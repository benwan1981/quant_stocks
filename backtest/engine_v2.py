# backtest/engine_v2.py
# -*- coding: utf-8 -*-
"""
简化版组合动态因子回测引擎（使用预计算因子 + Numba 核心）：

- 从 precomputed/*.parquet 读取 prices / factors / regime
- 根据 strategy_v2.yaml 组合打分
- 根据 base_regime / z_sigma 给出 buy/sell/target_exp（简化版）
- 调用 engine_numba_core.backtest_core 做回测
- 返回 equity_df，并构建 trades_df（基于持仓变动还原）

后续你可以在此基础上，逐步往原来伪代码 A 的完整版靠拢。
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

#   from .engine_numba_core import backtest_core

ROOT_DIR = Path(__file__).resolve().parents[1]
PRE_DIR = ROOT_DIR / "precomputed"
FAC_DIR = PRE_DIR / "factors"
REG_DIR = PRE_DIR / "regime"


# ------------------- 配置结构 -------------------

@dataclass
@dataclass
class StrategyConfigV2:
    factor_weights: Dict[str, float]
    buy_base: float
    sell_base: float
    target_exp_base: float

    @classmethod
    def from_yaml(cls, path: Path) -> "StrategyConfigV2":
        """
        兼容两类写法：
        1）简单版：
            weights:
              mom10: 0.4
              mom60: 0.4
              vol60: 0.2
            thresholds:
              buy: 0.60
              sell: 0.30
              target_exp: 0.95

        2）你之前给的 P1~P6 结构：
            THRESHOLDS:
              RISK_ON:
                buy: 0.60
                sell: 0.30
                base_target: 0.95
              ...

        这里简化为只用 RISK_ON 这块做“基础参数”。
        """
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # 兼容大小写的 weights
        w = cfg.get("weights") or cfg.get("WEIGHTS") or {}

        # 如果没写 weights，就给一个默认组合（后续可以在 yaml 里自己调）
        if not w:
            w = {
                "mom10": 0.4,
                "mom60": 0.4,
                "vol60": 0.2,
            }

        # 兼容大小写的 thresholds
        th_raw = cfg.get("thresholds") or cfg.get("THRESHOLDS") or {}

        # 如果是 P1~P6 那种结构，里面有 RISK_ON/NEUTRAL/RISK_OFF
        if isinstance(th_raw, dict) and "RISK_ON" in th_raw:
            th = th_raw["RISK_ON"]
            buy = th.get("buy", 0.60)
            sell = th.get("sell", 0.30)
            # 你 P1~P6 里用的是 base_target，这里映射到 target_exp_base
            target_exp = th.get("base_target", 0.95)
        else:
            # 扁平结构
            buy = th_raw.get("buy", 0.60)
            sell = th_raw.get("sell", 0.30)
            target_exp = th_raw.get("target_exp", 0.95)

        return cls(
            factor_weights={str(k): float(v) for k, v in w.items()},
            buy_base=float(buy),
            sell_base=float(sell),
            target_exp_base=float(target_exp),
        )


@dataclass
class ExecutionConfig:
    initial_cash: float = 1_000_000.0
    fee_rate: float = 0.0005
    min_fee: float = 5.0
    stamp_duty: float = 0.001
    lot_size: int = 100


@dataclass
class RegimeConfig:
    """
    市场状态 / 波动配置占位类。

    目前主要用于:
    - strategy_v2_loader 从 YAML 解析时的类型提示
    - 允许未来在引擎里根据 base_regime / z_sigma 调整阈值

    如果暂时不用这些功能，保持默认值即可，不会影响回测结果。
    """
    # 预计算好的 regime 文件所在目录（通常是 ROOT_DIR / "precomputed" / "regime"）
    regime_dir: str | Path | None = None

    # 基础市场状态文件名，例如 "base_regime.parquet"
    base_regime_file: str = "base_regime.parquet"

    # 波动 z-score 文件名，例如 "z_sigma.parquet"
    z_sigma_file: str = "z_sigma.parquet"

    # 是否启用 regime 调整（不用可以关掉）
    enable_regime_adjust: bool = True

    # z_sigma 的阈值（只是示例，loader 用得到就有字段）
    bull_z: float = -0.5
    bear_z: float = 0.5

# ------------------- 引擎主体 -------------------

def backtest_core_py(
    prices: np.ndarray,
    scores: np.ndarray,
    buy_th: np.ndarray,
    sell_th: np.ndarray,
    target_exp: np.ndarray,
    fee_rate: float,
    initial_cash: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    纯 Python 版组合回测核心（先跑通逻辑，后续再考虑 Numba/GPU）.

    参数:
        prices:    (T, N) 收盘价矩阵
        scores:    (T, N) 0-1 之间的打分（横截面 rank 后）
        buy_th:    (T,)   每日买入阈值
        sell_th:   (T,)   每日卖出阈值（当前简化版暂时没用到，可扩展）
        target_exp:(T,)   每日组合目标仓位（0~1）
        fee_rate:  手续费费率
        initial_cash: 初始现金

    返回:
        equity: (T,)  每日总权益
        cash:   (T,)  每日现金
        pos:    (T, N) 每日持仓股数
    """
    T, N = prices.shape

    equity = np.zeros(T, dtype=np.float64)
    cash = np.zeros(T, dtype=np.float64)
    pos = np.zeros((T, N), dtype=np.float64)

    cash_cur = float(initial_cash)
    pos_cur = np.zeros(N, dtype=np.float64)

    # t=0 先记录初始状态（全现金）
    equity[0] = cash_cur
    cash[0] = cash_cur
    pos[0, :] = pos_cur

    for t in range(1, T):
        p = prices[t].copy()

        # 若当天价格有 NaN，用前一日价格顶一下，避免 0 价/NaN 影响
        prev_p = prices[t - 1]
        for j in range(N):
            if not np.isfinite(p[j]) or p[j] <= 0:
                p[j] = prev_p[j]

        # 当前组合市值 & 权益（交易前）
        port_val = float(np.nansum(pos_cur * p))
        eq_before = cash_cur + port_val
        if eq_before <= 0:
            eq_before = 1e-9  # 防止除零

        # 决定今日要持有哪些股票：score >= buy_th[t] 的，等权分配 target_exp[t]
        long_mask = scores[t] >= buy_th[t]
        target_w = np.zeros(N, dtype=np.float64)
        cnt_long = int(long_mask.sum())
        if cnt_long > 0 and target_exp[t] > 0:
            w_each = float(target_exp[t]) / cnt_long
            for j in range(N):
                if long_mask[j]:
                    target_w[j] = w_each

        # 当前各股市值 & 目标市值
        cur_value = pos_cur * p
        target_value = target_w * eq_before
        trade_value = target_value - cur_value  # >0 买入, <0 卖出

        # 先卖出
        sell_mask = trade_value < 0
        if sell_mask.any():
            sell_amt = float(-trade_value[sell_mask].sum())
            if sell_amt > 0:
                fee_sell = sell_amt * fee_rate
                cash_cur += sell_amt - fee_sell
                # pos_cur += trade_value / p （trade_value 为负）
                for j in range(N):
                    if sell_mask[j]:
                        if p[j] > 0:
                            pos_cur[j] += trade_value[j] / p[j]

        # 再买入（受当前现金约束）
        buy_mask = trade_value > 0
        if buy_mask.any() and cash_cur > 0:
            buy_amt_plan = float(trade_value[buy_mask].sum())
            if buy_amt_plan > 0:
                # 考虑手续费的缩放
                scale = min(1.0, cash_cur / (buy_amt_plan * (1.0 + fee_rate)))
                if scale > 0:
                    buy_value = trade_value * 0.0
                    for j in range(N):
                        if buy_mask[j]:
                            buy_value[j] = trade_value[j] * scale
                    buy_amt_real = float(buy_value[buy_mask].sum())
                    fee_buy = buy_amt_real * fee_rate
                    cost = buy_amt_real + fee_buy
                    cash_cur -= cost
                    # 增加持仓
                    for j in range(N):
                        if buy_mask[j] and p[j] > 0:
                            pos_cur[j] += buy_value[j] / p[j]

        # 更新当日权益
        port_val = float(np.nansum(pos_cur * p))
        eq_today = cash_cur + port_val

        equity[t] = eq_today
        cash[t] = cash_cur
        pos[t, :] = pos_cur

    return equity, cash, pos


class BacktestEngineV2:
    def __init__(
        self,
        strat_cfg: StrategyConfigV2,
        universe_codes: Optional[List[str]] = None,
    ) -> None:
        """
        universe_codes:
            若为 None，则使用 prices.parquet 中的全部列；
            否则仅使用给定代码的交集。
        """
        self.strat_cfg = strat_cfg
        self.universe_codes = universe_codes

        # 运行后填充：
        self.trades_df: Optional[pd.DataFrame] = None
        self.debug_df: Optional[pd.DataFrame] = None

    # ---------- 数据加载 ----------

    def _load_prices_and_factors(
        self,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        prices = pd.read_parquet(PRE_DIR / "prices.parquet")
        if start is not None:
            prices = prices[prices.index >= start]
        if end is not None:
            prices = prices[prices.index <= end]

        if self.universe_codes:
            use_codes = [c for c in self.universe_codes if c in prices.columns]
            prices = prices[use_codes]

        # 加载因子
        factors: Dict[str, pd.DataFrame] = {}
        for fname in ["mom10", "mom60", "vol60"]:
            p = FAC_DIR / f"{fname}.parquet"
            if not p.exists():
                continue
            f = pd.read_parquet(p)
            f = f.reindex(prices.index)
            if self.universe_codes:
                f = f[prices.columns]
            factors[fname] = f

        if not factors:
            raise RuntimeError("未加载到任何因子，请先运行 precompute_factors.py")

        return prices, factors

    def _load_regime(self, index_like: pd.Index) -> Tuple[pd.Series, pd.Series]:
        # 读取并对齐 base_regime
        br = pd.read_parquet(REG_DIR / "base_regime.parquet").reindex(index_like)
        # 兼容不同 pandas 版本：单列 DataFrame -> Series
        if isinstance(br, pd.DataFrame):
            # 尝试取名为 'base_regime' 的列，否则取第一列
            if "base_regime" in br.columns:
                base_regime = br["base_regime"]
            else:
                base_regime = br.iloc[:, 0]
        else:
            base_regime = br

        # 同理处理 z_sigma
        zs = pd.read_parquet(REG_DIR / "z_sigma.parquet").reindex(index_like)
        if isinstance(zs, pd.DataFrame):
            if "z_sigma" in zs.columns:
                z_sigma = zs["z_sigma"]
            else:
                z_sigma = zs.iloc[:, 0]
        else:
            z_sigma = zs

        # 缺失值处理：用前值填充，剩余空补默认
        base_regime = base_regime.ffill().fillna("neutral")
        z_sigma = z_sigma.ffill().fillna(0.0)

        return base_regime, z_sigma


    # ---------- 核心：构造 Numba 输入 & 调用 ------------

    def run_backtest(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        exec_cfg: Optional[ExecutionConfig] = None,
    ) -> pd.DataFrame:
        """
        返回 equity_df（index=date, columns: equity, cash, portfolio_value）
        """
        # 1) 执行参数
        if exec_cfg is None:
            exec_cfg = ExecutionConfig()

        start_ts = pd.to_datetime(start) if start else None
        end_ts = pd.to_datetime(end) if end else None

        # 2) 加载价格 & 因子
        prices, factors = self._load_prices_and_factors(start_ts, end_ts)
        base_regime, z_sigma = self._load_regime(prices.index)

        # 3) 组合打分：score = Σ w_f * factor_f
        score_df = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for name, w in self.strat_cfg.factor_weights.items():
            if name not in factors:
                # YAML 里有但当前没预计算的因子，直接跳过
                continue
            score_df += float(w) * factors[name]

        # 4) 横截面 0-1 排名，增强稳健性
        score_rank = score_df.rank(axis=1, pct=True)

        # 5) 按市场状态生成基础买卖阈值 & 目标仓位
        n = len(prices)
        buy_th = np.full(n, self.strat_cfg.buy_base, dtype=np.float64)
        sell_th = np.full(n, self.strat_cfg.sell_base, dtype=np.float64)
        target_exp = np.full(n, self.strat_cfg.target_exp_base, dtype=np.float64)

        # base_regime 是一个 Series，逐日调整
        for i, reg in enumerate(base_regime.values):
            if reg == "bull":
                buy_th[i] = self.strat_cfg.buy_base - 0.05
                sell_th[i] = self.strat_cfg.sell_base + 0.02
                target_exp[i] = min(1.0, self.strat_cfg.target_exp_base * 1.1)
            elif reg == "bear":
                buy_th[i] = self.strat_cfg.buy_base + 0.05
                sell_th[i] = self.strat_cfg.sell_base + 0.05
                target_exp[i] = max(0.2, self.strat_cfg.target_exp_base * 0.5)

        # 简单的波动自适应：z_sigma 高时降低目标仓位
        z_arr = z_sigma.to_numpy()
        target_exp = target_exp * (1.0 / (1.0 + 0.3 * np.maximum(z_arr, 0.0)))
        target_exp = np.clip(target_exp, 0.0, 1.0)

        # 每天符合“打分 >= 买入阈值”的股票数量，用于排查是否一直没人可买
        long_counts = (score_rank.values >= buy_th[:, None]).sum(axis=1)

        # 核心输入（纯 Python 版）
        price_arr = prices.to_numpy(dtype=np.float64)
        score_arr = score_rank.to_numpy(dtype=np.float64)
        buy_arr = buy_th.astype(np.float64)
        sell_arr = sell_th.astype(np.float64)
        target_arr = target_exp.astype(np.float64)

        equity_arr, cash_arr, pos_arr = backtest_core_py(
            price_arr,
            score_arr,
            buy_arr,
            sell_arr,
            target_arr,
            float(exec_cfg.fee_rate),
            float(exec_cfg.initial_cash),
        )


        # 8) 调用 Numba 核心
        # 关键点：**只能用位置参数，不能再出现 prices= 这种关键字**
        #
        # 假设 engine_numba_core.backtest_core 的定义类似：
        #   backtest_core(prices, scores, buy_th, sell_th, target_exp, fee_rate, initial_cash)
        #
        '''     equity_arr, cash_arr, pos_arr = backtest_core(
                price_arr,              # prices
                score_arr,              # scores
                buy_arr,                # buy_th
                sell_arr,               # sell_th
                target_arr,             # target_exp
                exec_cfg.fee_rate,      # fee_rate
                exec_cfg.initial_cash,  # initial_cash
            )'''
        
                # Numba 核心输入 —— 现在先使用纯 Python 版本 backtest_core_py
        price_arr = prices.to_numpy(dtype=np.float64)
        score_arr = score_rank.to_numpy(dtype=np.float64)
        buy_arr = buy_th.astype(np.float64)
        sell_arr = sell_th.astype(np.float64)
        target_arr = target_exp.astype(np.float64)

        equity_arr, cash_arr, pos_arr = backtest_core_py(
            price_arr,
            score_arr,
            buy_arr,
            sell_arr,
            target_arr,
            float(exec_cfg.fee_rate),
            float(exec_cfg.initial_cash),
        )


        # 9) 构造 equity_df
        eq = pd.DataFrame(
            {
                "equity": equity_arr,
                "cash": cash_arr,
            },
            index=prices.index,
        )

        # 10) 组合市值与真实仓位
        port_val = (pos_arr * price_arr).sum(axis=1)
        eq["portfolio_value"] = port_val

        # 11) debug_df：把计算过程暴露出去
        self.debug_df = pd.DataFrame(
            {
                "base_regime": base_regime.values,
                "z_sigma": z_arr,
                "buy_th": buy_arr,
                "sell_th": sell_arr,
                "target_exp": target_arr,
                "actual_exp": port_val / np.maximum(eq["equity"].values, 1e-9),
                "long_candidate_count": long_counts,
            },
            index=prices.index,
        )

        # 12) 交易明细：通过 pos_arr 差分近似还原
        trades: list[tuple] = []
        dates = prices.index.to_list()
        codes = prices.columns.to_list()
        for t in range(1, len(dates)):
            dt = dates[t]
            for j, code in enumerate(codes):
                prev_shares = pos_arr[t - 1, j]
                cur_shares = pos_arr[t, j]
                if cur_shares == prev_shares:
                    continue
                price = price_arr[t, j]
                if price <= 0:
                    continue
                if cur_shares > prev_shares:
                    qty = cur_shares - prev_shares
                    trades.append((dt, "BUY", code, qty, price))
                else:
                    qty = prev_shares - cur_shares
                    trades.append((dt, "SELL", code, qty, price))

        self.trades_df = pd.DataFrame(
            trades,
            columns=["date", "action", "code", "shares", "price"],
        )

        return eq
