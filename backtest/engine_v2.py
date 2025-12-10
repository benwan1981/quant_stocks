# backtest/engine_v2.py
# -*- coding: utf-8 -*-
"""
简化版组合动态因子回测引擎（使用预计算因子，先纯 Python 版本）：

- 从 precomputed/*.parquet 读取 prices / factors / regime
- 根据 strategy_v2.yaml / params_px.yaml 组合打分
- 根据 base_regime / z_sigma 给出 buy/sell/target_exp（简化版）
- 使用纯 Python 回测核心 backtest_core_py（后续可无缝换成 Numba 版本）
- 返回 equity_df，并构建 trades_df（基于持仓变动还原）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# 若后面要切换为 Numba 核心，再按需打开这一行
# from .engine_numba_core import backtest_core


ROOT_DIR = Path(__file__).resolve().parents[1]
PRE_DIR = ROOT_DIR / "precomputed"
FAC_DIR = PRE_DIR / "factors"
REG_DIR = PRE_DIR / "regime"


# ------------------- 配置结构 -------------------


@dataclass
class StrategyConfigV2:
    factor_weights: Dict[str, float]
    buy_base: float
    sell_base: float
    target_exp_base: float
    # 新增：按档位保存阈值，键是 "RISK_ON" / "NEUTRAL" / "RISK_OFF"
    thresholds_by_regime: Dict[str, Dict[str, float]] | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> "StrategyConfigV2":
        """
        兼容多种写法的策略 YAML：

        1）简单版：
            weights:
              mom10: 0.4
              mom60: 0.4
              vol60: 0.2
            thresholds:
              buy: 0.60
              sell: 0.30
              target_exp: 0.95

        2）P1~P7 版（大写 + regime 分档）：
            WEIGHTS:
              mom10: 0.4
              mom60: 0.4
              vol60: 0.2

            THRESHOLDS:
              RISK_ON:
                buy: 0.60
                sell: 0.30
                base_target: 0.95

              NEUTRAL:
                buy: 0.57
                sell: 0.32
                base_target: 0.65

              RISK_OFF:
                buy: 0.00
                sell: -0.20
                base_target: 0.25
        """
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # 顶层 key 做一层“大小写不敏感”
        cfg = {str(k).lower(): v for k, v in raw.items()}

        # ===== 因子权重 =====
        w_block = (
            cfg.get("weights")
            or cfg.get("factors")
            or cfg.get("w")
            or {}
        )

        factor_weights: Dict[str, float] = {}
        if isinstance(w_block, dict):
            for k, v in w_block.items():
                try:
                    factor_weights[str(k)] = float(v)
                except Exception:
                    continue

        if not factor_weights:
            factor_weights = {
                "mom10": 0.4,
                "mom60": 0.4,
                "vol60": 0.2,
            }

        # ===== 阈值块 =====
        th_block = cfg.get("thresholds") or cfg.get("th") or {}

        thresholds_by_regime: Dict[str, Dict[str, float]] | None = None
        buy_val = 0.60
        sell_val = 0.30
        target_val = 0.95

        if isinstance(th_block, dict):
            # 先看是不是分档结构（RISK_ON / NEUTRAL / RISK_OFF）
            upper_keys = {str(k).upper(): k for k in th_block.keys()}
            has_regimes = any(k in upper_keys for k in ("RISK_ON", "NEUTRAL", "RISK_OFF"))

            if has_regimes:
                thresholds_by_regime = {}

                for regime_up in ("RISK_ON", "NEUTRAL", "RISK_OFF"):
                    if regime_up not in upper_keys:
                        continue
                    orig_key = upper_keys[regime_up]
                    block = th_block.get(orig_key) or {}
                    if not isinstance(block, dict):
                        continue

                    r_buy = float(block.get("buy", 0.60))
                    r_sell = float(block.get("sell", 0.30))
                    r_target = float(
                        block.get("target_exp", block.get("base_target", 0.95))
                    )
                    thresholds_by_regime[regime_up] = {
                        "buy": r_buy,
                        "sell": r_sell,
                        "target": r_target,
                    }

                # 选一个“默认档位”作为 buy_base/sell_base/target_exp_base
                default_regime_up = None
                for candidate in ("NEUTRAL", "RISK_ON", "RISK_OFF"):
                    if thresholds_by_regime and candidate in thresholds_by_regime:
                        default_regime_up = candidate
                        break

                if thresholds_by_regime and default_regime_up is not None:
                    base = thresholds_by_regime[default_regime_up]
                    buy_val = base["buy"]
                    sell_val = base["sell"]
                    target_val = base["target"]
                else:
                    # 理论上不会走到这里，但防御一下
                    buy_val, sell_val, target_val = 0.60, 0.30, 0.95

            else:
                # 扁平结构：thresholds 里直接 buy/sell/target_exp
                buy_val = float(th_block.get("buy", 0.60))
                sell_val = float(th_block.get("sell", 0.30))
                target_val = float(
                    th_block.get("target_exp", th_block.get("base_target", 0.95))
                )
                thresholds_by_regime = None
        else:
            # thresholds 不是 dict，就走默认
            buy_val, sell_val, target_val = 0.60, 0.30, 0.95
            thresholds_by_regime = None

        return cls(
            factor_weights=factor_weights,
            buy_base=buy_val,
            sell_base=sell_val,
            target_exp_base=target_val,
            thresholds_by_regime=thresholds_by_regime,
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

    regime_dir: str | Path | None = None
    base_regime_file: str = "base_regime.parquet"
    z_sigma_file: str = "z_sigma.parquet"
    enable_regime_adjust: bool = True
    bull_z: float = -0.5
    bear_z: float = 0.5


# ------------------- 纯 Python 回测核心 -------------------


def backtest_core_py(
    prices: np.ndarray,
    scores: np.ndarray,
    buy_th: np.ndarray,
    sell_th: np.ndarray,   # 当前简化版暂时没用到
    target_exp: np.ndarray,
    fee_rate: float,
    initial_cash: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    纯 Python 版组合回测核心（先跑通逻辑，后续再考虑 Numba/GPU）。

    参数:
        prices:     (T, N) 收盘价矩阵
        scores:     (T, N) 0-1 之间的打分（横截面 rank 后）
        buy_th:     (T,)   每日买入阈值
        sell_th:    (T,)   每日卖出阈值（当前没用，可扩展）
        target_exp: (T,)   每日组合目标仓位（0~1）
        fee_rate:   手续费费率
        initial_cash: 初始现金

    返回:
        equity: (T,) 每日总权益
        cash:   (T,) 每日现金
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
        prev_p = prices[t - 1]

        # 若当天价格 NaN/非正，用前一日价格顶一下，避免 0 价/NaN 影响
        for j in range(N):
            if not np.isfinite(p[j]) or p[j] <= 0:
                p[j] = prev_p[j]

        # 当前组合市值 & 权益（交易前）
        port_val = float(np.nansum(pos_cur * p))
        eq_before = cash_cur + port_val
        if eq_before <= 0:
            eq_before = 1e-9

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
                for j in range(N):
                    if sell_mask[j] and p[j] > 0:
                        pos_cur[j] += trade_value[j] / p[j]  # trade_value[j] 为负

        # 再买入（受当前现金约束）
        buy_mask = trade_value > 0
        if buy_mask.any() and cash_cur > 0:
            buy_amt_plan = float(trade_value[buy_mask].sum())
            if buy_amt_plan > 0:
                # 考虑手续费的缩放
                scale = min(1.0, cash_cur / (buy_amt_plan * (1.0 + fee_rate)))
                if scale > 0:
                    buy_value = np.zeros(N, dtype=np.float64)
                    for j in range(N):
                        if buy_mask[j]:
                            buy_value[j] = trade_value[j] * scale

                    buy_amt_real = float(buy_value[buy_mask].sum())
                    fee_buy = buy_amt_real * fee_rate
                    cost = buy_amt_real + fee_buy
                    cash_cur -= cost

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


# ------------------- 引擎主体 -------------------


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

        # 用于后续导出组合快照 / 复盘
        self._pos_arr: Optional[np.ndarray] = None   # (T, N) 每日持仓股数
        self._price_arr: Optional[np.ndarray] = None # (T, N) 收盘价矩阵
        self._dates: Optional[pd.DatetimeIndex] = None
        self._codes: Optional[list[str]] = None


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

        # 加载因子（目前先支持 mom10 / mom60 / vol60）
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
        # base_regime
        br = pd.read_parquet(REG_DIR / "base_regime.parquet").reindex(index_like)
        if isinstance(br, pd.DataFrame):
            if "base_regime" in br.columns:
                base_regime = br["base_regime"]
            else:
                base_regime = br.iloc[:, 0]
        else:
            base_regime = br

        # z_sigma
        zs = pd.read_parquet(REG_DIR / "z_sigma.parquet").reindex(index_like)
        if isinstance(zs, pd.DataFrame):
            if "z_sigma" in zs.columns:
                z_sigma = zs["z_sigma"]
            else:
                z_sigma = zs.iloc[:, 0]
        else:
            z_sigma = zs

        base_regime = base_regime.ffill().fillna("neutral")
        z_sigma = z_sigma.ffill().fillna(0.0)

        return base_regime, z_sigma

    # ---------- 核心：构造输入 & 调用回测核心 ------------

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

        # 2) 加载价格 & 因子 & regime
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

        th_by_regime = self.strat_cfg.thresholds_by_regime

        if th_by_regime:
            # 有分档配置：根据 base_regime -> RISK_ON / NEUTRAL / RISK_OFF
            for i, reg in enumerate(base_regime.values):
                if reg == "bull":
                    regime_up = "RISK_ON"
                elif reg == "bear":
                    regime_up = "RISK_OFF"
                else:
                    regime_up = "NEUTRAL"

                block = th_by_regime.get(regime_up) or {}
                buy_th[i] = float(block.get("buy", self.strat_cfg.buy_base))
                sell_th[i] = float(block.get("sell", self.strat_cfg.sell_base))
                target_exp[i] = float(block.get("target", self.strat_cfg.target_exp_base))
        else:
            # 没有分档配置，退回到旧逻辑：根据 bull/bear 做简单偏移
            for i, reg in enumerate(base_regime.values):
                if reg == "bull":
                    buy_th[i] = self.strat_cfg.buy_base - 0.05
                    sell_th[i] = self.strat_cfg.sell_base + 0.02
                    target_exp[i] = min(1.0, self.strat_cfg.target_exp_base * 1.1)
                elif reg == "bear":
                    buy_th[i] = self.strat_cfg.buy_base + 0.05
                    sell_th[i] = self.strat_cfg.sell_base + 0.05
                    target_exp[i] = max(0.2, self.strat_cfg.target_exp_base * 0.5)

        # 6) 简单的波动自适应：z_sigma 高时降低目标仓位
        z_arr = z_sigma.to_numpy()
        target_exp = target_exp * (1.0 / (1.0 + 0.3 * np.maximum(z_arr, 0.0)))
        target_exp = np.clip(target_exp, 0.0, 1.0)

        # 7) 每天符合“打分 >= 买入阈值”的股票数量（方便 debug 是否一直没人可买）
        long_counts = (score_rank.values >= buy_th[:, None]).sum(axis=1)

        # 8) 调用纯 Python 回测核心
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
        trades: List[Tuple] = []
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

        # 保存内部状态，方便后续导出 1天 / 1周 / 1月 组合结果
        self._pos_arr = pos_arr
        self._price_arr = price_arr
        self._dates = prices.index
        self._codes = list(prices.columns)

        return eq


    def export_portfolio_snapshots(self, freq: str = "D") -> pd.DataFrame:
        """
        导出不同频率的组合快照，基于最近一次 run_backtest 的结果。

        参数
        ----
        freq : {"D", "W", "M"}
            "D" - 每日快照
            "W" - 每周最后一个交易日快照
            "M" - 每月最后一个交易日快照

        返回
        ----
        DataFrame，列包含：
            - date   : 日期
            - code   : 股票代码
            - shares : 持仓股数
            - value  : 该股票市值
            - weight : 在当日组合中的权重（按市值算）
        """
        if self._pos_arr is None or self._price_arr is None:
            raise RuntimeError("请先调用 run_backtest()，再导出组合快照。")

        dates = self._dates
        codes = self._codes
        pos_arr = self._pos_arr
        price_arr = self._price_arr

        # 1) 选择要导出的时间点
        if freq == "D":
            idx_list = list(range(len(dates)))
        else:
            # 构造一个辅助序列，用于 resample 找每周/每月最后一个索引
            ser = pd.Series(np.arange(len(dates)), index=dates)
            if freq == "W":
                # 按周取最后一个交易日，使用周五为锚点
                idx_ser = ser.resample("W-FRI").last()
            elif freq == "M":
                # 每月最后一个交易日
                idx_ser = ser.resample("M").last()
            else:
                raise ValueError(f"不支持的 freq: {freq}，请使用 'D'/'W'/'M'。")
            idx_list = [int(i) for i in idx_ser.dropna().to_list()]

        records = []
        for t in idx_list:
            dt = dates[t]
            shares_t = pos_arr[t]
            prices_t = price_arr[t]
            values_t = shares_t * prices_t
            total_val = float(np.nansum(values_t))

            # 若当日组合总市值为 0（空仓），则 weight 全为 0
            if total_val <= 0:
                weights_t = np.zeros_like(values_t)
            else:
                weights_t = values_t / total_val

            for j, code in enumerate(codes):
                sh = float(shares_t[j])
                if sh == 0 and total_val == 0:
                    # 完全空仓时，可以选择跳过所有记录；这里还是输出，方便看“空仓日”
                    pass

                records.append(
                    {
                        "date": dt,
                        "code": code,
                        "shares": sh,
                        "value": float(values_t[j]),
                        "weight": float(weights_t[j]),
                    }
                )

        df_snap = pd.DataFrame(records)
        # 可以按日期+权重排序，方便查看
        df_snap = df_snap.sort_values(["date", "weight"], ascending=[True, False])
        return df_snap
