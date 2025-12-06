# -*- coding: utf-8 -*-
"""
多标的 T+1 执行引擎（简化版）：
- 先卖后买
- 每日根据 target_weight_t，在 t+1 日开盘成交
- 不做整手约束（后续你可以加 floor(/100)*100）
"""

from __future__ import annotations

from dataclasses import dataclass,fields
from typing import Dict

import pandas as pd

# backtest/execution_v2.py 顶部
from pathlib import Path
from typing import Any, Dict

import yaml  # 需要 pip install pyyaml（你项目里如果已有就不用管）

@dataclass
class ExecutionConfig:
    # 这里以你现在 engine_v2 真正用到的字段为准
    initial_cash: float = 1_000_000.0
    fee_rate: float = 0.0005
    slippage: float = 0.0005
    min_fee: float = 5.0
    stamp_duty: float = 0.001
    lot_size: int = 100
    # 如果你后来在 __init__ 里真的加了别的字段，比如：
    # min_fee: float = 5.0
    # stamp_duty: float = 0.001
    # lot_size: int = 100
    # 就一并写在这里

    @classmethod
    def from_yaml(cls, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            valid_keys = {f.name for f in fields(cls)}
            filtered = {k: v for k, v in data.items() if k in valid_keys}

            return cls(**filtered)

        except Exception as e:
            print(f"加载执行配置 {path} 失败，使用默认参数。错误: {e}")
            return cls()


def run_execution_t1_equal_weight(
    price_panel: Dict[str, pd.DataFrame],
    target_weights: pd.DataFrame,
    cfg: ExecutionConfig | None = None,
    collect_debug: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    price_panel: {code: df}，df 至少包含 ["open", "close"]，index 为日期
    target_weights: DataFrame，index 为日期，列为 code，值为当日目标权重（0~1）
                    约定：在 t 日给出的权重，在 t+1 日开盘成交。
    返回：资金曲线 DataFrame（index 为日期）：
      - equity, cash, market_value, ret
    """
    if cfg is None:
        cfg = ExecutionConfig()

    all_dates = sorted(target_weights.index)
    codes = list(price_panel.keys())

    shares = {code: 0.0 for code in codes}
    cash = cfg.initial_cash
    records = []
    exec_debug: list[dict] = []
    trades: list[dict] = []

    for i in range(len(all_dates) - 1):
        t = all_dates[i]
        t_next = all_dates[i + 1]

        w_t = target_weights.loc[t].fillna(0.0)

        mv = 0.0
        for code in codes:
            dfp = price_panel[code]
            if t not in dfp.index:
                continue
            close_t = dfp.at[t, "close"]
            mv += shares[code] * close_t
        equity_t = cash + mv

        target_mv = w_t * equity_t
        target_exposure = float(w_t.fillna(0.0).abs().sum())

        cash_t1 = cash
        for code in codes:
            dfp = price_panel[code]
            if t_next not in dfp.index:
                continue
            open_next = dfp.at[t_next, "open"]

            cur_shares = shares[code]
            cur_mv_t = cur_shares * open_next
            tgt_mv_t = target_mv.get(code, 0.0)
            diff_mv = tgt_mv_t - cur_mv_t

            if abs(diff_mv) < 1e-8:
                continue

            if diff_mv < 0:
                sell_mv = -diff_mv
                sell_shares = sell_mv / (open_next * (1 + cfg.slippage))
                sell_shares = min(sell_shares, cur_shares)
                proceeds = sell_shares * open_next * (1 - cfg.slippage)
                fee = proceeds * cfg.fee_rate
                cash_t1 += proceeds - fee
                shares[code] -= sell_shares
                if collect_debug and sell_shares > 0:
                    trades.append(
                        {
                            "date": t_next,
                            "code": code,
                            "action": "SELL",
                            "shares": float(sell_shares),
                            "price": float(open_next),
                            "amount": float(proceeds),
                            "fee": float(fee),
                        }
                    )
            else:
                buy_mv = diff_mv
                buy_shares = buy_mv / (open_next * (1 + cfg.slippage))
                cost = buy_shares * open_next * (1 + cfg.slippage)
                fee = cost * cfg.fee_rate
                total_cost = cost + fee
                if total_cost > cash_t1:
                    scale = cash_t1 / total_cost
                    buy_shares *= scale
                    cost = buy_shares * open_next * (1 + cfg.slippage)
                    fee = cost * cfg.fee_rate
                    total_cost = cost + fee
                cash_t1 -= total_cost
                shares[code] += buy_shares
                if collect_debug and buy_shares > 0:
                    trades.append(
                        {
                            "date": t_next,
                            "code": code,
                            "action": "BUY",
                            "shares": float(buy_shares),
                            "price": float(open_next),
                            "amount": float(total_cost),
                            "fee": float(fee),
                        }
                    )

        cash = cash_t1

        mv_t1 = 0.0
        for code in codes:
            dfp = price_panel[code]
            if t_next not in dfp.index:
                continue
            close_t1 = dfp.at[t_next, "close"]
            mv_t1 += shares[code] * close_t1
        equity_t1 = cash + mv_t1

        records.append(
            {
                "date": t_next,
                "equity": equity_t1,
                "cash": cash,
                "market_value": mv_t1,
            }
        )

        if collect_debug:
            actual_exposure = 0.0
            if equity_t1 > 0:
                actual_exposure = mv_t1 / equity_t1
            num_positions = sum(1 for s in shares.values() if s > 1e-8)
            exec_debug.append(
                {
                    "date": t_next,
                    "target_exposure_exec": target_exposure,
                    "actual_exposure": actual_exposure,
                    "num_positions": num_positions,
                }
            )

    eq = pd.DataFrame(records).set_index("date")
    eq["ret"] = eq["equity"].pct_change().fillna(0.0)
    if not collect_debug:
        return eq

    trades_df = pd.DataFrame(trades)
    exec_debug_df = pd.DataFrame(exec_debug).set_index("date") if exec_debug else pd.DataFrame()
    return eq, trades_df, exec_debug_df
