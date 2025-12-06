# backtest/engine_numba_core.py
# -*- coding: utf-8 -*-
"""
组合回测核心（Numba 加速版）：

输入：
    prices     (T, N)
    scores     (T, N)   —— 已经是组合打分（越大越想买）
    buy_th     (T,)     —— 当日使用的买入阈值
    sell_th    (T,)     —— 当日使用的卖出阈值
    target_exp (T,)     —— 目标总仓位（0~1）
    fee_rate   float    —— 手续费率
    min_fee    float    —— 单笔最低费用
    stamp_duty float    —— 卖出印花税率
    lot_size   int      —— 整手股数
    initial_cash float  —— 初始资金

策略逻辑（简化版）：
    - 使用 t-1 的 score 来决定 t 日的买卖；
    - 先卖：score[t-1, i] < sell_th[t] 且已持有 -> 全卖出；
    - 再买：score[t-1, i] > buy_th[t] 且未持有 -> 等权分配剩余资金，向下取整手；
    - 目标是接近 target_exp[t] * equity[t] 的市值。

输出：
    equity(T,)
    cash(T,)
    pos(T, N)     —— 每日持仓股数（用于后续在 Python 层还原交易明细）
"""

from __future__ import annotations
import numpy as np

try:
    from numba import njit
except Exception:  # 如果 numba 未安装，提供一个假的装饰器（方便先跑通）
    def njit(*args, **kwargs):
        def wrapper(f):
            return f
        return wrapper


@njit
def _calc_commission(gross: float, fee_rate: float, min_fee: float) -> float:
    fee = gross * fee_rate
    if fee < min_fee:
        fee = min_fee
    return fee


@njit
def backtest_core(
    prices: np.ndarray,
    scores: np.ndarray,
    buy_th: np.ndarray,
    sell_th: np.ndarray,
    target_exp: np.ndarray,
    fee_rate: float,
    min_fee: float,
    stamp_duty: float,
    lot_size: int,
    initial_cash: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T, N = prices.shape
    cash = initial_cash
    holdings = np.zeros(N, dtype=np.int64)

    equity = np.zeros(T, dtype=np.float64)
    cash_series = np.zeros(T, dtype=np.float64)
    pos = np.zeros((T, N), dtype=np.int64)

    # 约定：第 0 天不交易，只建权益基准
    for t in range(T):
        # === 估值 ===
        port_val = 0.0
        for i in range(N):
            if holdings[i] > 0:
                port_val += holdings[i] * prices[t, i]
        eq = cash + port_val
        equity[t] = eq
        cash_series[t] = cash

        # 记录当日持仓
        for i in range(N):
            pos[t, i] = holdings[i]

        # t == 0 不做交易，避免使用 score[-1]
        if t == 0:
            continue

        # === 先卖：使用 t-1 的 score 与 t 日价格 ===
        equity_t_before = eq  # 用于计算目标仓位
        for i in range(N):
            if holdings[i] <= 0:
                continue
            if scores[t - 1, i] < sell_th[t]:
                price = prices[t, i]
                if price <= 0:
                    continue
                gross = price * holdings[i]
                fee = _calc_commission(gross, fee_rate, min_fee)
                tax = gross * stamp_duty
                cash += gross - fee - tax
                holdings[i] = 0

        # === 再估值 ===
        port_val = 0.0
        for i in range(N):
            if holdings[i] > 0:
                port_val += holdings[i] * prices[t, i]
        eq = cash + port_val
        equity_t_before = eq

        # === 再买：目标仓位 target_exp[t] ===
        desired_value = target_exp[t] * equity_t_before
        add_value = desired_value - port_val
        if add_value <= 0:
            # 不加仓，仅记录
            equity[t] = eq
            cash_series[t] = cash
            for i in range(N):
                pos[t, i] = holdings[i]
            continue

        # 候选：score[t-1, i] > buy_th[t] 且 holdings == 0，等权分配 add_value
        # 先数一遍候选数量
        cnt = 0
        for i in range(N):
            if holdings[i] == 0 and scores[t - 1, i] > buy_th[t]:
                cnt += 1

        if cnt == 0:
            # 没候选可买
            equity[t] = eq
            cash_series[t] = cash
            for i in range(N):
                pos[t, i] = holdings[i]
            continue

        alloc_per = add_value / cnt

        for i in range(N):
            if holdings[i] != 0:
                continue
            if scores[t - 1, i] <= buy_th[t]:
                continue
            price = prices[t, i]
            if price <= 0:
                continue
            # 按整手买入
            lots = int(alloc_per / price / lot_size)
            if lots <= 0:
                continue
            shares = lots * lot_size
            gross = shares * price
            fee = _calc_commission(gross, fee_rate, min_fee)
            cost = gross + fee
            if cost > cash:
                continue
            cash -= cost
            holdings[i] = shares

        # 最终估值
        port_val = 0.0
        for i in range(N):
            if holdings[i] > 0:
                port_val += holdings[i] * prices[t, i]
        eq = cash + port_val

        equity[t] = eq
        cash_series[t] = cash
        for i in range(N):
            pos[t, i] = holdings[i]

    return equity, cash_series, pos
