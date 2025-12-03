# factors/price_momentum.py
from __future__ import annotations

import pandas as pd
import numpy as np


def factor_mom_pct(df: pd.DataFrame, window: int, col: str = "close") -> pd.Series:
    """
    价格动量：过去 window 日的累计收益率（(P_t / P_{t-window}) - 1）
    df: 必须包含 col 列（默认 close），index 为日期
    """
    px = df[col].astype(float)
    # pct_change(window) = P_t / P_{t-window} - 1
    mom = px.pct_change(periods=window)
    mom.name = f"mom_{window}"
    return mom


def factor_mom_rank01(df: pd.DataFrame, window: int, col: str = "close") -> pd.Series:
    """
    单标的时间序列版的“0-1 归一化动量”，方便单票策略用。
    multi-stock 截面排序建议用 panel 版本（在 strategy 模块里）。
    """
    mom = factor_mom_pct(df, window=window, col=col)
    # 在时间上做一个简单 0-1 归一，可以按你喜好调整
    valid = mom.dropna()
    if valid.empty:
        return mom

    rank = (valid - valid.min()) / (valid.max() - valid.min() + 1e-12)
    mom.loc[valid.index] = rank
    mom.name = f"mom_{window}_r01"
    return mom