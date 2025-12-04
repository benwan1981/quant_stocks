# factors/trend.py
from __future__ import annotations

import pandas as pd
import numpy as np


def factor_ma(df: pd.DataFrame, window: int, col: str = "close") -> pd.Series:
    """
    简单均线 MA
    """
    ma = df[col].rolling(window, min_periods=window).mean()
    ma.name = f"ma{window}"
    return ma


def factor_ma_ratio(df: pd.DataFrame, short: int, long: int, col: str = "close") -> pd.Series:
    """
    MA 比值：MA_short / MA_long - 1，用来刻画趋势强弱
    """
    ma_s = factor_ma(df, short, col=col)
    ma_l = factor_ma(df, long, col=col)

    ratio = ma_s / ma_l - 1
    ratio.name = f"ma{short}_ma{long}_ratio"
    return ratio