# factors/volatility.py
from __future__ import annotations

import pandas as pd
import numpy as np


def factor_volatility(
    df: pd.DataFrame,
    window: int = 20,
    col: str = "close",
) -> pd.Series:
    """
    以日收益率为基础的波动率（标准差）
    """
    ret = df[col].pct_change()
    vol = ret.rolling(window, min_periods=window).std()
    vol.name = f"vol_{window}"
    return vol