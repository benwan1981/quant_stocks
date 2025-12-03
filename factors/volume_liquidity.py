# factors/volume_liquidity.py
from __future__ import annotations

import pandas as pd
import numpy as np


def factor_vol_ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    成交量/均量比：volume / MA(volume, window)
    """
    vol = df["volume"].astype(float)
    vol_ma = vol.rolling(window, min_periods=window//2).mean()
    ratio = vol / vol_ma
    ratio.name = f"vol_ratio_{window}"
    return ratio