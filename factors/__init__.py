# factors/__init__.py
from __future__ import annotations

from typing import Callable, Dict, List

import pandas as pd

from .price_momentum import factor_mom_pct, factor_mom_rank01
from .trend import factor_ma, factor_ma_ratio
from .volatility import factor_volatility
from .volume_liquidity import factor_vol_ratio

# 单标的因子函数签名：f(df: DataFrame) -> Series
FactorFunc = Callable[[pd.DataFrame], pd.Series]

# 一个简单的注册表：因子名 -> 无参数版本的函数（参数写死一部分）
FACTOR_REGISTRY: Dict[str, FactorFunc] = {
    # 动量
    "mom_10": lambda df: factor_mom_pct(df, window=10),
    "mom_60": lambda df: factor_mom_pct(df, window=60),

    # 均线
    "ma20":   lambda df: factor_ma(df, window=20),
    "ma60":   lambda df: factor_ma(df, window=60),
    "ma20_ma60_ratio": lambda df: factor_ma_ratio(df, short=20, long=60),

    # 波动率
    "vol_20": lambda df: factor_volatility(df, window=20),
    "vol_60": lambda df: factor_volatility(df, window=60),

    # 量能
    "vol_ratio_20": lambda df: factor_vol_ratio(df, window=20),
}

def apply_factors(df: pd.DataFrame, factor_names: List[str]) -> pd.DataFrame:
    """
    给单个标的的日线 df（open/high/low/close/volume）批量打因子：
        df_with_fac = apply_factors(df_price, ["ma20", "ma60", "mom_10", "vol_20"])
    """
    out = df.copy()
    for name in factor_names:
        if name not in FACTOR_REGISTRY:
            raise KeyError(f"未注册的因子: {name}")
        s = FACTOR_REGISTRY[name](out)
        out[s.name] = s
    return out