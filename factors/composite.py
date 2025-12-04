# factors/composite.py
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict


def combine_scores(
    df: pd.DataFrame,
    weights: Dict[str, float],
    score_col: str = "total_score",
) -> pd.DataFrame:
    """
    将若干因子按权重线性组合成总分：
        total_score = sum( w_i * df[factor_i] )

    weights: {factor_name: weight}
    例如：
        {
            "trend_score": 1.0,
            "momentum_score": 1.0,
            "volume_score": 0.5,
            "risk_score": -0.5,
        }
    """
    out = df.copy()
    total = 0.0

    for name, w in weights.items():
        if name not in out.columns:
            # 没有的列按 0 处理
            out[name] = 0.0
        total = total + w * out[name].fillna(0.0)

    out[score_col] = total
    return out