# -*- coding: utf-8 -*-
"""
市场波动 regime：
- 计算指数 60 日年化波动
- 得到 z-score
- 简单给出 risk_on / neutral / risk_off 三档
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VolRegimeParams:
    vol_window: int = 60
    ann_factor: int = 252


def compute_index_vol_z(
    index_df: pd.DataFrame,
    params: VolRegimeParams | None = None,
) -> pd.DataFrame:
    """
    输入：指数 df（至少包含 close）
    输出：df_vol，包含：
      - ret_idx
      - vol60
      - vol_z
      - regime: risk_on / neutral / risk_off
    """
    if params is None:
        params = VolRegimeParams()

    df = index_df.sort_index().copy()
    df["ret_idx"] = df["close"].pct_change().fillna(0.0)

    vol = df["ret_idx"].rolling(params.vol_window, min_periods=params.vol_window).std()
    vol_ann = vol * np.sqrt(params.ann_factor)

    mu = vol_ann.expanding().mean()
    sigma = vol_ann.expanding().std().replace(0, np.nan)
    z = (vol_ann - mu) / (sigma + 1e-9)

    df["vol60"] = vol_ann
    df["vol_z"] = z

    # regime 划分（你后面可以自己微调阈值）
    cond_risk_off = df["vol_z"] >= 1.0
    cond_risk_on = df["vol_z"] <= -0.5
    regime = pd.Series("neutral", index=df.index)
    regime[cond_risk_on] = "risk_on"
    regime[cond_risk_off] = "risk_off"
    df["regime"] = regime

    return df
