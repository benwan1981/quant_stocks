# -*- coding: utf-8 -*-
"""
v2 基础因子：
- 残差收益（对行业/风格指数回归后得到的 resid_ret）
- mom_short / mom_long（残差动量）
- vol60（残差波动）
- 横截面排名 0-1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .residual_momentum import compute_residual_return_series


@dataclass
class FactorWindows:
    mom_short: int = 10
    mom_long: int = 60
    vol: int = 60


def _rolling_sum(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).sum()


def _rolling_std(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).std()


def compute_stock_factor_panel(
    stock_universe: Dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    windows: FactorWindows | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    对股票池中每只股票，计算：
      - resid_ret: 残差收益
      - mom_short, mom_long, vol60: 基于残差收益的动量/波动
    返回：{code: df}，index 为日期
    """
    if windows is None:
        windows = FactorWindows()

    idx = index_df.sort_index()
    idx["ret_idx"] = idx["close"].pct_change().fillna(0.0)

    result: Dict[str, pd.DataFrame] = {}

    for code, df in stock_universe.items():
        s = df.sort_index().copy()
        s["ret"] = s["close"].pct_change().fillna(0.0)

        resid = compute_residual_return_series(
            stock_ret=s["ret"], index_ret=idx["ret_idx"]
        )

        fac = pd.DataFrame(index=s.index)
        fac["resid_ret"] = resid
        fac["mom_short"] = _rolling_sum(resid, windows.mom_short)
        fac["mom_long"] = _rolling_sum(resid, windows.mom_long)
        fac["vol60"] = _rolling_std(resid, windows.vol)

        result[code] = fac

    return result


def cross_section_rank_0_1(
    factor_panel: Dict[str, pd.DataFrame],
    col: str,
) -> Dict[str, pd.Series]:
    """
    对某一列因子做横截面 0-1 排名。
    返回：{code: rank_series}
    """
    all_dates = sorted(set().union(*[df.index for df in factor_panel.values()]))
    ranks_by_code: Dict[str, pd.Series] = {
        code: pd.Series(dtype=float) for code in factor_panel.keys()
    }

    for dt in all_dates:
        vals = {}
        for code, df in factor_panel.items():
            if dt in df.index:
                v = df.at[dt, col]
                if np.isfinite(v):
                    vals[code] = float(v)

        if len(vals) <= 1:
            continue

        ser = pd.Series(vals)
        rank = ser.rank(method="average")
        rank_01 = (rank - 1) / (len(rank) - 1)

        for code, r in rank_01.items():
            s = ranks_by_code[code]
            ranks_by_code[code] = s.reindex(s.index.union([dt]))
            ranks_by_code[code].at[dt] = r

    for code, s in ranks_by_code.items():
        ranks_by_code[code] = s.sort_index()

    return ranks_by_code


def build_composite_score_panel(
    factor_panel: Dict[str, pd.DataFrame],
    alpha_mom_short: float = 0.6,
    alpha_mom_long: float = 0.3,
    alpha_vol: float = 0.1,
    gamma_resid_mom: float = 0.2,
) -> Dict[str, pd.DataFrame]:
    """
    基于残差动量的组合打分：
      score_base = α1*mom_short_r + α2*mom_long_r + α3*(-vol_r)
      score = score_base + γ*resid_mom_rank（这里 resid_mom_rank 用 mom_short_r 再加权视作近似）
    返回：{code: df}，df 至少包含：
      - score_base
      - score
      - mom_short_r / mom_long_r / vol_r
    """
    mom_short_r = cross_section_rank_0_1(factor_panel, "mom_short")
    mom_long_r = cross_section_rank_0_1(factor_panel, "mom_long")
    vol_r = cross_section_rank_0_1(factor_panel, "vol60")

    result: Dict[str, pd.DataFrame] = {}

    for code, df in factor_panel.items():
        fac = pd.DataFrame(index=df.index)
        fac["mom_short_r"] = mom_short_r.get(code, pd.Series(dtype=float))
        fac["mom_long_r"] = mom_long_r.get(code, pd.Series(dtype=float))
        fac["vol_r"] = vol_r.get(code, pd.Series(dtype=float))

        fac["resid_mom_rank"] = fac["mom_short_r"]

        fac["score_base"] = (
            alpha_mom_short * fac["mom_short_r"]
            + alpha_mom_long * fac["mom_long_r"]
            - alpha_vol * fac["vol_r"]
        )
        fac["score"] = fac["score_base"] + gamma_resid_mom * fac["resid_mom_rank"]

        result[code] = fac

    return result
