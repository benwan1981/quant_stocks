# -*- coding: utf-8 -*-
"""
对单只股票的收益率序列与行业/风格指数收益做一元回归，
得到残差收益序列 resid_ret。
"""

from __future__ import annotations

import pandas as pd


def compute_residual_return_series(
    stock_ret: pd.Series,
    index_ret: pd.Series,
    min_obs: int = 60,
) -> pd.Series:
    """
    输入：
      stock_ret: 股票日收益（Series，index 为日期）
      index_ret: 对应指数日收益（Series，index 为日期）
    输出：
      resid_ret: 残差收益，index 对齐 stock_ret
    实现方式：用滚动窗口计算 beta，这里先用“全样本一次回归”的近似，
    后面你可以改成真正的滚动回归。
    """
    s = stock_ret.align(index_ret, join="inner")[0]
    i = index_ret.align(stock_ret, join="inner")[0]
    df = pd.DataFrame({"s": s, "i": i}).dropna()

    if len(df) < min_obs:
        # 样本太少就直接视为 resid = s
        resid = stock_ret.copy()
        resid.loc[:] = stock_ret
        return resid

    x = df["i"].values
    y = df["s"].values

    # 最简 OLS：y = a + b x
    x_mean = x.mean()
    y_mean = y.mean()
    cov = ((x - x_mean) * (y - y_mean)).sum()
    var = ((x - x_mean) ** 2).sum()
    if var == 0:
        beta = 0.0
    else:
        beta = cov / var

    fitted = beta * df["i"]
    resid_local = df["s"] - fitted

    resid = stock_ret.copy()
    resid.loc[resid_local.index] = resid_local
    return resid
