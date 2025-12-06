# backtest/precompute_regime.py
# -*- coding: utf-8 -*-
"""
根据 precomputed/prices.parquet 中 000300（若存在）或 HS300 的 close，
预计算市场状态与波动 z-score：

- base_regime: bull / bear / neutral
- z_sigma    : 60 日波动率 z-score

保存到：
    precomputed/regime/base_regime.parquet
    precomputed/regime/z_sigma.parquet

用法：
    python -m backtest.precompute_regime
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
PRE_DIR = ROOT_DIR / "precomputed"
REG_DIR = PRE_DIR / "regime"


def _select_index_series(prices: pd.DataFrame) -> pd.Series:
    for cand in ["000300", "HS300", "沪深300"]:
        if cand in prices.columns:
            s = prices[cand].copy()
            s.name = "index_close"
            return s
    raise KeyError("prices 中未找到 000300/HS300/沪深300 对应列，请检查数据。")


def compute_base_regime(idx_close: pd.Series) -> pd.Series:
    df = idx_close.to_frame("close")
    df["ret"] = df["close"].pct_change()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma100"] = df["close"].rolling(100).mean()
    df["vol20"] = df["ret"].rolling(20).std()
    df["vol60"] = df["ret"].rolling(60).std()

    cond_bull = (df["ma20"] > df["ma100"]) & (df["vol20"] > df["vol60"])
    cond_bear = (df["ma20"] < df["ma100"]) & (df["vol20"] < df["vol60"])
    base = np.where(cond_bull, "bull", np.where(cond_bear, "bear", "neutral"))
    base = pd.Series(base, index=df.index, name="base_regime")

    # 用 t-1 信息（供 t 日使用）
    return base.shift(1)


def compute_z_sigma(idx_close: pd.Series, window: int = 60) -> pd.Series:
    ret = idx_close.pct_change()
    sigma = ret.rolling(window).std() * np.sqrt(252.0)
    mu = sigma.rolling(252).mean()
    std = sigma.rolling(252).std()
    z = (sigma - mu) / std.replace(0, np.nan)
    return z.rename("z_sigma").shift(1)


def main():
    REG_DIR.mkdir(parents=True, exist_ok=True)

    prices = pd.read_parquet(PRE_DIR / "prices.parquet")
    idx_s = _select_index_series(prices)

    base_regime = compute_base_regime(idx_s)
    z_sigma = compute_z_sigma(idx_s)

    (base_regime
        .to_frame("base_regime")
        .to_parquet(REG_DIR / "base_regime.parquet"))

    (z_sigma
        .to_frame("z_sigma")
        .to_parquet(REG_DIR / "z_sigma.parquet"))


    print("base_regime:", base_regime.value_counts(dropna=True))
    print("z_sigma 描述：")
    print(z_sigma.describe())


if __name__ == "__main__":
    main()
