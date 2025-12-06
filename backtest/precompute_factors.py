# backtest/precompute_factors.py
# -*- coding: utf-8 -*-
"""
从 precomputed/prices.parquet / returns.parquet 计算基础因子：

- mom10, mom60  : 价格动量 (close / close.shift(n) - 1)
- vol60         : 60 日波动率（年化）

保存到：
    precomputed/factors/mom10.parquet
    precomputed/factors/mom60.parquet
    precomputed/factors/vol60.parquet

用法：
    python -m backtest.precompute_factors
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
PRE_DIR = ROOT_DIR / "precomputed"
FAC_DIR = PRE_DIR / "factors"


def load_base_panels():
    prices = pd.read_parquet(PRE_DIR / "prices.parquet")
    returns = pd.read_parquet(PRE_DIR / "returns.parquet")
    return prices, returns


def compute_momentum(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    mom = prices / prices.shift(window) - 1.0
    return mom


def compute_volatility(ret: pd.DataFrame, window: int) -> pd.DataFrame:
    vol = ret.rolling(window).std() * np.sqrt(252.0)
    return vol


def main():
    FAC_DIR.mkdir(parents=True, exist_ok=True)

    prices, returns = load_base_panels()
    print("prices shape:", prices.shape)
    print("returns shape:", returns.shape)

    mom10 = compute_momentum(prices, 10).shift(1)
    mom60 = compute_momentum(prices, 60).shift(1)
    vol60 = compute_volatility(returns, 60).shift(1)

    mom10.to_parquet(FAC_DIR / "mom10.parquet")
    mom60.to_parquet(FAC_DIR / "mom60.parquet")
    vol60.to_parquet(FAC_DIR / "vol60.parquet")

    print("保存因子：")
    print("mom10.parquet", mom10.shape)
    print("mom60.parquet", mom60.shape)
    print("vol60.parquet", vol60.shape)


if __name__ == "__main__":
    main()
