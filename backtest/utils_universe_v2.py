# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from common.gm_loader import load_gm_ohlcv


def load_stock_universe_from_dir(data_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """
    从目录中读取所有 *_D_qfq_gm.csv，构造成 {code: df_ohlcv}。
    要求 df 至少包含 ["open", "close"]，index 为 date。
    """
    data_dir = Path(data_dir)
    stock_universe: Dict[str, pd.DataFrame] = {}

    for csv_path in sorted(data_dir.glob("*_D_qfq_gm.csv")):
        name = csv_path.name
        code = name.split("_", 1)[0]

        df = load_gm_ohlcv(csv_path)
        if "date" in df.columns:
            df = df.set_index("date").sort_index()
        if not {"open", "close"} <= set(df.columns):
            continue

        stock_universe[code] = df[["open", "close"]].copy()

    if not stock_universe:
        raise RuntimeError(f"在目录 {data_dir} 中未找到任何 *_D_qfq_gm.csv")

    return stock_universe


def load_hs300_index_from_universe_dir(data_dir: str | Path) -> pd.DataFrame:
    """
    尝试在 data_dir 中找到“沪深300/HS300”指数日线文件，返回 df(index=date, columns=[open, close])。
    命名兼容多种模式：
      - 000300*D*.csv
      - *沪深300*D*.csv
      - *HS300*D*.csv
    """
    data_dir = Path(data_dir)

    patterns = [
        "000300*D*qfq*.csv",
        "000300*D*.csv",
        "*沪深300*D*qfq*.csv",
        "*沪深300*D*.csv",
        "*HS300*D*qfq*.csv",
        "*HS300*D*.csv",
    ]

    candidates = []
    for pat in patterns:
        candidates.extend(data_dir.glob(pat))

    # 去重并排序
    candidates = sorted(set(candidates))

    if not candidates:
        sample = sorted(p.name for p in data_dir.glob("*.csv"))[:10]
        raise FileNotFoundError(
            f"未在 {data_dir} 找到可识别的沪深300指数文件。\n"
            f"已尝试模式: {patterns}\n"
            f"该目录下部分文件示例: {sample}"
        )

    path = candidates[0]

    df = load_gm_ohlcv(path)
    if "date" not in df.columns:
        raise ValueError(f"{path} 中找不到 date 列")
    df = df.set_index("date").sort_index()

    if not {"open", "close"} <= set(df.columns):
        raise ValueError(f"{path} 中找不到 open/close 列")

    return df[["open", "close"]].copy()


def build_index_from_universe(stock_universe: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    用成分股重建一个“等权指数”：
      - 汇总所有股票的 close
      - 做横截面平均收益
      - base = 100 累乘得到 close
      - open 用前一日 close 近似
    """
    closes = pd.DataFrame({code: df["close"] for code, df in stock_universe.items()})
    closes = closes.sort_index().ffill()

    ret = closes.pct_change().mean(axis=1).fillna(0.0)
    idx_close = (1.0 + ret).cumprod() * 100.0

    idx = pd.DataFrame(index=idx_close.index)
    idx["close"] = idx_close
    idx["open"] = idx["close"].shift(1).fillna(idx["close"])
    return idx
