# backtest/precompute_data.py
# -*- coding: utf-8 -*-
"""
从 data/gm_HS300_equity 下读取所有掘金日线 CSV，
生成对齐后的价格矩阵和收益矩阵：

- precomputed/prices.parquet   (index=date, columns=code)
- precomputed/returns.parquet  (同维度，pct_change 后)

用法：
    python -m backtest.precompute_data
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "gm_HS300_equity"
OUT_DIR = ROOT_DIR / "precomputed"


def _load_one_csv(path: Path) -> pd.Series:
    """读取单个掘金 CSV，返回收盘价序列（index=date, name=code）。"""
    df = pd.read_csv(path)
    # 兼容列名：date / eob
    if "date" in df.columns:
        df["dt"] = pd.to_datetime(df["date"])
    elif "eob" in df.columns:
        df["dt"] = pd.to_datetime(df["eob"]).dt.tz_localize(None)
    else:
        df["dt"] = pd.to_datetime(df.iloc[:, 0])

    df = df.sort_values("dt")
    code = str(path.name.split("_", 1)[0])
    s = df.set_index("dt")["close"].astype("float64")
    s.name = code
    return s


def build_price_and_return_panel(raw_dir: Path) -> pd.DataFrame:
    """扫描目录构建价格矩阵和收益矩阵，并保存 parquet。"""
    out_pre = OUT_DIR
    out_pre.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        raise FileNotFoundError(f"原始数据目录不存在: {raw_dir}")

    series_dict: Dict[str, pd.Series] = {}

    for p in sorted(raw_dir.glob("*.csv")):
        # 只处理包含 _D_ 的日线文件
        if "_D_" not in p.name:
            continue
        s = _load_one_csv(p)
        series_dict[s.name] = s

    if not series_dict:
        raise RuntimeError(f"在 {raw_dir} 下未找到 *_D_*.csv 文件")

    # 统一日期索引
    all_dates = sorted(set().union(*[s.index for s in series_dict.values()]))
    idx = pd.DatetimeIndex(all_dates, name="date")

    # ✅ 一次性构建价格矩阵，避免循环 insert 造成碎片化
    reindexed = {code: s.reindex(idx) for code, s in series_dict.items()}
    prices = pd.DataFrame(reindexed, index=idx)

    # 前向填充偶发的缺口
    prices = prices.ffill()

    returns = prices.pct_change()

    prices.to_parquet(out_pre / "prices.parquet")
    returns.to_parquet(out_pre / "returns.parquet")

    print(f"保存 prices.parquet: {prices.shape}")
    print(f"保存 returns.parquet: {returns.shape}")
    return prices

def main():
    print(f"读取原始 CSV 目录: {RAW_DIR}")
    build_price_and_return_panel(RAW_DIR)


if __name__ == "__main__":
    main()
