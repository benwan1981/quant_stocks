# common/gm_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]


def load_gm_ohlcv(
    path: PathLike,
    *,
    ensure_sort: bool = True,
) -> pd.DataFrame:
    """
    将“原始 gm CSV”（history 直接 to_csv）转成统一格式：
        date, open, high, low, close, volume

    约定原始 CSV 至少包含：
        - eob: 结束时间（如 "2025-01-02 15:00:00"）
        - open, high, low, close, volume

    兼容：
        - 老文件如果已经有 'date' 列、但没有 'eob'，会直接用 'date'。
    
    返回:
        DataFrame，列为:
            date (Timestamp, 只保留日期)
            open, high, low, close, volume (float)
    """
    path = Path(path)
    df_raw = pd.read_csv(path)

    # 1) 识别时间列：优先 eob，没有 eob 再看 date（兼容旧文件）
    if "eob" in df_raw.columns:
        dt_series = pd.to_datetime(df_raw["eob"])
    elif "date" in df_raw.columns:
        dt_series = pd.to_datetime(df_raw["date"])
    else:
        raise ValueError(
            f"{path} 既没有 'eob' 也没有 'date' 列，无法转换为标准格式"
        )

    # === 关键：如果有时区，先去掉时区 ===
    tz = getattr(getattr(dt_series, "dt", None), "tz", None)
    if tz is not None:
        # 把带时区时间转成“本地时间的 naive datetime”
        dt_series = dt_series.dt.tz_convert(None)

    df = pd.DataFrame()
    # 只保留日期部分（去掉时分秒），统一叫 date
    df["date"] = dt_series.dt.normalize()

    required_cols = ["open", "high", "low", "close", "volume"]
    
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"{path} 缺少必须字段: {missing}")

    for col in required_cols:
        df[col] = df_raw[col].astype(float)

    if ensure_sort:
        df = df.sort_values("date").reset_index(drop=True)

    return df


def load_gm_ohlcv_by_code(
    code: str,
    data_dir: PathLike = "./data/gm_equity",
    *,
    ensure_sort: bool = True,
) -> tuple[pd.DataFrame, Path]:
    data_dir = Path(data_dir)
    candidates = sorted(data_dir.glob(f"{code}_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"{data_dir} 下找不到以 {code}_ 开头的 gm csv 文件")

    path = candidates[0]
    df = load_gm_ohlcv(path, ensure_sort=ensure_sort)
    return df, path

