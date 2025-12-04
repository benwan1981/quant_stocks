# universe/universe_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def parse_code_name_from_filename(path: Path) -> Tuple[str, str]:
    """
    从文件名解析出 code 和 name：
        600941_中国移动_D_qfq_gm.csv  ->  ("600941", "中国移动")
        000001_平安银行_D_qfq_gm.csv  ->  ("000001", "平安银行")
    """
    stem = path.stem  # 600941_中国移动_D_qfq_gm
    parts = stem.split("_")
    if len(parts) < 2:
        return stem, ""
    code = parts[0]
    name = parts[1]
    return code, name


def scan_universe_from_dir(
    data_dir: str,
    pattern: str = "*.csv",
    freq_tag: str = "_D_",   # 只把日线纳入股票池
) -> Dict[str, Path]:
    """
    从 data_dir 扫描所有 CSV 文件，返回 股票代码 -> 文件路径 的映射。
    默认只收包含 "_D_" 的文件名（排除分钟线）。
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"股票池目录不存在: {root}")

    mapping: Dict[str, Path] = {}
    for fp in root.glob(pattern):
        if freq_tag not in fp.name:
            continue
        code, _ = parse_code_name_from_filename(fp)
        if not code:
            continue
        mapping[code] = fp

    return mapping


def build_price_panel_from_dir(
    data_dir: str,
    freq_tag: str = "_D_",
) -> pd.DataFrame:
    """
    直接用“某个目录下的所有日线CSV文件”构建收盘价 panel：
        price_panel: index=日期, columns=代码（值=收盘价）
    """
    code_to_path = scan_universe_from_dir(data_dir, freq_tag=freq_tag)

    series_dict = {}
    for code, path in code_to_path.items():
        df = pd.read_csv(path)
        # 兼容 gm 原始日线格式（eob）或已经转好的 format（date）
        if "date" in df.columns:
            dt = pd.to_datetime(df["date"])
        elif "eob" in df.columns:
            dt = pd.to_datetime(df["eob"]).dt.date
            dt = pd.to_datetime(dt)
        else:
            raise ValueError(f"{path} 既没有 date 也没有 eob 列")

        df = df.copy()
        df["date"] = dt
        df = df.set_index("date").sort_index()

        if "close" not in df.columns:
            raise ValueError(f"{path} 缺少 close 列")

        s_close = df["close"].astype(float)
        s_close.name = code
        series_dict[code] = s_close

    panel = pd.concat(series_dict, axis=1)
    panel.index.name = "date"
    panel = panel.sort_index()
    return panel