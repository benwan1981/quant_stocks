# universe/sector_mapping.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .universe_loader import scan_universe_from_dir, parse_code_name_from_filename


def build_sector_mapping_stub(
    data_dir: str,
    out_csv: str = "./config/universe_sectors.csv",
) -> None:
    """
    根据某个目录下的日线 CSV 文件，生成一份“股票 → 板块/行业”映射表的骨架：
        code, name, sector, industry, concept

    - 如果 out_csv 已存在，则在原有基础上补充新增的 code，不会覆盖已有填好的 sector/industry。
    - 适合作为“板块分类”的唯一真相源，策略里就可以统一用它来查每个股票属于什么板块。
    """
    data_dir_path = Path(data_dir)
    if not data_dir_path.exists():
        raise FileNotFoundError(data_dir)

    # 先从目录里拿所有 code, name
    rows: List[Tuple[str, str]] = []
    for fp in data_dir_path.glob("*.csv"):
        if "_D_" not in fp.name:
            continue
        code, name = parse_code_name_from_filename(fp)
        rows.append((code, name))

    df_new = pd.DataFrame(rows, columns=["code", "name"])
    df_new = df_new.drop_duplicates(subset="code").sort_values("code")

    out_path = Path(out_csv)
    if out_path.exists():
        df_old = pd.read_csv(out_path, dtype=str)
        # 统一字段
        for col in ["code", "name", "sector", "industry", "concept"]:
            if col not in df_old.columns:
                df_old[col] = ""
        df_old["code"] = df_old["code"].astype(str).str.strip()

        # 用旧表保留已填好的 sector / industry 信息
        df_merged = df_new.merge(df_old, on="code", how="left", suffixes=("", "_old"))

        # name 以新为准，分类列若旧有值则用旧值
        for col in ["name", "sector", "industry", "concept"]:
            new_col = col
            old_col = col + "_old"
            if old_col in df_merged.columns:
                df_merged[new_col] = df_merged[new_col].fillna(df_merged[old_col])
                df_merged.drop(columns=[old_col], inplace=True)

        df_out = df_merged[["code", "name", "sector", "industry", "concept"]]
    else:
        # 新建
        df_new["sector"] = ""
        df_new["industry"] = ""
        df_new["concept"] = ""
        df_out = df_new

    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ 板块映射骨架已生成/更新: {out_path}")


def load_sector_mapping(csv_path: str = "./config/universe_sectors.csv") -> pd.DataFrame:
    """
    加载板块/行业映射表，返回 DataFrame:
        index=code, columns=[name, sector, industry, concept]
    """
    df = pd.read_csv(csv_path, dtype=str)
    df["code"] = df["code"].astype(str).str.strip()
    df = df.set_index("code")
    return df