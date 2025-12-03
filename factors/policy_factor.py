# factors/policy_factor.py
# -*- coding: utf-8 -*-
"""
政策因子：
- 读取 config/policy_themes.csv（五年计划 + 主题）
- 读取 config/policy_stock_mapping.csv（个股 / ETF 与主题的静态映射）
- 对某只股票，在每个交易日给出一个 policy_score（越大代表政策“风口”越强）

设计简单版本：
policy_score = sum_over_active_themes( importance * exposure_level )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---- 路径工具 ----

ROOT_DIR = Path(__file__).resolve().parents[1]  # 项目根目录
CONFIG_DIR = ROOT_DIR / "config"

THEME_CSV = CONFIG_DIR / "policy_themes.csv"
MAP_CSV = CONFIG_DIR / "policy_stock_mapping.csv"


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[policy_factor] ⚠️ 未找到配置文件: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[policy_factor] ⚠️ 读取 {path} 失败: {e}")
        return None


def load_policy_tables():
    """
    读取两个配置表，如果缺失就返回 None，让上层优雅降级。
    """
    df_theme = _safe_read_csv(THEME_CSV)
    df_map = _safe_read_csv(MAP_CSV)

    if df_theme is None or df_map is None:
        return None, None

    # 规范列名格式
    for col in ("plan_start", "plan_end"):
        if col in df_theme.columns:
            df_theme[col] = pd.to_datetime(df_theme[col])

    for col in ("start_date", "end_date"):
        if col in df_map.columns:
            df_map[col] = pd.to_datetime(df_map[col], errors="coerce")

    return df_theme, df_map


def attach_policy_factor(
    df: pd.DataFrame,
    code: str,
    market: str = "SH",
    symbol: Optional[str] = None,
    col_name: str = "policy_score",
) -> pd.DataFrame:
    """
    给单只股票的因子表 df（按日期索引）增加一列政策因子 policy_score。

    参数
    ----
    df : DataFrame
        至少要有 DatetimeIndex（date），其它列不限制。
    code : str
        证券代码（不带市场），如 '601939'
    market : str
        'SH' / 'SZ' / 'INDEX' / 'ETF' 等，用来和 mapping 中的 market 匹配
    symbol : str | None
        掘金标准代码：如 'SHSE.601939'。不传也没关系，我们主要用 code + market。
    col_name : str
        输出列名，默认 'policy_score'

    返回
    ----
    new_df : DataFrame
        原 df 的拷贝，多了一列 policy_score（float）
    """
    df = df.copy()
    # 先准备一列默认 0
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    dates = df.index

    df[col_name] = 0.0

    df_theme, df_map = load_policy_tables()
    if df_theme is None or df_map is None:
        # 没有配置，直接返回 0 因子
        print("[policy_factor] ⚠️ 未加载到政策配置文件，policy_score 全部为 0")
        return df

    # 过滤出本标的的映射行
    m = df_map.copy()
    m["code"] = m["code"].astype(str).str.strip()
    m["market"] = m["market"].astype(str).str.upper()

    rows = m[(m["code"] == str(code)) & (m["market"] == market.upper())].copy()
    if symbol is not None and rows.empty:
        rows = m[m["symbol"].astype(str).str.upper() == symbol.upper()].copy()

    rows["theme_code"] = rows["theme_code"].astype(str).str.upper()     

    if rows.empty:
        # 没有映射，就认为这票和政策无关，保持 0
        # 这里不报错，只提醒一下
        print(f"[policy_factor] ℹ️ {market}.{code} 未在 policy_stock_mapping.csv 中找到映射，policy_score=0")
        return df

    # 主题表规范化
    t = df_theme.copy()
    if "importance" not in t.columns:
        t["importance"] = 1.0
    else:
        t["importance"] = t["importance"].fillna(1).astype(float)

    # 为了方便 join，保证 theme_code 小写/统一
    t["theme_code"] = t["theme_code"].astype(str).str.upper()
    rows["theme_code"] = rows["theme_code"].astype(str).str.upper()

    # merge 主题，得到每条映射对应的 plan_start/plan_end/importance
    merged = rows.merge(t, on="theme_code", how="left", suffixes=("_map", "_theme"))

    if merged.empty:
        print(f"[policy_factor] ℹ️ {market}.{code} 找到映射但没有对应主题，policy_score=0")
        return df

    # 开始按行叠加 policy_score
    score = np.zeros(len(dates), dtype=float)

    for _, r in merged.iterrows():
        exposure = float(r.get("exposure_level", 0) or 0)
        importance = float(r.get("importance", 1.0) or 1.0)

        # 股票侧有效期
        start_map = r.get("start_date")
        end_map = r.get("end_date")

        # 五年计划侧有效期
        plan_start = r.get("plan_start")
        plan_end = r.get("plan_end")

        # 合并两个窗口：取交集
        # 默认给一个很大的范围
        eff_start = pd.Timestamp("1900-01-01")
        eff_end = pd.Timestamp("2100-01-01")

        for d in (start_map, plan_start):
            if pd.notna(d):
                eff_start = max(eff_start, pd.Timestamp(d))

        for d in (end_map, plan_end):
            if pd.notna(d):
                eff_end = min(eff_end, pd.Timestamp(d))

        if eff_start > eff_end:
            # 没有交集，跳过
            continue

        mask = (dates >= eff_start) & (dates <= eff_end)
        if not mask.any():
            continue

        # 这里的打分逻辑可以以后再调，现在先 simplest:
        #   单条贡献 = importance * exposure
        contrib = importance * exposure
        score[mask] += contrib

    df[col_name] = score
    return df
