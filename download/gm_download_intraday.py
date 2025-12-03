# -*- coding: utf-8 -*-
"""
掘金分时数据下载模块（原始数据版）：
- A股个股 / 指数 分时（1分钟、5分钟等）

特点：
- 只调用 gm.history 把 DataFrame 原样保存为 CSV
- 不新增/删除/改名任何字段，不做任何清洗
- 和 gm_download_all.py 解耦：日线/分时分开下载

依赖：
    pip install gm.api pandas

配置：
    在 config/config.py 里设置 GM_TOKEN（和 gm_download_all.py 共用）
"""

from __future__ import annotations

import os
import sys
from datetime import datetime,timedelta,date
from pathlib import Path
from typing import Optional

import pandas as pd
from gm.api import *

# ========= 把项目根目录加入 sys.path =========
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.config import GM_TOKEN
from common import ensure_utf8_filename

# 直接复用 gm_download_all 里的工具函数，避免重复造轮子
from download.gm_download_all import (
    init_gm,           # 初始化掘金 SDK
    normalize_symbol,  # 标准化代码：600519 -> SHSE.600519
    get_symbol_cn_name # 获取中文名
)


# ========= 分时下载（股票 / 指数） =========

def download_intraday_equity(
    code: str,
    frequency: str = "1m",
    start_date: str = "1990-01-01",
    end_date: Optional[str] = None,
    out_dir: str = "./data/gm_equity_intraday",
    market: Optional[str] = None,
) -> str:
    """
    下载单只 A 股 / 指数的分时数据（原始 gm.history 结果直接落盘）

    参数：
        code:       600519 / 000300 / SHSE.000300 等
        frequency:  "1m", "5m", "15m", "30m", "60m" 等
        start_date: "YYYY-MM-DD"
        end_date:   "YYYY-MM-DD"，默认今天
        out_dir:    输出目录
        market:     可选 "SH"/"SZ"，只在裸代码时用来指定市场

    返回：
        保存的 csv 文件路径
    """
    init_gm()

    symbol = symbol = normalize_symbol(code, market=market)

    # === 掘金权限：只能下「除今天外，最近 180 个自然日」的分时 ===
    today = datetime.now().date()

    # 1) 限制 end_date：最多到昨天
    if end_date is None:
        end_d = today - timedelta(days=1)
    else:
        end_d = datetime.strptime(end_date, "%Y-%m-%d").date()
        end_d = min(end_d, today - timedelta(days=1))  # 不允许 >= 今天

    # 2) 掘金允许的最早起始日：end_d 往前数 180 个自然日（包含 end_d）
    gm_min_start = end_d - timedelta(days=180 - 1)

    # 3) 用户想要的 start_date
    user_start = datetime.strptime(start_date, "%Y-%m-%d").date()

    # 4) 真正用来请求的起始日：取“用户起始日”和“权限最早日”里较晚的那一个
    real_start = max(user_start, gm_min_start)

    if real_start > end_d:
        raise ValueError(
            f"{code} 分时数据：start_date 太晚，"
            f"在权限范围内没有任何可下载数据。"
            f"建议把 start_date 调整到 {gm_min_start.isoformat()} 或更早。"
        )

    # 5) 组装成 history 需要的 start_time / end_time
    start_time = real_start.strftime("%Y-%m-%d") + " 09:30:00"
    end_time = end_d.strftime("%Y-%m-%d") + " 15:00:00"

    print(f"📡 下载 {symbol} {frequency} 分时: {real_start} ~ {end_d}")



    # 不传 fields，让 gm 返回“原始全字段”
    df = history(
        symbol=symbol,
        frequency=frequency,
        start_time=start_time,
        end_time=end_time,
        df=True,
        fill_missing="last",
    )

    if df is None or df.empty:
        raise RuntimeError(
            f"{symbol} 在 {real_start}~{end_d} 没有拿到 {frequency} 数据 "
            f"(原始请求区间为 {start_date}~{end_date or datetime.now().date()})"
        )

    # 不做任何字段处理，直接保存
    os.makedirs(out_dir, exist_ok=True)

    raw_code = symbol.split(".")[-1]  # 600519 / 000300
    cn_name = ensure_utf8_filename(get_symbol_cn_name(symbol))
    # 命名中带上 frequency，且标记为 raw，方便后面 loader 识别
    file_name = ensure_utf8_filename(f"{raw_code}_{cn_name}_{frequency}_gm_raw.csv")
    out_path = ensure_utf8_filename(os.path.join(out_dir, file_name))

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存到: {out_path}, 共 {len(df)} 行")
    return out_path


def batch_download_intraday_from_csv(
    list_csv: str,
    frequency: str = "1m",
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    out_dir: str = "./data/gm_equity_intraday",
) -> None:
    """
    从 CSV 批量下载股票 / 指数分时（原始数据）

    CSV 示例（和 gm_download_all.py 里的列表格式一致）：
        code,name,market
        600519,贵州茅台,SH
        000300,沪深300,SH
        SZSE.399006,创业板指,
        159915,沪深300ETF,SH
    """
    init_gm()

    df_list = pd.read_csv(list_csv)
    total = len(df_list)
    print(f"📃 Intraday 待下载标的数量: {total}, frequency = {frequency}")

    for i, row in df_list.iterrows():
        raw_code = str(row.get("code", "")).strip()
        if not raw_code or raw_code.lower() == "nan":
            continue

        mkt = row.get("market", None)
        mkt = str(mkt).strip() if isinstance(mkt, str) else None
        if mkt == "":
            mkt = None

        print(f"\n==== [Intraday {i+1}/{total}] 下载 {raw_code} {frequency} ====")
        try:
            download_intraday_equity(
                code=raw_code,
                frequency=frequency,
                start_date=start_date,
                end_date=end_date,
                out_dir=out_dir,
                market=mkt,
            )
        except Exception as e:
            print(f"❌ {raw_code} {frequency} 下载失败: {e}")


# ========= 示例入口 =========

if __name__ == "__main__":
    # 默认用和日线同一份股票列表
    equity_list_csv = "./config/gm_equity_list.csv"

    if os.path.exists(equity_list_csv):
        batch_download_intraday_from_csv(
            list_csv=equity_list_csv,
            frequency="1m",                   # 这里改成 "5m" / "15m" 等也可以
            #start_date='2025-05-24',
            #end_date=None,                    # None = 截止到今天
            out_dir="./data/gm_equity_intraday",
        )
    else:
        print(f"⚠️ 未找到 equity 列表文件: {equity_list_csv}，请先在 config 里准备好列表 CSV")

