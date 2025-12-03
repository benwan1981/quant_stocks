# download/gm_download_all.py
# -*- coding: utf-8 -*-
"""
统一的掘金数据下载模块：
- A股个股 / 指数日线
- 股指期货日线

输出统一为 CSV，格式：
    date, open, high, low, close, volume

依赖：
    pip install gm.api pandas

配置：
    在项目根目录的 config/config.py 中配置 GM_TOKEN，例如：
        GM_TOKEN = "你的token"
"""

from __future__ import annotations

import os
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from gm.api import *

# ========= 让 Python 能找到 config 包 =========
ROOT_DIR = Path(__file__).resolve().parents[1]   # 项目根目录
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.config import GM_TOKEN  # 你的掘金 token
from common import ensure_utf8_filename


# ========= 公共工具函数 =========

def init_gm() -> None:
    """初始化掘金 SDK"""
    if not GM_TOKEN:
        raise RuntimeError("请先在 config/config.py 里设置 GM_TOKEN")
    set_token(GM_TOKEN)


def sanitize_name_for_filename(name: str) -> str:
    """
    把中文名 / 英文名变成适合文件名的形式：
    - 去掉前后空格
    - 去掉空格
    - 去掉不适合文件名的符号 / \ : * ? " < > |
    """
    name = (name or "").strip()
    name = name.replace(" ", "")
    name = re.sub(r'[\\/:*?"<>|]', "", name)
    return ensure_utf8_filename(name or "UNKNOWN")


def normalize_symbol(code: str, market: Optional[str] = None) -> str:
    """
    股票 / 指数代码标准化：
    - "600519"  -> "SHSE.600519"
    - "000001"  -> "SZSE.000001"
    - "SHSE.000300" 原样返回
    market:
        - 显式指定 "SH"/"SHSE" / "SZ"/"SZSE" 时强制用该市场
        - 不指定时，按首位 5/6/9 -> 上交所，其余深交所
    """
    code = code.strip().upper()
    if "." in code:   # 已经是标准格式
        return code

    if market is not None:
        m = market.upper()
        if m in ("SH", "SHSE"):
            prefix = "SHSE"
        elif m in ("SZ", "SZSE"):
            prefix = "SZSE"
        else:
            raise ValueError(f"无法识别的 market: {market}")
    else:
        prefix = "SHSE" if code.startswith(("5", "6", "9")) else "SZSE"

    return f"{prefix}.{code}"


def get_symbol_cn_name(symbol: str) -> str:
    """
    从掘金拿 sec_name（适用于股票、指数、期货等，只要在 get_instruments 里能查到）
    symbol 形如 "SHSE.600519" / "CFFEX.IF2501"
    """
    inst_df = get_instruments(symbols=symbol, df=True)
    if inst_df is None or inst_df.empty:
        return "UNKNOWN"
    raw_name = str(inst_df.iloc[0].get("sec_name", "") or "")
    return sanitize_name_for_filename(raw_name)


# ========= 个股 / 指数 日线 =========

def download_daily_equity(
    code: str,
    start_date: str = "2005-01-01",
    end_date: Optional[str] = None,
    out_dir: str = "./data/gm_equity",
    market: Optional[str] = None,
    adjust=ADJUST_PREV,      # ⭐ 依然默认前复权（这是请求参数，不算“后处理”）
) -> str:
    """
    下载单只 A 股 / 指数的日线数据（原始掘金字段原样保存）

    参数：
        code:   "600519" / "000300" / "SHSE.000300"
        start_date: "YYYY-MM-DD"
        end_date:   "YYYY-MM-DD"，默认为今天
        out_dir:    输出目录
        market:     可选 "SH"/"SZ"
        adjust:     复权方式，默认 ADJUST_PREV（前复权）
    """
    init_gm()

    symbol = normalize_symbol(code, market=market)
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    start_time = start_date + " 09:30:00"
    end_time = end_date + " 15:00:00"

    print(f"📡 下载 {symbol} 日线(前复权): {start_date} ~ {end_date}")

    df = history(
        symbol=symbol,
        frequency="1d",
        start_time=start_time,
        end_time=end_time,
        fields="eob,open,high,low,close,volume",
        adjust=adjust,      # ⭐ 这里还是前复权
        df=True,
        # ❌ 不再 fill_missing，不再做任何本地加工
        # fill_missing="last",
    )

    if df is None or df.empty:
        raise RuntimeError(f"{symbol} 在 {start_date}~{end_date} 没有拿到数据")

    os.makedirs(out_dir, exist_ok=True)

    raw_code = symbol.split(".")[-1]     # 600519 / 000300
    cn_name = ensure_utf8_filename(get_symbol_cn_name(symbol))  # 贵州茅台 / 沪深300

    # 名字你可以按喜好来，我这里仍然标记 qfq_gm 方便以后识别
    file_name = ensure_utf8_filename(f"{raw_code}_{cn_name}_D_qfq_gm.csv")
    out_path = ensure_utf8_filename(os.path.join(out_dir, file_name))

    # ⭐ 关键：直接把掘金的 df 原样落盘
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存到: {out_path}, 共 {len(df)} 行")
    return out_path

def download_intraday_equity(
    code: str,
    frequency: str = "60s",          # ⭐ 默认 1 分钟
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    out_dir: str = "./data/gm_intraday",
    market: Optional[str] = None,
    adjust=ADJUST_PREV,              # ⭐ 仍然是前复权
) -> str:
    """
    下载单只 A 股 / 指数的分钟线数据（原始掘金字段原样保存）

    参数：
        code:       "600519" / "000300" / "SHSE.000300"
        frequency:  "60s"、"300s" 等
        start_date: "YYYY-MM-DD"
        end_date:   "YYYY-MM-DD"
        out_dir:    输出目录
        market:     可选 "SH"/"SZ"
        adjust:     复权方式，默认前复权
    """
    init_gm()

    symbol = normalize_symbol(code, market=market)
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    start_time = start_date + " 09:30:00"
    end_time = end_date + " 15:00:00"

    print(f"📡 下载 {symbol} 分时({frequency},前复权): {start_date} ~ {end_date}")

    df = history(
        symbol=symbol,
        frequency=frequency,
        start_time=start_time,
        end_time=end_time,
        fields="eob,open,high,low,close,volume",
        adjust=adjust,
        df=True,
        # ❌ 不设置 fill_missing，不加列、不改顺序
        # fill_missing=None,
    )

    if df is None or df.empty:
        raise RuntimeError(f"{symbol} 分时在 {start_date}~{end_date} 没有拿到数据")

    os.makedirs(out_dir, exist_ok=True)

    raw_code = symbol.split(".")[-1]
    cn_name = ensure_utf8_filename(get_symbol_cn_name(symbol))
    freq_tag = frequency.replace("s", "S")

    file_name = ensure_utf8_filename(f"{raw_code}_{cn_name}_{freq_tag}_qfq_gm.csv")
    out_path = ensure_utf8_filename(os.path.join(out_dir, file_name))

    # ⭐ 直接落盘 history 返回的 df
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ 分时已保存到: {out_path}，共 {len(df)} 行")
    return out_path


def batch_download_equity_from_csv(
    list_csv: str,
    start_date: str = "2005-01-01",
    end_date: Optional[str] = None,
    out_dir: str = "./data/gm_equity",
) -> None:
    """
    从 CSV 批量下载股票 / 指数日线

    CSV 示例（UTF-8）：
        code,name,market
        600519,贵州茅台,SH
        000300,沪深300,SH
        SZSE.399006,创业板指,
        159915,沪深300ETF,SH

    说明：
        - code: 必填，可以是裸代码(600519)、也可以是 SHSE.600519
        - name: 只是方便你看，不参与下载
        - market: 可选 SH/SZ，主要用于裸代码时手动指定市场
    """
    init_gm()

    df_list = pd.read_csv(list_csv)
    total = len(df_list)
    print(f"📃 Equity 待下载标的数量: {total}")

    for i, row in df_list.iterrows():
        raw_code = str(row.get("code", "")).strip()
        if not raw_code or raw_code.lower() == "nan":
            continue

        mkt = row.get("market", None)
        mkt = str(mkt).strip() if isinstance(mkt, str) else None
        if mkt == "":
            mkt = None

        print(f"\n==== [Equity {i+1}/{total}] 下载 {raw_code} ====")
        try:
            download_daily_equity(
                code=raw_code,
                start_date=start_date,
                end_date=end_date,
                out_dir=out_dir,
                market=mkt,
            )
        except Exception as e:
            print(f"❌ {raw_code} 下载失败: {e}")


# ========= 期货 日线 =========

def download_future_kline(
    symbol: str,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    out_dir: str = "./data/gm_futures",
) -> str:
    """
    下载单个股指期货合约的日线数据

    参数：
        symbol: 例如 "CFFEX.IF2501" / "CFFEX.IC2503"
    """
    init_gm()

    symbol = symbol.strip().upper()
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    start_time = start_date + " 09:00:00"
    end_time = end_date + " 15:15:00"

    print(f"📡 下载期货 {symbol} 日线: {start_date} ~ {end_date}")

    df = history(
        symbol=symbol,
        frequency="1d",
        start_time=start_time,
        end_time=end_time,
        fields="eob,open,high,low,close,volume",
        df=True,
        fill_missing="last",
    )

    if df is None or df.empty:
        raise RuntimeError(f"{symbol} 在 {start_date}~{end_date} 没有拿到数据")

    df = df.copy()
    df["date"] = pd.to_datetime(df["eob"]).dt.strftime("%Y-%m-%d")
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("date")

    os.makedirs(out_dir, exist_ok=True)

    code = symbol.split(".")[-1]            # IF2501
    cn_name = ensure_utf8_filename(get_symbol_cn_name(symbol))    # 沪深300指数期货 等
    file_name = ensure_utf8_filename(f"{code}_{cn_name}_FUT_D_gm.csv")
    out_path = ensure_utf8_filename(os.path.join(out_dir, file_name))

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存到: {out_path}, 共 {len(df)} 行")
    return out_path


def batch_download_futures_from_csv(
    list_csv: str,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    out_dir: str = "./data/gm_futures",
) -> None:
    """
    从 CSV 批量下载期货日线

    CSV 示例：
        code,name
        CFFEX.IF2501,沪深300IF2501
        CFFEX.IC2503,中证500IC2503
        CFFEX.IM2503,中证1000IM2503
        CFFEX.IH2503,上证50IH2503
    """
    init_gm()

    df_list = pd.read_csv(list_csv)
    total = len(df_list)
    print(f"📃 Futures 待下载合约数量: {total}")

    for i, row in df_list.iterrows():
        raw_code = str(row.get("code", "")).strip().upper()
        if not raw_code or raw_code.lower() == "nan":
            continue

        print(f"\n==== [FUT {i+1}/{total}] 下载期货 {raw_code} ====")
        try:
            download_future_kline(
                symbol=raw_code,
                start_date=start_date,
                end_date=end_date,
                out_dir=out_dir,
            )
        except Exception as e:
            print(f"❌ {raw_code} 下载失败: {e}")


# ========= 示例入口 =========

if __name__ == "__main__":
    # 你可以只开其中一类，也可以两类一起跑

    # === 1. 批量下载股票 / 指数日线 ===
    equity_list_csv = "./config/gm_equity_list.csv"   # 自己维护这个列表
    if os.path.exists(equity_list_csv):
        batch_download_equity_from_csv(
            list_csv=equity_list_csv,
            start_date="1990-01-01",
            end_date=None,                # None = 截止到今天
            out_dir="./data/gm_equity",
        )
    else:
        print(f"⚠️ 未找到 equity 列表文件: {equity_list_csv}，跳过股票/指数下载")

    # === 2. 批量下载股指期货日线 ===
    futures_list_csv = "./config/gm_futures_list.csv"
    if os.path.exists(futures_list_csv):
        batch_download_futures_from_csv(
            list_csv=futures_list_csv,
            start_date="2015-01-01",
            end_date=None,
            out_dir="./data/gm_futures",
        )
    else:
        print(f"⚠️ 未找到 futures 列表文件: {futures_list_csv}，跳过期货下载")
