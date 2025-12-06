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
from datetime import datetime,timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from gm.api import *

import numpy as np

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
    code = code.strip().upper()
    if "." in code:
        return code

    if market is not None:
        m = market.upper()
        # ⭐ 支持 SH、SHSE、SHSZ 都当成上交所
        if m in ("SH", "SHSE", "SHSZ"):
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
def update_daily_equity(
    code: str,
    start_date: str = "2005-01-01",
    end_date: Optional[str] = None,
    out_dir: str = "./data/gm_equity",
    market: Optional[str] = None,
) -> str:
    """
    增量更新单只 A 股 / 指数的日线数据（基于已有 CSV 补到最新）

    逻辑：
    1）在 out_dir 里按 code 前缀找历史文件：
        000001_平安银行_D_qfq_gm.csv
        000001_平安银行_D_gm.csv
        000001_*.csv
    2）读出历史文件，取最后一个交易日 last_dt
    3）从 max(start_date, last_dt+1) 开始，用 gm.history 拉取增量数据
    4）新数据按旧文件的列顺序对齐后，**直接在原文件尾部追加**，不改旧数据

    返回：
        最终写回/生成的 CSV 路径
    """
    init_gm()

    # 结束日期：默认今天
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    end_d = datetime.strptime(end_date, "%Y-%m-%d").date()

    # 输出目录
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # ========= 1. 找已有文件（兼容 *_D_qfq_gm / *_D_gm / 其他前缀） =========
    patterns = [
        f"{code}_*_D_qfq_gm*.csv",
        f"{code}_*_D_gm*.csv",
        f"{code}_*.csv",   # 兜底：只要前缀是 code 的都认
    ]

    existing_files: list[Path] = []
    for pat in patterns:
        files = sorted(out_dir_path.glob(pat))
        if files:
            existing_files = files
            break

    # ========= 2. 如果没有历史文件，退回“全量下载” =========
    if not existing_files:
        print(f"ℹ️ 未找到 {code} 的历史日线文件，将从 {start_date} 全量下载")
        return download_daily_equity(
            code=code,
            start_date=start_date,
            end_date=end_date,
            out_dir=out_dir,
            market=market,
        )

    # 取最新的那一个文件（文件名排序后最后一个）
    csv_path = existing_files[-1]
    print(f"✅ 找到已有文件: {csv_path}")

    df_old = pd.read_csv(csv_path)

    # 尝试识别时间列：优先 eob，没有则用 date（兼容你以前的 D_gm 文件）
    if "eob" in df_old.columns:
        dt_series = pd.to_datetime(df_old["eob"])
        time_col = "eob"
    elif "date" in df_old.columns:
        dt_series = pd.to_datetime(df_old["date"])
        time_col = "date"
    else:
        raise ValueError(
            f"{csv_path} 既没有 'eob' 也没有 'date' 列，无法做增量更新"
        )

    last_dt = dt_series.max().date()
    print(f"📌 {code} 本地最后交易日: {last_dt}")

    # 用户要求的最早起始日
    user_start = datetime.strptime(start_date, "%Y-%m-%d").date()
    # 增量真正起点：last_dt + 1 和 user_start 取较晚者
    incr_start_d = max(user_start, last_dt + timedelta(days=1))

    if incr_start_d > end_d:
        print(f"✅ {code} 日线已更新到 {last_dt}，无需增量下载")
        return str(csv_path)

    incr_start = incr_start_d.strftime("%Y-%m-%d")
    print(f"📡 准备增量下载 {code} 日线: {incr_start} ~ {end_date}")

    # ========= 3. 调 gm.history 拉增量 =========
    symbol = normalize_symbol(code, market=market)

    start_time = incr_start + " 09:30:00"
    end_time = end_date + " 15:00:00"

    df_new = history(
        symbol=symbol,
        frequency="1d",
        start_time=start_time,
        end_time=end_time,
        # ⭐ 为了跟之前文件一致，继续用这几个字段
        fields="eob,open,high,low,close,volume",
        df=True,
        # 不再 fill_missing，不做本地插值
        # fill_missing="last",
    )

    if df_new is None or df_new.empty:
        print(f"⚠️ {code} 在 {incr_start}~{end_date} 没有新数据，本地已最新")
        return str(csv_path)

    print(f"✅ 新增 {len(df_new)} 行")

    # ========= 4. 对齐列，然后在原文件尾部追加 =========
    df_new = df_new.copy()

    old_cols = list(df_old.columns)

    # 先补齐新表中缺少的旧列（填 NA），保证列齐全
    for c in old_cols:
        if c not in df_new.columns:
            df_new[c] = pd.NA

    # 如果新数据里有旧文件没有的列，为了不破坏老文件结构，可以丢弃这些列
    extra_cols = [c for c in df_new.columns if c not in old_cols]
    if extra_cols:
        df_new = df_new.drop(columns=extra_cols)

    # 按旧文件的列顺序重排
    df_new = df_new[old_cols]

    # 确保时间列是 datetime（虽然这里只 append，不去重，还是习惯统一一下）
    df_new[time_col] = pd.to_datetime(df_new[time_col])

    # ⭐ 关键：不改旧数据，只在文件尾部追加新行，不写表头
    df_new.to_csv(
        csv_path,
        mode="a",
        index=False,
        header=False,
        encoding="utf-8-sig",
    )

    print(f"💾 {code} 日线已在原文件尾部追加 {len(df_new)} 行: {csv_path}")
    return str(csv_path)

def update_daily_equity_file(
    csv_path: Path,
    start_date: str = "2005-01-01",
    end_date: Optional[str] = None,
    market: Optional[str] = None,
) -> str:
    """
    针对“指定的某一个 CSV 文件”做日线增量更新：
    - 保留原路径、原文件名不变
    - 只在原文件尾部追加新数据（不覆盖旧数据）

    命名约定（不改你之前的规则）：
        000001_平安银行_D_qfq_gm.csv  等
        code_名字_频率_...csv

    逻辑：
    1）从文件名提取 code
    2）读取旧数据，找到最后交易日 last_dt
    3）从 max(start_date, last_dt + 1) 开始调 gm.history
    4）对齐列顺序，只把新行追加到原 CSV 尾部
    """
    init_gm()

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到文件: {csv_path}")

    # === 结束日期，默认今天 ===
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    end_d = datetime.strptime(end_date, "%Y-%m-%d").date()

    # === 1. 从文件名提取 code ===
    code = extract_code_from_filename(csv_path)
    symbol = normalize_symbol(code, market=market)

    print(f"🔍 正在检查 {csv_path.name} ({symbol}) 是否需要增量更新...")

    # === 2. 读取旧数据，找最后交易日 ===
    df_old = pd.read_csv(csv_path)
    if df_old.empty:
        # 空文件就当没有历史，直接全量下载到这个文件（覆盖）
        print(f"⚠️ {csv_path} 是空文件，将从 {start_date} 全量下载并覆盖")
        return download_daily_equity(
            code=code,
            start_date=start_date,
            end_date=end_date,
            out_dir=str(csv_path.parent),
            market=market,
        )

    if "eob" in df_old.columns:
        time_col = "eob"
        dt_series = pd.to_datetime(df_old["eob"])
    elif "date" in df_old.columns:
        time_col = "date"
        dt_series = pd.to_datetime(df_old["date"])
    else:
        raise ValueError(f"{csv_path} 既没有 'eob' 也没有 'date' 列，无法增量更新")

    last_dt = dt_series.max().date()
    print(f"📌 当前文件最后交易日: {last_dt}")

    user_start = datetime.strptime(start_date, "%Y-%m-%d").date()
    incr_start_d = max(user_start, last_dt + timedelta(days=1))

    if incr_start_d > end_d:
        print(f"✅ {csv_path.name} 已更新至 {last_dt}，无需增量下载")
        return str(csv_path)

    incr_start = incr_start_d.strftime("%Y-%m-%d")
    print(f"📡 准备增量下载 {symbol} 日线: {incr_start} ~ {end_date}")

    # === 3. 调 gm.history 拉增量 ===
    start_time = incr_start + " 09:30:00"
    end_time = end_date + " 15:00:00"

    df_new = history(
        symbol=symbol,
        frequency="1d",
        start_time=start_time,
        end_time=end_time,
        # 保持“原始字段”风格
        fields="eob,open,high,low,close,volume",
        adjust=ADJUST_PREV,
        df=True,
    )

    if df_new is None or df_new.empty:
        print(f"⚠️ {symbol} 在 {incr_start}~{end_date} 没有新增日线数据")
        return str(csv_path)

    print(f"✅ 新增 {len(df_new)} 行，将在原文件尾部追加")

    df_new = df_new.copy()

    # === 4. 对齐列：只保证“新数据列 ⊇ 旧文件列”，多余列丢弃 ===
    old_cols = list(df_old.columns)

    # 如果旧文件用的是 'date'，而新数据只有 'eob'，这里做个兼容转换
    if ("date" in old_cols) and ("eob" in df_new.columns) and ("date" not in df_new.columns):
        df_new["date"] = pd.to_datetime(df_new["eob"]).dt.strftime("%Y-%m-%d")

    # 补齐旧文件里有但新数据里缺少的列
    for c in old_cols:
        if c not in df_new.columns:
            df_new[c] = pd.NA

    # 只保留旧文件的列顺序，保证格式完全一致
    df_new = df_new[old_cols]

    # === 5. 追加写回：不覆盖旧内容 ===
    df_new.to_csv(
        csv_path,
        mode="a",           # 追加
        index=False,
        header=False,       # 不重复写表头
        encoding="utf-8-sig",
    )
    print(f"💾 已向 {csv_path.name} 追加 {len(df_new)} 行")
    return str(csv_path)


def batch_download_equity_from_csv(
    data_dirs: list[str],
    start_date: str = "2005-01-01",   # 仅在文件不存在时，用于全量下载的起始日
    end_date: Optional[str] = None,
    out_dir: str = "./data/gm_equity",
) -> None:
    """
    （已改造）
    不再从 config CSV 读取股票池，而是从给定的数据目录集合中：
        - 扫描已保存的日线 / 分钟线 CSV 文件
        - 按文件名抽取 code（例如 600519_*.csv → code = 600519）
        - 对这些 code 调用 update_daily_equity 做“日线增量更新”

    参数：
        data_dirs: 目录列表，例如：
            ["./data/gm_equity", "./data/gm_equity_intraday"]
            既可以只给日线目录，也可以把分钟线目录一起丢进来，
            我们只是用文件名提取 code。
        start_date: 如果某个 code 还没有任何日线文件时，
                    会退化为一次完整下载，用这个起始时间。
        end_date:   下载截止日期，None = 截止到今天
        out_dir:    日线 CSV 所在目录（增量更新目标目录）
    """
    init_gm()

    codes = collect_codes_from_dirs(data_dirs)
    if not codes:
        print("⚠️ 在指定目录中没有发现任何 CSV 文件，或无法提取代码")
        return

    print(f"📃 共识别到 {len(codes)} 只标的: {', '.join(codes)}")

    for i, code in enumerate(codes, start=1):
        print(f"\n==== [Equity UPDATE {i}/{len(codes)}] 补充 {code} 日线 ====")
        try:
            # update_daily_equity 内部逻辑：
            # - 若已有日线文件：从文件最后一天 + 1 开始补
            # - 若没有：调用 download_daily_equity 做一次完整下载
            update_daily_equity(
                code=code,
                start_date=start_date,
                end_date=end_date,
                out_dir=out_dir,
                market=None,
            )
        except Exception as e:
            print(f"❌ {code} 日线补充失败: {e}")


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
    list_csv: str | None = None,
    data_dirs: list[str] | None = None,
    start_date: str = "2005-01-01",
    end_date: Optional[str] = None,
    out_dir: str = "./data/gm_equity",
) -> None:
    """
    批量下载 / 补充股票日线数据（两种来源二选一）：

    1）老方式：通过 config 里的列表 CSV（list_csv）
        - CSV 示例：
            code,name,market
            600519,贵州茅台,SH
            000300,沪深300,SH
        - 走老逻辑，方便兼容之前脚本

    2）新方式：通过 data_dirs 里已经存在的 CSV 文件名提取 code
        - 例如 ./data/gm_equity/600519_贵州茅台_D_gm.csv
               ./data/gm_equity_intraday/600519_贵州茅台_1m_gm_raw.csv
        - 统一抽取出 600519 做增量更新
    """
    init_gm()

    codes: list[str] = []

    # === 新方式：优先使用 data_dirs ===
    if data_dirs:
        codes = collect_codes_from_dirs(data_dirs)
        if not codes:
            print("⚠️ data_dirs 中未找到任何 CSV 或无法提取代码")
    # === 旧方式：退回 list_csv ===
    elif list_csv:
        df_list = pd.read_csv(list_csv)
        for _, row in df_list.iterrows():
            raw_code = str(row.get("code", "")).strip()
            if not raw_code or raw_code.lower() == "nan":
                continue
            codes.append(raw_code)
        codes = sorted(set(codes))
    else:
        print("⚠️ 既没有传 data_dirs，也没有传 list_csv，不知道从哪里取标的列表")
        return

    if not codes:
        print("⚠️ 没有任何标的需要下载 / 更新，结束")
        return

    print(f"📃 待处理标的数量: {len(codes)}")
    print("  " + ", ".join(codes))

    for i, code in enumerate(codes, start=1):
        print(f"\n==== [Equity {i}/{len(codes)}] 处理 {code} ====")
        try:
            # 这里假定你已经实现了 update_daily_equity：
            # - 如果已有对应日线文件：只补最后一天之后的数据
            # - 如果没有：退化为 download_daily_equity 全量下载
            update_daily_equity(
                code=code,
                start_date=start_date,
                end_date=end_date,
                out_dir=out_dir,
                market=None,
            )
        except Exception as e:
            print(f"❌ {code} 日线补充失败: {e}")


def update_intraday_equity(
    code: str,
    frequency: str = "1m",
    out_dir: str = "./data/gm_equity_intraday",
    market: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """
    在已有“原始 gm 分时 CSV”基础上补充最新数据。
    - 文件格式：history(df=True) 直接 to_csv，有 eob, open, high, low, close, volume 等。
    - 文件命名：600519_贵州茅台_1m_gm_raw.csv 这种。
    - 若文件不存在，则退化为一次完整下载（调用 download_intraday_equity）。
    """
    init_gm()

    symbol = normalize_symbol(code, market=market)

    today = datetime.now().date()
    if end_date is None:
        end_d = today - timedelta(days=1)  # 掘金：分时不含当天
    else:
        end_d = datetime.strptime(end_date, "%Y-%m-%d").date()
        end_d = min(end_d, today - timedelta(days=1))

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    raw_code = symbol.split(".")[-1]

    # 找现有分时文件：600519_*_1m_gm_raw.csv
    candidates = sorted(out_dir_path.glob(f"{raw_code}_*_{frequency}_gm_raw.csv"))
    if not candidates:
        print(f"ℹ️ 未找到 {raw_code} {frequency} 现有分时文件，改为完整下载")
        # 注意：这里 start_date 受 180 日限制，只能从 end_d 往前最多 180 天
        start_d = end_d - timedelta(days=180 - 1)
        return download_intraday_equity(
            code=code,
            frequency=frequency,
            start_date=start_d.strftime("%Y-%m-%d"),
            end_date=end_d.strftime("%Y-%m-%d"),
            out_dir=out_dir,
            market=market,
        )

    out_path = candidates[0]
    df_old = pd.read_csv(out_path)
    if "eob" not in df_old.columns:
        raise RuntimeError(f"{out_path} 缺少 eob 列，无法做分时增量更新")

    last_dt = pd.to_datetime(df_old["eob"]).max()
    last_date = last_dt.date()

    # 新数据从“最后一天的下一天”开始
    new_start_date = last_date + timedelta(days=1)

    # 掘金权限：end_d 往前数 180 个自然日
    gm_min_start = end_d - timedelta(days=180 - 1)
    real_start = max(new_start_date, gm_min_start)

    if real_start > end_d:
        print(f"✅ {symbol} {frequency} 分时已更新至 {last_date}，无需补充")
        return str(out_path)

    start_time = real_start.strftime("%Y-%m-%d") + " 09:30:00"
    end_time = end_d.strftime("%Y-%m-%d") + " 15:00:00"

    print(f"📡 补充下载 {symbol} {frequency} 分时: {real_start} ~ {end_d}")

    df_new = history(
        symbol=symbol,
        frequency=frequency,
        start_time=start_time,
        end_time=end_time,
        df=True,
        fill_missing="last",
    )

    if df_new is None or df_new.empty:
        print(f"⚠️ {symbol} 没有新增 {frequency} 分时数据")
        return str(out_path)

    # 对齐旧文件的列顺序，避免列不一致
    missing_cols = [c for c in df_old.columns if c not in df_new.columns]
    for c in missing_cols:
        df_new[c] = np.nan
    df_new = df_new[df_old.columns]

    df_new.to_csv(out_path, mode="a", index=False, header=False, encoding="utf-8-sig")
    print(f"✅ {symbol} {frequency} 分时已补充 {len(df_new)} 条，文件: {out_path}")
    return str(out_path)


def batch_download_equity_from_csv(
    list_csv: str | None = None,
    data_dirs: list[str] | None = None,
    start_date: str = "2005-01-01",
    end_date: Optional[str] = None,
    out_dir: str = "./data/gm_equity",
) -> None:
    """
    批量下载 / 补充股票日线数据：

    两种模式（二选一）：

    1）旧模式：list_csv
        - 从列表 CSV 读取股票池（code,name,market）
        - 调用 update_daily_equity（在 out_dir 里找 / 补 / 下）

    2）新模式：data_dirs
        - 从若干个目录中遍历已有 CSV 文件（如 000001_平安银行_D_qfq_gm.csv）
        - 针对“每一个文件”调用 update_daily_equity_file
        - 在“该文件原路径”尾部追加新数据（不覆盖旧内容）
    """
    init_gm()

    # ========= 新模式：按目录里的 CSV 文件就地补充 =========
    if data_dirs:
        csv_files: list[Path] = []

        for d in data_dirs:
            p = Path(d)
            if not p.exists():
                print(f"⚠️ 目录不存在，跳过: {p}")
                continue
            if not p.is_dir():
                print(f"⚠️ 不是目录，跳过: {p}")
                continue

            # 只处理日线文件：约定包含 "_D_"（例如 600941_中国移动_D_qfq_gm.csv）
            for fp in p.glob("*.csv"):
                if "_D_" not in fp.name:
                    # 分钟线、期货等留给其他脚本（例如 batch_update_intraday_from_csv）
                    continue
                csv_files.append(fp)

        if not csv_files:
            print("⚠️ 在 data_dirs 中没有找到任何日线 CSV（包含 '_D_' 的文件名），结束")
            return

        print(f"📃 需要增量更新的日线文件数量: {len(csv_files)}")
        for i, csv_path in enumerate(sorted(csv_files), start=1):
            print(f"\n==== [Equity FILE {i}/{len(csv_files)}] {csv_path} ====")
            try:
                update_daily_equity_file(
                    csv_path=csv_path,
                    start_date=start_date,
                    end_date=end_date,
                    market=None,   # 如需区分市场，可以后从文件名里加规则
                )
            except Exception as e:
                print(f"❌ 更新 {csv_path.name} 失败: {e}")
        return

  # ========= 旧模式：通过列表 CSV 补充 / 下载 =========
    if not list_csv:
        print("⚠️ 既没有传 data_dirs，也没有传 list_csv，不知道从哪里取标的列表")
        return

    # ⭐ 用 dtype=str，避免 000001 → 1
    df_list = pd.read_csv(list_csv, dtype=str)

    records: list[tuple[str, str | None]] = []

    for _, row in df_list.iterrows():
        raw_code = (row.get("code") or "").strip()
        if not raw_code or raw_code.lower() == "nan":
            continue

        # ⭐ 补齐 6 位，保证 000001 不被吃掉
        code = raw_code.zfill(6)

        raw_mkt = (row.get("market") or "").strip().upper()
        market = raw_mkt if raw_mkt else None

        records.append((code, market))

    if not records:
        print("⚠️ 列表 CSV 中没有有效代码")
        return

    # 去重
    records = sorted(set(records), key=lambda x: (x[0], x[1] or ""))

    print(f"📃 待处理标的数量: {len(records)}")
    print("  " + ", ".join([r[0] for r in records]))

    for i, (code, market) in enumerate(records, start=1):
        print(f"\n==== [Equity {i}/{len(records)}] 处理 {code} (market={market or '-'}) ===")
        try:
            update_daily_equity(
                code=code,
                start_date=start_date,
                end_date=end_date,
                out_dir=out_dir,
                market=market,   # ⭐ 把 CSV 里的 market 传进来
            )
        except Exception as e:
            print(f"❌ {code} 日线补充失败: {e}")


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

def extract_code_from_filename(path: Path) -> str:
    """
    从保存好的 CSV 文件名里提取代码部分。

    约定命名规则类似：
        600519_贵州茅台_D_gm.csv
        600941_中国移动_D_qfq_gm.csv
        600519_贵州茅台_1m_gm_raw.csv

    则统一取文件名第一个 "_" 之前的部分作为 code：
        -> 600519 / 600941
    """
    stem = path.stem  # 不含扩展名
    if "_" not in stem:
        return stem.strip()
    return stem.split("_", 1)[0].strip()


def collect_codes_from_dirs(data_dirs: list[str]) -> list[str]:
    """
    从一组目录中收集所有 CSV 文件，并根据文件名提取 code，去重后返回排序好的列表。
    """
    codes: set[str] = set()

    for d in data_dirs:
        p = Path(d)
        if not p.exists():
            print(f"⚠️ 目录不存在，跳过: {p}")
            continue
        if not p.is_dir():
            print(f"⚠️ 不是目录，跳过: {p}")
            continue

        for csv_path in p.glob("*.csv"):
            code = extract_code_from_filename(csv_path)
            if code:
                codes.add(code)

    return sorted(codes)


# ========= 示例入口 =========

if __name__ == "__main__":
    # 你可以只开其中一类，也可以两类一起跑

    # === 1. 批量下载股票 / 指数日线 ===
        equity_list_csv = "./config/gm_HS300_daily_list.csv"   # 自己维护这个列表
        if os.path.exists(equity_list_csv):
            batch_download_equity_from_csv(
                list_csv=equity_list_csv,
                start_date="1990-01-01",
                end_date=None,                # None = 截止到今天
                out_dir="./data/gm_HS300_equity",
            )
        else:
            print(f"⚠️ 未找到 equity 列表文件: {equity_list_csv}，跳过股票/指数下载")

    # === 2. 批量下载股指期货日线 ===
        futures_list_csv = "./config/gm_futures_list.csv"
        if os.path.exists(futures_list_csv):
            batch_download_futures_from_csv(
                list_csv=futures_list_csv,
                start_date="1990-01-01",
                end_date=None,
                out_dir="./data/gm_futures",
            )
        else:
            print(f"⚠️ 未找到 futures 列表文件: {futures_list_csv}，跳过期货下载")
    

    # === 3. 从现有数据目录自动识别标的，并补充日线 ===
data_dirs = [
    "./data/gm_159599ETF_equity",     # 比如你以后新增的目录
]

'''batch_download_equity_from_csv(
    data_dirs=data_dirs,
    start_date="1990-01-01",
    end_date=None,   # None = 截止到今天
    # out_dir 参数在 data_dirs 模式下不会用到，可以留着兼容旧调用
)'''
