# download/gm_download_daily.py
# -*- coding: utf-8 -*-
"""
使用掘金 GM API 下载 A 股日线数据，保存为 CSV：
格式: date, open, high, low, close, volume

依赖:
    pip install gm.api pandas

准备:
    在 config/config.py 中填写 GM_TOKEN
"""

import os
from datetime import datetime
import sys
import pandas as pd
from gm.api import *
import re
from pathlib import Path

# —— 确保能找到项目根目录下的 config 包 ——
# gm_download_daily.py 位于: 项目根/download/gm_download_daily.py
ROOT_DIR = Path(__file__).resolve().parents[1]   # 上上级，就是项目根目录
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 现在可以稳稳地从 config/config.py 里导入 GM_TOKEN 了
from config.config import GM_TOKEN
from common import ensure_utf8_filename



# ---------- 工具：清洗中文名用于文件名 ----------

def sanitize_name_for_filename(name: str) -> str:
    """
    把股票中文名变成适合做文件名的形式：
    - 去掉空格
    - 去掉不适合文件名的符号 / \ : * ? " < > | 等
    """
    name = name.strip()
    # 去掉空格
    name = name.replace(" ", "")
    # 去掉不合法字符
    name = re.sub(r'[\\/:*?"<>|]', "", name)
    return ensure_utf8_filename(name or "UNKNOWN")


# -------- 工具函数 --------

def init_gm():
    """初始化掘金环境"""
    if not GM_TOKEN:
        raise RuntimeError("请先在 config/config.py 里设置 GM_TOKEN")
    set_token(GM_TOKEN)


def normalize_symbol(code: str, market: str | None = None) -> str:
    """
    把 '600048' -> 'SHSE.600048'
       '000001' -> 'SZSE.000001'
    如果已经是 'SHSE.600048' 这种格式就直接返回
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
        # 默认规则：5 / 6 / 9 开头是上交所，其余深交所
        prefix = "SHSE" if code.startswith(("5", "6", "9")) else "SZSE"

    return f"{prefix}.{code}"

def get_symbol_cn_name(symbol: str) -> str:
    """
    通过掘金 get_instruments 拿最新的中文名称（sec_name）
    symbol: 'SHSE.600048' 这种格式
    """
    inst_df = get_instruments(symbols=symbol, df=True)
    if inst_df is None or inst_df.empty:
        return "UNKNOWN"
    raw_name = str(inst_df.iloc[0].get("sec_name", "") or "")
    return sanitize_name_for_filename(raw_name)


def download_daily_kline(
    code: str,
    start_date: str = "1990-01-01",
    end_date: str | None = None,
    out_dir: str = "./data/gm",
) -> str:
    """
    下载单只股票的日线数据，保存为 CSV
    参数:
        code:      如 "600048" 或 "SHSE.600048"
        start_date: 起始日期 "YYYY-MM-DD"
        end_date:   结束日期 "YYYY-MM-DD"，默认到今天
        out_dir:    输出目录，默认 ./data/gm

    返回:
        保存的 csv 文件路径
    """
    init_gm()

    symbol = normalize_symbol(code)
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    # GM 的时间要带上时分秒
    start_time = start_date + " 09:30:00"
    end_time = end_date + " 15:00:00"

    print(f"📡 下载 {symbol} 日线: {start_date} ~ {end_date}")

    # fields 里 eob 是bar结束时间，用来当日期
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

    # 统一为你的回测格式: date, open, high, low, close, volume
    df = df.copy()
    df["date"] = pd.to_datetime(df["eob"]).dt.strftime("%Y-%m-%d")
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("date")

    os.makedirs(out_dir, exist_ok=True)

    # 文件名：如 600048_万科A_D_gm.csv
    raw_code = code.split(".")[-1]  # 取 '600048'
    cn_name = ensure_utf8_filename(get_symbol_cn_name(symbol))  # 从掘金拿中文名并清洗
    file_name = ensure_utf8_filename(f"{raw_code}_{cn_name}_D_gm.csv")
    out_path = ensure_utf8_filename(os.path.join(out_dir, file_name))


    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存到: {out_path}, 共 {len(df)} 行")

    return ensure_utf8_filename(out_path)

def download_batch_daily_kline(
    codes: list[str],
    start_date: str = "2005-01-01",
    end_date: str | None = None,
    out_dir: str = "./data/gm",
) -> dict[str, str]:
    """
    批量下载多只标的日线数据

    参数:
        codes:      代码列表，如 ["159599", "600048", "601939", "600519"]
        start_date: 起始日期
        end_date:   结束日期，None 表示到今天
        out_dir:    输出目录

    返回:
        { code: csv路径 } 的字典
    """
    results: dict[str, str] = {}

    for code in codes:
        try:
            path = download_daily_kline(
                code=code,
                start_date=start_date,
                end_date=end_date,
                out_dir=out_dir,
            )
            results[code] = path
        except Exception as e:
            print(f"❌ 下载 {code} 失败: {e}")

    print(f"\n✅ 批量下载完成，成功 {len(results)}/{len(codes)} 只")
    return results

def load_codes_from_csv(path: str, code_col: str | None = None) -> list[str]:
    """
    从 CSV 文件读取待下载的代码列表。

    CSV 示例（推荐）：
        code,name
        600048,万科A
        601939,建设银行
        600519,贵州茅台
        159599,中证A50ETF 

    规则：
    - 如果指定了 code_col，则使用该列；
    - 否则优先找列名 'code'，其次 '代码'；
    - 如果都没有，就默认使用第一列。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"代码列表 CSV 不存在: {p}")

    df = pd.read_csv(p, dtype=str)

    if df.empty:
        raise RuntimeError(f"CSV {p} 为空")

    # 选择 code 列
    if code_col is not None:
        if code_col not in df.columns:
            raise RuntimeError(f"CSV {p} 中找不到列: {code_col}")
        series = df[code_col]
    else:
        if "code" in df.columns:
            series = df["code"]
        elif "代码" in df.columns:
            series = df["代码"]
        else:
            # 默认用第一列
            first_col = df.columns[0]
            series = df[first_col]

    codes = (
        series.astype(str)
        .map(lambda x: x.strip())
        .replace("", pd.NA)
        .dropna()
        .tolist()
    )

    if not codes:
        raise RuntimeError(f"在 CSV {p} 中没有解析到任何代码")

    print(f"📃 从 CSV 读取到 {len(codes)} 个代码")
    return codes


# -------- 示例入口 --------

if __name__ == "__main__":
    # 代码列表 CSV：一行一只，推荐有表头 code,name
    # 例如 config/gm_daily_list.csv
    # code,name
    # 600048,万科A
    # 601939,建设银行
    # 600519,贵州茅台
    # 159599,中证A50ETF
    codes_file = ROOT_DIR / "config" / "gm_daily_list.csv"

    # 如果你的列名不是 code，可以传 code_col 参数
    codes = load_codes_from_csv(str(codes_file), code_col="code")

    download_batch_daily_kline(
        codes=codes,
        start_date="1990-01-01",
        end_date=None,          # None = 到今天
        out_dir="./data/gm",
    )

