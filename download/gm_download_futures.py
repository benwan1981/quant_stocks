# gm_download_futures.py（示例）
import os
from datetime import datetime

import pandas as pd
from gm.api import *
from pathlib import Path
import sys

# === 加在文件靠前的位置，紧接着 import 后面 ===

# 确保能找到项目根目录下的 config 包
ROOT_DIR = Path(__file__).resolve().parents[1]   # 上上级，就是项目根目录
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.config import GM_TOKEN  # 你已经配过的 token
from common import ensure_utf8_filename


from config.config import GM_TOKEN  # 你已经配过的 token

def init_gm():
    if not GM_TOKEN:
        raise RuntimeError("请先在 config/config.py 里设置 GM_TOKEN")
    set_token(GM_TOKEN)

def download_future_kline(
    symbol: str,
    start_date: str = "2015-01-01",
    end_date: str | None = None,
    out_dir: str = "./data/gm_futures",
) -> str:
    """
    下载单个股指期货合约的日线数据
    symbol: 例如 "CFFEX.IF2501" / "CFFEX.IC2503"
    """
    init_gm()

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
    df = df[["date", "open", "high", "low", "close", "volume"]].sort_values("date")

    os.makedirs(out_dir, exist_ok=True)
    code = symbol.split(".")[-1]
    file_name = ensure_utf8_filename(f"{code}_FUT_D_gm.csv")
    out_path = ensure_utf8_filename(os.path.join(out_dir, file_name))
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存到: {out_path}, 共 {len(df)} 行")
    return out_path

if __name__ == "__main__":
    # 示例：下沪深 300 股指期货某个合约
    download_future_kline("CFFEX.IF2501", start_date="2020-01-01")
