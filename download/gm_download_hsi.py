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

from config.config import GM_TOKEN
from common import ensure_utf8_filename


def init_gm():
    if not GM_TOKEN:
        raise RuntimeError("请先设置 GM_TOKEN")
    set_token(GM_TOKEN)


def download_hsi_main(
    symbol="HKEX.HSI",
    start_date="2000-01-01",
    end_date=None,
    out_dir="./data/gm_futures"
):
    """
    下载恒生指数主力期货日线
    symbol:
        HKEX.HSI  恒生指数主连
        HKEX.MHI  小型恒生主连

    返回:
        保存的 CSV 路径
    """
    init_gm()

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    start_time = start_date + " 09:00:00"
    end_time = end_date + " 16:00:00"

    print(f"📡 下载恒生主力期货 {symbol}: {start_date} ~ {end_date}")

    df = history(
        symbol=symbol,
        frequency="1d",
        start_time=start_time,
        end_time=end_time,
        fields="eob,open,high,low,close,volume",
        df=True,
        fill_missing="last"
    )

    if df is None or df.empty:
        raise RuntimeError("没有获取到数据，请检查 symbol 是否正确")

    df["date"] = pd.to_datetime(df["eob"]).dt.strftime("%Y-%m-%d")
    df = df[["date", "open", "high", "low", "close", "volume"]]

    os.makedirs(out_dir, exist_ok=True)
    out_path = ensure_utf8_filename(os.path.join(out_dir, "HSI_MAIN_D_gm.csv"))
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"✅ 已保存到: {out_path}，共 {len(df)} 行")
    return out_path


if __name__ == "__main__":
    download_hsi_main()
