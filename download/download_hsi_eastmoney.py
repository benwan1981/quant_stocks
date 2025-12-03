# download/download_hsi_eastmoney.py
# -*- coding: utf-8 -*-
"""
从东方财富下载恒生相关日线数据（JSON K 线接口）：
- 恒生主力期货：secid=134.HSI_M
- 恒生指数现货：secid=100.HSI

输出格式：date, open, high, low, close, volume
"""

import os
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common import ensure_utf8_filename


SECIDS = {
    "hsi_main": "134.HSI_M",  # 恒生主力期货
    "hsi_index": "100.HSI",   # 恒生指数现货
}


def _build_url(secid: str) -> str:
    """
    构造东方财富日 K 线接口 URL
    klt=101 日线, fqt=1 前复权（指数/期货实际没什么复权概念，这里保持一致）
    """
    return (
        "https://push2his.eastmoney.com/api/qt/stock/kline/get?"
        f"secid={secid}"
        "&fields1=f1,f2,f3,f4,f5"
        "&fields2=f51,f52,f53,f54,f55,f56,f57,f58"
        "&klt=101"
        "&fqt=1"
        "&beg=0&end=20500000"
    )


def download_hsi_from_eastmoney(
    kind: str = "hsi_main",
    out_csv: str | None = None,
) -> str:
    """
    下载恒生相关日线数据。

    kind:
        - "hsi_main"  恒生主力期货（推荐你做期货回测用）
        - "hsi_index" 恒生指数现货（做指数对比/大盘风格用）

    out_csv:
        输出文件路径；为 None 时自动按 kind 命名：
        - ./data/hk/HSI_MAIN_D_eastmoney.csv
        - ./data/hk/HSI_INDEX_D_eastmoney.csv
    """
    if kind not in SECIDS:
        raise ValueError(f"未知 kind={kind}，可选: {list(SECIDS.keys())}")

    secid = SECIDS[kind]
    url = _build_url(secid)

    if out_csv is None:
        os.makedirs("./data/hk", exist_ok=True)
        if kind == "hsi_main":
            out_csv = "./data/hk/HSI_MAIN_D_eastmoney.csv"
        else:
            out_csv = "./data/hk/HSI_INDEX_D_eastmoney.csv"

    out_csv = ensure_utf8_filename(out_csv)

    headers = {"User-Agent": "Mozilla/5.0"}

    print(f"📡 请求: {url}")
    r = requests.get(
        url,
        headers=headers,
        timeout=10,
        proxies={"http": None, "https": None},  # 不走系统代理，防止 proxy 干扰
    )
    r.raise_for_status()
    data = r.json()

    klines = data.get("data", {}).get("klines", [])
    if not klines:
        raise RuntimeError(f"无数据或格式异常: {data}")

    rows = []
    for line in klines:
        # "日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅"
        parts = line.split(",")
        dt = parts[0]
        open_p = float(parts[1])
        close_p = float(parts[2])
        high_p = float(parts[3])
        low_p = float(parts[4])
        vol = float(parts[5])
        rows.append([dt, open_p, high_p, low_p, close_p, vol])

    df = pd.DataFrame(
        rows,
        columns=["date", "open", "high", "low", "close", "volume"]
    )
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存到: {out_csv}, 共 {len(df)} 行")
    return out_csv


if __name__ == "__main__":
    # 1) 恒生主力期货
    download_hsi_from_eastmoney("hsi_main")

    # 2) 恒生指数现货（如果需要顺便下）
    download_hsi_from_eastmoney("hsi_index")
