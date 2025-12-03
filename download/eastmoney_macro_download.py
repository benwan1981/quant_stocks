# download/eastmoney_macro_download.py
# -*- coding: utf-8 -*-
"""
通用：用东财 secid 下载 K 线（指数 / 期货 / 汇率 都能用）

接口：
  https://push2his.eastmoney.com/api/qt/stock/kline/get

输出：
  CSV：date, open, high, low, close, volume
"""

import os
import time
import json
import requests
import pandas as pd
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common import ensure_utf8_filename


def download_eastmoney_kline_by_secid(
    secid: str,
    out_csv: str,
    klt: int = 101,
    fqt: int = 0,
):
    """
    通用下载函数（按 secid）：

    参数
    ----
    secid : str
        东财内部代码，例如：
        - "100.HSI"      恒生指数
        - "134.HSI_M"    恒生期货主连
        - "100.DINIW"    （假设）美元指数
        - "100.USDCNH"   （假设）美元兑离岸人民币
    out_csv : str
        输出的 CSV 路径
    klt : int
        K 线周期：
          101 = 日K
          102 = 周K
          103 = 月K
    fqt : int
        复权方式：
          0 = 不复权
          1 = 前复权
          2 = 后复权
        对指数/期货/汇率通常用 0 即可
    """
    url = (
        "https://push2his.eastmoney.com/api/qt/stock/kline/get?"
        f"secid={secid}"
        "&fields1=f1,f2,f3,f4,f5"
        "&fields2=f51,f52,f53,f54,f55,f56,f57,f58"
        f"&klt={klt}"
        f"&fqt={fqt}"
        "&beg=0&end=20500000"
    )

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    print(f"📡 请求: {url}")
    r = requests.get(
        url,
        headers=headers,
        timeout=10,
        proxies={"http": None, "https": None},  # 避免走系统代理
    )

    # 简单检查
    try:
        data = r.json()
    except json.JSONDecodeError as e:
        raise RuntimeError(f"{secid} 返回的不是合法 JSON: {e}")

    rc = data.get("rc")
    msg = data.get("msg", "")
    klines = (data.get("data") or {}).get("klines")

    print(f"  ↳ rc={rc}, msg={msg}, kline条数={0 if klines is None else len(klines)}")

    if rc != 0 or not klines:
        raise RuntimeError(f"{secid} 无数据或接口错误: rc={rc}, msg={msg}, data={data.get('data')}")

    rows = []
    for line in klines:
        # 每一条形如：
        # "2025-11-14,收盘,开盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率(部分字段可能缺失)"
        parts = line.split(",")
        dt = parts[0]
        close_p = float(parts[1])
        open_p  = float(parts[2])
        high_p  = float(parts[3])
        low_p   = float(parts[4])
        vol     = float(parts[5]) if len(parts) > 5 and parts[5] != "" else 0.0

        rows.append([dt, open_p, high_p, low_p, close_p, vol])

    df = pd.DataFrame(
        rows,
        columns=["date", "open", "high", "low", "close", "volume"],
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    out_csv = ensure_utf8_filename(out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存到: {out_csv}, 共 {len(df)} 行")


def main():
    """
    在这里配置你要下载的目标：
      key        = 你自己起的名字（用于文件名）
      value.secId = 你在东财页面 URL 里看到的 nid 值
    """
    out_dir = "./data/eastmoney_macro"
    os.makedirs(out_dir, exist_ok=True)

    TARGETS = {
        # 👇 下面这些 secid 一定要你自己在网页上确认后再填
        # 例子写法，仅作占位符，用你自己从 URL 拿到的 nid 替换
        "CNY_USD":  "133.USDCNH",   # TODO: 用实际的 nid 替换
        "USD_INDEX": "100.UDI",   # TODO: 用实际的 nid 替换
        "JPY_USD":  "119.USDJPY",   # TODO: 用实际的 nid 替换
    }

    for name, secid in TARGETS.items():
        out_csv = os.path.join(out_dir, f"{name}_D_eastmoney.csv")
        try:
            download_eastmoney_kline_by_secid(
                secid=secid,
                out_csv=out_csv,
                klt=101,   # 日线
                fqt=0,     # 指数/汇率用不复权
            )
        except Exception as e:
            print(f"❌ 下载 {name} ({secid}) 失败: {e}")


if __name__ == "__main__":
    main()
