# download_universe_em.py
"""
批量从东方财富下载 A 股 / ETF 日线数据，保存为 CSV：
格式兼容 practice_single_stock.py 里的 load_data_from_csv。

注意：需要安装 requests 和 pandas：
    pip install requests pandas
"""

import requests
import pandas as pd
import time
from pathlib import Path


# ===== 0. 配置区：股票池 & 输出目录 =====

UNIVERSE = [
    # 金融
    "601939",   # 建设银行
    "600036",   # 招商银行
    "601318",   # 中国平安

    # 消费 / 医药
    "600519",   # 贵州茅台
    "000858",   # 五粮液
    "600276",   # 恒瑞医药

    # 成长 / 新能源
    "300750",   # 宁德时代
    "002594",   # 比亚迪
    "601012",   # 隆基绿能

    # 指数 / ETF
    "510300",   # 沪深300 ETF
    "159915",   # 创业板 ETF
    "159892",   # 海外科技相关 ETF
]

DATA_DIR = Path("./data")          # 输出目录
KLT = 101                          # 101 = 日K
FQT = 1                            # 1 = 前复权 (0=不复权，2=后复权)
BEG = "0"                          # 从最早开始
END = "20500000"                   # 到很远的未来


# ===== 1. 工具函数：code -> secid 市场标识 =====

def code_to_secid(code: str) -> str:
    """
    东方财富 secid 格式: {market}.{code}
    market:
        1 = 上海（以 5/6/9 开头）
        0 = 深圳（其余）
    """
    code = code.strip()
    if code[0] in ("5", "6", "9"):
        market = "1"
    else:
        market = "0"
    return f"{market}.{code}"


# ===== 2. 核心下载函数 =====

def fetch_em_kline(code: str,
                   klt: int = KLT,
                   fqt: int = FQT,
                   beg: str = BEG,
                   end: str = END,
                   retry: int = 3,
                   pause: float = 0.5) -> pd.DataFrame:
    """
    使用东方财富 push2his 接口获取 K 线。
    返回 DataFrame：列为 [date, open, high, low, close, volume]，按日期升序。
    """
    secid = code_to_secid(code)
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

    params = {
        "secid": secid,
        "klt": klt,
        "fqt": fqt,
        "beg": beg,
        "end": end,
        "fields1": "f1,f2,f3,f4,f5,f6",      # 头部字段（不用）
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62",  # k线字段
    }

    last_err = None
    for _ in range(retry):
        try:
            print(f"📡 请求 {code} ({secid}) ...")
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not data or data.get("data") is None:
                print(f"❌ {code} 返回 data=None，可能被风控或代码无效")
                last_err = RuntimeError(f"data is None, rc={data.get('rc')}")
                time.sleep(pause)
                continue

            klines = data["data"].get("klines")
            if not klines:
                print(f"❌ {code} 未返回 klines")
                last_err = RuntimeError(f"klines is empty")
                time.sleep(pause)
                continue

            # klines 每项形如：
            # "2025-01-02,9.50,9.60,9.70,9.40,123456,xxx,..."
            rows = []
            for item in klines:
                parts = item.split(",")
                trade_date = parts[0]
                open_p = float(parts[1])
                close_p = float(parts[2])
                high_p = float(parts[3])
                low_p = float(parts[4])
                vol = float(parts[5])  # 成交量（东方财富一般是手）

                rows.append({
                    "date": trade_date,
                    "open": open_p,
                    "high": high_p,
                    "low": low_p,
                    "close": close_p,
                    "volume": vol,
                })

            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            return df

        except Exception as e:
            print(f"⚠️ {code} 请求出错: {e}")
            last_err = e
            time.sleep(pause)

    raise RuntimeError(f"获取 {code} K线失败: {last_err}")


# ===== 3. 主流程：批量下载并保存 CSV =====

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for code in UNIVERSE:
        try:
            df = fetch_em_kline(code)
        except Exception as e:
            print(f"❌ {code} 下载失败: {e}")
            continue

        if df.empty:
            print(f"⚠️ {code} 数据为空，跳过保存")
            continue

        out_path = DATA_DIR / f"{code}_D_qfq.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"✅ {code} 共 {len(df)} 条记录，已保存到 {out_path}")

        # 稍微等一下，避免请求太快
        time.sleep(0.3)


if __name__ == "__main__":
    main()
