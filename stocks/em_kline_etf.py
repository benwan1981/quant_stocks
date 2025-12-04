# em_kline_etf.py
import requests
import pandas as pd


def fetch_em_kline(code: str,
                   market: str = None,
                   klt: int = 101,
                   fqt: int = 1,
                   beg: str = "19900101",
                   end: str = "20500101",
                   limit: int = 100000) -> pd.DataFrame:
    """
    从东方财富获取历史K线（股票/ETF通用）.

    参数
    ----
    code : str
        证券代码，如 '159892'
    market : str or None
        市场标识:
        - 若为 None，则自动根据代码猜测:
          6/5/9 开头 -> '1' (沪市)
          0/1/2/3 开头 -> '0' (深市)
        - 也可手工传 '1' 或 '0'
    klt : int
        K线类型:
        - 1, 5, 15, 30, 60 = 分钟
        - 101 = 日K
        - 102 = 周K
        - 103 = 月K
    fqt : int
        复权方式:
        - 0 不复权
        - 1 前复权
        - 2 后复权
    beg, end : str
        开始/结束日期，格式 'YYYYMMDD'
    limit : int
        最多返回多少条（有些host带这个参数）

    返回
    ----
    DataFrame:
        index：DatetimeIndex（日期升序）
        列：['open','close','high','low','volume','amount',
             'change_pct','change_amt','amplitude','turnover_rate']
        若失败，返回空 DataFrame
    """
    if market is None:
        # 简单根据代码判断市场
        if code.startswith(("5", "6", "9")):
            market = "1"  # 沪
        else:
            market = "0"  # 深

    secid = f"{market}.{code}"

    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": klt,
        "fqt": fqt,
        "beg": beg,
        "end": end,
        "lmt": limit,
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://quote.eastmoney.com/",
    }

    r = requests.get(url, params=params, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()

    if not data or data.get("data") is None or "klines" not in data["data"]:
        print("⚠️ 东方财富无数据或被风控：", data)
        return pd.DataFrame()

    klines = data["data"]["klines"]
    rows = [k.split(",") for k in klines]

    cols = [
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "change_pct",
        "change_amt",
        "amplitude",
        "turnover_rate",
    ]
    df = pd.DataFrame(rows, columns=cols)

    # 类型转换
    num_cols = [
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "change_pct",
        "change_amt",
        "amplitude",
        "turnover_rate",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    return df