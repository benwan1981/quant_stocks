# download_600048_ak_sina.py
import akshare as ak
import pandas as pd


def download_sina_daily_to_csv(
    symbol: str,
    out_csv: str,
    start_date: str = "20000101",
    end_date: str = "20500101",
    adjust: str = "qfq",
):
    """
    用 akshare 的 新浪 日线接口，下载数据并存成 CSV:
    列: date, open, high, low, close, volume

    symbol: 必须类似 'sh600048' 或 'sz000002'
    start_date, end_date: 形如 '20200101'
    adjust: '' 不复权, 'qfq' 前复权, 'hfq' 后复权
    """
    df = ak.stock_zh_a_daily(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )

    # 只取回测需要的几列
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()

    # 确保 date 是字符串或标准日期
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存到: {out_csv}, 共 {len(df)} 行")


if __name__ == "__main__":
    # 保利发展 600048，对应上交所，所以用 sh600048
    out_path = "./data/600048_D_qfq.csv"
    download_sina_daily_to_csv(
        symbol="sh600048",
        out_csv=out_path,
        start_date="20070101",   # 可以按需改时间
        end_date="20251114",     # 或者 "20500101"
        adjust="qfq",
    )
