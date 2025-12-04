# download_159892_daily.py
from em_kline_etf import fetch_em_kline
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common import ensure_utf8_filename


def main():
    code = "601939"          # 纳指类ETF
    # 159892 是深交所ETF，所以市场是 '0'，但我们也可以让函数自动识别
    df = fetch_em_kline(code=code,
                        market=None,    # 让它自动根据代码判断
                        klt=101,        # 101 = 日K
                        fqt=1,          # 1 = 前复权；若要不复权改成 0
                        beg="20211019", # 足够早的开始时间
                        end="20251114") # 足够晚的结束时间

    if df.empty:
        print("❌ 未获取到任何数据，可能 IP 被风控或代码有误")
        return

    print("✅ 获取到的记录数:", len(df))
    print(df.tail())

    out_file = ensure_utf8_filename("601939_D_qfq.csv")   # 前复权日线
    df.to_csv(out_file, encoding="utf-8-sig")
    print(f"✅ 已保存到: {out_file}")


if __name__ == "__main__":
    main()
