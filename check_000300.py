# check_000300.py
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
pre_path = ROOT_DIR / "precomputed" / "prices.parquet"

print("读取:", pre_path)
prices = pd.read_parquet(pre_path)
print("列数量:", len(prices.columns))
print("前 10 列:", list(prices.columns[:10]))
print("'000300' in columns? ->", "000300" in prices.columns)
