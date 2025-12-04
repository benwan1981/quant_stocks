# backtest/test_factors_universe.py
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from factors import apply_factors
from universe.universe_loader import scan_universe_from_dir
from universe.sector_mapping import build_sector_mapping_stub, load_sector_mapping


DATA_DIR = "./data/gm_a_index"   # 股票池目录


def main():
    # 1) 先构建/更新一份板块映射骨架（你之后可以手工填 sector / industry）
    build_sector_mapping_stub(DATA_DIR, "./config/universe_sectors.csv")
    sector_map = load_sector_mapping("./config/universe_sectors.csv")

    # 2) 股票池 = 目录下所有日线 csv
    code_to_path = scan_universe_from_dir(DATA_DIR)

    for code, path in list(code_to_path.items())[:5]:  # 这里先测试 5 只
        print(f"\n=== 测试因子: {code}, 文件 = {path} ===")
        df = pd.read_csv(path)

        # 标准化日期
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        elif "eob" in df.columns:
            df["date"] = pd.to_datetime(df["eob"]).dt.date
            df["date"] = pd.to_datetime(df["date"])
        else:
            raise ValueError("缺少 date/eob 列")

        df = df.set_index("date").sort_index()

        # 3) 按需加载因子
        df_fac = apply_factors(df, ["ma20", "ma60", "mom_10", "vol_20", "vol_ratio_20"])
        print(df_fac[["close", "ma20", "ma60", "mom_10", "vol_20", "vol_ratio_20"]].tail(3))

        # 4) 可以顺手打印一下板块信息
        if code in sector_map.index:
            row = sector_map.loc[code]
            print(f"  板块: {row.get('sector', '')}  行业: {row.get('industry', '')}")

if __name__ == "__main__":
    main()