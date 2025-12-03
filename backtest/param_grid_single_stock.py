# backtest/param_grid_single_stock.py
# -*- coding: utf-8 -*-
"""
单个标的参数网格回测：
- 从 CSV 加载数据
- 计算因子 + 打分（复用 practice_single_stock / stock_factors）
- 扫一堆 (buy_score, sell_score, min_hold_days) 组合
- 输出一个参数表现表，并保存为 CSV
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# === 保证能 import 到项目内模块 ===
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 复用你已有的函数
from backtest.practice_single_stock import (
    load_data_from_csv,
    generate_signals_v2,
    simple_backtest,
)
from factors.stock_factors import compute_stock_factors, attach_scores
from factors.policy_factor import attach_policy_factor
from common import ensure_utf8_filename

# 和 practice_single_stock 保持一致（也可以改成参数）
START_DATE = "2018-01-01"
END_DATE   = "2025-12-30"


def run_param_grid(
    csv_path: str,
    code: str,
    market: str = "SH",
    buy_list=(4.0, 4.5, 5.0),
    sell_list=(2.5, 3.0, 3.5),
    hold_list=(5, 10, 20),
    out_dir: str = "./backtest/results",
):
    """
    对单个标的做参数网格回测，输出 DataFrame 并保存 CSV。

    参数：
    - csv_path: 数据文件路径，例如 "./data/gm/600383_金地集团_D_gm.csv"
    - code: 证券代码（不带交易所），例如 "600383"
    - market: "SH" / "SZ"
    - buy_list: buy_score_thresh 候选列表
    - sell_list: sell_score_thresh 候选列表
    - hold_list: min_hold_days 候选列表
    """

    # ===== 1. 加载数据 =====
    df = load_data_from_csv(csv_path)

    # 按时间过滤
    if START_DATE:
        df = df[df.index >= pd.to_datetime(START_DATE)]
    if END_DATE:
        df = df[df.index <= pd.to_datetime(END_DATE)]
    if df.empty:
        raise RuntimeError("时间段过滤后没有数据，请检查 START_DATE / END_DATE")

    print(f"数据区间: {df.index[0].date()} ~ {df.index[-1].date()}，共 {len(df)} 个交易日")

    # ===== 2. 计算因子 + 政策因子 + 打分 =====
    df_fac = compute_stock_factors(df)

    try:
        df_fac = attach_policy_factor(df_fac, code=code, market=market)
        print("✅ 已叠加政策因子")
    except Exception as e:
        print(f"⚠️ attach_policy_factor 失败（忽略）：{e}")

    df_scored = attach_scores(df_fac)

    # ===== 3. 扫参数网格 =====
    rows = []

    for buy in buy_list:
        for sell in sell_list:
            for hold in hold_list:

                df_sig = generate_signals_v2(
                    df_scored,
                    buy_score_thresh=buy,
                    sell_score_thresh=sell,
                    min_hold_days=hold,
                )

                eq = simple_backtest(
                    df_sig,
                    initial_cash=100000,
                    fee_rate=0.0005,
                    slippage=0.0005,
                    stop_loss_pct=0.10,
                    trail_stop_pct=0.15,
                    fee_engine=None,
                )

                start_eq = float(eq["equity"].iloc[0])
                end_eq   = float(eq["equity"].iloc[-1])
                total_ret = end_eq / start_eq - 1.0

                n_days = len(eq)
                if n_days > 1:
                    cagr = (end_eq / start_eq) ** (252.0 / n_days) - 1.0
                else:
                    cagr = 0.0

                cummax = eq["equity"].cummax()
                dd = eq["equity"] / cummax - 1.0
                max_dd = float(dd.min())

                rows.append(
                    {
                        "buy_score": buy,
                        "sell_score": sell,
                        "min_hold_days": hold,
                        "total_return": total_ret,
                        "cagr": cagr,
                        "max_drawdown": max_dd,
                    }
                )

    res = pd.DataFrame(rows)
    res = res.sort_values("total_return", ascending=False)

    # 漂亮一点的打印
    print("\n📊 参数组合表现（按总收益排序）：")
    if not res.empty:
        print(
            res.to_string(
                index=False,
                formatters={
                    "total_return": lambda x: f"{x:7.2%}",
                    "cagr":          lambda x: f"{x:7.2%}",
                    "max_drawdown":  lambda x: f"{x:7.2%}",
                },
            )
        )
    else:
        print("（结果为空？检查数据和参数范围）")

    # ===== 4. 保存为 CSV =====
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    base_name = ensure_utf8_filename(Path(csv_path).stem)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = ensure_utf8_filename(f"{base_name}_param_grid_{ts}.csv")
    out_file = Path(out_dir) / file_name
    res.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\n📄 参数表现表已保存: {out_file}")

    return res


if __name__ == "__main__":
    # 这里先默认用 600383 金地，你可以按需改：
    csv_path = "./data/gm/600941_中国移动_D_gm.csv"
    run_param_grid(
        csv_path=csv_path,
        code="600383",
        market="SH",
        buy_list=(4.0, 4.5, 5.0),
        sell_list=(2.5, 3.0, 3.5),
        hold_list=(5, 10, 20),
    )
