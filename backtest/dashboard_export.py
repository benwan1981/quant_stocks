# backtest/dashboard_export.py
# -*- coding: utf-8 -*-
"""
把单次回测结果导出成给网页 dashboard 用的一张 CSV：

列包含：
- date          交易日期
- open, high, low, close, volume  日线数据
- position      策略实际持仓（0/1）
- raw_position  策略意图持仓（如果有的话）
- equity        策略资金曲线
- bh_equity     Buy & Hold 资金曲线
"""

from pathlib import Path
import pandas as pd


def export_dashboard_csv(
    price_df: pd.DataFrame,
    df_sig: pd.DataFrame,
    eq: pd.DataFrame,
    bh: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    """
    参数：
    - price_df: 原始日线数据（index 为 date）
    - df_sig:   含 position/raw_position 的信号表（index 为 date）
    - eq:       策略资金曲线（simple_backtest 返回，index 为 date）
    - bh:       Buy & Hold 资金曲线（backtest_buy_and_hold 返回，index 为 date）
    - out_path: 输出 CSV 路径

    返回：
    - 实际保存的 Path
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 以策略资金曲线的日期范围为准，避免 warm-up 期的 NaN
    idx = eq.index

    # 对齐各个数据源
    price = price_df.reindex(idx)
    sig   = df_sig.reindex(idx)
    bh_eq = bh.reindex(idx)

    merged = pd.DataFrame(index=idx)

    for col in ["open", "high", "low", "close", "volume"]:
        if col in price.columns:
            merged[col] = price[col]

    if "position" in sig.columns:
        merged["position"] = sig["position"].fillna(0).astype(int)
    if "raw_position" in sig.columns:
        merged["raw_position"] = sig["raw_position"].fillna(0).astype(int)

    merged["equity"] = eq["equity"]
    if "equity" in bh_eq.columns:
        merged["bh_equity"] = bh_eq["equity"]

    merged = merged.reset_index().rename(columns={"index": "date"})
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")

    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"📄 dashboard CSV 已保存: {out_path}")
    return out_path