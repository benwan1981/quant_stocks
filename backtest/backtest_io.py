# backtest/backtest_io.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional

import json
import pandas as pd


@dataclass
class StrategyConfig:
    """
    策略配置：名字、版本、文字说明、参数（dict）
    """
    name: str
    version: str = "1.0"
    description: str = ""
    params: Dict[str, Any] | None = None


@dataclass
class BacktestMeta:
    """
    回测元信息：标的、数据来源、时间区间等
    """
    symbol: str                 # 如 "SHSE.600383"
    symbol_name: str = ""       # 如 "金地集团"
    data_source: str = ""       # 如 "gm", "eastmoney"
    start_date: str = ""        # "YYYY-MM-DD"
    end_date: str = ""          # "YYYY-MM-DD"
    initial_cash: float = 100000.0
    benchmark: str = ""         # 如 "沪深300", "Buy&Hold(本标的)"


def calc_basic_stats(eq: pd.DataFrame) -> Dict[str, Any]:
    """
    从 equity 曲线里算一些基本统计指标。
    eq: index 为日期，包含 'equity' 列。
    """
    if not isinstance(eq.index, pd.DatetimeIndex):
        eq = eq.copy()
        eq.index = pd.to_datetime(eq.index)

    start_date = eq.index[0].strftime("%Y-%m-%d")
    end_date = eq.index[-1].strftime("%Y-%m-%d")

    start_eq = float(eq["equity"].iloc[0])
    end_eq = float(eq["equity"].iloc[-1])
    total_ret = end_eq / start_eq - 1.0

    cummax = eq["equity"].cummax()
    drawdown = eq["equity"] / cummax - 1.0
    max_dd = float(drawdown.min())

    # 简单年化：用交易日个数 / 252
    n_days = len(eq)
    ann_ret = (1 + total_ret) ** (252.0 / n_days) - 1.0 if n_days > 0 else None

    return {
        "start_date": start_date,
        "end_date": end_date,
        "total_return": total_ret,
        "max_drawdown": max_dd,
        "annual_return": ann_ret,
        "n_days": n_days,
    }


def save_backtest_to_json(
    eq: pd.DataFrame,
    strategy: StrategyConfig,
    meta: BacktestMeta,
    out_path: str,
    extra_stats: Optional[Dict[str, Any]] = None,
) -> str:
    """
    把单标的回测结果保存为 JSON 文件，结构大致为：

    {
      "meta": {...},
      "strategy": {...},
      "stats": {...},
      "equity_curve": [
        {"date": "2024-01-02", "equity": 101234.5, "cash": ..., ...},
        ...
      ]
    }
    """
    df = eq.copy().reset_index()
    # 统一日期字段为字符串
    if "date" not in df.columns:
        df = df.rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    equity_curve = df.to_dict(orient="records")

    stats = calc_basic_stats(eq)
    if extra_stats:
        stats.update(extra_stats)

    payload = {
        "meta": asdict(meta),
        "strategy": asdict(strategy),
        "stats": stats,
        "equity_curve": equity_curve,
    }

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"💾 回测JSON已保存: {p} (共 {len(equity_curve)} 条记录)")
    return str(p)


def save_portfolio_backtests(
    results: Dict[str, pd.DataFrame],
    strategies: Dict[str, StrategyConfig],
    metas: Dict[str, BacktestMeta],
    out_path: str,
) -> str:
    """
    多标的回测结果打成一个 JSON，方便以后做“组合策略”。
    results: {symbol: eq_df}
    strategies: {symbol: StrategyConfig}
    metas: {symbol: BacktestMeta}
    JSON 结构概念上是：

    {
      "portfolio_name": "...",
      "items": [
        { "symbol": "SHSE.600383", "meta": {...}, "strategy": {...}, "stats": {...} },
        ...
      ]
    }
    """
    portfolio_items = []

    for symbol, eq in results.items():
        strat = strategies.get(symbol)
        meta = metas.get(symbol)
        if strat is None or meta is None:
            continue

        stats = calc_basic_stats(eq)
        portfolio_items.append({
            "symbol": symbol,
            "meta": asdict(meta),
            "strategy": asdict(strat),
            "stats": stats,
        })

    payload = {
        "portfolio_name": "custom_portfolio",
        "items": portfolio_items,
    }

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"💾 组合回测JSON已保存: {p} (标的数: {len(portfolio_items)})")
    return str(p)