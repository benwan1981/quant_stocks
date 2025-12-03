# backtest/plotting.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

try:
    # 可选：用 mplfinance 画蜡烛图
    from mplfinance.original_flavor import candlestick_ohlc
except ImportError:
    candlestick_ohlc = None
    
# backtest/plotting.py 里，原来的 save_backtest_overview_png 整个替换成下面这一段

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """保证 df 以 DatetimeIndex 为索引；如有 'date' 列则自动 set_index。"""
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df

    raise ValueError("DataFrame 既不是 DatetimeIndex，也没有 'date' 列，无法画日期轴")


def save_backtest_overview_png(
    price_df: pd.DataFrame,
    df_sig: pd.DataFrame,
    eq: pd.DataFrame,
    bh: pd.DataFrame,
    out_path: str = "./backtest/plots/overview.png",
    title: str = "回测总览",
):
    """
    画 2 行子图：
    - 上：价格 + 买卖点
    - 下：策略 vs Buy&Hold 归一化资金曲线
    """

    # ===== 1. 统一索引为日期 =====
    price_df = _ensure_datetime_index(price_df)
    df_sig = _ensure_datetime_index(df_sig)
    eq = _ensure_datetime_index(eq)
    bh = _ensure_datetime_index(bh)

    # 只要日 K 的基础字段
    price_cols = [c for c in ["open", "high", "low", "close"] if c in price_df.columns]
    price = price_df[price_cols].copy()

    # 把 position 拼到价格上（按日期对齐）
    merged = price.join(df_sig[["position"]], how="left")
    merged["position"] = merged["position"].ffill().fillna(0).astype(int)

    # 计算买卖点：position 从 0→1 为买入，1→0 为卖出
    pos = merged["position"]
    pos_shift = pos.shift(1).fillna(0)
    buy_mask = (pos == 1) & (pos_shift == 0)
    sell_mask = (pos == 0) & (pos_shift == 1)

    buy_dates = merged.index[buy_mask]
    sell_dates = merged.index[sell_mask]

    # ===== 2. 准备资金曲线（归一化） =====
    eq_norm = eq["equity"] / eq["equity"].iloc[0]
    bh_norm = bh["equity"] / bh["equity"].iloc[0]

    # 只保留两条曲线时间交集，避免一条太长一条太短
    common_index = eq_norm.index.union(bh_norm.index)
    eq_norm = eq_norm.reindex(common_index).ffill()
    bh_norm = bh_norm.reindex(common_index).ffill()

    # ===== 3. 作图 =====
    fig, (ax_price, ax_eq) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    fig.suptitle(title, fontsize=14)

    # --- 上图：价格 + 买卖点 ---
    ax_price.plot(merged.index, merged["close"], label="Close", linewidth=1)

    if len(buy_dates) > 0:
        ax_price.scatter(
            buy_dates,
            merged.loc[buy_dates, "close"],
            marker="^",
            color="g",
            s=40,
            label="Buy",
        )
    if len(sell_dates) > 0:
        ax_price.scatter(
            sell_dates,
            merged.loc[sell_dates, "close"],
            marker="v",
            color="r",
            s=40,
            label="Sell",
        )

    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left")

    # --- 下图：资金曲线 ---
    ax_eq.plot(eq_norm.index, eq_norm, label="Strategy", linewidth=1)
    ax_eq.plot(bh_norm.index, bh_norm, label="Buy & Hold", linewidth=1)

    ax_eq.set_ylabel("Equity (normalized)")
    ax_eq.legend(loc="upper left")

    # ===== 4. x 轴用年份格式化 =====
    ax_eq.xaxis.set_major_locator(mdates.YearLocator())
    ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"🖼 总览图已保存: {out_path}")


def _ensure_chinese_fonts() -> None:
    """
    尝试注册常见的中文字体（Windows/macOS），避免中文标题/注释触发 Missing Glyph warning。
    """
    candidate_fonts = [
        "C:/Windows/Fonts/msyh.ttc",   # Microsoft YaHei
        "C:/Windows/Fonts/simhei.ttf", # SimHei
        "C:/Windows/Fonts/simsun.ttc", # SimSun
        "/System/Library/Fonts/PingFang.ttc",  # macOS PingFang
    ]
    for font_path in candidate_fonts:
        p = Path(font_path)
        if p.exists():
            try:
                font_manager.fontManager.addfont(str(p))
            except Exception:
                pass


_ensure_chinese_fonts()
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "PingFang SC",
    "Heiti SC",
    "STHeiti",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def save_equity_curve_png(
    eq: pd.DataFrame,
    out_path: str,
    title: str = "Equity Curve",
    label: str = "strategy",
):
    """
    生成一张单策略的资金曲线 PNG 图
    """
    df = eq.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["equity"], label=label)
    ax.set_title(title)
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    ax.legend()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"🖼 资金曲线已保存为 PNG: {out_path}")


def save_multi_equity_curve_png(
    curves: Dict[str, pd.DataFrame],
    out_path: str,
    title: str = "Equity Comparison",
):
    """
    多条资金曲线对比，比如：策略 vs Buy&Hold
    curves: {name: eq_df}
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    for name, df in curves.items():
        _df = df.copy()
        if not isinstance(_df.index, pd.DatetimeIndex):
            _df.index = pd.to_datetime(_df.index)
        ax.plot(_df.index, _df["equity"], label=name)

    ax.set_title(title)
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    ax.legend()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"🖼 多曲线对比图已保存: {out_path}")

try:
    import plotly.graph_objs as go
except ImportError:
    go = None


def save_equity_curve_html(
    eq: pd.DataFrame,
    out_path: str,
    title: str = "Equity Curve",
    series_name: str = "strategy",
):
    """
    用 plotly 输出一个可交互的 HTML 资金曲线（悬浮显示、缩放等）
    """
    if go is None:
        raise RuntimeError("plotly 未安装，先 pip install plotly 再用这个函数。")

    df = eq.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["equity"],
        mode="lines",
        name=series_name,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Equity",
        template="plotly_white",
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"📊 交互式HTML资金曲线已保存: {out_path}")

def save_param_summary_scatter(
    df_summary: pd.DataFrame,
    out_path: str,
    title: str = "参数组合表现"
):
    """
    对 run_param_table 的汇总结果画一张散点图：
    - x: buy_score_thresh
    - y: sell_score_thresh
    - 点大小: min_hold_days
    - 颜色: strategy_total_return

    df_summary 约定包含列：
        symbol, buy_score_thresh, sell_score_thresh,
        min_hold_days, strategy_total_return
    """
    if df_summary is None or df_summary.empty:
        print("⚠️ df_summary 为空，跳过参数散点图绘制")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    x = df_summary["buy_score_thresh"]
    y = df_summary["sell_score_thresh"]
    sizes = 30 + df_summary["min_hold_days"] * 4
    colors = df_summary["strategy_total_return"]  # 已经是收益率（小数）

    sc = ax.scatter(
        x, y,
        s=sizes,
        c=colors,
        cmap="RdYlGn",     # 亏损偏红，盈利偏绿
        alpha=0.8,
        edgecolors="k",
        linewidths=0.5,
    )

    # 在点旁边标上 symbol，方便看
    for _, row in df_summary.iterrows():
        sym = str(row.get("symbol", ""))
        ax.text(
            row["buy_score_thresh"] + 0.02,
            row["sell_score_thresh"] + 0.02,
            sym,
            fontsize=8,
            alpha=0.7,
        )

    ax.set_xlabel("buy_score_thresh（买入阈值）")
    ax.set_ylabel("sell_score_thresh（卖出阈值）")
    ax.set_title(title)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("strategy_total_return（策略总收益，小数）")

    ax.grid(True, linestyle="--", alpha=0.3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"🖼 参数组合散点图已保存: {out_path}")


def save_param_summary_by_symbol(
    df_summary: pd.DataFrame,
    out_dir: str = "./backtest/plots/param_by_symbol",
):
    """
    按 symbol 拆分，每个标的一张参数散点图。
    """
    if df_summary is None or df_summary.empty:
        print("⚠️ df_summary 为空，跳过按 symbol 绘图")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sym, g in df_summary.groupby("symbol"):
        file_name = f"{str(sym).replace('.', '')}_param_summary.png"
        out_path = out_dir / file_name
        save_param_summary_scatter(
            g,
            out_path=str(out_path),
            title=f"{sym} 参数组合表现",
        )
