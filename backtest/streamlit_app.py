# backtest/streamlit_app.py
# -*- coding: utf-8 -*-
"""
简易单票回测 Dashboard（Streamlit 版）

功能：
- 从指定数据目录中选择一只股票（CSV）
- 配置回测区间和因子权重 / 打分阈值
- 计算因子 + 打分 + 生成信号 + 简单回测
- 展示：
    - 策略 vs Buy&Hold 收益、最大回撤
    - 资金曲线图
    - 最近若干天的信号/持仓表

后续可以逐步升级：
- 接入 param_table
- 接入多标的选股 / 排名
- 接入策略版本选择
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import streamlit as st

# === 把项目根目录加入 sys.path，方便 import 你现有模块 ===
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.gm_loader import load_gm_ohlcv
from factors.stock_factors import compute_stock_factors, attach_scores
from factors.policy_factor import attach_policy_factor
from fees.fee_engine import FeeConfig, FeeEngine
from backtest.engine_v2 import BacktestEngineV2
from backtest.execution_v2 import ExecutionConfig
from backtest.strategy_v2_loader import load_strategy_config_v2
from backtest.utils_universe_v2 import load_stock_universe_from_dir, build_index_from_universe

FACTOR_HELP = {
    "w_trend": "trend_score（趋势因子）：基于价格与均线等趋势指标，反映当前是否处于上升/下降趋势。一般给正权重。",
    "w_mom": "momentum_score（动量因子）：关注最近一段时间的涨跌幅，反映中短期上涨动能。一般给正权重。",
    "w_vol": "volume_score（量能因子）：关注成交量相对于过去均值的放大情况，放量配合上涨时更倾向加分。",
    "w_risk": "risk_score（风险因子）：对极端波动、快速回撤等做风险惩罚。通常权重较小，甚至可以给负权重压制高风险阶段。",
    "w_tech": "technical_score（技术形态因子）：若干技术形态信号的综合打分，一般给小正权重，可按策略微调。",
    "w_pol": "policy_score（政策因子）：结合政策/主题信息，对相关标的加减分，需要启用政策因子时才有意义。",
}


def _set_chinese_font() -> None:
    """Best-effort set a font that包含中文，避免 Matplotlib 缺字警告。"""
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",   # Microsoft YaHei
        r"C:\Windows\Fonts\simhei.ttf", # SimHei
        r"C:\Windows\Fonts\simsun.ttc", # SimSun
    ]
    for path in candidates:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            font_prop = font_manager.FontProperties(fname=path)
            rcParams["font.family"] = font_prop.get_name()
            rcParams["axes.unicode_minus"] = False
            return
    rcParams["axes.unicode_minus"] = False


_set_chinese_font()


# ========== 一些工具函数 ==========

def list_equity_csv_files(data_dir: Path) -> list[Path]:
    """
    列出目录下所有“日线股票 CSV”，约定命名形如：
        600941_中国移动_D_qfq_gm.csv
    这里只简单用：包含 "_D_" 且后缀为 .csv
    """
    if not data_dir.exists():
        return []
    files = [p for p in data_dir.glob("*.csv") if "_D_" in p.name]
    return sorted(files)


def calc_cagr(eq: pd.DataFrame) -> float:
    if len(eq) < 2:
        return 0.0
    start = float(eq["equity"].iloc[0])
    end = float(eq["equity"].iloc[-1])
    if start <= 0 or end <= 0:
        return 0.0
    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.0
    if years <= 0:
        return 0.0
    return (end / start) ** (1 / years) - 1


def calc_max_drawdown(eq: pd.DataFrame) -> float:
    equity = eq["equity"].values
    cummax = np.maximum.accumulate(equity)
    dd = equity / cummax - 1.0
    return float(dd.min())


# ========== 信号 & 回测（从 practice_single_stock 里简化过来的版本） ==========

def generate_signals_v2(
    df_scored: pd.DataFrame,
    buy_score_thresh: float = 4.5,
    sell_score_thresh: float = 3.0,
    min_hold_days: int = 10,
    min_trend_for_buy: float = 1.5,
    min_risk_for_buy: float = 1.5,
    max_trend_for_sell: float = 0.5,
) -> pd.DataFrame:
    df = df_scored.copy()
    raw_pos: list[int] = []
    hold_days_intent: list[int] = []

    pos = 0
    hold_days = 0

    for _, row in df.iterrows():
        score = row.get("total_score", np.nan)
        trend = row.get("trend_score", np.nan)
        risk = row.get("risk_score", np.nan)

        if np.isnan(score) or np.isnan(trend) or np.isnan(risk):
            pos = 0
            hold_days = 0
            raw_pos.append(pos)
            hold_days_intent.append(hold_days)
            continue

        if pos == 0:
            strong_buy = (
                (score >= buy_score_thresh)
                and (trend >= min_trend_for_buy)
                and (risk >= min_risk_for_buy)
            )
            if strong_buy:
                pos = 1
                hold_days = 0
        else:
            hold_days += 1
            weak_or_risk_off = (
                (score <= sell_score_thresh) or (trend <= max_trend_for_sell)
            )
            if weak_or_risk_off and hold_days >= min_hold_days:
                pos = 0
                hold_days = 0

        raw_pos.append(pos)
        hold_days_intent.append(hold_days if pos == 1 else 0)

    df["raw_position"] = raw_pos
    df["hold_days_intent"] = hold_days_intent
    df["position"] = df["raw_position"].shift(1).fillna(0).astype(int)
    return df


def simple_backtest(
    df_sig: pd.DataFrame,
    initial_cash: float = 100000,
    fee_rate: float = 0.0005,
    slippage: float = 0.0005,
    fee_engine: Optional[FeeEngine] = None,
) -> pd.DataFrame:
    df = df_sig.copy().reset_index()
    cash = initial_cash
    shares = 0.0

    if fee_engine is None:
        cfg = FeeConfig(
            trade_fee_rate=fee_rate,
            stamp_duty_rate=0.001,
            financing_rate_year=0.06,
        )
        fee_engine = FeeEngine(cfg)

    records = []
    prev_pos = 0

    for _, row in df.iterrows():
        date = row["date"]
        price_open = row["open"]
        price_close = row["close"]
        target_pos = int(row["position"])

        day_buy_amount = 0.0
        day_sell_amount = 0.0

        if prev_pos == 0 and target_pos == 1 and shares == 0:
            buy_amount = cash
            exec_price = price_open * (1 + slippage)
            shares = buy_amount / exec_price
            day_buy_amount = buy_amount
            cash -= buy_amount

        elif prev_pos == 1 and target_pos == 0 and shares > 0:
            exec_price = price_open * (1 - slippage)
            sell_amount = shares * exec_price
            day_sell_amount = sell_amount
            cash += sell_amount
            shares = 0.0

        day_fee = fee_engine.on_day(
            date=date,
            buy_amount=day_buy_amount,
            sell_amount=day_sell_amount,
            margin_balance=0.0,
            days=1,
        )
        cash -= day_fee

        market_value = shares * price_close
        equity = cash + market_value

        records.append(
            {
                "date": date,
                "cash": cash,
                "shares": shares,
                "market_value": market_value,
                "equity": equity,
                "position": target_pos,
                "day_fee": day_fee,
            }
        )

        prev_pos = target_pos

    eq = pd.DataFrame(records).set_index("date")
    eq["ret"] = eq["equity"].pct_change().fillna(0)
    eq._fee_engine = fee_engine
    return eq


def backtest_buy_and_hold(
    df: pd.DataFrame,
    initial_cash: float = 100000,
    fee_rate: float = 0.0005,
    slippage: float = 0.0005,
) -> pd.DataFrame:
    df = df.copy().reset_index()
    cash = initial_cash
    shares = 0.0
    records = []

    for i, row in df.iterrows():
        date = row["date"]
        po = row["open"]
        pc = row["close"]

        if i == 0:
            buy_amount = cash
            exec_price = po * (1 + slippage)
            shares = buy_amount / exec_price
            fee = buy_amount * fee_rate
            cash -= buy_amount + fee

        mv = shares * pc
        eqty = cash + mv
        records.append(
            {"date": date, "cash": cash, "shares": shares, "market_value": mv, "equity": eqty}
        )

    eq = pd.DataFrame(records).set_index("date")
    eq["ret"] = eq["equity"].pct_change().fillna(0)
    return eq


# ========== Streamlit 界面 ==========

def main():
    st.set_page_config(
        page_title="回测 Dashboard",
        layout="wide",
    )
    st.title("📈 回测 Dashboard")

    tab1, tab2 = st.tabs(["🧪 单票回测（V1）", "📈 组合策略（V2）"])

    # ---------- Tab1：单票回测（沿用原逻辑） ----------
    with tab1:
        st.sidebar.header("① 数据 & 标的")

        default_dir = "./data/gm_HS300_equity"
        data_dir_str = st.sidebar.text_input("数据目录", value=default_dir)
        data_dir = Path(data_dir_str)

        csv_files = list_equity_csv_files(data_dir)
        if not csv_files:
            st.sidebar.warning("该目录下没有找到 *_D_*.csv 日线文件")
            st.stop()

        file_options = {f.name: f for f in csv_files}
        selected_name = st.sidebar.selectbox(
            "选择标的 CSV 文件", options=list(file_options.keys())
        )
        csv_path = file_options[selected_name]
        code = csv_path.name.split("_", 1)[0]

        st.sidebar.markdown(f"**当前标的代码**: `{code}`")

        st.sidebar.header("② 回测时间")

        df_all = load_gm_ohlcv(csv_path)
        df_all = df_all.set_index("date").sort_index()

        min_date = df_all.index[0].date()
        max_date = df_all.index[-1].date()

        start_date = st.sidebar.date_input("开始日期", value=min_date, min_value=min_date, max_value=max_date)
        end_date = st.sidebar.date_input("结束日期", value=max_date, min_value=min_date, max_value=max_date)

        if start_date > end_date:
            st.sidebar.error("开始日期不能晚于结束日期")
            st.stop()

        st.sidebar.header("③ 策略参数")

        buy_th = st.sidebar.number_input("买入阈值（total_score）", value=4.5, step=0.5)
        sell_th = st.sidebar.number_input("卖出阈值（total_score）", value=3.0, step=0.5)
        min_hold = st.sidebar.number_input("最小持有天数", value=10, min_value=1, step=1)

        st.sidebar.subheader("因子权重（用于 total_score）")
        w_trend = st.sidebar.number_input(
            "w_trend",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.01,
            format="%.2f",
            help=FACTOR_HELP["w_trend"],
        )
        w_mom = st.sidebar.number_input(
            "w_mom",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.01,
            format="%.2f",
            help=FACTOR_HELP["w_mom"],
        )
        w_vol = st.sidebar.number_input(
            "w_vol",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.01,
            format="%.2f",
            help=FACTOR_HELP["w_vol"],
        )
        w_risk = st.sidebar.number_input(
            "w_risk",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.01,
            format="%.2f",
            help=FACTOR_HELP["w_risk"],
        )
        w_tech = st.sidebar.number_input(
            "w_tech",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.01,
            format="%.2f",
            help=FACTOR_HELP["w_tech"],
        )
        w_pol = st.sidebar.number_input(
            "w_policy",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.01,
            format="%.2f",
            help=FACTOR_HELP["w_pol"],
        )

        use_policy = st.sidebar.checkbox("启用政策因子", value=False)

        run_btn = st.sidebar.button("🚀 运行回测")

        if not run_btn:
            st.info("在左侧配置完参数后，点击 **🚀 运行回测**。")
            st.stop()

        with st.spinner("正在计算因子并回测，请稍候..."):
            df = df_all[(df_all.index >= pd.to_datetime(start_date)) & (df_all.index <= pd.to_datetime(end_date))]
            if df.empty:
                st.error("该时间段内没有数据，请调整开始/结束日期")
                st.stop()

            df_fac = compute_stock_factors(df)

            if use_policy:
                try:
                    df_fac = attach_policy_factor(df_fac, code=code, market=None)
                except Exception as e:
                    st.warning(f"政策因子未生效：{e}")

            df_scored = attach_scores(df_fac)

            for col in ["trend_score", "momentum_score", "volume_score", "risk_score", "technical_score"]:
                if col not in df_scored.columns:
                    df_scored[col] = 0.0
            if "policy_score" not in df_scored.columns:
                df_scored["policy_score"] = 0.0

            df_scored["total_score"] = (
                w_trend * df_scored["trend_score"]
                + w_mom * df_scored["momentum_score"]
                + w_vol * df_scored["volume_score"]
                + w_risk * df_scored["risk_score"]
                + w_tech * df_scored["technical_score"]
                + w_pol * df_scored["policy_score"]
            )

            df_sig = generate_signals_v2(
                df_scored,
                buy_score_thresh=buy_th,
                sell_score_thresh=sell_th,
                min_hold_days=min_hold,
            )

            eq = simple_backtest(df_sig)
            bh = backtest_buy_and_hold(df)

            strat_ret = float(eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1)
            strat_cagr = calc_cagr(eq)
            strat_mdd = calc_max_drawdown(eq)

            bh_ret = float(bh["equity"].iloc[-1] / bh["equity"].iloc[0] - 1)
            bh_cagr = calc_cagr(bh)
            bh_mdd = calc_max_drawdown(bh)

        st.subheader(f"标的：{code}  | 回测区间：{start_date} ~ {end_date}")
        col1, col2, col3 = st.columns(3)
        col1.metric("策略总收益", f"{strat_ret:,.2%}")
        col2.metric("策略年化收益(CAGR)", f"{strat_cagr:,.2%}")
        col3.metric("策略最大回撤", f"{strat_mdd:,.2%}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Buy&Hold 总收益", f"{bh_ret:,.2%}")
        col5.metric("Buy&Hold CAGR", f"{bh_cagr:,.2%}")
        col6.metric("Buy&Hold 最大回撤", f"{bh_mdd:,.2%}")

        st.markdown("### 资金曲线（策略 vs Buy&Hold）")
        fig, ax = plt.subplots(figsize=(10, 4))
        eq["equity_norm"] = eq["equity"] / eq["equity"].iloc[0]
        bh["equity_norm"] = bh["equity"] / bh["equity"].iloc[0]
        ax.plot(eq.index, eq["equity_norm"], label="策略")
        ax.plot(bh.index, bh["equity_norm"], label="Buy&Hold", linestyle="--")
        ax.set_ylabel("归一化权益")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

        st.markdown("### 最近 20 天信号 & 持仓")
        st.dataframe(
            df_sig[["close", "total_score", "raw_position", "position", "hold_days_intent"]].tail(20)
        )

    # ---------- Tab2：组合动态策略 V2 ----------
    with tab2:
        st.markdown("### 组合动态因子策略（V2）")

        data_dir_v2 = ROOT_DIR / "data" / "gm_HS300_equity"
        st.write(f"数据目录：`{data_dir_v2}`")

        run_combo = st.button("🚀 运行组合策略 V2")

        if run_combo:
            with st.spinner("正在加载股票池并运行组合回测（V2）..."):
                try:
                    stock_universe = load_stock_universe_from_dir(data_dir_v2)
                    index_df = build_index_from_universe(stock_universe)

                    strat_cfg = load_strategy_config_v2(ROOT_DIR / "config" / "strategy_v2.yaml")
                    exec_cfg = ExecutionConfig(
                        initial_cash=1_000_000,
                        fee_rate=0.0005,
                        slippage=0.0005,
                    )

                    engine = BacktestEngineV2(
                        stock_universe=stock_universe,
                        index_df=index_df,
                        strat_cfg=strat_cfg,
                    )
                    eq = engine.run_backtest(exec_cfg)

                    idx_norm = index_df["close"].reindex(eq.index).ffill()
                    idx_norm = idx_norm / idx_norm.iloc[0]
                except Exception as e:
                    st.error(f"组合回测运行失败：{e}")
                else:
                    strat_ret = float(eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1)
                    strat_cagr = calc_cagr(eq)
                    strat_mdd = calc_max_drawdown(eq)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("策略总收益", f"{strat_ret:,.2%}")
                    col2.metric("策略年化收益(CAGR)", f"{strat_cagr:,.2%}")
                    col3.metric("策略最大回撤", f"{strat_mdd:,.2%}")

                    st.markdown("#### 组合资金曲线 vs 等权指数")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    eq_norm = eq["equity"] / eq["equity"].iloc[0]
                    ax.plot(eq.index, eq_norm, label="策略组合")
                    ax.plot(eq.index, idx_norm, label="等权指数", linestyle="--")
                    ax.set_ylabel("归一化权益")
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    st.pyplot(fig)

                    st.markdown("#### 最近 10 日组合权益")
                    st.dataframe(eq[["equity", "cash", "market_value"]].tail(10))


if __name__ == "__main__":
    main()

'''终端执行：
streamlit run backtest/streamlit_app.py --server.port 8501
需要参数再追加；要换端口就改 --server.port。'''
