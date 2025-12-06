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
import plotly.graph_objects as go

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import streamlit as st
import traceback
import json

from backtest.engine_v2 import BacktestEngineV2, StrategyConfigV2
from backtest.execution_v2 import ExecutionConfig
from backtest.utils_universe_v2 import load_stock_universe_from_dir, build_index_from_universe

from common.gm_loader import load_gm_ohlcv
from factors.stock_factors import compute_stock_factors, attach_scores
from factors.policy_factor import attach_policy_factor
from fees.fee_engine import FeeConfig, FeeEngine


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
        "/System/Library/Fonts/PingFang.ttc",  # macOS PingFang
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB W3.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux 文泉驿
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansSC-Regular.otf",
    ]
    for path in candidates:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            font_prop = font_manager.FontProperties(fname=path)
            rcParams["font.family"] = font_prop.get_name()
            rcParams["axes.unicode_minus"] = False
            return
    # fallback: 从已安装字体中按名称模糊匹配
    for f in font_manager.fontManager.ttflist:
        name = f.name.lower()
        if any(k in name for k in ["pingfang", "heiti", "simhei", "noto sans sc", "source han", "sarasa ui sc"]):
            rcParams["font.family"] = f.name
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


RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def save_backtest_result_v2(
    eq: pd.DataFrame,
    trades: pd.DataFrame,
    dbg: pd.DataFrame,
    strat_cfg: StrategyConfigV2,
    exec_cfg: ExecutionConfig,
    window_label: str,
    universe: list[str],
) -> Path:
    """
    将本次组合回测结果落盘：
        - equity.csv
        - trades.csv
        - debug.csv
        - summary.json
    返回保存目录路径。
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"combo_v2_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    eq.to_csv(run_dir / "equity.csv", index=True)
    trades.to_csv(run_dir / "trades.csv", index=False)
    dbg.to_csv(run_dir / "debug.csv", index=True)

    # 简单汇总
    total_ret = float(eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1.0)
    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.0 if days > 0 else 0.0
    cagr = (eq["equity"].iloc[-1] / eq["equity"].iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
    rolling_max = eq["equity"].cummax()
    dd = eq["equity"] / rolling_max - 1.0
    mdd = float(dd.min())

    # 粗略胜率（以 equity 日收益 >0 计）
    ret_daily = eq["equity"].pct_change().dropna()
    win_rate = float((ret_daily > 0).mean())

    meta = {
        "window_label": window_label,
        "universe": universe,
        "strategy": {
            "factor_weights": strat_cfg.factor_weights,
            "buy_base": strat_cfg.buy_base,
            "sell_base": strat_cfg.sell_base,
            "target_exp_base": strat_cfg.target_exp_base,
        },
        "execution": {
            "initial_cash": exec_cfg.initial_cash,
            "fee_rate": exec_cfg.fee_rate,
            "stamp_duty": exec_cfg.stamp_duty,
            "lot_size": exec_cfg.lot_size,
        },
        "metrics": {
            "total_return": total_ret,
            "cagr": cagr,
            "max_drawdown": mdd,
            "win_rate": win_rate,
            "days": days,
        },
        "saved_at": ts,
    }

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return run_dir

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

        # 记住用户选过的日期；切换 CSV 也保持用户输入，只做边界裁剪
        sd = st.session_state.get("start_date_input", min_date)
        ed = st.session_state.get("end_date_input", max_date)
        sd = max(min_date, min(sd, max_date))
        ed = max(min_date, min(ed, max_date))
        if ed < sd:
            ed = sd
        st.session_state["start_date_input"] = sd
        st.session_state["end_date_input"] = ed

        start_date = st.sidebar.date_input(
            "开始日期",
            key="start_date_input",
            min_value=min_date,
            max_value=max_date,
        )
        end_date = st.sidebar.date_input(
            "结束日期",
            key="end_date_input",
            min_value=min_date,
            max_value=max_date,
        )

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
        else:
            with st.spinner("正在计算因子并回测，请稍候..."):
                df = df_all[
                    (df_all.index >= pd.to_datetime(start_date))
                    & (df_all.index <= pd.to_datetime(end_date))
                ]
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


            eq_norm = eq["equity"] / eq["equity"].iloc[0]
            bh_norm = bh["equity"] / bh["equity"].iloc[0]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=eq.index,
                y=eq_norm,
                mode="lines",
                name="策略"
            ))
            fig.add_trace(go.Scatter(
                x=bh.index,
                y=bh_norm,
                mode="lines",
                name="Buy&Hold",
                line=dict(dash="dash")
            ))
            fig.update_layout(
                height=400,
                margin=dict(l=40, r=20, t=40, b=40),
                yaxis_title="归一化权益",
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)


            st.markdown("### 最近 20 天信号 & 持仓")
            st.dataframe(
                df_sig[["close", "total_score", "raw_position", "position", "hold_days_intent"]].tail(20)
            )

    # ---------- Tab2：组合动态策略 V2 ----------
    with tab2:
        st.markdown("### 组合动态因子策略（V2）")


        data_dir_v2 = ROOT_DIR / "data" / "gm_HS300_equity"
        st.write(f"数据目录：`{data_dir_v2}`")

        # 回测区间选择
        window_label = st.selectbox(
            "回测区间",
            options=["最近1年", "最近3年", "最近5年", "全部"],
            index=1,
            help="用于组合策略 V2 的回测时间窗口",
        )

        # 股票数量上限（防止一次性全 A 股太慢）
        max_stocks = st.number_input(
            "最多使用的股票数量（按代码排序取前 N 个）",
            min_value=10,
            max_value=5000,
            value=300,
            step=10,
        )
        # 策略文件目录：自动枚举 *.yaml
        strategy_dir = ROOT_DIR / "config" / "strategies_v2"
        strategy_dir.mkdir(parents=True, exist_ok=True)

        strategy_files = sorted(strategy_dir.glob("*.yaml"))
        if not strategy_files:
            st.info(f"策略目录 {strategy_dir} 下暂时没有 .yaml 文件，将使用默认 config/strategy_v2.yaml")
            strat_path = ROOT_DIR / "config" / "strategy_v2.yaml"
        else:
            strat_name_options = [p.name for p in strategy_files]
            selected_strat_name = st.selectbox(
                "选择策略参数文件（YAML）",
                options=strat_name_options,
                index=0,
                help=f"来自目录：{strategy_dir}",
            )
            strat_path = strategy_dir / selected_strat_name

        st.write(f"当前使用的策略文件：`{strat_path}`")

        run_combo = st.button("🚀 运行组合策略 V2")


        if run_combo:
            with st.spinner("正在加载数据并运行组合回测（V2）..."):
                try:
                    # 1) 加载 universe：直接从 prices.parquet 拿列名
                    prices_all = pd.read_parquet(ROOT_DIR / "precomputed" / "prices.parquet")
                    all_codes = sorted(prices_all.columns.tolist())
                    universe_codes = all_codes[: int(max_stocks)]
                    st.write(f"本次使用股票数量：{len(universe_codes)}")

                    # 2) 计算时间窗口
                    start_ts = None
                    end_ts = prices_all.index.max()
                    if window_label != "全部":
                        years = 1 if "1年" in window_label else 3 if "3年" in window_label else 5
                        start_ts = end_ts - pd.DateOffset(years=years)

                    # === 这里修改：构造股票池时强制包含 000300 ===
                    max_n = int(max_stocks)
                    all_codes = sorted(prices_all.columns.tolist())

                    universe_codes = all_codes[:max_n]

                    # 如果 000300 在数据里，但不在当前截取的股票池中，就把它硬塞进去
                    if "000300" in all_codes and "000300" not in universe_codes:
                        if len(universe_codes) < max_n:
                            universe_codes.append("000300")
                        else:
                            # 替换最后一个代码，保证总数量不变
                            universe_codes[-1] = "000300"

                    # 去重一下，防止上面 append/替换造成重复
                    universe_codes = list(dict.fromkeys(universe_codes))

                    st.write(
                        f"本次使用股票数量：{len(universe_codes)}，"
                        f"{'已包含 000300 作为指数/状态基准' if '000300' in universe_codes else '注意：未找到 000300'}"
                    )

                    # 3) 加载策略配置 & 执行配置
                    strat_path = ROOT_DIR / "config" / "strategy_v2.yaml"
                    try:
                        strat_cfg = StrategyConfigV2.from_yaml(strat_path)
                    except Exception as e:
                        st.error(f"加载策略配置 {strat_path} 失败：{e}")
                        st.stop()

                    exec_cfg_path = ROOT_DIR / "config" / "execution_v2.yaml"
                    try:
                        exec_cfg = ExecutionConfig.from_yaml(exec_cfg_path)
                    except Exception as e:
                        st.warning(f"加载执行配置 {exec_cfg_path} 失败，使用代码内默认参数。错误: {e}")
                        exec_cfg = ExecutionConfig()

                    # 4) 运行引擎
                    engine_v2 = BacktestEngineV2(
                        strat_cfg=strat_cfg,
                        universe_codes=universe_codes,
                    )
                    eq = engine_v2.run_backtest(
                        start=None if start_ts is None else start_ts.strftime("%Y-%m-%d"),
                        end=end_ts.strftime("%Y-%m-%d"),
                        exec_cfg=exec_cfg,
                    )
                    #-----------------打印过程-----------------------#
                    strat_cfg = StrategyConfigV2.from_yaml(strat_path)

                    st.markdown("#### 当前生效的策略参数快照")
                    st.json(
                        {
                            "策略文件": str(strat_path),
                            "factor_weights": strat_cfg.factor_weights,
                            "buy_base": strat_cfg.buy_base,
                            "sell_base": strat_cfg.sell_base,
                            "target_exp_base": strat_cfg.target_exp_base,
                        }
                    )

                    exec_cfg = load_execution_config(...)

                    #-----------------打印过程-----------------------#

                    
                    trades_df = engine_v2.trades_df if engine_v2.trades_df is not None else pd.DataFrame()
                    dbg_df = engine_v2.debug_df if engine_v2.debug_df is not None else pd.DataFrame()

                except Exception as e:
                    st.error(f"组合回测运行失败：{e}")
                else:
                    # ===== 指标与曲线 =====
                    strat_ret = float(eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1)
                    days = (eq.index[-1] - eq.index[0]).days
                    years = days / 365.0 if days > 0 else 0.0
                    strat_cagr = (eq["equity"].iloc[-1] / eq["equity"].iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
                    rolling_max = eq["equity"].cummax()
                    dd = eq["equity"] / rolling_max - 1.0
                    strat_mdd = float(dd.min())
                    ret_daily = eq["equity"].pct_change().dropna()
                    win_rate = float((ret_daily > 0).mean())

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("策略总收益", f"{strat_ret:,.2%}")
                    col2.metric("年化收益(CAGR)", f"{strat_cagr:,.2%}")
                    col3.metric("最大回撤", f"{strat_mdd:,.2%}")
                    col4.metric("日胜率", f"{win_rate:,.2%}")

                    st.markdown("#### 组合资金曲线")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    eq_norm = eq["equity"] / eq["equity"].iloc[0]
                    ax.plot(eq.index, eq_norm, label="策略组合")
                    ax.set_ylabel("归一化权益")
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    st.pyplot(fig)

                    st.markdown("#### 最近 10 日组合权益")
                    st.dataframe(eq[["equity", "cash", "portfolio_value"]].tail(10))

                    # ===== 计算过程显示 =====
                    with st.expander("🧮 计算过程（模式 / 仓位 / 阈值 / 波动）", expanded=False):
                        if dbg_df is None or dbg_df.empty:
                            st.info("当前引擎未提供 debug_df。")
                        else:
                            st.dataframe(dbg_df.tail(50))
                            if {"target_exp", "actual_exp"}.issubset(dbg_df.columns):
                                fig2, ax2 = plt.subplots(figsize=(8, 3))
                                ax2.plot(dbg_df.index, dbg_df["target_exp"], label="目标仓位")
                                ax2.plot(dbg_df.index, dbg_df["actual_exp"], label="实际仓位", linestyle="--")
                                ax2.set_ylabel("仓位")
                                ax2.grid(True, alpha=0.3)
                                ax2.legend()
                                st.pyplot(fig2)

                    # ===== 交易明细与个股买卖点 =====
                    with st.expander("📊 交易明细与个股买卖点", expanded=False):
                        if trades_df is None or trades_df.empty:
                            st.info("本次回测没有交易。")
                        else:
                            trades_df = trades_df.copy()
                            trades_df["date"] = pd.to_datetime(trades_df["date"])
                            trades_df = trades_df.sort_values("date")

                            st.markdown(
                                f"共 **{len(trades_df)}** 笔交易，涉及 **{trades_df['code'].nunique()}** 只股票。"
                            )
                            st.dataframe(trades_df.tail(200))

                            codes = sorted(trades_df["code"].unique().tolist())
                            selected_codes = st.multiselect(
                                "选择个股查看买卖点（最多展示前 3 只）",
                                options=codes,
                                default=codes[:3],
                            )

                            prices_sel = pd.read_parquet(ROOT_DIR / "precomputed" / "prices.parquet").reindex(eq.index)

                            for c in selected_codes[:3]:
                                if c not in prices_sel.columns:
                                    continue
                                st.markdown(f"##### {c} 买卖点示意")
                                s = prices_sel[c].reindex(eq.index).ffill()

                                td_c = trades_df[trades_df["code"] == c]
                                buys = td_c[td_c["action"] == "BUY"]
                                sells = td_c[td_c["action"] == "SELL"]

                                fig3, ax3 = plt.subplots(figsize=(8, 3))
                                ax3.plot(s.index, s.values, label="收盘价")
                                if not buys.empty:
                                    ax3.scatter(
                                        buys["date"],
                                        s.reindex(buys["date"]),
                                        marker="^",
                                        label="买入",
                                    )
                                if not sells.empty:
                                    ax3.scatter(
                                        sells["date"],
                                        s.reindex(sells["date"]),
                                        marker="v",
                                        label="卖出",
                                    )
                                ax3.grid(True, alpha=0.3)
                                ax3.legend()
                                st.pyplot(fig3)

                    # ===== 保存结果 =====
                    run_dir = save_backtest_result_v2(
                        eq=eq,
                        trades=trades_df,
                        dbg=dbg_df,
                        strat_cfg=strat_cfg,
                        exec_cfg=exec_cfg,
                        window_label=window_label,
                        universe=universe_codes,
                    )
                    st.success(f"本次回测结果已保存到：{run_dir}")

if __name__ == "__main__":
    main()

'''这是个备注，不用管：终端执行：
streamlit run backtest/streamlit_app.py --server.port 8501
需要参数再追加；要换端口就改 --server.port。1205'''
