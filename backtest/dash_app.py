# backtest/dash_app.py
# -*- coding: utf-8 -*-
"""
单票回测 Dashboard（Dash 版本）

功能：
- 从指定数据目录中扫描日线 CSV（如 600941_中国移动_D_qfq_gm.csv）
- 选择标的 + 回测区间 + 策略参数
- 计算因子 + 打分 + 生成信号 + 简单回测
- 展示：
    - 策略 vs Buy&Hold 收益、CAGR、最大回撤
    - 资金曲线（归一化）
    - 最近 20 天信号表

依赖：
    pip install dash plotly pandas numpy matplotlib  (matplotlib 可选，这里没用到)
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

# === 项目根目录放进 sys.path，方便 import 你现有模块 ===
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.gm_loader import load_gm_ohlcv
from factors.stock_factors import compute_stock_factors, attach_scores
from factors.policy_factor import attach_policy_factor
from fees.fee_engine import FeeConfig, FeeEngine


# ========== 工具函数 ==========

def list_equity_csv_files(data_dir: Path) -> list[Path]:
    """
    列出目录下所有“日线股票 CSV”，约定命名形如：
        600941_中国移动_D_qfq_gm.csv
    规则：包含 '_D_' 且后缀为 .csv
    """
    if not data_dir.exists() or not data_dir.is_dir():
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
    if len(equity) == 0:
        return 0.0
    cummax = np.maximum.accumulate(equity)
    dd = equity / cummax - 1.0
    return float(dd.min())


# ========== 信号 & 回测（从 practice_single_stock 简化） ==========

def generate_signals_v2(
    df_scored: pd.DataFrame,
    buy_score_thresh: float = 4.5,
    sell_score_thresh: float = 3.0,
    min_hold_days: int = 10,
    min_trend_for_buy: float = 1.5,
    min_risk_for_buy: float = 1.5,
    max_trend_for_sell: float = 0.5,
) -> pd.DataFrame:
    """
    基于 total_score + trend_score + risk_score 的简单状态机 + 最小持有天数
    """
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
    """
    简单单标的全仓进出回测（带费用）
    """
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
    """
    Buy & Hold：第一天开盘全仓买入，持有到最后
    """
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
            {
                "date": date,
                "cash": cash,
                "shares": shares,
                "market_value": mv,
                "equity": eqty,
            }
        )

    eq = pd.DataFrame(records).set_index("date")
    eq["ret"] = eq["equity"].pct_change().fillna(0)
    return eq


# ========== Dash 应用 ==========

app = Dash(__name__)
app.title = "单票回测 Dashboard (Dash)"


app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "margin": "10px"},
    children=[
        html.H1("📈 单票回测 Dashboard（Dash）"),
        html.Div(
            style={"display": "flex", "gap": "20px", "alignItems": "flex-start"},
            children=[
                # ===== 左侧控制区 =====
                html.Div(
                    style={
                        "flex": "0 0 320px",
                        "border": "1px solid #ddd",
                        "padding": "10px",
                        "borderRadius": "4px",
                    },
                    children=[
                        html.H3("① 数据 & 标的"),
                        html.Label("数据目录"),
                        dcc.Input(
                            id="data-dir-input",
                            type="text",
                            value="./data/gm_equity",
                            style={"width": "100%", "marginBottom": "8px"},
                        ),
                        html.Button(
                            "扫描目录中的日线 CSV",
                            id="scan-btn",
                            n_clicks=0,
                            style={"width": "100%", "marginBottom": "8px"},
                        ),
                        dcc.Dropdown(
                            id="file-dropdown",
                            placeholder="选择标的 CSV 文件",
                            style={"marginBottom": "12px"},
                        ),
                        html.Div(id="selected-code-text", style={"marginBottom": "12px", "fontSize": "14px"}),

                        html.H3("② 回测时间"),
                        dcc.DatePickerRange(
                            id="date-range",
                            display_format="YYYY-MM-DD",
                            minimum_nights=0,
                            style={"marginBottom": "12px"},
                        ),

                        html.H3("③ 策略参数"),
                        html.Label("买入阈值（total_score）"),
                        dcc.Input(
                            id="buy-thresh",
                            type="number",
                            value=4.5,
                            step=0.5,
                            style={"width": "100%", "marginBottom": "6px"},
                        ),
                        html.Label("卖出阈值（total_score）"),
                        dcc.Input(
                            id="sell-thresh",
                            type="number",
                            value=3.0,
                            step=0.5,
                            style={"width": "100%", "marginBottom": "6px"},
                        ),
                        html.Label("最小持有天数"),
                        dcc.Input(
                            id="min-hold-days",
                            type="number",
                            value=10,
                            min=1,
                            step=1,
                            style={"width": "100%", "marginBottom": "10px"},
                        ),

                        html.Label("因子权重"),
                        html.Div("w_trend"),
                        dcc.Input(
                            id="w-trend",
                            type="number",
                            value=1.0,
                            step=0.1,
                            style={"width": "100%", "marginBottom": "4px"},
                        ),
                        html.Div("w_mom"),
                        dcc.Input(
                            id="w-mom",
                            type="number",
                            value=1.0,
                            step=0.1,
                            style={"width": "100%", "marginBottom": "4px"},
                        ),
                        html.Div("w_vol"),
                        dcc.Input(
                            id="w-vol",
                            type="number",
                            value=1.0,
                            step=0.1,
                            style={"width": "100%", "marginBottom": "4px"},
                        ),
                        html.Div("w_risk"),
                        dcc.Input(
                            id="w-risk",
                            type="number",
                            value=1.0,
                            step=0.1,
                            style={"width": "100%", "marginBottom": "4px"},
                        ),
                        html.Div("w_tech"),
                        dcc.Input(
                            id="w-tech",
                            type="number",
                            value=1.0,
                            step=0.1,
                            style={"width": "100%", "marginBottom": "4px"},
                        ),
                        html.Div("w_policy"),
                        dcc.Input(
                            id="w-pol",
                            type="number",
                            value=1.0,
                            step=0.1,
                            style={"width": "100%", "marginBottom": "10px"},
                        ),

                        dcc.Checklist(
                            id="use-policy",
                            options=[{"label": "启用政策因子", "value": "policy"}],
                            value=[],
                            style={"marginBottom": "10px"},
                        ),

                        html.Button(
                            "🚀 运行回测",
                            id="run-btn",
                            n_clicks=0,
                            style={"width": "100%", "backgroundColor": "#28a745", "color": "white"},
                        ),
                        html.Div(id="log-text", style={"marginTop": "10px", "fontSize": "12px", "color": "#666"}),
                    ],
                ),

                # ===== 右侧展示区 =====
                html.Div(
                    style={
                        "flex": "1 1 0",
                        "minWidth": "0",
                        "border": "1px solid #ddd",
                        "padding": "10px",
                        "borderRadius": "4px",
                        "overflow": "auto",
                    },
                    children=[
                        html.H3("回测结果"),
                        html.Div(
                            id="metrics-div",
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(3, 1fr)",
                                "gap": "8px",
                                "marginBottom": "10px",
                            },
                        ),
                        dcc.Graph(id="equity-graph", style={"height": "420px"}),
                        html.H4("最近 20 天信号"),
                        dash_table.DataTable(
                            id="signals-table",
                            page_size=20,
                            style_table={"overflowX": "auto"},
                            style_cell={"fontSize": 12, "padding": "4px"},
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ========== 回调 ==========

@app.callback(
    Output("file-dropdown", "options"),
    Output("file-dropdown", "value"),
    Input("scan-btn", "n_clicks"),
    State("data-dir-input", "value"),
    prevent_initial_call=False,
)
def scan_directory(n_clicks, data_dir_str):
    """
    扫描数据目录，列出所有 *_D_*.csv，让用户选择。
    首次加载时也会执行一次（n_clicks 可能为 None / 0）
    """
    if not data_dir_str:
        return [], None

    data_dir = Path(data_dir_str)
    files = list_equity_csv_files(data_dir)
    if not files:
        return [], None

    options = [{"label": f.name, "value": f.name} for f in files]
    # 默认选第一个
    return options, files[0].name


@app.callback(
    Output("selected-code-text", "children"),
    Output("date-range", "min_date_allowed"),
    Output("date-range", "max_date_allowed"),
    Output("date-range", "start_date"),
    Output("date-range", "end_date"),
    Input("file-dropdown", "value"),
    State("data-dir-input", "value"),
)
def update_date_range(selected_file, data_dir_str):
    """
    当选择不同 CSV 时，读取一次数据，确定可用日期区间。
    """
    if not selected_file or not data_dir_str:
        return "未选择标的", None, None, None, None

    csv_path = Path(data_dir_str) / selected_file
    if not csv_path.exists():
        return f"文件不存在: {csv_path}", None, None, None, None

    df_all = load_gm_ohlcv(csv_path)
    df_all = df_all.set_index("date").sort_index()

    if df_all.empty:
        return f"{csv_path.name} 中没有数据", None, None, None, None

    min_date = df_all.index[0].date()
    max_date = df_all.index[-1].date()

    code = csv_path.name.split("_", 1)[0]
    text = f"当前标的代码：{code}"

    return (
        text,
        min_date,
        max_date,
        min_date,
        max_date,
    )


@app.callback(
    Output("metrics-div", "children"),
    Output("equity-graph", "figure"),
    Output("signals-table", "data"),
    Output("signals-table", "columns"),
    Output("log-text", "children"),
    Input("run-btn", "n_clicks"),
    State("file-dropdown", "value"),
    State("data-dir-input", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    State("buy-thresh", "value"),
    State("sell-thresh", "value"),
    State("min-hold-days", "value"),
    State("w-trend", "value"),
    State("w-mom", "value"),
    State("w-vol", "value"),
    State("w-risk", "value"),
    State("w-tech", "value"),
    State("w-pol", "value"),
    State("use-policy", "value"),
)
def run_backtest(
    n_clicks,
    selected_file,
    data_dir_str,
    start_date,
    end_date,
    buy_th,
    sell_th,
    min_hold,
    w_trend,
    w_mom,
    w_vol,
    w_risk,
    w_tech,
    w_pol,
    use_policy_list,
):
    """
    点击“运行回测”后执行：
    - 读取数据
    - 计算因子 + 打分 + 信号
    - 回测 + Buy&Hold
    - 输出指标 + 图 + 最近 20 天信号表
    """
    if not n_clicks:
        # 初始不回测
        return [], go.Figure(), [], [], ""

    log_lines = []

    if not selected_file or not data_dir_str:
        return [], go.Figure(), [], [], "⚠️ 请先选择数据目录和标的 CSV 文件"

    csv_path = Path(data_dir_str) / selected_file
    if not csv_path.exists():
        return [], go.Figure(), [], [], f"⚠️ 文件不存在: {csv_path}"

    try:
        # 1) 读数据 + 时间过滤
        df_all = load_gm_ohlcv(csv_path)
        df_all = df_all.set_index("date").sort_index()

        if start_date:
            df_all = df_all[df_all.index >= pd.to_datetime(start_date)]
        if end_date:
            df_all = df_all[df_all.index <= pd.to_datetime(end_date)]

        if df_all.empty:
            return [], go.Figure(), [], [], "⚠️ 时间段内无数据，请调整日期"

        log_lines.append(f"数据区间：{df_all.index[0].date()} ~ {df_all.index[-1].date()}，共 {len(df_all)} 个交易日")

        # 2) 计算因子
        df_fac = compute_stock_factors(df_all)

        # 政策因子
        has_policy = False
        code = csv_path.name.split("_", 1)[0]
        if use_policy_list and "policy" in use_policy_list:
            try:
                df_fac = attach_policy_factor(df_fac, code=code, market=None)
                has_policy = True
                log_lines.append("✅ 已叠加政策因子")
            except Exception as e:
                log_lines.append(f"⚠️ 政策因子未生效：{e}")
        else:
            log_lines.append("ℹ️ 本次未启用政策因子")

        # 3) 打分
        df_scored = attach_scores(df_fac)

        for col in ["trend_score", "momentum_score", "volume_score", "risk_score", "technical_score"]:
            if col not in df_scored.columns:
                df_scored[col] = 0.0
        if "policy_score" not in df_scored.columns:
            df_scored["policy_score"] = 0.0

        df_scored["total_score"] = (
            float(w_trend or 0.0) * df_scored["trend_score"]
            + float(w_mom or 0.0) * df_scored["momentum_score"]
            + float(w_vol or 0.0) * df_scored["volume_score"]
            + float(w_risk or 0.0) * df_scored["risk_score"]
            + float(w_tech or 0.0) * df_scored["technical_score"]
            + float(w_pol or 0.0) * df_scored["policy_score"]
        )

        # 4) 生成信号 & 回测
        df_sig = generate_signals_v2(
            df_scored,
            buy_score_thresh=float(buy_th or 4.5),
            sell_score_thresh=float(sell_th or 3.0),
            min_hold_days=int(min_hold or 10),
        )

        eq = simple_backtest(df_sig)
        bh = backtest_buy_and_hold(df_all)

        # 5) 绩效指标
        strat_ret = float(eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1)
        strat_cagr = calc_cagr(eq)
        strat_mdd = calc_max_drawdown(eq)

        bh_ret = float(bh["equity"].iloc[-1] / bh["equity"].iloc[0] - 1)
        bh_cagr = calc_cagr(bh)
        bh_mdd = calc_max_drawdown(bh)

        # 指标展示
        metrics_children = [
            html.Div(
                style={
                    "border": "1px solid #eee",
                    "padding": "4px 8px",
                    "borderRadius": "4px",
                    "backgroundColor": "#fafafa",
                },
                children=[
                    html.Div("策略总收益"),
                    html.Strong(f"{strat_ret:,.2%}"),
                ],
            ),
            html.Div(
                style={
                    "border": "1px solid #eee",
                    "padding": "4px 8px",
                    "borderRadius": "4px",
                    "backgroundColor": "#fafafa",
                },
                children=[
                    html.Div("策略年化收益 (CAGR)"),
                    html.Strong(f"{strat_cagr:,.2%}"),
                ],
            ),
            html.Div(
                style={
                    "border": "1px solid #eee",
                    "padding": "4px 8px",
                    "borderRadius": "4px",
                    "backgroundColor": "#fafafa",
                },
                children=[
                    html.Div("策略最大回撤"),
                    html.Strong(f"{strat_mdd:,.2%}"),
                ],
            ),
            html.Div(
                style={
                    "border": "1px solid #eee",
                    "padding": "4px 8px",
                    "borderRadius": "4px",
                    "backgroundColor": "#fafafa",
                },
                children=[
                    html.Div("Buy&Hold 总收益"),
                    html.Strong(f"{bh_ret:,.2%}"),
                ],
            ),
            html.Div(
                style={
                    "border": "1px solid #eee",
                    "padding": "4px 8px",
                    "borderRadius": "4px",
                    "backgroundColor": "#fafafa",
                },
                children=[
                    html.Div("Buy&Hold CAGR"),
                    html.Strong(f"{bh_cagr:,.2%}"),
                ],
            ),
            html.Div(
                style={
                    "border": "1px solid #eee",
                    "padding": "4px 8px",
                    "borderRadius": "4px",
                    "backgroundColor": "#fafafa",
                },
                children=[
                    html.Div("Buy&Hold 最大回撤"),
                    html.Strong(f"{bh_mdd:,.2%}"),
                ],
            ),
        ]

        # ========= 资金曲线图 =========
        eq_plot = eq.copy()
        bh_plot = bh.copy()
        # 确保索引为日期类型并排序，避免图形坐标乱序/不显示
        eq_plot.index = pd.to_datetime(eq_plot.index)
        bh_plot.index = pd.to_datetime(bh_plot.index)
        eq_plot = eq_plot.sort_index()
        bh_plot = bh_plot.sort_index()
        eq_plot["equity_norm"] = eq_plot["equity"] / eq_plot["equity"].iloc[0]
        bh_plot["equity_norm"] = bh_plot["equity"] / bh_plot["equity"].iloc[0]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=eq_plot.index,
                y=eq_plot["equity_norm"],
                mode="lines",
                name="策略",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=bh_plot.index,
                y=bh_plot["equity_norm"],
                mode="lines",
                name="Buy&Hold",
                line={"dash": "dash"},
            )
        )
        fig.update_layout(
            margin=dict(l=40, r=20, t=40, b=40),
            yaxis_title="归一化权益",
            xaxis_title="日期",
            hovermode="x unified",
            template="plotly_white",
            height=420,
            legend=dict(orientation="h", y=-0.2),
        )
        fig.update_xaxes(type="date")

        # ========= 最近 20 天信号表 =========
        df_tail = df_sig[["close", "total_score", "raw_position", "position", "hold_days_intent"]].tail(20)
        df_tail = df_tail.reset_index()
        df_tail["date"] = df_tail["date"].dt.strftime("%Y-%m-%d")

        columns = [{"name": c, "id": c} for c in df_tail.columns]
        data = df_tail.to_dict("records")

        log_lines.append("回测完成 ✅")

        return metrics_children, fig, data, columns, html.Pre("\n".join(log_lines))

    except Exception as e:
        return [], go.Figure(), [], [], f"❌ 回测出错: {e}"


if __name__ == "__main__":
    # debug=True 开发阶段方便看报错
    # Dash 3+ 用 app.run（run_server 已弃用）
    # 只监听本机
    app.run(debug=True, host="127.0.0.1", port=8050)
