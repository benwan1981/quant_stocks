# backtest/run_param_table.py
# -*- coding: utf-8 -*-
"""
批量调参入口脚本：

- 从 config/param_table.csv 读取参数
- 每一行 = 一个“标的 + 策略参数 + 因子权重”组合
- 对每一行执行：
    1. 读 data_file 的日线数据
    2. compute_stock_factors + attach_policy_factor (可选)
    3. attach_scores → 按 w_xxx 组合 total_score
    4. generate_signals_v2 生成交易信号
    5. simple_backtest 与 backtest_buy_and_hold
- 输出汇总表 backtest/param_table_summary.csv
"""

import sys
from pathlib import Path
from datetime import datetime
import os

import pandas as pd
import numpy as np

# ==== 确保可以 import 你现有的模块 ====
ROOT_DIR = Path(__file__).resolve().parents[1]   # 项目根目录
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from factors.stock_factors import compute_stock_factors, attach_scores
from factors.policy_factor import attach_policy_factor
from fees.fee_engine import FeeConfig, FeeEngine

'''from backtest.plotting import (
    save_backtest_overview_png,
    save_param_summary_scatter,
    save_param_summary_by_symbol,
)'''
from common import ensure_utf8_filename

# 👇 新增
from backtest.dashboard_export import export_dashboard_csv

# ===== 工具函数 =====

def load_data_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


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

    raw_pos = []
    hold_days_intent = []

    pos = 0
    hold_days = 0

    for _, row in df.iterrows():
        score = row.get("total_score", np.nan)
        trend = row.get("trend_score", np.nan)
        risk  = row.get("risk_score",  np.nan)

        if np.isnan(score) or np.isnan(trend) or np.isnan(risk):
            pos = 0
            hold_days = 0
            raw_pos.append(pos)
            hold_days_intent.append(hold_days)
            continue

        if pos == 0:
            strong_buy = (
                (score >= buy_score_thresh) and
                (trend >= min_trend_for_buy) and
                (risk  >= min_risk_for_buy)
            )
            if strong_buy:
                pos = 1
                hold_days = 0
        else:
            hold_days += 1
            weak_or_risk_off = (
                (score <= sell_score_thresh) or
                (trend <= max_trend_for_sell)
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
    fee_engine: FeeEngine | None = None,
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


# ===== 主流程：从 param_table.csv 读取并逐行回测 =====

def run_from_param_table(
    param_csv: str = "./config/param_table.csv",
    out_summary_csv: str | None = "./backtest/param_table_summary.csv",
    start_date: str | None = None,
    end_date: str | None = None,
):
    param_path = Path(param_csv)
    if not param_path.exists():
        raise FileNotFoundError(f"找不到参数表: {param_path}")

    df_param = pd.read_csv(param_path, comment="#")
    rows_out = []

  # 如果参数表里有 start_date / end_date 列，则支持行级回测时间
    has_start_col = "start_date" in df_param.columns
    has_end_col = "end_date" in df_param.columns

    for idx, row in df_param.iterrows():
        code = str(row["symbol_code"]).strip()
        market = str(row["symbol_market"]).strip().upper()
        symbol = f"{market}.{code}"
        data_file = str(row["data_file"]).strip()
        use_policy = int(row.get("use_policy", 0))

        # === 每一行自己的回测区间 ===
        row_start = str(row.get("backtest_start", "")).strip()
        row_end   = str(row.get("backtest_end", "")).strip()
        # 空字符串 / NaN 就用全局默认
        use_start = row_start or start_date
        use_end   = row_end or end_date

        buy_th = float(row.get("buy_score_thresh", 4.5))
        sell_th = float(row.get("sell_score_thresh", 3.0))
        min_hold = int(row.get("min_hold_days", 10))

        w_trend = float(row.get("w_trend", 1.0))
        w_mom   = float(row.get("w_mom",   1.0))
        w_vol   = float(row.get("w_vol",   1.0))
        w_risk  = float(row.get("w_risk",  1.0))
        w_tech  = float(row.get("w_tech",  1.0))
        w_pol   = float(row.get("w_policy",1.0))

        print(f"\n=== [{idx}] 回测 {symbol} | "
              f"buy={buy_th}, sell={sell_th}, hold={min_hold}, "
              f"policy={use_policy} ===")

        df = load_data_from_csv(data_file)

        # 行内回测时间（如果参数表里有 start_date / end_date，就优先用表里的）
        row_start = None
        row_end = None
        if has_start_col:
            val = str(row["start_date"]).strip()
            if val and val.lower() != "nan":
                row_start = val
        if has_end_col:
            val = str(row["end_date"]).strip()
            if val and val.lower() != "nan":
                row_end = val

        eff_start = row_start or start_date
        eff_end = row_end or end_date

        # 时间过滤（优先用表里的 backtest_start/backtest_end）
        if use_start:
            df = df[df.index >= pd.to_datetime(use_start)]
        if use_end:
            df = df[df.index <= pd.to_datetime(use_end)]
        if df.empty:
            print("  ⚠️ 时间过滤后无数据，跳过")
            continue


        # 计算因子
        df_fac = compute_stock_factors(df)

        # 政策因子
        if use_policy:
            try:
                df_fac = attach_policy_factor(df_fac, code=code, market=market)
                print("  ✅ 已叠加政策因子")
            except Exception as e:
                print(f"  ⚠️ 政策因子失败: {e}")

        # 打分
        df_scored = attach_scores(df_fac)

        # 重新按权重合成 total_score（可选）
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

        # 生成信号 & 回测
        df_sig = generate_signals_v2(
            df_scored,
            buy_score_thresh=buy_th,
            sell_score_thresh=sell_th,
            min_hold_days=min_hold,
        )

        eq = simple_backtest(df_sig)
        bh = backtest_buy_and_hold(df)

        total_ret = float(eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1)
        cagr = calc_cagr(eq)
        mdd = calc_max_drawdown(eq)

        bh_ret = float(bh["equity"].iloc[-1] / bh["equity"].iloc[0] - 1)
        bh_cagr = calc_cagr(bh)
        bh_mdd = calc_max_drawdown(bh)

        print(
            f"  策略总收益: {total_ret:6.2%} | CAGR: {cagr:6.2%} | MDD: {mdd:6.2%}\n"
            f"  Buy&Hold:  {bh_ret:6.2%} | CAGR: {bh_cagr:6.2%} | MDD: {bh_mdd:6.2%}"
        )

        # === 为当前参数组合导出 dashboard 用的 CSV ===
        # 文件名里带上 symbol + 参数，方便在网页里选
        safe_sym = symbol.replace(".", "")
        dash_dir = Path("./backtest/results/dashboard_runs")
        dash_dir.mkdir(parents=True, exist_ok=True)
        dash_name = (
            f"{safe_sym}_buy{buy_th}_sell{sell_th}_hold{min_hold}_"
            f"policy{use_policy}_dashboard.csv"
        )
        dash_path = dash_dir / dash_name

        export_dashboard_csv(
            price_df=df,   # 本次回测的价格数据（已过滤时间）
            df_sig=df_sig, # 含 position/raw_position
            eq=eq,         # 策略资金曲线
            bh=bh,         # Buy & Hold 资金曲线
            out_path=dash_path,
        )


        rows_out.append(
            {
                "symbol": symbol,
                "data_file": data_file,
                "start_date": eff_start,
                "end_date": eff_end,
                "buy_score_thresh": buy_th,
                "sell_score_thresh": sell_th,
                "min_hold_days": min_hold,
                "use_policy": use_policy,
                "w_trend": w_trend,
                "w_mom": w_mom,
                "w_vol": w_vol,
                "w_risk": w_risk,
                "w_tech": w_tech,
                "w_policy": w_pol,
                "strategy_total_return": total_ret,
                "strategy_cagr": cagr,
                "strategy_max_drawdown": mdd,
                "bh_total_return": bh_ret,
                "bh_cagr": bh_cagr,
                "bh_max_drawdown": bh_mdd,
            }
        )

    if not rows_out:
        print("\n⚠️ 没有任何配置成功回测。")
        return

    df_out = pd.DataFrame(rows_out)
    print("\n===== 参数组合表现（按策略总收益排序） =====")
    df_show = df_out.sort_values(
        ["symbol", "strategy_total_return"], ascending=[True, False]
    )
    pd.set_option("display.max_columns", None)
    print(df_show)

    if out_summary_csv:
        out_csv_str = ensure_utf8_filename(out_summary_csv)
        out_path = Path(out_csv_str)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_show.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n📄 汇总结果已保存: {out_path}")

    # === 准备图表输出目录 ===
    plots_root = Path("backtest/plots")
    plots_root.mkdir(parents=True, exist_ok=True)



    # === 按 symbol 各画一张参数散点图 ===
    by_symbol_dir = plots_root / "param_by_symbol"
    by_symbol_dir.mkdir(parents=True, exist_ok=True)



if __name__ == "__main__":
    # 这里可以用 START_DATE/END_DATE，与你单票脚本保持一致
    run_from_param_table(
        param_csv="./config/param_table.csv",
        out_summary_csv="./backtest/param_table_summary.csv",
        start_date=None,   # 让位给表里的 backtest_start
        end_date=None,     # 让位给表里的 backtest_end
    )
