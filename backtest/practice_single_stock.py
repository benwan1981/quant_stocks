# practice_single_stock.py
import sys
from pathlib import Path
import time
import os 
from datetime import datetime

import pandas as pd
import numpy as np
import json

# === 把项目根目录加入 sys.path，确保能 import factors / fees / config 等 ===
ROOT_DIR = Path(__file__).resolve().parents[1]  # 上上级目录：项目根目录
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from factors.stock_factors import compute_stock_factors, attach_scores

from fees.fee_engine import FeeConfig, FeeEngine

# 如果你已经创建了 factors/policy_factor.py，并实现 attach_policy_factor，
# 就保留下面这一行；否则可以先注释掉：
from factors.policy_factor import attach_policy_factor  # NEW: 政策因子

from backtest.backtest_io import (
    StrategyConfig,
    BacktestMeta,
)

'''from backtest.plotting import (
    save_equity_curve_png,
    save_multi_equity_curve_png,
)
# 如果装了 plotly，再需要的话：
from backtest.plotting import save_equity_curve_html'''

# 👇 新增
from backtest.dashboard_export import export_dashboard_csv

from backtest.plotting import save_backtest_overview_png
from common import ensure_utf8_filename

from common.gm_loader import load_gm_ohlcv, load_gm_ohlcv_by_code


# === 新增：从参数表读取单行配置 ===
PARAM_TABLE_CSV = "./config/param_table.csv"
PARAM_ROW_IDX = 0   # 默认用第 0 行；你可以自己改成 1、2……

def load_config_from_param_table(row_idx: int = PARAM_ROW_IDX) -> dict:
    df_param = pd.read_csv(PARAM_TABLE_CSV, comment="#")
    if row_idx < 0 or row_idx >= len(df_param):
        raise IndexError(f"参数表只有 {len(df_param)} 行，row_idx={row_idx} 越界了")

    row = df_param.iloc[row_idx]

    def _clean_date_cell(val) -> str:
        """把 NaN / 'nan' / 空白统一变成 ''"""
        s = str(val).strip()
        if not s or s.lower() == "nan":
            return ""
        return s

    cfg = {
        "code": str(row["symbol_code"]).strip(),
        "market": str(row["symbol_market"]).strip().upper(),
        "data_file": str(row["data_file"]).strip(),

        "buy_score_thresh": float(row.get("buy_score_thresh", 4.5)),
        "sell_score_thresh": float(row.get("sell_score_thresh", 3.0)),
        "min_hold_days": int(row.get("min_hold_days", 10)),

        "use_policy": int(row.get("use_policy", 0)),

        "w_trend": float(row.get("w_trend", 1.0)),
        "w_mom":   float(row.get("w_mom",   1.0)),
        "w_vol":   float(row.get("w_vol",   1.0)),
        "w_risk":  float(row.get("w_risk",  1.0)),
        "w_tech":  float(row.get("w_tech",  1.0)),
        "w_policy": float(row.get("w_policy", 1.0)),

        # 这里用清洗函数
        "backtest_start": _clean_date_cell(row.get("backtest_start", "")),
        "backtest_end":   _clean_date_cell(row.get("backtest_end", "")),
    }
    return cfg



# ===== 回测时间段配置（可选） =====
START_DATE = "2018-01-01"
END_DATE   = "2025-12-30"

def load_data_from_csv(path: str) -> pd.DataFrame:
    """
    通用加载函数：
    - 如果是“原始 gm CSV”（带 eob/open/high/low/close/volume），
      由 load_gm_ohlcv 统一转为 date,open,high,low,close,volume
    - 如果是之前已经处理好的标准 CSV（有 date 列），
      load_gm_ohlcv 也能兼容
    """
    # 先用 loader 做统一格式转换
    df = load_gm_ohlcv(path)
    # 我们内部一直用 date 做 index
    df = df.set_index("date").sort_index()
    return df[["open", "high", "low", "close", "volume"]]



def generate_signals(df_scored: pd.DataFrame,
                     buy_q: float = 0.75,
                     sell_q: float = 0.40) -> pd.DataFrame:
    df = df_scored.copy()

    # 先算阈值（一个标的整个历史范围）
    q_buy  = df['total_score'].quantile(buy_q)
    q_sell = df['total_score'].quantile(sell_q)

    # 安全起见，避免 sell 阈值反而 > buy 阈值
    if q_sell > q_buy:
        q_sell = q_buy * 0.8

    # 状态机：raw_pos 是“策略意图”的持仓状态（0/1）
    raw_pos = []
    pos = 0  # 初始空仓

    for score in df['total_score']:
        if np.isnan(score):
            # 因子不足期，保持空仓
            pos = 0
        else:
            if pos == 0:
                # 当前空仓，只有当 score 突破高阈值时才开仓
                if score >= q_buy:
                    pos = 1
            else:
                # 当前持仓，只有当 score 跌破较低阈值时才清仓
                if score <= q_sell:
                    pos = 0
        raw_pos.append(pos)

    df['raw_position'] = raw_pos
    # T+1 执行：今天的实际 position 用昨天的 raw_position
    df['position'] = df['raw_position'].shift(1).fillna(0).astype(int)

    return df

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
    V2 信号逻辑（单标的）：
    - 只用 total_score + trend_score + risk_score 决定开/平仓
    - 引入“最小持有天数”约束，避免频繁进出

    规则（基于 df_scored 中的列）：
    - 开仓条件（强信号）：
        total_score >= buy_score_thresh
        且 trend_score >= min_trend_for_buy
        且 risk_score >= min_risk_for_buy
    - 平仓条件（弱信号，且已经持有 min_hold_days 以上）：
        total_score <= sell_score_thresh
        或 trend_score <= max_trend_for_sell
    - 因子缺失(NaN)：强制空仓

    raw_position：策略“意图”的仓位（今天根据今天的分数做决策）
    position：T+1 执行，今天的持仓 = 昨天的 raw_position
    """
    df = df_scored.copy()

    raw_pos: list[int] = []
    hold_days_intent: list[int] = []

    pos = 0              # 当前意图仓位（0/1）
    hold_days = 0        # 意图层面的持有天数（连续 pos==1 的天数）

    for idx, row in df.iterrows():
        score = row.get("total_score", np.nan)
        trend = row.get("trend_score", np.nan)
        risk  = row.get("risk_score",  np.nan)

        # 默认：因子缺失时直接空仓
        if np.isnan(score) or np.isnan(trend) or np.isnan(risk):
            pos = 0
            hold_days = 0
            raw_pos.append(pos)
            hold_days_intent.append(hold_days)
            continue

        if pos == 0:
            # 当前空仓：只在“强信号”出现时开仓
            strong_buy = (
                (score >= buy_score_thresh) and
                (trend >= min_trend_for_buy) and
                (risk  >= min_risk_for_buy)
            )
            if strong_buy:
                pos = 1
                hold_days = 0  # 新开仓，从 0 天开始
            # 空仓时 hold_days 记 0
        else:
            # 当前有仓，先增加持有天数
            hold_days += 1

            # 弱信号 / 风险恶化：考虑平仓
            weak_or_risk_off = (
                (score <= sell_score_thresh) or
                (trend <= max_trend_for_sell)
            )

            # 只有持有达到 min_hold_days 以后，才允许因为弱信号平仓
            if weak_or_risk_off and hold_days >= min_hold_days:
                pos = 0
                hold_days = 0

        raw_pos.append(pos)
        hold_days_intent.append(hold_days if pos == 1 else 0)

    df["raw_position"] = raw_pos
    df["hold_days_intent"] = hold_days_intent

    # T+1 执行：今天的实际 position = 昨天的 raw_position
    df["position"] = df["raw_position"].shift(1).fillna(0).astype(int)

    return df


def backtest_buy_and_hold(df: pd.DataFrame,
                          initial_cash: float = 100000,
                          fee_rate: float = 0.0005,
                          slippage: float = 0.0005) -> pd.DataFrame:
    """
    非严格版：第一个交易日开盘全仓买入，一直拿到最后一天收盘。
    用来对比策略是否离谱。
    """
    df = df.copy().reset_index()
    cash = initial_cash
    shares = 0.0
    records = []

    for i, row in df.iterrows():
        date = row['date']
        price_open = row['open']
        price_close = row['close']

        if i == 0:
            # 第一天开盘全仓买入
            buy_amount = cash
            exec_price = price_open * (1 + slippage)
            shares = buy_amount / exec_price
            fee = buy_amount * fee_rate
            cash -= buy_amount + fee

        market_value = shares * price_close
        equity = cash + market_value

        records.append({
            'date': date,
            'cash': cash,
            'shares': shares,
            'market_value': market_value,
            'equity': equity,
        })

    eq = pd.DataFrame(records).set_index('date')
    eq['ret'] = eq['equity'].pct_change().fillna(0)
    return eq

def summarize_annual_performance(eq: pd.DataFrame, label: str = "策略"):
    """
    按自然年统计收益和最大回撤。
    eq: simple_backtest 或 backtest_buy_and_hold 返回的 DataFrame，index 为日期，含 'equity' 列。
    label: 打印时的名字（比如 '策略' 或 'Buy & Hold'）
    """
    eq = eq.copy()
    # 确保 index 是 DatetimeIndex
    if not isinstance(eq.index, pd.DatetimeIndex):
        eq.index = pd.to_datetime(eq.index)

    print(f"\n📅 {label}年度表现：")
    years = sorted(eq.index.year.unique())

    year_stats: dict[int, tuple[float, float]] = {}

    for year in years:
        df_y = eq[eq.index.year == year]
        if df_y.empty:
            continue

        start_eq = df_y["equity"].iloc[0]
        end_eq = df_y["equity"].iloc[-1]
        year_ret = end_eq / start_eq - 1

        cummax = df_y["equity"].cummax()
        drawdown = df_y["equity"] / cummax - 1
        max_dd = drawdown.min()

        year_stats[year] = (float(year_ret), float(max_dd))

        print(
            f"  {year} 年: 收益 {year_ret:6.2%}  最大回撤 {max_dd:6.2%} "
            f"(期初 {start_eq:,.2f} → 期末 {end_eq:,.2f})"
        )
    return year_stats

def save_backtest_to_csv(
    eq: pd.DataFrame,
    bh: pd.DataFrame,
    csv_path: str,
    out_dir: str = "./backtest/results",
):
    """
    把本次回测的结果保存到 CSV，方便之后画图、回溯。

    eq: 策略回测结果（simple_backtest 返回）
    bh: Buy & Hold 回测结果
    csv_path: 原始数据文件路径（用来从文件名里解析标的）
    out_dir: 输出目录，默认 ./backtest/results
    """
    os.makedirs(out_dir, exist_ok=True)

    base_name = ensure_utf8_filename(Path(csv_path).stem)  # 如: 600383_金地集团_D_gm
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    eq_file = Path(out_dir) / ensure_utf8_filename(f"{base_name}_strategy_{ts}.csv")
    bh_file = Path(out_dir) / ensure_utf8_filename(f"{base_name}_buyhold_{ts}.csv")

    # 直接带 index 保存，index 就是 date，后面画图方便
    eq.to_csv(eq_file, encoding="utf-8-sig")
    bh.to_csv(bh_file, encoding="utf-8-sig")

    print("\n📄 回测结果已保存：")
    print(f"  策略曲线:   {eq_file}")
    print(f"  Buy&Hold: {bh_file}")

def save_backtest_report_to_json(
    eq: pd.DataFrame,
    bh: pd.DataFrame,
    df_sig: pd.DataFrame,
    annual_strategy: dict,
    annual_bh: dict,
    csv_path: str,
    out_dir: str = "./backtest/results",
    meta_extra: dict = None
):
    """
    保存完整回测结果到 JSON，适合后续回溯与扩展。
    """
    os.makedirs(out_dir, exist_ok=True)

    base_name = ensure_utf8_filename(Path(csv_path).stem)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = Path(out_dir) / ensure_utf8_filename(f"{base_name}_backtest_{ts}.json")

    # equity → list[dict]
    eq_list = [
        {"date": d.strftime("%Y-%m-%d"),
         "equity": float(v),
         "cash": float(eq.loc[d, "cash"]),
         "market_value": float(eq.loc[d, "market_value"])}
        for d, v in eq["equity"].items()
    ]
    bh_list = [
        {"date": d.strftime("%Y-%m-%d"), "equity": float(v)}
        for d, v in bh["equity"].items()
    ]

    # signals
    sig_list = [
        {
            "date": idx.strftime("%Y-%m-%d"),
            "raw_position": int(r.raw_position),
            "position": int(r.position)
        }
        for idx, r in df_sig.iterrows()
    ]

    result = {
        "meta": {
            "symbol": base_name.split("_")[0],
            "data_file": csv_path,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0"
        },
        "performance": {
            "strategy": annual_strategy,
            "buy_and_hold": annual_bh
        },
        "equity_curve": {
            "strategy": eq_list,
            "buy_and_hold": bh_list
        },
        "signals": sig_list,
    }

    if meta_extra:
        result["meta"].update(meta_extra)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n📄 JSON 回测结果已保存: {out_file}")


def simple_backtest(df_sig: pd.DataFrame,
                    initial_cash: float = 100000,
                    fee_rate: float = 0.0005,
                    slippage: float = 0.0005,
                    stop_loss_pct: float = 0.10,
                    trail_stop_pct: float = 0.15,
                    fee_engine: FeeEngine | None = None,
                    ) -> pd.DataFrame:
    """
    fee_engine: 若为 None，会根据 fee_rate 创建一个默认的 FeeEngine。
    """
    df = df_sig.copy().reset_index()
    cash = initial_cash
    shares = 0.0

    # 如果外面没传，就自己建一个
    if fee_engine is None:
        cfg = FeeConfig(trade_fee_rate=fee_rate,
                        stamp_duty_rate=0.001,
                        financing_rate_year=0.06)
        fee_engine = FeeEngine(cfg)

    entry_price = None
    peak_price = None

    records = []

    prev_pos = 0
    for i, row in df.iterrows():
        date = row['date']
        price_open = row['open']
        price_close = row['close']
        target_pos = int(row['position'])

        # ======== 先做风控、决定 target_pos (略) ========
        # ... 这里保留你之前的止损逻辑 ...

        # 统计当天买卖金额（用于费用模块）
        day_buy_amount = 0.0
        day_sell_amount = 0.0

        # ======== 按目标仓位执行交易 ========
        if prev_pos == 0 and target_pos == 1 and shares == 0:
            # 全仓买入
            buy_amount = cash
            exec_price = price_open * (1 + slippage)
            shares = buy_amount / exec_price
            day_buy_amount = buy_amount  # 记录今日买入金额

            cash -= buy_amount  # 暂不扣手续费，统一交给 FeeEngine

            entry_price = exec_price
            peak_price = price_close

        elif prev_pos == 1 and target_pos == 0 and shares > 0:
            # 全部卖出
            exec_price = price_open * (1 - slippage)
            sell_amount = shares * exec_price
            day_sell_amount = sell_amount  # 记录今日卖出金额

            cash += sell_amount  # 暂不扣手续费

            shares = 0.0
            entry_price = None
            peak_price = None

        # ======== 当日费用统一计算 & 扣除 ========
        # 当前没有融资，就先传 margin_balance=0.0，后面要做融资时再改
        day_fee = fee_engine.on_day(
            date=date,
            buy_amount=day_buy_amount,
            sell_amount=day_sell_amount,
            margin_balance=0.0,
            days=1,
        )
        cash -= day_fee

        # ======== 计算市值和权益 ========
        market_value = shares * price_close
        equity = cash + market_value

        records.append({
            'date': date,
            'cash': cash,
            'shares': shares,
            'market_value': market_value,
            'equity': equity,
            'position': target_pos,
            'day_fee': day_fee,
        })

        prev_pos = target_pos

    eq = pd.DataFrame(records).set_index('date')
    eq['ret'] = eq['equity'].pct_change().fillna(0)

    # 把 fee_engine 挂到结果上，方便外面 summary
    eq._fee_engine = fee_engine
    return eq



def main():
    start_time = time.time()

    # ===== 0. 从参数表读取本次配置 =====
    cfg = load_config_from_param_table(PARAM_ROW_IDX)
    code = cfg["code"]
    market = cfg["market"]
    csv_path = cfg["data_file"]

    print(f"本次回测标的: {market}.{code} | 数据文件: {csv_path}")

    # ===== 1. 加载数据（统一用 loader 转成标准 OHLCV） =====
    df = load_gm_ohlcv(csv_path)
    df = df.set_index("date").sort_index()
    # ===== 按时间段过滤（优先用表里的 backtest_start/backtest_end） =====
    eff_start = cfg["backtest_start"] or START_DATE
    eff_end   = cfg["backtest_end"] or END_DATE

    if eff_start:
        df = df[df.index >= pd.to_datetime(eff_start)]
    if eff_end:
        df = df[df.index <= pd.to_datetime(eff_end)]
    if len(df) == 0:
        raise RuntimeError("时间段过滤后没有数据，请检查参数表或 START_DATE / END_DATE")

    print(f"时间段过滤后: {df.index[0].date()} ~ {df.index[-1].date()}，共 {len(df)} 个交易日")



    # ===== 2. 计算因子 =====
    df_fac = compute_stock_factors(df)

    # ★★ 如果已经实现 attach_policy_factor，就打开下面这段；否则可以先注释掉 ★★
    # 当前回测标的是 159892（恒生医药ETF），所以 code 写 "159892"
    has_policy = False
    if cfg["use_policy"]:
        try:
            df_fac = attach_policy_factor(df_fac, code=code, market=market)
            has_policy = True
            print("✅ 本次回测已叠加政策因子")
        except Exception as e:
            print(f"⚠️ 政策因子未生效（可忽略）：{e}")
            has_policy = False
    else:
        print("ℹ️ 本次配置未启用政策因子（use_policy=0）")


    print("因子样例：")
    if has_policy and "policy_score" in df_fac.columns:
        print(df_fac[['close', 'ma20', 'mom20', 'vol_ratio_20', 'rsi14', 'policy_score']].tail())
    else:
        print(df_fac[['close', 'ma20', 'mom20', 'vol_ratio_20', 'rsi14']].tail())


    # ===== 3. 打分 =====
    df_scored = attach_scores(df_fac)

    # === 按参数表里的权重重算 total_score，和 run_param_table 保持一致 ===
    w_trend = cfg["w_trend"]
    w_mom   = cfg["w_mom"]
    w_vol   = cfg["w_vol"]
    w_risk  = cfg["w_risk"]
    w_tech  = cfg["w_tech"]
    w_pol   = cfg["w_policy"]

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


    print("\n打分样例（最近10天）：")
    print(df_scored[['close',
                     'trend_score', 'momentum_score',
                     'volume_score', 'risk_score',
                     'technical_score', 'total_score']].tail(10))

    # ===== 4. 生成信号（使用 V2 策略逻辑） =====
    df_sig = generate_signals_v2(
        df_scored,
        buy_score_thresh=cfg["buy_score_thresh"],
        sell_score_thresh=cfg["sell_score_thresh"],
        min_hold_days=cfg["min_hold_days"],
    )
    print("\n本次信号参数：",
          f"buy={cfg['buy_score_thresh']}, "
          f"sell={cfg['sell_score_thresh']}, "
          f"min_hold={cfg['min_hold_days']}")    
    print("\n最近20天信号（V2）：")
    print(df_sig[['close', 'total_score', 'raw_position', 'position', 'hold_days_intent']].tail(20))


    # ===== 5. 简单回测 =====
    eq = simple_backtest(
        df_sig,
        initial_cash=100000,
        fee_rate=0.0005,
        slippage=0.0005,
        stop_loss_pct=0.10,
        trail_stop_pct=0.15,
        fee_engine=None,
    )

    print("\n回测结果（最后10天）：")
    print(eq.tail(10))

    total_return = eq['equity'].iloc[-1] / eq['equity'].iloc[0] - 1
    cummax = eq['equity'].cummax()
    drawdown = eq['equity'] / cummax - 1
    max_dd = drawdown.min()
    print(f"\n总收益: {total_return:.2%}")
    print(f"最大回撤: {max_dd:.2%}")

    # ⭐ 新增：按年统计策略表现
    year_stats_strategy = summarize_annual_performance(eq, label="策略")


    # ===== 6. Buy & Hold 对照 =====
    bh = backtest_buy_and_hold(df)
    bh_total_return = bh['equity'].iloc[-1] / bh['equity'].iloc[0] - 1
    print(f"\nBuy & Hold 总收益: {bh_total_return:.2%}")

    # ===== 7. 保存回测结果为 JSON =====
    strategy_cfg = StrategyConfig(
        name="score_trend_v2_with_policy",
        version="1.0",
        description="打分 + 趋势 + 风控 + 政策因子，单标的波段策略",
        params={
            "buy_score_thresh": cfg["buy_score_thresh"],
            "sell_score_thresh": cfg["sell_score_thresh"],
            "min_hold_days": cfg["min_hold_days"],
            "min_trend_for_buy": 1.5,
            "min_risk_for_buy": 1.5,
            "max_trend_for_sell": 0.5,
            "initial_cash": 100000,
            "fee_rate": 0.0005,
            "slippage": 0.0005,
        },
    )


    meta = BacktestMeta(
        symbol=f"{market}.{code}",
        symbol_name=code,  # 如果以后想用中文名，可以从 csv_path 的文件名里解析
        data_source="gm",
        start_date=str(df.index[0].date()),
        end_date=str(df.index[-1].date()),
        initial_cash=100000,
        benchmark="Buy&Hold(本标的)",
    )



    # ===== 8. 导出图表 =====
    # 单策略资金曲线

    # ===== 9. 总览图：K线 + 买卖点 + 资金曲线 =====
    overview_png = f"./backtest/plots/{code}_overview.png"
    save_backtest_overview_png(
        price_df=df,          # 日K数据（过滤时间段后的）
        df_sig=df_sig,        # 含 position 的信号表
        eq=eq,                # 策略资金曲线
        bh=bh,                # Buy & Hold 资金曲线
        out_path=overview_png,
        title=f"{code} 回测总览",
    )


    # 策略 vs Buy&Hold 对比图


    # 如需HTML可交互图（需安装 plotly）
    # save_equity_curve_html(
    #     eq,
    #     out_path="./backtest/plots/600383_strategy_equity.html",
    #     title="600383 策略资金曲线(交互)",
    #     series_name="strategy",
    # )

    # ⭐ 新增：按年统计 Buy & Hold 表现
    year_stats_bh = summarize_annual_performance(bh, label="Buy & Hold")
    
    # ⭐ 把本次回测结果存盘，方便以后画图/分析
    save_backtest_to_csv(eq, bh, csv_path)

    # annual strategy summary → dict
    annual_strategy = {
        y: {"return": float(ret), "max_dd": float(dd)}
        for y, (ret, dd) in year_stats_strategy.items()
    }
    # annual bh summary → dict
    annual_bh = {
        y: {"return": float(ret), "max_dd": float(dd)}
        for y, (ret, dd) in year_stats_bh.items()
    }

    # 把策略配置 + 回测元信息写进 JSON 的 meta 里
    meta_extra = {
        "strategy": {
            "name": strategy_cfg.name,
            "version": strategy_cfg.version,
            "description": strategy_cfg.description,
            "params": strategy_cfg.params,
        },
        "backtest_meta": {
            "symbol": meta.symbol,
            "symbol_name": meta.symbol_name,
            "data_source": meta.data_source,
            "start_date": meta.start_date,
            "end_date": meta.end_date,
            "initial_cash": meta.initial_cash,
            "benchmark": meta.benchmark,
        },
    }

    save_backtest_report_to_json(
        eq=eq,
        bh=bh,
        df_sig=df_sig,
        annual_strategy=annual_strategy,
        annual_bh=annual_bh,
        csv_path=csv_path,
        meta_extra=meta_extra,
    )

    # === 导出给网页 dashboard 用的 CSV ===
    dash_csv_name = Path(csv_path).stem + "_dashboard.csv"
    dash_csv_path = Path("./backtest/results") / dash_csv_name
    export_dashboard_csv(
        price_df=df,
        df_sig=df_sig,
        eq=eq,
        bh=bh,
        out_path=dash_csv_path,
    )
    # ===== 打印费用汇总 =====
    fee_engine = getattr(eq, "_fee_engine", None)
    if fee_engine is not None:
        fee_summary = fee_engine.summary()
        print("\n费用汇总：")
        print(f"  交易佣金总额: {fee_summary['total_trade_fee']:.2f}")
        print(f"  印花税总额:   {fee_summary['total_stamp_duty']:.2f}")
        print(f"  融资利息总额: {fee_summary['total_financing_fee']:.2f}")
        print(f"  费用合计:     {fee_summary['total_fee']:.2f}")

        fee_ratio = fee_summary['total_fee'] / 100000  # 初始资金 10 万
        print(f"  费用占初始资金比例: {fee_ratio:.2%}")

if __name__ == "__main__":
    main()
