# practice_single_stock.py
import pandas as pd
import numpy as np
import time
from fees import FeeEngine,FeeConfig

# ===== 回测时间段配置（可选） =====
# 用字符串写就行，比如 "2015-01-01"；不限制就填 None
START_DATE ="2018-01-01"      # 例如 "2015-01-01"
END_DATE   ="2025-12-30"      # 例如 "2020-12-31"



# === 如果你有自己的数据API，可以换成 from data_api import DataAPI ===
# 这里示范用 CSV 或你已有的 df
from factors import compute_stock_factors, attach_scores


def load_data_from_csv(path: str) -> pd.DataFrame:
    """
    假设你有一个csv，包含列：date, open, high, low, close, volume
    date: 形如 2025-01-01
    """
    df = pd.read_csv(path)
    # 统一格式
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    return df[['open', 'high', 'low', 'close', 'volume']]


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

        print(
            f"  {year} 年: 收益 {year_ret:6.2%}  最大回撤 {max_dd:6.2%} "
            f"(期初 {start_eq:,.2f} → 期末 {end_eq:,.2f})"
        )


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

    # ===== 1. 加载数据 =====
    csv_path = r"./data/159892_D_qfq.csv"  # TODO: 改成你自己的路径
    df = load_data_from_csv(csv_path)

    # ===== 按时间段过滤（如果设置了 START_DATE / END_DATE） =====
    if START_DATE or END_DATE:
        if START_DATE:
            df = df[df.index >= pd.to_datetime(START_DATE)]
        if END_DATE:
            df = df[df.index <= pd.to_datetime(END_DATE)]
        if len(df) == 0:
            raise RuntimeError("时间段过滤后没有数据，请检查 START_DATE / END_DATE")
        print(f"时间段过滤后: {df.index[0].date()} ~ {df.index[-1].date()}，共 {len(df)} 个交易日")


    print("原始数据：", df.head())

    # ===== 2. 计算因子 =====
    df_fac = compute_stock_factors(df)
    print("因子样例：")
    print(df_fac[['close', 'ma20', 'mom20', 'vol_ratio_20', 'rsi14']].tail())

    # ===== 3. 打分 =====
    df_scored = attach_scores(df_fac)
    print("\n打分样例（最近10天）：")
    print(df_scored[['close',
                     'trend_score', 'momentum_score',
                     'volume_score', 'risk_score',
                     'technical_score', 'total_score']].tail(10))

    # ===== 4. 生成信号（使用 V2 策略逻辑） =====
    df_sig = generate_signals_v2(
        df_scored,
        buy_score_thresh=4.5,
        sell_score_thresh=3.0,
        min_hold_days=10,
    )
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
    summarize_annual_performance(eq, label="策略")


    # ===== 6. Buy & Hold 对照 =====
    bh = backtest_buy_and_hold(df)
    bh_total_return = bh['equity'].iloc[-1] / bh['equity'].iloc[0] - 1
    print(f"\nBuy & Hold 总收益: {bh_total_return:.2%}")

    # ⭐ 新增：按年统计 Buy & Hold 表现
    summarize_annual_performance(bh, label="Buy & Hold")
    
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