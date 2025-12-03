# practice_universe.py
import pandas as pd
import numpy as np
from pathlib import Path

from factors import compute_stock_factors, attach_scores
from practice_single_stock import load_data_from_csv
from fees import FeeEngine, FeeConfig

# ===== 配置区 =====
DATA_DIR = Path("./data")
CSV_SUFFIX = "_D_qfq.csv"

# 回测时间范围（包含端点），留 None 表示不限制
# 例如：START_DATE = "2023-01-01"  END_DATE = "2025-11-17"
START_DATE = "2019-01-01"
END_DATE = "2025-11-14"

TOP_N = 5  # 组合回测时选前多少只

UNIVERSE = [
    # 金融
    "601939",   # 建设银行
    "600036",   # 招商银行
    "601318",   # 中国平安

    # 消费 / 医药
    "600519",   # 贵州茅台
    "000858",   # 五粮液
    "600276",   # 恒瑞医药

    # 成长 / 新能源
    "300750",   # 宁德时代
    "002594",   # 比亚迪
    "601012",   # 隆基绿能

    # 指数 / ETF
    "510300",   # 沪深300ETF
    "159915",   # 创业板ETF
    "159892",   # 恒生医药ETF
]

# 代码 -> 名称
NAME_MAP = {
    "601939": "建设银行",
    "600036": "招商银行",
    "601318": "中国平安",
    "600519": "贵州茅台",
    "000858": "五粮液",
    "600276": "恒瑞医药",
    "300750": "宁德时代",
    "002594": "比亚迪",
    "601012": "隆基绿能",
    "510300": "沪深300ETF",
    "159915": "创业板ETF",
    "159892": "恒生医药ETF",
}


# ===== 工具函数：加载股票池数据 =====
def load_universe_data(universe,
                       data_dir: Path = DATA_DIR,
                       suffix: str = CSV_SUFFIX):
    """
    读取股票池里所有标的的历史数据，返回 {code: df}
    要求每个标的有一个 csv: data_dir / f"{code}{suffix}"
    """
    data = {}
    for code in universe:
        csv_path = data_dir / f"{code}{suffix}"
        if not csv_path.exists():
            print(f"⚠️ 未找到数据文件: {csv_path}")
            continue
        df = load_data_from_csv(str(csv_path))
        data[code] = df
    return data


def _filter_dates(dates: pd.DatetimeIndex,
                  start_date: str | None,
                  end_date: str | None) -> pd.DatetimeIndex:
    """按 START_DATE / END_DATE 过滤日期"""
    if start_date:
        dates = dates[dates >= pd.to_datetime(start_date)]
    if end_date:
        dates = dates[dates <= pd.to_datetime(end_date)]
    return dates


def find_common_dates(data_dict,
                      start_date: str | None = None,
                      end_date: str | None = None) -> pd.DatetimeIndex:
    """
    找到所有标的的公共交易日期交集，并按开始/结束日期过滤。
    用于单日打分时选择 as_of_date。
    """
    if not data_dict:
        return pd.DatetimeIndex([])

    sets = [set(df.index) for df in data_dict.values()]
    common = set.intersection(*sets)
    common_index = pd.DatetimeIndex(sorted(common))
    common_index = _filter_dates(common_index, start_date, end_date)
    return common_index


# ===== 单日打分（排名用） =====
def score_universe_on_date(data_dict,
                           as_of_date: pd.Timestamp) -> pd.DataFrame:
    """
    对股票池在某一天进行打分：
    - 对每只股票算因子 + 打分
    - 取截至 as_of_date 的 total_score 作为当日得分
    返回: index=code, 列包含 name, close, 各子分 & total_score
    """
    rows = []

    for code, df in data_dict.items():
        # 只用 as_of_date 之前的数据
        df_sub = df[df.index <= as_of_date]
        if len(df_sub) < 60:  # 数据太短因子不稳定，跳过
            print(f"⚠️ {code} 在 {as_of_date.date()} 之前数据不足，跳过")
            continue

        # 计算因子和打分
        df_fac = compute_stock_factors(df_sub)
        df_scored = attach_scores(df_fac)

        if as_of_date not in df_scored.index:
            print(f"⚠️ {code} 在 {as_of_date.date()} 因子数据缺失，跳过")
            continue

        row = df_scored.loc[as_of_date]
        rows.append({
            "code": code,
            "name": NAME_MAP.get(code, code),
            "date": as_of_date,
            "close": row["close"],
            "trend_score": row["trend_score"],
            "momentum_score": row["momentum_score"],
            "volume_score": row["volume_score"],
            "risk_score": row["risk_score"],
            "technical_score": row["technical_score"],
            "total_score": row["total_score"],
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).set_index("code")
    # 按 total_score 从高到低排序
    result = result.sort_values("total_score", ascending=False)
    return result


# ===== 组合回测相关函数 =====
def prepare_scored_universe(data_dict: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    为股票池里每只标的计算因子和打分，返回 {code: df_scored}
    df_scored 中至少要有 total_score 列。
    """
    scored = {}
    for code, df in data_dict.items():
        df_fac = compute_stock_factors(df)
        df_scored = attach_scores(df_fac)
        scored[code] = df_scored
    return scored


def get_backtest_dates(scored_dict: dict[str, pd.DataFrame],
                       start_date: str | None = None,
                       end_date: str | None = None) -> pd.DatetimeIndex:
    """
    找到所有标的 total_score 都非空的共同日期，用于组合回测。
    """
    if not scored_dict:
        return pd.DatetimeIndex([])

    date_sets = []
    for df in scored_dict.values():
        valid_dates = df.index[df["total_score"].notna()]
        date_sets.append(set(valid_dates))

    common = set.intersection(*date_sets)
    dates = pd.DatetimeIndex(sorted(common))
    dates = _filter_dates(dates, start_date, end_date)
    return dates


def pick_topN_on_date(scored_dict: dict[str, pd.DataFrame],
                      scoring_date: pd.Timestamp,
                      top_n: int) -> list[str]:
    """
    在给定 scoring_date 上，根据 total_score 选出前 top_n 只股票代码。
    """
    rows = []
    for code, df in scored_dict.items():
        if scoring_date not in df.index:
            continue
        score = df.at[scoring_date, "total_score"]
        if pd.isna(score):
            continue
        rows.append((code, score))

    if not rows:
        return []

    rows.sort(key=lambda x: x[1], reverse=True)
    return [code for code, _ in rows[:top_n]]


def backtest_topN_portfolio(data_dict: dict[str, pd.DataFrame],
                            scored_dict: dict[str, pd.DataFrame],
                            top_n: int = TOP_N,
                            initial_cash: float = 100000,
                            fee_rate: float = 0.0005,
                            slippage: float = 0.0005,
                            dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """
    简单组合回测：
    - 每个交易日 t，用 t-1 日的 total_score 选出 TopN
    - 在 t 日开盘价等权买入/调仓
    - 每天全换仓（先卖完昨天持仓，再买当天 TopN），方便理解
    - 用 FeeEngine 统计佣金 + 印花税
    返回组合权益曲线 eq_portfolio（含 day_fee、holding_codes）
    """
    if dates is None:
        dates = get_backtest_dates(scored_dict, START_DATE, END_DATE)

    if len(dates) < 2:
        raise RuntimeError("可回测的共同日期太少，无法进行组合回测")

    cfg = FeeConfig(
        trade_fee_rate=fee_rate,
        stamp_duty_rate=0.001,
        financing_rate_year=0.06,
    )
    fee_engine = FeeEngine(cfg)

    cash = initial_cash
    holdings: dict[str, float] = {}  # code -> shares
    records = []

    for i in range(1, len(dates)):
        scoring_date = dates[i - 1]
        trade_date = dates[i]

        # 1) 选出 scoring_date 的 TopN
        selected = pick_topN_on_date(scored_dict, scoring_date, top_n)

        day_buy_amount = 0.0
        day_sell_amount = 0.0

        # 2) 先全部卖出旧持仓（简单版：每天完全重平衡）
        for code, shares in list(holdings.items()):
            if shares <= 0:
                continue
            df = data_dict[code]
            if trade_date not in df.index:
                continue
            price_open = df.loc[trade_date, "open"] * (1 - slippage)
            amount = shares * price_open
            cash += amount
            day_sell_amount += amount
        holdings = {}

        # 3) 再等权买入新的 TopN
        if selected:
            cash_per_stock = cash / len(selected)
            for code in selected:
                df = data_dict[code]
                if trade_date not in df.index:
                    continue
                price_open = df.loc[trade_date, "open"] * (1 + slippage)
                if price_open <= 0:
                    continue
                shares = cash_per_stock / price_open
                amount = shares * price_open
                if amount <= 0:
                    continue
                cash -= amount
                day_buy_amount += amount
                holdings[code] = shares

        # 4) 费用（佣金 + 印花税），统一从现金扣
        day_fee = fee_engine.on_day(
            date=trade_date,
            buy_amount=day_buy_amount,
            sell_amount=day_sell_amount,
            margin_balance=0.0,
            days=1,
        )
        cash -= day_fee

        # 5) 用收盘价计算当日市值 & 权益
        market_value = 0.0
        for code, shares in holdings.items():
            df = data_dict[code]
            if trade_date not in df.index:
                continue
            price_close = df.loc[trade_date, "close"]
            market_value += shares * price_close

        equity = cash + market_value

        records.append({
            "date": trade_date,
            "cash": cash,
            "market_value": market_value,
            "equity": equity,
            "day_fee": day_fee,
            "holding_codes": ",".join(sorted(holdings.keys())),
        })

    eq = pd.DataFrame(records).set_index("date")
    eq["ret"] = eq["equity"].pct_change().fillna(0)

    # 挂上 fee_engine，方便外部 summary
    eq._fee_engine = fee_engine
    return eq


def backtest_buy_and_hold_universe(data_dict: dict[str, pd.DataFrame],
                                   dates: pd.DatetimeIndex,
                                   initial_cash: float = 100000,
                                   fee_rate: float = 0.0005,
                                   slippage: float = 0.0005) -> pd.DataFrame:
    """
    Buy & Hold 组合回测：
    - 在第一个交易日开盘等权买入 UNIVERSE 里的所有标的
    - 中间不调仓，一直持有到最后一日
    - 只在买入时收取佣金，未计卖出印花税（相当于“持有到期、未真正卖出”）
    """
    if len(dates) < 1:
        raise RuntimeError("日期序列为空，无法回测 Buy & Hold")

    cfg = FeeConfig(
        trade_fee_rate=fee_rate,
        stamp_duty_rate=0.001,
        financing_rate_year=0.06,
    )
    fee_engine = FeeEngine(cfg)

    first_date = dates[0]
    codes = list(data_dict.keys())

    cash = initial_cash
    holdings: dict[str, float] = {}

    # 第一天等权买入
    day_buy_amount = 0.0
    for code in codes:
        df = data_dict[code]
        if first_date not in df.index:
            continue
        price_open = df.loc[first_date, "open"] * (1 + slippage)
        if price_open <= 0:
            continue
        cash_per_stock = initial_cash / len(codes)
        shares = cash_per_stock / price_open
        amount = shares * price_open
        cash -= amount
        day_buy_amount += amount
        holdings[code] = shares

    # 计算第一天的手续费（只买不卖）
    day_fee = fee_engine.on_day(
        date=first_date,
        buy_amount=day_buy_amount,
        sell_amount=0.0,
        margin_balance=0.0,
        days=1,
    )
    cash -= day_fee

    # 逐日估值
    records = []
    for d in dates:
        market_value = 0.0
        for code, shares in holdings.items():
            df = data_dict[code]
            if d not in df.index:
                continue
            price_close = df.loc[d, "close"]
            market_value += shares * price_close
        equity = cash + market_value

        records.append({
            "date": d,
            "cash": cash,
            "market_value": market_value,
            "equity": equity,
            "day_fee": day_fee if d == first_date else 0.0,
        })

    eq = pd.DataFrame(records).set_index("date")
    eq["ret"] = eq["equity"].pct_change().fillna(0)

    # 同样挂一个 fee_engine，方便外部统计费用（这里只会有第一次买入的费用）
    eq._fee_engine = fee_engine
    return eq


# ===== main：先做单日排行，再做区间 Top5 组合 & Buy & Hold 对比 =====
def main():
    # 1. 读取股票池数据
    data_dict = load_universe_data(UNIVERSE)
    if not data_dict:
        print("❌ 股票池数据为空，请检查 CSV 路径和 UNIVERSE 配置")
        return

    # 2. 找公共交易日期，并选择一个 as_of_date 做“当天排行”
    common_dates = find_common_dates(data_dict, START_DATE, END_DATE)
    if len(common_dates) == 0:
        print("❌ 各股票日期没有交集，请检查数据完整性或调整 START/END_DATE")
        return

    as_of_date = common_dates[-1]  # 用区间内最后一个共同交易日
    print(f"📅 选取评级日期: {as_of_date.date()}")
    if START_DATE or END_DATE:
        print(f"⏱ 回测区间限制: {START_DATE or '最早'} ~ {END_DATE or '最新'}")

    # 3. 对当日进行打分排行
    rank_df = score_universe_on_date(data_dict, as_of_date)
    if rank_df.empty:
        print("❌ 打分结果为空，请检查因子/打分函数")
        return

    N = min(10, len(rank_df))
    print(f"\n股票池在 {as_of_date.date()} 的打分排行（Top {N}）：")
    cols = ["name", "close", "trend_score", "momentum_score",
            "volume_score", "risk_score", "technical_score", "total_score"]
    print(rank_df[cols].head(N))

    print("\nscore 分布：")
    print(rank_df["total_score"].describe())

    # 4. 准备全历史（或指定区间）的打分数据
    scored_dict = prepare_scored_universe(data_dict)
    dates_bt = get_backtest_dates(scored_dict, START_DATE, END_DATE)
    if len(dates_bt) < 2:
        print("❌ 区间内共同有效日期不足，无法回测组合")
        return

    print(f"\n组合回测日期范围: {dates_bt[0].date()} ~ {dates_bt[-1].date()} "
          f"(共 {len(dates_bt)} 个交易日)")

    # 5. TopN 组合回测
    print(f"\n开始前 Top{TOP_N} 组合回测...")
    eq_port = backtest_topN_portfolio(
        data_dict=data_dict,
        scored_dict=scored_dict,
        top_n=TOP_N,
        initial_cash=100000,
        fee_rate=0.0005,
        slippage=0.0005,
        dates=dates_bt,
    )

    print("\nTopN 组合回测结果（最后10天）：")
    print(eq_port[["cash", "market_value", "equity", "day_fee", "holding_codes"]].tail(10))

    total_return = eq_port["equity"].iloc[-1] / eq_port["equity"].iloc[0] - 1
    cummax = eq_port["equity"].cummax()
    drawdown = eq_port["equity"] / cummax - 1
    max_dd = drawdown.min()
    print(f"\nTop{TOP_N} 组合总收益: {total_return:.2%}")
    print(f"Top{TOP_N} 组合最大回撤: {max_dd:.2%}")

    fee_engine = getattr(eq_port, "_fee_engine", None)
    if fee_engine is not None:
        fee_summary = fee_engine.summary()
        print("\nTopN 组合费用汇总：")
        print(f"  交易佣金总额: {fee_summary['total_trade_fee']:.2f}")
        print(f"  印花税总额:   {fee_summary['total_stamp_duty']:.2f}")
        print(f"  融资利息总额: {fee_summary['total_financing_fee']:.2f}")
        print(f"  费用合计:     {fee_summary['total_fee']:.2f}")

    # 6. Buy & Hold 组合回测（等权持有全股票池）
    print("\n开始 Buy & Hold 组合回测（等权持有全股票池）...")
    eq_bh = backtest_buy_and_hold_universe(
        data_dict=data_dict,
        dates=dates_bt,
        initial_cash=100000,
        fee_rate=0.0005,
        slippage=0.0005,
    )

    print("\nBuy & Hold 回测结果（最后10天）：")
    print(eq_bh[["cash", "market_value", "equity", "day_fee"]].tail(10))

    bh_total_return = eq_bh["equity"].iloc[-1] / eq_bh["equity"].iloc[0] - 1
    bh_cummax = eq_bh["equity"].cummax()
    bh_drawdown = eq_bh["equity"] / bh_cummax - 1
    bh_max_dd = bh_drawdown.min()
    print(f"\nBuy & Hold 总收益: {bh_total_return:.2%}")
    print(f"Buy & Hold 最大回撤: {bh_max_dd:.2%}")

    fee_engine_bh = getattr(eq_bh, "_fee_engine", None)
    if fee_engine_bh is not None:
        fee_summary_bh = fee_engine_bh.summary()
        print("\nBuy & Hold 费用汇总：")
        print(f"  交易佣金总额: {fee_summary_bh['total_trade_fee']:.2f}")
        print(f"  印花税总额:   {fee_summary_bh['total_stamp_duty']:.2f}")
        print(f"  融资利息总额: {fee_summary_bh['total_financing_fee']:.2f}")
        print(f"  费用合计:     {fee_summary_bh['total_fee']:.2f}")


if __name__ == "__main__":
    main()
