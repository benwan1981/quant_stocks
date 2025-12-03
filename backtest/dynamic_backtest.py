"""
动态因子回测脚本（单文件版）
--------------------------------
功能：
1. 读取所有股票 CSV（日线）
2. 构造公共交易日 price_panel, ret_panel
3. 用沪深300计算牛熊状态 & 强反弹(rally_flag)
4. compute_factors_dyn：长短动量 + 波动 + beta
5. run_dynamic_for_year：按年回测动态因子策略

约定：
- CSV 文件目录：DATA_DIR
- 个股文件名形如：000001_平安银行_D_qfq_gm.csv
- 沪深300文件：000300_沪深300_D_gm.csv
"""

import os
from pathlib import Path
from math import floor
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

# ==================== 参数区 ====================

DATA_DIR = "./data/gm_equity/"  # TODO: 改成你自己的 CSV 文件目录

INITIAL_CAPITAL = 100000.0

# 手续费：万2，单笔最低 5 元；卖出另收千一印花税
COMM_RATE = 0.0002
STAMP_DUTY = 0.001

def calc_commission(gross: float) -> float:
    """券商佣金：万2，单笔最低 5 元"""
    return max(5.0, gross * COMM_RATE)


def calc_buy_cost(price: float, shares: int) -> float:
    """买入总成本（含佣金，不含印花税）"""
    gross = price * shares
    comm = calc_commission(gross)
    return gross + comm


def calc_sell_proceeds(price: float, shares: int) -> float:
    """卖出净收入（扣掉佣金+印花税之后到手的钱）"""
    gross = price * shares
    comm = calc_commission(gross)
    tax = gross * STAMP_DUTY
    return gross - comm - tax
# 动量 & 波动窗口
LOOKBACK_MOM_LONG = 60
LOOKBACK_MOM_SHORT = 10
LOOKBACK_VOL = 60
LOOKBACK_BETA = 120


# ==================== 工具函数 ====================

def load_price(file_path: Path) -> pd.DataFrame | None:
    """
    从单个 CSV 读取日线数据，返回仅包含 dt, close 的 DataFrame
    """
    df = pd.read_csv(file_path)
    # 自动找日期列
    col_dt = [c for c in df.columns if "date" in c.lower() or "eob" in c.lower()]
    if not col_dt:
        return None
    dt_col = col_dt[0]
    df["dt"] = pd.to_datetime(df[dt_col]).dt.tz_localize(None)
    df = df[["dt", "close"]].sort_values("dt")
    return df


# ==================== 1. 读取所有股票 CSV ====================

def load_all_stocks(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    扫描目录中的 *_D_qfq_gm.csv 个股文件，返回 {代码: df(dt, close)}
    """
    base = Path(data_dir)
    files = sorted(os.listdir(base))
    stock_files = [f for f in files if f.endswith("_D_qfq_gm.csv") and not f.startswith("159")]  # 排除ETF如需可放开

    price_dict: Dict[str, pd.DataFrame] = {}
    for f in stock_files:
        code = f.split("_")[0]
        df = load_price(base / f)
        if df is not None:
            price_dict[code] = df

    return price_dict


# ==================== 2. 构造公共交易日 price_panel / ret_panel ====================

def build_price_ret_panel(price_dict: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    根据所有个股 df，构造：
    - price_panel: index=公共交易日, columns=股票代码, 值=收盘价
    - ret_panel:   index=公共交易日, columns=股票代码, 值=日收益率
    这里用的是“所有股票的日期交集”，简单干净；如你想用并集可自行改动。
    """
    # 取所有股票日期交集
    all_dates_set = None
    for df in price_dict.values():
        s = set(df["dt"])
        if all_dates_set is None:
            all_dates_set = s
        else:
            all_dates_set = all_dates_set & s
    if not all_dates_set:
        raise ValueError("没有公共交易日，检查数据是否重叠过少。")

    all_dates = sorted(all_dates_set)
    all_dates = pd.Series(all_dates)

    price_panel = pd.DataFrame(index=all_dates)
    for code, df in price_dict.items():
        price_panel[code] = df.set_index("dt").reindex(all_dates)["close"]

    ret_panel = price_panel.pct_change().dropna()
    return price_panel, ret_panel


# ==================== 3. HS300 牛熊 & rally ====================

def load_hs300(data_dir: str, all_dates: pd.Index) -> pd.DataFrame:
    """
    加载 000300_沪深300_D_gm.csv，返回对齐 all_dates 的 index_df(close, volume)
    """
    base = Path(data_dir)
    hs300_file = base / "000300_沪深300_D_gm.csv"
    if not hs300_file.exists():
        raise FileNotFoundError(f"找不到沪深300文件: {hs300_file}")

    raw = pd.read_csv(hs300_file)
    col_dt = [c for c in raw.columns if "date" in c.lower() or "eob" in c.lower()]
    if not col_dt:
        raise ValueError("沪深300文件中找不到日期列")
    dt_col = col_dt[0]
    raw["dt"] = pd.to_datetime(raw[dt_col]).dt.tz_localize(None)

    # 找成交量列（可能是 volume/vol/成交量/amount/成交额 任一）
    vol_col = None
    for c in raw.columns:
        if "vol" in c.lower() or "量" in c:
            vol_col = c
            break
    if vol_col is None:
        for c in raw.columns:
            if "amount" in c.lower() or "额" in c:
                vol_col = c
                break
    if vol_col is None:
        # 没有量，就填NaN
        df = raw[["dt", "close"]].copy()
        df["volume"] = np.nan
    else:
        df = raw[["dt", "close", vol_col]].copy()
        df.rename(columns={vol_col: "volume"}, inplace=True)

    index_df = df.set_index("dt").reindex(all_dates).ffill()
    return index_df


def compute_market_regime(index_df: pd.DataFrame) -> pd.Series:
    """
    用 MA20 / MA100 + VOL20 / VOL60 判定 bull / bear / neutral，并整体 shift(1) 做到 T+1
    """
    df = index_df.copy()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma100"] = df["close"].rolling(100).mean()
    df["vol20"] = df["volume"].rolling(20).mean()
    df["vol60"] = df["volume"].rolling(60).mean()

    def _judge(row):
        if pd.isna(row["ma20"]) or pd.isna(row["ma100"]):
            return "neutral"
        trend_up = row["ma20"] > row["ma100"]
        vol_up = False
        if not pd.isna(row["vol20"]) and not pd.isna(row["vol60"]) and row["vol60"] > 0:
            vol_up = row["vol20"] > row["vol60"]
        if trend_up and vol_up:
            return "bull"
        elif (not trend_up) and (not vol_up):
            return "bear"
        else:
            return "neutral"

    regime = df.apply(_judge, axis=1)
    return regime.shift(1)  # 全体后移一天，避免未来函数


def compute_rally_flag(index_df: pd.DataFrame) -> pd.Series:
    """
    基于前一日涨幅 + 3日涨幅 + 放量，标记“强反弹/爆发行情”。
    返回的是 Series[bool]，同样不使用未来信息。
    """
    df = index_df.copy()
    ret = df["close"].pct_change()

    # 前一日涨幅
    ret_shifted = ret.shift(1)
    # 近3日累计（使用已shift的）
    short_ret3 = ret_shifted.rolling(3).sum()
    # 放量比
    vol60 = df["volume"].rolling(60).mean()
    vol_ratio = df["volume"] / vol60  # 当日/60日

    cond1 = ret_shifted > 0.02          # 前一日涨幅 > 2%
    cond2 = short_ret3 > 0.05           # 前3日累计涨幅 > 5%
    cond3 = vol_ratio.shift(1) > 1.5    # 前一日成交量 > 60日1.5倍

    rally = cond1 | cond2 | cond3
    return rally  # 这里已经只用到 t-1 及以前的信息


# ==================== 4. 动态因子 compute_factors_dyn ====================

def compute_factors_dyn(
    ret_panel: pd.DataFrame,
    cut_date: pd.Timestamp,
    lookback_mom_long: int = LOOKBACK_MOM_LONG,
    lookback_mom_short: int = LOOKBACK_MOM_SHORT,
    lookback_vol: int = LOOKBACK_VOL,
    lookback_beta: int = LOOKBACK_BETA,
) -> pd.DataFrame | None:
    """
    截止 cut_date（不含当日）的因子矩阵：
    - mom_long: 最近60日累计收益
    - mom_short: 最近10日累计收益
    - vol: 最近60日波动率（年化）
    - beta: 相对“等权市场组合”的120日beta
    并做 0–1 排名：mom_long_r, mom_short_r, vol_r, beta_r
    """
    hist = ret_panel[ret_panel.index < cut_date].tail(lookback_beta)
    if len(hist) < lookback_beta // 2:
        return None

    mkt = hist.mean(axis=1)  # 简单等权收益序列

    rows = []
    for code in hist.columns:
        s = hist[code].dropna()
        if len(s) < lookback_mom_long:
            continue
        mom_long = (1 + s.tail(lookback_mom_long)).prod() - 1
        mom_short = (1 + s.tail(lookback_mom_short)).prod() - 1 if len(s) >= lookback_mom_short else np.nan
        vol = s.tail(lookback_vol).std() * np.sqrt(252)
        beta = s.cov(mkt) / mkt.var() if mkt.var() != 0 else 0.0
        rows.append((code, mom_long, mom_short, vol, beta))

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["code", "mom_long", "mom_short", "vol", "beta"]).set_index("code")
    df["mom_long_r"] = df["mom_long"].rank(pct=True)
    df["mom_short_r"] = df["mom_short"].rank(pct=True)
    df["beta_r"] = df["beta"].rank(pct=True)
    df["vol_r"] = df["vol"].rank(pct=True)
    return df


# ==================== 5. 单年回测 run_dynamic_for_year ====================

def decide_mode(base_regime: str, rally: bool) -> str:
    """
    根据 base_regime + rally 决定当日模式：
    - risk_on  ： bull 或者 rally=True
    - risk_off ： bear 且 rally=False
    - neutral  ： 其他
    """
    if base_regime == "bull" or rally:
        return "risk_on"
    elif base_regime == "bear":
        return "risk_off"
    else:
        return "neutral"


def run_dynamic_for_year(
    year: int,
    price_panel: pd.DataFrame,
    ret_panel: pd.DataFrame,
    market_regime: pd.Series,
    rally_flag: pd.Series,
    initial_capital: float = INITIAL_CAPITAL,
) -> Tuple[float, float, int, pd.DataFrame]:
    """
    按“自然年”跑一遍动态因子策略：
    - 每年年初资金重置为 initial_capital
    - 年内按 day by day 进行 T+1 调仓
    - 年末清仓，返回 final_capital, 年度收益率, 交易笔数, 交易明细 DataFrame
    """
    # 当年全部交易日
    year_dates = price_panel.index[price_panel.index.year == year]
    if len(year_dates) < 2:
        return np.nan, np.nan, 0, pd.DataFrame()

    capital = initial_capital
    portfolio: Dict[str, int] = {}
    trade_log: List[tuple] = []

    # 从第二个交易日开始（T+1）
    for dt in year_dates[1:]:
        base_reg = market_regime.loc[dt]
        rally = bool(rally_flag.loc[dt])
        mode = decide_mode(base_reg, rally)

        factors = compute_factors_dyn(ret_panel, cut_date=dt)
        if factors is None:
            continue

        prices_today = price_panel.loc[dt]

        # === 不同模式下的打分 & 参数 ===
        if mode == "risk_on":
            score = (
                0.8 * factors["mom_short_r"]
                + 0.2 * factors["mom_long_r"]
                # 你可以加一点 beta_r 权重，例如 + 0.1 * factors["beta_r"]
            )
            buy_th = 0.6
            sell_th = 0.25
            target_exposure = 0.95
            max_positions = 12

        elif mode == "neutral":
            score = (
                0.6 * factors["mom_long_r"]
                - 0.4 * factors["vol_r"]
            )
            buy_th = 0.55
            sell_th = 0.3
            target_exposure = 0.6
            max_positions = 8

        else:  # risk_off
            score = -factors["vol_r"]
            buy_th = 0.0        # 不再新开仓
            sell_th = -0.2      # 低于这个就卖（波动最高的一批）
            target_exposure = 0.2
            max_positions = 4

        factors = factors.assign(score=score)

        # === 先卖 ===
        sell_codes = []
        for code in list(portfolio.keys()):
            if code not in factors.index or factors.loc[code, "score"] < sell_th:
                price = float(prices_today.get(code, np.nan))
                if not np.isfinite(price):
                    continue
                shares = portfolio[code]
                proceeds = calc_sell_proceeds(price, shares)
                capital += proceeds
                trade_log.append((
                    dt.date(), "SELL", mode, base_reg, rally,
                    code, shares, price, capital
                ))
                sell_codes.append(code)
        for c in sell_codes:
            del portfolio[c]

        # === risk_off 模式不新开仓 ===
        if mode == "risk_off":
            continue

        # === 选买入候选 ===
        buy_candidates = factors[factors["score"] > buy_th].sort_values("score", ascending=False)
        if buy_candidates.empty:
            continue

        # 当前市值 & 总资产
        portfolio_value = 0.0
        for code, sh in portfolio.items():
            p = float(prices_today.get(code, np.nan))
            if np.isfinite(p):
                portfolio_value += p * sh
        total_equity = capital + portfolio_value
        if total_equity <= 0:
            continue

        target_value = target_exposure * total_equity
        additional_value = max(0.0, target_value - portfolio_value)
        if additional_value <= capital * 0.05:
            # 增仓太少就不要折腾
            continue

        slots_left = max_positions - len(portfolio)
        if slots_left <= 0:
            continue

        to_buy = buy_candidates.index.difference(portfolio.keys())[:slots_left]
        if len(to_buy) == 0:
            continue

        per_cap = additional_value / len(to_buy)

        # === 按 100 股一手买入 ===
        for code in to_buy:
            price = float(prices_today.get(code, np.nan))
            if not np.isfinite(price):
                continue

            raw_shares = per_cap / price
            lots = floor(raw_shares / 100)
            shares = lots * 100
            if shares < 100:
                continue

            gross = price * shares
            buy_cost = gross + calc_commission(gross)
            if buy_cost > capital:
                continue

            capital -= buy_cost
            portfolio[code] = portfolio.get(code, 0) + shares
            trade_log.append((
                dt.date(), "BUY", mode, base_reg, rally,
                code, shares, price, capital
            ))

    # === 年末清仓 ===
    last_dt = year_dates[-1]
    last_prices = price_panel.loc[last_dt]
    for code, shares in list(portfolio.items()):
        price = float(last_prices.get(code, np.nan))
        if not np.isfinite(price):
            continue
        proceeds = calc_sell_proceeds(price, shares)
        capital += proceeds
        trade_log.append((
            last_dt.date(), "SELL_END", "final", market_regime.loc[last_dt],
            bool(rally_flag.loc[last_dt]), code, shares, price, capital
        ))
        del portfolio[code]

    final_capital = capital
    total_return = final_capital / initial_capital - 1.0

    trade_df = pd.DataFrame(
        trade_log,
        columns=["Date", "Action", "Mode", "BaseRegime", "Rally",
                 "Stock", "Shares", "Price", "Remain_Capital"]
    )
    return final_capital, total_return, len(trade_df), trade_df


# ==================== 主程序示例 ====================

if __name__ == "__main__":
    # 1. 读取股票数据
    price_dict = load_all_stocks(DATA_DIR)

    # 2. 构造 price_panel, ret_panel
    price_panel, ret_panel = build_price_ret_panel(price_dict)
    print("price_panel 形状:", price_panel.shape)
    print("ret_panel 日期区间:", ret_panel.index.min(), "→", ret_panel.index.max())

    # 3. 加载沪深300 & 计算牛熊 + rally
    index_df = load_hs300(DATA_DIR, price_panel.index)
    market_regime = compute_market_regime(index_df)
    rally_flag = compute_rally_flag(index_df)

    # 4. 按年回测
    results = {}
    for yr in sorted(set(price_panel.index.year)):
        cap, r, n_trades, _trades = run_dynamic_for_year(
            yr, price_panel, ret_panel, market_regime, rally_flag
        )
        results[yr] = (cap, r, n_trades)
        print(f"年份 {yr}: 期末资金 {cap:,.2f}，收益率 {r*100:.2f}%，交易笔数 {n_trades}")