# practice_single_stock.py
import pandas as pd
import numpy as np

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
                     buy_thresh: float = 2.5,
                     sell_thresh: float = 1.0) -> pd.DataFrame:
    """
    T+1 执行版：
    - 第T天用 total_score 生成 raw_signal_T（1=想持有, 0=想空仓）
    - 真正的 position_T = raw_signal_{T-1}（上一天的信号）
    这样就不会用到未来数据。
    """
    df = df_scored.copy()

    df['raw_signal'] = 0
    df.loc[df['total_score'] >= buy_thresh, 'raw_signal'] = 1
    df.loc[df['total_score'] <= sell_thresh, 'raw_signal'] = 0

    # 信号整体往后移一天：今天的仓位根据昨天的 raw_signal 决定
    df['position'] = df['raw_signal'].shift(1).fillna(0).astype(int)

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

def simple_backtest(df_sig: pd.DataFrame,
                    initial_cash: float = 100000,
                    fee_rate: float = 0.0005,
                    slippage: float = 0.0005) -> pd.DataFrame:
    """
    非严格版小回测：
    - 当天 position 决定当天开盘是否买/卖（有未来函数风险，先练手）
    - 买入：用所有现金，以开盘价*(1+滑点)买
    - 卖出：全部卖出，以开盘价*(1-滑点)卖
    - 每天收盘，用收盘价计算市值
    """
    df = df_sig.copy().reset_index()
    cash = initial_cash
    shares = 0.0
    records = []

    prev_pos = 0
    for i, row in df.iterrows():
        date = row['date']
        price_open = row['open']
        price_close = row['close']
        pos = int(row['position'])

        # 仓位变化
        if prev_pos == 0 and pos == 1 and shares == 0:
            # 全仓买入
            buy_amount = cash
            exec_price = price_open * (1 + slippage)
            shares = buy_amount / exec_price
            fee = buy_amount * fee_rate
            cash -= buy_amount + fee

        elif prev_pos == 1 and pos == 0 and shares > 0:
            # 全部卖出
            exec_price = price_open * (1 - slippage)
            sell_amount = shares * exec_price
            fee = sell_amount * fee_rate
            cash += sell_amount - fee
            shares = 0.0

        market_value = shares * price_close
        equity = cash + market_value

        records.append({
            'date': date,
            'cash': cash,
            'shares': shares,
            'market_value': market_value,
            'equity': equity,
            'position': pos
        })

        prev_pos = pos

    eq = pd.DataFrame(records).set_index('date')
    eq['ret'] = eq['equity'].pct_change().fillna(0)
    return eq


def main():
    # ===== 1. 加载数据 =====
    # 方案A：用你自己的数据API（比如东方财富封装）
    # from data_api import DataAPI
    # api = DataAPI()
    # df = api.get_kline('600519', period='D', limit=600)

    # 方案B：先用本地CSV练手
    # 你可以先随便导出一只股票的数据成csv再练
    csv_path = r"/Users/benwan/Downloads/PyProjects/stocks/stocks/159892_D_qfq.csv"  # TODO: 改成你自己的路径
    df = load_data_from_csv(csv_path)

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

    # ===== 4. 生成信号 =====
    df_sig = generate_signals(df_scored, buy_thresh=2.5, sell_thresh=1.0)
    print("\n最近20天信号：")
    print(df_sig[['close', 'total_score', 'position']].tail(20))

    # ===== 5. 简单回测 =====
    eq = simple_backtest(df_sig)
    print("\n回测结果（最后10天）：")
    print(eq.tail(10))

    # ===== 6. Buy & Hold 对照 =====
    bh = backtest_buy_and_hold(df)
    bh_total_return = bh['equity'].iloc[-1] / bh['equity'].iloc[0] - 1
    print(f"\nBuy & Hold 总收益: {bh_total_return:.2%}")

    # 简单输出总收益和最大回撤
    total_return = eq['equity'].iloc[-1] / eq['equity'].iloc[0] - 1
    cummax = eq['equity'].cummax()
    drawdown = eq['equity'] / cummax - 1
    max_dd = drawdown.min()

    print(f"\n总收益: {total_return:.2%}")
    print(f"最大回撤: {max_dd:.2%}")


if __name__ == "__main__":
    main()