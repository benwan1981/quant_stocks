# factors.py
import pandas as pd
import numpy as np


def compute_stock_factors(df: pd.DataFrame, min_periods: int = 60) -> pd.DataFrame:
    """
    对单只股票的历史K线计算基础因子。
    
    参数
    ----
    df : DataFrame
        必须包含列: ['open','high','low','close','volume']，index 为日期(升序)
    min_periods : int
        至少需要多少条历史数据才认为因子可靠（前面不足的行打 NaN）

    返回
    ----
    df_f : DataFrame
        在原 df 的基础上新增一系列因子列，不做打分。
    """
    df = df.copy().sort_index()

    close = df['close']
    vol   = df['volume']

    # ---------- 趋势 / 均线 ----------
    df['ma5']  = close.rolling(5,  min_periods=1).mean()
    df['ma20'] = close.rolling(20, min_periods=1).mean()
    df['ma60'] = close.rolling(60, min_periods=1).mean()

    # 价格相对均线的位置（百分比）
    df['close_ma20_pct'] = (close / df['ma20'] - 1.0)
    df['close_ma60_pct'] = (close / df['ma60'] - 1.0)

    # ---------- 动量 ----------
    df['mom5']  = close.pct_change(5)   # 5日涨幅
    df['mom20'] = close.pct_change(20)  # 20日涨幅

    # ---------- 量能 ----------
    df['vol5']  = vol.rolling(5,  min_periods=1).mean()
    df['vol20'] = vol.rolling(20, min_periods=1).mean()

    # 相对放量比例: 今日量 / 20日均量
    df['vol_ratio_20'] = df['volume'] / (df['vol20'] + 1e-9)

    # ---------- 风险：波动 & 回撤 ----------
    # 20日收益率标准差（波动率）
    df['ret1'] = close.pct_change(1)
    df['volatility_20d'] = df['ret1'].rolling(20, min_periods=5).std()

    # 60日最大回撤（滚动）
    df['max_drawdown_60d'] = _rolling_max_drawdown(close, window=60)

    # ---------- MACD ----------
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # ---------- RSI14 ----------
    df['rsi14'] = _rsi(close, period=14)

    # 对前期数据不足的行统一打 NaN，避免乱用
    df.loc[df.index[:min_periods-1], ['ma60','mom20','vol20',
                                      'volatility_20d','max_drawdown_60d',
                                      'macd','macd_signal','macd_hist','rsi14']] = np.nan

    return df


def score_stock_row(row: pd.Series) -> dict:
    """
    对单行数据（某一天的因子）打分。
    注意：这里只负责把“因子数值 -> 分数”，不决定买卖。
    
    返回一个 dict，包含各子因子分数和 total_score。
    """
    # 若关键因子缺失，返回 NaN 分数
    if pd.isna(row.get('ma20')) or pd.isna(row.get('ma60')):
        return {
            "trend_score":    np.nan,
            "momentum_score": np.nan,
            "volume_score":   np.nan,
            "risk_score":     np.nan,
            "technical_score":np.nan,
            "total_score":    np.nan,
        }

    # ===== 1. 趋势分 trend_score =====
    # 基本逻辑：close > ma20 > ma60 加基础分，离均线越远，适度加分/减分
    trend_score = 0.0
    if row['close'] > row['ma20'] > row['ma60']:
        trend_score += 1.0
    elif row['close'] > row['ma20']:
        trend_score += 0.5

    # close 相对 ma20 的偏离（过度偏离反而扣分）
    x = row['close_ma20_pct']  # 例如 0.1 = 高于20日线10%
    if not pd.isna(x):
        if -0.05 <= x <= 0.15:
            trend_score += 0.5   # 在合理偏离区间内
        elif x < -0.05:
            trend_score -= 0.5   # 跌太多
        elif x > 0.25:
            trend_score -= 0.5   # 乖离过大，可能过热

    # ===== 2. 动量分 momentum_score =====
    momentum_score = 0.0
    mom20 = row.get('mom20', np.nan)
    if not pd.isna(mom20):
        if 0.10 <= mom20 <= 0.60:
            momentum_score += 1.0
        elif 0.05 <= mom20 < 0.10:
            momentum_score += 0.5
        elif mom20 > 0.60:
            momentum_score -= 0.5  # 涨得太猛，短期有回吐风险

    mom5 = row.get('mom5', np.nan)
    if not pd.isna(mom5):
        if mom5 > 0:
            momentum_score += 0.5
        elif mom5 < -0.05:
            momentum_score -= 0.5

    # ===== 3. 量能分 volume_score =====
    volume_score = 0.0
    vr = row.get('vol_ratio_20', np.nan)
    if not pd.isna(vr):
        if 1.2 <= vr <= 3.0:
            volume_score += 1.0   # 温和放量
        elif 0.7 <= vr < 1.2:
            volume_score += 0.3   # 正常量
        elif vr > 5.0:
            volume_score -= 0.5   # 极端放量，有可能是出货
        elif vr < 0.5:
            volume_score -= 0.3   # 极度缩量，关注度差

    # ===== 4. 风险分 risk_score（分越高表示越“安全”） =====
    risk_score = 0.0
    vol20 = row.get('volatility_20d', np.nan)
    mdd60 = row.get('max_drawdown_60d', np.nan)

    # 波动率越低越好
    if not pd.isna(vol20):
        if vol20 < 0.02:
            risk_score += 1.0
        elif vol20 < 0.04:
            risk_score += 0.5
        elif vol20 > 0.08:
            risk_score -= 0.5

    # 近60日最大回撤越小越好（注意 mdd60 通常为负数）
    if not pd.isna(mdd60):
        if mdd60 > -0.10:
            risk_score += 1.0
        elif mdd60 > -0.20:
            risk_score += 0.5
        elif mdd60 < -0.30:
            risk_score -= 0.5

    # ===== 5. 技术形态附加分（MACD & RSI） =====
    technical_score = 0.0

    macd_hist = row.get('macd_hist', np.nan)
    if not pd.isna(macd_hist):
        if macd_hist > 0:
            technical_score += 0.5
        else:
            technical_score -= 0.3

    rsi = row.get('rsi14', np.nan)
    if not pd.isna(rsi):
        if 40 <= rsi <= 70:
            technical_score += 0.5
        elif rsi > 75:
            technical_score -= 0.5
        elif rsi < 25:
            # 超卖可以给一点弹性加分，但不要太高
            technical_score += 0.2

    # ===== 6. 综合得分 total_score =====
    # 这里的权重你后面可以调参
    total_score = (
        1.2 * trend_score +
        1.0 * momentum_score +
        0.8 * volume_score +
        1.0 * risk_score +
        0.8 * technical_score
    )

    return {
        "trend_score":     trend_score,
        "momentum_score":  momentum_score,
        "volume_score":    volume_score,
        "risk_score":      risk_score,
        "technical_score": technical_score,
        "total_score":     total_score,
    }


def attach_scores(df_factors: pd.DataFrame) -> pd.DataFrame:
    """
    在已经有因子列的 DataFrame 上，逐行计算得分并附加到 DataFrame。
    返回包含各子分数及 total_score 的 df。
    """
    df = df_factors.copy()
    scores = df.apply(score_stock_row, axis=1, result_type='expand')
    # scores 是一个 DataFrame，列名就是 score_stock_row 返回的 key
    for col in scores.columns:
        df[col] = scores[col]
    return df


# ========== 辅助函数 ==========

def _rolling_max_drawdown(close: pd.Series, window: int = 60) -> pd.Series:
    """
    计算滚动窗口内最大回撤（负数），例如 -0.25 表示最大回撤 25%
    """
    # 使用 expanding + rolling 的方式
    # 对于每个窗口： (最低价/最高价 - 1)
    roll_max = close.rolling(window, min_periods=1).max()
    roll_min = close.rolling(window, min_periods=1).min()
    mdd = roll_min / roll_max - 1.0
    return mdd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    经典 RSI 计算，返回 0-100 区间。
    """
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()

    rs = roll_up / (roll_down + 1e-9)
    rsi = 100 - 100 / (1 + rs)
    return rsi