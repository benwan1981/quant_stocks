# factors/stock_factors.py
import numpy as np
import pandas as pd

def compute_stock_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入: 原始日线数据 DataFrame，index 为 date，列至少包含:
        ['open', 'high', 'low', 'close', 'volume']

    输出: 在原基础上增加:
        ma20, mom20, vol_ratio_20, rsi14
    """
    df = df.copy().sort_index()

    close = df["close"].astype(float)
    vol = df["volume"].astype(float)

    # 20 日均线
    df["ma20"] = close.rolling(20, min_periods=1).mean()

    # 20 日动量: 近 20 日涨跌幅
    df["mom20"] = close.pct_change(20)

    # 20 日量能放大倍数: 今日量 / 20 日均量
    vol_mean_20 = vol.rolling(20, min_periods=1).mean()
    df["vol_ratio_20"] = vol / vol_mean_20.replace(0, np.nan)

    # RSI14（Wilder 算法简化版）
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.ewm(alpha=1 / 14, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / 14, adjust=False).mean()

    rs = roll_up / (roll_down + 1e-9)
    df["rsi14"] = 100 - 100 / (1 + rs)

    return df


def attach_scores(df_fac: pd.DataFrame) -> pd.DataFrame:
    """
    在因子基础上打分，输出:
      trend_score, momentum_score, volume_score, risk_score, technical_score, total_score
    这套规则和你之前示例的风格一致，但可以后续再逐步微调。
    """
    df = df_fac.copy()

    close = df["close"]
    ma20 = df["ma20"]
    mom20 = df["mom20"]
    vol_ratio = df["vol_ratio_20"]
    rsi = df["rsi14"]

    # —— 趋势分数: 看价格相对 MA20 的位置 —— 
    trend = []
    for c, m in zip(close, ma20):
        if pd.isna(m):
            trend.append(np.nan)
            continue
        ratio = c / m
        if ratio >= 1.05:
            trend.append(1.5)
        elif ratio >= 1.02:
            trend.append(1.0)
        elif ratio >= 0.99:
            trend.append(0.5)
        elif ratio >= 0.95:
            trend.append(0.0)
        else:
            trend.append(-0.5)
    df["trend_score"] = trend

    # —— 动量分数: 看 20 日涨跌 —— 
    mom_score = []
    for m in mom20:
        if pd.isna(m):
            mom_score.append(np.nan)
            continue
        if m >= 0.20:
            mom_score.append(1.5)
        elif m >= 0.10:
            mom_score.append(1.0)
        elif m >= 0.03:
            mom_score.append(0.5)
        elif m >= -0.05:
            mom_score.append(0.0)
        else:
            mom_score.append(-0.5)
    df["momentum_score"] = mom_score

    # —— 量能分数: 看量比 —— 
    vol_score = []
    for vr in vol_ratio:
        if pd.isna(vr):
            vol_score.append(np.nan)
            continue
        if vr >= 2.0:
            vol_score.append(1.0)
        elif vr >= 1.5:
            vol_score.append(0.5)
        elif vr >= 0.8:
            vol_score.append(0.0)
        elif vr >= 0.5:
            vol_score.append(-0.3)
        else:
            vol_score.append(-0.5)
    df["volume_score"] = vol_score

    # —— 风险分数: 用 RSI 控制风险 —— 
    risk_score = []
    for r in rsi:
        if pd.isna(r):
            risk_score.append(np.nan)
            continue
        score = 1.5
        if r > 75:
            score -= 1.0   # 超买减分
        elif r < 30:
            score += 0.5   # 超卖适度加一点
        risk_score.append(score)
    df["risk_score"] = risk_score

    # —— 技术分数: 综合趋势 + RSI 的简单额外加权 —— 
    tech_score = []
    for t, r in zip(df["trend_score"], rsi):
        if pd.isna(t) or pd.isna(r):
            tech_score.append(np.nan)
            continue
        extra = 0.0
        if 40 <= r <= 60:
            extra = 1.0
        elif 30 <= r < 40 or 60 < r <= 70:
            extra = 0.5
        elif r > 80 or r < 20:
            extra = -0.3
        tech_score.append(extra)
    df["technical_score"] = tech_score

    # —— 总分 —— 
    df["total_score"] = (
        df["trend_score"]
        + df["momentum_score"]
        + df["volume_score"]
        + df["risk_score"]
        + df["technical_score"]
    )

    return df
