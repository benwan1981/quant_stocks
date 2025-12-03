# fees.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict
import pandas as pd


@dataclass
class FeeConfig:
    """费用配置（后面可以继续扩展）"""
    trade_fee_rate: float = 0.0005   # 佣金：双边
    stamp_duty_rate: float = 0.001   # 印花税：通常只对卖出
    financing_rate_year: float = 0.06  # 年化融资利率（以后用）
    days_in_year: int = 252          # 用于融资日费率换算


@dataclass
class FeeRecord:
    """记录每天产生的各项费用"""
    date: pd.Timestamp
    buy_amount: float = 0.0
    sell_amount: float = 0.0
    trade_fee: float = 0.0
    stamp_duty: float = 0.0
    financing_fee: float = 0.0


class FeeEngine:
    """
    统一管理交易费用、印花税、融资费用的模块。
    目前主要用交易费+印花税，融资费预留接口。
    """
    def __init__(self, config: FeeConfig | None = None):
        self.config = config or FeeConfig()
        self.records: List[FeeRecord] = []

    # --- 核心计算函数 ---

    def calc_trade_fee(self, amount: float) -> float:
        """按成交金额计算佣金（单边）"""
        return amount * self.config.trade_fee_rate

    def calc_stamp_duty(self, sell_amount: float) -> float:
        """按卖出金额计算印花税（可以买入不收、卖出收）"""
        return sell_amount * self.config.stamp_duty_rate if sell_amount > 0 else 0.0

    def calc_financing_fee(self, margin_balance: float, days: int = 1) -> float:
        """
        融资利息：按资金占用计算。
        目前你还没用到，可以先传 0，将来加融资时直接复用。
        """
        if margin_balance <= 0:
            return 0.0
        daily_rate = self.config.financing_rate_year / self.config.days_in_year
        return margin_balance * daily_rate * days

    def on_day(self,
               date,
               buy_amount: float = 0.0,
               sell_amount: float = 0.0,
               margin_balance: float = 0.0,
               days: int = 1) -> float:
        """
        在回测主循环中，每天调用一次：
        - 输入当日买入金额、卖出金额、融资余额
        - 返回当日总费用，并记录到明细
        """
        trade_fee = 0.0
        if buy_amount > 0:
            trade_fee += self.calc_trade_fee(buy_amount)
        if sell_amount > 0:
            trade_fee += self.calc_trade_fee(sell_amount)

        stamp = self.calc_stamp_duty(sell_amount)
        financing = self.calc_financing_fee(margin_balance, days=days)

        rec = FeeRecord(
            date=pd.to_datetime(date),
            buy_amount=buy_amount,
            sell_amount=sell_amount,
            trade_fee=trade_fee,
            stamp_duty=stamp,
            financing_fee=financing,
        )
        self.records.append(rec)

        return trade_fee + stamp + financing

    # --- 汇总输出 ---

    def summary(self) -> Dict[str, float]:
        """返回总费用汇总，用于打印"""
        if not self.records:
            return {
                "total_trade_fee": 0.0,
                "total_stamp_duty": 0.0,
                "total_financing_fee": 0.0,
                "total_fee": 0.0,
            }

        df = pd.DataFrame([r.__dict__ for r in self.records])
        total_trade = float(df["trade_fee"].sum())
        total_stamp = float(df["stamp_duty"].sum())
        total_fin = float(df["financing_fee"].sum())
        return {
            "total_trade_fee": total_trade,
            "total_stamp_duty": total_stamp,
            "total_financing_fee": total_fin,
            "total_fee": total_trade + total_stamp + total_fin,
        }
