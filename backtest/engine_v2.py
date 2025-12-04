# -*- coding: utf-8 -*-
"""
BacktestEngineV2：
- 使用 v2 因子流水线（残差动量 + vol）
- 基于市场波动 regime 动态设置 buy_th / sell_th / base_target / max_positions
- 使用 soft drawdown gate 缩放 target_exposure
- 生成每日目标权重矩阵 -> 交给 execution_v2 模拟 T+1 回测
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal

import numpy as np
import pandas as pd

from factors.core_factors_v2 import (
    FactorWindows,
    compute_stock_factor_panel,
    build_composite_score_panel,
)
from factors.market_vol_regime import compute_index_vol_z, VolRegimeParams
from .execution_v2 import run_execution_t1_equal_weight, ExecutionConfig


RegimeType = Literal["risk_on", "neutral", "risk_off"]


@dataclass
class RegimeConfig:
    buy_th: float
    sell_th: float
    base_target: float
    max_positions: int


@dataclass
class VolConfig:
    a: float = 0.35
    k1: float = 0.08
    k2: float = 0.08


@dataclass
class DDGateConfig:
    start_dd: float = -0.05
    end_dd: float = -0.15


@dataclass
class StrategyConfigV2:
    factor_windows: FactorWindows = field(default_factory=FactorWindows)
    regime_params: dict[RegimeType, RegimeConfig] = None
    vol_regime_params: VolRegimeParams = field(default_factory=VolRegimeParams)
    vol_config: VolConfig = field(default_factory=VolConfig)
    dd_gate: DDGateConfig = field(default_factory=DDGateConfig)
    kelly_lambda: float = 0.45

    def __post_init__(self):
        if self.regime_params is None:
            self.regime_params = {
                "risk_on": RegimeConfig(0.60, 0.30, 0.95, 14),
                "neutral": RegimeConfig(0.57, 0.32, 0.65, 10),
                "risk_off": RegimeConfig(0.00, -0.20, 0.25, 6),
            }


def _soft_dd_gate(eq: pd.Series, cfg: DDGateConfig) -> pd.Series:
    peak = eq.cummax()
    dd = eq / peak - 1.0
    g = pd.Series(index=eq.index, dtype=float)
    for dt, d in dd.items():
        if d >= cfg.start_dd:
            g[dt] = 1.0
        elif d <= cfg.end_dd:
            g[dt] = 0.0
        else:
            g[dt] = (d - cfg.end_dd) / (cfg.start_dd - cfg.end_dd)
    return g


class BacktestEngineV2:
    def __init__(
        self,
        stock_universe: Dict[str, pd.DataFrame],
        index_df: pd.DataFrame,
        strat_cfg: StrategyConfigV2 | None = None,
    ) -> None:
        """
        stock_universe: {code: df}，df 至少包含 ["open","close"]
        index_df: 指数 df，至少包含 ["close"]
        """
        self.stock_universe = stock_universe
        self.index_df = index_df
        self.cfg = strat_cfg or StrategyConfigV2()

    def compute_scores(self) -> Dict[str, pd.DataFrame]:
        base_panel = compute_stock_factor_panel(
            self.stock_universe, self.index_df, self.cfg.factor_windows
        )
        score_panel = build_composite_score_panel(base_panel)
        return score_panel

    def _build_regime_series(self) -> pd.DataFrame:
        df_vol = compute_index_vol_z(self.index_df, self.cfg.vol_regime_params)
        return df_vol[["vol_z", "regime"]]

    def _get_regime_params_on_date(self, regime: RegimeType) -> RegimeConfig:
        return self.cfg.regime_params[regime]

    def build_target_weights(self) -> pd.DataFrame:
        scores = self.compute_scores()
        reg_df = self._build_regime_series()

        all_dates = sorted(set(reg_df.index).intersection(*[df.index for df in scores.values()]))
        codes = list(self.stock_universe.keys())
        tw = pd.DataFrame(index=all_dates, columns=codes, dtype=float)
        meta_rows: list[dict] = []

        equity_dummy = pd.Series(1.0, index=all_dates)
        dd_gate = _soft_dd_gate(equity_dummy, self.cfg.dd_gate)

        for dt in all_dates:
            if dt not in reg_df.index:
                continue
            vol_z = float(reg_df.at[dt, "vol_z"])
            regime_str: RegimeType = reg_df.at[dt, "regime"]  # type: ignore

            rp = self._get_regime_params_on_date(regime_str)

            a = self.cfg.vol_config.a
            k1 = self.cfg.vol_config.k1
            k2 = self.cfg.vol_config.k2

            buy_th = rp.buy_th - k1 * vol_z
            sell_th = rp.sell_th + k2 * vol_z  # currently unused，保留可扩展

            base_target = rp.base_target * (1.0 / (1.0 + a * vol_z))
            base_target = float(np.clip(base_target, 0.0, 1.0))

            gate = float(dd_gate.get(dt, 1.0))
            target_exposure = base_target * gate

            scores_today = {}
            vol_today = {}
            for code, df in scores.items():
                if dt in df.index:
                    scores_today[code] = float(df.at[dt, "score"])
                    vol_today[code] = float(df.at[dt, "vol_r"] if "vol_r" in df.columns else 0.2)

            if not scores_today:
                continue

            ser_score = pd.Series(scores_today)
            candidates = ser_score[ser_score > buy_th]
            if candidates.empty:
                continue
            candidates = candidates.sort_values(ascending=False).iloc[: rp.max_positions]

            score_plus = candidates - buy_th
            vol_series = pd.Series({c: max(vol_today.get(c, 0.2), 1e-4) for c in candidates.index})
            raw = score_plus.clip(lower=0.0) / vol_series
            if raw.sum() <= 0:
                continue
            w = raw / raw.sum()
            w = w * (target_exposure * self.cfg.kelly_lambda)

            for code in candidates.index:
                tw.at[dt, code] = w[code]

            meta_rows.append(
                {
                    "date": dt,
                    "regime": regime_str,
                    "vol_z": vol_z,
                    "buy_th": buy_th,
                    "sell_th": sell_th,
                    "target_exposure": target_exposure,
                    "gate": gate,
                    "num_candidates": len(candidates),
                }
            )

        tw = tw.fillna(0.0)
        self.target_meta_df = pd.DataFrame(meta_rows).set_index("date") if meta_rows else pd.DataFrame()
        return tw

    def run_backtest(self, exec_cfg: ExecutionConfig | None = None) -> pd.DataFrame:
        target_weights = self.build_target_weights()
        exec_result = run_execution_t1_equal_weight(
            price_panel=self.stock_universe,
            target_weights=target_weights,
            cfg=exec_cfg,
            collect_debug=True,
        )

        if isinstance(exec_result, tuple):
            eq, trades_df, exec_debug_df = exec_result
        else:
            eq = exec_result
            trades_df = pd.DataFrame()
            exec_debug_df = pd.DataFrame()

        # 计算 drawdown
        dd_series = eq["equity"] / eq["equity"].cummax() - 1.0
        eq["dd"] = dd_series

        meta_df = getattr(self, "target_meta_df", pd.DataFrame())
        debug_df = pd.DataFrame()
        if not meta_df.empty:
            meta_df = meta_df.sort_index()
            if len(meta_df.index) > 1:
                meta_shift = meta_df.iloc[:-1].copy()
                meta_shift.index = meta_df.index[1:]
                debug_df = meta_shift
        if not exec_debug_df.empty:
            exec_debug_df = exec_debug_df.sort_index()
            if debug_df.empty:
                debug_df = exec_debug_df
            else:
                debug_df = debug_df.join(exec_debug_df, how="outer")
        if not debug_df.empty:
            debug_df["dd"] = dd_series.reindex(debug_df.index)
            debug_df["actual_exposure"] = debug_df.get("actual_exposure", pd.Series(index=debug_df.index))
            debug_df["num_positions"] = debug_df.get("num_positions", pd.Series(index=debug_df.index))
            debug_df["mode"] = debug_df.get("regime", pd.Series(index=debug_df.index))

        self.debug_df = debug_df.sort_index()
        self.trades_df = trades_df
        return eq
