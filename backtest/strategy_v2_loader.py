# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml

from backtest.engine_v2 import (
    StrategyConfigV2,
    RegimeConfig,
    VolConfig,
    DDGateConfig,
    FactorWindows,
)

PathLike = Union[str, Path]


def load_strategy_config_v2(config_path: PathLike | None = None) -> StrategyConfigV2:
    """
    从 config/strategy_v2.yaml 读取参数，构造 StrategyConfigV2。
    如果文件不存在，则返回默认 StrategyConfigV2()。
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "strategy_v2.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return StrategyConfigV2()

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    look = cfg.get("LOOKBACK", {})
    windows = FactorWindows(
        mom_short=look.get("MOM_SHORT", 10),
        mom_long=look.get("MOM_LONG", 60),
        vol=look.get("VOL", 60),
    )

    reg = cfg.get("REGIMES", {})
    regime_params = {
        "risk_on": RegimeConfig(
            buy_th=reg["risk_on"]["buy_th"],
            sell_th=reg["risk_on"]["sell_th"],
            base_target=reg["risk_on"]["base_target"],
            max_positions=reg["risk_on"]["max_positions"],
        ),
        "neutral": RegimeConfig(
            buy_th=reg["neutral"]["buy_th"],
            sell_th=reg["neutral"]["sell_th"],
            base_target=reg["neutral"]["base_target"],
            max_positions=reg["neutral"]["max_positions"],
        ),
        "risk_off": RegimeConfig(
            buy_th=reg["risk_off"]["buy_th"],
            sell_th=reg["risk_off"]["sell_th"],
            base_target=reg["risk_off"]["base_target"],
            max_positions=reg["risk_off"]["max_positions"],
        ),
    }

    vol_cfg = cfg.get("VOL", {})
    vol = VolConfig(
        a=vol_cfg.get("a", 0.35),
        k1=vol_cfg.get("k1", 0.08),
        k2=vol_cfg.get("k2", 0.08),
    )

    dd_cfg = cfg.get("DD_GATE", {})
    dd_gate = DDGateConfig(
        start_dd=dd_cfg.get("start_dd", -0.05),
        end_dd=dd_cfg.get("end_dd", -0.15),
    )

    kelly_lambda = cfg.get("KELLY", {}).get("lambda", 0.45)

    return StrategyConfigV2(
        factor_windows=windows,
        regime_params=regime_params,
        vol_config=vol,
        dd_gate=dd_gate,
        kelly_lambda=kelly_lambda,
    )
