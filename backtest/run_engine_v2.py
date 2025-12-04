# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from backtest.engine_v2 import BacktestEngineV2
from backtest.execution_v2 import ExecutionConfig
from backtest.strategy_v2_loader import load_strategy_config_v2
from backtest.utils_universe_v2 import (
    load_stock_universe_from_dir,
    build_index_from_universe,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "gm_HS300_equity"

    # 1) 构造股票池和指数（用成分股重建等权指数）
    stock_universe = load_stock_universe_from_dir(data_dir)
    index_df = build_index_from_universe(stock_universe)

    # 2) 初始化策略与执行配置（从 YAML 读取）
    strat_cfg = load_strategy_config_v2(root / "config" / "strategy_v2.yaml")
    exec_cfg = ExecutionConfig(
        initial_cash=1_000_000,
        fee_rate=0.0005,
        slippage=0.0005,
    )

    # 3) 运行回测
    engine = BacktestEngineV2(
        stock_universe=stock_universe,
        index_df=index_df,
        strat_cfg=strat_cfg,
    )
    equity = engine.run_backtest(exec_cfg)

    # 4) 输出结果
    out_dir = root / "backtest" / "output_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "equity_v2.csv"
    equity.to_csv(out_path)

    print("回测完成，资金曲线已导出到:", out_path)
    print(equity.tail())


if __name__ == "__main__":
    main()
