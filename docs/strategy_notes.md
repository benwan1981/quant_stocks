# 股票回测项目说明（速记版）

## 目录结构
- data/gm/         掘金日线数据（A股、ETF 等）
- download/        各类数据下载脚本（掘金 + 东方财富）
- factors/
  - stock_factors.py    价格/量因子计算 + attach_scores
  - policy_factor.py    五年计划 / 主题 映射出的政策因子
- backtest/
  - practice_single_stock.py  单标的策略测试 + JSON/CSV 输出 + 画图
  - run_param_table.py        读取参数表批量回测
  - plotting.py               资金曲线绘图
  - backtest_io.py            回测结果 JSON IO（策略定义、元信息等）
- config/
  - config.py                 掘金 GM_TOKEN 等
  - policy_themes.csv         五年计划主题表
  - policy_stock_mapping.csv  个股/ETF -> 主题映射
  - param_table.csv           批量调参表（本项目调参入口）

## 策略核心

1. 原始数据：日线 OHLCV（按 date 索引）
2. 因子计算：compute_stock_factors(df)
3. （可选）叠加政策因子：attach_policy_factor(df_fac, code, market)
4. 打分：attach_scores(df_fac)，得到
   - trend_score, momentum_score, volume_score, risk_score, technical_score, （以及 policy_score）
5. total_score 组合：
   - 在 practice_single_stock.py 中使用默认权重
   - 在 run_param_table.py 中按参数表的 w_xxx 动态加权
6. 信号：
   - 使用 generate_signals_v2：
     - 开仓：total_score >= buy_score_thresh 且 trend_score / risk_score 达标
     - 平仓：total_score <= sell_score_thresh 或 trend 变弱，且持有天数 >= min_hold_days
   - T+1 执行（今天的持仓 = 昨天的 raw_position）
7. 回测：
   - simple_backtest：单票满仓进出，带佣金 / 印花税
   - backtest_buy_and_hold：首日全仓买入持有到结束
8. 输出：
   - CSV：策略资金曲线 + Buy&Hold
   - JSON：包含 meta / 年度表现 / 信号 / 资金曲线
   - PNG：策略资金曲线 + 策略 vs Buy&Hold 对比图
