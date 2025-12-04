# 量化项目目录结构说明（V2）

新版结构在原始脚手架的基础上增加了数据来源区分、模板与测试资产、老版本归档等模块，方便扩展多标的回测与策略验证。

## 顶层目录总览

```
stocks/
├── backtest/           回测引擎、模板与结果
├── common/             通用工具（掘金数据读取等）
├── config/             全局配置与静态列表
├── data/               各类原始/处理后行情
├── docs/               策略及结构说明
├── download/           数据下载脚本
├── factors/            因子计算模块
├── fees/               费用与滑点模型
├── logs/               运行日志
├── old_ver/            历史版本保留
├── signals/            信号打分与持仓意图
├── stocks/             早期快速验证脚本
├── universe/           股票池定义
└── create_quant_project.py / PROJECT_STRUCTURE.md 等辅助脚本
```

## 目录与关键文件

### `backtest/`
- `practice_single_stock.py`、`practice_universe.py`：单票与多票回测入口。
- `backtest_io.py`、`plotting.py`、`dashboard_export.py`：读写、图表和仪表盘导出工具。
- `results/`、`plots/`：策略结果 CSV/JSON 与图形输出，`template/` 提供仪表盘 HTML 与示例资产 CSV。
- `run_param_table.py`、`param_grid_single_stock.py`：参数表与网格实验。

### `common/`
- `gm_loader.py`：封装掘金数据接口及通用工具；`__init__.py` 暴露公共方法。

### `config/`
- `config.py`：GM Token、路径、默认参数等核心配置。
- `gm_*_list.csv`：掘金股票/期货/日线清单；`param_table.csv` 统一参数表。
- `policy_stock_mapping.csv`、`policy_themes.csv`：政策主题及个股映射。

### `data/`
- `raw/`：原封不动的下载数据（东方财富等）。
- `gm/`、`gm_equity/`、`gm_equity_intraday/`、`gm_futures/`：掘金日线/分钟/期货数据。
- `hk/`：港股及 ETF 相关行情；`eastmoney_macro/`：汇率、指数等宏观时间序列。
- `processed/`：清洗、对齐及带因子的中间结果。

### `docs/`
- `strategy_notes.md`、`stru.md`：策略逻辑、结构说明及记录。

### `download/`
- `gm_download_all.py`、`gm_download_daily.py`、`gm_download_intraday.py` 等脚本负责掘金批量下载。
- `download_universe_em.py`、`download_hsi_eastmoney.py`、`eastmoney_macro_download.py` 等针对东方财富或恒指来源。
- 其他按标的（如 `download_159892_daily.py`、`download_600048.py`）的示例脚本可直接改造复用。

### `factors/`
- `basic_factors.py`、`stock_factors.py`、`policy_factor.py`：技术面、基本面及政策主题相关因子实现，统一在 `__init__.py` 中组织。

### `fees/`
- `fee_engine.py`：佣金、印花税、融资利息等费用模型。
- `__init__.py`：导出 FeeEngine，方便在回测或交易层调用。

### `signals/`
- 存放信号合成与打分逻辑，生成 `raw_position` 并衔接回测执行层。

### `universe/`
- 股票池定义、列表及辅助函数（如全市场、指数成分、行业主题等）。

### 其它辅助目录
- `logs/`：记录回测运行日志和异常。
- `old_ver/`：保留早期实验脚本及数据，便于回溯。
- `stocks/`：快速验证脚本、旧版下载和数据示例，与 `old_ver/` 类似但更偏向单独实验。
- `__pycache__/`：Python 缓存，可按需清理。

### 顶层脚本
- `create_quant_project.py`：初始化/扩展项目结构的自动化脚本。
- `PROJECT_STRUCTURE.md`：本文档，可在脚手架或部署时同步更新。

## 使用建议
- 单票回测可基于 `backtest/practice_single_stock.py` 调整参数表或策略模块。
- 股票池策略以 `practice_universe.py` 为模板，配合 `universe/` 定义和 `signals/` 输出排名。
- 数据获取统一放在 `download/`，下载结果对应归档至 `data/` 下的分类目录。
- 新的因子/信号模块建议在 `factors/`、`signals/` 中分文件维护，并在 `config/param_table.csv` 中记录参数。
- 发布或备份策略时，将说明整理到 `docs/`，并把新结构变化同步至本文档。





————————————-----------------------------————————————————————
# 量化项目目录结构说明（V2 对齐版）

当前结构在原始脚手架基础上做了两件事：

1. **数据源拆分清晰**：东方财富 / 掘金（日线 & 分时）分目录存放；
2. **回测入口统一用参数表驱动**：`practice_single_stock.py` & `run_param_table.py` 共用 `config/param_table.csv`。

---

## 顶层目录总览

```bash
stocks/
├── backtest/           回测引擎、入口脚本、图表与结果
├── common/             通用工具（gm 数据 loader 等）
├── config/             全局配置、参数表、股票池清单
├── data/               各类原始 / 处理后行情数据
├── docs/               策略说明、结构文档
├── download/           各类数据下载脚本（gm / 东方财富 等）
├── factors/            因子计算与打分模块
├── fees/               费用与滑点模型
├── logs/               运行日志
├── old_ver/            历史版本代码与数据
├── signals/            （预留）信号合成模块
├── stocks/             早期单票实验脚本（旧）
├── universe/           股票池定义与工具
├── create_quant_project.py
└── PROJECT_STRUCTURE.md



backtest/ 回测模块

核心脚本：
	•	practice_single_stock.py
	•	单票回测入口，从 config/param_table.csv 读取一行配置（PARAM_ROW_IDX 控制行号）。
	•	数据统一通过 common/gm_loader.py 转成标准 OHLCV：
	•	支持“原始 gm CSV”（带 eob）；
	•	也兼容已经有 date,open,high,low,close,volume 的旧 CSV。
	•	计算因子 → 打分 → generate_signals_v2 → simple_backtest → Buy & Hold 对比。
	•	导出：
	•	策略 & B&H 资金曲线 CSV（backtest/results/）
	•	JSON 回测报告（含 meta、年度表现、信号等）
	•	Web dashboard 用 CSV（backtest/results/*_dashboard.csv）
	•	总览图 PNG（backtest/plots/{code}_overview.png）
	•	practice_universe.py
	•	多标的 / 股票池回测入口（按股票池＋参数集批量跑，逻辑与单票类似）。
	•	run_param_table.py
	•	表驱动批量回测入口：对 config/param_table.csv 中每一行执行回测。
	•	与 practice_single_stock.py 共享：
	•	因子逻辑（compute_stock_factors + attach_scores）
	•	新版信号逻辑 generate_signals_v2
	•	费用引擎 FeeEngine
	•	输出：
	•	汇总结果：backtest/param_table_summary.csv
	•	每个参数组合一份 dashboard CSV：backtest/results/dashboard_runs/…

工具 & 输出：
	•	backtest_io.py：StrategyConfig、BacktestMeta 等回测元信息结构体。
	•	plotting.py：
	•	save_backtest_overview_png：K 线 + 买卖点 + 策略 & B&H 资金曲线总览图。
	•	其他散点 / 参数可视化工具（按需使用）。
	•	dashboard_export.py：将一次回测的数据整理为前端仪表盘可用的 CSV。
	•	results/：
	•	*_strategy_*.csv / *_buyhold_*.csv：单次回测的资金曲线；
	•	*_backtest_*.json：完整回测报告；
	•	*_dashboard.csv：单次回测的 dashboard 源数据；
	•	dashboard_runs/：参数表批量跑出来的多组合 dashboard CSV。
	•	plots/：
	•	{code}_overview.png：单票总览图；
	•	param_by_symbol/、param_runs/：参数扫描图等。
	•	template/：
	•	dashboard.html / json_dashboard.html：前端仪表盘模板；
	•	示例 JSON / CSV 资产，可用于本地调试前端展示。

⸻

common/ 通用工具
	•	gm_loader.py：
	•	load_gm_ohlcv(path)
	•	统一把“原始 gm CSV”（history(..., df=True) 直接保存）转成标准格式：
	•	时间：eob → date（日期粒度，去掉时分秒）；
	•	列：open, high, low, close, volume。
	•	若文件本身已有 date 列，也能兼容。
	•	load_gm_ohlcv_by_code(code, data_dir=...)
	•	在指定目录（如 ./data/gm_equity）按 {code}_*.csv 自动匹配文件并加载。
	•	__init__.py：按需导出公共函数（例如直接从 common import loader）。

⸻

config/ 配置与静态资源
	•	config.py：
	•	GM_TOKEN：掘金 Token；
	•	默认路径、全局回测参数等。
	•	gm_equity_list.csv：
	•	掘金股票 / 指数日线下载清单；
	•	供 gm_download_all.py 和 gm_download_intraday.py 复用。
	•	gm_futures_list.csv：
	•	掘金股指期货合约下载清单。
	•	param_table.csv（核心）：
	•	行 = 「标的 + 策略参数 + 因子权重 + 回测周期」：
	•	symbol_code, symbol_market, data_file
	•	buy_score_thresh, sell_score_thresh, min_hold_days
	•	use_policy
	•	w_trend, w_mom, w_vol, w_risk, w_tech, w_policy
	•	backtest_start, backtest_end（可选，不填则用脚本默认周期）
	•	由 practice_single_stock.py 与 run_param_table.py 共同使用。
	•	policy_stock_mapping.csv / policy_themes.csv：
	•	政策主题与个股映射表，供 factors/policy_factor.py 使用。

⸻

data/ 数据目录

建议约定如下分层（部分旧目录可以逐步迁移）：
	•	raw/
	•	原始下载数据（例如东方财富接口的原始 CSV），不做任何字段改动。
	•	gm_equity/
	•	掘金 股票 / 指数日线：
	•	命名约定：600941_中国移动_D_qfq_gm.csv 或类似；
	•	字段：当前是 history(..., frequency="1d", df=True) 的结果，可以保持原始字段，读取时统一用 gm_loader 转换。
	•	gm_equity_intraday/
	•	掘金 股票 / 指数分时（1m / 5m 等）：
	•	命名约定：600941_中国移动_1m_gm_raw.csv；
	•	保留 history 返回的所有原始字段。
	•	gm_futures/
	•	掘金 股指期货日线：
	•	如 IF2501_沪深300指数期货_FUT_D_gm.csv。
	•	gm/
	•	早期掘金数据存放目录，逐步被 gm_equity/、gm_equity_intraday/ 替代，可作为兼容 / 过渡。
	•	hk/
	•	港股与恒指相关行情（如 HSI_INDEX_D_eastmoney.csv、HSI_MAIN_D_eastmoney.csv）。
	•	eastmoney_macro/
	•	汇率、宏观指数等东方财富下载的时间序列（CNY_USD_D_eastmoney.csv 等）。
	•	processed/
	•	各类预处理结果、对齐后的面板数据、带因子的中间结果等。

约定：回测脚本不要直接改写 raw、gm_*，写入统一放 backtest/results/ 或 data/processed/。

⸻

download/ 下载脚本
	•	掘金类：
	•	gm_download_all.py
	•	股票 / 指数日线：download_daily_equity & batch_download_equity_from_csv
输出：./data/gm_equity/
	•	股指期货日线：download_future_kline & batch_download_futures_from_csv
输出：./data/gm_futures/
	•	gm_download_intraday.py（你写的分时模块，实际文件名以你本地为准）
	•	download_intraday_equity：单标的分时（1m / 5m / 15m…）
	•	batch_download_intraday_from_csv：从 gm_equity_list.csv 批量；
	•	特点：
	•	用 gm.history 原样落盘，不做字段清洗；
	•	自动处理「只能下载最近 180 个自然日且不含当天」的权限限制；
	•	输出：./data/gm_equity_intraday/。
	•	东方财富类：
	•	download_universe_em.py：A 股 / ETF 日线（前复权）；
	•	download_hsi_eastmoney.py：恒指相关；
	•	eastmoney_macro_download.py：宏观时间序列。
	•	示例 / 旧脚本：
	•	download_159892_daily.py、download_600048.py 等，可参考其写法改造。

⸻

factors/ 因子层
	•	stock_factors.py：
	•	compute_stock_factors(df)：基于日线 OHLCV 计算技术类因子（MA、动量、成交量比、RSI 等）。
	•	attach_scores(df_fac)：将各类因子打分、合成基础 score。
	•	policy_factor.py：
	•	attach_policy_factor(df_fac, code, market)：根据政策主题、新闻或自定义逻辑给出 policy_score。
	•	basic_factors.py：可放与具体策略无关的通用因子。
	•	__init__.py：统一导出核心因子函数，方便其他模块调用。

⸻

fees/ 费用模型
	•	fee_engine.py：
	•	FeeConfig：配置佣金率、印花税率、融资利率等；
	•	FeeEngine：根据每日买卖金额 & 融资余额计算当日费用。
	•	在回测中：
	•	simple_backtest 使用 FeeEngine 统一计算当日手续费、印花税和融资成本，并挂在结果 DataFrame 上（eq._fee_engine），方便后续汇总。

⸻

signals/ 信号层（预留）

目前主要信号逻辑在回测脚本中（generate_signals_v2），后续可以迁移到 signals/ 目录中：
	•	把：
	•	generate_signals_v2(...)
	•	以及可能的「多标的排名 / 择时信号」
	•	抽出来放 signals/stock_signals.py 之类，回测脚本只负责「调度 + 回测」，信号策略独立版本管理。

⸻

universe/ 股票池
	•	维护各类股票池定义：
	•	指数成分（沪深 300、中证 500 等）
	•	行业 / 主题池
	•	自定义观察池等。
	•	可提供工具函数：
	•	load_universe(name) 返回 symbol 列表；
	•	和 config/gm_equity_list.csv 配合使用。

⸻

其它辅助目录
	•	logs/：记录批量跑回测时的日志与异常。
	•	old_ver/：
	•	旧版数据 / 旧版脚本整体存放位置，避免影响现在逻辑，但保留可回溯性。
	•	stocks/：
	•	早期单票测试脚本、独立实验用代码（例如最早的 practice_single_stock*.py），逐步已被 backtest/ 里的新版取代。

⸻

顶层脚本
	•	create_quant_project.py：
	•	初始化 / 扩展项目结构的自动化脚本，可以根据需要生成空目录或模板文件。
	•	PROJECT_STRUCTURE.md：
	•	即本文档；建议在：
	•	新增目录、
	•	调整数据路径、
	•	回测入口更改
时同步更新，保证结构与代码一致。




