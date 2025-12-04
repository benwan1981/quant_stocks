# init_project_structure.py
# -*- coding: utf-8 -*-

"""
一次性运行脚本：
在当前目录下自动生成量化项目的目录结构和部分模板文件。

使用方法：
    1. 把本文件放到你的项目根目录，比如:
       D:\PyProjects\Projects\stocks\init_project_structure.py

    2. 在该目录执行：
       (stocks) python init_project_structure.py

    3. 运行后会在当前目录下看到：
       data/, universe/, factors/, backtest/, fees/, download/, config/ 等目录
"""

from pathlib import Path
import textwrap

# ================== 1. 配置区域 ==================

# 项目根目录：默认使用当前工作目录
PROJECT_ROOT = Path.cwd()

# 需要创建的目录（相对 PROJECT_ROOT）
DIRS = [
    "data",
    "data/raw",        # 原始数据（东方财富 / 掘金原始下载）
    "data/gm",         # 掘金下载的原始 CSV
    "data/processed",  # 清洗后的数据、特征数据
    "universe",        # 股票池 & 指数成分管理
    "factors",         # 因子计算相关
    "signals",         # 信号与打分逻辑
    "backtest",        # 单票 & 组合回测引擎
    "fees",            # 手续费 / 融资费率模块
    "download",        # 各种数据下载脚本（东方财富 / 掘金 / 其他）
    "common",          # 公共工具函数（时间、日志、配置等）
    "config",          # 配置文件（token、路径、参数）
    "logs",            # 回测日志 / 运行日志
]

# 需要视为 Python 包的目录（会自动生成 __init__.py）
PKG_DIRS = [
    "universe",
    "factors",
    "signals",
    "backtest",
    "fees",
    "download",
    "common",
    "config",
]

# 需要自动生成的模板文件（相对 PROJECT_ROOT）
TEMPLATE_FILES = {
    "config/config.py": textwrap.dedent(
        """\
        # -*- coding: utf-8 -*-
        \"\"\"项目配置文件

        在这里放一些全局配置，例如：
        - 掘金 GM_TOKEN
        - 默认数据目录
        - 回测参数（默认资金、费率等）

        使用方式：
            from config.config import GM_TOKEN, DATA_DIR
        \"\"\"

        from pathlib import Path

        # TODO: 把你的掘金 token 填到这里
        GM_TOKEN: str = "YOUR_GM_TOKEN_HERE"

        # 数据根目录（默认就是项目下的 data 目录）
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        DATA_DIR = PROJECT_ROOT / "data"
        RAW_DATA_DIR = DATA_DIR / "raw"
        GM_DATA_DIR = DATA_DIR / "gm"
        PROCESSED_DATA_DIR = DATA_DIR / "processed"

        # 默认回测参数（可以按需修改）
        DEFAULT_INITIAL_CASH: float = 100000.0
        DEFAULT_FEE_RATE: float = 0.0005
        DEFAULT_STAMP_DUTY: float = 0.001
        DEFAULT_SLIPPAGE: float = 0.0005
        """
    ),
    "PROJECT_STRUCTURE.md": textwrap.dedent(
        """\
        # 量化项目目录结构说明（V1）

        本结构由 `init_project_structure.py` 自动生成，主要用于：
        - 单票回测
        - 股票池多标的一致回测
        - 后续接入掘金数据、扩展因子、组合管理

        ## 目录说明

        - `data/`
          - `raw/`        原始数据（不做任何清洗，东方财富等）
          - `gm/`         掘金下载的原始 CSV
          - `processed/`  清洗、对齐、带因子的中间结果

        - `universe/`
          股票池定义，如：
          - A 股全市场
          - HS300 成分股
          - 中证1000 成分股
          - 行业/主题股票池（白酒、高股息等）

        - `factors/`
          因子计算模块：
          - MA / MOM / VOL / RSI / MACD 等
          - 指数相关性、汇率、期货等扩展因子

        - `signals/`
          信号与打分逻辑：
          - 如何从因子组合出分数
          - 生成原始持仓意图 raw_position
          - 执行层（T+1、持有天数、风控）连接 backtest

        - `backtest/`
          回测引擎：
          - 单标的回测（类似 practice_single_stock）
          - 股票池 TopN 组合回测（类似 practice_universe）
          - 年度分段回测 / 绩效分析

        - `fees/`
          手续费与融资费率模块：
          - 统一管理佣金、印花税、融资利息计算
          - 未来可以扩展股票/期货不同费率

        - `download/`
          下载脚本：
          - 东方财富日线 / 分时下载
          - 掘金日线 / 分钟线下载
          - 指数、期货、汇率等数据下载

        - `common/`
          放通用工具：
          - 时间/交易日处理
          - 日志封装
          - 配置加载等

        - `config/`
          配置文件：
          - `config.py` 中记录 GM_TOKEN、目录路径、默认参数

        - `logs/`
          回测日志、错误信息、运行记录等

        ## 使用建议

        - 单票回测脚本可以放在 `backtest/single_stock_xxx.py`
        - 股票池回测脚本可以放在 `backtest/universe_xxx.py`
        - 下载脚本，比如 `gm_download_all_a_daily.py`，建议放在 `download/` 中
        - 费用模块 `FeeEngine` 建议放在 `fees/engine.py`，并在 `fees/__init__.py` 中导出
        """
    ),
}


# ================== 2. 具体实现 ==================

def create_directories():
    print(f"项目根目录: {PROJECT_ROOT}\n")

    for rel in DIRS:
        path = PROJECT_ROOT / rel
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"📁 已创建目录: {rel}")
        else:
            print(f"📂 目录已存在(跳过): {rel}")


def create_init_files():
    for rel in PKG_DIRS:
        pkg_path = PROJECT_ROOT / rel
        if not pkg_path.exists():
            # 目录本身若不存在，这里顺带建一下
            pkg_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 已创建包目录: {rel}")

        init_file = pkg_path / "__init__.py"
        if not init_file.exists():
            init_file.write_text("# -*- coding: utf-8 -*-\n", encoding="utf-8")
            print(f"🧩 已创建: {rel}/__init__.py")
        else:
            print(f"🧩 __init__.py 已存在(跳过): {rel}/__init__.py")


def create_template_files():
    for rel, content in TEMPLATE_FILES.items():
        path = PROJECT_ROOT / rel
        if path.exists():
            print(f"📄 模板已存在(跳过): {rel}")
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"📄 已生成模板文件: {rel}")


def main():
    print("🚀 开始初始化量化项目目录结构...\n")

    # 1) 创建目录
    create_directories()
    print("")

    # 2) 创建 __init__.py
    create_init_files()
    print("")

    # 3) 创建模板文件
    create_template_files()
    print("")

    print("✅ 完成。现在你可以把现有的脚本按功能分类，迁移到对应目录中。")


if __name__ == "__main__":
    main()