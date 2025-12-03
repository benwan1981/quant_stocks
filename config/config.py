# -*- coding: utf-8 -*-
"""项目配置文件

在这里放一些全局配置，例如：
- 掘金 GM_TOKEN
- 默认数据目录
- 回测参数（默认资金、费率等）

使用方式：
    from config.config import GM_TOKEN, DATA_DIR
"""

from pathlib import Path

# TODO: 把你的掘金 token 填到这里
GM_TOKEN: str = "c9860b7a402983f14c43dcde1feda7591cd50747"
#GM_TOKEN: str = "bd4ff20b-c176-11f0-9675-52560acd7da0"

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
