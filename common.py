from pathlib import Path

# 数据路径
DEFAULT_DATA_PATH = Path(r"D:/科大云盘/A股数据")
PATH = (
    DEFAULT_DATA_PATH if DEFAULT_DATA_PATH.exists() else Path(__file__).parent / "data"
)

# 复权常数
ADJUST_NONE = 0
ADJUST_PREV = 1
ADJUST_POST = 2
