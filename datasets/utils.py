import os
import psutil
from pathlib import Path
from collections.abc import Callable
from pandas import DataFrame, Series
from shapely import box
from tqdm import trange
from pyproj import Geod

type LevelFunc = Callable[[DataFrame, int], Series]
type ChildFunc = Callable[[DataFrame, any], Series]
type IntersectionFunc = Callable[[box, DataFrame, int], DataFrame]

geod = Geod(ellps="WGS84")
max_memory_usage = 0


def output_by_level(max_level: int, df: DataFrame, level_fn: LevelFunc, file_path: Path) -> list[Path]:
    """
    Splits the input DataFrame into multiple files based on the level.
    """

    output_file_names = []

    for i in trange(max_level, desc=f"Saving to {str(file_path.parent)}"):
        level = df[level_fn(df, i)]

        if not level.has_sindex:
            _ = level.sindex
            
        suffix = file_path.suffix
        file_name = file_path.with_suffix(f".level{i:02}{suffix}")
        output_file_names.append(file_name)
        level.to_file(file_name)

    return output_file_names


def create_directories() -> tuple[Path, Path]:
    """
    Creates necessary paths for output and caching.
    """

    if not Path("output").exists():
        Path("output").mkdir()

    if not Path(".cache").exists():
        Path(".cache").mkdir()

    return Path("output").resolve(), Path(".cache").resolve()


def get_process_memory_use():
    proc = psutil.Process(os.getpid())
    mem_info = proc.memory_info()
    return mem_info.rss


def checkpoint_memory_usage():
    global max_memory_usage
    max_memory_usage = max(max_memory_usage, get_process_memory_use())
