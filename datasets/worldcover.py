import os
from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import psutil
from pandas import DataFrame, Series
from pyproj import Geod
from rasterio.errors import WindowError
from tqdm import trange, tqdm
from rasterio import features, DatasetReader
import numpy as np
from shapely import box
import boto3
import humanize
from botocore import UNSIGNED
from botocore.config import Config
import rasterio


# Land cover classification mapping
LAND_COVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

# Numpy's histogram bins are half-open except the last bin, so we need to add a right edge to the last bin.
# This can be anything > 100, but 101 seems safe enough if new categories get added to the end but we haven't updated the code.
LAND_COVER_CLASSES_BINS = list(LAND_COVER_CLASSES.keys())
LAND_COVER_CLASSES_BINS.append(101)
land_cover_names = list(LAND_COVER_CLASSES.values())
geod = Geod(ellps="WGS84")
max_memory_usage = 0
s3_bucket = "esa-worldcover"

type LevelFunc = Callable[[DataFrame, int], Series]
type ChildFunc = Callable[[DataFrame, any], Series]
type IntersectionFunc = Callable[[box, DataFrame, int], DataFrame]


def get_process_memory_use():
    proc = psutil.Process(os.getpid())
    mem_info = proc.memory_info()
    return mem_info.rss


def output_by_level(max_level: int, stats_df: DataFrame, level_fn: LevelFunc, file_path: Path) -> list[Path]:
    output_file_names = []

    for i in trange(max_level, desc=f"Saving to {str(file_path.parent)}"):
        level = stats_df[level_fn(stats_df, i)]
        suffix = file_path.suffix
        file_name = file_path.with_suffix(f".level{i:02}{suffix}")
        output_file_names.append(file_name)
        level.to_file(file_name)

    return output_file_names


def calculate_total_area(statistics: DataFrame):
    areas = []
    unknowns = []

    for _, row in tqdm(statistics.iterrows(), total=statistics.shape[0], desc="Calculating total area"):
        total_area = abs(geod.geometry_area_perimeter(row.geometry)[0]) / 1000000.0
        covered_area = row[land_cover_names].sum()
        areas.append(total_area)
        unknowns.append(max(0.0, total_area - covered_area))

    statistics["total_area"] = areas
    statistics["Unknown"] = unknowns


def calculate_values(source_raster: DatasetReader, intersections: DataFrame) -> DataFrame:
    global max_memory_usage

    ds_bbox = box(*source_raster.bounds)
    src_file_name = Path(source_raster.name)

    geometry_progress_bar = trange(
        len(intersections), leave=False, desc=src_file_name.name
    )
    # TODO: Parallelize
    for i in geometry_progress_bar:
        region = intersections.iloc[i]
        max_memory_usage = max(max_memory_usage, get_process_memory_use())
        current_memory_usage = get_process_memory_use()
        geometry_progress_bar.set_postfix_str(f"{humanize.naturalsize(current_memory_usage)}")
        geometry_progress_bar.set_description_str(f"{src_file_name.name}")
        geom = region.geometry

        try:
            window = features.geometry_window(source_raster, [geom])
        except WindowError as e:
            continue

        window_xform = source_raster.window_transform(window)
        data = source_raster.read(window=window)
        clipped_geom = geom.intersection(ds_bbox)
        max_memory_usage = max(max_memory_usage, get_process_memory_use())

        if clipped_geom.is_empty:
            continue

        clipped_geom_area = (
                abs(geod.geometry_area_perimeter(clipped_geom)[0]) / 1000000.0
        )
        max_memory_usage = max(max_memory_usage, get_process_memory_use())
        mask = features.geometry_mask(
            [clipped_geom], [window.height, window.width], window_xform, invert=True
        )
        max_memory_usage = max(max_memory_usage, get_process_memory_use())
        values = np.histogram(data[0][mask], bins=LAND_COVER_CLASSES_BINS)[0]
        np_sum = np.sum(values)

        # Avoid divide by zero errors.
        if not np.isclose(0.0, np_sum, atol=1e-6):
            values = values * (clipped_geom_area / np_sum)
            intersections.loc[region.name, land_cover_names] += values

        max_memory_usage = max(max_memory_usage, get_process_memory_use())

    return intersections


def create_directories() -> tuple[Path, Path]:
    """
    Creates necessary paths for output and caching.
    """

    if not Path("output").exists():
        Path("output").mkdir()

    if not Path(".cache").exists():
        Path(".cache").mkdir()

    return Path("output").resolve(), Path(".cache").resolve()


def get_tile_keys(geom_df: DataFrame, level: int, level_fn: LevelFunc) -> list[str]:
    countries = geom_df[level_fn(geom_df, level)]
    tile_index_df = gpd.read_file("input/esa_worldcover_grid.fgb", engine="pyogrio")
    intersected_tiles = gpd.sjoin(tile_index_df, countries, how="inner")
    unique_tiles = intersected_tiles.drop_duplicates(subset="ll_tile").copy()

    tile_names = unique_tiles["ll_tile"].tolist()
    tile_keys = [f"v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile_name.strip()}_Map.tif" for tile_name in tile_names]

    unique_tiles["tile_key"] = tile_keys

    unique_tiles = unique_tiles[["ll_tile", "tile_key", "geometry"]]

    unique_tiles.to_file("output/worldcover-tiles.fgb")

    return tile_keys


def sum_children(stats_df: DataFrame, bottom_level: int, level_fn: LevelFunc, child_fn: ChildFunc, code_column_name: str) -> None:
    level_progress_bar = trange(bottom_level - 1, 0, -1, desc="Calculating parent statistics")
    for level in level_progress_bar:
        df = stats_df[level_fn(stats_df, level - 1)]
        for index, nut in tqdm(df.iterrows(), total=df.shape[0], leave=False):
            code = nut[code_column_name]
            children = stats_df[child_fn(stats_df, code)]

            if children.empty:
                continue

            stats_df.loc[index, land_cover_names] = children[land_cover_names].sum()


def process(tif_file_names: list[str],
            stats_df: DataFrame,
            bottom_level: int,
            level_fn: LevelFunc,
            child_fn: ChildFunc,
            intersection_fn: IntersectionFunc,
            code_column_name: str,
            output_path: Path
            ) -> list[Path]:
    for name in land_cover_names:
        stats_df[name] = 0.0

    stats_df["Unknown"] = 0.0
    stats_df["total_area"] = 0.0

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    print("Processing")
    tiff_progress_bar = trange(len(tif_file_names))
    for idx in tiff_progress_bar:
        tiff_progress_bar.set_postfix_str(f"{humanize.naturalsize(max_memory_usage)}")
        src_file_name = Path("./.cache/" + Path(tif_file_names[idx]).name)

        if not src_file_name.exists():
            s3.download_file(s3_bucket, tif_file_names[idx], str(src_file_name.resolve()))

        ds = rasterio.open(src_file_name)
        ds_bbox = box(*ds.bounds)

        intersections = intersection_fn(ds_bbox, stats_df, bottom_level)

        if len(intersections) == 0:
            ds.close()
            continue

        results = calculate_values(ds, intersections)

        stats_df.loc[results.index] = results

        ds.close()

    sum_children(stats_df, bottom_level, level_fn, child_fn, code_column_name)
    calculate_total_area(stats_df)

    file_names = output_by_level(bottom_level, stats_df, level_fn, output_path)
    print("Done")

    return file_names
