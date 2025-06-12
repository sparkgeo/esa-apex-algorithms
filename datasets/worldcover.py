"""
Calculates the coverage of each land cover class in a polygon in square km. Also adds a total area
so that proportions can be calculated.
"""
from pathlib import Path
from joblib import Parallel, delayed

import geopandas as gpd
from pandas import DataFrame
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

from datasets.utils import (
    LevelFunc,
    ChildFunc,
    IntersectionFunc,
    output_by_level,
    get_process_memory_use,
    geod,
    checkpoint_memory_usage,
    max_memory_usage,
)

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
s3_bucket = "esa-worldcover"


def attribute_keys() -> list[str]:
    """
    Returns the names of the columns that we want to render in the front end.
    """
    return land_cover_names + ["Unknown"]


def _compute_area(row):
    total_area = abs(geod.geometry_area_perimeter(row.geometry)[0]) / 1000000.0
    covered_area = row[land_cover_names].sum()
    unknown = max(0.0, total_area - covered_area)

    return total_area, unknown


def calculate_total_area(df: DataFrame) -> None:
    """
    Calculates the total area of each polygon in square km and adds it to the dataframe.
    """

    results = Parallel(n_jobs=-1)(
        delayed(_compute_area)(row)
        for _, row in tqdm(
            df.iterrows(), total=df.shape[0], desc="Calculating total area"
        )
    )

    # Unzip results
    total_areas, unknowns = zip(*results)

    df["total"] = total_areas
    df["Unknown"] = unknowns


def calculate_values(source_raster: DatasetReader, intersections: DataFrame) -> DataFrame:
    """
    Calculates the values for each land cover class that partially or wholly intersects the given polygon.
    """

    ds_bbox = box(*source_raster.bounds)
    src_file_name = Path(source_raster.name)

    geometry_progress_bar = trange(
        len(intersections), leave=False, desc=src_file_name.name
    )

    for i in geometry_progress_bar:
        region = intersections.iloc[i]
        checkpoint_memory_usage()
        current_memory_usage = get_process_memory_use()
        geometry_progress_bar.set_postfix_str(f"{humanize.naturalsize(current_memory_usage)}")
        geometry_progress_bar.set_description_str(f"{src_file_name.name}")
        geom = region.geometry

        try:
            window = features.geometry_window(source_raster, [geom])
        except WindowError:
            continue

        window_xform = source_raster.window_transform(window)
        data = source_raster.read(window=window)
        clipped_geom = geom.intersection(ds_bbox)
        checkpoint_memory_usage()

        if clipped_geom.is_empty:
            continue

        clipped_geom_area = (
                abs(geod.geometry_area_perimeter(clipped_geom)[0]) / 1000000.0
        )
        checkpoint_memory_usage()
        mask = features.geometry_mask(
            [clipped_geom], [window.height, window.width], window_xform, invert=True
        )
        checkpoint_memory_usage()
        values = np.histogram(data[0][mask], bins=LAND_COVER_CLASSES_BINS)[0]
        np_sum = np.sum(values)

        # Avoid divide by zero errors.
        if not np.isclose(0.0, np_sum, atol=1e-6):
            values = values * (clipped_geom_area / np_sum)
            intersections.loc[region.name, land_cover_names] += values

        checkpoint_memory_usage()

    return intersections



def get_tile_keys(df: DataFrame, level: int, level_fn: LevelFunc) -> list[str]:
    """
    Returns a list of raster tiles as S3 keys for the given level of polygon data.
    """

    countries = df[level_fn(df, level)]
    tile_index_df = gpd.read_file("input/esa_worldcover_grid.fgb", engine="pyogrio")
    intersected_tiles = gpd.sjoin(tile_index_df, countries, how="inner")
    unique_tiles = intersected_tiles.drop_duplicates(subset="ll_tile").copy()

    tile_names = unique_tiles["ll_tile"].tolist()
    tile_keys = [f"v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile_name.strip()}_Map.tif" for tile_name in tile_names]

    unique_tiles["tile_key"] = tile_keys

    unique_tiles = unique_tiles[["ll_tile", "tile_key", "geometry"]]

    unique_tiles.to_file("output/worldcover-tiles.fgb")

    return tile_keys


def sum_children(df: DataFrame, bottom_level: int, level_fn: LevelFunc, child_fn: ChildFunc, code_column_name: str) -> None:
    """
    Iterates over each polygon from the bottom up, summing the values in the children polygons.
    """

    level_progress_bar = trange(bottom_level - 1, 0, -1, desc="Calculating parent statistics")

    for level in level_progress_bar:
        level_df = df[level_fn(df, level - 1)]
        for index, nut in tqdm(level_df.iterrows(), total=level_df.shape[0], leave=False):
            code = nut[code_column_name]
            children = df[child_fn(df, code) & df["touched"]]

            if children.empty:
                continue

            df.loc[index, land_cover_names] = children[land_cover_names].sum()
            df.loc[index, "touched"] = True


def process(tif_file_names: list[str],
            df: DataFrame,
            bottom_level: int,
            level_fn: LevelFunc,
            child_fn: ChildFunc,
            intersection_fn: IntersectionFunc,
            code_column_name: str,
            output_path: Path
            ) -> list[Path]:
    """
    For each raster tile, calculate the values for each land cover class that partially or wholly intersects the given polygon.
    """

    for name in land_cover_names:
        df[name] = 0.0

    df["Unknown"] = 0.0
    df["total"] = 0.0
    df["touched"] = False

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

        intersections = intersection_fn(ds_bbox, df, bottom_level)

        if len(intersections) == 0:
            ds.close()
            continue

        intersections["touched"] = True
        results = calculate_values(ds, intersections)

        df.loc[results.index] = results

        ds.close()

    sum_children(df, bottom_level, level_fn, child_fn, code_column_name)
    df = df[df["touched"]]
    calculate_total_area(df)
    df = df.to_crs(crs="EPSG:4326")

    df.drop(columns=["touched"], axis=1, inplace=True)

    file_names = output_by_level(bottom_level, df, level_fn, output_path)
    print("Done")

    return file_names
