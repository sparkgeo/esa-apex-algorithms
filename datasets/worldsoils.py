import math
from pathlib import Path

import numpy as np
import humanize
import rasterio
import requests
import urllib3
from geopandas import GeoDataFrame
from pandas import DataFrame
from rasterio import features, DatasetReader
from rasterio.errors import WindowError
from shapely import box
from tqdm import trange, tqdm

from datasets.utils import (
    LevelFunc,
    ChildFunc,
    IntersectionFunc,
    output_by_level,
    get_process_memory_use,
    checkpoint_memory_usage,
    max_memory_usage
)

# On 2025/05/21, world-soils's certificate was invalid.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
base_url = "https://gui.world-soils.com/mapserver/Europe"
coverage_id = "soc_0-5cm_mean_europe_2020-2022"


def round_down(val, base) -> float:
    """
    Rounds down a given value to the nearest multiple of a specified base.

    :param val:
        The value to be rounded down.
    :param base:
        The base to round down to.
    :return:
        The result of rounding down the value to the nearest multiple of
        the base.
    """

    return math.floor(val / base) * base


def round_up(val, base) -> float:
    """
    Rounds up a given value to the nearest multiple of a specified base.

    :param val:
        The value to be rounded down.
    :param base:
        The base to round down to.
    :return:
        The result of rounding down the value to the nearest multiple of
        the base.
    """

    return math.ceil(val / base) * base


def get_2deg_cells(gdf: GeoDataFrame) -> list[tuple]:
    """
    Extracts unique 2x2 degree cells from the geometric boundaries of a GeoDataFrame.

    :param gdf:
        A GeoDataFrame containing geometric data to be analyzed.
    :return:
        A list of tuples representing the unique 2x2 degree cells. Each tuple
        is structured as (min_lon, min_lat, max_lon, max_lat).
    """
    cells = set()

    for geom in gdf.geometry:
        if geom.is_empty:
            continue

        minx, miny, maxx, maxy = geom.bounds

        # Round bounding box to the nearest 2 degrees
        min_lon = round_down(minx, 2)
        max_lon = round_up(maxx, 2)
        min_lat = round_down(miny, 2)
        max_lat = round_up(maxy, 2)

        # Generate 2x2 degree cells within the bounds
        for lon in range(int(min_lon), int(max_lon), 2):
            for lat in range(int(min_lat), int(max_lat), 2):
                cell = (lon, lat, lon + 2, lat + 2)
                cells.add(cell)

    return list(cells)


def attribute_keys() -> list[str]:
    """
    Returns the names of the columns that we want to render in the front end.
    """

    return ["soil_min", "soil_max", "soil_mean"]


def get_tile_keys(df: DataFrame, level: int, level_fn: LevelFunc) -> list[tuple]:
    """
    Returns a list of raster tiles as WCS query parameters for the given level of polygon data.
    """

    polygons = df[level_fn(df, level)]

    cells = get_2deg_cells(polygons)
    params = []

    for cell in cells:
        param = {
            "SERVICE": "WCS",
            "VERSION": "2.0.1",
            "REQUEST": "GetCoverage",
            "COVERAGEID": coverage_id,
            "FORMAT": "image/tiff",
            "OUTPUTCRS": "http://www.opengis.net/def/crs/EPSG/0/4326",
            "SUBSETTINGCRS": "http://www.opengis.net/def/crs/EPSG/0/4326",
            "SUBSET": [f"long({cell[0]},{cell[2]})", f"lat({cell[1]},{cell[3]})"],
        }
        params.append(param)

    return list(zip(cells, params))


def calculate_values(source_raster: DatasetReader, intersections: DataFrame) -> DataFrame:
    """
    Calculates the values for each land cover class that partially or wholly intersects the given polygon.
    """

    ds_bbox = box(*source_raster.bounds)
    src_file_name = Path(source_raster.name)

    geometry_progress_bar = trange(len(intersections), leave=False)

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
        data = source_raster.read(window=window, masked=True)
        data.mask |= (data.data == 32767)
        clipped_geom = geom.intersection(ds_bbox)
        checkpoint_memory_usage()

        if clipped_geom.is_empty:
            continue

        checkpoint_memory_usage()
        mask = features.geometry_mask(
            [clipped_geom], [window.height, window.width], window_xform, invert=True
        )
        checkpoint_memory_usage()

        masked_data = np.ma.array(data[0], mask=mask)

        if not masked_data.mask.all():
            np_sum = np.ma.sum(masked_data)
            intersections.loc[region.name, "soil_min"] = min(intersections.loc[region.name, "soil_min"], np.min(masked_data))
            intersections.loc[region.name, "soil_max"] = max(intersections.loc[region.name, "soil_max"], np.max(masked_data))
            intersections.loc[region.name, "value_sum"] += np_sum
            intersections.loc[region.name, "sample_count"] += (~masked_data.mask).sum()

        checkpoint_memory_usage()

    return intersections


def calculate_mean(df: DataFrame) -> None:
    df["soil_mean"] = np.where(
        df["sample_count"] == 0, 0.0, df["value_sum"] / df["sample_count"]
    )
    df["soil_min"] = df["soil_min"].replace(np.inf, 0.0)
    df["soil_max"] = df["soil_max"].replace(-np.inf, 0.0)


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

            df.loc[index, "soil_min"] = children["soil_min"].min()
            df.loc[index, "soil_max"] = children["soil_max"].max()
            df.loc[index, "value_sum"] = children["value_sum"].sum()
            df.loc[index, "sample_count"] = children["sample_count"].sum()
            df.loc[index, "touched"] = True


def process(wcs_params: list[tuple],
            df: DataFrame,
            bottom_level: int,
            level_fn: LevelFunc,
            child_fn: ChildFunc,
            intersection_fn: IntersectionFunc,
            code_column_name: str,
            output_path: Path
            ) -> list[Path]:
    """
    For each raster tile, calculate the soil organic carbon that partially or wholly intersects the given polygon.
    """

    df["soil_min"] = float("inf")
    df["soil_max"] = -float("inf")
    df["value_sum"] = 0.0
    df["sample_count"] = 0
    df["touched"] = False

    print("Processing")
    tiff_progress_bar = trange(len(wcs_params))
    for idx in tiff_progress_bar:
        tiff_progress_bar.set_postfix_str(f"{humanize.naturalsize(max_memory_usage)}")
        src_file_name = Path("./.cache") / f"{coverage_id}_{wcs_params[idx][0][0]}_{wcs_params[idx][0][1]}_{wcs_params[idx][0][2]}_{wcs_params[idx][0][3]}.tif"
        tiff_progress_bar.set_description_str(f"{src_file_name.name}")

        if not src_file_name.exists():
            response = requests.get(base_url, params=wcs_params[idx][1], verify=False, stream=True)
            if response.status_code == 200:
                with open(src_file_name, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                continue

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
    calculate_mean(df)
    df = df.to_crs(crs="EPSG:4326")

    df.drop(columns=["value_sum", "sample_count", "touched"], axis=1, inplace=True)

    file_names = output_by_level(bottom_level, df, level_fn, output_path)
    print("Done")

    return file_names
