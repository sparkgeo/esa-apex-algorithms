# https://gui.world-soils.com/mapserver/Europe?SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage&COVERAGEID=soc_0-5cm_mean_europe_2020-2022&FORMAT=image/tiff&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326&SUBSET=long(8,10)&SUBSET=lat(50,52)&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326
import os
from collections.abc import Callable
import math
from pathlib import Path

import humanize
import psutil
import rasterio
import requests
from pandas import DataFrame, Series
from shapely import box
from tqdm import trange

type LevelFunc = Callable[[DataFrame, int], Series]
type ChildFunc = Callable[[DataFrame, any], Series]
type IntersectionFunc = Callable[[box, DataFrame, int], DataFrame]


def get_process_memory_use():
    proc = psutil.Process(os.getpid())
    mem_info = proc.memory_info()
    return mem_info.rss


base_url = "https://gui.world-soils.com/mapserver/Europe"
coverage_id = "soc_0-5cm_mean_europe_2018-2020"
max_memory_usage = 0


def output_by_level(max_level: int, stats_df: DataFrame, level_fn: LevelFunc, file_path: Path) -> list[Path]:
    output_file_names = []

    for i in trange(max_level, desc=f"Saving to {str(file_path.parent)}"):
        level = stats_df[level_fn(stats_df, i)]
        suffix = file_path.suffix
        file_name = file_path.with_suffix(f".level{i:02}{suffix}")
        output_file_names.append(file_name)
        level.to_file(file_name)

    return output_file_names


def round_down(val, base):
    return math.floor(val / base) * base


def round_up(val, base):
    return math.ceil(val / base) * base


def get_2deg_cells(gdf):
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


def get_tile_keys(geom_df: DataFrame, level: int, level_fn: LevelFunc) -> list[tuple]:
    countries = geom_df[level_fn(geom_df, level)]

    cells = get_2deg_cells(countries)
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


def nuts_level_func(stats_df: DataFrame, level: int) -> Series:
    return Series(stats_df["LEVL_CODE"] == level)


def process(wcs_params: list[tuple],
            stats_df: DataFrame,
            bottom_level: int,
            level_fn: LevelFunc,
            child_fn: ChildFunc,
            intersection_fn: IntersectionFunc,
            code_column_name: str,
            output_path: Path
            ) -> list[Path]:

    stats_df["mean"] = 0.0
    stats_df["min"] = 0.0
    stats_df["max"] = 0.0
    stats_df["sd"] = 0.0

    print("Processing")
    tiff_progress_bar = trange(len(wcs_params))
    for idx in tiff_progress_bar:
        tiff_progress_bar.set_postfix_str(f"{humanize.naturalsize(max_memory_usage)}")
        src_file_name = Path("./.cache") / f"{coverage_id}_{wcs_params[idx][0][0]}_{wcs_params[idx][0][1]}_{wcs_params[idx][0][2]}_{wcs_params[idx][0][3]}.tif"

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

        intersections = intersection_fn(ds_bbox, stats_df, bottom_level)

        if len(intersections) == 0:
            ds.close()
            continue

        # results = calculate_values(ds, intersections)

        # stats_df.loc[results.index] = results

        ds.close()

    # sum_children(stats_df, bottom_level, level_fn, child_fn, code_column_name)
    # calculate_total_area(stats_df)

    file_names = output_by_level(bottom_level, stats_df, level_fn, output_path)
    print("Done")

    return file_names
