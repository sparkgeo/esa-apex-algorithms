from pathlib import Path
import re

import boto3
import humanize
from botocore import UNSIGNED
from botocore.config import Config
import geopandas as gpd
import rasterio
from pandas import DataFrame, Series
from shapely import box
from tqdm import trange

from worldcover import (
    output_by_level,
    calculate_total_area,
    calculate_values,
    max_memory_usage,
    land_cover_names,
    s3_bucket,
    create_directories,
    get_tile_keys,
    sum_children,
)


def process(tif_file_names: list[str], stats_df: DataFrame, bottom_level: int):
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
            s3.download_file(s3_bucket, tif_file_names[idx], src_file_name)

        ds = rasterio.open(src_file_name)
        ds_bbox = box(*ds.bounds)

        intersections = nuts_intersections(ds_bbox, stats_df, bottom_level)

        if len(intersections) == 0:
            ds.close()
            continue

        results = calculate_values(ds, intersections)
        stats_df.update(results)

        ds.close()

    sum_children(stats_df, bottom_level, nuts_level_func, "NUTS_ID")
    calculate_total_area(stats_df)

    output_by_level(bottom_level, stats_df, nuts_level_func, Path("output-nuts/worldcover-stats-nuts.fgb"))
    print("Done")


def nuts_level_func(stats_df: DataFrame, level: int) -> Series:
    return Series(stats_df["LEVL_CODE"] == level)


def nuts_intersections(ds_bbox: box, stats_df: DataFrame, level: int):
    return stats_df[
            (stats_df.geometry.intersects(ds_bbox)) & (stats_df["children"] == "")
        ]


def fix_children(stats_df: DataFrame) -> None:
    """
    The formatting of the _children_ property has got mangled, so we clean it up here.
    """
    chlidren = []

    pattern = re.compile(r"([A-Z0-9]+)+")

    for _, row in stats_df.iterrows():
        g = pattern.findall(row.children)
        chlidren.append(",".join(g))

    stats_df["children"] = chlidren


# def sum_children(stats_df: DataFrame) -> None:
#     print("Calculating parent statistics")
#
#     for index, nut in stats_df[stats_df["LEVL_CODE"] == 2].iterrows():
#         code = nut["NUTS_ID"]
#         children = stats_df[
#             (stats_df["NUTS_ID"].str.startswith(code)) & (stats_df["LEVL_CODE"] == 3)]
#
#         if children.empty:
#             continue
#
#         stats_df.loc[index, land_cover_names] = children[land_cover_names].sum()
#     for index, nut in stats_df[stats_df["LEVL_CODE"] == 1].iterrows():
#         code = nut["NUTS_ID"]
#         children = stats_df[
#             (stats_df["NUTS_ID"].str.startswith(code)) & (stats_df["LEVL_CODE"] == 2)]
#
#         if children.empty:
#             continue
#
#         stats_df.loc[index, land_cover_names] = children[land_cover_names].sum()
#     for index, nut in stats_df[stats_df["LEVL_CODE"] == 0].iterrows():
#         code = nut["NUTS_ID"]
#         children = stats_df[
#             (stats_df["NUTS_ID"].str.startswith(code)) & (stats_df["LEVL_CODE"] == 1)]
#
#         if children.empty:
#             continue
#
#         stats_df.loc[index, land_cover_names] = children[land_cover_names].sum()


def main():
    create_directories()
    geometry_path = Path("data/NUTS_with_children.fgb")
    geom_df = gpd.read_file(geometry_path.resolve(), engine="pyogrio")
    fix_children(geom_df)
    keys = get_tile_keys(geom_df, 0, nuts_level_func)
    process(keys, geom_df, 4)


if __name__ == "__main__":
    main()
