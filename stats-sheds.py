from pathlib import Path

import boto3
import humanize
import pandas as pd
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
    land_cover_names,
    max_memory_usage,
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

    tiff_progress_bar = trange(len(tif_file_names), desc="processing")
    for idx in tiff_progress_bar:
        tiff_progress_bar.set_postfix_str(f"{humanize.naturalsize(max_memory_usage)}")
        src_file_name = Path("./.cache/" + Path(tif_file_names[idx]).name)

        if not src_file_name.exists():
            s3.download_file(s3_bucket, tif_file_names[idx], src_file_name)

        ds = rasterio.open(src_file_name)
        ds_bbox = box(*ds.bounds)

        intersections = hydrosheds_intersections(ds_bbox, stats_df, bottom_level)

        if len(intersections) == 0:
            ds.close()
            continue

        results = calculate_values(ds, intersections)

        for res_idx, row in results.iterrows():
            stats_df.loc[res_idx, row.index] = row

        ds.close()

    sum_children(stats_df, bottom_level, hydrosheds_level_func, hydrosheds_children, "PFAF_ID")
    calculate_total_area(stats_df)
    output_by_level(bottom_level, stats_df, hydrosheds_level_func, Path("output/worldcover-stats-sheds.fgb"))

    print("Done")


def hydrosheds_level_func(stats_df: DataFrame, level: int) -> Series:
    return Series(stats_df["PFAF_ID"].astype("str").str.len() == level + 1)


def hydrosheds_intersections(ds_bbox: box, stats_df: DataFrame, level: int):
    return stats_df[stats_df.geometry.intersects(ds_bbox) & hydrosheds_level_func(stats_df, level - 1)]


def hydrosheds_children(stats_df: DataFrame, pfaf_id: int) -> Series:
    """
    Returns a boolean Series indicating whether the PFAF_ID is a child of the given PFAF_ID.
    """

    child_lower = pfaf_id * 10
    child_upper = child_lower + 9
    return Series((stats_df["PFAF_ID"] >= child_lower) & (stats_df["PFAF_ID"] <= child_upper))


def merge_datasets(root_path: Path) -> DataFrame:
    files = root_path.glob("*.shp")

    return pd.concat([gpd.read_file(file.resolve(), engine="pyogrio") for file in files], ignore_index=True)


def main():
    create_directories()
    geom_df = merge_datasets(Path("/home/merseyviking/Documents/Sparkgeo/Projects/watershed/hybas_eu_lev01-12_v1c/"))
    keys = get_tile_keys(geom_df, 2, hydrosheds_level_func)
    process(keys, geom_df, 12)


if __name__ == "__main__":
    main()
