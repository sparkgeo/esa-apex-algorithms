from pathlib import Path

import geopandas as gpd
from pandas import DataFrame, Series
from shapely import box

from datasets.utils import create_directories
from datasets.worldcover import (
    get_tile_keys,
    process,
    land_cover_names,
)
from inject_metadata import inject_metadata

watersheds_worldcover_metadata = {
    "identifierKey": "HYBAS_ID",
    "nameKey": "HYBAS_ID",
    "levelKey": "level",
    "childrenKey": "children",
    "attributeKeys": land_cover_names
}


def hydrosheds_level_func(stats_df: DataFrame, level: int) -> Series:
    return Series(stats_df["PFAF_ID"].astype("str").str.len() == level + 1)


def hydrosheds_intersections(ds_bbox: box, stats_df: DataFrame, level: int) -> DataFrame:
    return stats_df[stats_df.geometry.intersects(ds_bbox) & hydrosheds_level_func(stats_df, level - 1)]


def hydrosheds_children(stats_df: DataFrame, pfaf_id: int) -> Series:
    """
    Returns a boolean Series indicating whether the PFAF_ID is a child of the given PFAF_ID.
    """

    child_lower = pfaf_id * 10
    child_upper = child_lower + 9
    return Series((stats_df["PFAF_ID"] >= child_lower) & (stats_df["PFAF_ID"] <= child_upper))


def main():
    create_directories()
    geom_df = gpd.read_file("input/hybas_eu_lev01-12_v1c.fgb", engine="pyogrio")
    keys = get_tile_keys(geom_df, 2, hydrosheds_level_func)
    file_names = process(
        keys,
        geom_df,
        12,
        hydrosheds_level_func,
        hydrosheds_children,
        hydrosheds_intersections,
        "PFAF_ID",
        Path("output/worldcover-stats-watersheds.fgb")
    )

    for file_name in file_names:
        inject_metadata(file_name, watersheds_worldcover_metadata)


if __name__ == "__main__":
    main()
