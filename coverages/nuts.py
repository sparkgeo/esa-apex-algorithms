from pathlib import Path

import geopandas as gpd
from pandas import DataFrame, Series
from shapely import box

from datasets.worldcover import (
    create_directories,
    get_tile_keys,
    process,
    land_cover_names
)
from inject_metadata import inject_metadata

nuts_worldcover_metadata = {
    "identifierKey": "NUTS_ID",
    "nameKey": "NUTS_NAME",
    "levelKey": "LEVL_CODE",
    "childrenKey": "children",
    "attributeKeys": land_cover_names
}


def nuts_level_func(stats_df: DataFrame, level: int) -> Series:
    return Series(stats_df["LEVL_CODE"] == level)


def foo(row, df):
    if not row["children"]:
        return []

    return row["children"].split(",")


def nuts_children(df: DataFrame, nuts_id: str) -> Series:
    children = df.loc[df["NUTS_ID"] == nuts_id, "children"].iloc[0].split(",")

    return Series(df["NUTS_ID"].isin(children))


def nuts_intersections(ds_bbox: box, stats_df: DataFrame, level: int):
    return stats_df[
            (stats_df.geometry.intersects(ds_bbox)) & (stats_df["children"] == "")
        ]


def main():
    create_directories()
    geom_df = gpd.read_file("input/NUTS_with_children.fgb", engine="pyogrio")
    keys = get_tile_keys(geom_df, 0, nuts_level_func)
    file_names = process(
        keys,
        geom_df,
        4,
        nuts_level_func,
        nuts_children,
        nuts_intersections,
        "NUTS_ID",
        Path("output/worldcover-stats-nuts.fgb")
    )

    for file_name in file_names:
        inject_metadata(file_name, nuts_worldcover_metadata)


if __name__ == "__main__":
    main()
