"""
Interface functions for processing statistical raster data using
HydroBASINS https://www.hydrosheds.org/products/hydrobasins polygons.
"""

from pathlib import Path

import geopandas as gpd
from pandas import DataFrame, Series
from shapely import box

from datasets.utils import create_directories
from inject_metadata import inject_metadata, MetadataIn


metadata = MetadataIn(identifierKey="HYBAS_ID", nameKey="HYBAS_ID", levelKey="level", childrenKey="children", attributeKeys=[])


def hydrosheds_level_func(df: DataFrame, level: int) -> Series:
    """
    Filters the input DataFrame based on a specified hierarchy level.
    This function evaluates whether the 'PFAF_ID' column of the input DataFrame
    matches the provided level and returns the resulting boolean Series.

    Conforms to :py:type:`datasets.utils.LevelFunc`.

    :param df:
        The input DataFrame.
    :param level:
        The specific hierarchy level to filter by.
    :return:
        A boolean Series where each entry is True if the corresponding level in the
        DataFrame matches the given level, otherwise False.
    """
    return Series(df["PFAF_ID"].astype("str").str.len() == level + 1)


def hydrosheds_intersections(bbox: box, df: DataFrame, level: int) -> DataFrame:
    """
    Determines the intersections of a bounding box with features in a GeoDataFrame
    that match a specific hydroshed level.

    This function applies spatial filtering on a GeoDataFrame to identify features
    (overlapping with a provided bounding box) that also satisfy a specific hydroshed
    level. The identified features are returned as a new GeoDataFrame.

    :param bbox:
        The geometric bounding box used for spatial filtering.
    :param df:
        The GeoDataFrame containing spatial features to be filtered.
    :param level:
        Hydroshed level used for filtering the GeoDataFrame.

    :return:
        A new GeoDataFrame containing features that intersect the bounding box
        and satisfy the hydroshed level criteria.
    """
    return df[df.geometry.intersects(bbox) & hydrosheds_level_func(df, level - 1)]


def hydrosheds_children(df: DataFrame, pfaf_id: int) -> Series:
    """
    Returns a boolean Series indicating whether the PFAF_ID is a child of the given PFAF_ID.
    """

    child_lower = pfaf_id * 10
    child_upper = child_lower + 9
    return Series((df["PFAF_ID"] >= child_lower) & (df["PFAF_ID"] <= child_upper))


def process_worldcover():
    from datasets.worldcover import get_tile_keys, process, attribute_keys

    metadata.attributeKeys = attribute_keys()

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
        inject_metadata(file_name, metadata)


def process_worldsoils():
    from datasets.worldsoils import get_tile_keys, process, coverage_id, attribute_keys

    metadata.attributeKeys = attribute_keys()

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
        Path(f"output/{coverage_id}-watersheds.fgb")
    )

    for file_name in file_names:
        inject_metadata(file_name, metadata)


if __name__ == "__main__":
    process_worldsoils()
