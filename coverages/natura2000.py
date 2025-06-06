"""
Interface functions for processing statistical raster data using
Natura 2000 https://www.eea.europa.eu/themes/biodiversity/natura-2000/the-natura-2000-protected-areas-network polygons.
"""

from pathlib import Path

import geopandas as gpd
from pandas import DataFrame, Series
from shapely import box

from datasets.utils import create_directories
from inject_metadata import inject_metadata, MetadataIn


metadata = MetadataIn(identifierKey="SITECODE", nameKey="SITENAME", levelKey="level", childrenKey="children", attributeKeys=[])


def natura_level_func(df: DataFrame, level: int) -> Series:
    """
    Filters the input DataFrame based on a specified hierarchy level.

    Conforms to :py:type:`datasets.utils.LevelFunc`.

    :param df:
        The input DataFrame.
    :param level:
        The specific hierarchy level to filter by.
    :return:
        A boolean Series where each entry is True if the corresponding level in the
        DataFrame matches the given level, otherwise False.
    """

    if level == 0:
        return Series(True, index=df.index)
    else:
        return Series(False, index=df.index)


def natura_intersections(bbox: box, df: DataFrame, level: int) -> DataFrame:
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
    return df[df.geometry.intersects(bbox)]


def natura_children(df: DataFrame, unused_id: int) -> Series:
    """
    Returns a boolean Series. Natura 2000 dataset is one level, so return false
    which means no children.
    """

    return Series(False, index=df.index)


def natura_create_columns(df: DataFrame) -> DataFrame:
    df["level"] = 0
    df["children"] = ""
    return df


def process_worldcover():
    from datasets.worldcover import get_tile_keys, process, attribute_keys

    metadata.attributeKeys = attribute_keys()

    create_directories()
    geom_df = gpd.read_file("input/Natura2000_end2023.gpkg", layer="NaturaSite_polygon", engine="pyogrio")
    geom_df.to_crs("EPSG:4326", inplace=True)
    natura_create_columns(geom_df)
    keys = get_tile_keys(geom_df, 0, natura_level_func)
    file_names = process(
        keys,
        geom_df,
        1,
        natura_level_func,
        natura_children,
        natura_intersections,
        "SITECODE",
        Path("output/worldcover-stats-natura.fgb")
    )

    for file_name in file_names:
        inject_metadata(file_name, metadata)


def process_worldsoils():
    from datasets.worldsoils import get_tile_keys, process, coverage_id, attribute_keys

    metadata.attributeKeys = attribute_keys()

    create_directories()
    geom_df = gpd.read_file("input/Natura2000_end2023.gpkg", layer="NaturaSite_polygon", engine="pyogrio")
    geom_df.to_crs("EPSG:4326", inplace=True)
    natura_create_columns(geom_df)
    keys = get_tile_keys(geom_df, 0, natura_level_func)
    file_names = process(
        keys,
        geom_df,
        1,
        natura_level_func,
        natura_children,
        natura_intersections,
        "SITECODE",
        Path(f"output/{coverage_id}-natura.fgb")
    )

    for file_name in file_names:
        inject_metadata(file_name, metadata)


if __name__ == "__main__":
    process_worldcover()
