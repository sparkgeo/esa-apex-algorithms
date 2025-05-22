"""
Interface functions for processing statistical raster data using
NUTS (Nomenclature of Units for Territorial Statistics) polygons.
"""

from pathlib import Path

import geopandas as gpd
from pandas import DataFrame, Series
from shapely import box

from datasets.utils import create_directories
from datasets.worldsoils import coverage_id
from inject_metadata import inject_metadata, MetadataIn


def nuts_level_func(df: DataFrame, level: int) -> Series:
    """
    Filters the input DataFrame based on a specified NUTS level.
    This function evaluates whether the 'LEVL_CODE' column of the input DataFrame
    matches the provided level and returns the resulting boolean Series.

    Conforms to :py:type:`datasets.utils.LevelFunc`.

    :param df:
        The input DataFrame containing at least a 'LEVL_CODE' column,
        which specifies the NUTS level classification for each row.
    :param level:
        The specific NUTS level to filter by. The function checks if rows in 'LEVL_CODE'
        match this parameter.
    :return:
        A boolean Series where each entry is True if the corresponding 'LEVL_CODE' in the
        DataFrame matches the given level, otherwise False.
    """
    return Series(df["LEVL_CODE"] == level)


def nuts_children(df: DataFrame, nuts_id: str) -> Series:
    """
    Fetches the child regions of a given NUTS region by its identifier and returns
    a boolean Series indicating whether each region in the DataFrame is a child
    of the specified NUTS region.

    Conforms to :py:type:`datasets.utils.ChildFunc`.


    :param df:
        A pandas DataFrame containing NUTS data, which must include a "NUTS_ID"
        column representing the identifiers of regions and a "children" column
        with a string listing child identifiers, separated by commas.
    :param nuts_id:
        A string representing the NUTS Identifier of the region whose children
        are being queried.
    :return:
        A pandas Series containing boolean values where each entry denotes
        whether the corresponding region in the input DataFrame is a child
        of the specified NUTS Identifier.
    """
    children = df.loc[df["NUTS_ID"] == nuts_id, "children"].iloc[0].split(",")

    return Series(df["NUTS_ID"].isin(children))


def nuts_intersections(bbox: box, df: DataFrame, level: int) -> DataFrame:
    """
    Filters a DataFrame to find rows where the geometry intersects with a given
    bounding box (bbox) and the row has no children. The function operates
    on a specified level.

    Conforms to :py:type:`datasets.utils.IntersectionFunc`.

    :param bbox:
        The bounding box (bbox) to compute spatial intersections with.
    :param df:
        A DataFrame containing spatial geometries.
    :param level:
        An integer indicating the desired level of operation.
    :return:
        A filtered DataFrame with rows satisfying the intersection
        and level conditions.
    """
    return df[
        (df.geometry.intersects(bbox)) & (df["children"] == "")
        ]

metadata = MetadataIn(identifierKey="NUTS_ID", nameKey="NUTS_NAME", levelKey="LEVL_CODE", childrenKey="children", attributeKeys=[])

def process_worldcover():
    """
    Processes geospatial data using the WorldCover dataset utilities to compute
    and inject metadata statistics for specified NUTS regions and their related
    hierarchies. This function performs multiple tasks including reading input
    data, processing geospatial keys, and saving outputs while appending metadata.
    """
    from datasets.worldcover import get_tile_keys, process, attribute_keys

    metadata.attributeKeys = attribute_keys()

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
        inject_metadata(file_name, metadata)


def process_worldsoils():
    """
    Executes the main process for handling world soils data by processing spatial datasets,
    generating tile keys, and injecting metadata into processed files. This function serves
    as the central workflow for managing and transforming input geospatial data and outputs.

    The function reads input geometry data, processes spatial tiles based on specified
    parameters, and translates them into output files enriched with metadata.
    """
    from datasets.worldsoils import get_tile_keys, process, attribute_keys

    metadata.attributeKeys = attribute_keys()

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
        Path(f"output/{coverage_id}-nuts.fgb")
    )

    for file_name in file_names:
        inject_metadata(file_name, metadata)


if __name__ == "__main__":
    process_worldsoils()
