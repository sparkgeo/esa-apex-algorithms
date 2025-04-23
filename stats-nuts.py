from pathlib import Path
import re

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import geopandas as gpd
from icecream import ic
import numpy as np
import rasterio
from pandas import DataFrame
from pyproj import Geod
from rasterio import features, DatasetReader
from shapely import box
from tqdm import trange

ic.configureOutput(prefix="")

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
geod = Geod(ellps="WGS84")
s3_bucket = "esa-worldcover"

def process(tif_file_names: list[str], geometry_path: Path):
    ic("Loading boundaries")

    stats_df = gpd.read_file(geometry_path.resolve(), engine="pyogrio")

    for name in land_cover_names:
        stats_df[name] = 0.0

    stats_df["Unknown"] = 0.0
    stats_df["total_area"] = 0.0
    stats_df["classifications"] = ",".join(land_cover_names) + ",Unknown"
    fix_children(stats_df)

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    ic("Processing")
    tiff_progress_bar = trange(len(tif_file_names))
    for idx in tiff_progress_bar:
        src_file_name = Path("./.cache/" + Path(tif_file_names[idx]).name)

        if not src_file_name.exists():
            s3.download_file(s3_bucket, tif_file_names[idx], src_file_name)

        ds = rasterio.open(src_file_name)
        ds_bbox = box(*ds.bounds)

        intersections = stats_df[
            (stats_df.geometry.intersects(ds_bbox)) & (stats_df["children"] == "")
        ]

        if len(intersections) == 0:
            ds.close()
            continue

        results = calculate_values(ds, intersections)
        stats_df.update(results)

        ds.close()

    sum_children(stats_df)
    calculate_total_area(stats_df)

    ic("Saving results")
    stats_df.to_file("output/worldcover-nuts-stats.fgb")
    ic("Done")


def calculate_values(source_raster: DatasetReader, intersections: DataFrame) -> DataFrame:
    ds_bbox = box(*source_raster.bounds)
    src_file_name = Path(source_raster.name)

    geometry_progress_bar = trange(
        len(intersections), leave=False, desc=src_file_name.name
    )
    # TODO: Parallelize
    for i in geometry_progress_bar:
        region = intersections.iloc[i]
        geometry_progress_bar.set_postfix_str(region["NUTS_ID"])
        geom = region.geometry
        window = features.geometry_window(source_raster, [geom])
        window_xform = source_raster.window_transform(window)
        data = source_raster.read(window=window)
        clipped_geom = geom.intersection(ds_bbox)

        if clipped_geom.is_empty:
            continue

        clipped_geom_area = (
                abs(geod.geometry_area_perimeter(clipped_geom)[0]) / 1000000.0
        )
        mask = features.geometry_mask(
            [clipped_geom], [window.height, window.width], window_xform, invert=True
        )
        values = np.histogram(data[0][mask], bins=LAND_COVER_CLASSES_BINS)[0]
        values = values * (clipped_geom_area / np.sum(values))
        intersections.loc[region.name, land_cover_names] += values

    return intersections


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


def sum_children(stats_df: DataFrame) -> None:
    ic("Calculating parent statistics")

    for index, nut in stats_df[stats_df["LEVL_CODE"] == 2].iterrows():
        code = nut["NUTS_ID"]
        children = stats_df[
            (stats_df["NUTS_ID"].str.startswith(code)) & (stats_df["LEVL_CODE"] == 3)]

        if children.empty:
            continue

        stats_df.loc[index, land_cover_names] = children[land_cover_names].sum()
    for index, nut in stats_df[stats_df["LEVL_CODE"] == 1].iterrows():
        code = nut["NUTS_ID"]
        children = stats_df[
            (stats_df["NUTS_ID"].str.startswith(code)) & (stats_df["LEVL_CODE"] == 2)]

        if children.empty:
            continue

        stats_df.loc[index, land_cover_names] = children[land_cover_names].sum()
    for index, nut in stats_df[stats_df["LEVL_CODE"] == 0].iterrows():
        code = nut["NUTS_ID"]
        children = stats_df[
            (stats_df["NUTS_ID"].str.startswith(code)) & (stats_df["LEVL_CODE"] == 1)]

        if children.empty:
            continue

        stats_df.loc[index, land_cover_names] = children[land_cover_names].sum()


def calculate_total_area(statistics: DataFrame):
    ic("Calculating total area")
    areas = []
    unknowns = []

    for _, row in statistics.iterrows():
        total_area = abs(geod.geometry_area_perimeter(row.geometry)[0]) / 1000000.0
        covered_area = row[land_cover_names].sum()
        areas.append(total_area)
        unknowns.append(max(0.0, total_area - covered_area))

    statistics["total_area"] = areas
    statistics["Unknown"] = unknowns


def get_tile_keys(geometry_path: Path):
    geom_df = gpd.read_file(geometry_path.resolve(), engine="pyogrio")
    countries = geom_df[(geom_df["LEVL_CODE"] == 0)]
    tile_index_df = gpd.read_file("data/esa_worldcover_grid.fgb", engine="pyogrio")
    intersected_tiles = gpd.sjoin(tile_index_df, countries, how="inner")
    unique_tiles = intersected_tiles.drop_duplicates(subset="ll_tile").copy()

    tile_names = unique_tiles["ll_tile"].tolist()
    tile_keys = [f"v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile_name.strip()}_Map.tif" for tile_name in tile_names]

    unique_tiles["tile_key"] = tile_keys

    unique_tiles = unique_tiles[["ll_tile", "tile_key", "geometry"]]

    unique_tiles.to_file("output/worldcover-tiles.fgb")

    return tile_keys



def create_directories() -> tuple[Path, Path]:
    """
    Creates necessary paths for output and caching.
    """

    if not Path("output").exists():
        Path("output").mkdir()

    if not Path(".cache").exists():
        Path(".cache").mkdir()

    return Path("output").resolve(), Path(".cache").resolve()


def main():
    # TODO: Command line options.
    # TODO: Write comments.
    geometry_path = Path("data/NUTS_with_children.fgb")
    keys = get_tile_keys(geometry_path)
    process(keys, geometry_path)


if __name__ == "__main__":
    main()
