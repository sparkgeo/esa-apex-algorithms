from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Geod
from rasterio import features
from shapely import box
from tqdm import trange

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

def main(tif_file_names: list[str], geometry_path: Path):
    geom_df = gpd.read_file(geometry_path.resolve(), engine="pyogrio")
    stats_df = pd.DataFrame(0.0, index=range(len(geom_df)), columns=land_cover_names, dtype=float)
    stats_df = pd.concat([geom_df, stats_df], axis=1)

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    tiff_progress_bar = trange(len(tif_file_names))
    for idx in tiff_progress_bar:
        src_file_name = Path("./.cache/" + Path(tif_file_names[idx]).name)

        if not src_file_name.exists():
            s3.download_file(s3_bucket, tif_file_names[idx], src_file_name)

        ds = rasterio.open(src_file_name)
        ds_bbox = box(*ds.bounds)

        result = stats_df[
            (stats_df.geometry.intersects(ds_bbox)) & (stats_df["LEVL_CODE"] == 3)
        ]

        if len(result) == 0:
            ds.close()
            continue

        geometry_progress_bar = trange(
            len(result), leave=False, desc=src_file_name.name
        )
        # TODO: Parallelize
        for i in geometry_progress_bar:
            region = result.iloc[i]
            geometry_progress_bar.set_postfix_str(region["NUTS_ID"])
            geom = region.geometry
            window = features.geometry_window(ds, [geom])
            window_xform = ds.window_transform(window)
            data = ds.read(window=window)
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
            stats_df.loc[region.name, land_cover_names] += values

        ds.close()

    for index, nut in stats_df[stats_df["LEVL_CODE"] == 2].iterrows():
        code = nut["NUTS_ID"]
        children = stats_df[(stats_df["NUTS_ID"].str.startswith(code)) & (stats_df["LEVL_CODE"] == 3)]
        stats_df.loc[index, land_cover_names] = children[land_cover_names].sum()

    for index, nut in stats_df[stats_df["LEVL_CODE"] == 1].iterrows():
        code = nut["NUTS_ID"]
        children = stats_df[(stats_df["NUTS_ID"].str.startswith(code)) & (stats_df["LEVL_CODE"] == 2)]
        stats_df.loc[index, land_cover_names] = children[land_cover_names].sum()

    for index, nut in stats_df[stats_df["LEVL_CODE"] == 0].iterrows():
        code = nut["NUTS_ID"]
        children = stats_df[(stats_df["NUTS_ID"].str.startswith(code)) & (stats_df["LEVL_CODE"] == 1)]
        stats_df.loc[index, land_cover_names] = children[land_cover_names].sum()

    stats_df.to_file("output/worldcover-stats.fgb")


def get_tile_keys(geometry_path: Path):
    geom_df = gpd.read_file(geometry_path.resolve(), engine="pyogrio")
    countries = geom_df[(geom_df["LEVL_CODE"] == 0)]
    tile_index_df = gpd.read_file("data/esa_worldcover_grid.fgb", engine="pyogrio")
    intersected_tiles = gpd.sjoin(tile_index_df, countries, how="inner")
    unique_tiles = intersected_tiles.drop_duplicates(subset="ll_tile")
    tile_names = unique_tiles["ll_tile"].tolist()
    tile_keys = [f"v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile_name.strip()}_Map.tif" for tile_name in tile_names]

    return tile_keys


if __name__ == "__main__":
    # TODO: Command line options.
    # TODO: Create directories.
    # TODO: Write comments.
    main(get_tile_keys(Path("data/NUTS_with_children.fgb")), Path("data/NUTS_with_children.fgb"))
