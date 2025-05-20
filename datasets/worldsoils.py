# https://gui.world-soils.com/mapserver/Europe?SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage&COVERAGEID=soc_0-5cm_mean_europe_2020-2022&FORMAT=image/tiff&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326&SUBSET=long(8,10)&SUBSET=lat(50,52)&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326
from collections.abc import Callable

from pandas import DataFrame, Series
import geopandas as gpd

type LevelFunc = Callable[[DataFrame, int], Series]


def get_tile_keys(geom_df: DataFrame, level: int, level_fn: LevelFunc) -> list[str]:
    countries = geom_df[level_fn(geom_df, level)]

    tile_index_df = gpd.read_file("input/esa_worldcover_grid.fgb", engine="pyogrio")
    intersected_tiles = gpd.sjoin(tile_index_df, countries, how="inner")
    unique_tiles = intersected_tiles.drop_duplicates(subset="ll_tile").copy()

    tile_names = unique_tiles["ll_tile"].tolist()
    tile_keys = [f"v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile_name.strip()}_Map.tif" for tile_name in tile_names]

    unique_tiles["tile_key"] = tile_keys

    unique_tiles = unique_tiles[["ll_tile", "tile_key", "geometry"]]

    unique_tiles.to_file("output/worldcover-tiles.fgb")

    return tile_keys
