"""
Calculates the coverage of each land cover class in a polygon in square km. Also adds a total area
so that proportions can be calculated.
"""
from pathlib import Path

from joblib import Parallel, delayed

from pandas import DataFrame
import requests
from rasterio.errors import WindowError
from tqdm import trange, tqdm
from rasterio import features, DatasetReader
import numpy as np
from shapely import box
import humanize
import rasterio

from datasets.utils import (
    LevelFunc,
    ChildFunc,
    IntersectionFunc,
    output_by_level,
    get_process_memory_use,
    checkpoint_memory_usage,
    max_memory_usage,
)

HABITAT_CLASSES = {
    20000: "Costal Habitats",
    30000: "Inland Surface Waters",
    40000: "Mires, bogs and fens",
    50000: "Grasslands and lands dominated by forbs, mosses or lichens",
    60000: "Heathland, scrub and tundra",
    70000: "Woodland, forest and other wooded land",
    80000: "Inland unvegetated or sparsely vegetated habitats",
    90000: "Regularly or recently cultivated agricultural, horticultural and domestic habitats",
    100000: "Constructed, industrial and other artificial habitats",
    110000: "Complex Habitats",
}

# Numpy's histogram bins are half-open except the last bin, so we need to add a right edge to the last bin.
# This can be anything > 110000, but 110001 seems safe enough if new categories get added to the end but we haven't updated the code.
HABITAT_CLASSES_BINS = list(HABITAT_CLASSES.keys())
HABITAT_CLASSES_BINS.append(110001)
habitat_names = list(HABITAT_CLASSES.values())
server_url = "https://eoresults.esa.int"


def attribute_keys() -> list[str]:
    """
    Returns the names of the columns that we want to render in the front end.
    """
    return habitat_names + ["Unknown"]


def _compute_area(row):
    total_area = row.geometry.area / 1000000.0
    covered_area = row[habitat_names].sum()
    unknown = max(0.0, total_area - covered_area)

    return total_area, unknown


def calculate_total_area(df: DataFrame) -> None:
    """
    Calculates the total area of each polygon in square km and adds it to the dataframe.
    """

    results = Parallel(n_jobs=-1)(
        delayed(_compute_area)(row)
        for _, row in tqdm(
            df.iterrows(), total=df.shape[0], desc="Calculating total area"
        )
    )

    # Unzip results
    total_areas, unknowns = zip(*results)

    df["total_area"] = total_areas
    df["Unknown"] = unknowns


def calculate_values(source_raster: DatasetReader, intersections: DataFrame) -> DataFrame:
    """
    Calculates the values for each land cover class that partially or wholly intersects the given polygon.
    """

    ds_bbox = box(*source_raster.bounds)
    src_file_name = Path(source_raster.name)

    geometry_progress_bar = trange(
        len(intersections), leave=False, desc=src_file_name.name
    )

    for i in geometry_progress_bar:
        region = intersections.iloc[i]
        checkpoint_memory_usage()
        current_memory_usage = get_process_memory_use()
        geometry_progress_bar.set_postfix_str(f"{humanize.naturalsize(current_memory_usage)}")
        geometry_progress_bar.set_description_str(f"{src_file_name.name}")
        geom = region.geometry

        try:
            window = features.geometry_window(source_raster, [geom])
        except WindowError:
            continue

        window_xform = source_raster.window_transform(window)
        data = source_raster.read(window=window)
        clipped_geom = geom.intersection(ds_bbox)
        checkpoint_memory_usage()

        if clipped_geom.is_empty:
            continue

        checkpoint_memory_usage()
        mask = features.geometry_mask(
            [clipped_geom], [window.height, window.width], window_xform, invert=True
        )
        checkpoint_memory_usage()
        values = np.histogram(data[0][mask], bins=HABITAT_CLASSES_BINS)[0]
        np_sum = np.sum(values)

        # Avoid divide by zero errors.
        if not np.isclose(0.0, np_sum, atol=1e-6):
            values = values * ((clipped_geom.area / 1000000.0) / np_sum)
            intersections.loc[region.name, habitat_names] += values

        checkpoint_memory_usage()

    return intersections



def get_tile_keys(df: DataFrame, level: int, level_fn: LevelFunc) -> list[str]:
    """
    Returns a list of raster tiles.
    """
    tile_keys = [server_url + "/d/ESA_PEOPLE_EA_HABITAT_MAPS_EUNIS_2021/2020/01/01/ESA_PEOPLE_EA_HABITAT_MAPS_EUNIS_2021-GR_2020/GR_L1_pp_2020.tif"]

    return tile_keys


def sum_children(df: DataFrame, bottom_level: int, level_fn: LevelFunc, child_fn: ChildFunc, code_column_name: str) -> None:
    """
    Iterates over each polygon from the bottom up, summing the values in the children polygons.
    """

    level_progress_bar = trange(bottom_level - 1, 0, -1, desc="Calculating parent statistics")

    for level in level_progress_bar:
        level_df = df[level_fn(df, level - 1)]
        for index, nut in tqdm(level_df.iterrows(), total=level_df.shape[0], leave=False):
            code = nut[code_column_name]
            children = df[child_fn(df, code)]

            if children.empty:
                continue

            df.loc[index, habitat_names] = children[habitat_names].sum()


def process(tif_file_names: list[str],
            df: DataFrame,
            bottom_level: int,
            level_fn: LevelFunc,
            child_fn: ChildFunc,
            intersection_fn: IntersectionFunc,
            code_column_name: str,
            output_path: Path
            ) -> list[Path]:
    """
    For each raster tile, calculate the values for each land cover class that partially or wholly intersects the given polygon.
    """

    for name in habitat_names:
        df[name] = 0.0

    df["Unknown"] = 0.0
    df["total_area"] = 0.0

    df = df.to_crs(crs="EPSG:3035")

    print("Processing")
    tiff_progress_bar = trange(len(tif_file_names))
    for idx in tiff_progress_bar:
        tiff_progress_bar.set_postfix_str(f"{humanize.naturalsize(max_memory_usage)}")
        src_file_name = Path("./.cache/" + Path(tif_file_names[idx]).name)
        tiff_progress_bar.set_description_str(f"{src_file_name.name}")

        if not src_file_name.exists():
            response = requests.get(tif_file_names[idx], verify=False, stream=True)
            if response.status_code == 200:
                with open(src_file_name, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                continue

        ds = rasterio.open(src_file_name)
        ds_bbox = box(*ds.bounds)

        intersections = intersection_fn(ds_bbox, df, bottom_level)

        if len(intersections) == 0:
            ds.close()
            continue

        results = calculate_values(ds, intersections)

        df.loc[results.index] = results

        ds.close()

    sum_children(df, bottom_level, level_fn, child_fn, code_column_name)
    calculate_total_area(df)
    df = df.to_crs(crs="EPSG:3857")

    file_names = output_by_level(bottom_level, df, level_fn, output_path)
    print("Done")

    return file_names
