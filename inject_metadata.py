import json
from pathlib import Path

from osgeo import ogr
from pydantic import BaseModel, field_serializer


class MissingColumn(Exception):
    pass


class MetadataIn(BaseModel):
    identifierKey: str
    nameKey: str
    levelKey: str
    childrenKey: str
    attributeKeys: list[str]


class MetadataOut(BaseModel):
    identifierKey: str
    nameKey: str
    levelKey: str
    childrenKey: str
    attributeKeys: list[int]

    # FGB metadata fields can only be strings.
    @field_serializer("attributeKeys")
    def serialize_attribute_keys(self, keys, _):
        return ",".join([str(v) for v in keys])


def inject_metadata(fgb_file_path: Path, metadata: MetadataIn):
    if not fgb_file_path.exists():
        raise FileNotFoundError(f"Input file '{fgb_file_path}' not found")
    if fgb_file_path.suffix != ".fgb":
        raise ValueError(f"Input file '{fgb_file_path}' is not a FlatGeobuf file")

    src_ds = ogr.Open(fgb_file_path)
    src_layer = src_ds.GetLayer()

    # Check columns exist in the input file.
    defn = src_layer.GetLayerDefn()

    if defn.GetFieldIndex(metadata.identifierKey) == -1:
        raise MissingColumn(f"Index column '{metadata.identifierKey}' not found in {fgb_file_path}")

    if defn.GetFieldIndex(metadata.nameKey) == -1:
        raise MissingColumn(f"Name column '{metadata.nameKey}' not found in {fgb_file_path}")

    if defn.GetFieldIndex(metadata.levelKey) == -1:
        raise MissingColumn(f"Level column '{metadata.levelKey}' not found in {fgb_file_path}")

    if defn.GetFieldIndex(metadata.childrenKey) == -1:
        raise MissingColumn(f"Level column '{metadata.childrenKey}' not found in {fgb_file_path}")

    attribute_column_indices = []

    for column in metadata.attributeKeys:
        column_index = defn.GetFieldIndex(column)
        if column_index == -1:
            raise MissingColumn(f"Column '{column}' not found in {fgb_file_path}")
        else:
            attribute_column_indices.append(column_index)

    metadata_out = MetadataOut(
        identifierKey=metadata.identifierKey,
        nameKey=metadata.nameKey,
        levelKey=metadata.levelKey,
        attributeKeys=attribute_column_indices,
        childrenKey=metadata.childrenKey
    )

    # Overwrite the input file.
    driver = ogr.GetDriverByName("FlatGeobuf")
    dst_ds = driver.CreateDataSource(fgb_file_path)
    dst_ds.CopyLayer(src_layer, src_layer.GetName())

    print(f"Metadata injected into {fgb_file_path}")
    print(f"{metadata_out}")

    # We serialize the metadata to a string to take advantage of Pydantic's field serializer,
    # then load it back up as a regular Python object so OGR can set the metadata.
    dst_ds.SetMetadata(json.loads(metadata_out.model_dump_json()))

    dst_ds = None
    src_ds = None
