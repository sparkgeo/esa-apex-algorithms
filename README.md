# This project has been archived!

Better and more up-to-date repositories can be found here:

- https://github.com/sparkgeo/apex-convert-boundaries
- https://github.com/sparkgeo/apex-create-constraints
- https://github.com/sparkgeo/apex-build-statistics

# APEX statistics preprocessor

This repository contains algorithms developed by Sparkgeo for the European Space Agency's APEX project.
The project focuses on processing and analyzing geospatial data to support environmental monitoring and assessment.

## Summary of operation

The application takes two datasets - a heirarchical series of polygons, and a raster dataset. It calculates statistics
for each of the lowest-level polygons based on the raster dataset. The statistics are then calculated for each parent
based on the values of their children. For example we can take the
[NUTS](https://ec.europa.eu/eurostat/web/nuts) (Nomenclature of Units for Territorial Statistics) polygons and
calculate the total land use for each class from the [ESA WorldCover](https://esa-worldcover.org/en) land use classes.
The statistics are written to a polygon dataset in [FlatGeobuf](https://flatgeobuf.org/) format. There is one file per
level of polygon data to make rendering more efficient.
It iterates over each raster tile and calculates the possibly partial statistics for each covering polygon. This
reduces the memory footprint by only ever considering one raster tile at a time then discarding it. This allows very
large polygons to be used without generating massive composite rasters.

## Vector polygon datasets

The code expects two columns to be present in the input polygon dataset: 1) an integer representing the level in
a heirarchy, with level 0 as the top level; and 2) a comma-separated string of IDs that reference the children (if any)
of that polygon. In the current iteration of the code, this is assumed to be present in the data already, but an
interface will be provided to allow developers to write a function that genreates them before processing.

## Raster datasets

Raster data must have a CRS of EPSG:4326, but can be of any resolution or size, and can have nodata values. As each
dataset is different, speicific processing functions need to be created to obtain the necessary values for calculation.
In the examples here, the WorldCover dataset is downloaded from an S3 bucket based on whether the bounding box of the
input polygons intersects the raster tile's bounds.

Raster data is downloaded and stored in a local cache folder, but because each raster is used only once, this behavior
can be turned off to keep disk usage to a minimum.

## Metadata

To be rendered by the front end, some metadata needs to appear in the output files. This must conform to the following
JSON schema:

```json
{
  "properties": {
    "identifierKey": {
      "type": "string"
    },
    "nameKey": {
      "type": "string"
    },
    "levelKey": {
      "type": "string"
    },
    "childrenKey": {
      "type": "string"
    },
    "attributeKeys": {
      "items": {
        "type": "string"
      },
      "type": "array"
    }
  },
  "required": [
    "identifierKey",
    "nameKey",
    "levelKey",
    "childrenKey",
    "attributeKeys"
  ],
  "title": "MetadataIn",
  "type": "object"
}
```

Where:
- `identifierKey` is the attribute that is the unique identifier for each polygon.
- `nameKey` is the attribute that is the human-readable name for each polygon.
- `levelKey` is the attribute that describes which level of the hierarchy this polygon is at.
- `childrenKey` is the attribute that lists any children of a polygon as a comma-separated string of `identifierKey`s.
- `attributeKeys` is a list of attributes that are used to display on the front end as a graph or table.
