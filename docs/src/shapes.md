# OME-Arrow Shapes

OME-Arrow Shapes is an Arrow-native convention for bioimaging object tables.
It is intended to complement OME-Zarr and OME-NGFF, not replace them.

The core model is:

- one row is one biological object
- geometry is one logical column
- measurements are ordinary Arrow columns
- dense label rasters remain canonical OME-Zarr labels
- shape rows reference labels instead of duplicating mask payloads
- coordinate metadata follows OME axis, unit, and coordinate-space concepts

## The Gap

OME has strong, mature conventions for image pixels and label images.
OME-Zarr and OME-NGFF are a natural fit for dense multiscale arrays, including
segmentation rasters.

Object tables sit in a different access pattern.
Common bioimage analysis needs to query, filter, join, and aggregate objects:

- load all cell centroids
- filter detections by class or confidence
- join cells to nuclei
- compute population summaries
- associate each object with a source image and label value
- use DuckDB, Polars, DataFusion, or PyArrow without unpacking image arrays

Those workflows are columnar and relational.
They benefit from Arrow and Parquet, but still need OME-compatible semantics for
axes, units, labels, provenance, and relationships.

OME-Arrow Shapes fills that table-shaped gap.

## Alignment With OME

OME-Arrow Shapes stays with OME's existing direction in several ways.

Images stay dense.
Pixels should remain in OME-Zarr, OME-TIFF, or OME-Arrow image storage.

Labels stay dense.
Segmentation masks should remain canonical label rasters.
Shape rows can reference label objects with `label_image_id` and `label_value`.

Coordinates stay explicit.
Shape schemas store axes, units, coordinate space, and geometry encoding in Arrow
schema metadata.
This keeps table values interpretable without inventing a separate coordinate
system model.

Measurements stay columnar.
Area, volume, intensity, texture, model scores, and morphology features are plain
Arrow columns.
No special nested measurement encoding is required.

## Shape Schema

Use `make_shape_table` to create a validated Arrow table.

```python
from ome_arrow import make_shape_table

shapes = make_shape_table(
    [
        {
            "object_id": "cell-1",
            "image_id": "image-1",
            "label_image_id": "labels-1",
            "label_value": 7,
            "geometry": [128.0, 256.0],
            "centroid": [128.0, 256.0],
            "class": "cell",
            "confidence": 0.98,
            "area_um2": 84.2,
        }
    ],
    geometry_encoding="geoarrow.point",
    axes=("y", "x"),
    units=("pixel", "pixel"),
    coordinate_space="pixel",
)
```

The canonical columns are:

| Column | Purpose |
| --- | --- |
| `object_id` | stable object identifier |
| `image_id` | source image identifier |
| `label_image_id` | source label image identifier |
| `label_value` | integer label value in the label raster |
| `geometry` | one logical geometry value |
| `centroid` | coordinate vector for object center |
| `bbox` | min/max coordinate bounds |
| `class` | object class or annotation category |
| `confidence` | detection or classification confidence |

Any additional columns are measurements.
They remain ordinary Arrow columns and can be queried directly.

## Geometry Encodings

The current registry includes:

| Encoding | Storage intent |
| --- | --- |
| `geoarrow.point` | point coordinate vector |
| `geoarrow.linestring` | list of coordinate vectors |
| `geoarrow.polygon` | list of rings |
| `geoarrow.multipolygon` | list of polygons |
| `ome.labelmask` | label image reference |
| `ome.pointcloud` | list of coordinate vectors |
| `ome.boundingbox` | min/max coordinate vectors |
| `ome.mesh3d` | vertices and faces |

The registry lets 2D shapes reuse GeoArrow-like nested Arrow layouts while
leaving room for bioimaging-specific geometry such as label masks, point clouds,
and meshes.

## Label References

For segmentation objects, prefer references over embedded masks.

```python
from ome_arrow import make_shape_table

objects = make_shape_table(
    [
        {
            "object_id": "nucleus-1",
            "image_id": "image-1",
            "label_image_id": "nuclear-labels",
            "label_value": 42,
            "geometry": {
                "label_image_id": "nuclear-labels",
                "label_value": 42,
            },
            "class": "nucleus",
        }
    ],
    geometry_encoding="ome.labelmask",
)
```

This avoids duplicating raster masks in every object row.
The label image remains the canonical segmentation, and the object table remains
small, queryable, and provenance-preserving.

## Relationships

Object relationships are also ordinary Arrow rows.

```python
from ome_arrow import make_relationship_table

relationships = make_relationship_table(
    [
        {
            "parent_id": "cell-1",
            "child_id": "nucleus-1",
            "relationship_type": "contains",
            "confidence": 1.0,
        }
    ]
)
```

Supported relationship types are:

- `contains`
- `adjacent`
- `touches`
- `parent`
- `track`
- `derived_from`

## Performance Contract

Shape tables are built on Arrow-native arrays and schema metadata.
The implementation avoids binary geometry blobs where nested Arrow types are
practical, which keeps column projection, filtering, and Parquet serialization
available to Arrow-compatible engines.

The test suite includes pytest performance canaries for shape table construction
and image-id filtering.
They are intentionally lightweight regression checks rather than rigorous
hardware-specific benchmarks.

Run them with:

```sh
python -m pytest tests/test_shapes.py tests/test_shapes_performance.py
```

For deeper benchmark work, add workload-specific scripts under `benchmarks/`
that report medians over repeated runs and compare against checked-in baselines.
