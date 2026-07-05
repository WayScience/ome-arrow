"""Tests for Arrow-native OME-Arrow shape tables."""

import json
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ome_arrow import (
    OME_ARROW_SHAPES_METADATA_KEY,
    OMEArrowShapes,
    make_relationship_table,
    make_shape_table,
    read_shape_parquet,
    shape_schema,
    validate_relationship_table,
    validate_shape_table,
    write_shape_parquet,
)
from ome_arrow.shapes import SUPPORTED_GEOMETRY_ENCODINGS


def test_make_shape_table_stores_single_geometry_column_with_metadata() -> None:
    """Create a shape table with one logical geometry value per object row."""
    table = make_shape_table(
        [
            {
                "object_id": "cell-1",
                "image_id": "image-1",
                "label_image_id": "labels-1",
                "label_value": 7,
                "geometry": [[0.0, 0.0], [2.0, 0.0], [2.0, 3.0]],
                "centroid": [1.25, 1.5],
                "bbox": {"min": [0.0, 0.0], "max": [2.0, 3.0]},
                "class": "cell",
                "confidence": 0.98,
                "area": 5.5,
            }
        ],
        geometry_encoding="geoarrow.linestring",
        axes=("y", "x"),
        units=("pixel", "pixel"),
        coordinate_space="pixel",
    )

    metadata = json.loads(table.schema.metadata[OME_ARROW_SHAPES_METADATA_KEY])

    assert table.num_rows == 1
    assert table.column_names.count("geometry") == 1
    assert table.schema.field("geometry").type == pa.list_(pa.list_(pa.float64()))
    assert table.schema.field("area").type == pa.float64()
    assert metadata["type"] == "ome.arrow.shapes"
    assert metadata["geometry_encoding"] == "geoarrow.linestring"
    assert metadata["axes"] == ["y", "x"]
    assert metadata["coordinate_space"] == "pixel"
    assert validate_shape_table(table) is None


def test_shape_table_supports_labelmask_geometry_references() -> None:
    """Represent label masks by reference instead of embedding raster masks."""
    table = make_shape_table(
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

    geometry = table["geometry"].to_pylist()[0]

    assert geometry == {"label_image_id": "nuclear-labels", "label_value": 42}
    assert validate_shape_table(table) is None


def test_shape_table_rejects_invalid_geometry_encoding() -> None:
    """Fail before creating tables for unknown geometry encodings."""
    assert "geoarrow.polygon" in SUPPORTED_GEOMETRY_ENCODINGS

    with pytest.raises(ValueError, match="Unsupported geometry_encoding"):
        shape_schema("wkb")


def test_validate_shape_table_requires_metadata_and_identity() -> None:
    """Require shape metadata and object identity columns."""
    table = pa.table({"geometry": [[[0.0, 1.0]]]})

    with pytest.raises(ValueError, match="schema metadata"):
        validate_shape_table(table)

    table = make_shape_table(
        [{"geometry": [0.0, 1.0]}],
        geometry_encoding="geoarrow.point",
        validate=False,
    )

    with pytest.raises(ValueError, match="object_id"):
        validate_shape_table(table)


def test_ome_arrow_shapes_wrapper_exposes_metadata_and_filtering() -> None:
    """Wrap shape tables with small convenience accessors."""
    shapes = OMEArrowShapes.from_rows(
        [
            {
                "object_id": "cell-1",
                "image_id": "image-1",
                "label_image_id": "labels",
                "label_value": 1,
                "geometry": [0.0, 1.0],
            },
            {
                "object_id": "cell-2",
                "image_id": "image-2",
                "label_image_id": "labels",
                "label_value": 2,
                "geometry": [2.0, 3.0],
            },
        ],
        geometry_encoding="geoarrow.point",
    )

    subset = shapes.for_image("image-1")

    assert shapes.geometry_encoding == "geoarrow.point"
    assert shapes.axes == ("y", "x")
    assert subset.table["object_id"].to_pylist() == ["cell-1"]


def test_relationship_table_models_object_edges() -> None:
    """Build object relationship edges as ordinary Arrow rows."""
    table = make_relationship_table(
        [
            {
                "parent_id": "cell-1",
                "child_id": "nucleus-1",
                "relationship_type": "contains",
                "confidence": 1.0,
            }
        ]
    )

    assert table.schema.field("relationship_type").type == pa.string()
    assert table["parent_id"].to_pylist() == ["cell-1"]
    assert validate_relationship_table(table) is None


def test_shape_parquet_roundtrip_preserves_metadata(
    tmp_path: pathlib.Path,
) -> None:
    """Write and read complete shape tables with schema metadata intact."""
    path = tmp_path / "cells.ome-shapes.parquet"
    table = make_shape_table(
        [
            {
                "object_id": "cell-1",
                "image_id": "image-1",
                "label_image_id": "labels-1",
                "label_value": 1,
                "geometry": [10.0, 20.0],
                "area_um2": 42.5,
            }
        ],
        geometry_encoding="geoarrow.point",
        axes=("y", "x"),
        units=("micrometer", "micrometer"),
        coordinate_space="physical",
    )

    write_shape_parquet(table, path)
    roundtrip = read_shape_parquet(path)

    assert roundtrip.schema.metadata[OME_ARROW_SHAPES_METADATA_KEY]
    assert roundtrip["object_id"].to_pylist() == ["cell-1"]
    assert roundtrip["area_um2"].to_pylist() == [42.5]
    assert pq.ParquetFile(path).metadata.num_rows == 1


def test_shape_parquet_projection_and_filters_support_label_references(
    tmp_path: pathlib.Path,
) -> None:
    """Read only needed label-reference columns with Parquet predicate filters."""
    path = tmp_path / "label_refs.ome-shapes.parquet"
    table = make_shape_table(
        [
            {
                "object_id": "nucleus-1",
                "image_id": "image-1",
                "label_image_id": "nuclear-labels",
                "label_value": 1,
                "geometry": {
                    "label_image_id": "nuclear-labels",
                    "label_value": 1,
                },
                "class": "nucleus",
            },
            {
                "object_id": "nucleus-2",
                "image_id": "image-2",
                "label_image_id": "nuclear-labels",
                "label_value": 2,
                "geometry": {
                    "label_image_id": "nuclear-labels",
                    "label_value": 2,
                },
                "class": "nucleus",
            },
        ],
        geometry_encoding="ome.labelmask",
    )

    write_shape_parquet(table, path, row_group_size=1)
    projected = read_shape_parquet(
        path,
        columns=["object_id", "image_id", "label_value"],
        filters=[("image_id", "==", "image-2")],
    )

    assert projected.column_names == ["object_id", "image_id", "label_value"]
    assert projected.to_pydict() == {
        "object_id": ["nucleus-2"],
        "image_id": ["image-2"],
        "label_value": [2],
    }
