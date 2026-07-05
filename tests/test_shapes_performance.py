"""Performance canaries for Arrow-native OME-Arrow shape tables."""

from __future__ import annotations

import time

from ome_arrow import OMEArrowShapes, make_shape_table


def _shape_rows(count: int) -> list[dict[str, object]]:
    """Create deterministic point rows for shape performance tests."""
    return [
        {
            "object_id": f"cell-{i}",
            "image_id": f"image-{i % 5}",
            "label_image_id": "labels",
            "label_value": i,
            "geometry": [float(i), float(i + 1)],
            "centroid": [float(i), float(i + 1)],
            "class": "cell",
            "area": float(i % 997),
            "mean_intensity": float(i % 255),
        }
        for i in range(count)
    ]


def test_make_shape_table_handles_large_object_tables_quickly() -> None:
    """Keep shape table construction on Arrow's fast path."""
    rows = _shape_rows(20_000)

    start = time.perf_counter()
    table = make_shape_table(rows, geometry_encoding="geoarrow.point")
    elapsed = time.perf_counter() - start

    assert table.num_rows == 20_000
    assert table.schema.field("area").type.bit_width == 64
    assert elapsed < 1.0


def test_shape_filtering_stays_vectorized_for_image_ids() -> None:
    """Keep image-id filtering backed by Arrow compute operations."""
    table = make_shape_table(_shape_rows(20_000), geometry_encoding="geoarrow.point")
    shapes = OMEArrowShapes(table)

    start = time.perf_counter()
    subset = shapes.for_image("image-3")
    elapsed = time.perf_counter() - start

    assert subset.table.num_rows == 4_000
    assert subset.table["image_id"].to_pylist() == ["image-3"] * 4_000
    assert elapsed < 0.25
