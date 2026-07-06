"""Performance canaries for Arrow-native OME-Arrow shape tables."""

from __future__ import annotations

import pathlib
import statistics
import time
from collections.abc import Callable

from ome_arrow import (
    OMEArrowShapes,
    make_shape_table,
    read_shape_parquet,
    write_shape_parquet,
)


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


def _median_seconds(fn: Callable[[], object], repeats: int = 3) -> float:
    """Return median runtime across repeated calls."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def test_make_shape_table_handles_large_object_tables_quickly() -> None:
    """Keep shape table construction on Arrow's fast path."""
    rows = _shape_rows(20_000)

    table = make_shape_table(rows, geometry_encoding="geoarrow.point")
    elapsed = _median_seconds(
        lambda: make_shape_table(rows, geometry_encoding="geoarrow.point")
    )

    assert table.num_rows == 20_000
    assert table.schema.field("area").type.bit_width == 64
    assert elapsed < 5.0


def test_shape_filtering_stays_vectorized_for_image_ids() -> None:
    """Keep image-id filtering backed by Arrow compute operations."""
    table = make_shape_table(_shape_rows(20_000), geometry_encoding="geoarrow.point")
    shapes = OMEArrowShapes(table)

    subset = shapes.for_image("image-3")
    elapsed = _median_seconds(lambda: shapes.for_image("image-3"))

    assert subset.table.num_rows == 4_000
    assert subset.table["image_id"].to_pylist() == ["image-3"] * 4_000
    assert elapsed < 2.0


def test_shape_parquet_projection_stays_fast(tmp_path: pathlib.Path) -> None:
    """Keep projected shape reads on Parquet's columnar fast path."""
    path = tmp_path / "cells.ome-shapes.parquet"
    table = make_shape_table(_shape_rows(20_000), geometry_encoding="geoarrow.point")

    write_shape_parquet(table, path, row_group_size=5_000)
    write_elapsed = _median_seconds(
        lambda: write_shape_parquet(table, path, row_group_size=5_000)
    )

    projected = read_shape_parquet(
        path,
        columns=["object_id", "image_id", "area", "mean_intensity"],
        filters=[("image_id", "==", "image-3")],
    )
    read_elapsed = _median_seconds(
        lambda: read_shape_parquet(
            path,
            columns=["object_id", "image_id", "area", "mean_intensity"],
            filters=[("image_id", "==", "image-3")],
        )
    )

    assert projected.num_rows == 4_000
    assert projected.column_names == ["object_id", "image_id", "area", "mean_intensity"]
    assert write_elapsed < 5.0
    assert read_elapsed < 2.0
