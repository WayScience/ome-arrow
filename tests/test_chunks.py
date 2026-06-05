"""
Tests for chunked pixel support.
"""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ome_arrow import OME_ARROW_BYTE_STRUCT, OMEArrow
from ome_arrow.export import plane_from_chunks, to_numpy, to_ome_parquet
from ome_arrow.ingest import from_numpy, from_ome_parquet, to_ome_arrow


def test_to_numpy_from_chunks(example_correct_data: dict) -> None:
    """Reconstruct dense arrays from chunked pixels."""
    data = dict(example_correct_data)
    data["planes"] = []

    arr = to_numpy(data)

    assert arr.shape == (1, 2, 1, 3, 4)
    np.testing.assert_array_equal(
        arr[0, 0, 0],
        np.array([[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]]),
    )
    np.testing.assert_array_equal(
        arr[0, 1, 0],
        np.array([[100, 101, 102, 103], [110, 111, 112, 113], [120, 121, 122, 123]]),
    )


def test_to_ome_arrow_builds_chunks() -> None:
    """Build chunked pixels from planes when requested."""
    planes = [
        {
            "z": 0,
            "t": 0,
            "c": 0,
            "pixels": [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23],
        }
    ]

    scalar = to_ome_arrow(
        image_id="img-0002",
        name="Chunky",
        image_type="image",
        dimension_order="XYCT",
        dtype="uint16",
        size_x=4,
        size_y=3,
        size_z=1,
        size_c=1,
        size_t=1,
        channels=[{"id": "ch-0", "name": "C0"}],
        planes=planes,
        chunk_shape=(1, 2, 2),
        build_chunks=True,
    )

    record = scalar.as_py()
    assert record["chunk_grid"]["chunk_y"] == 2
    assert record["chunk_grid"]["chunk_x"] == 2
    assert len(record["chunks"]) == 4

    first_chunk = record["chunks"][0]
    assert first_chunk["t"] == 0
    assert first_chunk["c"] == 0
    assert first_chunk["z"] == 0
    assert first_chunk["y"] == 0
    assert first_chunk["x"] == 0
    assert first_chunk["shape_y"] == 2
    assert first_chunk["shape_x"] == 2
    assert first_chunk["pixels"] == [0, 1, 10, 11]


def test_plane_from_chunks(example_correct_data: dict) -> None:
    """Extract a 2D plane directly from chunked pixels."""
    data = dict(example_correct_data)
    data["planes"] = []

    plane = plane_from_chunks(data, t=0, c=1, z=0)

    np.testing.assert_array_equal(
        plane,
        np.array([[100, 101, 102, 103], [110, 111, 112, 113], [120, 121, 122, 123]]),
    )


def test_to_numpy_from_byte_chunks() -> None:
    """Decode inline typed chunk bytes without numeric pixel lists."""
    arr = np.arange(1 * 1 * 2 * 3 * 4, dtype=np.uint16).reshape(1, 1, 2, 3, 4)

    scalar = from_numpy(
        arr,
        dim_order="TCZYX",
        chunk_shape=(1, 2, 2),
        chunk_encoding="bytes",
    )
    record = scalar.as_py()

    assert scalar.type == OME_ARROW_BYTE_STRUCT
    assert record["planes"] == []
    assert "pixel_bytes" in record["chunks"][0]
    assert "pixels" not in record["chunks"][0]
    np.testing.assert_array_equal(to_numpy(scalar), arr)
    np.testing.assert_array_equal(
        plane_from_chunks(scalar, t=0, c=0, z=1), arr[0, 0, 1]
    )


def test_inline_byte_parquet_roundtrip_and_tensor_view(tmp_path: Path) -> None:
    """Write/read an ergonomic inline OME value backed by typed chunk bytes."""
    arr = np.arange(1 * 1 * 2 * 3 * 4, dtype=np.uint16).reshape(1, 1, 2, 3, 4)
    scalar = from_numpy(arr, dim_order="TCZYX", build_chunks=True)
    out = tmp_path / "inline-bytes.ome.parquet"

    to_ome_parquet(
        scalar,
        str(out),
        column_name="ome_arrow",
        inline_chunk_encoding="bytes",
    )

    table = pq.read_table(out)
    assert table["ome_arrow"].type == OME_ARROW_BYTE_STRUCT

    roundtrip, struct_array = from_ome_parquet(
        out,
        column_name="ome_arrow",
        return_array=True,
    )
    np.testing.assert_array_equal(to_numpy(roundtrip), arr)
    np.testing.assert_array_equal(
        OMEArrow(str(out)).tensor_view(layout="TCZYX").to_numpy(contiguous=True),
        arr,
    )
    np.testing.assert_array_equal(
        OMEArrow(str(out)).tensor_view(t=0, z=1, c=0, layout="YX").to_numpy(),
        arr[0, 0, 1],
    )
    assert struct_array.type == OME_ARROW_BYTE_STRUCT
