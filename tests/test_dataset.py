"""Tests for typed chunk-buffer OME-Arrow datasets."""

import pathlib

import numpy as np
import pyarrow.parquet as pq

from ome_arrow import (
    OMEArrow,
    OMEArrowDataset,
    choose_chunking,
    write_ome_arrow_dataset,
)


def test_choose_chunking_presets() -> None:
    """Resolve layout and chunk shape from access-pattern presets."""
    choice = choose_chunking(
        (1, 1, 4, 1024, 1024),
        np.uint16,
        access_pattern="fast_crop_2d",
    )

    assert choice.layout == "tile"
    assert choice.chunk_shape == (1, 1, 1, 256, 256)
    assert "fast_crop_2d" in choice.rationale


def test_write_dataset_preserves_dtype_and_reads_image(tmp_path: pathlib.Path) -> None:
    """Round-trip a typed byte-buffer dataset without widening pixels."""
    arr = np.arange(1 * 2 * 2 * 5 * 6, dtype=np.uint8).reshape(1, 2, 2, 5, 6)
    out = tmp_path / "typed_dataset"

    choice = write_ome_arrow_dataset(
        [arr],
        out,
        layout="image",
        compression=None,
    )
    dataset = OMEArrowDataset(out)
    image_id = dataset.images["image_id"].to_pylist()[0]
    roundtrip = dataset.pixels.read_image(image_id)

    assert choice.layout == "image"
    assert roundtrip.dtype == np.uint8
    np.testing.assert_array_equal(roundtrip, arr)
    assert pq.ParquetFile(out / "chunks.parquet").metadata.num_row_groups == 1


def test_dataset_reads_plane_and_region_from_indexed_tiles(
    tmp_path: pathlib.Path,
) -> None:
    """Read selected planes and crops through the physical chunk index."""
    arr = np.arange(1 * 1 * 2 * 6 * 7, dtype=np.uint16).reshape(1, 1, 2, 6, 7)
    out = tmp_path / "tiles"

    write_ome_arrow_dataset(
        [arr],
        out,
        layout="tile",
        chunk_shape=(1, 1, 1, 3, 4),
        compression="zstd",
    )
    dataset = OMEArrowDataset(out)
    image_id = dataset.images["image_id"].to_pylist()[0]

    plane = dataset.pixels.read_plane(image_id, z=1)
    region = dataset.pixels.read_region(
        image_id,
        z=1,
        y=slice(2, 6),
        x=slice(3, 7),
    )

    np.testing.assert_array_equal(plane, arr[0, 0, 1])
    np.testing.assert_array_equal(region, arr[:, :, 1:2, 2:6, 3:7])
    assert pq.ParquetFile(out / "chunks.parquet").metadata.num_row_groups == 8


def test_dataset_reads_from_packed_chunk_row_groups(tmp_path: pathlib.Path) -> None:
    """Read indexed chunks when several chunk rows share one row group."""
    arr = np.arange(1 * 1 * 2 * 6 * 7, dtype=np.uint16).reshape(1, 1, 2, 6, 7)
    out = tmp_path / "packed_tiles"

    write_ome_arrow_dataset(
        [arr],
        out,
        layout="tile",
        chunk_shape=(1, 1, 1, 3, 4),
        compression="zstd",
        chunk_rows_per_row_group=4,
    )
    dataset = OMEArrowDataset(out)
    image_id = dataset.images["image_id"].to_pylist()[0]

    region = dataset.pixels.read_region(
        image_id,
        z=1,
        y=slice(2, 6),
        x=slice(3, 7),
    )

    np.testing.assert_array_equal(region, arr[:, :, 1:2, 2:6, 3:7])
    assert pq.ParquetFile(out / "chunks.parquet").metadata.num_row_groups == 2


def test_dataset_accepts_ome_arrow_records(tmp_path: pathlib.Path) -> None:
    """Write OME-Arrow records to typed byte-buffer datasets."""
    arr = np.arange(1 * 1 * 1 * 3 * 4, dtype=np.uint16).reshape(1, 1, 1, 3, 4)
    oa = OMEArrow(arr, dim_order="TCZYX")
    out = tmp_path / "records"

    write_ome_arrow_dataset([oa], out, layout="z-plane", compression=None)
    dataset = OMEArrowDataset(out)
    image_id = dataset.images["image_id"].to_pylist()[0]

    roundtrip = dataset.pixels.read_image(image_id)
    assert roundtrip.dtype == np.uint16
    np.testing.assert_array_equal(roundtrip, arr)
