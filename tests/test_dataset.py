"""Tests for typed chunk-buffer OME-Arrow datasets."""

import pathlib

import numpy as np
import pyarrow.parquet as pq
import pytest

from ome_arrow import (
    AccessPattern,
    ChunkChoice,
    Layout,
    OMEArrow,
    OMEArrowDataset,
    choose_chunking,
    from_numpy,
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


def test_public_dataset_typing_exports() -> None:
    """Expose dataset typing helpers from the package root."""
    assert ChunkChoice is not None
    assert Layout is not None
    assert AccessPattern is not None


def test_choose_chunking_rejects_nonpositive_target_chunk_mb() -> None:
    """Fail before chunk scaling when target_chunk_mb is invalid."""
    with pytest.raises(ValueError, match=r"target_chunk_mb.*size_mb"):
        choose_chunking(
            (1, 1, 4, 1024, 1024),
            np.uint16,
            access_pattern="fast_crop_2d",
            target_chunk_mb=0,
        )


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
    image_id = dataset.image_ids[0]
    roundtrip = dataset.read_image()

    assert choice.layout == "image"
    assert dataset.image_ids == [image_id]
    assert roundtrip.dtype == np.uint8
    np.testing.assert_array_equal(roundtrip, arr)
    assert pq.ParquetFile(out / "chunks.parquet").metadata.num_row_groups == 1


def test_write_dataset_rejects_duplicate_image_ids(tmp_path: pathlib.Path) -> None:
    """Reject duplicate image IDs before chunk rows can merge at read time."""
    arr = np.arange(1 * 1 * 1 * 3 * 4, dtype=np.uint16).reshape(1, 1, 1, 3, 4)
    first = from_numpy(arr, image_id="duplicate", build_chunks=False)
    second = from_numpy(arr + 1, image_id="duplicate", build_chunks=False)

    with pytest.raises(ValueError, match="Duplicate image_id"):
        write_ome_arrow_dataset([first, second], tmp_path / "duplicates")


def test_write_dataset_removes_stale_index_when_index_disabled(
    tmp_path: pathlib.Path,
) -> None:
    """Avoid loading stale physical index metadata after rewriting a dataset."""
    arr = np.arange(1 * 1 * 2 * 6 * 7, dtype=np.uint16).reshape(1, 1, 2, 6, 7)
    out = tmp_path / "stale_index"

    write_ome_arrow_dataset(
        [arr],
        out,
        layout="tile",
        chunk_shape=(1, 1, 1, 3, 4),
        compression=None,
        build_physical_index=True,
    )
    assert (out / "index.parquet").exists()

    write_ome_arrow_dataset(
        [arr],
        out,
        layout="tile",
        chunk_shape=(1, 1, 1, 3, 4),
        compression=None,
        build_physical_index=False,
    )

    assert not (out / "index.parquet").exists()
    assert OMEArrowDataset(out).index is None


def test_write_dataset_can_normalize_pixel_dtype(tmp_path: pathlib.Path) -> None:
    """Explicitly widen stored pixels when callers ask for a target dtype."""
    arr = np.array([[[[[0, 255], [256, 70000]]]]], dtype=np.uint32)
    out = tmp_path / "typed_dataset_uint16"

    write_ome_arrow_dataset(
        [arr],
        out,
        layout="image",
        compression=None,
        pixel_dtype="uint16",
    )
    dataset = OMEArrowDataset(out)
    roundtrip = dataset.read_image()

    assert roundtrip.dtype == np.uint16
    np.testing.assert_array_equal(
        roundtrip,
        np.array([[[[[0, 255], [256, 65535]]]]], dtype=np.uint16),
    )


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

    plane = dataset.read_plane(z=1)
    region = dataset.read_region(
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

    region = dataset.read_region(
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

    roundtrip = dataset.read_image()
    assert roundtrip.dtype == np.uint16
    np.testing.assert_array_equal(roundtrip, arr)


def test_dataset_read_torch_return_type(tmp_path: pathlib.Path) -> None:
    """Read typed dataset pixels directly as torch tensors."""
    torch = pytest.importorskip("torch")
    arr = np.arange(1 * 1 * 1 * 3 * 4, dtype=np.uint16).reshape(1, 1, 1, 3, 4)
    out = tmp_path / "torch_return"

    write_ome_arrow_dataset([arr], out, layout="image", compression=None)
    dataset = OMEArrowDataset(out)

    tensor = dataset.read_image(return_type="torch")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.uint16
    np.testing.assert_array_equal(tensor.numpy(), arr)


def test_dataset_read_jax_return_type(tmp_path: pathlib.Path) -> None:
    """Read typed dataset pixels directly as JAX arrays."""
    jnp = pytest.importorskip("jax.numpy")
    arr = np.arange(1 * 1 * 1 * 3 * 4, dtype=np.uint16).reshape(1, 1, 1, 3, 4)
    out = tmp_path / "jax_return"

    write_ome_arrow_dataset([arr], out, layout="image", compression=None)
    dataset = OMEArrowDataset(out)

    jax_arr = dataset.read_image(return_type="jax")
    assert isinstance(jax_arr, jnp.ndarray)
    assert jax_arr.dtype == jnp.uint16
    np.testing.assert_array_equal(np.asarray(jax_arr), arr)
