"""Tests for image-file ingestion helpers."""

from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest

import ome_arrow.ingest as ingest_mod
from ome_arrow import OME_ARROW_BYTE_STRUCT
from ome_arrow.export import to_numpy


@pytest.mark.parametrize(
    ("converter", "path"),
    [
        (ingest_mod.from_tiff, Path("image.tiff")),
        (ingest_mod.from_ome_zarr, Path("image.ome.zarr")),
    ],
)
def test_file_ingest_forwards_byte_chunk_options(
    monkeypatch: pytest.MonkeyPatch,
    converter: Callable[..., object],
    path: Path,
) -> None:
    """Encode and compress chunks requested through file ingest helpers."""
    arr = np.zeros((1, 1, 1, 32, 32), dtype=np.uint16)
    image = SimpleNamespace(
        data=arr,
        dims=SimpleNamespace(T=1, C=1, Z=1, Y=32, X=32),
        physical_pixel_sizes=SimpleNamespace(X=1.0, Y=1.0, Z=1.0, unit="µm"),
        channel_names=["C0"],
    )
    monkeypatch.setattr(ingest_mod, "BioImage", lambda **_kwargs: image)

    scalar = converter(
        path,
        chunk_encoding="bytes",
        chunk_compression="zstd",
        chunk_compression_level=1,
    )

    assert scalar.type == OME_ARROW_BYTE_STRUCT
    assert scalar.as_py()["chunks"][0]["compression"] == "zstd"
    np.testing.assert_array_equal(to_numpy(scalar), arr)


def test_file_ingest_keeps_list_chunks_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the historical list encoding for existing callers."""
    arr = np.arange(12, dtype=np.uint16).reshape(1, 1, 1, 3, 4)
    image = SimpleNamespace(
        data=arr,
        dims=SimpleNamespace(T=1, C=1, Z=1, Y=3, X=4),
        physical_pixel_sizes=SimpleNamespace(X=1.0, Y=1.0, Z=1.0, unit="µm"),
    )
    monkeypatch.setattr(ingest_mod, "BioImage", lambda **_kwargs: image)

    scalar = ingest_mod.from_tiff("image.tiff")

    assert "pixels" in scalar.as_py()["chunks"][0]
    np.testing.assert_array_equal(to_numpy(scalar), arr)
