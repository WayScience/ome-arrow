"""Lightweight OME-IRIS-aligned benchmark for selective pixel reads.

This benchmark intentionally follows the simple style of
``benchmarks/benchmark_lazy_tensor.py``. It accepts local image paths, builds
temporary OME-Zarr and typed OME-Arrow dataset artifacts with matched chunking,
and times public read APIs for 2D, 3D, 4D, and 5D-style access patterns.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from bioio import BioImage
from bioio_ome_zarr import Reader as OMEZarrReader

from ome_arrow import OMEArrow, OMEArrowDataset, write_ome_arrow_dataset


@dataclass(frozen=True)
class BenchmarkResult:
    """Summary stats for one benchmark case."""

    dataset: str
    case: str
    format: str
    median_ms: float
    min_ms: float
    max_ms: float
    shape: tuple[int, ...]
    dtype: str
    bytes_on_disk: int


@dataclass(frozen=True)
class Fixture:
    """One source image fixture to benchmark."""

    name: str
    path: Path
    preferred_chunk_shape: tuple[int, int, int, int, int]
    chunk_rows_per_row_group: int


@dataclass(frozen=True)
class ArrowArtifact:
    """Typed OME-Arrow artifacts for one dtype mode."""

    label: str
    block_path: Path
    image_path: Path
    block: OMEArrowDataset
    block_image_id: str
    image: OMEArrowDataset
    image_image_id: str


def _default_fixtures() -> list[Fixture]:
    """Return local fixtures that cover 2D, 3D, 4D, and 5D patterns."""
    return [
        Fixture(
            name="2d-human",
            path=Path("tests/data/examplehuman/AS_09125_050116030001_D03f00d0.tif"),
            preferred_chunk_shape=(1, 1, 1, 256, 256),
            chunk_rows_per_row_group=1,
        ),
        Fixture(
            name="3d-nuclei",
            path=Path("tests/data/cp-3d-nuclei/nuclei1_out_c00_dr90_image.tif"),
            preferred_chunk_shape=(1, 1, 16, 128, 128),
            chunk_rows_per_row_group=1,
        ),
        Fixture(
            name="4d-time-volume",
            path=Path("tests/data/ome-artificial-5d-datasets/4D-series.ome.tiff"),
            preferred_chunk_shape=(1, 1, 2, 128, 128),
            chunk_rows_per_row_group=8,
        ),
        Fixture(
            name="5d-multichannel-time-volume",
            path=Path(
                "tests/data/ome-artificial-5d-datasets/multi-channel-4D-series.ome.tiff"
            ),
            preferred_chunk_shape=(1, 1, 2, 128, 128),
            chunk_rows_per_row_group=8,
        ),
    ]


def _parse_fixture_arg(raw: str) -> Fixture:
    """Parse ``name=path`` or bare path CLI fixture input."""
    if "=" in raw:
        name, path = raw.split("=", 1)
    else:
        path = raw
        name = Path(path).stem
    return Fixture(
        name=name,
        path=Path(path),
        preferred_chunk_shape=(1, 1, 1, 256, 256),
        chunk_rows_per_row_group=1,
    )


def _dir_size(path: Path) -> int:
    """Return total bytes for a file or directory."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _time_case(
    *,
    dataset: str,
    case: str,
    format_name: str,
    fn: Callable[[], Any],
    repeats: int,
    warmup: int,
    bytes_on_disk: int,
) -> BenchmarkResult:
    """Time one case and return summary stats."""
    out: Any | None = None
    for _ in range(warmup):
        out = fn()
        _sync_result(out)

    times_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        _sync_result(out)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    if out is None:
        raise RuntimeError(f"Case did not produce output: {dataset} {case}")
    shape, dtype = _result_shape_dtype(out)
    return BenchmarkResult(
        dataset=dataset,
        case=case,
        format=format_name,
        median_ms=statistics.median(times_ms),
        min_ms=min(times_ms),
        max_ms=max(times_ms),
        shape=shape,
        dtype=dtype,
        bytes_on_disk=int(bytes_on_disk),
    )


def _sync_result(out: Any) -> None:
    """Synchronize lazy array backends for fair timing."""
    block_until_ready = getattr(out, "block_until_ready", None)
    if callable(block_until_ready):
        block_until_ready()


def _result_shape_dtype(out: Any) -> tuple[tuple[int, ...], str]:
    """Return shape and dtype strings for NumPy, Torch, and JAX-like arrays."""
    shape = tuple(int(v) for v in getattr(out, "shape", ()))
    dtype = getattr(out, "dtype", None)
    if dtype is None:
        dtype_str = type(out).__name__
    else:
        try:
            dtype_str = np.dtype(dtype).name
        except TypeError:
            dtype_str = str(dtype)
    return shape, dtype_str


def _torch_available() -> bool:
    """Return whether torch is importable."""
    return importlib.util.find_spec("torch") is not None


def _jax_available() -> bool:
    """Return whether JAX is importable."""
    return importlib.util.find_spec("jax") is not None


def _read_zarr_full(path: Path) -> np.ndarray:
    """Read a Zarr image directly through BioImage as TCZYX NumPy."""
    return np.asarray(BioImage(str(path), reader=OMEZarrReader).data)


def _read_zarr_plane(path: Path, *, t: int, c: int, z: int) -> np.ndarray:
    """Read one Zarr YX plane directly through BioImage."""
    image = BioImage(str(path), reader=OMEZarrReader)
    return np.asarray(image.get_image_data("YX", T=t, C=c, Z=z))


def _read_zarr_crop(
    path: Path,
    *,
    t: int,
    c: int,
    z: int,
    y: slice,
    x: slice,
) -> np.ndarray:
    """Read one Zarr YX crop directly through BioImage."""
    return _read_zarr_plane(path, t=t, c=c, z=z)[y, x]


def _read_zarr_subvolume(
    path: Path,
    *,
    t: int,
    c: int,
    z: slice,
    y: slice,
    x: slice,
) -> np.ndarray:
    """Read one Zarr subvolume directly through BioImage as TCZYX NumPy."""
    image = BioImage(str(path), reader=OMEZarrReader)
    cropped = np.asarray(image.get_image_data("ZYX", T=t, C=c))[z, y, x]
    return cropped.reshape(1, 1, z.stop - z.start, y.stop - y.start, x.stop - x.start)


def _read_tiff_full(path: Path) -> np.ndarray:
    """Read a TIFF image directly through BioImage as TCZYX NumPy."""
    return np.asarray(BioImage(str(path)).data)


def _read_tiff_plane(path: Path, *, t: int, c: int, z: int) -> np.ndarray:
    """Read one TIFF YX plane directly through BioImage."""
    image = BioImage(str(path))
    return np.asarray(image.get_image_data("YX", T=t, C=c, Z=z))


def _read_tiff_crop(
    path: Path,
    *,
    t: int,
    c: int,
    z: int,
    y: slice,
    x: slice,
) -> np.ndarray:
    """Read one TIFF YX crop directly through BioImage."""
    return _read_tiff_plane(path, t=t, c=c, z=z)[y, x]


def _read_tiff_subvolume(
    path: Path,
    *,
    t: int,
    c: int,
    z: slice,
    y: slice,
    x: slice,
) -> np.ndarray:
    """Read one TIFF subvolume directly through BioImage as TCZYX NumPy."""
    image = BioImage(str(path))
    cropped = np.asarray(image.get_image_data("ZYX", T=t, C=c))[z, y, x]
    return cropped.reshape(1, 1, z.stop - z.start, y.stop - y.start, x.stop - x.start)


def _load_source(path: Path) -> tuple[OMEArrow, np.ndarray, np.ndarray]:
    """Load source image arrays for normalized and raw dtype comparisons."""
    oa = OMEArrow(str(path))
    normalized = oa.export(
        how="numpy",
        dtype=np.dtype(oa.data.as_py()["pixels_meta"]["type"]),
    )
    if not isinstance(normalized, np.ndarray):
        raise TypeError("OMEArrow numpy export did not return an ndarray")
    raw = _read_tiff_full(path)
    return oa, normalized, raw


def _write_artifacts(
    fixture: Fixture,
    source: OMEArrow,
    normalized_arr: np.ndarray,
    raw_arr: np.ndarray,
    workdir: Path,
) -> tuple[Path, list[ArrowArtifact]]:
    """Write matched OME-Zarr and typed OME-Arrow artifacts for one fixture."""
    zarr_path = workdir / f"{fixture.name}.ome.zarr"
    dtype = np.dtype(normalized_arr.dtype)

    source.export(
        how="ome-zarr",
        out=str(zarr_path),
        dtype=dtype,
        chunks=fixture.preferred_chunk_shape,
        zarr_compressor="zstd",
        zarr_level=3,
    )

    artifacts: list[ArrowArtifact] = []
    for label, arr, pixel_dtype in (
        ("ome-arrow-src", raw_arr, None),
        ("ome-arrow-u16", normalized_arr, "uint16"),
    ):
        typed_block_path = workdir / f"{fixture.name}.{label}.block.ome-arrow"
        typed_image_path = workdir / f"{fixture.name}.{label}.image.ome-arrow"
        write_ome_arrow_dataset(
            [arr],
            typed_block_path,
            layout="block",
            chunk_shape=fixture.preferred_chunk_shape,
            compression="zstd",
            chunk_rows_per_row_group=fixture.chunk_rows_per_row_group,
            pixel_dtype=pixel_dtype,
        )
        write_ome_arrow_dataset(
            [arr],
            typed_image_path,
            layout="image",
            compression="zstd",
            pixel_dtype=pixel_dtype,
        )
        typed_block = OMEArrowDataset(typed_block_path)
        typed_image = OMEArrowDataset(typed_image_path)
        artifacts.append(
            ArrowArtifact(
                label=label,
                block_path=typed_block_path,
                image_path=typed_image_path,
                block=typed_block,
                block_image_id=str(typed_block.image_ids[0]),
                image=typed_image,
                image_image_id=str(typed_image.image_ids[0]),
            )
        )

    return zarr_path, artifacts


def _center_crop(size_y: int, size_x: int) -> tuple[slice, slice]:
    """Return a modest centered YX crop."""
    height = min(128, size_y)
    width = min(128, size_x)
    y0 = max(0, (size_y - height) // 2)
    x0 = max(0, (size_x - width) // 2)
    return slice(y0, y0 + height), slice(x0, x0 + width)


def _subvolume_slice(size_z: int) -> slice:
    """Return a small centered Z slice."""
    depth = min(4, size_z)
    z0 = max(0, (size_z - depth) // 2)
    return slice(z0, z0 + depth)


def _cases_for_fixture(
    fixture: Fixture,
    zarr_path: Path,
    arrow_artifacts: list[ArrowArtifact],
    arr: np.ndarray,
    *,
    repeats: int,
    warmup: int,
) -> list[BenchmarkResult]:
    """Build and execute benchmark cases for one fixture."""
    st, sc, sz, sy, sx = (int(v) for v in arr.shape)
    z_mid = sz // 2
    t_mid = st // 2
    c_mid = sc // 2
    crop_y, crop_x = _center_crop(sy, sx)
    roi = (
        crop_x.start,
        crop_y.start,
        crop_x.stop - crop_x.start,
        crop_y.stop - crop_y.start,
    )
    z_sub = _subvolume_slice(sz)

    zarr_size = _dir_size(zarr_path)
    tiff_size = _dir_size(fixture.path)

    case_defs: list[tuple[str, str, Callable[[], Any], int]] = [
        (
            "full-image",
            "ome-tiff-tensor-numpy",
            lambda: (
                OMEArrow.scan(str(fixture.path))
                .tensor_view(layout="TCZYX")
                .to_numpy(contiguous=True)
            ),
            tiff_size,
        ),
        (
            "full-image",
            "ome-tiff-bioio-numpy",
            lambda: _read_tiff_full(fixture.path),
            tiff_size,
        ),
        (
            "full-image",
            "ome-zarr-tensor-numpy",
            lambda: (
                OMEArrow.scan(str(zarr_path))
                .tensor_view(layout="TCZYX")
                .to_numpy(contiguous=True)
            ),
            zarr_size,
        ),
        (
            "full-image",
            "ome-zarr-bioio-numpy",
            lambda: _read_zarr_full(zarr_path),
            zarr_size,
        ),
        (
            "plane",
            "ome-tiff-tensor-numpy",
            lambda: (
                OMEArrow.scan(str(fixture.path))
                .tensor_view(t=0, c=0, z=z_mid, layout="YX")
                .to_numpy(contiguous=True)
            ),
            tiff_size,
        ),
        (
            "plane",
            "ome-tiff-bioio-numpy",
            lambda: _read_tiff_plane(fixture.path, t=0, c=0, z=z_mid),
            tiff_size,
        ),
        (
            "plane",
            "ome-zarr-tensor-numpy",
            lambda: (
                OMEArrow.scan(str(zarr_path))
                .tensor_view(t=0, c=0, z=z_mid, layout="YX")
                .to_numpy(contiguous=True)
            ),
            zarr_size,
        ),
        (
            "plane",
            "ome-zarr-bioio-numpy",
            lambda: _read_zarr_plane(zarr_path, t=0, c=0, z=z_mid),
            zarr_size,
        ),
        (
            "crop-2d",
            "ome-tiff-tensor-numpy",
            lambda: (
                OMEArrow.scan(str(fixture.path))
                .tensor_view(t=0, c=0, z=z_mid, roi=roi, layout="YX")
                .to_numpy(contiguous=True)
            ),
            tiff_size,
        ),
        (
            "crop-2d",
            "ome-tiff-bioio-numpy",
            lambda: _read_tiff_crop(
                fixture.path,
                t=0,
                c=0,
                z=z_mid,
                y=crop_y,
                x=crop_x,
            ),
            tiff_size,
        ),
        (
            "crop-2d",
            "ome-zarr-tensor-numpy",
            lambda: (
                OMEArrow.scan(str(zarr_path))
                .tensor_view(t=0, c=0, z=z_mid, roi=roi, layout="YX")
                .to_numpy(contiguous=True)
            ),
            zarr_size,
        ),
        (
            "crop-2d",
            "ome-zarr-bioio-numpy",
            lambda: _read_zarr_crop(
                zarr_path,
                t=0,
                c=0,
                z=z_mid,
                y=crop_y,
                x=crop_x,
            ),
            zarr_size,
        ),
    ]

    for artifact in arrow_artifacts:
        block_size = _dir_size(artifact.block_path)
        image_size = _dir_size(artifact.image_path)
        case_defs.extend(
            [
                (
                    "full-image",
                    f"{artifact.label}-numpy",
                    lambda artifact=artifact: artifact.image.read_image(
                        artifact.image_image_id
                    ),
                    image_size,
                ),
                (
                    "plane",
                    f"{artifact.label}-numpy",
                    lambda artifact=artifact: artifact.block.read_plane(
                        artifact.block_image_id,
                        t=0,
                        c=0,
                        z=z_mid,
                    ),
                    block_size,
                ),
                (
                    "crop-2d",
                    f"{artifact.label}-numpy",
                    lambda artifact=artifact: artifact.block.read_region(
                        artifact.block_image_id,
                        t=0,
                        c=0,
                        z=z_mid,
                        y=crop_y,
                        x=crop_x,
                    ),
                    block_size,
                ),
            ]
        )

    if _torch_available():
        case_defs.extend(
            [
                (
                    "full-image",
                    "ome-tiff-tensor-torch",
                    lambda: (
                        OMEArrow.scan(str(fixture.path))
                        .tensor_view(layout="TCZYX")
                        .to_torch(mode="numpy")
                    ),
                    tiff_size,
                ),
                (
                    "full-image",
                    "ome-zarr-tensor-torch",
                    lambda: (
                        OMEArrow.scan(str(zarr_path))
                        .tensor_view(layout="TCZYX")
                        .to_torch(mode="numpy")
                    ),
                    zarr_size,
                ),
                (
                    "crop-2d",
                    "ome-tiff-tensor-torch",
                    lambda: (
                        OMEArrow.scan(str(fixture.path))
                        .tensor_view(t=0, c=0, z=z_mid, roi=roi, layout="YX")
                        .to_torch(mode="numpy")
                    ),
                    tiff_size,
                ),
                (
                    "crop-2d",
                    "ome-zarr-tensor-torch",
                    lambda: (
                        OMEArrow.scan(str(zarr_path))
                        .tensor_view(t=0, c=0, z=z_mid, roi=roi, layout="YX")
                        .to_torch(mode="numpy")
                    ),
                    zarr_size,
                ),
            ]
        )
        for artifact in arrow_artifacts:
            block_size = _dir_size(artifact.block_path)
            image_size = _dir_size(artifact.image_path)
            case_defs.extend(
                [
                    (
                        "full-image",
                        f"{artifact.label}-torch",
                        lambda artifact=artifact: artifact.image.read_image(
                            artifact.image_image_id,
                            return_type="torch",
                        ),
                        image_size,
                    ),
                    (
                        "crop-2d",
                        f"{artifact.label}-torch",
                        lambda artifact=artifact: artifact.block.read_region(
                            artifact.block_image_id,
                            t=0,
                            c=0,
                            z=z_mid,
                            y=crop_y,
                            x=crop_x,
                            return_type="torch",
                        ),
                        block_size,
                    ),
                ]
            )

    if _jax_available():
        case_defs.extend(
            [
                (
                    "full-image",
                    "ome-tiff-tensor-jax",
                    lambda: (
                        OMEArrow.scan(str(fixture.path))
                        .tensor_view(layout="TCZYX")
                        .to_jax(mode="numpy")
                    ),
                    tiff_size,
                ),
                (
                    "full-image",
                    "ome-zarr-tensor-jax",
                    lambda: (
                        OMEArrow.scan(str(zarr_path))
                        .tensor_view(layout="TCZYX")
                        .to_jax(mode="numpy")
                    ),
                    zarr_size,
                ),
                (
                    "crop-2d",
                    "ome-tiff-tensor-jax",
                    lambda: (
                        OMEArrow.scan(str(fixture.path))
                        .tensor_view(t=0, c=0, z=z_mid, roi=roi, layout="YX")
                        .to_jax(mode="numpy")
                    ),
                    tiff_size,
                ),
                (
                    "crop-2d",
                    "ome-zarr-tensor-jax",
                    lambda: (
                        OMEArrow.scan(str(zarr_path))
                        .tensor_view(t=0, c=0, z=z_mid, roi=roi, layout="YX")
                        .to_jax(mode="numpy")
                    ),
                    zarr_size,
                ),
            ]
        )
        for artifact in arrow_artifacts:
            block_size = _dir_size(artifact.block_path)
            image_size = _dir_size(artifact.image_path)
            case_defs.extend(
                [
                    (
                        "full-image",
                        f"{artifact.label}-jax",
                        lambda artifact=artifact: artifact.image.read_image(
                            artifact.image_image_id,
                            return_type="jax",
                        ),
                        image_size,
                    ),
                    (
                        "crop-2d",
                        f"{artifact.label}-jax",
                        lambda artifact=artifact: artifact.block.read_region(
                            artifact.block_image_id,
                            t=0,
                            c=0,
                            z=z_mid,
                            y=crop_y,
                            x=crop_x,
                            return_type="jax",
                        ),
                        block_size,
                    ),
                ]
            )

    if sz > 1:
        case_defs.extend(
            [
                (
                    "subvolume",
                    "ome-tiff-tensor-numpy",
                    lambda: (
                        OMEArrow.scan(str(fixture.path))
                        .tensor_view(t=0, c=0, z=z_sub, roi=roi, layout="TCZYX")
                        .to_numpy(contiguous=True)
                    ),
                    tiff_size,
                ),
                (
                    "subvolume",
                    "ome-tiff-bioio-numpy",
                    lambda: _read_tiff_subvolume(
                        fixture.path,
                        t=0,
                        c=0,
                        z=z_sub,
                        y=crop_y,
                        x=crop_x,
                    ),
                    tiff_size,
                ),
                (
                    "subvolume",
                    "ome-zarr-tensor-numpy",
                    lambda: (
                        OMEArrow.scan(str(zarr_path))
                        .tensor_view(t=0, c=0, z=z_sub, roi=roi, layout="TCZYX")
                        .to_numpy(contiguous=True)
                    ),
                    zarr_size,
                ),
                (
                    "subvolume",
                    "ome-zarr-bioio-numpy",
                    lambda: _read_zarr_subvolume(
                        zarr_path,
                        t=0,
                        c=0,
                        z=z_sub,
                        y=crop_y,
                        x=crop_x,
                    ),
                    zarr_size,
                ),
            ]
        )
        for artifact in arrow_artifacts:
            block_size = _dir_size(artifact.block_path)
            case_defs.append(
                (
                    "subvolume",
                    f"{artifact.label}-numpy",
                    lambda artifact=artifact: artifact.block.read_region(
                        artifact.block_image_id,
                        t=0,
                        c=0,
                        z=z_sub,
                        y=crop_y,
                        x=crop_x,
                    ),
                    block_size,
                )
            )

    if st > 1:
        case_defs.extend(
            [
                (
                    "timepoint-plane",
                    "ome-tiff-tensor-numpy",
                    lambda: (
                        OMEArrow.scan(str(fixture.path))
                        .tensor_view(t=t_mid, c=0, z=z_mid, layout="YX")
                        .to_numpy(contiguous=True)
                    ),
                    tiff_size,
                ),
                (
                    "timepoint-plane",
                    "ome-tiff-bioio-numpy",
                    lambda: _read_tiff_plane(fixture.path, t=t_mid, c=0, z=z_mid),
                    tiff_size,
                ),
                (
                    "timepoint-plane",
                    "ome-zarr-tensor-numpy",
                    lambda: (
                        OMEArrow.scan(str(zarr_path))
                        .tensor_view(t=t_mid, c=0, z=z_mid, layout="YX")
                        .to_numpy(contiguous=True)
                    ),
                    zarr_size,
                ),
                (
                    "timepoint-plane",
                    "ome-zarr-bioio-numpy",
                    lambda: _read_zarr_plane(zarr_path, t=t_mid, c=0, z=z_mid),
                    zarr_size,
                ),
            ]
        )
        for artifact in arrow_artifacts:
            block_size = _dir_size(artifact.block_path)
            case_defs.append(
                (
                    "timepoint-plane",
                    f"{artifact.label}-numpy",
                    lambda artifact=artifact: artifact.block.read_plane(
                        artifact.block_image_id,
                        t=t_mid,
                        c=0,
                        z=z_mid,
                    ),
                    block_size,
                )
            )

    if sc > 1:
        case_defs.extend(
            [
                (
                    "channel-plane",
                    "ome-tiff-tensor-numpy",
                    lambda: (
                        OMEArrow.scan(str(fixture.path))
                        .tensor_view(t=0, c=c_mid, z=z_mid, layout="YX")
                        .to_numpy(contiguous=True)
                    ),
                    tiff_size,
                ),
                (
                    "channel-plane",
                    "ome-tiff-bioio-numpy",
                    lambda: _read_tiff_plane(fixture.path, t=0, c=c_mid, z=z_mid),
                    tiff_size,
                ),
                (
                    "channel-plane",
                    "ome-zarr-tensor-numpy",
                    lambda: (
                        OMEArrow.scan(str(zarr_path))
                        .tensor_view(t=0, c=c_mid, z=z_mid, layout="YX")
                        .to_numpy(contiguous=True)
                    ),
                    zarr_size,
                ),
                (
                    "channel-plane",
                    "ome-zarr-bioio-numpy",
                    lambda: _read_zarr_plane(zarr_path, t=0, c=c_mid, z=z_mid),
                    zarr_size,
                ),
            ]
        )
        for artifact in arrow_artifacts:
            block_size = _dir_size(artifact.block_path)
            case_defs.append(
                (
                    "channel-plane",
                    f"{artifact.label}-numpy",
                    lambda artifact=artifact: artifact.block.read_plane(
                        artifact.block_image_id,
                        t=0,
                        c=c_mid,
                        z=z_mid,
                    ),
                    block_size,
                )
            )

    results = []
    for case, format_name, fn, bytes_on_disk in case_defs:
        results.append(
            _time_case(
                dataset=fixture.name,
                case=case,
                format_name=format_name,
                fn=fn,
                repeats=repeats,
                warmup=warmup,
                bytes_on_disk=bytes_on_disk,
            )
        )
    return results


def _print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results grouped in a compact table."""
    print("")
    print("OME-IRIS-style benchmark (ms)")
    print(
        f"{'dataset':24} {'case':18} {'format':24} {'median':>9} "
        f"{'min':>9} {'max':>9} {'shape':>18} {'dtype':>8} {'MB':>8}"
    )
    print("-" * 120)
    for r in results:
        print(
            f"{r.dataset:24} {r.case:18} {r.format:24} "
            f"{r.median_ms:9.2f} {r.min_ms:9.2f} {r.max_ms:9.2f} "
            f"{r.shape!s:>18} {r.dtype:>8} {r.bytes_on_disk / 1_000_000:8.2f}"
        )


def _write_json(path: Path, results: list[BenchmarkResult]) -> None:
    """Write benchmark results as JSON."""
    path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "dataset": r.dataset,
                        "case": r.case,
                        "format": r.format,
                        "median_ms": r.median_ms,
                        "min_ms": r.min_ms,
                        "max_ms": r.max_ms,
                        "shape": list(r.shape),
                        "dtype": r.dtype,
                        "bytes_on_disk": r.bytes_on_disk,
                    }
                    for r in results
                ]
            },
            indent=2,
        )
    )


def main() -> None:
    """Run OME-IRIS-style benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--fixture",
        action="append",
        default=[],
        help="Fixture as name=path or path. Can be provided more than once.",
    )
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    fixtures = [_parse_fixture_arg(raw) for raw in args.fixture]
    if not fixtures:
        fixtures = [f for f in _default_fixtures() if f.path.exists()]
    if not fixtures:
        raise FileNotFoundError("No benchmark fixtures were found.")

    all_results: list[BenchmarkResult] = []
    with tempfile.TemporaryDirectory(prefix="ome_arrow_ome_iris_bench_") as tmp:
        workdir = Path(tmp)
        for fixture in fixtures:
            if not fixture.path.exists():
                print(f"Skipping missing fixture: {fixture.name} -> {fixture.path}")
                continue
            print(f"Preparing {fixture.name}: {fixture.path}")
            source, arr, raw_arr = _load_source(fixture.path)
            zarr_path, arrow_artifacts = _write_artifacts(
                fixture,
                source,
                arr,
                raw_arr,
                workdir,
            )
            all_results.extend(
                _cases_for_fixture(
                    fixture,
                    zarr_path,
                    arrow_artifacts,
                    arr,
                    repeats=args.repeats,
                    warmup=args.warmup,
                )
            )

    _print_results(all_results)
    if args.json_out is not None:
        _write_json(args.json_out, all_results)


if __name__ == "__main__":
    main()
