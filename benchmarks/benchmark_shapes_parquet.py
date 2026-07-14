"""Benchmark OME-Arrow Shapes Parquet read/write paths.

The benchmark uses synthetic but biologically interpretable object tables:

- one row is one segmented object
- each object belongs to one source image
- label-reference rows point at canonical label rasters by label value
- measurement columns model common morphology and intensity features

The results are directional local signals, not universal format rankings.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import pyarrow as pa

from ome_arrow import make_shape_table, read_shape_parquet, write_shape_parquet


@dataclass(frozen=True)
class ShapeFixture:
    """Synthetic scientific shape workload description."""

    name: str
    geometry_encoding: str
    image_count: int
    objects_per_image: int
    measurement_count: int

    @property
    def object_count(self) -> int:
        """Return the total number of object rows."""
        return self.image_count * self.objects_per_image


@dataclass(frozen=True)
class ShapeBenchmarkResult:
    """One shape Parquet benchmark result."""

    fixture: str
    geometry_encoding: str
    compression: str | None
    image_count: int
    object_count: int
    measurement_count: int
    write_ms: float
    full_read_ms: float
    projected_read_ms: float
    filtered_read_ms: float
    file_size_mb: float
    projected_columns: int
    filtered_rows: int


def _time(fn: Callable[[], object], *, repeats: int, warmup: int) -> float:
    """Return median runtime in milliseconds."""
    for _ in range(warmup):
        fn()
    times_ms = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times_ms)


def _measurement_payload(index: int) -> dict[str, float]:
    """Create deterministic morphology and intensity measurements."""
    return {
        "area_um2": float(50.0 + (index % 200) * 0.5),
        "perimeter_um": float(20.0 + (index % 120) * 0.25),
        "mean_intensity_dna": float(100.0 + (index % 4096)),
        "mean_intensity_mito": float(80.0 + ((index * 3) % 2048)),
        "eccentricity": float((index % 100) / 100.0),
        "solidity": float(0.75 + (index % 25) / 100.0),
    }


def _point_rows(fixture: ShapeFixture) -> list[dict[str, object]]:
    """Build centroid-like point geometry rows."""
    rows: list[dict[str, object]] = []
    for image_idx in range(fixture.image_count):
        image_id = f"image-{image_idx:03d}"
        label_image_id = f"{image_id}-labels"
        for object_idx in range(fixture.objects_per_image):
            index = image_idx * fixture.objects_per_image + object_idx
            y = float((object_idx // 100) * 12 + 6)
            x = float((object_idx % 100) * 12 + 6)
            rows.append(
                {
                    "object_id": f"{image_id}-cell-{object_idx:05d}",
                    "image_id": image_id,
                    "label_image_id": label_image_id,
                    "label_value": object_idx + 1,
                    "geometry": [y, x],
                    "centroid": [y, x],
                    "class": "cell",
                    "confidence": 0.95,
                    **_measurement_payload(index),
                }
            )
    return rows


def _labelmask_rows(fixture: ShapeFixture) -> list[dict[str, object]]:
    """Build rows that reference canonical label rasters."""
    rows: list[dict[str, object]] = []
    for image_idx in range(fixture.image_count):
        image_id = f"image-{image_idx:03d}"
        label_image_id = f"{image_id}-nuclear-labels"
        for object_idx in range(fixture.objects_per_image):
            index = image_idx * fixture.objects_per_image + object_idx
            label_value = object_idx + 1
            rows.append(
                {
                    "object_id": f"{image_id}-nucleus-{object_idx:05d}",
                    "image_id": image_id,
                    "label_image_id": label_image_id,
                    "label_value": label_value,
                    "geometry": {
                        "label_image_id": label_image_id,
                        "label_value": label_value,
                    },
                    "class": "nucleus",
                    "confidence": 0.98,
                    **_measurement_payload(index),
                }
            )
    return rows


def _shape_table(fixture: ShapeFixture) -> pa.Table:
    """Create one benchmark shape table."""
    if fixture.geometry_encoding == "ome.labelmask":
        rows = _labelmask_rows(fixture)
    else:
        rows = _point_rows(fixture)
    return make_shape_table(
        rows,
        geometry_encoding=fixture.geometry_encoding,
        axes=("y", "x"),
        units=("pixel", "pixel"),
    )


def _benchmark_fixture(
    fixture: ShapeFixture,
    *,
    compression: str | None,
    repeats: int,
    warmup: int,
    workdir: Path,
) -> ShapeBenchmarkResult:
    """Run write/read benchmark cases for one shape fixture."""
    table = _shape_table(fixture)
    out = workdir / f"{fixture.name}.{compression or 'none'}.ome-shapes.parquet"
    projected_columns = [
        "object_id",
        "image_id",
        "label_value",
        "area_um2",
        "mean_intensity_dna",
        "eccentricity",
    ]
    filtered_image_id = f"image-{fixture.image_count // 2:03d}"

    def write() -> None:
        write_shape_parquet(
            table,
            out,
            compression=compression,
            row_group_size=fixture.objects_per_image,
            use_dictionary=[
                "object_id",
                "image_id",
                "label_image_id",
                "class",
            ],
        )

    write()
    write_ms = _time(write, repeats=repeats, warmup=warmup)

    def full_read() -> pa.Table:
        return read_shape_parquet(out)

    def projected_read() -> pa.Table:
        return read_shape_parquet(out, columns=projected_columns)

    def filtered_read() -> pa.Table:
        return read_shape_parquet(
            out,
            columns=projected_columns,
            filters=[("image_id", "==", filtered_image_id)],
        )

    full = full_read()
    projected = projected_read()
    filtered = filtered_read()
    if full.num_rows != fixture.object_count:
        raise AssertionError(f"full read returned {full.num_rows} rows")
    if projected.num_columns != len(projected_columns):
        raise AssertionError("projected read returned unexpected columns")
    if filtered.num_rows != fixture.objects_per_image:
        raise AssertionError(f"filtered read returned {filtered.num_rows} rows")

    return ShapeBenchmarkResult(
        fixture=fixture.name,
        geometry_encoding=fixture.geometry_encoding,
        compression=compression,
        image_count=fixture.image_count,
        object_count=fixture.object_count,
        measurement_count=fixture.measurement_count,
        write_ms=write_ms,
        full_read_ms=_time(full_read, repeats=repeats, warmup=warmup),
        projected_read_ms=_time(projected_read, repeats=repeats, warmup=warmup),
        filtered_read_ms=_time(filtered_read, repeats=repeats, warmup=warmup),
        file_size_mb=out.stat().st_size / (1024 * 1024),
        projected_columns=len(projected_columns),
        filtered_rows=filtered.num_rows,
    )


def run(
    *,
    repeats: int,
    warmup: int,
    image_count: int,
    objects_per_image: int,
) -> list[ShapeBenchmarkResult]:
    """Run shape Parquet benchmarks."""
    fixtures = [
        ShapeFixture(
            name="cell-centroids",
            geometry_encoding="geoarrow.point",
            image_count=image_count,
            objects_per_image=objects_per_image,
            measurement_count=6,
        ),
        ShapeFixture(
            name="nucleus-label-refs",
            geometry_encoding="ome.labelmask",
            image_count=image_count,
            objects_per_image=objects_per_image,
            measurement_count=6,
        ),
    ]
    results: list[ShapeBenchmarkResult] = []
    with tempfile.TemporaryDirectory(prefix="ome_arrow_shapes_parquet_") as tmp:
        workdir = Path(tmp)
        for fixture in fixtures:
            for compression in (None, "zstd"):
                results.append(
                    _benchmark_fixture(
                        fixture,
                        compression=compression,
                        repeats=repeats,
                        warmup=warmup,
                        workdir=workdir,
                    )
                )
    return results


def _print_results(results: list[ShapeBenchmarkResult]) -> None:
    """Print benchmark results as a compact scientific table."""
    print("")
    print("OME-Arrow Shapes Parquet benchmark")
    print(
        f"{'fixture':20} {'geometry':16} {'codec':8} "
        f"{'images':>6} {'objects':>8} {'meas':>5} "
        f"{'write':>9} {'full':>9} {'project':>9} {'filter':>9} "
        f"{'size MB':>9}"
    )
    print("-" * 118)
    for result in results:
        codec = result.compression or "none"
        print(
            f"{result.fixture:20} {result.geometry_encoding:16} {codec:8} "
            f"{result.image_count:6d} {result.object_count:8d} "
            f"{result.measurement_count:5d} {result.write_ms:9.2f} "
            f"{result.full_read_ms:9.2f} {result.projected_read_ms:9.2f} "
            f"{result.filtered_read_ms:9.2f} {result.file_size_mb:9.2f}"
        )


def main() -> None:
    """Run the command-line benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--image-count", type=int, default=24)
    parser.add_argument("--objects-per-image", type=int, default=2_000)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    results = run(
        repeats=args.repeats,
        warmup=args.warmup,
        image_count=args.image_count,
        objects_per_image=args.objects_per_image,
    )
    _print_results(results)
    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps({"results": [asdict(result) for result in results]}, indent=2)
        )


if __name__ == "__main__":
    main()
