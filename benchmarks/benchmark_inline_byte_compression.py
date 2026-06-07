"""Benchmark inline byte-backed OME values with leaf-level compression."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ome_arrow import from_numpy
from ome_arrow.export import to_numpy, to_ome_parquet
from ome_arrow.ingest import from_ome_parquet


@dataclass(frozen=True)
class Result:
    """One inline byte compression benchmark result."""

    dataset: str
    codec: str
    parquet_compression: str | None
    write_ms: float
    read_ms: float
    size_mb: float


def _time(fn: Callable[[], object], *, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


def _make_arrays() -> dict[str, np.ndarray]:
    y, x = np.mgrid[:512, :512]
    smooth = ((y * 3 + x * 5) % 4096).astype(np.uint16)
    rng = np.random.default_rng(42)
    noisy = rng.integers(0, 65535, size=(512, 512), dtype=np.uint16)
    volume = np.stack(
        [((smooth + z * 17) % 4096).astype(np.uint16) for z in range(16)],
        axis=0,
    )
    return {
        "2d-smooth": smooth.reshape(1, 1, 1, 512, 512),
        "2d-noisy": noisy.reshape(1, 1, 1, 512, 512),
        "3d-smooth": volume.reshape(1, 1, 16, 512, 512),
    }


def _cases() -> list[tuple[str, str | None, int | None, str | None]]:
    return [
        ("leaf-none/parquet-none", None, None, None),
        ("leaf-none/parquet-zstd", None, None, "zstd"),
        ("leaf-auto/parquet-none", "auto", None, None),
        ("leaf-auto/parquet-zstd", "auto", None, "zstd"),
        ("leaf-fast/parquet-none", "fast", None, None),
        ("leaf-small/parquet-zstd", "small", None, "zstd"),
        ("leaf-lz4/parquet-none", "lz4", None, None),
        ("leaf-zstd1/parquet-none", "zstd", 1, None),
        ("leaf-zstd3/parquet-none", "zstd", 3, None),
        ("leaf-zstd1/parquet-zstd", "zstd", 1, "zstd"),
        ("leaf-brotli3/parquet-none", "brotli", 3, None),
    ]


def run(*, repeats: int, warmup: int) -> list[Result]:
    """Run inline byte compression benchmarks."""
    results: list[Result] = []
    with tempfile.TemporaryDirectory(prefix="ome_arrow_inline_compression_") as tmp:
        tmpdir = Path(tmp)
        for dataset, arr in _make_arrays().items():
            base = from_numpy(
                arr,
                dim_order="TCZYX",
                chunk_shape=(1, 256, 256),
                build_chunks=True,
            )
            expected = arr
            for label, codec, level, parquet_compression in _cases():
                filename_label = label.replace("/", "_")
                out = tmpdir / f"{dataset}.{filename_label}.ome.parquet"

                def write() -> None:
                    to_ome_parquet(
                        base,
                        str(out),
                        column_name="ome_arrow",
                        compression=parquet_compression,
                        inline_chunk_encoding="bytes",
                        inline_chunk_compression=codec,
                        inline_chunk_compression_level=level,
                    )

                write_ms = _time(write, repeats=repeats, warmup=warmup)

                def read() -> np.ndarray:
                    value = from_ome_parquet(out, column_name="ome_arrow")
                    decoded = to_numpy(value, dtype=expected.dtype)
                    if decoded.shape != expected.shape:
                        raise AssertionError(
                            f"decoded shape {decoded.shape} != {expected.shape}"
                        )
                    return decoded

                decoded = read()
                np.testing.assert_array_equal(decoded, expected)
                read_ms = _time(read, repeats=repeats, warmup=warmup)
                results.append(
                    Result(
                        dataset=dataset,
                        codec=label,
                        parquet_compression=parquet_compression,
                        write_ms=write_ms,
                        read_ms=read_ms,
                        size_mb=out.stat().st_size / (1024 * 1024),
                    )
                )
    return results


def main() -> None:
    """Run the command-line benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    results = run(repeats=args.repeats, warmup=args.warmup)
    print("")
    print("Inline byte compression benchmark")
    print(
        f"{'dataset':12} {'codec':26} {'write ms':>10} {'read ms':>10} {'size MB':>10}"
    )
    print("-" * 72)
    for result in results:
        print(
            f"{result.dataset:12} {result.codec:26} "
            f"{result.write_ms:10.2f} {result.read_ms:10.2f} "
            f"{result.size_mb:10.2f}"
        )
    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps({"results": [asdict(r) for r in results]}, indent=2)
        )


if __name__ == "__main__":
    main()
