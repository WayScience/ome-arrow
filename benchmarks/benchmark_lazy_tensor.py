"""Lightweight benchmark for lazy tensor read paths.

This script compares `OMEArrow.scan(...).tensor_view(...).to_numpy()` across:
- TIFF source-backed lazy plane loading
- OME-Parquet (planes payload)
- OME-Parquet (chunked payload)

It is intended as a quick local signal, not a rigorous microbenchmark.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ome_arrow import OMEArrow
from ome_arrow.export import to_ome_parquet
from ome_arrow.ingest import from_numpy


@dataclass(frozen=True)
class BenchmarkResult:
    """Summary stats for one benchmark case."""

    name: str
    median_ms: float
    min_ms: float
    max_ms: float
    shape: tuple[int, ...]


@dataclass(frozen=True)
class RegressionCheck:
    """Regression check output for one benchmark case."""

    name: str
    baseline_ms: float | None
    threshold_ms: float | None
    regressed: bool


def _time_case(
    name: str,
    fn: Callable[[], np.ndarray],
    *,
    repeats: int,
    warmup: int,
) -> BenchmarkResult:
    """Run one benchmark case and return timing stats."""
    out: np.ndarray | None = None
    for _ in range(warmup):
        out = fn()

    times_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    if out is None:
        raise RuntimeError("Benchmark case did not produce output.")
    return BenchmarkResult(
        name=name,
        median_ms=statistics.median(times_ms),
        min_ms=min(times_ms),
        max_ms=max(times_ms),
        shape=tuple(out.shape),
    )


def _build_parquet_fixtures(workdir: Path) -> tuple[Path, Path]:
    """Create small planes/chunks parquet fixtures for local benchmarking."""
    arr = np.arange(1 * 2 * 3 * 256 * 256, dtype=np.uint16).reshape(1, 2, 3, 256, 256)

    planes_scalar = from_numpy(arr, build_chunks=False, image_id="bench-planes")
    chunks_scalar = from_numpy(
        arr,
        build_chunks=True,
        chunk_shape=(1, 64, 64),
        image_id="bench-chunks",
    )

    planes_path = workdir / "bench_planes.ome.parquet"
    chunks_path = workdir / "bench_chunks.ome.parquet"
    to_ome_parquet(planes_scalar, out_path=str(planes_path), column_name="ome_arrow")
    to_ome_parquet(chunks_scalar, out_path=str(chunks_path), column_name="ome_arrow")
    return planes_path, chunks_path


def _print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results in a compact table."""
    print("")
    print("Lazy tensor benchmark (ms)")
    print(f"{'case':38} {'median':>10} {'min':>10} {'max':>10} {'shape':>16}")
    print("-" * 92)
    for r in results:
        print(
            f"{r.name:38} {r.median_ms:10.2f} {r.min_ms:10.2f} {r.max_ms:10.2f} {str(r.shape):>16}"
        )


def _load_baseline(path: Path | None) -> dict[str, float]:
    """Load baseline medians from JSON, if provided."""
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text())
    cases = payload.get("cases", {})
    return {str(k): float(v) for k, v in cases.items()}


def _check_regressions(
    results: list[BenchmarkResult],
    *,
    baseline: dict[str, float],
    regression_factor: float,
    absolute_slack_ms: float,
) -> list[RegressionCheck]:
    """Compare benchmark medians against baseline thresholds."""
    checks: list[RegressionCheck] = []
    for r in results:
        baseline_ms = baseline.get(r.name)
        if baseline_ms is None:
            checks.append(
                RegressionCheck(
                    name=r.name,
                    baseline_ms=None,
                    threshold_ms=None,
                    regressed=False,
                )
            )
            continue

        threshold_ms = baseline_ms * regression_factor + absolute_slack_ms
        checks.append(
            RegressionCheck(
                name=r.name,
                baseline_ms=baseline_ms,
                threshold_ms=threshold_ms,
                regressed=r.median_ms > threshold_ms,
            )
        )
    return checks


def _print_regressions(checks: list[RegressionCheck]) -> None:
    """Print regression-comparison details."""
    with_baseline = [c for c in checks if c.baseline_ms is not None]
    if not with_baseline:
        print("\nNo baseline cases configured; skipping regression checks.")
        return
    print("\nCanary comparison")
    print(f"{'case':38} {'baseline':>10} {'threshold':>10} {'status':>10}")
    print("-" * 74)
    for c in checks:
        if c.baseline_ms is None or c.threshold_ms is None:
            status = "no-baseline"
            baseline = "-"
            threshold = "-"
        else:
            status = "regressed" if c.regressed else "ok"
            baseline = f"{c.baseline_ms:.2f}"
            threshold = f"{c.threshold_ms:.2f}"
        print(f"{c.name:38} {baseline:>10} {threshold:>10} {status:>10}")


def _write_json_report(
    path: Path,
    *,
    results: list[BenchmarkResult],
    checks: list[RegressionCheck],
    repeats: int,
    warmup: int,
) -> None:
    """Write a machine-readable benchmark report for CI artifacts."""
    payload = {
        "repeats": repeats,
        "warmup": warmup,
        "results": [
            {
                "name": r.name,
                "median_ms": r.median_ms,
                "min_ms": r.min_ms,
                "max_ms": r.max_ms,
                "shape": list(r.shape),
            }
            for r in results
        ],
        "regression_checks": [
            {
                "name": c.name,
                "baseline_ms": c.baseline_ms,
                "threshold_ms": c.threshold_ms,
                "regressed": c.regressed,
            }
            for c in checks
        ],
    }
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    """Run lightweight lazy tensor benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5, help="Timed iterations.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations.")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON output path for CI/reporting.",
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help="Optional baseline JSON with {'cases': {'case-name': median_ms}}.",
    )
    parser.add_argument(
        "--regression-factor",
        type=float,
        default=1.25,
        help="Allowed multiplicative slowdown vs baseline.",
    )
    parser.add_argument(
        "--absolute-slack-ms",
        type=float,
        default=2.0,
        help="Allowed absolute slowdown slack in ms.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero when any case exceeds regression threshold.",
    )
    parser.add_argument(
        "--tiff-path",
        type=Path,
        default=Path("tests/data/ome-artificial-5d-datasets/single-channel.ome.tiff"),
        help="TIFF file used for the source-backed lazy case.",
    )
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0.")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0.")
    if args.regression_factor <= 0:
        raise ValueError("--regression-factor must be > 0.")
    if args.absolute_slack_ms < 0:
        raise ValueError("--absolute-slack-ms must be >= 0.")

    with tempfile.TemporaryDirectory(prefix="ome_arrow_lazy_bench_") as tmp:
        tmpdir = Path(tmp)
        planes_path, chunks_path = _build_parquet_fixtures(tmpdir)

        cases: list[tuple[str, Callable[[], np.ndarray]]] = []
        if args.tiff_path.exists():
            cases.append(
                (
                    "scan+tiff -> tensor_view(YX)",
                    lambda: OMEArrow.scan(str(args.tiff_path))
                    .tensor_view(t=0, z=0, c=0, layout="YX")
                    .to_numpy(contiguous=True),
                )
            )
        else:
            print(f"Skipping TIFF case; file not found: {args.tiff_path}")

        cases.extend(
            [
                (
                    "scan+parquet(planes) -> tensor_view(YX)",
                    lambda: OMEArrow.scan(str(planes_path))
                    .tensor_view(t=0, z=1, c=1, layout="YX")
                    .to_numpy(contiguous=True),
                ),
                (
                    "scan+parquet(chunks) -> tensor_view(YX)",
                    lambda: OMEArrow.scan(str(chunks_path))
                    .tensor_view(t=0, z=1, c=1, layout="YX")
                    .to_numpy(contiguous=True),
                ),
            ]
        )

        results = [
            _time_case(name, fn, repeats=args.repeats, warmup=args.warmup)
            for name, fn in cases
        ]
        _print_results(results)
        baseline = _load_baseline(args.baseline_json)
        checks = _check_regressions(
            results,
            baseline=baseline,
            regression_factor=args.regression_factor,
            absolute_slack_ms=args.absolute_slack_ms,
        )
        _print_regressions(checks)
        if args.json_out is not None:
            _write_json_report(
                args.json_out,
                results=results,
                checks=checks,
                repeats=args.repeats,
                warmup=args.warmup,
            )
        if args.fail_on_regression and any(c.regressed for c in checks):
            sys.exit(1)


if __name__ == "__main__":
    main()
