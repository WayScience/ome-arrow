<img width="300" src="https://raw.githubusercontent.com/wayscience/ome-arrow/main/docs/src/_static/ome-arrow-with-text.png?raw=true">

![PyPI - Version](https://img.shields.io/pypi/v/ome-arrow)
[![Build Status](https://github.com/wayscience/ome-arrow/actions/workflows/run-tests.yml/badge.svg?branch=main)](https://github.com/wayscience/ome-arrow/actions/workflows/run-tests.yml?query=branch%3Amain)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Software DOI badge](https://zenodo.org/badge/DOI/10.5281/zenodo.17664969.svg)](https://doi.org/10.5281/zenodo.17664969)

# Open, interoperable, and queryable microscopy images with OME Arrow

OME-Arrow uses [Open Microscopy Environment (OME)](https://github.com/ome) specifications through [Apache Arrow](https://arrow.apache.org/) for fast, queryable, and language agnostic bioimage data.

> 📐 Benchmark results that inform OME Arrow's design decisions are available in the [ome-arrow-benchmarks](https://github.com/WayScience/ome-arrow-benchmarks) repository.

<img height="200" src="https://raw.githubusercontent.com/wayscience/ome-arrow/main/docs/src/_static/references_to_files.png">

__Images are often left behind from the data model, referenced but excluded from databases.__

<img height="200" src="https://raw.githubusercontent.com/wayscience/ome-arrow/main/docs/src/_static/various_ome_arrow_schema.png">

__OME-Arrow brings images back into the story.__

OME Arrow enables image data to be stored alongside metadata or derived data such as single-cell morphology features.
Images in OME Arrow are composed of mutlilayer [structs](https://arrow.apache.org/docs/python/generated/pyarrow.struct.html) so they may be stored as values within tables.
This means you can store, query, and build relationships on data from the same location using any system which is compatible with Apache Arrow (including Parquet) through common data interfaces (such as SQL and DuckDB).

## Project focus

This package is intentionally dedicated to work at a per-image level and not large batch handling (though it may be used for those purposes by users or in other projects).

- For visualizing OME Arrow and OME Parquet data in Napari, please see the [`napari-ome-arrow`](https://github.com/WayScience/napari-ome-arrow) Napari plugin.
- For more comprehensive handling of many images and features in the context of the OME Parquet format please see the [`CytoDataFrame`](https://github.com/cytomining/CytoDataFrame) project (and relevant [example notebook](https://github.com/cytomining/CytoDataFrame/blob/main/docs/src/examples/cytodataframe_at_a_glance.ipynb)).

## Installation

Install OME Arrow from PyPI or from source:

```sh
# install from pypi
pip install ome-arrow

# install directly from source
pip install git+https://github.com/wayscience/ome-arrow.git
```

## Quick start

See below for a quick start guide.
Please also reference an example notebook: [Learning to fly with OME-Arrow](https://github.com/wayscience/ome-arrow/tree/main/docs/src/examples/learning_to_fly_with_ome-arrow.ipynb).

```python
from ome_arrow import OMEArrow

# Ingest a tif image through a convenient OME Arrow class
# We can also ingest OME-Zarr or NumPy arrays.
oa_image = OMEArrow(
    data="your_image.tif"
)

# Access the OME Arrow struct itself
# (compatible with Arrow-compliant data storage).
oa_image.data

# Show information about the image.
oa_image.info()

# Display the image with matplotlib.
oa_image.view(how="matplotlib")

# Display the image with pyvista
# (great for ZYX 3D images; install extras: `pip install 'ome-arrow[viz]'`).
oa_image.view(how="pyvista")

# Export to OME-Parquet. This writes the typed chunk dataset layout.
# We can also export OME-TIFF, OME-Zarr or NumPy arrays.
oa_image.export(how="ome-parquet", out="your_image.ome.parquet")

# Export to Vortex (install extras: `pip install 'ome-arrow[vortex]'`).
oa_image.export(how="vortex", out="your_image.vortex")
```

## Tensor view (DLPack)

For tensor-focused workflows (PyTorch/JAX), use `tensor_view` and DLPack export.

```python
from ome_arrow import OMEArrow

oa = OMEArrow("your_image.ome.parquet")

# Spatial ROI per plane (YX convention)
view = oa.tensor_view(t=0, z=0, roi=(32, 32, 128, 128), layout="CYX")

# Convenience 3D ROI (x, y, z, w, h, d)
view3d = oa.tensor_view(roi3d=(32, 32, 2, 128, 128, 4), layout="TZCYX")

# 3D tiled iteration over (z, y, x)
for cap in view3d.iter_tiles_3d(tile_size=(2, 64, 64), mode="numpy"):
    pass
```

Lazy scan-style convention (Polars-like):

```python
from ome_arrow import OMEArrow

oa = OMEArrow.scan("your_image.ome.parquet")  # deferred load
# First: queue lazy spatial/index slicing
lazy_crop = oa.slice_lazy(0, 512, 0, 512).slice_lazy(64, 256, 64, 256)
cropped = lazy_crop.collect()

# slice_lazy returns a new OMEArrow plan; collect does not mutate `oa`.
# Build tensor_view from the returned sliced object to reuse that plan.
tensor_view_result = cropped.tensor_view(t=0, z=slice(0, 4), roi=(0, 0, 192, 192))
arr = tensor_view_result.to_numpy()
```

Advanced options:

- `chunk_policy="auto" | "combine" | "keep"` controls ChunkedArray handling.
- `channel_policy="error" | "first"` controls behavior when dropping `C` from layout.

See full docs: [`docs/src/dlpack.md`](docs/src/dlpack.md)

## Inline byte-backed OME values

The historical nested table stores pixel payloads as numeric lists inside `chunks[].pixels` and `planes[].pixels`.
For faster one-row-per-image Parquet tables, write inline chunk bytes instead:

```python
from ome_arrow import from_numpy, to_ome_parquet

record = from_numpy(arr, dim_order="TCZYX", chunk_encoding="bytes")
to_ome_parquet(record, "image.ome.parquet", column_name="ome_arrow")
```

You can also convert an existing OME-Arrow record at write time:

```python
to_ome_parquet(
    record,
    "image.ome.parquet",
    column_name="ome_arrow",
    inline_chunk_encoding="bytes",
)
```

This keeps the ergonomic inline OME value while storing chunk payloads as typed `pixel_bytes: large_binary`.
Use it for moderate image-level tables and whole-image reads.
For large 3D/5D selective reads, prefer the typed chunk dataset API below.

Leaf-level chunk compression is also available for inline byte chunks:

```python
record = from_numpy(
    arr,
    dim_order="TCZYX",
    chunk_encoding="bytes",
    chunk_compression="auto",
)

to_ome_parquet(
    record,
    "image.ome.parquet",
    column_name="ome_arrow",
    compression="zstd",
)
```

Compression guidance from `benchmarks/benchmark_inline_byte_compression.py`:

| Data/workload                               | Suggested setting                                                                   | Why                                                                           |
| ------------------------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| General inline-byte tables                  | `chunk_compression="auto"` and Parquet `compression="zstd"`                         | Compresses chunks only when they shrink, then lets Parquet compress metadata. |
| Faster reads on compressible images         | `chunk_compression="fast"` with Parquet `compression=None`                          | Uses LZ4 only when chunks shrink, keeping decode overhead low.                |
| Best storage on compressible 3D/volume data | `chunk_compression="small"` plus Parquet `compression="zstd"`                       | Uses Zstd level 1 only when chunks shrink, then applies Parquet compression.  |
| Noisy/high-entropy images                   | `chunk_compression="auto"` or no leaf compression; use Parquet `compression="zstd"` | Auto skips chunks that would grow; noisy data often does not compress.        |

Explicit codecs such as `chunk_compression="zstd"` with `chunk_compression_level=1` and `chunk_compression="lz4"` are also supported when you want fixed behavior instead of a preset.

## Typed chunk datasets

Typed chunk datasets are the optimized pixel IO path for OME-Arrow.
Their goal is to keep image metadata small and queryable while storing pixels as typed byte chunks that can be read directly by image, plane, channel, region, or volume.
Use this layout when performance matters for selective reads, larger 3D/5D images, or data engineering workflows that need predictable chunk indexing.

`OMEArrow.export(how="ome-parquet")` writes the typed byte-buffer dataset layout.
For explicit control over layout and chunks, use the dataset writer directly.
By default, this stores image metadata separately from pixel chunks and writes one chunk per Parquet row group, so `read_plane()` and `read_region()` can jump through a physical index instead of materializing the older nested struct payload.
You can change that row-group packing with `chunk_rows_per_row_group`.

```python
import numpy as np
from ome_arrow import OMEArrowDataset, write_ome_arrow_dataset

arr = np.zeros((1, 1, 1, 1024, 1024), dtype=np.uint16)  # TCZYX

choice = write_ome_arrow_dataset(
    [arr],
    "image.ome-arrow",
    layout="tile",
    chunk_shape=(1, 1, 1, 512, 512),
    compression="zstd",
    chunk_rows_per_row_group=1,
)
print(choice.rationale)

dataset = OMEArrowDataset("image.ome-arrow")
image_id = dataset.images["image_id"].to_pylist()[0]
plane = dataset.pixels.read_plane(image_id, t=0, c=0, z=0)
crop = dataset.pixels.read_region(image_id, y=slice(128, 384), x=slice(128, 384))

# Dataset-level shortcuts return NumPy by default and can return Torch/JAX
# arrays when those packages are installed.
plane_np = dataset.read_plane(t=0, c=0, z=0)
plane_torch = dataset.read_plane(t=0, c=0, z=0, return_type="torch")
plane_jax = dataset.read_plane(t=0, c=0, z=0, return_type="jax")
```

Use `chunk_rows_per_row_group=1` for the fastest direct chunk reads.
Use a larger value, such as `8`, to reduce row-group overhead for small chunks when storage size matters.

The writer preserves source pixel dtype by default.
To normalize stored pixel buffers explicitly, pass `pixel_dtype`, for example `pixel_dtype="uint16"`.
Integer casts clamp by default; pass `clamp=False` to use NumPy casting behavior directly.

## Tensor ingest (PyTorch/JAX)

You can ingest torch or JAX arrays directly with `OMEArrow(...)`.
You can also use explicit helper functions from `ome_arrow.ingest`.

Why this is useful:

- It reduces compute overhead by removing conversion code boilerplate in separate model/data pipelines that already use torch or JAX tensors (i.e., it provides a direct port of OME-arrow into popular deep learning libraries).
- However, this is more about clean interoperability than dramatic end-to-end speedups (although we expect fewer handoffs to result in speedups). Specifically:
- It makes it easier for a user to update dimension ordering input in the same place without requiring separate functionality (see argument `dim_order`).
- This smooths handoffs and reduces mistakes when moving between tensor layouts and OME-Arrow records. For example, CPU torch tensors often expose a NumPy view without an extra copy.
- Ingest still materializes OME-Arrow planes/chunks.

```python
from ome_arrow import OMEArrow

# Direct constructor support:
# inferred defaults are rank-based:
# 2D -> "YX", 3D -> "ZYX", 4D -> "TCYX", 5D -> "TCZYX"
oa_torch = OMEArrow(torch_tensor)
oa_jax = OMEArrow(jax_array)

# Optional: override dim order when shape is ambiguous
oa_zyx = OMEArrow(torch_volume, dim_order="ZYX")
```

```python
from ome_arrow.ingest import from_torch_array, from_jax_array

scalar_torch = from_torch_array(torch_tensor, dim_order="TCYX")
scalar_jax = from_jax_array(jax_array, dim_order="TCYX")
```

Notes:

- Torch/JAX support is optional.
- Install extras as needed: `pip install "ome-arrow[dlpack-torch]"` or `pip install "ome-arrow[dlpack-jax]"`.
- Torch tensors are detached and converted on CPU for ingest.
- `dim_order` is accepted only for NumPy/torch/JAX array inputs.
- Ingest now passes flattened NumPy pixel buffers directly to Arrow.
- This avoids materializing Python `list` payloads per plane/chunk.

## Benchmarking lazy reads

Use the lightweight benchmark utility in `benchmarks/` to compare lazy tensor read paths (TIFF source-backed, Parquet planes, Parquet chunks).
For more detailed benchmark results and analysis, see the [ome-arrow-benchmarks](https://github.com/WayScience/ome-arrow-benchmarks) repository.

```bash
uv run python benchmarks/benchmark_lazy_tensor.py --repeats 5 --warmup 1
```

For OME-IRIS-style 2D/3D/4D/5D access patterns, use `benchmark_ome_iris.py`.
This benchmark is intended to answer practical questions about pixel IO: how fast each format writes a matched artifact, how fast it reads full images or volumes, and how fast it serves selective access patterns such as planes, crops, subvolumes, timepoints, and channels.

```bash
uv run python benchmarks/benchmark_ome_iris.py --repeats 3 --warmup 1
```

By default, the benchmark uses local test-data fixtures when available.
You can also pass real local TIFF fixtures explicitly:

```bash
uv run python benchmarks/benchmark_ome_iris.py \
  --fixture 2d=/path/to/plate-image.tif \
  --fixture 3d=/path/to/volume.tif \
  --fixture 5d=/path/to/tczyx-image.ome.tif \
  --repeats 3 \
  --warmup 1 \
  --json-out benchmark-results.json
```

Each `--fixture` argument is `name=/path/to/image.tif`.
The `name` label is used only in the output table, so choose labels that describe the dimensionality or dataset source.
Inputs must be TIFF files; the benchmark creates temporary matched OME-Zarr and OME-Arrow artifacts for the same source image, then reports latency, returned shape, dtype, and artifact size.
Temporary artifacts are deleted automatically after the run.

Use the printed table for quick local iteration and `--json-out` when comparing runs over time or attaching results to an issue/PR.
Prefer multiple repeats when making performance claims, because local filesystem cache, codec warmup, and Torch/JAX initialization can affect single-run timings.

The OME-IRIS-style benchmark separates return/API paths:

- `ome-zarr-tensor-numpy`: OME-Arrow `tensor_view(...).to_numpy()` over OME-Zarr.
- `ome-zarr-bioio-numpy`: direct BioImage NumPy reads over OME-Zarr.
- `ome-tiff-tensor-numpy`: OME-Arrow `tensor_view(...).to_numpy()` over TIFF.
- `ome-tiff-bioio-numpy`: direct BioImage NumPy reads over TIFF.
- `ome-arrow-src-numpy`: source-dtype typed OME-Arrow dataset NumPy reads.
- `ome-arrow-u16-numpy`: typed OME-Arrow dataset NumPy reads normalized to `uint16` for apples-to-apples comparisons with normalized paths.
- `ome-arrow-u16-raw-numpy`: normalized `uint16` typed OME-Arrow reads with uncompressed chunk bytes for local speed comparisons.
- `ome-arrow-*-chunks`: Arrow-native raw chunk-row reads that return `pixel_bytes` without decoding into NumPy.
- `ome-tiff-tensor-torch` / `ome-tiff-tensor-jax`: OME-Arrow tensor-view Torch/JAX returns over TIFF.
- `ome-zarr-tensor-torch` / `ome-zarr-tensor-jax`: OME-Arrow tensor-view Torch/JAX returns over OME-Zarr.
- `ome-arrow-src-torch` / `ome-arrow-src-jax`: source-dtype typed OME-Arrow dataset reads with `return_type="torch"` or `return_type="jax"`.
- `ome-arrow-u16-torch` / `ome-arrow-u16-jax`: normalized `uint16` typed OME-Arrow dataset reads with Torch/JAX returns.

Notes:

- This benchmark is for local iteration and relative comparisons.
- It is not part of CI pass/fail checks.
- CI also runs this benchmark in a dedicated `benchmark_canary` job and uploads `benchmark-results.json` as a workflow artifact.

Recalibrating `benchmarks/ci-baseline.json`:

1. Run the benchmark on `main` a few times (for example 3-5 runs):
   `uv run python benchmarks/benchmark_lazy_tensor.py --repeats 7 --warmup 2 --json-out benchmark-results.json`
1. For each case, collect the observed `median_ms` values.
1. Update `benchmarks/ci-baseline.json` with stable medians from those runs (prefer a conservative value near the slower side, not the fastest sample).
1. Keep CI canary tolerance (`regression_factor` + `absolute_slack_ms`) unchanged unless you have repeated false positives.

## Contributing, Development, and Testing

Please see our [contributing documentation](https://github.com/wayscience/ome-arrow/tree/main/CONTRIBUTING.md) for more details on contributions, development, and testing.

## Related projects

OME Arrow is used or inspired by the following projects, check them out!

- [`napari-ome-arrow`](https://github.com/WayScience/napari-ome-arrow): enables you to view OME Arrow and related images.
- [`CytoDataFrame`](https://github.com/cytomining/CytoDataFrame): provides a DataFrame-like experience for viewing feature and microscopy image data within Jupyter notebook interfaces and creating OME Parquet files.
- [`coSMicQC`](https://github.com/cytomining/coSMicQC): performs quality control on microscopy feature datasets, visualized using CytoDataFrames.
- [`pycytominer`](https://github.com/cytomining/pycytominer): supports feature profiling, normalization, and downstream analysis workflows for image-based profiling datasets.
- [`iceberg-bioimage`](https://github.com/WayScience/iceberg-bioimage): defines warehouse-oriented patterns for connecting bioimage formats and analytical tables at scale.
- [`ome-arrow-benchmarks`](https://github.com/WayScience/ome-arrow-benchmarks): contains benchmark results and analysis that inform the design choices in OME Arrow.
- [`CytoTable`](https://github.com/cytomining/CytoTable): converts image-based profiling outputs into analysis-ready tabular formats such as Parquet.
