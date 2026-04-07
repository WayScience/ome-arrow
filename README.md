<img width="600" src="https://raw.githubusercontent.com/wayscience/ome-arrow/main/docs/src/_static/logo.png?raw=true">

![PyPI - Version](https://img.shields.io/pypi/v/ome-arrow)
[![Build Status](https://github.com/wayscience/ome-arrow/actions/workflows/run-tests.yml/badge.svg?branch=main)](https://github.com/wayscience/ome-arrow/actions/workflows/run-tests.yml?query=branch%3Amain)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Software DOI badge](https://zenodo.org/badge/DOI/10.5281/zenodo.17664969.svg)](https://doi.org/10.5281/zenodo.17664969)

# Open, interoperable, and queryable microscopy images with OME Arrow

OME-Arrow uses [Open Microscopy Environment (OME)](https://github.com/ome) specifications through [Apache Arrow](https://arrow.apache.org/) for fast, queryable, and language agnostic bioimage data.

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

# Export to OME-Parquet.
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

## Tensor ingest (PyTorch/JAX)

You can ingest torch or JAX arrays directly with `OMEArrow(...)`.
You can also use explicit helper functions from `ome_arrow.ingest`.

Why this is useful:

- It removes conversion boilerplate in model/data pipelines that already use torch or JAX tensors.
- It keeps axis handling in one place (`dim_order`).
- This reduces mistakes when moving between tensor layouts and OME-Arrow records.
- It can reduce overhead in some paths.
- For example, CPU torch tensors often expose a NumPy view without an extra copy.
- Ingest still materializes OME-Arrow planes/chunks.
- This is more about clean interoperability than dramatic end-to-end speedups.

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
- Install extras as needed:
  `pip install "ome-arrow[dlpack-torch]"` or `pip install "ome-arrow[dlpack-jax]"`.
- Torch tensors are detached and converted on CPU for ingest.
- `dim_order` is accepted only for NumPy/torch/JAX array inputs.
- Ingest now passes flattened NumPy pixel buffers directly to Arrow.
- This avoids materializing Python `list` payloads per plane/chunk.

## Benchmarking lazy reads

Use the lightweight benchmark utility in `benchmarks/` to compare lazy tensor
read paths (TIFF source-backed, Parquet planes, Parquet chunks):

```bash
uv run python benchmarks/benchmark_lazy_tensor.py --repeats 5 --warmup 1
```

Notes:

- This benchmark is for local iteration and relative comparisons.
- It is not part of CI pass/fail checks.
- CI also runs this benchmark in a dedicated `benchmark_canary` job and
  uploads `benchmark-results.json` as a workflow artifact.

Recalibrating `benchmarks/ci-baseline.json`:

1. Run the benchmark on `main` a few times (for example 3-5 runs):
   `uv run python benchmarks/benchmark_lazy_tensor.py --repeats 7 --warmup 2 --json-out benchmark-results.json`
1. For each case, collect the observed `median_ms` values.
1. Update `benchmarks/ci-baseline.json` with stable medians from those runs
   (prefer a conservative value near the slower side, not the fastest sample).
1. Keep CI canary tolerance (`regression_factor` + `absolute_slack_ms`) unchanged
   unless you have repeated false positives.

## Contributing, Development, and Testing

Please see our [contributing documentation](https://github.com/wayscience/ome-arrow/tree/main/CONTRIBUTING.md) for more details on contributions, development, and testing.

## Related projects

OME Arrow is used or inspired by the following projects, check them out!

- [`napari-ome-arrow`](https://github.com/WayScience/napari-ome-arrow): enables you to view OME Arrow and related images.
- [`nViz`](https://github.com/WayScience/nViz): focuses on ingesting and visualizing various 3D image data.
- [`CytoDataFrame`](https://github.com/cytomining/CytoDataFrame): provides a DataFrame-like experience for viewing feature and microscopy image data within Jupyter notebook interfaces and creating OME Parquet files.
- [`coSMicQC`](https://github.com/cytomining/coSMicQC): performs quality control on microscopy feature datasets, visualized using CytoDataFrames.
