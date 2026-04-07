# Exporting OME-Arrow pixel data via DLPack

OME-Arrow exposes a small tensor view API for pixel data. The returned
`TensorView` can export DLPack capsules for zero-copy interoperability on CPU
and (optionally) GPU.

Key defaults:

- OME-Arrow tensor layouts always include channels (`C`) as a tensor axis.
- Default layout is `CHW` (equivalent to `CYX`) when both `T` and `Z` are singleton in the source.
- Otherwise, default layout is `TZCHW` (equivalent to `TZCYX`, with singleton `T`/`Z` retained unless you override layout).
- You can override with any valid TZCHW/TZCYX permutation/subset, for example `YXC`, `ZCYX`, or `CYX`.

Layout nomenclature:

- `T`: time index
- `Z`: z/depth index
- `C`: channel index
- `Y`: image row axis (height)
- `X`: image column axis (width)
  (`H/W` aliases are also accepted for compatibility).

Practical mapping:

- 2D image content (`YX`) is typically exposed as `CYX`.
- 3D z-stack content (`ZYX`) is typically exposed as `ZCYX` or `TZCYX` (with `T=1`).
- Time-lapse and volumetric content use `TZCYX`/`TZCHW` by default.

## PyTorch

```python
from ome_arrow import OMEArrow

obj = OMEArrow("example.ome.parquet")
view = obj.tensor_view(t=0, z=0, c=0)

# DLPack capsule -> torch.Tensor
import torch

capsule = view.to_dlpack(mode="arrow", device="cpu")
flat = torch.utils.dlpack.from_dlpack(capsule)
tensor = flat.reshape(view.shape)
```

You can also ingest torch tensors directly:

```python
from ome_arrow import OMEArrow
import torch

# 2D tensor interpreted as YX by default.
torch_tensor = torch.randint(0, 256, (128, 128), dtype=torch.uint16)
oa = OMEArrow(torch_tensor)

# 3D tensors are inferred as ZYX by default.
# Use dim_order when your tensor is arranged differently (for example CYX).
torch_volume = torch.randint(0, 256, (16, 128, 128), dtype=torch.uint16)
oa_cyx = OMEArrow(torch_volume, dim_order="CYX")
```

Use `dim_order` when the inferred axis order does not match your tensor layout.
`dim_order` is only supported for array/tensor ingest paths.

To persist with this interpreted axis mapping, export the resulting OME-Arrow
record (for example to parquet):

```python
from ome_arrow import OMEArrow
import torch

torch_volume = torch.randint(0, 256, (16, 128, 128), dtype=torch.uint16)
oa = OMEArrow(torch_volume, dim_order="ZYX")
oa.export(how="parquet", out="volume.ome.parquet")
```

OME-Arrow stores pixels in canonical OME-style fields (`size_t`, `size_c`,
`size_z`, `size_y`, `size_x`) rather than preserving a free-form input label
string. The interpreted mapping is preserved through those axis sizes and can
be read back with `tensor_view(...)` layouts.

"Batch" dimension note:

- There is no separate `B` axis in the OME-Arrow schema.
- For model batches, map batch to `T` during ingest.
- Examples:
  - `B,C,Y,X` -> use `dim_order="TCYX"`
  - `B,C,Z,Y,X` -> use `dim_order="TCZYX"`
  - `B,Y,X,C` -> use `dim_order="TYXC"`
- If `T` is already meaningful in your data, represent batch as table rows
  (one OME-Arrow record per batch item) instead of overloading another image axis.

## Lazy scan-style slicing

```python
from ome_arrow import OMEArrow

obj = OMEArrow.scan("example.ome.parquet")
# Prioritize lazy slice planning first.
lazy_crop = obj.slice_lazy(0, 512, 0, 512).slice_lazy(64, 256, 64, 256)
cropped = lazy_crop.collect()

# Then execute tensor selections on the sliced result.
tensor_view = cropped.tensor_view(t=0, z=slice(0, 8), roi=(64, 64, 128, 128))
arr = tensor_view.to_numpy()

# Note: executing a LazyTensorView from OMEArrow.scan(...) does not
# materialize the original OMEArrow object itself.
# Call obj.collect() explicitly if you need to materialize `obj`.
```

## JAX

```python
from ome_arrow import OMEArrow

obj = OMEArrow("example.ome.parquet")
view = obj.tensor_view(t=0, z=0, c=0, layout="CYX")

import jax.numpy as jnp

capsule = view.to_dlpack(mode="arrow", device="cpu")
flat = jnp.from_dlpack(capsule)
arr = flat.reshape(view.shape)
```

You can also ingest JAX arrays directly:

```python
from ome_arrow import OMEArrow
import jax.numpy as jnp

# 2D array interpreted as YX by default.
jax_array = jnp.arange(128 * 128, dtype=jnp.uint16).reshape(128, 128)
oa = OMEArrow(jax_array)

# 3D arrays are inferred as ZYX by default.
# Use dim_order when your array is arranged differently (for example CYX).
jax_volume = jnp.arange(16 * 128 * 128, dtype=jnp.uint16).reshape(16, 128, 128)
oa_cyx = OMEArrow(jax_volume, dim_order="CYX")
```

## Iteration examples

```python
from ome_arrow import OMEArrow
import numpy as np

obj = OMEArrow("example.ome.parquet")
view = obj.tensor_view()

# Batch over time (T) dimension.
for cap in view.iter_dlpack(batch_size=2, shuffle=False, mode="numpy"):
    batch = np.from_dlpack(cap)
    # batch shape: (batch, Z, C, Y, X) in TZCYX layout
```

```python
from ome_arrow import OMEArrow
import numpy as np

obj = OMEArrow("example.ome.parquet")
view = obj.tensor_view(t=0, z=0)

# Tile over spatial region.
for cap in view.iter_dlpack(
    tile_size=(256, 256), shuffle=True, seed=123, mode="numpy"
):
    tile = np.from_dlpack(cap)
    # tile shape: (C, Y, X) in CYX layout
```

## Ownership and lifetime

`TensorView.to_dlpack()` returns a DLPack-capable object (with `__dlpack__`)
that references the underlying Arrow values buffer in `mode="arrow"`, or a
NumPy buffer in `mode="numpy"`. Keep the `TensorView` (or any NumPy array
returned by `to_numpy`) alive until the consumer finishes using the DLPack
object.

`mode="arrow"` currently requires a single `(t, z, c)` selection and a full-frame
ROI. Use `mode="numpy"` for batches, crops, or layout reshaping beyond a simple
reshape.

Zero-copy guarantees depend on the source: Arrow-backed inputs preserve buffers,
while records built from Python lists or NumPy arrays will materialize once into
Arrow buffers. The same applies to `StructScalar` inputs, which are normalized
through Python objects before Arrow-mode export.
For Parquet/Vortex sources, zero-copy also requires the on-disk struct schema
to match `OME_ARROW_STRUCT`; non-strict schema normalization materializes via
Python objects.

## Optional dependencies

CPU DLPack export uses Arrow buffers by default. For framework helpers and GPU
paths, install only what you need:

```bash
pip install "ome-arrow[dlpack-torch]"  # torch only
pip install "ome-arrow[dlpack-jax]"    # jax only
pip install "ome-arrow[dlpack]"        # both
```

## Benchmarking lazy reads

To quickly compare lazy tensor read paths (TIFF source-backed execution,
Parquet planes, Parquet chunks), run:

```bash
uv run python benchmarks/benchmark_lazy_tensor.py --repeats 5 --warmup 1
```

This is a lightweight local benchmark intended for directional performance
checks during development.

In CI, the `tests` workflow runs a `benchmark_canary` job that executes the
same script and uploads a JSON report artifact.

### Recalibrating `ci-baseline.json`

When performance changes are intentional (or runner behavior shifts), update
`benchmarks/ci-baseline.json` as follows:

1. Check out the latest `main`.
1. Run the benchmark multiple times:
   `uv run python benchmarks/benchmark_lazy_tensor.py --repeats 7 --warmup 2 --json-out benchmark-results.json`
1. Record `median_ms` per case across runs.
1. Set each baseline value to a stable, slightly conservative median.
1. Open a PR that updates baseline values only, with benchmark evidence.

Expected variability:

- Small fluctuations are normal on GitHub-hosted runners.
- Relative ordering of cases is usually stable.
- Typical drift should be modest, but occasional jumps can happen due to
  runner image or dependency changes.
