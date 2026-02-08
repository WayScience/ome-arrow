# Exporting OME-Arrow pixel data via DLPack

OME-Arrow exposes a small tensor view API for pixel data. The returned
`TensorView` can export DLPack capsules for zero-copy interoperability on CPU
and (optionally) GPU.

Key defaults:

- 2D views default to `CHW` layout.
- 5D views default to `TZCHW` layout.
- Use `layout="HWC"` (or any TZCHW permutation) to override.

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

## JAX

```python
from ome_arrow import OMEArrow

obj = OMEArrow("example.ome.parquet")
view = obj.tensor_view(t=0, z=0, c=0, layout="HWC")

import jax

capsule = view.to_dlpack(mode="arrow", device="cpu")
flat = jax.dlpack.from_dlpack(capsule)
arr = flat.reshape(view.shape)
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
    # batch shape: (batch, Z, C, H, W) in TZCHW layout
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
    # tile shape: (C, H, W) in CHW layout
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
