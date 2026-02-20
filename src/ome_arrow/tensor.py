"""Tensor view utilities for OME-Arrow pixel data."""

from __future__ import annotations

import random
import threading
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Literal, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ome_arrow.export import plane_from_chunks
from ome_arrow.meta import OME_ARROW_STRUCT

_TZCHW = "TZCHW"
_ALLOWED_DIMS = set(_TZCHW)
_ALLOWED_MODES = {"arrow", "numpy"}


class _Unset:
    """Typed sentinel for arguments that were not provided."""


_UNSET: _Unset = _Unset()


@dataclass(frozen=True)
class _Selection:
    """Normalized index selection for a TensorView.

    Attributes:
        t: Selected time indices.
        z: Selected depth indices.
        c: Selected channel indices.
        roi: Spatial ROI as `(x, y, w, h)` in pixel coordinates. This ROI is
            2D and is applied per selected `(t, z, c)` plane; time/depth/channel
            are represented separately by `t`, `z`, and `c`.
    """

    t: List[int]
    z: List[int]
    c: List[int]
    roi: tuple[int, int, int, int]


class LazyTensorView:
    """Deferred TensorView plan with Polars-style collect semantics."""

    def __init__(
        self,
        *,
        loader: Callable[
            [],
            dict[str, Any] | pa.StructScalar | pa.StructArray | pa.ChunkedArray,
        ],
        t: int | slice | Sequence[int] | None = None,
        z: int | slice | Sequence[int] | None = None,
        c: int | slice | Sequence[int] | None = None,
        roi: tuple[int, int, int, int] | None = None,
        roi3d: tuple[int, int, int, int, int, int] | None = None,
        tile: tuple[int, int] | None = None,
        layout: str | None = None,
        dtype: np.dtype | None = None,
        chunk_policy: Literal["auto", "combine", "keep"] = "auto",
        channel_policy: Literal["error", "first"] = "error",
    ) -> None:
        """Initialize a deferred TensorView plan.

        Args:
            loader: Callable that returns concrete OME-Arrow pixel data.
            t: Time index selection.
            z: Depth index selection.
            c: Channel index selection.
            roi: Spatial crop as ``(x, y, w, h)``.
            roi3d: Spatial + depth crop as ``(x, y, z, w, h, d)``.
            tile: Tile index as ``(tile_y, tile_x)``.
            layout: Requested output layout (TZCHW letters).
            dtype: Output dtype override.
            chunk_policy: Chunk handling strategy for ChunkedArray inputs.
            channel_policy: Behavior when dropping ``C`` from layout.
        """
        self._loader = loader
        self._kwargs: dict[str, Any] = {
            "t": t,
            "z": z,
            "c": c,
            "roi": roi,
            "roi3d": roi3d,
            "tile": tile,
            "layout": layout,
            "dtype": dtype,
            "chunk_policy": chunk_policy,
            "channel_policy": channel_policy,
        }
        self._resolved: TensorView | None = None
        self._collect_lock = threading.Lock()

    def _spawn(self, **updates: Any) -> "LazyTensorView":
        kwargs = dict(self._kwargs)
        kwargs.update(updates)
        return LazyTensorView(loader=self._loader, **kwargs)

    def collect(self) -> "TensorView":
        """Materialize this lazy plan into a concrete TensorView."""
        resolved = self._resolved
        if resolved is not None:
            return resolved

        with self._collect_lock:
            resolved = self._resolved
            if resolved is None:
                resolved = TensorView(self._loader(), **self._kwargs)
                self._resolved = resolved
        return resolved

    def with_layout(self, layout: str) -> "LazyTensorView":
        """Return a new lazy view with an updated layout."""
        return self._spawn(layout=layout)

    def select(
        self,
        *,
        t: int | slice | Sequence[int] | None | _Unset = _UNSET,
        z: int | slice | Sequence[int] | None | _Unset = _UNSET,
        c: int | slice | Sequence[int] | None | _Unset = _UNSET,
        roi: tuple[int, int, int, int] | None | _Unset = _UNSET,
        roi3d: tuple[int, int, int, int, int, int] | None | _Unset = _UNSET,
        tile: tuple[int, int] | None | _Unset = _UNSET,
    ) -> "LazyTensorView":
        """Return a new lazy plan with updated index/ROI selections."""
        updates = {}
        if t is not _UNSET:
            updates["t"] = t
        if z is not _UNSET:
            updates["z"] = z
        if c is not _UNSET:
            updates["c"] = c
        if roi is not _UNSET:
            updates["roi"] = roi
        if roi3d is not _UNSET:
            updates["roi3d"] = roi3d
        if tile is not _UNSET:
            updates["tile"] = tile
        return self._spawn(**updates)

    @property
    def dtype(self) -> np.dtype:
        """Return the tensor dtype.

        Note:
            Accessing this property calls ``collect()`` and may materialize data
            from source files (for example Parquet/TIFF), which can be expensive.
        """
        return self.collect().dtype

    @property
    def device(self) -> str:
        """Return the tensor storage device.

        Note:
            For unresolved lazy plans, this returns ``"cpu"`` without calling
            ``collect()``.
        """
        if self._resolved is None:
            return "cpu"
        return self._resolved.device

    @property
    def layout(self) -> str:
        """Return the effective tensor layout.

        Note:
            Accessing this property calls ``collect()`` and may materialize data
            from source files (for example Parquet/TIFF), which can be expensive.
        """
        layout = self._kwargs.get("layout")
        if layout is not None:
            return layout
        return self.collect().layout

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the tensor shape.

        Note:
            Accessing this property calls ``collect()`` and may materialize data
            from source files (for example Parquet/TIFF), which can be expensive.
        """
        return self.collect().shape

    @property
    def strides(self) -> tuple[int, ...]:
        """Return tensor strides in bytes.

        Note:
            Accessing this property calls ``collect()`` and may materialize data
            from source files (for example Parquet/TIFF), which can be expensive.
        """
        return self.collect().strides

    def to_numpy(self, *, contiguous: bool = False) -> np.ndarray:
        """Materialize as a NumPy array.

        Args:
            contiguous: When True, return a contiguous array copy.

        Returns:
            np.ndarray: Materialized array.
        """
        return self.collect().to_numpy(contiguous=contiguous)

    def to_dlpack(
        self,
        *,
        device: str = "cpu",
        contiguous: bool = True,
        mode: str = "arrow",
    ) -> Any:
        """Export the planned view as a DLPack object.

        Args:
            device: Target device (``"cpu"`` or ``"cuda"``).
            contiguous: When True, materialize contiguous data when needed.
            mode: Export mode (``"arrow"`` or ``"numpy"``).

        Returns:
            Any: DLPack-compatible object.
        """
        return self.collect().to_dlpack(device=device, contiguous=contiguous, mode=mode)

    def to_torch(
        self,
        *,
        device: str = "cpu",
        contiguous: bool = True,
        mode: str = "arrow",
    ) -> Any:
        """Convert the planned view to a torch tensor.

        Args:
            device: Target device (``"cpu"`` or ``"cuda"``).
            contiguous: When True, materialize contiguous data when needed.
            mode: Export mode (``"arrow"`` or ``"numpy"``).

        Returns:
            Any: ``torch.Tensor`` when torch is installed.
        """
        return self.collect().to_torch(device=device, contiguous=contiguous, mode=mode)

    def to_jax(
        self,
        *,
        device: str = "cpu",
        contiguous: bool = True,
        mode: str = "arrow",
    ) -> Any:
        """Convert the planned view to a JAX array.

        Args:
            device: Target device (``"cpu"`` or ``"cuda"``).
            contiguous: When True, materialize contiguous data when needed.
            mode: Export mode (``"arrow"`` or ``"numpy"``).

        Returns:
            Any: JAX array when JAX is installed.
        """
        return self.collect().to_jax(device=device, contiguous=contiguous, mode=mode)

    def iter_dlpack(
        self,
        *,
        batch_size: int | None = None,
        tile_size: tuple[int, int] | None = None,
        tiles: tuple[int, int] | None = None,
        shuffle: bool = False,
        seed: int | None = None,
        prefetch: int = 0,
        device: str = "cpu",
        contiguous: bool = True,
        mode: str = "arrow",
    ) -> Iterator[Any]:
        """Iterate DLPack outputs in batches or 2D tiles.

        Args:
            batch_size: Number of time indices per batch.
            tile_size: Optional tile size as ``(tile_h, tile_w)``.
            tiles: Deprecated alias for ``tile_size``.
            shuffle: Whether to shuffle iteration order.
            seed: Optional random seed for deterministic shuffling.
            prefetch: Placeholder prefetch count.
            device: Target device (``"cpu"`` or ``"cuda"``).
            contiguous: When True, materialize contiguous data when needed.
            mode: Export mode (``"arrow"`` or ``"numpy"``).

        Returns:
            Iterator[Any]: Iterator of DLPack-compatible objects.
        """
        return self.collect().iter_dlpack(
            batch_size=batch_size,
            tile_size=tile_size,
            tiles=tiles,
            shuffle=shuffle,
            seed=seed,
            prefetch=prefetch,
            device=device,
            contiguous=contiguous,
            mode=mode,
        )

    def iter_tiles_3d(
        self,
        *,
        tile_size: tuple[int, int, int],
        shuffle: bool = False,
        seed: int | None = None,
        prefetch: int = 0,
        device: str = "cpu",
        contiguous: bool = True,
        mode: str = "numpy",
    ) -> Iterator[Any]:
        """Iterate DLPack outputs in 3D tiles.

        Args:
            tile_size: Tile shape as ``(tile_z, tile_h, tile_w)``.
            shuffle: Whether to shuffle iteration order.
            seed: Optional random seed for deterministic shuffling.
            prefetch: Placeholder prefetch count.
            device: Target device (``"cpu"`` or ``"cuda"``).
            contiguous: When True, materialize contiguous data when needed.
            mode: Export mode (currently ``"numpy"`` only).

        Returns:
            Iterator[Any]: Iterator of DLPack-compatible objects.
        """
        return self.collect().iter_tiles_3d(
            tile_size=tile_size,
            shuffle=shuffle,
            seed=seed,
            prefetch=prefetch,
            device=device,
            contiguous=contiguous,
            mode=mode,
        )


class TensorView:
    """View OME-Arrow pixel data as a tensor-like object.

    Args:
        data: OME-Arrow dict, StructScalar, or 1-row StructArray/ChunkedArray.
        t: Time index selection (int, slice, or sequence). Default: all.
        z: Z index selection (int, slice, or sequence). Default: all.
        c: Channel index selection (int, slice, or sequence). Default: all.
        roi: Spatial crop (x, y, w, h) in pixels. Default: full frame.
        roi3d: Spatial + depth crop (x, y, z, w, h, d). This is a
            convenience alias for ``roi=(x, y, w, h)`` and
            ``z=slice(z, z + d)``.
        tile: Tile index (tile_y, tile_x) based on chunk grid.
        layout: Desired layout string using TZCHW letters where
            T=time, Z=depth, C=channel, H=height (Y), W=width (X).
        dtype: Output dtype override. Defaults to pixels_meta.type when valid.
        chunk_policy: Handling for ``pyarrow.ChunkedArray`` inputs. "auto"
            keeps multi-chunk arrays and unwraps single-chunk arrays.
            "combine" always combines multi-chunk arrays eagerly.
            "keep" always keeps chunked storage.
        channel_policy: Behavior when dropping `C` from layout while
            multiple channels are selected. "error" raises (default).
            "first" keeps the first channel.
    """

    def __init__(
        self,
        data: dict[str, Any] | pa.StructScalar | pa.StructArray | pa.ChunkedArray,
        *,
        t: int | slice | Sequence[int] | None = None,
        z: int | slice | Sequence[int] | None = None,
        c: int | slice | Sequence[int] | None = None,
        roi: tuple[int, int, int, int] | None = None,
        roi3d: tuple[int, int, int, int, int, int] | None = None,
        tile: tuple[int, int] | None = None,
        layout: str | None = None,
        dtype: np.dtype | None = None,
        chunk_policy: Literal["auto", "combine", "keep"] = "auto",
        channel_policy: Literal["error", "first"] = "error",
    ) -> None:
        """Initialize a TensorView over selected OME-Arrow pixels.

        Args:
            data: OME-Arrow record as dict/StructScalar/StructArray/ChunkedArray.
            t: Time index selection (int, slice, or sequence). Default: all.
            z: Z index selection (int, slice, or sequence). Default: all.
            c: Channel index selection (int, slice, or sequence). Default: all.
            roi: Spatial crop (x, y, w, h) in pixels. Default: full frame.
            roi3d: Spatial + depth crop (x, y, z, w, h, d). This is a
                convenience alias for ``roi=(x, y, w, h)`` and
                ``z=slice(z, z + d)``.
            tile: Tile index (tile_y, tile_x) derived from chunk_grid.
            layout: Desired layout string using TZCHW letters where
                T=time, Z=depth, C=channel, H=height (Y), W=width (X).
            dtype: Output dtype override.
            chunk_policy: Handling for ``pyarrow.ChunkedArray`` inputs.
            channel_policy: Behavior when dropping `C` from layout while
                multiple channels are selected. "error" raises (default).
                "first" keeps the first channel.
        """
        self._chunk_policy = _normalize_chunk_policy(chunk_policy)
        self._struct_array: pa.StructArray | None = None
        self._struct_scalar: pa.StructScalar | None = None
        self._chunked_array: pa.ChunkedArray | None = None
        if isinstance(data, pa.ChunkedArray):
            if data.num_chunks == 0:
                data = pa.array([], type=OME_ARROW_STRUCT)
            elif self._chunk_policy == "combine":
                data = data.chunk(0) if data.num_chunks == 1 else data.combine_chunks()
            elif self._chunk_policy == "auto" and data.num_chunks == 1:
                data = data.chunk(0)
        if isinstance(data, pa.StructArray):
            self._struct_array = data
            self._struct_scalar = data[0] if len(data) > 0 else None
            self._data_py: dict[str, Any] | None = None
        elif isinstance(data, pa.ChunkedArray):
            self._chunked_array = data
            self._struct_scalar = _first_struct_scalar_from_chunked(data)
            self._data_py = None
        elif isinstance(data, pa.StructScalar):
            self._struct_scalar = data
            self._data_py = None
        else:
            self._data_py = data
        # Keep normalized backing data so child TensorViews do not repeatedly
        # combine chunked Arrow arrays during iteration.
        self._data = data
        self._layout_override = _normalize_layout(layout) if layout else None
        self._channel_policy = _normalize_channel_policy(channel_policy)

        pm = self._pixels_meta()
        self._size_x = int(pm["size_x"])
        self._size_y = int(pm["size_y"])
        self._size_z = int(pm["size_z"])
        self._size_c = int(pm["size_c"])
        self._size_t = int(pm["size_t"])

        if dtype is None:
            dtype = _dtype_from_meta(pm.get("type"))
        self._dtype = np.dtype(dtype)

        self._selection = self._normalize_selection(
            t=t, z=z, c=c, roi=roi, roi3d=roi3d, tile=tile
        )
        self._array: np.ndarray | None = None
        self._array_layout: str | None = None
        self._chunks_present: bool | None = None

    @property
    def dtype(self) -> np.dtype:
        """Return the tensor dtype."""

        return self._dtype

    @property
    def device(self) -> str:
        """Return the storage device for the view (currently always "cpu")."""

        return "cpu"

    @property
    def layout(self) -> str:
        """Return the effective layout for this view."""

        if self._layout_override:
            return self._layout_override
        if len(self._selection.t) == 1 and len(self._selection.z) == 1:
            return "CHW"
        return _TZCHW

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the tensor shape for the current layout."""
        shape, _ = self._shape_and_strides_for_layout(self.layout)
        return shape

    @property
    def strides(self) -> tuple[int, ...]:
        """Return the tensor strides in bytes for the current layout."""
        _, strides = self._shape_and_strides_for_layout(self.layout)
        return strides

    def with_layout(self, layout: str) -> "TensorView":
        """Return a new TensorView with a layout override.

        Args:
            layout: Desired layout string using TZCHW letters where
                T=time, Z=depth, C=channel, H=height (Y), W=width (X).

        Returns:
            TensorView: New view with the requested layout.
        """

        return TensorView(
            self._data,
            t=self._selection.t,
            z=self._selection.z,
            c=self._selection.c,
            roi=self._selection.roi,
            layout=layout,
            dtype=self._dtype,
            chunk_policy=self._chunk_policy,
            channel_policy=self._channel_policy,
        )

    def to_numpy(self, *, contiguous: bool = False) -> np.ndarray:
        """Materialize the view as a NumPy array.

        Args:
            contiguous: When True, return a contiguous array copy.

        Returns:
            np.ndarray: Array in the requested layout.
        """

        layout = self.layout
        arr = self._materialize(layout)
        if contiguous:
            return np.ascontiguousarray(arr)
        return arr

    def to_dlpack(
        self,
        *,
        device: str = "cpu",
        contiguous: bool = True,
        mode: str = "arrow",
    ) -> Any:
        """Export the view as a DLPack capsule.

        Args:
            device: Target device ("cpu" or "cuda").
            contiguous: When True, materialize a contiguous buffer if needed.
            mode: Export mode. "arrow" returns a capsule for the Arrow values
                buffer (1D). "numpy" materializes a tensor-shaped NumPy view.
                Zero-copy Arrow mode requires Arrow-backed inputs (typically
                Parquet/Vortex ingestion with canonical schema); StructScalar
                and dict inputs are normalized through Python objects.

        Returns:
            DLPack object compatible with torch/jax import utilities.
            The returned object is single-use per DLPack ownership semantics:
            after a consumer imports it, the capsule must not be reused.

        Raises:
            ValueError: If an unsupported device is requested.
            RuntimeError: If required optional dependencies are missing.
        """

        device = _normalize_device(device)
        mode = _normalize_mode(mode)
        if device == "cpu":
            if mode == "arrow":
                values = self._arrow_values()
                capsule = values.__dlpack__()
                return _DLPackWrapper(capsule, _dlpack_device(values))
            arr = self.to_numpy(contiguous=contiguous)
            capsule = arr.__dlpack__()
            return _DLPackWrapper(capsule, _dlpack_device(arr))

        torch = _require_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch reports no CUDA.")

        if mode == "arrow":
            raise ValueError("mode='arrow' does not support device='cuda'.")
        arr = self.to_numpy(contiguous=contiguous)
        torch_tensor = torch.from_numpy(arr)
        torch_tensor = torch_tensor.to(device="cuda")
        capsule = torch.utils.dlpack.to_dlpack(torch_tensor)
        return _DLPackWrapper(capsule, _dlpack_device(torch_tensor))

    def to_torch(
        self,
        *,
        device: str = "cpu",
        contiguous: bool = True,
        mode: str = "arrow",
    ) -> Any:
        """Convert the view into a torch.Tensor using DLPack.

        Args:
            device: Target device ("cpu" or "cuda").
            contiguous: When True, materialize a contiguous buffer if needed.
            mode: Export mode. "arrow" returns a 1D values buffer.

        Returns:
            torch.Tensor: Tensor backed by the DLPack capsule.
        """

        torch = _require_torch()
        dlpack = self.to_dlpack(device=device, contiguous=contiguous, mode=mode)
        return torch.utils.dlpack.from_dlpack(dlpack)

    def to_jax(
        self,
        *,
        device: str = "cpu",
        contiguous: bool = True,
        mode: str = "arrow",
    ) -> Any:
        """Convert the view into a JAX array using DLPack.

        Args:
            device: Target device ("cpu" or "cuda").
            contiguous: When True, materialize a contiguous buffer if needed.
            mode: Export mode. "arrow" returns a 1D values buffer.

        Returns:
            jax.Array: Array backed by the DLPack capsule.
        """

        try:
            import jax.numpy as jnp
        except ImportError as exc:
            raise RuntimeError("JAX is not installed.") from exc

        dlpack = self.to_dlpack(device=device, contiguous=contiguous, mode=mode)
        return jnp.from_dlpack(dlpack)

    def iter_dlpack(
        self,
        *,
        batch_size: int | None = None,
        tile_size: tuple[int, int] | None = None,
        tiles: tuple[int, int] | None = None,
        shuffle: bool = False,
        seed: int | None = None,
        prefetch: int = 0,
        device: str = "cpu",
        contiguous: bool = True,
        mode: str = "arrow",
    ) -> Iterator[Any]:
        """Iterate over DLPack capsules in batches or tiles.

        Args:
            batch_size: Number of T indices per batch. Defaults to full range.
            tile_size: Tile size (tile_h, tile_w) in pixels for spatial tiling.
            tiles: Deprecated alias for ``tile_size``.
            shuffle: Whether to shuffle the iteration order.
            seed: Seed for deterministic shuffling.
            prefetch: Placeholder for future asynchronous prefetch support.
                Currently validated but does not change synchronous iteration.
            device: Target device ("cpu" or "cuda").
            contiguous: When True, materialize contiguous buffers if needed.
            mode: Export mode. "arrow" returns 1D values buffers.

        Yields:
            DLPack object per batch or tile.
        """

        if prefetch < 0:
            raise ValueError("prefetch must be >= 0")

        if tiles is not None:
            if tile_size is not None:
                raise ValueError("Provide only one of tile_size or tiles.")
            warnings.warn(
                "iter_dlpack(tiles=...) is deprecated; use tile_size=... instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            tile_size = tiles

        if tile_size is not None:
            yield from self._iter_tiles(
                tiles=tile_size,
                shuffle=shuffle,
                seed=seed,
                prefetch=prefetch,
                device=device,
                contiguous=contiguous,
                mode=mode,
            )
            return

        yield from self._iter_batches(
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            prefetch=prefetch,
            device=device,
            contiguous=contiguous,
            mode=mode,
        )

    def iter_tiles_3d(
        self,
        *,
        tile_size: tuple[int, int, int],
        shuffle: bool = False,
        seed: int | None = None,
        prefetch: int = 0,
        device: str = "cpu",
        contiguous: bool = True,
        mode: str = "numpy",
    ) -> Iterator[Any]:
        """Iterate over 3D tiles (z, y, x) as DLPack capsules.

        Args:
            tile_size: Tile size as ``(tile_z, tile_h, tile_w)``.
            shuffle: Whether to shuffle the tile order.
            seed: Seed for deterministic shuffling.
            prefetch: Placeholder for future asynchronous prefetch support.
            device: Target device ("cpu" or "cuda").
            contiguous: When True, materialize contiguous buffers if needed.
            mode: Export mode. Must be ``"numpy"`` for tiled 3D iteration.

        Yields:
            DLPack object per 3D tile.
        """

        if mode != "numpy":
            raise ValueError("iter_tiles_3d currently supports only mode='numpy'.")
        if prefetch < 0:
            raise ValueError("prefetch must be >= 0")

        tile_z, tile_h, tile_w = tile_size
        if tile_z <= 0 or tile_h <= 0 or tile_w <= 0:
            raise ValueError("tile_size entries must be positive")

        t_indices = list(self._selection.t)
        z_indices = list(self._selection.z)
        if not t_indices or not z_indices:
            return

        x0, y0, w0, h0 = self._selection.roi
        z_batches = [
            tuple(z_indices[i : i + tile_z]) for i in range(0, len(z_indices), tile_z)
        ]
        tasks = []
        for t in t_indices:
            for z_batch in z_batches:
                for y in range(y0, y0 + h0, tile_h):
                    for x in range(x0, x0 + w0, tile_w):
                        tw = min(tile_w, x0 + w0 - x)
                        th = min(tile_h, y0 + h0 - y)
                        tasks.append((t, z_batch, x, y, tw, th))

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(tasks)

        for batch in _batched(tasks, 1, prefetch=prefetch):
            t, z_batch, x, y, w, h = batch[0]
            view = TensorView(
                self._data,
                t=[t],
                z=list(z_batch),
                c=self._selection.c,
                roi=(x, y, w, h),
                layout=self.layout,
                dtype=self._dtype,
                chunk_policy=self._chunk_policy,
                channel_policy=self._channel_policy,
            )
            yield view.to_dlpack(device=device, contiguous=contiguous, mode=mode)

    def _iter_batches(
        self,
        *,
        batch_size: int | None,
        shuffle: bool,
        seed: int | None,
        prefetch: int,
        device: str,
        contiguous: bool,
        mode: str,
    ) -> Iterator[Any]:
        t_indices = list(self._selection.t)
        if not t_indices:
            return
        if batch_size is None:
            batch_size = len(t_indices)
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(t_indices)

        for batch in _batched(t_indices, batch_size, prefetch=prefetch):
            view = TensorView(
                self._data,
                t=batch,
                z=self._selection.z,
                c=self._selection.c,
                roi=self._selection.roi,
                layout=self.layout,
                dtype=self._dtype,
                chunk_policy=self._chunk_policy,
                channel_policy=self._channel_policy,
            )
            yield view.to_dlpack(device=device, contiguous=contiguous, mode=mode)

    def _iter_tiles(
        self,
        *,
        tiles: tuple[int, int],
        shuffle: bool,
        seed: int | None,
        prefetch: int,
        device: str,
        contiguous: bool,
        mode: str,
    ) -> Iterator[Any]:
        tile_h, tile_w = tiles
        if tile_h <= 0 or tile_w <= 0:
            raise ValueError("tiles must be positive")

        x0, y0, w0, h0 = self._selection.roi
        tile_positions = []
        for y in range(y0, y0 + h0, tile_h):
            for x in range(x0, x0 + w0, tile_w):
                tw = min(tile_w, x0 + w0 - x)
                th = min(tile_h, y0 + h0 - y)
                tile_positions.append((x, y, tw, th))

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(tile_positions)

        for batch in _batched(tile_positions, 1, prefetch=prefetch):
            x, y, w, h = batch[0]
            view = TensorView(
                self._data,
                t=self._selection.t,
                z=self._selection.z,
                c=self._selection.c,
                roi=(x, y, w, h),
                layout=self.layout,
                dtype=self._dtype,
                chunk_policy=self._chunk_policy,
                channel_policy=self._channel_policy,
            )
            yield view.to_dlpack(device=device, contiguous=contiguous, mode=mode)

    def _materialize(self, layout: str) -> np.ndarray:
        if self._array is not None and self._array_layout == layout:
            return self._array

        base = self._build_tzchw()
        if "C" not in layout and base.shape[2] != 1 and self._channel_policy == "first":
            # Explicit opt-in: keep first channel when layout drops C.
            base = base[:, :, :1, :, :]
        arr = _apply_layout(base, layout)
        self._array = arr
        self._array_layout = layout
        return arr

    def _build_tzchw(self) -> np.ndarray:
        t_idx, z_idx, c_idx = self._selection.t, self._selection.z, self._selection.c
        x0, y0, w, h = self._selection.roi

        out = np.zeros((len(t_idx), len(z_idx), len(c_idx), h, w), dtype=self._dtype)
        plane_map = self._plane_map()
        # Baseline implementation: reads one plane per (t, z, c) combination.
        # For large 5D selections this may be a hotspot, especially with chunks.
        for ti, t in enumerate(t_idx):
            for zi, z in enumerate(z_idx):
                for ci, c in enumerate(c_idx):
                    arr2d = self._read_plane(plane_map, t=t, z=z, c=c)
                    out[ti, zi, ci] = arr2d[y0 : y0 + h, x0 : x0 + w]
        return out

    def _plane_map(self) -> dict[tuple[int, int, int], list[Any]]:
        if self._has_chunks():
            return {}
        plane_map = {}
        data_py = self._data_py_dict()
        for plane in data_py.get("planes", []):
            key = (int(plane["t"]), int(plane["z"]), int(plane["c"]))
            plane_map[key] = plane["pixels"]
        return plane_map

    def _read_plane(
        self,
        plane_map: dict[tuple[int, int, int], list[Any]],
        *,
        t: int,
        z: int,
        c: int,
    ) -> np.ndarray:
        if self._has_chunks():
            data_py = self._data_py_dict()
            return plane_from_chunks(data_py, t=t, z=z, c=c, dtype=self._dtype)

        key = (t, z, c)
        if key not in plane_map:
            raise ValueError(f"Plane (t={t}, z={z}, c={c}) not found.")
        pix = plane_map[key]
        try:
            arr = np.asarray(pix, dtype=self._dtype)
        except Exception as exc:
            raise ValueError(f"Plane (t={t}, z={z}, c={c}) pixels invalid.") from exc
        return arr.reshape(self._size_y, self._size_x)

    def _normalize_selection(
        self,
        *,
        t: int | slice | Sequence[int] | None,
        z: int | slice | Sequence[int] | None,
        c: int | slice | Sequence[int] | None,
        roi: tuple[int, int, int, int] | None,
        roi3d: tuple[int, int, int, int, int, int] | None,
        tile: tuple[int, int] | None,
    ) -> _Selection:
        if roi3d is not None:
            if roi is not None or tile is not None:
                raise ValueError("Provide only one of roi3d, roi, or tile.")
            if z is not None:
                raise ValueError("Provide either z or roi3d, not both.")
            if len(roi3d) != 6:
                raise ValueError("roi3d must be (x, y, z, w, h, d)")
            x3, y3, z3, w3, h3, d3 = roi3d
            if d3 <= 0:
                raise ValueError("roi3d depth must be positive")
            if z3 < 0:
                raise ValueError("roi3d z origin must be non-negative")
            if z3 + d3 > self._size_z:
                raise ValueError("roi3d depth is out of bounds")
            roi = (x3, y3, w3, h3)
            z = slice(z3, z3 + d3)

        t_idx = _normalize_index(t, self._size_t, "t")
        z_idx = _normalize_index(z, self._size_z, "z")
        c_idx = _normalize_index(c, self._size_c, "c")

        if roi is not None and tile is not None:
            raise ValueError("Provide either roi or tile, not both.")

        if tile is not None:
            roi = _tile_to_roi(self._chunk_grid(), tile, self._size_x, self._size_y)

        if roi is None:
            roi = (0, 0, self._size_x, self._size_y)
        x, y, w, h = roi
        if w <= 0 or h <= 0:
            raise ValueError("roi width and height must be positive")
        if x < 0 or y < 0:
            raise ValueError("roi origin must be non-negative")
        if x + w > self._size_x or y + h > self._size_y:
            raise ValueError("roi is out of bounds")

        return _Selection(t=t_idx, z=z_idx, c=c_idx, roi=roi)

    def _data_py_dict(self) -> dict[str, Any]:
        """Return the backing record as a Python dict.

        Returns:
            dict[str, Any]: Materialized OME-Arrow record.

        Raises:
            ValueError: If no scalar-backed record is available.
        """
        if self._data_py is None:
            if self._struct_scalar is None:
                raise ValueError("No OME-Arrow scalar available.")
            self._data_py = self._struct_scalar.as_py()
        return self._data_py

    def _pixels_meta(self) -> dict[str, Any]:
        """Return ``pixels_meta`` from the active backing representation.

        Returns:
            dict[str, Any]: Pixel metadata for shape and dtype decisions.

        Raises:
            ValueError: If metadata is missing or no scalar/array is available.
        """
        if self._data_py is not None:
            return self._data_py["pixels_meta"]
        if self._struct_array is not None:
            pm_arr = self._struct_array.field("pixels_meta")
            if len(pm_arr) == 0:
                raise ValueError(
                    "Cannot create TensorView from an empty OME-Arrow array."
                )
            return pm_arr[0].as_py()
        if self._struct_scalar is not None:
            pm_scalar = self._struct_scalar["pixels_meta"]
            if not pm_scalar.is_valid:
                raise ValueError("pixels_meta is missing from OME-Arrow scalar.")
            return pm_scalar.as_py()
        raise ValueError("No OME-Arrow scalar available.")

    def _has_chunks(self) -> bool:
        """Return whether chunked pixel payloads are present.

        Returns:
            bool: True when ``chunks`` contains at least one chunk.
        """
        if self._chunks_present is None:
            self._chunks_present = self._compute_has_chunks()
        return self._chunks_present

    def _compute_has_chunks(self) -> bool:
        """Compute chunk presence from dict/array/scalar backing data.

        Returns:
            bool: True when chunk payloads are present and non-empty.
        """
        result = False
        if self._data_py is not None:
            result = bool(self._data_py.get("chunks"))
        elif self._struct_array is not None:
            chunks_arr = self._struct_array.field("chunks")
            if len(chunks_arr) > 0 and not chunks_arr.is_null().to_pylist()[0]:
                lengths = chunks_arr.value_lengths()
                first_len = lengths[0]
                result = bool(first_len.is_valid and int(first_len.as_py()) > 0)
        elif self._struct_scalar is not None:
            chunks_scalar = self._struct_scalar["chunks"]
            if chunks_scalar.is_valid:
                result = len(chunks_scalar.values) > 0
        return result

    def _chunk_grid(self) -> dict[str, Any]:
        """Return chunk grid metadata when available.

        Returns:
            dict[str, Any]: Chunk grid metadata, or an empty dict.
        """
        if self._data_py is not None:
            return self._data_py.get("chunk_grid") or {}
        if self._struct_array is not None:
            grid_arr = self._struct_array.field("chunk_grid")
            if len(grid_arr) == 0 or grid_arr.is_null().to_pylist()[0]:
                return {}
            return grid_arr[0].as_py()
        if self._struct_scalar is not None:
            grid_scalar = self._struct_scalar["chunk_grid"]
            if not grid_scalar.is_valid:
                return {}
            return grid_scalar.as_py()
        return {}

    def _shape_and_strides_for_layout(
        self, layout: str
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        current = list(_TZCHW)
        shape = [
            len(self._selection.t),
            len(self._selection.z),
            len(self._selection.c),
            self._selection.roi[3],
            self._selection.roi[2],
        ]

        itemsize = int(self._dtype.itemsize)
        strides = [0] * len(shape)
        stride = itemsize
        for axis in range(len(shape) - 1, -1, -1):
            strides[axis] = stride
            stride *= shape[axis]

        # Mirror _apply_layout behavior without touching pixel buffers.
        for dim in list(current):
            if dim not in layout:
                axis = current.index(dim)
                if shape[axis] != 1:
                    if dim == "C" and self._channel_policy == "first":
                        shape[axis] = 1
                    else:
                        raise ValueError(
                            f"layout '{layout}' drops non-singleton dimension '{dim}'."
                        )
                shape.pop(axis)
                strides.pop(axis)
                current.pop(axis)

        if list(layout) == current:
            return tuple(shape), tuple(strides)

        axes = [current.index(dim) for dim in layout]
        return tuple(shape[axis] for axis in axes), tuple(
            strides[axis] for axis in axes
        )

    def _arrow_values(self) -> pa.Array:
        t_idx, z_idx, c_idx = self._selection.t, self._selection.z, self._selection.c
        if len(t_idx) != 1 or len(z_idx) != 1 or len(c_idx) != 1:
            raise ValueError("mode='arrow' requires a single (t, z, c) selection.")

        struct_arr = _ensure_struct_array(self._data)
        if struct_arr is None:
            raise ValueError("mode='arrow' requires Arrow-backed data.")

        if self._has_chunks():
            return _select_chunk_values(
                struct_arr,
                t=t_idx[0],
                z=z_idx[0],
                c=c_idx[0],
                roi=self._selection.roi,
            )

        x0, y0, w, h = self._selection.roi
        if not (x0 == 0 and y0 == 0 and w == self._size_x and h == self._size_y):
            raise ValueError(
                "mode='arrow' requires a full-frame ROI; use mode='numpy' for crops."
            )

        return _select_plane_values(struct_arr, t=t_idx[0], z=z_idx[0], c=c_idx[0])


def _normalize_layout(layout: str | None) -> str | None:
    if layout is None:
        return None
    layout = layout.strip().upper()
    if not layout:
        raise ValueError("layout must be non-empty")
    if any(dim not in _ALLOWED_DIMS for dim in layout):
        raise ValueError("layout must use only TZCHW letters")
    if len(set(layout)) != len(layout):
        raise ValueError("layout cannot repeat dimensions")
    return layout


def _normalize_device(device: str) -> str:
    device = str(device).strip().lower()
    if device in {"cpu", "cuda"}:
        return device
    raise ValueError(f"Unsupported device '{device}'.")


def _normalize_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode not in _ALLOWED_MODES:
        raise ValueError(f"Unsupported mode '{mode}'.")
    return mode


def _normalize_channel_policy(channel_policy: str) -> str:
    policy = str(channel_policy).strip().lower()
    if policy not in {"error", "first"}:
        raise ValueError(
            "Unsupported channel_policy "
            f"'{channel_policy}'. Expected one of: error, first."
        )
    return policy


def _normalize_chunk_policy(chunk_policy: str) -> str:
    """Normalize and validate chunk handling policy.

    Args:
        chunk_policy: Requested chunk handling mode.

    Returns:
        str: Normalized policy value in ``{"auto", "combine", "keep"}``.

    Raises:
        ValueError: If the policy value is unsupported.
    """
    policy = str(chunk_policy).strip().lower()
    if policy not in {"auto", "combine", "keep"}:
        raise ValueError(
            "Unsupported chunk_policy "
            f"'{chunk_policy}'. Expected one of: auto, combine, keep."
        )
    return policy


def _normalize_index(
    index: int | slice | Sequence[int] | None, size: int, name: str
) -> List[int]:
    if size <= 0:
        raise ValueError(f"{name} size must be positive")
    if index is None:
        return list(range(size))
    if isinstance(index, int):
        idx = index + size if index < 0 else index
        if not 0 <= idx < size:
            raise ValueError(f"{name} index {index} out of range")
        return [idx]
    if isinstance(index, slice):
        return list(range(*index.indices(size)))
    if isinstance(index, Sequence) and not isinstance(index, (str, bytes)):
        out = []
        for item in index:
            if not isinstance(item, int):
                raise ValueError(f"{name} indices must be integers")
            idx = item + size if item < 0 else item
            if not 0 <= idx < size:
                raise ValueError(f"{name} index {item} out of range")
            out.append(idx)
        return out
    raise ValueError(f"Unsupported {name} index type")


def _apply_layout(arr: np.ndarray, layout: str) -> np.ndarray:
    layout = _normalize_layout(layout) or _TZCHW
    current = list(_TZCHW)
    for dim in list(current):
        if dim not in layout:
            axis = current.index(dim)
            if arr.shape[axis] != 1:
                raise ValueError(
                    f"layout '{layout}' drops non-singleton dimension '{dim}'."
                )
            arr = np.squeeze(arr, axis=axis)
            current.pop(axis)

    if list(layout) == current:
        return arr
    axes = [current.index(dim) for dim in layout]
    return np.transpose(arr, axes)


def _dtype_from_meta(dtype: Any) -> np.dtype:
    if dtype is None:
        return np.dtype(np.uint16)
    try:
        return np.dtype(dtype)
    except Exception as exc:
        warnings.warn(
            (
                "_dtype_from_meta: failed to convert dtype "
                f"{dtype!r} ({exc}); falling back to uint16."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return np.dtype(np.uint16)


def _tile_to_roi(
    chunk_grid: dict[str, Any], tile: tuple[int, int], size_x: int, size_y: int
) -> tuple[int, int, int, int]:
    if len(tile) != 2:
        raise ValueError("tile must be (tile_y, tile_x)")
    tile_y, tile_x = tile
    if not chunk_grid:
        raise ValueError("tile selection requires chunk_grid metadata")
    try:
        chunk_y = int(chunk_grid["chunk_y"])
        chunk_x = int(chunk_grid["chunk_x"])
    except Exception as exc:
        raise ValueError("chunk_grid is missing chunk_x/chunk_y") from exc
    if chunk_x <= 0 or chunk_y <= 0:
        raise ValueError("chunk_grid chunk sizes must be positive")
    if tile_x < 0 or tile_y < 0:
        raise ValueError("tile indices must be non-negative")

    x = tile_x * chunk_x
    y = tile_y * chunk_y
    if x >= size_x or y >= size_y:
        raise ValueError("tile indices out of bounds")
    w = min(chunk_x, size_x - x)
    h = min(chunk_y, size_y - y)
    return x, y, w, h


def _batched(items: List[Any], size: int, *, prefetch: int) -> Iterator[List[Any]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    # prefetch is intentionally a no-op placeholder until async prefetching
    # is implemented.
    _ = prefetch
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch is not installed.") from exc
    return torch


def _dlpack_device(obj: Any) -> tuple[int, int]:
    if hasattr(obj, "__dlpack_device__"):
        return obj.__dlpack_device__()
    return (1, 0)


class _DLPackWrapper:
    """Thin DLPack producer wrapper around a single capsule.

    The wrapped capsule follows standard DLPack ownership transfer semantics:
    it is intended for one consumer call to ``__dlpack__``.
    """

    def __init__(self, capsule: Any, device: tuple[int, int]) -> None:
        self._capsule = capsule
        self._device = device

    def __dlpack__(self, stream: Any | None = None) -> Any:
        _ = stream
        if self._capsule is None:
            raise RuntimeError(
                "DLPack capsule has already been consumed and cannot be reused."
            )
        capsule = self._capsule
        self._capsule = None
        return capsule

    def __dlpack_device__(self) -> tuple[int, int]:
        return self._device


def _ensure_struct_array(
    data: dict[str, Any] | pa.StructScalar | pa.StructArray | pa.ChunkedArray,
) -> pa.StructArray | None:
    """Coerce supported inputs into a StructArray for Arrow-mode export.

    Args:
        data: Candidate backing data for a TensorView.

    Returns:
        pa.StructArray | None: A StructArray when conversion is possible,
            otherwise ``None``.
    """
    if isinstance(data, pa.ChunkedArray):
        if data.num_chunks == 0:
            return pa.array([], type=OME_ARROW_STRUCT)
        if data.num_chunks == 1:
            data = data.chunk(0)
        else:
            scalar = _first_struct_scalar_from_chunked(data)
            if scalar is None:
                return pa.array([], type=OME_ARROW_STRUCT)
            warnings.warn(
                "mode='arrow' received a multi-chunk ChunkedArray; using first "
                "record without eager combine_chunks().",
                UserWarning,
                stacklevel=3,
            )
            data = pa.array([scalar], type=OME_ARROW_STRUCT)
    if isinstance(data, pa.StructArray):
        return data
    if isinstance(data, pa.StructScalar):
        warnings.warn(
            "mode='arrow' received a StructScalar; converting via as_py(), "
            "so zero-copy is not guaranteed for this export.",
            UserWarning,
            stacklevel=3,
        )
        return pa.array([data.as_py()], type=OME_ARROW_STRUCT)
    if isinstance(data, dict):
        warnings.warn(
            "mode='arrow' received a dict; converting to Arrow buffers, "
            "so zero-copy is not guaranteed for this export.",
            UserWarning,
            stacklevel=3,
        )
        scalar = pa.scalar(data, type=OME_ARROW_STRUCT)
        return pa.array([scalar])
    return None


def _first_struct_scalar_from_chunked(data: pa.ChunkedArray) -> pa.StructScalar | None:
    """Return the first non-empty struct scalar from a chunked array.

    Args:
        data: Chunked struct column.

    Returns:
        pa.StructScalar | None: First scalar found, or ``None`` for empty input.
    """
    for chunk in data.chunks:
        if len(chunk) > 0:
            return chunk[0]
    return None


def _select_plane_values(
    struct_arr: pa.StructArray, *, t: int, z: int, c: int
) -> pa.Array:
    planes_arr = struct_arr.field("planes")
    if len(planes_arr) == 0 or planes_arr.is_null().to_pylist()[0]:
        raise ValueError("mode='arrow' requires planes data.")

    plane_list = planes_arr[0].values
    mask = pc.and_(
        pc.equal(plane_list.field("t"), t),
        pc.and_(pc.equal(plane_list.field("z"), z), pc.equal(plane_list.field("c"), c)),
    )
    selected = pc.filter(plane_list, mask)
    if len(selected) != 1:
        raise ValueError("Plane not found for the requested (t, z, c).")
    pixels_list = selected.field("pixels")[0]
    return pixels_list.values


def _select_chunk_values(
    struct_arr: pa.StructArray,
    *,
    t: int,
    z: int,
    c: int,
    roi: tuple[int, int, int, int],
) -> pa.Array:
    chunks_arr = struct_arr.field("chunks")
    if len(chunks_arr) == 0 or chunks_arr.is_null().to_pylist()[0]:
        raise ValueError("mode='arrow' requires chunked pixels data.")

    chunks = chunks_arr[0].values
    x, y, w, h = roi
    mask = pc.and_(
        pc.equal(chunks.field("t"), t),
        pc.and_(
            pc.equal(chunks.field("z"), z),
            pc.and_(
                pc.equal(chunks.field("c"), c),
                pc.and_(pc.equal(chunks.field("x"), x), pc.equal(chunks.field("y"), y)),
            ),
        ),
    )
    selected = pc.filter(chunks, mask)
    if len(selected) != 1:
        raise ValueError("Chunk not found for the requested (t, z, c, roi).")

    if int(selected.field("shape_z")[0].as_py()) != 1:
        raise ValueError("mode='arrow' requires shape_z == 1 for chunked selections.")
    if (
        int(selected.field("shape_x")[0].as_py()) != w
        or int(selected.field("shape_y")[0].as_py()) != h
    ):
        raise ValueError("mode='arrow' requires ROI to match chunk size.")

    pixels_list = selected.field("pixels")[0]
    return pixels_list.values
