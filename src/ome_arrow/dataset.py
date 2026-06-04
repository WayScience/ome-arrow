"""Typed chunk-buffer dataset support for OME-Arrow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ome_arrow.export import to_numpy
from ome_arrow.meta import OME_ARROW_TAG_TYPE, OME_ARROW_TAG_VERSION

Layout = Literal["auto", "image", "tile", "volume", "z-plane", "block", "hybrid"]
ArrayReturn = Literal["numpy", "torch", "jax"]
AccessPattern = Literal[
    "balanced",
    "full_image",
    "random_image",
    "crop_2d",
    "fast_crop_2d",
    "full_volume",
    "plane_3d",
    "subvolume_3d",
    "balanced_5d",
    "time_series",
]

IMAGE_TABLE_SCHEMA = pa.schema(
    [
        pa.field("image_id", pa.string()),
        pa.field("name", pa.string()),
        pa.field("image_type", pa.string()),
        pa.field("dtype", pa.string()),
        pa.field("size_t", pa.int32()),
        pa.field("size_c", pa.int32()),
        pa.field("size_z", pa.int32()),
        pa.field("size_y", pa.int32()),
        pa.field("size_x", pa.int32()),
        pa.field("layout", pa.string()),
        pa.field("chunk_t", pa.int32()),
        pa.field("chunk_c", pa.int32()),
        pa.field("chunk_z", pa.int32()),
        pa.field("chunk_y", pa.int32()),
        pa.field("chunk_x", pa.int32()),
        pa.field("compression", pa.string()),
        pa.field("channels_json", pa.string()),
    ]
)

CHUNK_TABLE_SCHEMA = pa.schema(
    [
        pa.field("image_id", pa.string()),
        pa.field("chunk_id", pa.int64()),
        pa.field("t", pa.int32()),
        pa.field("c", pa.int32()),
        pa.field("z", pa.int32()),
        pa.field("y", pa.int32()),
        pa.field("x", pa.int32()),
        pa.field("shape_t", pa.int32()),
        pa.field("shape_c", pa.int32()),
        pa.field("shape_z", pa.int32()),
        pa.field("shape_y", pa.int32()),
        pa.field("shape_x", pa.int32()),
        pa.field("dtype", pa.string()),
        pa.field("pixel_bytes", pa.large_binary()),
    ]
)

INDEX_TABLE_SCHEMA = pa.schema(
    [
        pa.field("image_id", pa.string()),
        pa.field("chunk_id", pa.int64()),
        pa.field("row_group_id", pa.int32()),
        pa.field("row_index_in_group", pa.int32()),
        pa.field("fragment_path", pa.string()),
        pa.field("t", pa.int32()),
        pa.field("c", pa.int32()),
        pa.field("z", pa.int32()),
        pa.field("y", pa.int32()),
        pa.field("x", pa.int32()),
        pa.field("shape_t", pa.int32()),
        pa.field("shape_c", pa.int32()),
        pa.field("shape_z", pa.int32()),
        pa.field("shape_y", pa.int32()),
        pa.field("shape_x", pa.int32()),
    ]
)


@dataclass(frozen=True)
class ChunkChoice:
    """Resolved chunking strategy for a typed pixel dataset."""

    layout: str
    chunk_shape: tuple[int, int, int, int, int]
    compression: str | None
    rationale: str


def choose_chunking(
    shape: Sequence[int],
    dtype: np.dtype | str,
    *,
    layout: Layout = "auto",
    access_pattern: AccessPattern = "balanced",
    target_chunk_mb: float = 4.0,
    compression: str | None = "zstd",
    chunk_shape: Sequence[int] | None = None,
) -> ChunkChoice:
    """Choose a chunk layout for a TCZYX image.

    Args:
        shape: Image shape as ``(T, C, Z, Y, X)``.
        dtype: NumPy dtype or dtype string.
        layout: Explicit layout or ``"auto"``.
        access_pattern: Preset used when layout/chunk shape are not explicit.
        target_chunk_mb: Soft maximum chunk payload size in megabytes.
        compression: Parquet compression setting.
        chunk_shape: Optional explicit chunk shape as ``(T, C, Z, Y, X)``.

    Returns:
        ChunkChoice: Resolved layout, chunk shape, compression, and rationale.
    """
    if len(shape) != 5:
        raise ValueError("shape must be a five-value TCZYX sequence")
    st, sc, sz, sy, sx = (int(v) for v in shape)
    if min(st, sc, sz, sy, sx) <= 0:
        raise ValueError("shape values must be positive")
    dtype_obj = np.dtype(dtype)
    try:
        target_chunk_mb = float(target_chunk_mb)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "target_chunk_mb must be a positive number; cannot compare "
            "target_chunk_mb to computed size_mb."
        ) from exc
    if target_chunk_mb <= 0:
        raise ValueError(
            "target_chunk_mb must be > 0 before computing chunk scale from size_mb."
        )

    if layout == "auto":
        layout_by_pattern = {
            "balanced": "block",
            "full_image": "image",
            "random_image": "image",
            "crop_2d": "tile",
            "fast_crop_2d": "tile",
            "full_volume": "volume",
            "plane_3d": "z-plane",
            "subvolume_3d": "block",
            "balanced_5d": "block",
            "time_series": "block",
        }
        resolved_layout = layout_by_pattern[access_pattern]
    else:
        resolved_layout = str(layout)

    if chunk_shape is not None:
        if len(chunk_shape) != 5:
            raise ValueError("chunk_shape must be a five-value TCZYX sequence")
        ct, cc, cz, cy, cx = (int(v) for v in chunk_shape)
    elif resolved_layout in {"image", "hybrid"}:
        ct, cc, cz, cy, cx = st, sc, sz, sy, sx
    elif resolved_layout == "volume":
        ct, cc, cz, cy, cx = 1, 1, sz, sy, sx
    elif resolved_layout == "z-plane":
        ct, cc, cz, cy, cx = 1, 1, 1, sy, sx
    elif access_pattern == "fast_crop_2d":
        ct, cc, cz, cy, cx = 1, 1, 1, min(sy, 256), min(sx, 256)
    elif resolved_layout == "tile":
        ct, cc, cz, cy, cx = 1, 1, 1, min(sy, 512), min(sx, 512)
    elif access_pattern == "time_series":
        ct, cc, cz, cy, cx = 1, 1, min(sz, 8), min(sy, 256), min(sx, 256)
    elif access_pattern == "balanced_5d":
        ct, cc, cz, cy, cx = 1, 1, min(sz, 16), min(sy, 256), min(sx, 256)
    else:
        ct, cc, cz, cy, cx = 1, 1, min(sz, 16), min(sy, 128), min(sx, 128)

    bounds = (st, sc, sz, sy, sx)
    chunk = tuple(
        max(1, min(int(v), limit)) for v, limit in zip((ct, cc, cz, cy, cx), bounds)
    )
    size_mb = float(np.prod(chunk) * dtype_obj.itemsize) / (1024 * 1024)
    if (
        chunk_shape is None
        and size_mb > target_chunk_mb
        and resolved_layout
        not in {
            "image",
            "volume",
            "z-plane",
        }
    ):
        scale = (target_chunk_mb / size_mb) ** 0.5
        cy = max(1, min(sy, int(chunk[3] * scale)))
        cx = max(1, min(sx, int(chunk[4] * scale)))
        chunk = (chunk[0], chunk[1], chunk[2], cy, cx)
        size_mb = float(np.prod(chunk) * dtype_obj.itemsize) / (1024 * 1024)

    rationale = (
        f'Selected layout="{resolved_layout}", chunk_shape={chunk}, '
        f"compression={compression!r} because dtype={dtype_obj.name}, "
        f"estimated chunk size is {size_mb:.3g} MB, and "
        f'access_pattern="{access_pattern}".'
    )
    return ChunkChoice(
        layout=resolved_layout,
        chunk_shape=chunk,
        compression=compression,
        rationale=rationale,
    )


def _as_tczyx_array(image: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a TCZYX array and metadata dictionary for a supported image input."""
    try:
        from ome_arrow.core import OMEArrow
    except Exception:
        OMEArrow = ()  # type: ignore

    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = arr.reshape(1, 1, 1, *arr.shape)
        elif arr.ndim == 3:
            arr = arr.reshape(1, 1, *arr.shape)
        elif arr.ndim == 4:
            arr = arr.reshape(arr.shape[0], 1, *arr.shape[1:])
        elif arr.ndim != 5:
            raise ValueError("NumPy image inputs must be 2D, 3D, 4D, or 5D")
        return np.asarray(arr), {}

    if OMEArrow and isinstance(image, OMEArrow):
        record = image.data.as_py()
        arr = to_numpy(image.data, dtype=np.dtype(record["pixels_meta"]["type"]))
        return arr, record

    if isinstance(image, pa.StructScalar):
        record = image.as_py()
        return to_numpy(image, dtype=np.dtype(record["pixels_meta"]["type"])), record

    if isinstance(image, dict) and "pixels_meta" in image:
        return to_numpy(image, dtype=np.dtype(image["pixels_meta"]["type"])), image

    raise TypeError(
        "images must contain numpy.ndarray, OMEArrow, pa.StructScalar, "
        "or OME-Arrow dict values"
    )


def _cast_pixel_dtype(
    arr: np.ndarray,
    pixel_dtype: np.dtype | str | None,
    *,
    clamp: bool,
) -> np.ndarray:
    """Return ``arr`` in the requested pixel dtype, preserving dtype by default."""
    if pixel_dtype is None:
        return arr

    dtype = np.dtype(pixel_dtype)
    if arr.dtype == dtype:
        return arr

    out = arr
    if clamp and np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        out = np.clip(out, info.min, info.max)
    return out.astype(dtype, copy=False)


def _image_metadata(
    *,
    image: Any,
    arr: np.ndarray,
    record: dict[str, Any],
    index: int,
    choice: ChunkChoice,
) -> dict[str, Any]:
    """Build one image metadata row."""
    pm = record.get("pixels_meta") or {}
    image_id = record.get("id") or getattr(image, "name", None) or f"image-{index:05d}"
    name = record.get("name") or str(image_id)
    channels = pm.get("channels") or []
    st, sc, sz, sy, sx = (int(v) for v in arr.shape)
    return {
        "image_id": str(image_id),
        "name": str(name),
        "image_type": (
            None if record.get("image_type") is None else str(record.get("image_type"))
        ),
        "dtype": np.dtype(arr.dtype).name,
        "size_t": st,
        "size_c": sc,
        "size_z": sz,
        "size_y": sy,
        "size_x": sx,
        "layout": choice.layout,
        "chunk_t": choice.chunk_shape[0],
        "chunk_c": choice.chunk_shape[1],
        "chunk_z": choice.chunk_shape[2],
        "chunk_y": choice.chunk_shape[3],
        "chunk_x": choice.chunk_shape[4],
        "compression": choice.compression,
        "channels_json": json.dumps(channels),
    }


def _iter_chunk_rows(
    image_id: str,
    arr: np.ndarray,
    *,
    start_chunk_id: int,
    choice: ChunkChoice,
) -> Iterable[dict[str, Any]]:
    """Yield typed byte-buffer chunk rows for one TCZYX image."""
    ct, cc, cz, cy, cx = choice.chunk_shape
    st, sc, sz, sy, sx = arr.shape
    chunk_id = int(start_chunk_id)
    for t0 in range(0, st, ct):
        nt = min(ct, st - t0)
        for c0 in range(0, sc, cc):
            nc = min(cc, sc - c0)
            for z0 in range(0, sz, cz):
                nz = min(cz, sz - z0)
                for y0 in range(0, sy, cy):
                    ny = min(cy, sy - y0)
                    for x0 in range(0, sx, cx):
                        nx = min(cx, sx - x0)
                        chunk = np.ascontiguousarray(
                            arr[
                                t0 : t0 + nt,
                                c0 : c0 + nc,
                                z0 : z0 + nz,
                                y0 : y0 + ny,
                                x0 : x0 + nx,
                            ]
                        )
                        yield {
                            "image_id": image_id,
                            "chunk_id": chunk_id,
                            "t": t0,
                            "c": c0,
                            "z": z0,
                            "y": y0,
                            "x": x0,
                            "shape_t": nt,
                            "shape_c": nc,
                            "shape_z": nz,
                            "shape_y": ny,
                            "shape_x": nx,
                            "dtype": np.dtype(arr.dtype).name,
                            "pixel_bytes": chunk.tobytes(order="C"),
                        }
                        chunk_id += 1


def write_ome_arrow_dataset(
    images: Sequence[Any],
    output_path: str | Path,
    *,
    layout: Layout = "auto",
    access_pattern: AccessPattern = "balanced",
    target_chunk_mb: float = 4.0,
    chunk_shape: Sequence[int] | None = None,
    compression: str | None = "zstd",
    build_physical_index: bool = True,
    chunk_rows_per_row_group: int = 1,
    pixel_dtype: np.dtype | str | None = None,
    clamp: bool = True,
) -> ChunkChoice:
    """Write a metadata table and typed pixel chunk table.

    Args:
        images: Sequence of image inputs. NumPy arrays are interpreted as
            ``TCZYX`` for 5D, ``TZYX`` for 4D, ``ZYX`` for 3D, and ``YX`` for 2D.
            Existing OME-Arrow records preserve their metadata.
        output_path: Directory to write.
        layout: Physical layout mode or ``"auto"``.
        access_pattern: Preset used for automatic layout/chunk shape selection.
        target_chunk_mb: Soft maximum chunk payload size in megabytes.
        chunk_shape: Optional explicit ``TCZYX`` chunk shape.
        compression: Parquet compression setting, e.g. ``"zstd"`` or ``None``.
        build_physical_index: Whether to emit row-group index metadata.
        chunk_rows_per_row_group: Number of chunk rows to pack into each row
            group. ``1`` gives fastest direct chunk reads. Larger values reduce
            Parquet row-group overhead and can improve storage for small chunks.
        pixel_dtype: Optional output dtype for stored pixel buffers. ``None``
            preserves the source dtype. Set to ``"uint16"`` to normalize inputs
            to the legacy OME-Arrow pixel dtype.
        clamp: Whether to clamp values to the output dtype range before casting
            when ``pixel_dtype`` is set to an integer dtype.

    Returns:
        ChunkChoice: The choice used for the first image. All images currently
        share the same explicit options, but their edge chunks are clipped.
    """
    if not images:
        raise ValueError("images must contain at least one image")
    if chunk_rows_per_row_group <= 0:
        raise ValueError("chunk_rows_per_row_group must be > 0")
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    choices: list[ChunkChoice] = []
    next_chunk_id = 0
    for i, image in enumerate(images):
        arr, record = _as_tczyx_array(image)
        arr = _cast_pixel_dtype(arr, pixel_dtype, clamp=clamp)
        choice = choose_chunking(
            arr.shape,
            arr.dtype,
            layout=layout,
            access_pattern=access_pattern,
            target_chunk_mb=target_chunk_mb,
            compression=compression,
            chunk_shape=chunk_shape,
        )
        choices.append(choice)
        meta = _image_metadata(
            image=image,
            arr=arr,
            record=record,
            index=i,
            choice=choice,
        )
        if any(row["image_id"] == meta["image_id"] for row in image_rows):
            raise ValueError(
                f"Duplicate image_id in OME-Arrow dataset write: {meta['image_id']!r}"
            )
        image_rows.append(meta)
        new_rows = list(
            _iter_chunk_rows(
                meta["image_id"],
                arr,
                start_chunk_id=next_chunk_id,
                choice=choice,
            )
        )
        chunk_rows.extend(new_rows)
        next_chunk_id += len(new_rows)

    metadata = {
        b"ome.arrow.type": str(OME_ARROW_TAG_TYPE).encode(),
        b"ome.arrow.version": str(OME_ARROW_TAG_VERSION).encode(),
        b"ome.arrow.layout": b"typed-chunk-dataset",
    }
    images_table = pa.Table.from_pylist(image_rows, schema=IMAGE_TABLE_SCHEMA)
    chunks_table = pa.Table.from_pylist(chunk_rows, schema=CHUNK_TABLE_SCHEMA)
    images_table = images_table.replace_schema_metadata(metadata)
    chunks_table = chunks_table.replace_schema_metadata(metadata)

    pq.write_table(images_table, out_dir / "images.parquet", compression=compression)
    pq.write_table(
        chunks_table,
        out_dir / "chunks.parquet",
        compression=compression,
        row_group_size=int(chunk_rows_per_row_group),
    )

    index_path = out_dir / "index.parquet"
    if build_physical_index:
        index_rows = [
            {
                "image_id": row["image_id"],
                "chunk_id": row["chunk_id"],
                "row_group_id": i // int(chunk_rows_per_row_group),
                "row_index_in_group": i % int(chunk_rows_per_row_group),
                "fragment_path": "chunks.parquet",
                "t": row["t"],
                "c": row["c"],
                "z": row["z"],
                "y": row["y"],
                "x": row["x"],
                "shape_t": row["shape_t"],
                "shape_c": row["shape_c"],
                "shape_z": row["shape_z"],
                "shape_y": row["shape_y"],
                "shape_x": row["shape_x"],
            }
            for i, row in enumerate(chunk_rows)
        ]
        index_table = pa.Table.from_pylist(index_rows, schema=INDEX_TABLE_SCHEMA)
        index_table = index_table.replace_schema_metadata(metadata)
        pq.write_table(index_table, index_path, compression=compression)
    elif index_path.exists():
        index_path.unlink()

    manifest = {
        "type": OME_ARROW_TAG_TYPE,
        "version": OME_ARROW_TAG_VERSION,
        "layout": "typed-chunk-dataset",
        "image_count": len(image_rows),
        "chunk_count": len(chunk_rows),
        "choice": choices[0].__dict__,
    }
    (out_dir / "_ome_arrow_dataset.json").write_text(json.dumps(manifest, indent=2))
    return choices[0]


class OMEArrowPixels:
    """Direct typed pixel reader for an OME-Arrow dataset."""

    def __init__(self, dataset: "OMEArrowDataset") -> None:
        """Initialize a pixel reader from its parent dataset."""
        self._dataset = dataset

    def read_image(self, image_id: str) -> np.ndarray:
        """Read one image as a dense ``TCZYX`` NumPy array."""
        meta = self._dataset.image_metadata(image_id)
        return self._read_region(
            str(image_id),
            t=slice(0, int(meta["size_t"])),
            c=slice(0, int(meta["size_c"])),
            z=slice(0, int(meta["size_z"])),
            y=slice(0, int(meta["size_y"])),
            x=slice(0, int(meta["size_x"])),
        )

    def read_many(self, image_ids: Iterable[str]) -> list[np.ndarray]:
        """Read multiple images by identifier."""
        return [self.read_image(str(image_id)) for image_id in image_ids]

    def read_channel(self, image_id: str, channel: int) -> np.ndarray:
        """Read one channel as a ``TZYX`` NumPy array."""
        arr = self._read_region(
            str(image_id),
            c=slice(int(channel), int(channel) + 1),
        )
        return arr[:, 0]

    def read_plane(
        self,
        image_id: str,
        *,
        t: int = 0,
        c: int = 0,
        z: int = 0,
    ) -> np.ndarray:
        """Read one ``YX`` plane."""
        arr = self._read_region(
            str(image_id),
            t=slice(int(t), int(t) + 1),
            c=slice(int(c), int(c) + 1),
            z=slice(int(z), int(z) + 1),
        )
        return arr[0, 0, 0]

    def read_region(
        self,
        image_id: str,
        *,
        y: slice,
        x: slice,
        t: int | slice | None = None,
        c: int | slice | None = None,
        z: int | slice | None = None,
    ) -> np.ndarray:
        """Read a spatial region as a dense ``TCZYX`` NumPy array."""
        return self._read_region(
            str(image_id),
            t=_as_slice(t),
            c=_as_slice(c),
            z=_as_slice(z),
            y=y,
            x=x,
        )

    def _read_region(
        self,
        image_id: str,
        *,
        t: slice | None = None,
        c: slice | None = None,
        z: slice | None = None,
        y: slice | None = None,
        x: slice | None = None,
    ) -> np.ndarray:
        meta = self._dataset.image_metadata(image_id)
        dtype = np.dtype(meta["dtype"])
        bounds = (
            int(meta["size_t"]),
            int(meta["size_c"]),
            int(meta["size_z"]),
            int(meta["size_y"]),
            int(meta["size_x"]),
        )
        ts, cs, zs, ys, xs = _normalize_region((t, c, z, y, x), bounds)
        out = np.zeros(
            (
                ts.stop - ts.start,
                cs.stop - cs.start,
                zs.stop - zs.start,
                ys.stop - ys.start,
                xs.stop - xs.start,
            ),
            dtype=dtype,
        )

        chunks = self._dataset._matching_chunks(
            image_id,
            t=ts,
            c=cs,
            z=zs,
            y=ys,
            x=xs,
        )
        parquet_file = self._dataset._chunks_file
        for row in chunks:
            chunk = self._dataset._read_chunk_row(parquet_file, row)
            arr = np.frombuffer(chunk["pixel_bytes"], dtype=np.dtype(chunk["dtype"]))
            arr = arr.reshape(
                (
                    int(chunk["shape_t"]),
                    int(chunk["shape_c"]),
                    int(chunk["shape_z"]),
                    int(chunk["shape_y"]),
                    int(chunk["shape_x"]),
                )
            )
            _copy_overlap(out, arr, chunk, (ts, cs, zs, ys, xs))
        return out


class OMEArrowDataset:
    """Reader for metadata plus typed byte-buffer OME-Arrow datasets."""

    def __init__(self, path: str | Path) -> None:
        """Open an OME-Arrow dataset directory."""
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"No such dataset: {self.path}")
        self.images = pq.read_table(self.path / "images.parquet")
        self._chunks_file = pq.ParquetFile(self.path / "chunks.parquet")
        index_path = self.path / "index.parquet"
        self.index = pq.read_table(index_path) if index_path.exists() else None
        self.pixels = OMEArrowPixels(self)

    @property
    def image_ids(self) -> list[str]:
        """Return image identifiers in this dataset."""
        return [str(v) for v in self.images["image_id"].to_pylist()]

    def image_metadata(self, image_id: str) -> dict[str, Any]:
        """Return one image metadata row as a dictionary."""
        mask = pc.equal(self.images["image_id"], str(image_id))
        rows = self.images.filter(mask).to_pylist()
        if not rows:
            raise KeyError(f"image_id not found: {image_id}")
        return rows[0]

    def read_image(
        self,
        image_id: str | None = None,
        *,
        return_type: ArrayReturn = "numpy",
    ) -> Any:
        """Read one image as a dense ``TCZYX`` NumPy array.

        Args:
            image_id: Image identifier. When omitted, the first image is read.
            return_type: Array backend to return: ``"numpy"``, ``"torch"``,
                or ``"jax"``.

        Returns:
            Any: Dense TCZYX image in the requested array backend.
        """
        arr = self.pixels.read_image(self._resolve_image_id(image_id))
        return _convert_return(arr, return_type)

    def read_many(
        self,
        image_ids: Iterable[str],
        *,
        return_type: ArrayReturn = "numpy",
    ) -> list[Any]:
        """Read multiple images by identifier."""
        arrays = self.pixels.read_many(image_ids)
        return [_convert_return(arr, return_type) for arr in arrays]

    def read_channel(
        self,
        image_id: str | None = None,
        channel: int = 0,
        *,
        return_type: ArrayReturn = "numpy",
    ) -> Any:
        """Read one channel as a ``TZYX`` NumPy array."""
        arr = self.pixels.read_channel(self._resolve_image_id(image_id), channel)
        return _convert_return(arr, return_type)

    def read_plane(
        self,
        image_id: str | None = None,
        *,
        t: int = 0,
        c: int = 0,
        z: int = 0,
        return_type: ArrayReturn = "numpy",
    ) -> Any:
        """Read one ``YX`` plane."""
        arr = self.pixels.read_plane(self._resolve_image_id(image_id), t=t, c=c, z=z)
        return _convert_return(arr, return_type)

    def read_region(
        self,
        image_id: str | None = None,
        *,
        y: slice,
        x: slice,
        t: int | slice | None = None,
        c: int | slice | None = None,
        z: int | slice | None = None,
        return_type: ArrayReturn = "numpy",
    ) -> Any:
        """Read a spatial region as a dense ``TCZYX`` NumPy array."""
        arr = self.pixels.read_region(
            self._resolve_image_id(image_id),
            y=y,
            x=x,
            t=t,
            c=c,
            z=z,
        )
        return _convert_return(arr, return_type)

    def _resolve_image_id(self, image_id: str | None) -> str:
        """Resolve an optional image ID to the first image when omitted."""
        if image_id is not None:
            return str(image_id)
        image_ids = self.image_ids
        if not image_ids:
            raise ValueError("Dataset contains no image IDs.")
        return image_ids[0]

    def _matching_chunks(
        self,
        image_id: str,
        *,
        t: slice,
        c: slice,
        z: slice,
        y: slice,
        x: slice,
    ) -> list[dict[str, Any]]:
        if self.index is not None:
            source = self.index
        else:
            source = pq.read_table(self.path / "chunks.parquet")
        mask = pc.equal(source["image_id"], str(image_id))
        for axis, region in {
            "t": t,
            "c": c,
            "z": z,
            "y": y,
            "x": x,
        }.items():
            start = source[axis]
            stop = pc.add(source[axis], source[f"shape_{axis}"])
            axis_mask = pc.and_(
                pc.less(start, region.stop),
                pc.greater(stop, region.start),
            )
            mask = pc.and_(mask, axis_mask)
        return source.filter(mask).to_pylist()

    def _read_chunk_row(
        self,
        parquet_file: pq.ParquetFile,
        row: dict[str, Any],
    ) -> dict[str, Any]:
        if "row_group_id" in row and row["row_group_id"] is not None:
            table = parquet_file.read_row_group(
                int(row["row_group_id"]),
                columns=["chunk_id", "dtype", "pixel_bytes"],
            )
            mask = pc.equal(table["chunk_id"], int(row["chunk_id"]))
            chunk_rows = table.filter(mask).to_pylist()
            if not chunk_rows:
                raise ValueError(
                    f"Chunk {row['chunk_id']} not found in row group "
                    f"{row['row_group_id']}"
                )
            return {**row, **chunk_rows[0]}

        table = pq.read_table(
            self.path / "chunks.parquet",
            filters=[
                ("image_id", "=", row["image_id"]),
                ("chunk_id", "=", row["chunk_id"]),
            ],
        )
        chunk_rows = table.to_pylist()
        if not chunk_rows:
            raise ValueError(f"Chunk not found: {row['chunk_id']}")
        return chunk_rows[0]


def _as_slice(value: int | slice | None) -> slice | None:
    """Normalize an optional scalar index to a slice."""
    if value is None:
        return None
    if isinstance(value, slice):
        return value
    i = int(value)
    return slice(i, i + 1)


def _convert_return(arr: np.ndarray, return_type: ArrayReturn) -> Any:
    """Convert a NumPy read result to the requested array backend."""
    if return_type == "numpy":
        return arr
    if return_type == "torch":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Torch is not installed.") from exc
        return torch.as_tensor(arr)
    if return_type == "jax":
        try:
            import jax.numpy as jnp
        except ImportError as exc:
            raise RuntimeError("JAX is not installed.") from exc
        return jnp.asarray(arr)
    raise ValueError("return_type must be one of 'numpy', 'torch', or 'jax'")


def _normalize_region(
    slices: tuple[slice | None, slice | None, slice | None, slice | None, slice | None],
    bounds: tuple[int, int, int, int, int],
) -> tuple[slice, slice, slice, slice, slice]:
    """Normalize optional slices against TCZYX image bounds."""
    normalized = []
    for raw_slice, bound in zip(slices, bounds):
        axis_slice = slice(0, bound) if raw_slice is None else raw_slice
        start, stop, step = axis_slice.indices(bound)
        if step != 1:
            raise ValueError("OMEArrowDataset pixel reads support only step=1 slices")
        stop = max(stop, start)
        normalized.append(slice(start, stop))
    return tuple(normalized)  # type: ignore[return-value]


def _copy_overlap(
    out: np.ndarray,
    chunk: np.ndarray,
    row: dict[str, Any],
    region: tuple[slice, slice, slice, slice, slice],
) -> None:
    """Copy the overlap between a decoded chunk and requested output region."""
    chunk_starts = (
        int(row["t"]),
        int(row["c"]),
        int(row["z"]),
        int(row["y"]),
        int(row["x"]),
    )
    chunk_shapes = (
        int(row["shape_t"]),
        int(row["shape_c"]),
        int(row["shape_z"]),
        int(row["shape_y"]),
        int(row["shape_x"]),
    )
    out_slices = []
    chunk_slices = []
    for start, size, requested in zip(chunk_starts, chunk_shapes, region):
        overlap_start = max(start, requested.start)
        overlap_stop = min(start + size, requested.stop)
        if overlap_stop <= overlap_start:
            return
        out_slices.append(
            slice(overlap_start - requested.start, overlap_stop - requested.start)
        )
        chunk_slices.append(slice(overlap_start - start, overlap_stop - start))
    out[tuple(out_slices)] = chunk[tuple(chunk_slices)]
