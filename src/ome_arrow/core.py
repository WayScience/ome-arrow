"""
Core of the ome_arrow package, used for classes and such.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

import matplotlib
import numpy as np
import pyarrow as pa

from ome_arrow.export import (
    to_numpy,
    to_ome_parquet,
    to_ome_tiff,
    to_ome_vortex,
    to_ome_zarr,
)
from ome_arrow.ingest import (
    from_numpy,
    from_ome_parquet,
    from_ome_vortex,
    from_ome_zarr,
    from_stack_pattern_path,
    from_tiff,
)
from ome_arrow.meta import OME_ARROW_STRUCT
from ome_arrow.tensor import LazyTensorView, TensorView
from ome_arrow.transform import slice_ome_arrow
from ome_arrow.utils import describe_ome_arrow
from ome_arrow.view import view_matplotlib, view_pyvista

# if not in runtime, import pyvista for type hints
if TYPE_CHECKING:
    import pyvista


@dataclass(frozen=True)
class _LazySourceSpec:
    """Deferred source description for lazy OMEArrow loading."""

    data: str
    column_name: str
    row_index: int
    image_type: str | None


@dataclass(frozen=True)
class _LazySliceSpec:
    """Deferred spatial/index slice specification."""

    x_min: int
    x_max: int
    y_min: int
    y_max: int
    t_indices: tuple[int, ...] | None
    c_indices: tuple[int, ...] | None
    z_indices: tuple[int, ...] | None
    fill_missing: bool


class OMEArrow:
    """
    Small convenience toolkit for working with ome-arrow data.

    If `input` is a TIFF path, this loads it via `tiff_to_ome_arrow`.
    If `input` is a dict, it will be converted using `to_struct_scalar`.
    If `input` is already a `pa.StructScalar`, it is used as-is.

    In Jupyter, evaluating the instance will render the first plane using
    matplotlib (via `_repr_html_`). Call `view_matplotlib()` to select a
    specific (z, t, c) plane.

    Args:
        input: TIFF path, nested dict, or `pa.StructScalar`.
        struct: Expected Arrow StructType (e.g., OME_ARROW_STRUCT).
    """

    def __init__(
        self,
        data: str | dict | pa.StructScalar | "np.ndarray",
        tcz: Tuple[int, int, int] = (0, 0, 0),
        column_name: str = "ome_arrow",
        row_index: int = 0,
        image_type: str | None = None,
        lazy: bool = False,
    ) -> None:
        """
        Construct an OMEArrow from:
        - a Bio-Formats-style stack pattern string (contains '<', '>', or '*')
        - a path/URL to an OME-TIFF (.tif/.tiff)
        - a path/URL to an OME-Zarr store (.zarr / .ome.zarr)
        - a path/URL to an OME-Parquet file (.parquet / .pq)
        - a path/URL to a Vortex file (.vortex)
        - a NumPy ndarray (2D-5D; interpreted
            with from_numpy defaults)
        - a dict already matching the OME-Arrow schema
        - a pa.StructScalar already typed to OME_ARROW_STRUCT
        - optionally override/set image_type metadata on ingest
        - optionally defer source-file ingestion with lazy=True
        """

        # set the tcz for viewing
        self.tcz = tcz
        self._data: pa.StructScalar | None = None
        self._struct_array: pa.StructArray | None = None
        self._lazy_source: _LazySourceSpec | None = None
        self._lazy_slices: list[_LazySliceSpec] = []

        if lazy:
            if not isinstance(data, str):
                raise TypeError("lazy=True currently supports only string file inputs.")
            if any(c in data for c in "<>*"):
                raise TypeError(
                    "lazy=True does not support Bio-Formats pattern strings. "
                    "Use OMEArrow(..., lazy=False) for pattern ingestion via "
                    "from_stack_pattern_path."
                )
            self._lazy_source = _LazySourceSpec(
                data=data,
                column_name=column_name,
                row_index=row_index,
                image_type=image_type,
            )
            return

        # --- 1) Stack pattern (Bio-Formats-style) --------------------------------
        if isinstance(data, str) and any(c in data for c in "<>*"):
            self.data = from_stack_pattern_path(
                data,
                default_dim_for_unspecified="C",
                map_series_to="T",
                clamp_to_uint16=True,
                image_type=image_type,
            )

        # --- 2) String path/URL: OME-Zarr / OME-Parquet / OME-TIFF ---------------
        elif isinstance(data, str):
            self.data, self._struct_array = self._load_from_string_source(
                data,
                column_name=column_name,
                row_index=row_index,
                image_type=image_type,
            )

        # --- 3) NumPy ndarray ----------------------------------------------------
        elif isinstance(data, np.ndarray):
            # Uses from_numpy defaults: dim_order="TCZYX", clamp_to_uint16=True, etc.
            # If the array is YX/ZYX/CYX/etc.,
            # from_numpy will expand/reorder accordingly.
            self.data = from_numpy(data, image_type=image_type)

        # --- 4) Already-typed Arrow scalar ---------------------------------------
        elif isinstance(data, pa.StructScalar):
            self.data = data
            if image_type is not None:
                self.data = self._wrap_with_image_type(self.data, image_type)

        # --- 5) Plain dict matching the schema -----------------------------------
        elif isinstance(data, dict):
            record = {f.name: data.get(f.name) for f in OME_ARROW_STRUCT}
            self.data = pa.scalar(record, type=OME_ARROW_STRUCT)
            if image_type is not None:
                self.data = self._wrap_with_image_type(self.data, image_type)

        # --- otherwise ------------------------------------------------------------
        else:
            raise TypeError(
                "input data must be str, dict, pa.StructScalar, or numpy.ndarray"
            )

    @classmethod
    def scan(
        cls,
        data: str,
        *,
        tcz: Tuple[int, int, int] = (0, 0, 0),
        column_name: str = "ome_arrow",
        row_index: int = 0,
        image_type: str | None = None,
    ) -> "OMEArrow":
        """Create a lazily-loaded OMEArrow, similar to Polars scan semantics.

        Args:
            data: Input source path/URL.
            tcz: Default `(t, c, z)` indices used for view helpers.
            column_name: OME-Arrow column name for tabular sources.
            row_index: Row index for tabular sources.
            image_type: Optional image type override.

        Returns:
            OMEArrow: Lazily planned OMEArrow instance.
        """
        return cls(
            data=data,
            tcz=tcz,
            column_name=column_name,
            row_index=row_index,
            image_type=image_type,
            lazy=True,
        )

    @property
    def is_lazy(self) -> bool:
        """Return whether this instance still has deferred work."""
        return self._lazy_source is not None or bool(self._lazy_slices)

    @property
    def data(self) -> pa.StructScalar:
        """Return the materialized OME-Arrow StructScalar.

        Returns:
            pa.StructScalar: Materialized OME-Arrow record.

        Raises:
            RuntimeError: If the record could not be initialized.
        """
        self._ensure_materialized()
        if self._data is None:
            raise RuntimeError("OMEArrow data is not initialized.")
        return self._data

    @data.setter
    def data(self, value: pa.StructScalar) -> None:
        self._data = value

    def collect(self) -> "OMEArrow":
        """Materialize deferred source data and return ``self``.

        Returns:
            OMEArrow: The same instance after materialization.
        """
        self._ensure_materialized()
        return self

    @staticmethod
    def _load_from_string_source(
        data: str,
        *,
        column_name: str,
        row_index: int,
        image_type: str | None,
    ) -> tuple[pa.StructScalar, pa.StructArray | None]:
        s = data.strip()
        path = pathlib.Path(s)
        struct_array: pa.StructArray | None = None

        if (
            s.lower().endswith(".zarr")
            or s.lower().endswith(".ome.zarr")
            or ".zarr/" in s.lower()
            or (path.exists() and path.is_dir() and path.suffix.lower() == ".zarr")
        ):
            scalar = from_ome_zarr(s)
            if image_type is not None:
                scalar = OMEArrow._wrap_with_image_type(scalar, image_type)
            return scalar, None

        if s.lower().endswith((".parquet", ".pq")) or path.suffix.lower() in {
            ".parquet",
            ".pq",
        }:
            parquet_result = from_ome_parquet(
                s,
                column_name=column_name,
                row_index=row_index,
                return_array=True,
            )
            scalar, struct_array = parquet_result
            if image_type is not None:
                scalar = OMEArrow._wrap_with_image_type(scalar, image_type)
            return scalar, struct_array

        if s.lower().endswith(".vortex") or path.suffix.lower() == ".vortex":
            vortex_result = from_ome_vortex(
                s,
                column_name=column_name,
                row_index=row_index,
                return_array=True,
            )
            scalar, struct_array = vortex_result
            if image_type is not None:
                scalar = OMEArrow._wrap_with_image_type(scalar, image_type)
            return scalar, struct_array

        if path.suffix.lower() in {".tif", ".tiff"} or s.lower().endswith(
            (".tif", ".tiff")
        ):
            scalar = from_tiff(s)
            if image_type is not None:
                scalar = OMEArrow._wrap_with_image_type(scalar, image_type)
            return scalar, None

        if path.exists() and path.is_dir():
            raise ValueError(
                f"Directory '{s}' exists but does not look like an OME-Zarr store "
                "(expected suffix '.zarr' or '.ome.zarr')."
            )

        raise ValueError(
            "String input must be one of:\n"
            "  • Bio-Formats pattern string (contains '<', '>' or '*')\n"
            "  • OME-Zarr path/URL ending with '.zarr' or '.ome.zarr'\n"
            "  • OME-Parquet file ending with '.parquet' or '.pq'\n"
            "  • Vortex file ending with '.vortex'\n"
            "  • OME-TIFF path/URL ending with '.tif' or '.tiff'"
        )

    def _ensure_materialized(self) -> None:
        if self._lazy_source is None:
            return
        lazy_source = self._lazy_source
        # Intentionally do not clear `_lazy_source` / `_lazy_slices` before load.
        # If `_load_from_string_source(...)` raises, lazy state is preserved so
        # callers can inspect/retry without losing the deferred plan.
        scalar, struct_array = self._load_from_string_source(
            lazy_source.data,
            column_name=lazy_source.column_name,
            row_index=lazy_source.row_index,
            image_type=lazy_source.image_type,
        )
        if self._lazy_slices:
            data = scalar
            for spec in self._lazy_slices:
                data = slice_ome_arrow(
                    data=data,
                    x_min=spec.x_min,
                    x_max=spec.x_max,
                    y_min=spec.y_min,
                    y_max=spec.y_max,
                    t_indices=spec.t_indices,
                    c_indices=spec.c_indices,
                    z_indices=spec.z_indices,
                    fill_missing=spec.fill_missing,
                )
            # Applying lazy slices via `slice_ome_arrow` materializes through a
            # StructScalar path, so we intentionally drop `_struct_array` here.
            # Consequence: Arrow-backed zero-copy tensor paths
            # (for example `tensor_view(...).to_dlpack(mode="arrow")`) are not
            # available after lazy slicing.
            self.data = data
            self._struct_array = None
        else:
            self.data, self._struct_array = scalar, struct_array
        # Lazy state is cleared only after a successful materialization.
        self._lazy_source = None
        self._lazy_slices = []

    def _tensor_source(self) -> pa.StructScalar | pa.StructArray:
        self._ensure_materialized()
        return self._struct_array if self._struct_array is not None else self.data

    @staticmethod
    def _wrap_with_image_type(
        data: pa.StructScalar, image_type: str
    ) -> pa.StructScalar:
        return pa.scalar(
            {
                **data.as_py(),
                "image_type": str(image_type),
            },
            type=OME_ARROW_STRUCT,
        )

    def export(  # noqa: PLR0911
        self,
        how: str = "numpy",
        dtype: np.dtype = np.uint16,
        strict: bool = True,
        clamp: bool = False,
        *,
        # common writer args
        out: str | None = None,
        dim_order: str = "TCZYX",
        # OME-TIFF args
        compression: str | None = "zlib",
        compression_level: int = 6,
        tile: tuple[int, int] | None = None,
        # OME-Zarr args
        chunks: tuple[int, int, int, int, int] | None = None,  # (T,C,Z,Y,X)
        zarr_compressor: str | None = "zstd",
        zarr_level: int = 7,
        # optional display metadata (both paths guard/ignore if unsafe)
        use_channel_colors: bool = False,
        # Parquet args
        parquet_column_name: str = "ome_arrow",
        parquet_compression: str | None = "zstd",
        parquet_metadata: dict[str, str] | None = None,
        vortex_column_name: str = "ome_arrow",
        vortex_metadata: dict[str, str] | None = None,
    ) -> np.array | dict | pa.StructScalar | str:
        """
        Export the OME-Arrow content in a chosen representation.

        Args
        ----
        how:
            "numpy"     → TCZYX np.ndarray
            "dict"      → plain Python dict
            "scalar"    → pa.StructScalar (as-is)
            "ome-tiff"  → write OME-TIFF via BioIO
            "ome-zarr"  → write OME-Zarr (OME-NGFF) via BioIO
            "parquet"   → write a single-row Parquet with one struct column
            "vortex"    → write a single-row Vortex file with one struct column
        dtype:
            Target dtype for "numpy"/writers (default: np.uint16).
        strict:
            For "numpy": raise if a plane has wrong pixel length.
        clamp:
            For "numpy"/writers: clamp values into dtype range before cast.

        Keyword-only (writer specific)
        ------------------------------
        out:
            Output path (required for 'ome-tiff', 'ome-zarr', and 'parquet').
        dim_order:
            Axes string for BioIO writers; default "TCZYX".
        compression / compression_level / tile:
            OME-TIFF options (passed through to tifffile via BioIO).
        chunks / zarr_compressor / zarr_level :
            OME-Zarr options (chunk shape, compressor hint, level). If chunks is
            None, a TCZYX default is chosen (1,1,<=4,<=512,<=512).
        use_channel_colors:
            Try to embed per-channel display colors when safe; otherwise omitted.
        parquet_*:
            Options for Parquet export (column name, compression, file metadata).
        vortex_*:
            Options for Vortex export (column name, file metadata).

        Returns
        -------
        Any
            - "numpy": np.ndarray (T, C, Z, Y, X)
            - "dict":  dict
            - "scalar": pa.StructScalar
            - "ome-tiff": output path (str)
            - "ome-zarr": output path (str)
            - "parquet": output path (str)
            - "vortex": output path (str)

        Raises
        ------
        ValueError:
            Unknown 'how' or missing required params.
        """
        self._ensure_materialized()

        # existing modes
        if how == "numpy":
            return to_numpy(self.data, dtype=dtype, strict=strict, clamp=clamp)
        if how == "dict":
            return self.data.as_py()
        if how == "scalar":
            return self.data

        mode = how.lower().replace("_", "-")

        # OME-TIFF via BioIO
        if mode in {"ome-tiff", "ometiff", "tiff"}:
            if not out:
                raise ValueError("export(how='ome-tiff') requires 'out' path.")
            to_ome_tiff(
                self.data,
                out,
                dtype=dtype,
                clamp=clamp,
                dim_order=dim_order,
                compression=compression,
                compression_level=int(compression_level),
                tile=tile,
                use_channel_colors=use_channel_colors,
            )
            return out

        # OME-Zarr via BioIO
        if mode in {"ome-zarr", "omezarr", "zarr"}:
            if not out:
                raise ValueError("export(how='ome-zarr') requires 'out' path.")
            to_ome_zarr(
                self.data,
                out,
                dtype=dtype,
                clamp=clamp,
                dim_order=dim_order,
                chunks=chunks,
                compressor=zarr_compressor,
                compressor_level=int(zarr_level),
            )
            return out

        # Parquet (single row, single struct column)
        if mode in {"ome-parquet", "omeparquet", "parquet"}:
            if not out:
                raise ValueError("export(how='parquet') requires 'out' path.")
            to_ome_parquet(
                data=self.data,
                out_path=out,
                column_name=parquet_column_name,
                compression=parquet_compression,  # default 'zstd'
                file_metadata=parquet_metadata,
            )
            return out

        # Vortex (single row, single struct column)
        if mode in {"ome-vortex", "omevortex", "vortex"}:
            if not out:
                raise ValueError("export(how='vortex') requires 'out' path.")
            to_ome_vortex(
                data=self.data,
                out_path=out,
                column_name=vortex_column_name,
                file_metadata=vortex_metadata,
            )
            return out

        raise ValueError(f"Unknown export method: {how}")

    def info(self) -> Dict[str, Any]:
        """
        Describe the OME-Arrow data structure.

        Returns:
            dict with keys:
                - shape: (T, C, Z, Y, X)
                - type: classification string
                - summary: human-readable text
        """
        self._ensure_materialized()
        return describe_ome_arrow(self.data)

    def view(
        self,
        how: str = "matplotlib",
        tcz: tuple[int, int, int] = (0, 0, 0),
        autoscale: bool = True,
        vmin: int | None = None,
        vmax: int | None = None,
        cmap: str = "gray",
        show: bool = True,
        c: int | None = None,
        downsample: int = 1,
        opacity: str | float = "sigmoid",
        clim: tuple[float, float] | None = None,
        show_axes: bool = True,
        scaling_values: tuple[float, float, float] | None = None,
    ) -> matplotlib.figure.Figure | "pyvista.Plotter":
        """
        Render an OME-Arrow record using Matplotlib or PyVista.

        This convenience method supports two rendering backends:

        * ``how="matplotlib"`` — renders a single (t, c, z) plane as a 2D image.
        Returns a Matplotlib :class:`~matplotlib.figure.Figure` (or whatever
        :func:`view_matplotlib` returns) and optionally displays it with
        ``plt.show()`` when ``show=True``.

        * ``how="pyvista"`` — creates an interactive 3D PyVista visualization in
        Jupyter. When ``show=True``, displays the widget. Independently, a static
        PNG snapshot is embedded in the notebook (inside a collapsed
        ``<details>`` block) for non-interactive renderers (e.g., GitHub).

        Args:
        how: Rendering backend. One of ``"matplotlib"`` or ``"pyvista"``.
        tcz: The (t, c, z) indices of the plane to display when using Matplotlib.
            Defaults to ``(0, 0, 0)``.
        autoscale: If ``True`` and ``vmin``/``vmax`` are not provided, infer
            display limits from the image data range (Matplotlib path only).
        vmin: Lower display limit for intensity scaling (Matplotlib path only).
        vmax: Upper display limit for intensity scaling (Matplotlib path only).
        cmap: Matplotlib colormap name for single-channel display (Matplotlib only).
        show: Whether to display the plot immediately. For Matplotlib, calls
            ``plt.show()``. For PyVista, calls ``plotter.show()``.
        c: Channel index override for the PyVista view. If ``None``, uses
            ``tcz[1]`` (the ``c`` from ``tcz``).
        downsample: Integer downsampling factor for the PyVista volume or slices.
            Must be ``>= 1``.
        opacity: Opacity specification for PyVista. Either a float in ``[0, 1]``
            or the string ``"sigmoid"`` (backend interprets as a preset transfer
            function).
        clim: Contrast limits (``(low, high)``) for PyVista rendering.
        show_axes: If ``True``, display axes in the PyVista scene.
        scaling_values: Physical scale multipliers for the (x, y, z) axes used by
            PyVista, typically to express anisotropy. If ``None``, uses metadata
            scaling from the OME-Arrow record (pixels_meta.physical_size_x/y/z).
            These scaling values will default to 1µm if metadata is missing in
            source image metadata.

        Returns:
        matplotlib.figure.Figure | pyvista.Plotter:
            * If ``how="matplotlib"``, returns the figure created by
            :func:`view_matplotlib` (often a :class:`~matplotlib.figure.Figure`).
            * If ``how="pyvista"``, returns the created :class:`pyvista.Plotter`.

        Raises:
        ValueError: If a requested plane (``t,c,z``) is not found or if pixel
            array dimensions are inconsistent (propagated from
            :func:`view_matplotlib`).
        TypeError: If parameter types are invalid (e.g., negative ``downsample``).

        Notes:
        * The PyVista path embeds a static PNG snapshot via Pillow (``PIL``). If
            Pillow is unavailable, the method logs a warning and skips the snapshot,
            but the interactive viewer is still returned.
        * When ``show=False`` and ``how="pyvista"``, no interactive window is
            opened, but the returned :class:`pyvista.Plotter` can be shown later.

        Examples:
        Display a single plane with Matplotlib:

        >>> fig = obj.view(how="matplotlib", tcz=(0, 1, 5), cmap="magma")

        Create an interactive PyVista scene in a Jupyter notebook:

        >>> plotter = obj.view(how="pyvista", c=0, downsample=2, show=True)

        Configure PyVista contrast limits and keep axes hidden:

        >>> plotter = obj.view(how="pyvista", clim=(100, 2000), show_axes=False)
        """
        self._ensure_materialized()

        if how == "matplotlib":
            return view_matplotlib(
                self.data,
                tcz=tcz,
                autoscale=autoscale,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                show=show,
            )

        if how == "pyvista":
            import base64
            import io

            from IPython.display import HTML, display

            c_idx = int(tcz[1] if c is None else c)
            plotter = view_pyvista(
                data=self.data,
                c=c_idx,
                downsample=downsample,
                opacity=opacity,
                clim=clim,
                show_axes=show_axes,
                scaling_values=scaling_values,
                show=False,
            )

            # 1) show the interactive widget for live work
            if show:
                plotter.show()

            # 2) capture a PNG and embed it in a collapsed details block
            try:
                img = plotter.screenshot(return_img=True)  # ndarray
                if img is not None:
                    buf = io.BytesIO()
                    # use matplotlib-free writer: PyVista returns RGB(A) uint8
                    from PIL import (
                        Image as PILImage,
                    )  # pillow is a light dep most envs have

                    PILImage.fromarray(img).save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                    display(
                        HTML(
                            f"""
                        <details>
                        <summary>Static snapshot (for non-interactive view)</summary>
                        <img src="data:image/png;base64,{b64}" />
                        </details>
                        """
                        )
                    )
            except Exception as e:
                print(f"Warning: could not save PyVista snapshot: {e}")

            return plotter

    def tensor_view(
        self,
        *,
        scene: int | None = None,
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
    ) -> TensorView | LazyTensorView:
        """Create a TensorView of the pixel data.

        Args:
            scene: Scene index (only 0 is supported for single-image records).
            t: Time index selection (int, slice, or sequence). Default: all.
            z: Z index selection (int, slice, or sequence). Default: all.
            c: Channel index selection (int, slice, or sequence). Default: all.
            roi: Spatial crop (x, y, w, h) in pixels.
            roi3d: Spatial + depth crop (x, y, z, w, h, d) in pixels/planes.
                This is a convenience alias for ``roi=(x, y, w, h)`` and
                ``z=slice(z, z + d)``.
            tile: Tile index (tile_y, tile_x) based on chunk grid.
            layout: Desired layout string using TZCHW letters where
                T=time, Z=depth, C=channel, H=height (Y), W=width (X).
            dtype: Output dtype override.
            chunk_policy: Handling for ``pyarrow.ChunkedArray`` inputs.
            channel_policy: Behavior when dropping `C` from layout while
                multiple channels are selected. "error" raises (default).
                "first" keeps the first channel.

        Returns:
            TensorView | LazyTensorView: Tensor view over selected pixels.
            In lazy mode, this returns a deferred ``LazyTensorView`` that
            materializes on first execution call (for example ``to_numpy()``).

        Raises:
            ValueError: If an unsupported scene is requested.
        """

        if scene not in (None, 0):
            raise ValueError("Only scene=0 is supported for single-image records.")

        if self._lazy_source is not None:
            return LazyTensorView(
                loader=self._tensor_source,
                t=t,
                z=z,
                c=c,
                roi=roi,
                roi3d=roi3d,
                tile=tile,
                layout=layout,
                dtype=dtype,
                chunk_policy=chunk_policy,
                channel_policy=channel_policy,
            )

        # TensorView uses an internal canonical axis basis (TZCHW) for shape/stride
        # math, then applies the requested layout permutation for output.
        return TensorView(
            self._struct_array if self._struct_array is not None else self.data,
            t=t,
            z=z,
            c=c,
            roi=roi,
            roi3d=roi3d,
            tile=tile,
            layout=layout,
            dtype=dtype,
            chunk_policy=chunk_policy,
            channel_policy=channel_policy,
        )

    def slice(
        self,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        t_indices: Optional[Iterable[int]] = None,
        c_indices: Optional[Iterable[int]] = None,
        z_indices: Optional[Iterable[int]] = None,
        fill_missing: bool = True,
    ) -> OMEArrow:
        """
        Create a cropped copy of an OME-Arrow record.

        Crops spatially to [y_min:y_max, x_min:x_max] (half-open) and, if provided,
        filters/reindexes T/C/Z to the given index sets.

        Parameters
        ----------
        x_min, x_max, y_min, y_max : int
            Half-open crop bounds in pixels (0-based).
        t_indices, c_indices, z_indices : Iterable[int] | None
            Optional explicit indices to keep for T, C, Z. If None, keep all.
            Selected indices are reindexed to 0..len-1 in the output.
        fill_missing : bool
            If True, any missing (t,c,z) planes in the selection are zero-filled.

        Returns
        -------
        OMEArrow object
            New OME-Arrow record with updated sizes and planes.
        """
        self._ensure_materialized()

        return OMEArrow(
            data=slice_ome_arrow(
                data=self.data,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                t_indices=t_indices,
                c_indices=c_indices,
                z_indices=z_indices,
                fill_missing=fill_missing,
            )
        )

    def slice_lazy(
        self,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        t_indices: Optional[Iterable[int]] = None,
        c_indices: Optional[Iterable[int]] = None,
        z_indices: Optional[Iterable[int]] = None,
        fill_missing: bool = True,
    ) -> OMEArrow:
        """Return a lazily planned slice, collected on first execution.

        For lazy sources created with ``OMEArrow.scan(...)``, this queues a
        deferred slice operation and returns a new lazy OMEArrow plan produced
        from ``OMEArrow.scan(...)``.
        For already materialized sources, this falls back to eager ``slice()``.
        This method does not mutate ``self``.

        Notes:
            ``slice_lazy`` always returns a new plan object. Internally, the
            returned plan gets a fresh ``_lazy_slices`` list
            (``[*self._lazy_slices, new_slice]``), so chained plans do not
            share mutable slice state with the original ``OMEArrow``.
            A common footgun is:
            ``oa.slice_lazy(...).collect()`` followed by ``oa.tensor_view(...)``.
            Those calls can load/materialize the same source twice because
            ``oa`` remains the original plan. For a single-load workflow, keep
            working from the value returned by ``slice_lazy`` / ``collect``.

        Args:
            x_min: Inclusive minimum X index for the crop.
            x_max: Exclusive maximum X index for the crop.
            y_min: Inclusive minimum Y index for the crop.
            y_max: Exclusive maximum Y index for the crop.
            t_indices: Optional time indices to retain.
            c_indices: Optional channel indices to retain.
            z_indices: Optional depth indices to retain.
            fill_missing: Whether to zero-fill missing `(t, c, z)` planes.

        Returns:
            OMEArrow: Lazy plan when source is lazy; eager slice result otherwise.
        """
        if self._lazy_source is None:
            return self.slice(
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                t_indices=t_indices,
                c_indices=c_indices,
                z_indices=z_indices,
                fill_missing=fill_missing,
            )

        lazy_source = self._lazy_source
        planned = OMEArrow.scan(
            lazy_source.data,
            tcz=self.tcz,
            column_name=lazy_source.column_name,
            row_index=lazy_source.row_index,
            image_type=lazy_source.image_type,
        )
        planned._lazy_slices = [
            *self._lazy_slices,
            _LazySliceSpec(
                x_min=int(x_min),
                x_max=int(x_max),
                y_min=int(y_min),
                y_max=int(y_max),
                t_indices=(
                    None if t_indices is None else tuple(int(i) for i in t_indices)
                ),
                c_indices=(
                    None if c_indices is None else tuple(int(i) for i in c_indices)
                ),
                z_indices=(
                    None if z_indices is None else tuple(int(i) for i in z_indices)
                ),
                fill_missing=bool(fill_missing),
            ),
        ]
        return planned

    def _repr_html_(self) -> str:
        """
        Auto-render a plane as inline PNG in Jupyter.
        """
        try:
            self._ensure_materialized()
            view_matplotlib(
                data=self.data,
                tcz=self.tcz,
                autoscale=True,
                vmin=None,
                vmax=None,
                cmap="gray",
                show=False,
            )
            # return blank string to avoid showing class representation below image
            return self.info()["summary"]
        except Exception as e:
            # Fallback to a tiny text status if rendering fails.
            return f"<pre>OMEArrowKit: render failed: {e}</pre>"
