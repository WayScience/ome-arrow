"""
Core of the ome_arrow package, used for classes and such.
"""
from __future__ import annotations

import pathlib
from typing import Any, Optional, Tuple, Dict

import pyarrow as pa
import numpy as np
from ome_arrow.meta import OME_ARROW_STRUCT
from ome_arrow.view import view_matplotlib, view_pyvista
from ome_arrow.ingest import from_tiff, from_stack_pattern_path, from_ome_zarr
from ome_arrow.export import to_numpy, to_ome_tiff, to_ome_zarr, to_ome_parquet
from ome_arrow.utils import describe_ome_arrow


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
        data: str | dict | pa.StructScalar,
    ):
        """
        Construct an OMEArrow from:
        - a Bio-Formats-style stack pattern string (contains '<', '>', or '*')
        - a path/URL to an OME-TIFF (.tif/.tiff)
        - a path/URL to an OME-Zarr store (.zarr / .ome.zarr)
        - a path/URL to an OME-Parquet file (.parquet / .pq)
        - a dict already matching the OME-Arrow schema
        - a pa.StructScalar already typed to OME_ARROW_STRUCT
        """
        import pathlib
        from ome_arrow.meta import OME_ARROW_STRUCT
        from ome_arrow.ingest import (
            from_tiff,
            from_ome_zarr,
            from_stack_pattern_path,
            from_parquet,        # ← NEW
        )

        # --- 1) Stack pattern (Bio-Formats-style) --------------------------------
        if isinstance(data, str) and any(c in data for c in "<>*"):
            self.data = from_stack_pattern_path(
                data,
                default_dim_for_unspecified="C",
                map_series_to="T",
                clamp_to_uint16=True,
            )
            return

        # --- 2) String path/URL: OME-Zarr / OME-Parquet / OME-TIFF ---------------
        if isinstance(data, str):
            s = data.strip()
            path = pathlib.Path(s)

            # Inline Zarr detection: suffix or substring check
            if (
                s.lower().endswith(".zarr")
                or s.lower().endswith(".ome.zarr")
                or ".zarr/" in s.lower()
                or (path.exists() and path.is_dir() and path.suffix.lower() == ".zarr")
            ):
                self.data = from_ome_zarr(s)
                return

            # OME-Parquet detection (single-file parquet container)
            if s.lower().endswith((".parquet", ".pq")) or path.suffix.lower() in {".parquet", ".pq"}:
                # Uses defaults: column_name="ome_arrow", row_index=0
                self.data = from_parquet(s)
                return

            # TIFF ingest
            if path.suffix.lower() in {".tif", ".tiff"} or s.lower().endswith((".tif", ".tiff")):
                self.data = from_tiff(s)
                return

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
                "  • OME-TIFF path/URL ending with '.tif' or '.tiff'"
            )

        # --- 3) Already-typed Arrow scalar ---------------------------------------
        if isinstance(data, pa.StructScalar):
            self.data = data
            return

        # --- 4) Plain dict matching the schema -----------------------------------
        if isinstance(data, dict):
            self.data = pa.scalar(data, type=OME_ARROW_STRUCT)
            return

        # --- otherwise ------------------------------------------------------------
        raise TypeError("input data must be str, dict, or pa.StructScalar")
    def export(
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
        chunks: tuple[int, int, int, int, int] | None = None,   # (T,C,Z,Y,X)
        zarr_compressor: str | None = "zstd",
        zarr_level: int = 7,
        # optional display metadata (both paths guard/ignore if unsafe)
        use_channel_colors: bool = False,
        # Parquet args
        parquet_column_name: str = "ome_arrow",
        parquet_compression: str | None = "zstd",
        parquet_metadata: dict[str, str] | None = None,
    ) -> Any:
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
            OME-Zarr options (chunk shape, compressor hint, level).
        use_channel_colors:
            Try to embed per-channel display colors when safe; otherwise omitted.
        parquet_*:
            Options for Parquet export (column name, compression, file metadata).

        Returns
        -------
        Any
            - "numpy": np.ndarray (T, C, Z, Y, X)
            - "dict":  dict
            - "scalar": pa.StructScalar
            - "ome-tiff": output path (str)
            - "ome-zarr": output path (str)
            - "parquet": output path (str)

        Raises
        ------
        ValueError:
            Unknown 'how' or missing required params.
        """
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
                compression=parquet_compression,   # default 'zstd'
                file_metadata=parquet_metadata,
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
        return describe_ome_arrow(self.data)
        

    def view(
        self,
        how: str = "matplotlib",
        ztc: Tuple[int, int, int] = (0, 0, 0),
        autoscale: bool = True,
        vmin: Optional[int] = None,
        vmax: Optional[int] = None,
        cmap: str = "gray",
        show: bool = True,
        c: Optional[int] = None,
        downsample: int = 1,
        opacity: str | float = "sigmoid",
        clim: Optional[Tuple[float, float]] = None,
        show_axes: bool = True,
        scaling_values: Optional[tuple[float, float, float]] = (1.0, 0.1, 0.1),
    ) -> Any:
        if how == "matplotlib":
            return view_matplotlib(
                self.data,
                ztc=ztc,
                autoscale=autoscale,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                show=show,
            )

        if how == "pyvista":
            c_idx = int(ztc[2] if c is None else c)
            return view_pyvista(
                data=self.data,
                c=c_idx,
                downsample=downsample,
                opacity=opacity,
                clim=clim,
                show_axes=show_axes,
                scaling_values=scaling_values,
            )

        raise ValueError(f"Unknown view method: {how}")
        
    def _repr_html_(self) -> str:
        """
        Auto-render first plane (0,0,0) as inline PNG in Jupyter.
        """
        try:
            view_matplotlib(
                data=self.data,
                ztc=(0, 0, 0),
                autoscale=True,
                vmin=None,
                vmax=None,
                cmap="gray",
                show=False
            )
            # return blank string to avoid showing class representation below image
            return self.info()["summary"]
        except Exception as e:
            # Fallback to a tiny text status if rendering fails.
            return f"<pre>OMEArrowKit: render failed: {e}</pre>"
        