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
from ome_arrow.convert import tiff_to_ome_arrow, to_numpy
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

        if isinstance(data, str):
            path = pathlib.Path(data)
            if path.suffix.lower() in {".tif", ".tiff"}:
                self.data = tiff_to_ome_arrow(path, OME_ARROW_STRUCT)
            else:
                raise ValueError(
                    "String input data must be a .tif/.tiff path."
                )
        elif isinstance(data, pa.StructScalar):
            self.data = data
        elif isinstance(data, dict):
            # Assumes dict matches the schema (or raises).
            self.data = pa.scalar(data, type=OME_ARROW_STRUCT)
        else:
            raise TypeError("input data must be str, dict, or pa.StructScalar")

    def export(
        self,
        how: str = "numpy",
        dtype: np.dtype = np.uint16,
        strict: bool = True,
        clamp: bool = False,
    ) -> Any:
        """
        Export the OME-Arrow content in a chosen representation.

        Args:
            how:
                "numpy"  → TCZYX np.ndarray
                "dict"   → plain Python dict
                "scalar" → pa.StructScalar (as-is)
            dtype:
                Target dtype for "numpy" export (default: np.uint16).
            strict:
                For "numpy": raise if a plane has wrong pixel length.
            clamp:
                For "numpy": clamp values into dtype range before cast.

        Returns:
            The exported object per the `how` argument.

        Raises:
            ValueError: If `how` is unknown.
        """
        if how == "numpy":
            return to_numpy(
                self.data, dtype=dtype, strict=strict, clamp=clamp
            )
        if how == "dict":
            return self.data.as_py()
        if how == "scalar":
            return self.data
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
        export_html: Optional[str] = None,
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
        