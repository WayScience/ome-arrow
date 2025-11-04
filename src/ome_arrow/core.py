"""
Core of the ome_arrow package, used for classes and such.
"""
from __future__ import annotations

import pathlib
from typing import Any, Optional, Tuple

import pyarrow as pa
import numpy as np
from ome_arrow.meta import OME_ARROW_STRUCT
from ome_arrow.view import view_matplotlib
from ome_arrow.convert import tiff_to_ome_arrow, to_numpy


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

    def view(
        self,
        how: str = "matplotlib",
        ztc: Tuple[int, int, int] = (0, 0, 0),
        autoscale: bool = True,
        vmin: Optional[int] = None,
        vmax: Optional[int] = None,
        cmap: str = "gray",
        show: bool = True,
    ) -> Any:
        """
        Render a (z, t, c) plane with various tools.

        Args:
            ztc: Tuple of (z, t, c). Defaults to (0, 0, 0).
            autoscale: When True and vmin/vmax unset, use data min/max.
            vmin: Optional lower display bound.
            vmax: Optional upper display bound.
            cmap: Matplotlib colormap (single-channel).

        Returns:
            A visualization of the object.
        """

        if how == "matplotlib":
            view_matplotlib(
                self.data,
                ztc=ztc,
                autoscale=autoscale,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                show=show
            )
        else:
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
            return ""
        except Exception as e:
            # Fallback to a tiny text status if rendering fails.
            return f"<pre>OMEArrowKit: render failed: {e}</pre>"
        