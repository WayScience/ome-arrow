"""
Viewing utilities for OME-Arrow data.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pyarrow as pa
import matplotlib.pyplot as plt

import warnings


def view_matplotlib(
    data: Dict[str, Any] | pa.StructScalar,
    ztc: Tuple[int, int, int] = (0, 0, 0),
    autoscale: bool = True,
    vmin: int | None = None,
    vmax: int | None = None,
    cmap: str = "gray",
    show: bool = True,
) -> plt:
    """Display a single (z,t,c) plane from an OME-Arrow record.

    Minimal deps: pyarrow (if scalar), numpy, matplotlib.

    Args:
      data:
        OME-Arrow data as a Python dict or StructScalar.
      ztc:
        (z, t, c) indices of the plane to display. Defaults to (0,0,0).
      autoscale:
        If True and vmin/vmax not passed, set vmin/vmax from data range.
      vmin:
        Optional lower limit for display scaling.
      vmax:
        Optional upper limit for display scaling.
      cmap:
        Matplotlib colormap for single-channel display.
      show:
        If True (default), calls plt.show() to display the image.

    Raises:
      ValueError: If the requested plane is not found or shapes mismatch.

    Notes:
      * Expects data["pixels_meta"] with size_x, size_y.
      * Expects data["planes"] list of {z,t,c,pixels}, where pixels is a
        flat uint16 sequence of length size_x * size_y.
    """
    # Unwrap Arrow scalar to plain Python dict if needed.
    if isinstance(data, pa.StructScalar):
        data = data.as_py()

    pm = data["pixels_meta"]
    sx, sy = int(pm["size_x"]), int(pm["size_y"])
    target = tuple(int(x) for x in ztc)

    plane = None
    for p in data["planes"]:
        if (int(p["z"]), int(p["t"]), int(p["c"])) == target:
            plane = p
            break
    if plane is None:
        raise ValueError(f"plane {target} not found")

    pix = plane["pixels"]
    if len(pix) != sx * sy:
        raise ValueError(
            f"pixels len {len(pix)} != size_x*size_y ({sx*sy})"
        )

    # Make 2D array. Cast to uint16; copy avoids view traps.
    img = np.asarray(pix, dtype=np.uint16).reshape(sy, sx).copy()

    # Decide intensity scaling.
    if vmin is None or vmax is None:
        if autoscale:
            lo = int(img.min())
            hi = int(img.max())
            # Avoid degenerate scale when flat image.
            if hi == lo:
                hi = lo + 1
            vmin = lo if vmin is None else vmin
            vmax = hi if vmax is None else vmax

    plt.figure()
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    z, t, c = target
    plt.title(f"OME-Arrow plane z={z} t={t} c={c}  ({sx}×{sy})")
    plt.axis("off")

    if show:
        plt.show()

    return plt