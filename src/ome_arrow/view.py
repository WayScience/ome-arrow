"""
Viewing utilities for OME-Arrow data.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pyarrow as pa
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, Sequence
import numpy as np
import pyarrow as pa
import pyvista as pv


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
    plt.axis("off")

    if show:
        plt.show()

    return plt



def view_pyvista(
    data: dict | pa.StructScalar,
    c: int = 0,
    downsample: int = 1,
    scaling_values: tuple[float, float, float] | None = None,  # NEW: (Z, Y, X)
    opacity: str | float = "sigmoid",
    clim: tuple[float, float] | None = None,
    show_axes: bool = True,
    backend: str = "auto",  # "auto" | "trame" | "html" | "static"
):
    """
    Jupyter-inline interactive volume view using PyVista 0.46+ backends.
    Tries 'trame' → 'html' → 'static' when backend='auto'.

    Parameters
    ----------
    scaling_values : (Z, Y, X) or None
        Legacy-style voxel spacing tuple (Z, Y, X). If provided, overrides
        pixels_meta.physical_size_{x,y,z}. If None, uses pixels_meta.
    """
    import warnings
    import numpy as np
    import pyvista as pv

    # ---- unwrap OME-Arrow row
    row = data.as_py() if isinstance(data, pa.StructScalar) else data
    pm = row["pixels_meta"]
    sx, sy, sz = int(pm["size_x"]), int(pm["size_y"]), int(pm["size_z"])
    sc, st = int(pm["size_c"]), int(pm["size_t"])
    if not (0 <= c < sc):
        raise ValueError(f"Channel out of range: 0..{sc-1}")

    # ---- spacing (PyVista expects (dx, dy, dz) ≡ world units along X,Y,Z)
    # default from OME-Arrow metadata
    dx = float(pm.get("physical_size_x", 1.0) or 1.0)
    dy = float(pm.get("physical_size_y", 1.0) or 1.0)
    dz = float(pm.get("physical_size_z", 1.0) or 1.0)

    # optional override from legacy scaling tuple (Z, Y, X)
    if scaling_values is None and "scaling_values" in pm:
        # if you stored it in pixels_meta before, honor it
        try:
            sz_legacy, sy_legacy, sx_legacy = pm["scaling_values"]
            dz, dy, dx = float(sz_legacy), float(sy_legacy), float(sx_legacy)
        except Exception:
            pass
    elif scaling_values is not None:
        sz_legacy, sy_legacy, sx_legacy = scaling_values
        dz, dy, dx = float(sz_legacy), float(sy_legacy), float(sx_legacy)

    # ---- rebuild (Z,Y,X) for T=0, channel c
    vol_zyx = np.zeros((sz, sy, sx), dtype=np.uint16)
    for p in row["planes"]:
        if int(p["t"]) == 0 and int(p["c"]) == c:
            z = int(p["z"])
            vol_zyx[z] = np.asarray(p["pixels"], dtype=np.uint16).reshape(sy, sx)

    # optional downsampling: scale both data and spacing
    if downsample > 1:
        vol_zyx = vol_zyx[::downsample, ::downsample, ::downsample]
        dz, dy, dx = dz * downsample, dy * downsample, dx * downsample

    # VTK expects (X,Y,Z) memory order and spacing=(dx,dy,dz)
    vol_xyz = vol_zyx.transpose(2, 1, 0)   # (nx, ny, nz)
    nx, ny, nz = map(int, vol_xyz.shape)

    if clim is None:
        vmin, vmax = float(vol_xyz.min()), float(vol_xyz.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
        clim = (vmin, vmax)

    # ---- select backend
    def _try_backend(name: str) -> bool:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*notebook backend.*",
                                    category=UserWarning)
            try:
                pv.set_jupyter_backend(name)
                return True
            except Exception:
                return False

    if backend == "auto":
        backend_used = "trame" if _try_backend("trame") \
                       else "html" if _try_backend("html") \
                       else "static"
    else:
        backend_used = backend if _try_backend(backend) else "static"

    pv.OFF_SCREEN = False

    # ---- build dataset as ImageData (UniformGrid)
    img = pv.ImageData()
    img.dimensions = (nx, ny, nz)   # number of points along each axis (X,Y,Z)
    img.spacing = (dx, dy, dz)      # physical spacing along each axis (X,Y,Z)
    img.origin = (0.0, 0.0, 0.0)
    img.point_data.clear()
    img.point_data["scalars"] = np.asfortranarray(vol_xyz).ravel(order="F")

    # ---- render
    pl = pv.Plotter()
    pl.set_background("#555555")
    pl.add_volume(
        img,
        cmap="binary",
        opacity=opacity,
        clim=clim,
        scalar_bar_args={"title": "intensity"},
    )
    if show_axes:
        pl.add_axes()
    pl.add_text(f"T=0 / {max(0, st-1)}  C={c}  [{backend_used}]",
                font_size=10)
    return pl.show()