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
    tcz: Tuple[int, int, int] = (0, 0, 0),
    autoscale: bool = True,
    vmin: int | None = None,
    vmax: int | None = None,
    cmap: str = "gray",
    show: bool = True,
) -> plt:
    """Display a single (t,c,z) plane from an OME-Arrow record.

    Minimal deps: pyarrow (if scalar), numpy, matplotlib.

    Args:
      data:
        OME-Arrow data as a Python dict or StructScalar.
      tcz:
        (t, c, z) indices of the plane to display. Defaults to (0,0,0).
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
    """
    if isinstance(data, pa.StructScalar):
        data = data.as_py()

    pm = data["pixels_meta"]
    sx, sy = int(pm["size_x"]), int(pm["size_y"])
    t, c, z = (int(x) for x in tcz)

    plane = next(
        (p for p in data["planes"] if int(p["t"]) == t and int(p["c"]) == c and int(p["z"]) == z),
        None,
    )
    if plane is None:
        raise ValueError(f"plane (t={t}, c={c}, z={z}) not found")

    pix = plane["pixels"]
    if len(pix) != sx * sy:
        raise ValueError(f"pixels len {len(pix)} != size_x*size_y ({sx*sy})")

    img = np.asarray(pix, dtype=np.uint16).reshape(sy, sx).copy()

    if (vmin is None or vmax is None) and autoscale:
        lo, hi = int(img.min()), int(img.max())
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
    scaling_values: tuple[float, float, float] | None = None,  # (Z, Y, X)
    opacity: str | float = "sigmoid",
    clim: tuple[float, float] | None = None,
    show_axes: bool = True,
    backend: str = "auto",  # "auto" | "trame" | "html" | "static"
    interpolation: str = "nearest",  # "nearest" or "linear"
    background: str = "black",
    percentile_clim: tuple[float, float] = (1.0, 99.9),  # robust contrast
    sampling_scale: float = 0.5,  # smaller = denser rays (sharper, slower)
):
    """
    Jupyter-inline interactive volume view using PyVista backends.
    Tries 'trame' → 'html' → 'static' when backend='auto'.

    sampling_scale controls ray step via the mapper after add_volume.
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

    # ---- spacing (dx, dy, dz) in world units
    dx = float(pm.get("physical_size_x", 1.0) or 1.0)
    dy = float(pm.get("physical_size_y", 1.0) or 1.0)
    dz = float(pm.get("physical_size_z", 1.0) or 1.0)

    # optional override from legacy scaling tuple (Z, Y, X)
    if scaling_values is None and "scaling_values" in pm:
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

    # optional downsampling (keep spacing consistent)
    if downsample > 1:
        vol_zyx = vol_zyx[::downsample, ::downsample, ::downsample]
        dz, dy, dx = dz * downsample, dy * downsample, dx * downsample

    # VTK expects (X,Y,Z) memory order
    vol_xyz = vol_zyx.transpose(2, 1, 0)  # (nx, ny, nz)
    nx, ny, nz = map(int, vol_xyz.shape)

    # ---- contrast limits (robust percentiles, like napari)
    if clim is None:
        lo, hi = np.percentile(vol_xyz, percentile_clim)
        lo = float(lo)
        hi = float(hi if hi > lo else lo + 1.0)
        clim = (lo, hi)

    # ---- backend selection
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

    # ---- build dataset
    img = pv.ImageData()
    img.dimensions = (nx, ny, nz)
    img.spacing = (dx, dy, dz)
    img.origin = (0.0, 0.0, 0.0)
    img.point_data.clear()
    img.point_data["scalars"] = np.asfortranarray(vol_xyz).ravel(order="F")

    # Make "scalars" active across PyVista versions
    try:
        img.point_data.set_active_scalars("scalars")
    except AttributeError:
        try:
            img.point_data.active_scalars_name = "scalars"
        except Exception:
            img.set_active_scalars("scalars")

    # ---- render
    pl = pv.Plotter()
    pl.set_background(background)

    # sensible opacity behavior relative to spacing
    base_sample = max(min(dx, dy, dz), 1e-6)  # avoid zero

    vol_actor = pl.add_volume(
        img,
        cmap="gray",                   # napari-like
        opacity=opacity,
        clim=clim,
        shade=False,                   # microscopy usually unshaded
        scalar_bar_args={"title": "intensity"},
        opacity_unit_distance=base_sample,   # keep opacity consistent
        # no sampling_distance kwarg here (set via mapper below)
    )

    # -- crispness & interpolation (version-safe)
    try:
        prop = getattr(vol_actor, "prop", None) or vol_actor.GetProperty()
        # nearest vs linear sampling
        if interpolation.lower().startswith("near"):
            prop.SetInterpolationTypeToNearest()
        else:
            prop.SetInterpolationTypeToLinear()
        # stop pre-map smoothing if available (big win for microscopy)
        if hasattr(prop, "SetInterpolateScalarsBeforeMapping"):
            prop.SetInterpolateScalarsBeforeMapping(False)
        # also expose scalar opacity unit distance in case kwarg unsupported
        if hasattr(prop, "SetScalarOpacityUnitDistance"):
            prop.SetScalarOpacityUnitDistance(base_sample)
    except Exception:
        pass

    # -- ray sampling density via mapper (works across many VTK versions)
    try:
        mapper = getattr(vol_actor, "mapper", None) or vol_actor.GetMapper()
        # lock sample distance if API allows
        if hasattr(mapper, "SetAutoAdjustSampleDistances"):
            mapper.SetAutoAdjustSampleDistances(False)
        if hasattr(mapper, "SetUseJittering"):
            mapper.SetUseJittering(False)
        if hasattr(mapper, "SetSampleDistance"):
            mapper.SetSampleDistance(float(base_sample * sampling_scale))
    except Exception:
        pass

    if show_axes:
        pl.add_axes()
    
    pl.show_bounds(
        color="white",
        grid="front",
        location="outer",
        ticks="both",
        xtitle="X (µm)",
        ytitle="Y (µm)",
        ztitle="Z (µm)",
    )

    def _force_white_bounds(*_args, **_kwargs):
        try:
            ren = pl.renderer

            # Modern cube-axes path
            if getattr(ren, "cube_axes_actor", None):
                ca = ren.cube_axes_actor
                # axis line colors
                for prop in (
                    ca.GetXAxesLinesProperty(),
                    ca.GetYAxesLinesProperty(),
                    ca.GetZAxesLinesProperty(),
                ):
                    prop.SetColor(1, 1, 1)
                # titles and tick labels
                for i in (0, 1, 2):  # 0:X, 1:Y, 2:Z
                    ca.GetTitleTextProperty(i).SetColor(1, 1, 1)
                    ca.GetLabelTextProperty(i).SetColor(1, 1, 1)
                ca.Modified()

            # Older/internal bounds actors
            if getattr(ren, "_bounds_actors", None):
                for actor in ren._bounds_actors.values():
                    actor.GetProperty().SetColor(1, 1, 1)
                    actor.Modified()

        except Exception:
            pass

    # run BEFORE drawing the frame so it’s visible immediately
    pl.ren_win.AddObserver("StartEvent", _force_white_bounds)

    # keep the old safety net if you like (optional):
    pl.iren.add_observer("RenderEvent", _force_white_bounds)

    def _recolor_and_render():
        _force_white_bounds()
        try:
            pl.render()  # immediate redraw so you see the white bounds now
        except Exception:
            pass

    pl.add_key_event("r", _recolor_and_render)

    return pl.show()