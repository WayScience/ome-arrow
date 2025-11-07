"""
Converting to and from OME-Arrow formats.
"""

import itertools
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import bioio_ome_tiff
import bioio_tifffile
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from bioio import BioImage
from bioio_ome_zarr import Reader as OMEZarrReader

from ome_arrow.meta import OME_ARROW_STRUCT, OME_ARROW_TAG_TYPE, OME_ARROW_TAG_VERSION


def to_ome_arrow(
    type_: str = OME_ARROW_TAG_TYPE,
    version: str = OME_ARROW_TAG_VERSION,
    image_id: str = "unnamed",
    name: str = "unknown",
    acquisition_datetime: Optional[datetime] = None,
    dimension_order: str = "XYZCT",
    dtype: str = "uint16",
    size_x: int = 1,
    size_y: int = 1,
    size_z: int = 1,
    size_c: int = 1,
    size_t: int = 1,
    physical_size_x: float = 1.0,
    physical_size_y: float = 1.0,
    physical_size_z: float = 1.0,
    physical_size_unit: str = "µm",
    channels: Optional[List[Dict[str, Any]]] = None,
    planes: Optional[List[Dict[str, Any]]] = None,
    masks: Any = None,
) -> pa.StructScalar:
    """
    Create a typed OME-Arrow StructScalar with sensible defaults.

    This builds and validates a nested dict that conforms to the given
    StructType (e.g., OME_ARROW_STRUCT). You can override any field
    explicitly; others use safe defaults.

    Args:
        type_: Top-level type string ("ome.arrow" by default).
        version: Specification version string.
        image_id: Unique image identifier.
        name: Human-friendly name.
        acquisition_datetime: Datetime of acquisition (defaults to now).
        dimension_order: Dimension order ("XYZCT" or "XYCT").
        dtype: Pixel data type string (e.g., "uint16").
        size_x, size_y, size_z, size_c, size_t: Axis sizes.
        physical_size_x/y/z: Physical scaling in µm.
        physical_size_unit: Unit string, default "µm".
        channels: List of channel dicts. Autogenerates one if None.
        planes: List of plane dicts. Empty if None.
        masks: Optional placeholder for future annotations.

    Returns:
        pa.StructScalar: A validated StructScalar for the schema.

    Example:
        >>> s = to_struct_scalar(OME_ARROW_STRUCT, image_id="img001")
        >>> s.type == OME_ARROW_STRUCT
        True
    """

    type_ = str(type_)
    version = str(version)
    image_id = str(image_id)
    name = str(name)
    dimension_order = str(dimension_order)
    dtype = str(dtype)
    physical_size_unit = str(physical_size_unit)

    # Sensible defaults for channels and planes
    if channels is None:
        channels = [
            {
                "id": "ch-0",
                "name": "default",
                "emission_um": 0.0,
                "excitation_um": 0.0,
                "illumination": "Unknown",
                "color_rgba": 0xFFFFFFFF,
            }
        ]
    else:
        # --- NEW: coerce channel text fields to str ------------------
        for ch in channels:
            if "id" in ch:
                ch["id"] = str(ch["id"])
            if "name" in ch:
                ch["name"] = str(ch["name"])
            if "illumination" in ch:
                ch["illumination"] = str(ch["illumination"])

    if planes is None:
        planes = [{"z": 0, "t": 0, "c": 0, "pixels": [0] * (size_x * size_y)}]

    record = {
        "type": type_,
        "version": version,
        "id": image_id,
        "name": name,
        "acquisition_datetime": acquisition_datetime or datetime.now(timezone.utc),
        "pixels_meta": {
            "dimension_order": dimension_order,
            "type": dtype,
            "size_x": size_x,
            "size_y": size_y,
            "size_z": size_z,
            "size_c": size_c,
            "size_t": size_t,
            "physical_size_x": physical_size_x,
            "physical_size_y": physical_size_y,
            "physical_size_z": physical_size_z,
            "physical_size_x_unit": physical_size_unit,
            "physical_size_y_unit": physical_size_unit,
            "physical_size_z_unit": physical_size_unit,
            "channels": channels,
        },
        "planes": planes,
        "masks": masks,
    }

    return pa.scalar(record, type=OME_ARROW_STRUCT)


def from_numpy(
    arr: np.ndarray,
    *,
    dim_order: str = "TCZYX",
    image_id: Optional[str] = None,
    name: Optional[str] = None,
    channel_names: Optional[Sequence[str]] = None,
    acquisition_datetime: Optional[datetime] = None,
    clamp_to_uint16: bool = True,
    # meta
    physical_size_x: float = 1.0,
    physical_size_y: float = 1.0,
    physical_size_z: float = 1.0,
    physical_size_unit: str = "µm",
    dtype_meta: Optional[str] = None,  # if None, inferred from output dtype
) -> pa.StructScalar:
    """
    Build an OME-Arrow StructScalar from a NumPy array.

    Parameters
    ----------
    arr : np.ndarray
        Image data with axes described by `dim_order`.
    dim_order : str, default "TCZYX"
        Axis labels for `arr`. Must include "Y" and "X".
        Supported examples: "YX", "ZYX", "CYX", "CZYX", "TYX", "TCYX", "TCZYX".
    image_id, name : Optional[str]
        Identifiers to embed in the record.
    channel_names : Optional[Sequence[str]]
        Names for channels; defaults to C0..C{n-1}.
    acquisition_datetime : Optional[datetime]
        Defaults to now (UTC) if None.
    clamp_to_uint16 : bool, default True
        If True, clamp/cast planes to uint16 before serialization.
    physical_size_x/y/z : float
        Spatial pixel sizes (µm), Z used if present.
    physical_size_unit : str
        Unit string for spatial axes (default "µm").
    dtype_meta : Optional[str]
        Pixel dtype string to place in metadata; if None, inferred from the
        (possibly cast) array's dtype.

    Returns
    -------
    pa.StructScalar
        Typed OME-Arrow record (schema = OME_ARROW_STRUCT).

    Notes
    -----
    - If Z is not in `dim_order`, `size_z` will be 1 and the meta
      dimension_order becomes "XYCT"; otherwise "XYZCT".
    - If T/C are absent in `dim_order`, they default to size 1.
    """

    if not isinstance(arr, np.ndarray):
        raise TypeError("from_numpy expects a NumPy ndarray.")

    dims = dim_order.upper()
    if "Y" not in dims or "X" not in dims:
        raise ValueError("dim_order must include 'Y' and 'X' axes.")

    # Map current axes -> indices
    axis_to_idx: Dict[str, int] = {ax: i for i, ax in enumerate(dims)}

    # Extract sizes with defaults for missing axes
    size_x = int(arr.shape[axis_to_idx["X"]])
    size_y = int(arr.shape[axis_to_idx["Y"]])
    size_z = int(arr.shape[axis_to_idx["Z"]]) if "Z" in axis_to_idx else 1
    size_c = int(arr.shape[axis_to_idx["C"]]) if "C" in axis_to_idx else 1
    size_t = int(arr.shape[axis_to_idx["T"]]) if "T" in axis_to_idx else 1

    if size_x <= 0 or size_y <= 0:
        raise ValueError("Image must have positive Y and X dimensions.")

    # Reorder to a standard (T, C, Z, Y, X) view for plane extraction
    desired_axes = ["T", "C", "Z", "Y", "X"]
    current_axes = list(dims)
    # Insert absent axes with size 1 using np.expand_dims
    view = arr
    for ax in desired_axes:
        if ax not in axis_to_idx:
            # Append a new singleton axis at the end, then we'll permute
            view = np.expand_dims(view, axis=-1)
            # Pretend this new axis now exists at the last index
            current_axes.append(ax)
            axis_to_idx = {a: i for i, a in enumerate(current_axes)}

    # Permute to TCZYX
    perm = [axis_to_idx[a] for a in desired_axes]
    tczyx = np.transpose(view, axes=perm)

    # Validate final shape
    if tuple(tczyx.shape) != (size_t, size_c, size_z, size_y, size_x):
        # This should not happen, but guard just in case
        raise ValueError(
            "Internal axis reordering mismatch: "
            f"got {tczyx.shape} vs expected {(size_t, size_c, size_z, size_y, size_x)}"
        )

    # Clamp/cast
    if clamp_to_uint16 and tczyx.dtype != np.uint16:
        tczyx = np.clip(tczyx, 0, 65535).astype(np.uint16, copy=False)

    # Channel names
    if not channel_names or len(channel_names) != size_c:
        channel_names = [f"C{i}" for i in range(size_c)]
    channel_names = [str(x) for x in channel_names]

    channels = [
        {
            "id": f"ch-{i}",
            "name": channel_names[i],
            "emission_um": 0.0,
            "excitation_um": 0.0,
            "illumination": "Unknown",
            "color_rgba": 0xFFFFFFFF,
        }
        for i in range(size_c)
    ]

    # Build planes: flatten YX per (t,c,z)
    planes: List[Dict[str, Any]] = []
    for t in range(size_t):
        for c in range(size_c):
            for z in range(size_z):
                plane = tczyx[t, c, z]
                planes.append(
                    {"z": z, "t": t, "c": c, "pixels": plane.ravel().tolist()}
                )

    # Meta dimension_order: mirror your other ingests
    meta_dim_order = "XYCT" if size_z == 1 else "XYZCT"

    # Pixel dtype in metadata
    dtype_str = dtype_meta or np.dtype(tczyx.dtype).name

    return to_ome_arrow(
        image_id=str(image_id or "unnamed"),
        name=str(name or "unknown"),
        acquisition_datetime=acquisition_datetime or datetime.now(timezone.utc),
        dimension_order=meta_dim_order,
        dtype=dtype_str,
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        size_c=size_c,
        size_t=size_t,
        physical_size_x=float(physical_size_x),
        physical_size_y=float(physical_size_y),
        physical_size_z=float(physical_size_z),
        physical_size_unit=str(physical_size_unit),
        channels=channels,
        planes=planes,
        masks=None,
    )


def from_tiff(
    tiff_path: str | Path,
    image_id: Optional[str] = None,
    name: Optional[str] = None,
    channel_names: Optional[Sequence[str]] = None,
    acquisition_datetime: Optional[datetime] = None,
    clamp_to_uint16: bool = True,
) -> pa.StructScalar:
    """
    Read a TIFF and return a typed OME-Arrow StructScalar.

    Uses bioio to read TCZYX (or XY) data, flattens each YX plane, and
    delegates struct creation to `to_struct_scalar`.

    Args:
        tiff_path: Path to a TIFF readable by bioio.
        image_id: Optional stable image identifier (defaults to stem).
        name: Optional human label (defaults to file name).
        channel_names: Optional channel names; defaults to C0..C{n-1}.
        acquisition_datetime: Optional acquisition time (UTC now if None).
        clamp_to_uint16: If True, clamp/cast planes to uint16.

    Returns:
        pa.StructScalar validated against `struct`.
    """

    p = Path(tiff_path)

    img = BioImage(
        image=str(p),
        reader=(
            bioio_ome_tiff.Reader
            if str(p).lower().endswith(("ome.tif", "ome.tiff"))
            else bioio_tifffile.Reader
        ),
    )

    arr = np.asarray(img.data)  # (T, C, Z, Y, X)
    dims = img.dims
    size_t = int(dims.T or 1)
    size_c = int(dims.C or 1)
    size_z = int(dims.Z or 1)
    size_y = int(dims.Y or arr.shape[-2])
    size_x = int(dims.X or arr.shape[-1])
    if size_x <= 0 or size_y <= 0:
        raise ValueError("Image must have positive Y and X dims.")

    pps = getattr(img, "physical_pixel_sizes", None)
    try:
        psize_x = float(getattr(pps, "X", None) or 1.0)
        psize_y = float(getattr(pps, "Y", None) or 1.0)
        psize_z = float(getattr(pps, "Z", None) or 1.0)
    except Exception:
        psize_x = psize_y = psize_z = 1.0

    # --- NEW: coerce top-level strings --------------------------------
    img_id = str(image_id or p.stem)
    display_name = str(name or p.name)

    # --- NEW: ensure channel_names is list[str] ------------------------
    if not channel_names or len(channel_names) != size_c:
        channel_names = [f"C{i}" for i in range(size_c)]
    channel_names = [str(x) for x in channel_names]

    channels = [
        {
            "id": f"ch-{i}",
            "name": channel_names[i],
            "emission_um": 0.0,
            "excitation_um": 0.0,
            "illumination": "Unknown",
            "color_rgba": 0xFFFFFFFF,
        }
        for i in range(size_c)
    ]

    planes: List[Dict[str, Any]] = []
    for t in range(size_t):
        for c in range(size_c):
            for z in range(size_z):
                plane = arr[t, c, z]
                if clamp_to_uint16 and plane.dtype != np.uint16:
                    plane = np.clip(plane, 0, 65535).astype(np.uint16)
                planes.append(
                    {"z": z, "t": t, "c": c, "pixels": plane.ravel().tolist()}
                )

    dim_order = "XYCT" if size_z == 1 else "XYZCT"

    return to_ome_arrow(
        image_id=img_id,
        name=display_name,
        acquisition_datetime=acquisition_datetime or datetime.now(timezone.utc),
        dimension_order=dim_order,
        dtype="uint16",
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        size_c=size_c,
        size_t=size_t,
        physical_size_x=psize_x,
        physical_size_y=psize_y,
        physical_size_z=psize_z,
        physical_size_unit="µm",
        channels=channels,
        planes=planes,
        masks=None,
    )


def from_stack_pattern_path(
    pattern_path: str | Path,
    default_dim_for_unspecified: str = "C",
    map_series_to: Optional[str] = "T",
    clamp_to_uint16: bool = True,
    channel_names: Optional[List[str]] = None,
    image_id: Optional[str] = None,
    name: Optional[str] = None,
) -> pa.StructScalar:
    path = Path(pattern_path)
    folder = path.parent
    line = path.name.strip()
    if not line:
        raise ValueError("Pattern path string is empty or malformed")

    DIM_TOKENS = {
        "C": {"c", "ch", "w", "wavelength"},
        "T": {"t", "tl", "tp", "timepoint"},
        "Z": {"z", "zs", "sec", "fp", "focal", "focalplane"},
        "S": {"s", "sp", "series"},
    }
    NUM_RANGE_RE = re.compile(r"^(?P<a>\d+)\-(?P<b>\d+)(?::(?P<step>\d+))?$")

    def detect_dim(before_text: str) -> Optional[str]:
        m = re.search(r"([A-Za-z]+)$", before_text)
        if not m:
            return None
        token = m.group(1).lower()
        for dim, names in DIM_TOKENS.items():
            if token in names:
                return dim
        return None

    def expand_raw_token(raw: str) -> Tuple[List[str], bool]:
        raw = raw.strip()
        if "," in raw and not NUM_RANGE_RE.match(raw):
            parts = [p.strip() for p in raw.split(",")]
            return parts, all(p.isdigit() for p in parts)
        m = NUM_RANGE_RE.match(raw)
        if m:
            a, b = m.group("a"), m.group("b")
            step = int(m.group("step") or "1")
            start, stop = int(a), int(b)
            if stop < start:
                raise ValueError(f"Inverted range not supported: <{raw}>")
            width = max(len(a), len(b))
            nums = [str(v).zfill(width) for v in range(start, stop + 1, step)]
            return nums, True
        return [raw], raw.isdigit()

    def parse_bracket_pattern(s: str) -> Tuple[str, List[Dict[str, Any]]]:
        placeholders, out = [], []
        i = ph_i = 0
        while i < len(s):
            if s[i] == "<":
                j = s.find(">", i + 1)
                if j == -1:
                    raise ValueError("Unclosed '<' in pattern.")
                raw_inside = s[i + 1 : j]
                before = "".join(out)
                dim = detect_dim(before) or "?"
                choices, is_num = expand_raw_token(raw_inside)
                placeholders.append(
                    {
                        "idx": ph_i,
                        "raw": raw_inside,
                        "choices": choices,
                        "dim": dim,
                        "is_numeric": is_num,
                    }
                )
                out.append(f"{{{ph_i}}}")
                ph_i += 1
                i = j + 1
            else:
                out.append(s[i])
                i += 1
        return "".join(out), placeholders

    def regex_match(folder: Path, regex: str) -> List[Path]:
        r = re.compile(regex)
        return sorted(
            [p for p in folder.iterdir() if p.is_file() and r.fullmatch(p.name)]
        )

    matched: Dict[Tuple[int, int, int], Path] = {}
    literal_channel_names: Optional[List[str]] = None

    if "<" in line and ">" in line:
        template, placeholders = parse_bracket_pattern(line)
        for ph in placeholders:
            ph["dim"] = (ph["dim"] or "?").upper()
            if ph["dim"] == "?":
                ph["dim"] = default_dim_for_unspecified.upper()

        for combo in itertools.product(*[ph["choices"] for ph in placeholders]):
            fname = template.format(*combo)
            fpath = folder / fname
            if not fpath.exists():
                continue

            t = c = z = 0
            for ph, val in zip(placeholders, combo):
                idx = ph["choices"].index(val)
                dim = ph["dim"]
                if dim == "S":
                    if not map_series_to:
                        raise ValueError("Encountered 'series' but map_series_to=None")
                    dim = map_series_to.upper()
                if dim == "T":
                    t = idx
                elif dim == "C":
                    c = idx
                elif dim == "Z":
                    z = idx

            if literal_channel_names is None:
                for ph in placeholders:
                    dim_eff = ph["dim"] if ph["dim"] != "S" else (map_series_to or "S")
                    if dim_eff == "C" and not ph["is_numeric"]:
                        literal_channel_names = ph["choices"]
                        break

            matched[(t, c, z)] = fpath
    else:
        for z, p in enumerate(regex_match(folder, line)):
            matched[(0, 0, z)] = p

    if not matched:
        raise FileNotFoundError(f"No files matched pattern: {pattern_path}")

    size_t = max(k[0] for k in matched) + 1
    size_c = max(k[1] for k in matched) + 1
    size_z = max(k[2] for k in matched) + 1

    if channel_names and len(channel_names) != size_c:
        raise ValueError(
            f"channel_names length {len(channel_names)} != size_c {size_c}"
        )
    if not channel_names:
        channel_names = literal_channel_names or [f"C{i}" for i in range(size_c)]

    # ---- PROBE SHAPE (NEW: accept TCZYX and squeeze singleton axes) ----
    sample = next(iter(matched.values()))
    is_ome = sample.suffix.lower() in (".ome.tif", ".ome.tiff")
    img0 = BioImage(
        image=str(sample),
        reader=(bioio_ome_tiff.Reader if is_ome else bioio_tifffile.Reader),
    )
    a0 = np.asarray(img0.data)
    # bioio returns TCZYX or YX; normalize to TCZYX
    if a0.ndim == 2:
        _T0, _C0, _Z0, Y0, X0 = 1, 1, 1, a0.shape[0], a0.shape[1]
    else:
        # Heuristic: last two are (Y,X); leading dims are (T,C,Z) possibly singleton
        Y0, X0 = a0.shape[-2], a0.shape[-1]
        lead = a0.shape[:-2]
        # Pad leading dims to T,C,Z (left-aligned)
        _T0, _C0, _Z0 = ([*list(lead), 1, 1, 1])[:3]
    size_y, size_x = Y0, X0

    # physical pixel sizes
    pps = getattr(img0, "physical_pixel_sizes", None)
    try:
        psize_x = float(getattr(pps, "X", None) or 1.0)
        psize_y = float(getattr(pps, "Y", None) or 1.0)
        psize_z = float(getattr(pps, "Z", None) or 1.0)
    except Exception:
        psize_x = psize_y = psize_z = 1.0

    # ---- BUILD PLANES (NEW: support Z-stacks within a single file when T=C=1) ----
    planes: List[Dict[str, Any]] = []

    def _ensure_u16(arr: np.ndarray) -> np.ndarray:
        if clamp_to_uint16 and arr.dtype != np.uint16:
            arr = np.clip(arr, 0, 65535).astype(np.uint16)
        return arr

    for t in range(size_t):
        for c in range(size_c):
            for z in range(size_z):
                fpath = matched.get((t, c, z))
                if fpath is None:
                    # missing plane: zero-fill
                    planes.append(
                        {"z": z, "t": t, "c": c, "pixels": [0] * (size_x * size_y)}
                    )
                    continue

                reader = (
                    bioio_ome_tiff.Reader
                    if fpath.suffix.lower() in (".ome.tif", ".ome.tiff")
                    else bioio_tifffile.Reader
                )
                im = BioImage(image=str(fpath), reader=reader)
                arr = np.asarray(im.data)

                if arr.ndim == 2:
                    # Direct YX
                    if arr.shape != (size_y, size_x):
                        raise ValueError(
                            f"Shape mismatch for {fpath.name}:"
                            f" {arr.shape} vs {(size_y, size_x)}"
                        )
                    arr = _ensure_u16(arr)
                    planes.append(
                        {"z": z, "t": t, "c": c, "pixels": arr.ravel().tolist()}
                    )
                else:
                    # Treat as TCZYX; extract dims
                    Y, X = arr.shape[-2], arr.shape[-1]
                    lead = arr.shape[:-2]
                    Tn, Cn, Zn = ([*list(lead), 1, 1, 1])[:3]
                    if (size_y, size_x) != (Y, X):
                        raise ValueError(
                            f"Shape mismatch for {fpath.name}:"
                            f" {(Y, X)} vs {(size_y, size_x)}"
                        )

                    # Case A: singleton TCZ -> squeeze to YX
                    if Tn == 1 and Cn == 1 and Zn == 1:
                        plane2d = _ensure_u16(arr.reshape(Y, X))
                        planes.append(
                            {"z": z, "t": t, "c": c, "pixels": plane2d.ravel().tolist()}
                        )
                    # Case B: multi-Z only (expand across Z)
                    elif Tn == 1 and Cn == 1 and Zn > 1:
                        # spill Z pages starting at this z index
                        for z_local in range(Zn):
                            plane2d = _ensure_u16(
                                arr.reshape(1, 1, Zn, Y, X)[0, 0, z_local]
                            )
                            z_idx = z + z_local
                            planes.append(
                                {
                                    "z": z_idx,
                                    "t": t,
                                    "c": c,
                                    "pixels": plane2d.ravel().tolist(),
                                }
                            )
                        # bump global size_z if we exceeded it
                        size_z = max(size_z, z + Zn)
                    else:
                        # For now, we require multi-T/C pages to be
                        # expressed by the filename pattern,
                        # not embedded inside a single file.
                        raise ValueError(
                            f"{fpath.name} contains "
                            f"multiple pages across T/C/Z={Tn, Cn, Zn}; "
                            f"only Z>1 with T=C=1 is supported inside one file. "
                            f"Please express T/C via the filename pattern."
                        )

    # Adjust channels (meta)
    channels_meta = [
        {
            "id": f"ch-{i}",
            "name": str((channel_names or [f"C{i}" for i in range(size_c)])[i]),
            "emission_um": 0.0,
            "excitation_um": 0.0,
            "illumination": "Unknown",
            "color_rgba": 0xFFFFFFFF,
        }
        for i in range(size_c)
    ]

    dim_order = "XYZCT" if size_z > 1 else "XYCT"
    display_name = name or str(pattern_path)
    img_id = image_id or path.stem

    return to_ome_arrow(
        image_id=str(img_id),
        name=str(display_name),
        acquisition_datetime=None,
        dimension_order=dim_order,
        dtype="uint16",
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        size_c=size_c,
        size_t=size_t,
        physical_size_x=psize_x,
        physical_size_y=psize_y,
        physical_size_z=psize_z,
        physical_size_unit="µm",
        channels=channels_meta,
        planes=planes,
        masks=None,
    )


def from_ome_zarr(
    zarr_path: str | Path,
    image_id: Optional[str] = None,
    name: Optional[str] = None,
    channel_names: Optional[Sequence[str]] = None,
    acquisition_datetime: Optional[datetime] = None,
    clamp_to_uint16: bool = True,
) -> pa.StructScalar:
    """
    Read an OME-Zarr directory and return a typed OME-Arrow StructScalar.

    Uses BioIO with the OMEZarrReader backend to read TCZYX (or XY) data,
    flattens each YX plane into OME-Arrow planes, and builds a validated
    StructScalar via `to_ome_arrow`.

    Args:
        zarr_path:
            Path to the OME-Zarr directory (e.g., "image.ome.zarr").
        image_id:
            Optional stable image identifier (defaults to directory stem).
        name:
            Optional display name (defaults to directory name).
        channel_names:
            Optional list of channel names. Defaults to C0, C1, ...
        acquisition_datetime:
            Optional datetime (defaults to UTC now).
        clamp_to_uint16:
            If True, cast pixels to uint16.

    Returns:
        pa.StructScalar: Validated OME-Arrow struct for this image.
    """
    p = Path(zarr_path)

    img = BioImage(image=str(p), reader=OMEZarrReader)

    arr = np.asarray(img.data)  # shape (T, C, Z, Y, X)
    dims = img.dims

    size_t = int(dims.T or 1)
    size_c = int(dims.C or 1)
    size_z = int(dims.Z or 1)
    size_y = int(dims.Y or arr.shape[-2])
    size_x = int(dims.X or arr.shape[-1])

    if size_x <= 0 or size_y <= 0:
        raise ValueError("Image must have positive Y and X dimensions.")

    pps = getattr(img, "physical_pixel_sizes", None)
    try:
        psize_x = float(getattr(pps, "X", None) or 1.0)
        psize_y = float(getattr(pps, "Y", None) or 1.0)
        psize_z = float(getattr(pps, "Z", None) or 1.0)
    except Exception:
        psize_x = psize_y = psize_z = 1.0

    img_id = str(image_id or p.stem)
    display_name = str(name or p.name)

    # Infer or assign channel names
    if not channel_names or len(channel_names) != size_c:
        try:
            chs = getattr(img, "channel_names", None)
            if chs is None:
                chs = [getattr(ch, "name", None) for ch in getattr(img, "channels", [])]
            if chs and len(chs) == size_c and all(c is not None for c in chs):
                channel_names = [str(c) for c in chs]
            else:
                channel_names = [f"C{i}" for i in range(size_c)]
        except Exception:
            channel_names = [f"C{i}" for i in range(size_c)]
    channel_names = [str(x) for x in channel_names]

    channels = [
        {
            "id": f"ch-{i}",
            "name": channel_names[i],
            "emission_um": 0.0,
            "excitation_um": 0.0,
            "illumination": "Unknown",
            "color_rgba": 0xFFFFFFFF,
        }
        for i in range(size_c)
    ]

    planes: List[Dict[str, Any]] = []
    for t in range(size_t):
        for c in range(size_c):
            for z in range(size_z):
                plane = arr[t, c, z]
                if clamp_to_uint16 and plane.dtype != np.uint16:
                    plane = np.clip(plane, 0, 65535).astype(np.uint16)
                planes.append(
                    {"z": z, "t": t, "c": c, "pixels": plane.ravel().tolist()}
                )

    dim_order = "XYCT" if size_z == 1 else "XYZCT"

    return to_ome_arrow(
        image_id=img_id,
        name=display_name,
        acquisition_datetime=acquisition_datetime or datetime.now(timezone.utc),
        dimension_order=dim_order,
        dtype="uint16",
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        size_c=size_c,
        size_t=size_t,
        physical_size_x=psize_x,
        physical_size_y=psize_y,
        physical_size_z=psize_z,
        physical_size_unit="µm",
        channels=channels,
        planes=planes,
        masks=None,
    )


def from_ome_parquet(
    parquet_path: str | Path,
    *,
    column_name: Optional[str] = "ome_arrow",
    row_index: int = 0,
    strict_schema: bool = False,
) -> pa.StructScalar:
    """
    Read an OME-Arrow record from a Parquet file and return a typed StructScalar.

    Expected layout (as produced by `to_ome_parquet`):
      - single Parquet file
      - a single column (default name "ome_arrow") of `OME_ARROW_STRUCT` type
      - one row (row_index=0)

    This function is forgiving:
      - If `column_name` is None or not found, it will auto-detect a struct column
        that matches the OME-Arrow field names.
      - If the table has multiple rows, you can choose which record to read
        via `row_index`.

    Parameters
    ----------
    parquet_path : str | Path
        Path to the .parquet file.
    column_name : Optional[str], default "ome_arrow"
        Name of the column that stores the OME-Arrow struct. If None, auto-detect.
    row_index : int, default 0
        Which row to read if the table contains multiple rows.
    strict_schema : bool, default False
        If True, require the column's type to equal `OME_ARROW_STRUCT` exactly.
        If False, we only require the column to be a Struct with the same field
        names (order can vary).

    Returns
    -------
    pa.StructScalar
        A validated OME-Arrow struct scalar.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If a suitable column/row cannot be found or schema checks fail.
    """
    p = Path(parquet_path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")

    table = pq.read_table(p)

    if table.num_rows == 0:
        raise ValueError("Parquet file contains 0 rows; expected at least 1.")
    if not (0 <= row_index < table.num_rows):
        raise ValueError(f"row_index {row_index} out of range [0, {table.num_rows}).")

    # 1) Locate the OME-Arrow column
    def _struct_matches_ome_fields(t: pa.StructType) -> bool:
        ome_fields = {f.name for f in OME_ARROW_STRUCT}
        col_fields = {f.name for f in t}
        return ome_fields == col_fields

    candidate_col = None

    if column_name is not None and column_name in table.column_names:
        arr = table[column_name]
        if not pa.types.is_struct(arr.type):
            raise ValueError(f"Column '{column_name}' is not a Struct; got {arr.type}.")
        if strict_schema and arr.type != OME_ARROW_STRUCT:
            raise ValueError(
                f"Column '{column_name}' schema != OME_ARROW_STRUCT.\n"
                f"Got:   {arr.type}\n"
                f"Expect:{OME_ARROW_STRUCT}"
            )
        if not strict_schema and not _struct_matches_ome_fields(arr.type):
            raise ValueError(
                f"Column '{column_name}' does not have the expected OME-Arrow fields."
            )
        candidate_col = arr
    else:
        # Auto-detect a struct column that matches OME-Arrow fields
        for name in table.column_names:
            arr = table[name]
            if pa.types.is_struct(arr.type):
                if strict_schema and arr.type == OME_ARROW_STRUCT:
                    candidate_col = arr
                    column_name = name
                    break
                if not strict_schema and _struct_matches_ome_fields(arr.type):
                    candidate_col = arr
                    column_name = name
                    break
        if candidate_col is None:
            if column_name is None:
                hint = "no struct column with OME-Arrow fields was found."
            else:
                hint = f"column '{column_name}' not found and auto-detection failed."
            raise ValueError(f"Could not locate an OME-Arrow struct column: {hint}")

    # 2) Extract the row as a Python dict
    #    (Using to_pylist() for the single element slice is simple & reliable.)
    record_dict: Dict[str, Any] = candidate_col.slice(row_index, 1).to_pylist()[0]

    # 3) Reconstruct a typed StructScalar using the canonical schema
    #    (this validates field names/types and normalizes order)
    scalar = pa.scalar(record_dict, type=OME_ARROW_STRUCT)

    # Optional: soft validation via file-level metadata (if present)
    try:
        meta = table.schema.metadata or {}
        meta.get(b"ome.arrow.type", b"").decode() == str(
            OME_ARROW_TAG_TYPE
        ) and meta.get(b"ome.arrow.version", b"").decode() == str(OME_ARROW_TAG_VERSION)
        # You could log/print a warning if tag_ok is False, but don't fail.
    except Exception:
        pass

    return scalar
