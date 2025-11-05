"""
Converting to and from OME-Arrow formats.
"""

from datetime import datetime, timezone
from typing import Optional, Sequence, List, Dict, Any

import pyarrow as pa
import numpy as np
import pyarrow as pa

from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pyarrow as pa
from bioio import BioImage
import bioio_tifffile
import bioio_ome_tiff


import re
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pyarrow as pa
from bioio import BioImage
import bioio_tifffile
import bioio_ome_tiff

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


def tiff_to_ome_arrow(
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


def to_numpy(
    data: Dict[str, Any] | pa.StructScalar,
    dtype: np.dtype = np.uint16,
    strict: bool = True,
    clamp: bool = False,
) -> np.ndarray:
    """
    Convert an OME-Arrow record into a NumPy array shaped (T,C,Z,Y,X).

    The OME-Arrow "planes" are flattened YX slices indexed by (z, t, c).
    This function reconstitutes them into a dense TCZYX ndarray.

    Args:
        data:
            OME-Arrow data as a Python dict or a `pa.StructScalar`.
        dtype:
            Output dtype (default: np.uint16). If different from plane
            values, a cast (and optional clamp) is applied.
        strict:
            When True, raise if a plane has wrong pixel length. When
            False, truncate/pad that plane to the expected length.
        clamp:
            If True, clamp values to the valid range of the target
            dtype before casting.

    Returns:
        np.ndarray: Dense array with shape (T, C, Z, Y, X).

    Raises:
        KeyError: If required OME-Arrow fields are missing.
        ValueError: If dimensions are invalid or planes are malformed.

    Examples:
        >>> arr = ome_arrow_to_tczyx(my_row)  # (T, C, Z, Y, X)
        >>> arr.shape
        (1, 2, 1, 512, 512)
    """
    # Unwrap Arrow scalar to plain Python dict if needed.
    if isinstance(data, pa.StructScalar):
        data = data.as_py()

    pm = data["pixels_meta"]
    sx, sy = int(pm["size_x"]), int(pm["size_y"])
    sz, sc, st = int(pm["size_z"]), int(pm["size_c"]), int(pm["size_t"])
    if sx <= 0 or sy <= 0 or sz <= 0 or sc <= 0 or st <= 0:
        raise ValueError("All size_* fields must be positive integers.")

    expected_len = sx * sy

    # Prepare target array (T,C,Z,Y,X), zero-filled by default.
    out = np.zeros((st, sc, sz, sy, sx), dtype=dtype)

    # Helper: cast (with optional clamp) to the output dtype.
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        lo, hi = info.min, info.max
    elif np.issubdtype(dtype, np.floating):
        lo, hi = -np.inf, np.inf
    else:
        # Rare dtypes: no clamping logic; rely on astype.
        lo, hi = -np.inf, np.inf

    def _cast_plane(a: np.ndarray) -> np.ndarray:
        if clamp:
            a = np.clip(a, lo, hi)
        return a.astype(dtype, copy=False)

    # Fill planes.
    for i, p in enumerate(data.get("planes", [])):
        z = int(p["z"])
        t = int(p["t"])
        c = int(p["c"])

        if not (0 <= z < sz and 0 <= t < st and 0 <= c < sc):
            raise ValueError(f"planes[{i}] index out of range: (z,t,c)=({z},{t},{c})")

        pix = p["pixels"]
        # Ensure sequence-like and correct length.
        try:
            n = len(pix)
        except Exception as e:
            raise ValueError(f"planes[{i}].pixels is not a sequence") from e

        if n != expected_len:
            if strict:
                raise ValueError(
                    f"planes[{i}].pixels length {n} != size_x*size_y {expected_len}"
                )
            # Lenient mode: fix length by truncation or zero-pad.
            if n > expected_len:
                pix = pix[:expected_len]
            else:
                pix = list(pix) + [0] * (expected_len - n)

        # Reshape to (Y,X) and cast.
        arr2d = np.asarray(pix).reshape(sy, sx)
        arr2d = _cast_plane(arr2d)
        out[t, c, z] = arr2d

    return out

def stack_from_pattern_path(
    pattern_path: str | Path,
    *,
    default_dim_for_unspecified: str = "C",
    map_series_to: Optional[str] = "T",
    clamp_to_uint16: bool = True,
    channel_names: Optional[List[str]] = None,
    image_id: Optional[str] = None,
    name: Optional[str] = None,
) -> pa.StructScalar:
    """
    Construct an OME-Arrow image stack directly from a file pattern path.
    This mimics the Bio-Formats 'pattern file' behavior, but without needing
    an external `.pattern` file.

    Example usage:
        stack_from_pattern_path("frames/<red,green,blue>.tiff")
        stack_from_pattern_path("images/test_Z<0-1>_C<0-2:2>.png")
        stack_from_pattern_path("images/test_.*\\.png")  # regex mode

    Parameters
    ----------
    pattern_path : str | Path
        A string containing both the folder and the filename pattern.
        Examples:
            "images/<red,green,blue>.tiff"
            "frames/test_Z<0-1>_C<0-2:2>.png"
            "data/experiment_.*\\.tif"
    default_dim_for_unspecified : str, optional
        The default dimension to assign for any placeholder with no preceding
        token (e.g., "<0-5>" → default is "C" = channels).
    map_series_to : str or None, optional
        If placeholders use the "series" token (S, sp, series),
        remap that dimension to another axis (default "T" for time).
        Set to None to disable series handling.
    clamp_to_uint16 : bool, optional
        Clamp and cast non-uint16 images to uint16 before embedding in Arrow.
    channel_names : list[str], optional
        Optional explicit names for channels. If not provided, inferred from
        literal placeholders (e.g. "<red,green,blue>") or defaulted to "C0..Cn".
    image_id : str, optional
        Optional OME image identifier.
    name : str, optional
        Human-friendly display name. Defaults to the pattern string itself.

    Returns
    -------
    pa.StructScalar
        An OME-Arrow-compatible StructScalar, ready for conversion, storage,
        or in-memory visualization.

    Notes
    -----
    - This function only supports single-plane image files (2D YX).
      Multi-page TIFFs will raise a ValueError.
    - Missing files are silently skipped and zero-filled to preserve shape.
    - The folder is inferred from the prefix of `pattern_path`.
    - Regex mode is used if the pattern contains no '<' or '>'.
    """

    # ------------------------------------------------------------
    # Separate folder and pattern components from the input path.
    # Example: "images/test_Z<0-1>_C<0-2:2>.png"
    #   folder = "images", line = "test_Z<0-1>_C<0-2:2>.png"
    # ------------------------------------------------------------
    path = Path(pattern_path)
    folder = path.parent
    line = path.name.strip()

    if not line:
        raise ValueError("Pattern path string is empty or malformed")

    # ------------------------------------------------------------
    # Dimension token map and range parsing regex
    # (matches Bio-Formats conventions)
    # ------------------------------------------------------------
    DIM_TOKENS = {
        "C": {"c", "ch", "w", "wavelength"},
        "T": {"t", "tl", "tp", "timepoint"},
        "Z": {"z", "zs", "sec", "fp", "focal", "focalplane"},
        "S": {"s", "sp", "series"},
    }
    NUM_RANGE_RE = re.compile(r"^(?P<a>\d+)\-(?P<b>\d+)(?::(?P<step>\d+))?$")

    # ------------------------------------------------------------
    # Helper: detect dimension type preceding a '<...>' placeholder.
    # ------------------------------------------------------------
    def detect_dim(before_text: str) -> Optional[str]:
        """Look back from '<' to detect a preceding token (e.g., '_Z<0-1>')"""
        m = re.search(r"([A-Za-z]+)$", before_text)
        if not m:
            return None
        token = m.group(1).lower()
        for dim, names in DIM_TOKENS.items():
            if token in names:
                return dim
        return None

    # ------------------------------------------------------------
    # Helper: expand a '<...>' block into a concrete list of values.
    # Examples:
    #   "<red,green,blue>" → ["red", "green", "blue"]
    #   "<00-03:2>" → ["00", "02"]
    # ------------------------------------------------------------
    def expand_raw_token(raw: str) -> Tuple[List[str], bool]:
        raw = raw.strip()
        # Case 1: explicit comma-separated list
        if "," in raw and not NUM_RANGE_RE.match(raw):
            parts = [p.strip() for p in raw.split(",")]
            return parts, all(p.isdigit() for p in parts)
        # Case 2: numeric range (with optional step)
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
        # Case 3: single literal (string or number)
        return [raw], raw.isdigit()

    # ------------------------------------------------------------
    # Helper: parse the string pattern and extract placeholders
    # Returns:
    #   - a template string, e.g., "test_Z{0}_C{1}.tif"
    #   - a list of placeholder dicts (choices, dim, etc.)
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Helper: regex fallback (no <...> present)
    # ------------------------------------------------------------
    def regex_match(folder: Path, regex: str) -> List[Path]:
        """Find all files in folder whose names fully match the regex."""
        r = re.compile(regex)
        return sorted([p for p in folder.iterdir() if p.is_file() and r.fullmatch(p.name)])

    # ------------------------------------------------------------
    # Step 1. Expand the pattern into a mapping (t, c, z) → Path
    # ------------------------------------------------------------
    matched: Dict[Tuple[int, int, int], Path] = {}
    literal_channel_names: Optional[List[str]] = None

    if "<" in line and ">" in line:
        # Bracket-pattern mode
        template, placeholders = parse_bracket_pattern(line)

        # Normalize missing dims to default (e.g., C if unspecified)
        for ph in placeholders:
            ph["dim"] = (ph["dim"] or "?").upper()
            if ph["dim"] == "?":
                ph["dim"] = default_dim_for_unspecified.upper()

        # Cartesian product across all placeholders → all possible filenames
        for combo in itertools.product(*[ph["choices"] for ph in placeholders]):
            fname = template.format(*combo)
            fpath = folder / fname
            if not fpath.exists():
                continue  # silently skip missing combinations (Bio-Formats behavior)

            # Initialize coordinates
            t = c = z = 0
            for ph, val in zip(placeholders, combo):
                idx = ph["choices"].index(val)
                dim = ph["dim"]
                # Convert 'S' (series) into another dimension, e.g., T
                if dim == "S":
                    if not map_series_to:
                        raise ValueError("Encountered 'series' but map_series_to=None")
                    dim = map_series_to.upper()
                if dim == "T": t = idx
                elif dim == "C": c = idx
                elif dim == "Z": z = idx

            # Capture literal channel names once (e.g., <red,green,blue>)
            if literal_channel_names is None:
                for ph in placeholders:
                    dim_eff = ph["dim"] if ph["dim"] != "S" else (map_series_to or "S")
                    if dim_eff == "C" and not ph["is_numeric"]:
                        literal_channel_names = ph["choices"]
                        break

            matched[(t, c, z)] = fpath

    else:
        # Regex mode — treat entire string as a filename regex
        for z, p in enumerate(regex_match(folder, line)):
            matched[(0, 0, z)] = p

    if not matched:
        raise FileNotFoundError(f"No files matched pattern: {pattern_path}")

    # ------------------------------------------------------------
    # Step 2. Infer stack dimensions (T, C, Z)
    # ------------------------------------------------------------
    size_t = max(k[0] for k in matched) + 1
    size_c = max(k[1] for k in matched) + 1
    size_z = max(k[2] for k in matched) + 1

    # Resolve channel naming
    if channel_names and len(channel_names) != size_c:
        raise ValueError(f"channel_names length {len(channel_names)} != size_c {size_c}")
    if not channel_names:
        channel_names = literal_channel_names or [f"C{i}" for i in range(size_c)]

    # ------------------------------------------------------------
    # Step 3. Probe first image to determine size and physical scale
    # ------------------------------------------------------------
    sample = next(iter(matched.values()))
    is_ome = sample.suffix.lower() in (".ome.tif", ".ome.tiff")
    img0 = BioImage(image=str(sample), reader=(bioio_ome_tiff.Reader if is_ome else bioio_tifffile.Reader))
    a0 = np.asarray(img0.data)
    if a0.ndim != 2:
        raise ValueError(f"{sample.name} is not single-plane (YX); got {a0.shape}")
    size_y, size_x = a0.shape

    pps = getattr(img0, "physical_pixel_sizes", None)
    try:
        psize_x = float(getattr(pps, "X", None) or 1.0)
        psize_y = float(getattr(pps, "Y", None) or 1.0)
        psize_z = float(getattr(pps, "Z", None) or 1.0)
    except Exception:
        # Missing or malformed physical scale metadata
        psize_x = psize_y = psize_z = 1.0

    # ------------------------------------------------------------
    # Step 4. Load each plane into the Arrow struct representation
    # ------------------------------------------------------------
    planes: List[Dict[str, Any]] = []
    for t in range(size_t):
        for c in range(size_c):
            for z in range(size_z):
                path = matched.get((t, c, z))
                if not path:
                    # If a plane is missing, fill with zeros
                    plane2d = np.zeros((size_y, size_x), dtype=np.uint16)
                else:
                    reader = bioio_ome_tiff.Reader if path.suffix.lower() in (".ome.tif", ".ome.tiff") else bioio_tifffile.Reader
                    im = BioImage(image=str(path), reader=reader)
                    a = np.asarray(im.data)
                    if a.ndim != 2:
                        raise ValueError(f"{path.name} is not single-plane; got {a.shape}")
                    if a.shape != (size_y, size_x):
                        raise ValueError(f"Shape mismatch for {path.name}: {a.shape} vs {(size_y, size_x)}")
                    if clamp_to_uint16 and a.dtype != np.uint16:
                        # Safely cast to uint16, clamping to valid range
                        a = np.clip(a, 0, 65535).astype(np.uint16)
                    plane2d = a
                planes.append({"z": z, "t": t, "c": c, "pixels": plane2d.ravel().tolist()})

    # ------------------------------------------------------------
    # Step 5. Construct OME-Arrow metadata (channels, pixels, etc.)
    # ------------------------------------------------------------
    channels_meta = [
        {
            "id": f"ch-{i}",
            "name": str(channel_names[i]),
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

    # ------------------------------------------------------------
    # Step 6. Delegate to to_ome_arrow() to build StructScalar
    # ------------------------------------------------------------
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
