"""
Converting to and from OME-Arrow formats.
"""

from datetime import datetime
from typing import Optional, Sequence, List, Dict, Any

import pyarrow as pa

from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pyarrow as pa
from bioio import BioImage

from ome_arrow.meta import OME_ARROW_STRUCT


def to_ome_arrow(
    type_: str = "ome.arrow",
    version: str = "1.0.0",
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
    """Create a typed OME-Arrow StructScalar with sensible defaults.

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
    if planes is None:
        planes = [
            {"z": 0, "t": 0, "c": 0, "pixels": [0] * (size_x * size_y)}
        ]

    record = {
        "type": type_,
        "version": version,
        "id": image_id,
        "name": name,
        "acquisition_datetime": acquisition_datetime or datetime.utcnow(),
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

    # Validate against struct (raises if mismatched)
    return pa.scalar(record, type=OME_ARROW_STRUCT)


def tiff_to_ome_arrow(
    tiff_path: str | Path,
    image_id: Optional[str] = None,
    name: Optional[str] = None,
    channel_names: Optional[Sequence[str]] = None,
    acquisition_datetime: Optional[datetime] = None,
    clamp_to_uint16: bool = True,
) -> pa.StructScalar:
    """Read a TIFF via bioio and return a typed OME-Arrow StructScalar.

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
    img = BioImage(str(p))

    # BioIO data is TCZYX (broadcasting missing dims to length 1).
    arr = np.asarray(img.data)            # shape: (T, C, Z, Y, X)
    dims = img.dims
    size_t = int(dims.T or 1)
    size_c = int(dims.C or 1)
    size_z = int(dims.Z or 1)
    size_y = int(dims.Y or arr.shape[-2])
    size_x = int(dims.X or arr.shape[-1])
    if size_x <= 0 or size_y <= 0:
        raise ValueError("Image must have positive Y and X dims.")

    # Physical pixel sizes (µm); default to 1.0 if unavailable.
    pps = getattr(img, "physical_pixel_sizes", None)
    try:
        psize_x = float(getattr(pps, "X", None) or 1.0)
        psize_y = float(getattr(pps, "Y", None) or 1.0)
        psize_z = float(getattr(pps, "Z", None) or 1.0)
    except Exception:
        psize_x = psize_y = psize_z = 1.0

    # Channels block.
    if not channel_names or len(channel_names) != size_c:
        channel_names = [f"C{i}" for i in range(size_c)]
    channels = [{
        "id": f"ch-{i}", "name": channel_names[i],
        "emission_um": 0.0, "excitation_um": 0.0,
        "illumination": "Unknown", "color_rgba": 0xFFFFFFFF
    } for i in range(size_c)]

    # Planes: flattened YX per (t, c, z).
    planes: List[Dict[str, Any]] = []
    for t in range(size_t):
        for c in range(size_c):
            for z in range(size_z):
                plane = arr[t, c, z]      # (Y, X)
                if clamp_to_uint16 and plane.dtype != np.uint16:
                    plane = np.clip(plane, 0, 65535).astype(np.uint16)
                planes.append({
                    "z": z, "t": t, "c": c,
                    "pixels": plane.ravel().tolist()
                })

    # Dimension order: no Z implies XYCT.
    dim_order = "XYCT" if size_z == 1 else "XYZCT"

    # Delegate final struct creation + type validation.
    return to_ome_arrow(
        type_="ome.arrow",
        version="1.0.0",
        image_id=image_id or p.stem,
        name=name or p.name,
        acquisition_datetime=acquisition_datetime or datetime.utcnow(),
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