"""
Utility functions for ome-arrow.
"""

from typing import Any, Dict
import pyarrow as pa

def verify_ome_arrow(data: Any, struct: pa.StructType) -> bool:
    """Return True if `data` conforms to the given Arrow StructType.

    This tries to convert `data` into a pyarrow scalar using `struct`
    as the declared type. If conversion fails, the data does not match.

    Args:
        data: A nested Python dict/list structure to test.
        struct: The expected pyarrow.StructType schema.

    Returns:
        bool: True if conversion succeeds, False otherwise.
    """
    try:
        pa.scalar(data, type=struct)
        return True
    except (TypeError, pa.ArrowInvalid, pa.ArrowTypeError):
        return False
    
def describe_ome_arrow(data: pa.StructScalar | dict) -> Dict[str, Any]:
    """
    Describe the structure of an OME-Arrow image record.

    Reads `pixels_meta` from the OME-Arrow struct to report TCZYX
    dimensions and classify whether it's a 2D image, 3D z-stack,
    movie/timelapse, or 4D timelapse-volume. Also flags whether it is
    multi-channel (C > 1) or single-channel.

    Args:
        data: OME-Arrow row as a pa.StructScalar or plain dict.

    Returns:
        dict with keys:
            - shape: (T, C, Z, Y, X)
            - type: classification string
            - summary: human-readable text
    """
    # --- Unwrap StructScalar if needed ---
    if isinstance(data, pa.StructScalar):
        data = data.as_py()

    pm = data.get("pixels_meta", {})
    t = int(pm.get("size_t", 1))
    c = int(pm.get("size_c", 1))
    z = int(pm.get("size_z", 1))
    y = int(pm.get("size_y", 1))
    x = int(pm.get("size_x", 1))

    # --- Basic dimensional classification ---
    if t == 1 and z == 1:
        kind = "2D image"
    elif t == 1 and z > 1:
        kind = "3D image (z-stack)"
    elif t > 1 and z == 1:
        kind = "movie / timelapse"
    elif t > 1 and z > 1:
        kind = "4D timelapse-volume"
    else:
        kind = "unknown"

    # --- Channel classification ---
    if c > 1:
        channel_info = f"multi-channel ({c} channels)"
    else:
        channel_info = "single-channel"

    # --- Summary ---
    summary = f"{kind}, {channel_info} - shape (T={t}, C={c}, Z={z}, Y={y}, X={x})"

    return {
        "shape": (t, c, z, y, x),
        "type": kind,
        "channels": c,
        "is_multichannel": c > 1,
        "summary": summary,
    }