"""
Utility functions for ome-arrow.
"""

from typing import Any
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