"""
Tests for the main module.
"""

from ome_arrow.meta import OME_ARROW_STRUCT
from ome_arrow.utils import verify_ome_arrow


def test_verify_ome_arrow_valid(example_correct_data: dict):
    """
    Test that a valid OME-Arrow structure is verified correctly.
    """

    assert verify_ome_arrow(1, OME_ARROW_STRUCT) is False

    assert verify_ome_arrow(example_correct_data, OME_ARROW_STRUCT) is True
