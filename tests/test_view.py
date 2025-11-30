"""
Tests for visualization helpers.
"""

import pytest

import ome_arrow.view as view


def test_view_pyvista_requires_optional_extras(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Ensure a clear warning/error is raised when PyVista deps are missing.
    """
    monkeypatch.setattr(view, "pv", None)

    with pytest.warns(RuntimeWarning, match=r"ome-arrow\[viz\]"):
        with pytest.raises(RuntimeError, match="PyVista-based visualization"):
            view.view_pyvista(data={})

