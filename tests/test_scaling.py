"""
Scaling metadata tests for OME-TIFF and OME-Zarr.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from bioio import BioImage
from bioio_ome_tiff import Reader as OMETiffReader
from bioio_tifffile import Reader as TiffReader

from ome_arrow.core import OMEArrow


def _read_ngff_scale(zarr_group: Path) -> tuple[float, float, float, str | None]:
    # Read NGFF metadata from the group-level attributes.
    attrs = json.loads((zarr_group / "zarr.json").read_text()).get("attributes", {})
    multiscales = attrs.get("ome", {}).get("multiscales", [])
    assert multiscales, "Expected NGFF multiscales metadata in zarr.json"
    ms = multiscales[0]
    axes = ms.get("axes", [])
    datasets = ms.get("datasets", [])
    assert axes and datasets, "Expected axes/datasets metadata in NGFF multiscales"

    # Find the scale transform for the base (path="0") dataset.
    ds = next((d for d in datasets if str(d.get("path")) == "0"), datasets[0])
    cts = ds.get("coordinateTransformations", [])
    scale_ct = next((ct for ct in cts if ct.get("type") == "scale"), None)
    assert scale_ct, "Expected scale coordinateTransformations in NGFF metadata"

    scale = scale_ct.get("scale", [])
    assert len(scale) == len(axes), "Scale length must match axes length"

    # Map axis names to scale values and units, only for spatial axes.
    axis_scale = {}
    axis_unit = {}
    for i, ax in enumerate(axes):
        name = str(ax.get("name", "")).lower()
        if name in {"x", "y", "z"}:
            axis_scale[name] = float(scale[i])
            unit = ax.get("unit")
            if unit:
                axis_unit[name] = str(unit)

    # Default to 1.0 when a spatial axis is missing in metadata.
    psize_x = axis_scale.get("x", 1.0)
    psize_y = axis_scale.get("y", 1.0)
    psize_z = axis_scale.get("z", 1.0)
    # Use the unit only if all provided spatial units agree.
    units = [axis_unit.get(a) for a in ("x", "y", "z") if axis_unit.get(a)]
    unit = units[0] if units and len(set(units)) == 1 else None
    return psize_x, psize_y, psize_z, unit


def test_ome_zarr_scale_from_metadata() -> None:
    """
    Ensure NGFF scale metadata is mapped into OME-Arrow physical sizes.
    """
    zarr_path = Path("tests/data/idr0062A/6001240_labels.zarr/labels/0")
    expected_x, expected_y, expected_z, unit = _read_ngff_scale(zarr_path)

    obj = OMEArrow(str(zarr_path))
    pm = obj.data.as_py()["pixels_meta"]

    assert pm["physical_size_x"] == pytest.approx(expected_x)
    assert pm["physical_size_y"] == pytest.approx(expected_y)
    assert pm["physical_size_z"] == pytest.approx(expected_z)
    if unit is not None:
        assert pm["physical_size_x_unit"] == "µm"
        assert pm["physical_size_y_unit"] == "µm"
        assert pm["physical_size_z_unit"] == "µm"


def test_ome_tiff_scale_from_metadata() -> None:
    """
    Ensure OME-TIFF physical pixel sizes map to OME-Arrow physical sizes.
    """
    tiff_path = Path("tests/data/examplehuman/AS_09125_050116030001_D03f00d2.tif")
    reader = (
        OMETiffReader
        if tiff_path.suffix.lower() in {".ome.tif", ".ome.tiff"}
        else TiffReader
    )
    img = BioImage(image=str(tiff_path), reader=reader)
    pps = getattr(img, "physical_pixel_sizes", None)
    assert pps is not None

    expected_x = float(getattr(pps, "X", None) or 1.0)
    expected_y = float(getattr(pps, "Y", None) or 1.0)
    expected_z = float(getattr(pps, "Z", None) or 1.0)

    obj = OMEArrow(str(tiff_path))
    pm = obj.data.as_py()["pixels_meta"]

    assert pm["physical_size_x"] == pytest.approx(expected_x)
    assert pm["physical_size_y"] == pytest.approx(expected_y)
    assert pm["physical_size_z"] == pytest.approx(expected_z)
