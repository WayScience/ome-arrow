"""
Tests for the core module
"""

import pathlib

import matplotlib
import numpy as np
import pytest

from ome_arrow.core import OMEArrow


@pytest.mark.parametrize(
    "input_data, expected_info",
    [
        (
            "tests/data/ome-artificial-5d-datasets/z-series.ome.tiff",
            {
                "channels": 1,
                "is_multichannel": False,
                "shape": (
                    1,
                    1,
                    5,
                    167,
                    439,
                ),
                "summary": "3D image (z-stack), single-channel - shape (T=1, C=1, Z=5, Y=167, X=439)",
                "type": "3D image (z-stack)",
            },
        ),
        (
            "tests/data/ome-artificial-5d-datasets/time-series.ome.tif",
            {
                "channels": 1,
                "is_multichannel": False,
                "shape": (
                    7,
                    1,
                    1,
                    167,
                    439,
                ),
                "summary": "movie / timelapse, single-channel - shape (T=7, C=1, Z=1, Y=167, X=439)",
                "type": "movie / timelapse",
            },
        ),
        (
            "tests/data/ome-artificial-5d-datasets/single-channel.ome.tiff",
            {
                "channels": 1,
                "is_multichannel": False,
                "shape": (
                    1,
                    1,
                    1,
                    167,
                    439,
                ),
                "summary": "2D image, single-channel - shape (T=1, C=1, Z=1, Y=167, X=439)",
                "type": "2D image",
            },
        ),
        (
            "tests/data/ome-artificial-5d-datasets/multi-channel.ome.tiff",
            {
                "channels": 3,
                "is_multichannel": True,
                "shape": (
                    1,
                    3,
                    1,
                    167,
                    439,
                ),
                "summary": "2D image, multi-channel (3 channels) - shape (T=1, C=3, Z=1, Y=167, "
                "X=439)",
                "type": "2D image",
            },
        ),
        (
            "tests/data/ome-artificial-5d-datasets/multi-channel-z-series.ome.tiff",
            {
                "channels": 3,
                "is_multichannel": True,
                "shape": (
                    1,
                    3,
                    5,
                    167,
                    439,
                ),
                "summary": "3D image (z-stack), multi-channel (3 channels) - shape (T=1, C=3, Z=5, "
                "Y=167, X=439)",
                "type": "3D image (z-stack)",
            },
        ),
        (
            "tests/data/ome-artificial-5d-datasets/multi-channel-time-series.ome.tiff",
            {
                "channels": 3,
                "is_multichannel": True,
                "shape": (
                    7,
                    3,
                    1,
                    167,
                    439,
                ),
                "summary": "movie / timelapse, multi-channel (3 channels) - shape (T=7, C=3, Z=1, "
                "Y=167, X=439)",
                "type": "movie / timelapse",
            },
        ),
        (
            "tests/data/ome-artificial-5d-datasets/multi-channel-4D-series.ome.tiff",
            {
                "channels": 3,
                "is_multichannel": True,
                "shape": (
                    7,
                    3,
                    5,
                    167,
                    439,
                ),
                "summary": "4D timelapse-volume, multi-channel (3 channels) - shape (T=7, C=3, Z=5, "
                "Y=167, X=439)",
                "type": "4D timelapse-volume",
            },
        ),
        (
            "tests/data/ome-artificial-5d-datasets/4D-series.ome.tiff",
            {
                "channels": 1,
                "is_multichannel": False,
                "shape": (
                    7,
                    1,
                    5,
                    167,
                    439,
                ),
                "summary": "4D timelapse-volume, single-channel - shape (T=7, C=1, Z=5, Y=167, X=439)",
                "type": "4D timelapse-volume",
            },
        ),
        (
            "tests/data/nviz-artificial-4d-dataset/E99_C<111,222>_ZS<000-021>.tif",
            {
                "channels": 2,
                "is_multichannel": True,
                "shape": (
                    1,
                    2,
                    22,
                    128,
                    128,
                ),
                "summary": "3D image (z-stack), multi-channel (2 channels) - shape (T=1, C=2, Z=22, "
                "Y=128, X=128)",
                "type": "3D image (z-stack)",
            },
        ),
        (
            "tests/data/nviz-artificial-4d-dataset/E99_C111_ZS<000-021>.tif",
            {
                "channels": 1,
                "is_multichannel": False,
                "shape": (
                    1,
                    1,
                    22,
                    128,
                    128,
                ),
                "summary": "3D image (z-stack), single-channel - shape (T=1, C=1, Z=22, Y=128, X=128)",
                "type": "3D image (z-stack)",
            },
        ),
        (
            "tests/data/nviz-artificial-4d-dataset/E99_C<111,222>_ZS000.tif",
            {
                "channels": 2,
                "is_multichannel": True,
                "shape": (
                    1,
                    2,
                    1,
                    128,
                    128,
                ),
                "summary": "2D image, multi-channel (2 channels) - shape (T=1, C=2, Z=1, Y=128, "
                "X=128)",
                "type": "2D image",
            },
        ),
        (
            "tests/data/examplehuman/AS_09125_050116030001_D03f00d2.tif",
            {
                "channels": 1,
                "is_multichannel": False,
                "shape": (
                    1,
                    1,
                    1,
                    512,
                    512,
                ),
                "summary": "2D image, single-channel - shape (T=1, C=1, Z=1, Y=512, X=512)",
                "type": "2D image",
            },
        ),
        (
            "tests/data/examplehuman/AS_09125_050116030001_D03f00d1.tif",
            {
                "channels": 1,
                "is_multichannel": False,
                "shape": (
                    1,
                    1,
                    1,
                    512,
                    512,
                ),
                "summary": "2D image, single-channel - shape (T=1, C=1, Z=1, Y=512, X=512)",
                "type": "2D image",
            },
        ),
        (
            "tests/data/examplehuman/AS_09125_050116030001_D03f00d0.tif",
            {
                "channels": 1,
                "is_multichannel": False,
                "shape": (
                    1,
                    1,
                    1,
                    512,
                    512,
                ),
                "summary": "2D image, single-channel - shape (T=1, C=1, Z=1, Y=512, X=512)",
                "type": "2D image",
            },
        ),
        (
            "tests/data/JUMP-BR00117006/BR00117006.ome.parquet",
            {
                "channels": 1,
                "is_multichannel": False,
                "shape": (
                    1,
                    1,
                    1,
                    72,
                    84,
                ),
                "summary": "2D image, single-channel - shape (T=1, C=1, Z=1, Y=72, X=84)",
                "type": "2D image",
            },
        ),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:As of version 0.4.0, the parser argument is ignored.*:DeprecationWarning"
)
@pytest.mark.filterwarnings(
    "ignore:OME-Arrow column schema differs from OME_ARROW_STRUCT.*:UserWarning"
)
def test_ome_arrow_base_expectations(
    input_data: str, expected_info: dict, tmp_path: pathlib.Path
):
    """
    Test that OMEArrow initializes correctly with valid data.
    """

    if input_data.endswith(".ome.parquet"):
        with pytest.warns(UserWarning, match="Requested column 'ome_arrow'"):
            oa_image = OMEArrow(data=input_data)
    else:
        oa_image = OMEArrow(data=input_data)

    assert oa_image.info() == expected_info

    # test visualization
    assert isinstance(
        oa_image.view(how="matplotlib", show=False)[0], matplotlib.figure.Figure
    )

    pv = pytest.importorskip(
        "pyvista",
        reason="PyVista visualization stack is optional (install extras: viz)",
    )
    assert isinstance(oa_image.view(how="pyvista", show=False), pv.Plotter)

    # test info description consistency across data inputs
    assert OMEArrow(data=oa_image.data).info() == expected_info

    # test conversions to other formats retain info
    assert OMEArrow(data=oa_image.export(how="numpy")).info() == expected_info

    assert (
        OMEArrow(
            data=oa_image.export(how="ometiff", out=f"{tmp_path}/example.ome.tiff")
        ).info()
        == expected_info
    )

    assert (
        OMEArrow(
            data=oa_image.export(how="omezarr", out=f"{tmp_path}/example.ome.zarr")
        ).info()
        == expected_info
    )

    assert (
        OMEArrow(
            data=oa_image.export(
                how="omeparquet", out=f"{tmp_path}/example.ome.parquet"
            )
        ).info()
        == expected_info
    )


@pytest.mark.parametrize(
    "input_data, column_name, row_index, expected_info",
    [
        (
            "tests/data/JUMP-BR00117006/BR00117006.ome.parquet",
            "Image_FileName_OrigDNA_OMEArrow_LABL",
            2,
            {
                "channels": 1,
                "is_multichannel": False,
                "shape": (
                    1,
                    1,
                    1,
                    73,
                    97,
                ),
                "summary": "2D image, single-channel - shape (T=1, C=1, Z=1, Y=73, X=97)",
                "type": "2D image",
            },
        ),
    ],
)
def test_ome_parquet_specific_col_and_row(
    input_data: str,
    column_name: str,
    row_index: int,
    expected_info: dict,
    tmp_path: pathlib.Path,
):
    """
    Test that OMEArrow initializes correctly with valid data.
    """

    with pytest.warns(UserWarning, match="schema differs from OME_ARROW_STRUCT"):
        oa_image = OMEArrow(
            data=input_data, column_name=column_name, row_index=row_index
        )

    assert oa_image.info() == expected_info


def test_vortex_roundtrip(tmp_path: pathlib.Path) -> None:
    """Smoke-test the Vortex round-trip export/import path."""
    pytest.importorskip(
        "vortex", reason="Vortex support is optional (install extras: vortex)."
    )

    arr = np.arange(16, dtype=np.uint16).reshape(1, 1, 1, 4, 4)
    oa = OMEArrow(arr)
    out = tmp_path / "example.vortex"

    oa.export(how="vortex", out=str(out))
    reloaded = OMEArrow(str(out))

    assert reloaded.info() == oa.info()


def test_parquet_roundtrip_preserves_image_type(tmp_path: pathlib.Path) -> None:
    """Ensure image_type round-trips through OME-Parquet."""
    arr = np.arange(16, dtype=np.uint16).reshape(1, 1, 1, 4, 4)
    oa = OMEArrow(arr, image_type="label")
    out = tmp_path / "example.ome.parquet"

    oa.export(how="omeparquet", out=str(out))
    reloaded = OMEArrow(str(out))

    assert reloaded.data.as_py()["image_type"] == "label"


def test_vortex_custom_column_name(tmp_path: pathlib.Path) -> None:
    """Ensure custom Vortex column names are preserved on round-trip."""
    pytest.importorskip(
        "vortex", reason="Vortex support is optional (install extras: vortex)."
    )

    arr = np.arange(12, dtype=np.uint16).reshape(1, 1, 1, 3, 4)
    oa = OMEArrow(arr)
    out = tmp_path / "custom_column.vortex"

    oa.export(how="vortex", out=str(out), vortex_column_name="custom_ome_arrow")
    reloaded = OMEArrow(str(out), column_name="custom_ome_arrow")

    assert reloaded.info() == oa.info()


def test_scan_collect_roundtrip() -> None:
    """Materialize a lazily scanned parquet source via collect()."""
    oa = OMEArrow.scan("tests/data/JUMP-BR00117006/BR00117006.ome.parquet")
    assert oa.is_lazy

    with pytest.warns(UserWarning, match="Requested column 'ome_arrow'"):
        oa.collect()
    assert not oa.is_lazy
    assert oa.info()["shape"] == (1, 1, 1, 72, 84)


def test_slice_lazy_scan_collect() -> None:
    """Queue a lazy slice and materialize it via collect()."""
    oa = OMEArrow.scan("tests/data/JUMP-BR00117006/BR00117006.ome.parquet")
    sliced = oa.slice_lazy(0, 10, 0, 8)

    assert sliced.is_lazy
    with pytest.warns(UserWarning, match="Requested column 'ome_arrow'"):
        sliced.collect()
    assert sliced.info()["shape"] == (1, 1, 1, 8, 10)


def test_slice_lazy_chain_scan_collect() -> None:
    """Allow chaining lazy slices before materialization."""
    oa = OMEArrow.scan("tests/data/JUMP-BR00117006/BR00117006.ome.parquet")
    sliced = oa.slice_lazy(0, 20, 0, 20).slice_lazy(5, 15, 2, 12)

    with pytest.warns(UserWarning, match="Requested column 'ome_arrow'"):
        shape = sliced.collect().info()["shape"]
    assert shape == (1, 1, 1, 10, 10)


def test_slice_lazy_on_materialized_falls_back_to_eager() -> None:
    """Use eager slice behavior when source is already materialized."""
    arr = np.arange(1 * 1 * 1 * 6 * 7, dtype=np.uint16).reshape(1, 1, 1, 6, 7)
    oa = OMEArrow(arr)
    out = oa.slice_lazy(1, 5, 1, 4)

    assert not out.is_lazy
    assert out.info()["shape"] == (1, 1, 1, 3, 4)
