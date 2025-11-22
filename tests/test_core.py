"""
Tests for the core module
"""

import pathlib

import matplotlib
import pytest
import pyvista as pv

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
def test_ome_arrow_base_expectations(
    input_data: str, expected_info: dict, tmp_path: pathlib.Path
):
    """
    Test that OMEArrow initializes correctly with valid data.
    """

    oa_image = OMEArrow(data=input_data)

    assert oa_image.info() == expected_info

    # test visualization
    assert isinstance(
        oa_image.view(how="matplotlib", show=False)[0], matplotlib.figure.Figure
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

    oa_image = OMEArrow(data=input_data, column_name=column_name, row_index=row_index)

    assert oa_image.info() == expected_info
