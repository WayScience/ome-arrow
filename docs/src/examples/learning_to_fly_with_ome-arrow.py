# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: ome-arrow
#     language: python
#     name: python3
# ---

# # Learning to fly with OME-Arrow
#
# This notebook provides a quick demonstration of what you can do with OME Arrow.

# we import a single class, OMEArrow
# which handles all data I/O and manipulation
from ome_arrow import OMEArrow

# read a TIFF file and convert it to OME-Arrow
oa_image = OMEArrow(
    data="../../../tests/data/examplehuman/AS_09125_050116030001_D03f00d0.tif"
)
# by default, the image and metadata are shown
oa_image

# we can also get a summary of the OME-Arrow object
oa_image.info()

# we can export the data into a number
# of different formats, e.g. numpy
oa_image.export(how="numpy")

# We can also read in TIFF stacks following OME bfconvert API conventions
stack = OMEArrow(
    data="../../../tests/data/nviz-artificial-4d-dataset/E99_C<111,222>_ZS<000-021>.tif",
    # this is an optional for which
    # timepoint, channel, and z-slice to show by default
    tcz=(0, 0, 20),
)
stack

# we can visualize the stack using pyvista for 3D rendering
stack.view(how="pyvista")

# here we demonstrate that the data can be exported again
# into numpy format and re-imported
# into a new OME-Arrow object (from numpy data).
stack_np = stack.export(how="numpy")
OMEArrow(data=stack_np, tcz=(0, 0, 20))

# here we demonstrate that the data can be exported again
# into OME-TIFF format and re-imported
# into a new OME-Arrow object (from OME-TIFF data).
stack.export(how="ome-tiff", out="example.ome.tiff")
OMEArrow(data="example.ome.tiff", tcz=(0, 0, 20))

# here we demonstrate that the data can be exported again
# into OME-ZARR format and re-imported
# into a new OME-Arrow object (from OME-ZARR data).
stack.export(how="ome-zarr", out="example.ome.zarr")
OMEArrow(data="example.ome.zarr", tcz=(0, 0, 20))

# here we demonstrate that the data can be exported again
# into OME-Parquet format and re-imported
# into a new OME-Arrow object (from OME-Parquet data).
stack.export(how="ome-parquet", out="example.ome.parquet")
OMEArrow(data="example.ome.parquet", tcz=(0, 0, 20))

# we can also slice the data to get a smaller region of interest
stack.slice(
    x_min=40,
    y_min=80,
    x_max=70,
    y_max=110,
    t_indices=[0],
    c_indices=[0],
    z_indices=[20],
)

# read from a multi-image OME Parquet file as OME-Arrow
oa_image = OMEArrow(
    data="../../../tests/data/JUMP-BR00117006/BR00117006.ome.parquet",
    # we can specify which column and row to read
    # (or rely on OMEArrow to find a suitable default)
    column_name="Image_FileName_OrigDNA_OMEArrow_LABL",
    row_index=2,
)
# by default, the image and metadata are shown
oa_image
