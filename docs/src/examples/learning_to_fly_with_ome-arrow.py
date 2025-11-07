# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Learning to fly with OME-Arrow
#

from ome_arrow import OMEArrow

oa_image = OMEArrow(
    data="../../../tests/data/examplehuman/AS_09125_050116030001_D03f00d0.tif"
)
oa_image

oa_image.info()

oa_image.export(how="numpy")

stack = OMEArrow(
    data="../../../tests/data/nviz-artificial-4d-dataset/E99_C<111,222>_ZS<000-047>.tif",
    tcz=(0, 0, 20),
)
stack

stack_np = stack.export(how="numpy")
OMEArrow(data=stack_np, tcz=(0, 0, 20))

stack.export(how="ome-tiff", out="example.ome.tiff")
OMEArrow(data="example.ome.tiff", tcz=(0, 0, 20))

stack.export(how="ome-zarr", out="example.ome.zarr")
OMEArrow(data="example.ome.zarr", tcz=(0, 0, 20))

stack.export(how="ome-parquet", out="example.ome.parquet")
OMEArrow(data="example.ome.parquet", tcz=(0, 0, 20))

stack.view(how="pyvista")

stack.slice(
    x_min=40,
    y_min=80,
    x_max=70,
    y_max=110,
    t_indices=[0],
    c_indices=[0],
    z_indices=[20],
)


