# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # OME-Arrow quiver
#

# +
from ome_arrow import OMEArrow

OMEArrow(
    data="../../../tests/data/nviz-artificial-4d-dataset/E99_C<111-222>_ZS<000-047>.tif"
).export(how="ome-tiff", out="example.ome.tiff")
# -

OMEArrow(data="example.ome.tiff")

OMEArrow(data="../../../tests/data/nviz-artificial-4d-dataset/E99_C111_ZS040.tif")

oa_image = OMEArrow(
    data="../../../tests/data/examplehuman/AS_09125_050116030001_D03f00d0.tif"
)
oa_image

oa_image.info()

oa_image.view(how="matplotlib")

OMEArrow(data="../../../tests/data/ome-artificial-5d-datasets/z-series.ome.tiff").view(
    how="pyvista"
)

oa_image.export(how="numpy")


