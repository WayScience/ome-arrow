# ome-arrow

OME-Arrow uses OME specifications with Apache Arrow for fast, queryable, and language agnostic bioimage data.

Images are often referenced through databases as filepath links instead of the data itself.
OME-Arrow enables image data to be stored alongside metadata or derived data such as single-cell morphology features.
This means you can store and query data from the same location using any system which is compatible with Apache Arrow.

## Quick start

See below for a quick start guide.
Please also reference an example notebook: [Learning to fly with OME-Arrow](docs/src/examples/learning_to_fly_with_ome-arrow.ipynb).

```python
from ome_arrow import OMEArrow

# Ingest a tif image through a convenient OME-Arrow class
# We can also ingest OME-Zarr or NumPy arrays.
oa_image = OMEArrow(
    data="your_image.tif"
)

# Access the OME-Arrow struct itself
# (compatible with Arrow-compliant data storage).
oa_image.data

# Show information about the image.
oa_image.info()

# Display the image with matplotlib.
oa_image.view(how="matplotlib")

# Display the image with pyvista
# (great for ZYX 3D images).
oa_image.view(how="pyvista")

# Export to OME-Parquet.
# We can also export OME-TIFF, OME-Zarr or NumPy arrays.
oa_image.export(how="ome-parquet", out="your_image.ome.parquet")
```
