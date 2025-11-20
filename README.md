<img height="200" src="https://raw.githubusercontent.com/wayscience/ome-arrow/main/docs/src/_static/logo.png?raw=true">

![PyPI - Version](https://img.shields.io/pypi/v/ome-arrow)
[![Build Status](https://github.com/wayscience/ome-arrow/actions/workflows/run-tests.yml/badge.svg?branch=main)](https://github.com/wayscience/ome-arrow/actions/workflows/run-tests.yml?query=branch%3Amain)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

# Open, interoperable, and queryable microscopy images with OME Arrow

OME-Arrow uses [Open Microscopy Environment (OME)](https://github.com/ome) specifications through [Apache Arrow](https://arrow.apache.org/) for fast, queryable, and language agnostic bioimage data.

<img height="200" src="https://raw.githubusercontent.com/wayscience/ome-arrow/main/docs/src/_static/references_to_files.png">

__Images are often left behind from the data model, referenced but excluded from databases.__

<img height="200" src="https://raw.githubusercontent.com/wayscience/ome-arrow/main/docs/src/_static/various_ome_arrow_schema.png">

__OME-Arrow brings images back into the story.__

OME Arrow enables image data to be stored alongside metadata or derived data such as single-cell morphology features.
Images in OME Arrow are composed of mutlilayer struct values so they may be stored as values within tables.
This means you can store, query, and build relationships on data from the same location using any system which is compatible with Apache Arrow (including Parquet) through common data interfaces (such as SQL and DuckDB).

## Installation

Install OME Arrow from PyPI or from source:

```sh
# install from pypi
pip install ome-arrow

# install directly from source
pip install git+https://github.com/wayscience/ome-arrow.git
```

## Quick start

See below for a quick start guide.
Please also reference an example notebook: [Learning to fly with OME-Arrow](https://github.com/wayscience/ome-arrow/tree/main/docs/src/examples/learning_to_fly_with_ome-arrow.ipynb).

```python
from ome_arrow import OMEArrow

# Ingest a tif image through a convenient OME Arrow class
# We can also ingest OME-Zarr or NumPy arrays.
oa_image = OMEArrow(
    data="your_image.tif"
)

# Access the OME Arrow struct itself
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

## Contributing, Development, and Testing

Please see our [contributing documentation](https://github.com/wayscience/ome-arrow/tree/main/CONTRIBUTING.md) for more details on contributions, development, and testing.

## Related projects

OME Arrow is used or inspired by the following projects, check them out!

- [`napari-ome-arrow`](https://github.com/WayScience/napari-ome-arrow): enables you to view OME Arrow and related images.
- [`nViz`](https://github.com/WayScience/nViz): focuses on ingesting and visualizing various 3D image data.
- [`CytoDataFrame`](https://github.com/cytomining/CytoDataFrame): provides a DataFrame-like experience for viewing feature and microscopy image data within Jupyter notebook interfaces.
- [`coSMicQC`](https://github.com/cytomining/coSMicQC): performs quality control on microscopy feature datasets, visualized using CytoDataFrames.
