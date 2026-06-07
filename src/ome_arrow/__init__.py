"""
Init file for ome_arrow package.
"""

from ome_arrow._version import version as ome_arrow_version
from ome_arrow.core import OMEArrow
from ome_arrow.dataset import (
    AccessPattern,
    ChunkChoice,
    Layout,
    OMEArrowDataset,
    OMEArrowPixels,
    choose_chunking,
    write_ome_arrow_dataset,
)
from ome_arrow.export import (
    to_numpy,
    to_ome_parquet,
    to_ome_tiff,
    to_ome_vortex,
    to_ome_zarr,
)
from ome_arrow.ingest import (
    from_jax_array,
    from_numpy,
    from_ome_parquet,
    from_ome_vortex,
    from_ome_zarr,
    from_tiff,
    from_torch_array,
    to_ome_arrow,
)
from ome_arrow.meta import (
    OME_ARROW_BYTE_STRUCT,
    OME_ARROW_STRUCT,
    OME_ARROW_TAG_TYPE,
    OME_ARROW_TAG_VERSION,
)
from ome_arrow.tensor import LazyTensorView, TensorView
from ome_arrow.utils import describe_ome_arrow, verify_ome_arrow
from ome_arrow.view import view_matplotlib, view_pyvista

__version__ = ome_arrow_version
