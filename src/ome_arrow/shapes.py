"""Arrow-native shape and relationship tables for OME-Arrow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from os import PathLike
from typing import Any, Iterable, Literal, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ome_arrow.meta import OME_ARROW_TAG_VERSION

GeometryEncoding = Literal[
    "geoarrow.point",
    "geoarrow.linestring",
    "geoarrow.polygon",
    "geoarrow.multipolygon",
    "ome.mesh3d",
    "ome.labelmask",
    "ome.pointcloud",
    "ome.boundingbox",
]

SUPPORTED_GEOMETRY_ENCODINGS: frozenset[str] = frozenset(
    {
        "geoarrow.point",
        "geoarrow.linestring",
        "geoarrow.polygon",
        "geoarrow.multipolygon",
        "ome.mesh3d",
        "ome.labelmask",
        "ome.pointcloud",
        "ome.boundingbox",
    }
)

OME_ARROW_SHAPES_METADATA_KEY = b"ome.arrow.shapes"
OME_ARROW_RELATIONSHIPS_METADATA_KEY = b"ome.arrow.relationships"

DEFAULT_SHAPE_COLUMNS = {
    "object_id",
    "image_id",
    "label_image_id",
    "label_value",
    "geometry",
    "centroid",
    "bbox",
    "class",
    "confidence",
}

RELATIONSHIP_TYPES: frozenset[str] = frozenset(
    {
        "contains",
        "adjacent",
        "touches",
        "parent",
        "track",
        "derived_from",
    }
)

RELATIONSHIP_SCHEMA = pa.schema(
    [
        pa.field("parent_id", pa.string(), nullable=False),
        pa.field("child_id", pa.string(), nullable=False),
        pa.field("relationship_type", pa.string(), nullable=False),
        pa.field("confidence", pa.float32()),
    ]
)


def _coordinate_type(dimensions: int) -> pa.ListType:
    """Return the Arrow coordinate vector type for a geometry dimension."""
    if dimensions < 1:
        raise ValueError("geometry dimensions must be at least 1.")
    return pa.list_(pa.float64())


def geometry_storage_type(
    geometry_encoding: GeometryEncoding | str,
    *,
    dimensions: int = 2,
) -> pa.DataType:
    """Return the Arrow storage type for a registered geometry encoding.

    Args:
        geometry_encoding: Registered OME-Arrow geometry encoding name.
        dimensions: Number of coordinate dimensions for coordinate vectors.

    Returns:
        Arrow data type for the logical geometry column.

    Raises:
        ValueError: If the encoding is unknown or dimensions are invalid.
    """
    if geometry_encoding not in SUPPORTED_GEOMETRY_ENCODINGS:
        raise ValueError(f"Unsupported geometry_encoding: {geometry_encoding!r}.")

    coordinate = _coordinate_type(dimensions)
    if geometry_encoding == "geoarrow.point":
        return coordinate
    if geometry_encoding in {"geoarrow.linestring", "ome.pointcloud"}:
        return pa.list_(coordinate)
    if geometry_encoding == "geoarrow.polygon":
        return pa.list_(pa.list_(coordinate))
    if geometry_encoding == "geoarrow.multipolygon":
        return pa.list_(pa.list_(pa.list_(coordinate)))
    if geometry_encoding == "ome.boundingbox":
        return pa.struct(
            [
                pa.field("min", coordinate, nullable=False),
                pa.field("max", coordinate, nullable=False),
            ]
        )
    if geometry_encoding == "ome.labelmask":
        return pa.struct(
            [
                pa.field("label_image_id", pa.string(), nullable=False),
                pa.field("label_value", pa.int64(), nullable=False),
            ]
        )
    return pa.struct(
        [
            pa.field("vertices", pa.list_(_coordinate_type(3)), nullable=False),
            pa.field("faces", pa.list_(pa.list_(pa.int32())), nullable=False),
        ]
    )


def shape_metadata(
    *,
    geometry_encoding: GeometryEncoding | str,
    axes: Sequence[str] = ("y", "x"),
    units: Sequence[str] | None = None,
    coordinate_space: str = "pixel",
    geometry_column: str = "geometry",
) -> dict[str, Any]:
    """Build JSON-serializable schema metadata for a shape table.

    Args:
        geometry_encoding: Registered OME-Arrow geometry encoding name.
        axes: Coordinate axis names for geometry values.
        units: Units aligned to axes. Defaults to ``"pixel"`` for each axis.
        coordinate_space: Name of the coordinate space for geometry values.
        geometry_column: Name of the logical geometry column.

    Returns:
        Metadata dictionary stored under ``OME_ARROW_SHAPES_METADATA_KEY``.

    Raises:
        ValueError: If the encoding, axes, or units are invalid.
    """
    if geometry_encoding not in SUPPORTED_GEOMETRY_ENCODINGS:
        raise ValueError(f"Unsupported geometry_encoding: {geometry_encoding!r}.")
    if not axes:
        raise ValueError("axes must contain at least one axis name.")
    if units is None:
        units = tuple("pixel" for _ in axes)
    if len(units) != len(axes):
        raise ValueError("units must have the same length as axes.")

    return {
        "type": "ome.arrow.shapes",
        "version": str(OME_ARROW_TAG_VERSION),
        "geometry_column": geometry_column,
        "geometry_encoding": str(geometry_encoding),
        "axes": list(axes),
        "units": list(units),
        "coordinate_space": coordinate_space,
    }


def _schema_with_json_metadata(
    schema: pa.Schema,
    *,
    key: bytes,
    payload: dict[str, Any],
) -> pa.Schema:
    """Attach compact JSON metadata to an Arrow schema."""
    metadata = dict(schema.metadata or {})
    metadata[key] = json.dumps(payload, sort_keys=True).encode("utf-8")
    return schema.with_metadata(metadata)


def shape_schema(
    geometry_encoding: GeometryEncoding | str,
    *,
    axes: Sequence[str] = ("y", "x"),
    units: Sequence[str] | None = None,
    coordinate_space: str = "pixel",
    geometry_column: str = "geometry",
    measurement_fields: Iterable[pa.Field] | None = None,
) -> pa.Schema:
    """Create an OME-Arrow shape table schema.

    Args:
        geometry_encoding: Registered OME-Arrow geometry encoding name.
        axes: Coordinate axis names for geometry, centroid, and bounding boxes.
        units: Units aligned to axes. Defaults to ``"pixel"`` for each axis.
        coordinate_space: Name of the coordinate space for geometry values.
        geometry_column: Name of the logical geometry column.
        measurement_fields: Extra Arrow fields for ordinary measurement columns.

    Returns:
        Arrow schema with OME-Arrow shape metadata attached.
    """
    metadata = shape_metadata(
        geometry_encoding=geometry_encoding,
        axes=axes,
        units=units,
        coordinate_space=coordinate_space,
        geometry_column=geometry_column,
    )
    coordinate = _coordinate_type(len(axes))
    fields = [
        pa.field("object_id", pa.string(), nullable=False),
        pa.field("image_id", pa.string()),
        pa.field("label_image_id", pa.string()),
        pa.field("label_value", pa.int64()),
        pa.field(
            geometry_column,
            geometry_storage_type(geometry_encoding, dimensions=len(axes)),
        ),
        pa.field("centroid", coordinate),
        pa.field(
            "bbox",
            pa.struct(
                [
                    pa.field("min", coordinate, nullable=False),
                    pa.field("max", coordinate, nullable=False),
                ]
            ),
        ),
        pa.field("class", pa.string()),
        pa.field("confidence", pa.float32()),
    ]
    if measurement_fields is not None:
        fields.extend(measurement_fields)

    return _schema_with_json_metadata(
        pa.schema(fields),
        key=OME_ARROW_SHAPES_METADATA_KEY,
        payload=metadata,
    )


def _infer_measurement_fields(rows: Sequence[dict[str, Any]]) -> list[pa.Field]:
    """Infer measurement fields for columns outside the canonical shape columns."""
    if not rows:
        return []

    fields: list[pa.Field] = []
    row_columns = set().union(*(row.keys() for row in rows))
    for name in sorted(row_columns - DEFAULT_SHAPE_COLUMNS):
        values = [row.get(name) for row in rows]
        fields.append(pa.field(name, pa.array(values).type))
    return fields


def make_shape_table(
    rows: Sequence[dict[str, Any]],
    *,
    geometry_encoding: GeometryEncoding | str,
    axes: Sequence[str] = ("y", "x"),
    units: Sequence[str] | None = None,
    coordinate_space: str = "pixel",
    geometry_column: str = "geometry",
    validate: bool = True,
) -> pa.Table:
    """Create an OME-Arrow shape table from Python row dictionaries.

    Args:
        rows: Shape rows, where each row represents one biological object.
        geometry_encoding: Registered OME-Arrow geometry encoding name.
        axes: Coordinate axis names for geometry, centroid, and bounding boxes.
        units: Units aligned to axes. Defaults to ``"pixel"`` for each axis.
        coordinate_space: Name of the coordinate space for geometry values.
        geometry_column: Name of the logical geometry column.
        validate: Validate the table after construction.

    Returns:
        Arrow table with OME-Arrow shape schema metadata.
    """
    row_list = list(rows)
    schema = shape_schema(
        geometry_encoding,
        axes=axes,
        units=units,
        coordinate_space=coordinate_space,
        geometry_column=geometry_column,
        measurement_fields=_infer_measurement_fields(row_list),
    )
    table = pa.Table.from_pylist(row_list, schema=schema)
    if validate:
        validate_shape_table(table)
    return table


def _shape_metadata_from_schema(schema: pa.Schema) -> dict[str, Any]:
    """Read OME-Arrow shape JSON metadata from a schema."""
    raw_metadata = schema.metadata or {}
    raw_payload = raw_metadata.get(OME_ARROW_SHAPES_METADATA_KEY)
    if raw_payload is None:
        raise ValueError("Shape table schema metadata is missing OME-Arrow shapes.")
    metadata = json.loads(raw_payload.decode("utf-8"))
    if metadata.get("type") != "ome.arrow.shapes":
        raise ValueError("Shape table metadata type must be 'ome.arrow.shapes'.")
    return metadata


def validate_shape_table(table: pa.Table) -> None:
    """Validate an OME-Arrow shape table.

    Args:
        table: Arrow table to validate.

    Raises:
        ValueError: If required metadata, columns, encoding, or IDs are invalid.
    """
    metadata = _shape_metadata_from_schema(table.schema)
    geometry_column = metadata.get("geometry_column", "geometry")
    geometry_encoding = metadata.get("geometry_encoding")
    axes = metadata.get("axes", [])

    if geometry_encoding not in SUPPORTED_GEOMETRY_ENCODINGS:
        raise ValueError(f"Unsupported geometry_encoding: {geometry_encoding!r}.")
    if "object_id" not in table.column_names:
        raise ValueError("Shape table must contain an object_id column.")
    if geometry_column not in table.column_names:
        raise ValueError(f"Shape table must contain {geometry_column!r} column.")
    if table.schema.field(geometry_column).type != geometry_storage_type(
        geometry_encoding,
        dimensions=len(axes),
    ):
        raise ValueError("Shape table geometry column does not match metadata.")
    if pc.any(pc.is_null(table["object_id"])).as_py():
        raise ValueError("Shape table object_id values must not be null.")


def write_shape_parquet(
    table: pa.Table,
    path: str | PathLike[str],
    *,
    compression: str | None = "zstd",
    row_group_size: int | None = 65_536,
    use_dictionary: bool | list[str] = True,
    validate: bool = True,
) -> None:
    """Write an OME-Arrow shape table to Parquet.

    Args:
        table: OME-Arrow shape table to write.
        path: Output Parquet path.
        compression: Parquet compression codec, or ``None`` for uncompressed.
        row_group_size: Number of rows per Parquet row group.
        use_dictionary: Dictionary-encode eligible columns. This is useful for
            repeated scientific labels such as image IDs, label image IDs, and
            object classes.
        validate: Validate the table before writing.

    Raises:
        ValueError: If validation fails.
    """
    if validate:
        validate_shape_table(table)
    pq.write_table(
        table,
        path,
        compression=compression,
        row_group_size=row_group_size,
        use_dictionary=use_dictionary,
    )


def read_shape_parquet(
    path: str | PathLike[str],
    *,
    columns: Sequence[str] | None = None,
    filters: Any | None = None,
    memory_map: bool = True,
    validate: bool = True,
) -> pa.Table:
    """Read an OME-Arrow shape Parquet table.

    Args:
        path: Input Parquet path.
        columns: Optional column projection for analytical reads.
        filters: Optional PyArrow Parquet filters for predicate pushdown.
        memory_map: Use memory mapping where supported.
        validate: Validate complete shape tables after reading. Projected reads
            that omit required columns still validate schema metadata but skip
            full table validation.

    Returns:
        Arrow table read from Parquet.

    Raises:
        ValueError: If OME-Arrow shape metadata or complete-table validation
            fails.
    """
    schema = pq.read_schema(path, memory_map=memory_map)
    metadata = _shape_metadata_from_schema(schema)
    table = pq.read_table(
        path,
        columns=columns,
        filters=filters,
        memory_map=memory_map,
    )
    if not validate:
        return table

    geometry_column = metadata.get("geometry_column", "geometry")
    required = {"object_id", geometry_column}
    if required.issubset(table.column_names):
        validate_shape_table(table)
    return table


def relationship_metadata() -> dict[str, Any]:
    """Build JSON-serializable schema metadata for object relationships."""
    return {
        "type": "ome.arrow.relationships",
        "version": str(OME_ARROW_TAG_VERSION),
        "relationship_types": sorted(RELATIONSHIP_TYPES),
    }


def relationship_schema() -> pa.Schema:
    """Create an OME-Arrow relationship table schema.

    Returns:
        Arrow schema with OME-Arrow relationship metadata attached.
    """
    return _schema_with_json_metadata(
        RELATIONSHIP_SCHEMA,
        key=OME_ARROW_RELATIONSHIPS_METADATA_KEY,
        payload=relationship_metadata(),
    )


def make_relationship_table(
    rows: Sequence[dict[str, Any]],
    *,
    validate: bool = True,
) -> pa.Table:
    """Create an OME-Arrow relationship table from edge rows.

    Args:
        rows: Relationship rows with parent, child, and relationship type.
        validate: Validate the table after construction.

    Returns:
        Arrow table with OME-Arrow relationship metadata.
    """
    table = pa.Table.from_pylist(list(rows), schema=relationship_schema())
    if validate:
        validate_relationship_table(table)
    return table


def validate_relationship_table(table: pa.Table) -> None:
    """Validate an OME-Arrow relationship table.

    Args:
        table: Arrow table to validate.

    Raises:
        ValueError: If required metadata, columns, IDs, or relationship types fail.
    """
    raw_payload = (table.schema.metadata or {}).get(
        OME_ARROW_RELATIONSHIPS_METADATA_KEY
    )
    if raw_payload is None:
        raise ValueError(
            "Relationship table schema metadata is missing OME-Arrow relationships."
        )
    metadata = json.loads(raw_payload.decode("utf-8"))
    if metadata.get("type") != "ome.arrow.relationships":
        raise ValueError(
            "Relationship table metadata type must be 'ome.arrow.relationships'."
        )

    for name in ("parent_id", "child_id", "relationship_type"):
        if name not in table.column_names:
            raise ValueError(f"Relationship table must contain a {name} column.")
        if pc.any(pc.is_null(table[name])).as_py():
            raise ValueError(f"Relationship table {name} values must not be null.")

    unknown = set(table["relationship_type"].to_pylist()) - RELATIONSHIP_TYPES
    if unknown:
        raise ValueError(f"Unsupported relationship_type values: {sorted(unknown)}.")


@dataclass(frozen=True)
class OMEArrowShapes:
    """Convenience wrapper around a validated OME-Arrow shape table."""

    table: pa.Table

    def __post_init__(self) -> None:
        """Validate the wrapped shape table."""
        validate_shape_table(self.table)

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[dict[str, Any]],
        *,
        geometry_encoding: GeometryEncoding | str,
        axes: Sequence[str] = ("y", "x"),
        units: Sequence[str] | None = None,
        coordinate_space: str = "pixel",
        geometry_column: str = "geometry",
    ) -> OMEArrowShapes:
        """Create a shape wrapper from Python row dictionaries.

        Args:
            rows: Shape rows, where each row represents one biological object.
            geometry_encoding: Registered OME-Arrow geometry encoding name.
            axes: Coordinate axis names for geometry, centroid, and bounding boxes.
            units: Units aligned to axes. Defaults to ``"pixel"`` for each axis.
            coordinate_space: Name of the coordinate space for geometry values.
            geometry_column: Name of the logical geometry column.

        Returns:
            Validated OME-Arrow shapes wrapper.
        """
        return cls(
            make_shape_table(
                rows,
                geometry_encoding=geometry_encoding,
                axes=axes,
                units=units,
                coordinate_space=coordinate_space,
                geometry_column=geometry_column,
            )
        )

    @property
    def metadata(self) -> dict[str, Any]:
        """Return decoded OME-Arrow shapes metadata."""
        return _shape_metadata_from_schema(self.table.schema)

    @property
    def geometry_encoding(self) -> str:
        """Return the registered geometry encoding for the table."""
        return str(self.metadata["geometry_encoding"])

    @property
    def axes(self) -> tuple[str, ...]:
        """Return coordinate axis names for the shape table."""
        return tuple(self.metadata["axes"])

    def for_image(self, image_id: str) -> OMEArrowShapes:
        """Return shapes that reference one image ID.

        Args:
            image_id: Image identifier to filter on.

        Returns:
            New wrapper containing only matching shape rows.
        """
        if "image_id" not in self.table.column_names:
            return type(self)(self.table.slice(0, 0))
        mask = pc.equal(self.table["image_id"], image_id)
        return type(self)(self.table.filter(mask))
