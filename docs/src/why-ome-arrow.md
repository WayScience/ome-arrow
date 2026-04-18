# Why OME Arrow?

OME Arrow is a table-oriented representation for OME-aligned image data built on Apache Arrow and commonly persisted as Parquet.
It complements, rather than replaces, Open Microscopy Environment (OME)-Zarr and OME-TIFF.
OME-Zarr and OME-TIFF are core image formats for storage, exchange, and visualization, while OME Arrow focuses on query-centric analytics where image payloads and tabular data need to be handled together.
Open Microscopy Environment (OME)-Zarr is already a strong default for many bioimaging workflows.
It provides cloud-oriented, multiscale image storage and standardized metadata through the Next-Generation File Formats (NGFF) / OME-Zarr specification.
OME Arrow is also cloud-oriented, especially for object-store-backed, table-native analytics workflows in Arrow/Parquet ecosystems.

This page explains why OME Arrow still matters.

## Where current OME formats are strong

- [OME-TIFF](https://ome-model.readthedocs.io/en/stable/ome-tiff/index.html):
  mature image + OME-XML (Extensible Markup Language) metadata packaging in TIFF (Tagged Image File Format)-based files.
- [OME-Zarr (NGFF)](https://ngff.openmicroscopy.org/0.5/):
  standardized Zarr hierarchy, multiscales, labels, and HCS (high-content screening) plate/well metadata.

These formats are excellent for image representation and interoperability.

## Gap OME Arrow is designed to address

In analysis-heavy workflows, image pixels often need to be handled together with tabular data (for example per-cell features, QC (quality control) metrics, and joins across many images).
OME-Zarr and OME-TIFF define image data structures well, but they are not table formats.
The [Zarr core specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html) is centered on typed N-dimensional arrays and groups rather than a canonical table model.
The [OME-Zarr specification (v0.5)](https://ngff.openmicroscopy.org/0.5/) adds strong guidance for image and metadata layout, but does not define a single cross-project table specification.
In practice, teams often introduce project-level table conventions inside Zarr hierarchies (for example, [Annotated Data (AnnData) in Zarr](https://anndata.readthedocs.io/en/stable/fileformat-prose.html)), which can produce surprising or inconsistent data structures across tools.

OME Arrow uses [Apache Arrow](https://arrow.apache.org/overview/) primitives to represent image payloads as typed, queryable values inside table-like data systems (including Parquet).
This makes image-linked analytics easier in SQL (Structured Query Language) / DataFrame-style pipelines.

## OME Arrow vs OME-Zarr

OME Arrow is not a universal replacement for OME-Zarr.

- Use OME-Zarr when your primary need is standards-based multiscale image storage and ecosystem compatibility for image-first tools.
- Use OME Arrow when your primary need is tighter integration between image data and tabular analytics workflows.
- Use both when needed: OME-Zarr for distribution/visualization paths and OME Arrow for query-centric pipelines.

## Preliminary benchmark signal

Preliminary results in [ome-arrow-benchmarks](https://github.com/WayScience/ome-arrow-benchmarks) show that outcomes are highly workload- and layout-dependent.
In the repository's synthetic wide-table plus image-column runs, Arrow-table-native backends can reduce full-table read time and storage size relative to some alternatives, while write performance varies by backend.
In the OME-Arrow-only benchmark that compares against directory-per-image OME-Zarr and TIFF layouts, full write/read timings and random-read timings diverge in different directions depending on operation type.
This impacts comparisons directly: these are not pure "format A vs format B" tests, because the benchmark also reflects table model choices, directory layout choices, and access pattern choices.
For this reason, benchmark results should be treated as preliminary guidance for scenario fit, not universal rankings.

## Why this matters for big-picture data repositories in image-based profiling

[iceberg-bioimage](https://github.com/WayScience/iceberg-bioimage) is one concrete example: it positions [Apache Iceberg](https://iceberg.apache.org/) as a control plane (cataloging, schemas, joins, snapshots), while Zarr and OME-TIFF remain data-plane formats.
Its README also lists OME Arrow as an optional integration for Arrow-native tabular image payloads and lazy image access.

That is the key fit: OME Arrow helps bridge bioimage formats and modern table engines without requiring every workflow to abandon OME-Zarr.
