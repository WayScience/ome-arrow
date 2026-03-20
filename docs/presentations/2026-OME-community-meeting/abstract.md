# 2026 OME Community Meeting - OME-Arrow

## Authors

Dave Bunten, Jenna Tomkinson, Michael Lippincott, Cameron Mattson, Gregory P. Way

## Title

OME-Arrow: Unifying Images, Metadata, and Features in an Interoperable Data Model

## Abstract

Modern bioimaging workflows increasingly combine images, metadata, and derived measurements across many tools and platforms. Enabling these components to work together seamlessly is key to interoperable and scalable analysis.

OME-Arrow is a project that applies Open Microscopy Environment (OME) conventions through Apache Arrow to integrate imaging data with modern analytical workflows. By representing images as Arrow-compatible structures alongside metadata and features, OME-Arrow enables programmatic and relational access using a consistent data model across languages while supporting familiar tools such as SQL engines, DuckDB, and Parquet-based pipelines.

The library supports ingestion from TIFF, OME-Zarr, and NumPy, with export to OME-Parquet, OME-Zarr, and OME-TIFF, along with lazy scan-style access for large datasets and tensor pathways for machine learning. OME-Arrow also integrates with napari-ome-arrow for visualization and CytoDataFrame for scalable feature-centric workflows, offering a modular, standards-aligned approach that complements the broader open bioimaging ecosystem.