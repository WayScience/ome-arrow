# 2026 OME Community Meeting - OME-Arrow

## Authors

Dave Bunten, Jenna Tomkinson, Michael Lippincott, Cameron Mattson, Julia B. Curd, and Gregory P. Way

## Title

OME-Arrow: Unifying Images, Metadata, and Morphology in an Interoperable Data Model for High-Content Imaging

## Abstract

Modern bioimaging workflows combine images, metadata, and derived measurements across many tools, but these components are often stored in incompatible formats and disconnected systems. This fragmentation makes it difficult to join data, reproduce analyses, and scale from small experiments to large, multi-sample studies.

OME-Arrow is a data model and toolkit for working with bioimaging data in modern analytical environments, where data are processed in code, queried with SQL, and analyzed across tools such as Python and R. It brings images, metadata, and derived measurements into a single structure organized as linked tables, rather than leaving them split across separate files and systems. This allows imaging data to be directly joined, filtered, and analyzed using familiar operations, enabling image-derived measurements, metadata, and experimental context to be queried together in a single system. In contrast to existing workflows, where these relationships must be manually reconstructed across files and tools, OME-Arrow makes them explicit and queryable.

OME-Arrow builds on Open Microscopy Environment (OME) conventions and represents data using Apache Arrow, a columnar in-memory data format designed for fast analytics and efficient data sharing across programming languages. It supports ingestion from formats such as TIFF, OME-Zarr, and NumPy, and export to Arrow-native formats (e.g., Parquet, Lance, Vortex) as well as OME-TIFF and OME-Zarr. Data can be processed directly in standalone workflows using these formats, enabling local analysis, scripting, and integration with tools such as SQL engines and DuckDB. For larger-scale use cases, the same data can be organized into an Apache Iceberg-style table structure, which supports dataset versioning, schema evolution, and concurrent access across systems. These two modes use the same underlying data model, allowing workflows to scale from local analysis to warehouse environments without restructuring data. The library also provides lazy scan-style access for large datasets, supports tensor-based pathways for machine learning, and integrates with napari-ome-arrow for advanced visualization and CytoDataFrame for feature-centric analysis within Jupyter notebook environments.

These capabilities enable end-to-end image-based profiling workflows in which raw images, single-cell features, and experimental metadata are analyzed together without intermediate data reshaping. In pediatric cancer research settings, this supports direct querying across imaging data and derived measurements, enabling researchers to relate cellular morphology to perturbations such as compounds, genetic modifications, or treatment conditions. By making these relationships explicit and queryable, OME-Arrow reduces the need for custom data integration steps and improves the consistency of downstream analyses. This approach is being applied to pediatric cancer datasets in collaboration with Alex’s Lemonade Stand Foundation, where integrated access to imaging and profiling data supports systematic exploration of phenotype–treatment relationships and more reproducible analytical workflows.
