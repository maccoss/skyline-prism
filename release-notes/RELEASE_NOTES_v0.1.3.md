# Skyline-PRISM v0.1.3 Release Notes

## Overview

This release addresses critical bugs in data scaling and outlier handling, enhances metadata recording for reproducibility, and provides significant improvements to the interactive Viewer.

## New Features

### Enhanced Metadata Recording & Provenance
- **Merged metadata always saved**: When using `-m` with multiple metadata files, the merged result is now saved to `sample_metadata.tsv` (previously only auto-generated metadata was saved).
- **Metadata provenance**: `metadata.json` now records `metadata_files` (original input paths) and `metadata_source` (`user_provided` or `auto_generated`).
- **Auto-load from provenance**: When using `--from-provenance`, input files (`-i`) and metadata files (`-m`) are automatically loaded from the previous run if still available. Falls back to saved `sample_metadata.tsv` if original files have moved.
- **Batch column preserved**: Batch assignments (explicit or inferred from filename) are now preserved in the saved metadata.

### Unified Viewer Improvements
- **Robust Grouping**: Fixed metadata merging issues to ensure data is correctly grouped even when filenames have matching issues (e.g. extension mismatches).
- **Dynamic Coloring**: Experimental subgroups now automatically receive distinct colors while preserving standard QC/Ref coloring.
- **Extended QC Options**: PCA plot can now be colored by any metadata column (e.g. Batch, Treatment), with smart handling for QC/Reference samples.
- **UI Simplification**: Removed redundant "Group by" dropdown to declutter interface.
- **Enhanced Plots**: Boxplots updated for cleaner look (dots only), increased font sizes for better readability.
- **QC Alignment**: Viewer QC plots (PCA, CV) now strictly match the algorithms used in the HTML/PDF QC reports.
- **UI UX**: Simplified dropdown menus by removing redundant fields.

### QC Report Plot Improvements
- **Consistent PCA Colors**: PCA plots now use consistent colors across all views (blue=experimental, red=reference, orange=qc).
- **PCA Draw Order**: Reference and QC samples are now plotted last so they appear on top of experimental samples.
- **Intensity Plot Grouping**: Peptide intensity boxplots now sort samples by sample_type first, grouping reference, qc, and experimental samples together.

## Bug Fixes

- **Scale Handling**: Fixed "Double Log" issues where data was redundantly logged; ensured strict Log2 (internal) vs Linear (output) handling.
- **CV Calculations**: Fixed Coefficient of Variation (CV) calculations to correctly use linear scale, ensuring accurate QC metrics.
- **Outlier Handling**: Fixed a crash in protein rollup when outlier samples were excluded (updated sample list propagation).
- **None Sample ID Handling**: Fixed crash when Sample ID column contains `None` values by filtering nulls before sorting.

