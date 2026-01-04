# Skyline-PRISM v0.1.2 Release Notes

## Overview

This release brings significant performance improvements for large datasets, enhanced Quality Control (QC) reports, robust batch estimation features, and Python 3.10+ support. The adaptive rollup logic has been optimized for speed (O(1) lookups) and stability, and the pipeline now includes automatic batch estimation capabilities.

## Breaking Changes

- **Python 3.10+ now required**: Dropped Python 3.9 support to use modern type features.

## New Features

### Automatic Batch Estimation
When metadata doesn't include a `batch` column, PRISM can now estimate batches from acquisition times:
- **Auto/Gap mode**: Detects natural batch breaks from time gaps.
- **Fixed mode**: Divides samples evenly into N batches by time.

### Multiple Metadata Files
The `-m/--metadata` argument now accepts multiple files which are automatically merged.

### QC Report Enhancements
- **Rich Metadata Header**: Detailed summary of input files, pipeline version, and processing parameters.
- **Improved Visualization**: Simplified plot labels, better correlation color scaling, and clearer layout.
- **Correct Scaling**: QC plots now automatically handle linear-scale raw data vs log-scale normalized data comparisons.


## Performance Improvements

- **Adaptive Rollup Speedup**:
  - Optimized data loading (loading only required columns).
  - O(1) peptide lookups and pre-built dictionary access, significantly reducing processing time.
  - Reduced optimization search space (fewer parameters).
- **Memory Management**:
  - DuckDB configured for external sorting to handle datasets >10GB without OOM errors.
  - Increased default sort buffer.

## Bug Fixes

- **Sample ID Matching**: Robust handling of `Sample ID` (with batch suffix) vs `Replicate Name` mismatches throughout the pipeline.
- **Adaptive Fallback**: Fixed logic where meaningless parameters were logged; proper fallback to sum method when adaptive learning yields no improvement.

- **QC Plot Labels**: Fixed Reference and QC samples not being correctly identified in plots due to ID format issues.
- **Duplicate Logging**: Fixed redundant progress messages in streaming rollup.
- **Config Validation**: Added warnings for unknown/misspelled config keys.


## Output Files

The pipeline produces:
- `corrected_peptides.parquet` / `corrected_proteins.parquet` (Normalized, Batch-Corrected, Linear Scale)
- `peptides_rollup.parquet` / `proteins_raw.parquet` (Raw Abundances)
- `protein_groups.tsv` (Group Definitions)
- `qc_report.html` (Interactive QC Report)
- `metadata.json` (Provenance)

## Installation

```bash
pip install skyline-prism==0.1.2
```

Or from source:

```bash
git clone https://github.com/maccoss/skyline-prism.git
cd skyline-prism
pip install -e ".[viz]"
```
