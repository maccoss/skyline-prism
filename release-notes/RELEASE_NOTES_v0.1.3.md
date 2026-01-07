# Skyline-PRISM v0.1.3 Release Notes

## Overview

This release introduces library-assisted transition rollup for spectral library-based interference detection, significant performance optimizations for large datasets, and enhanced metadata recording for reproducibility.

## New Features

### Library-Assisted Transition Rollup
- **New `library_assist` method**: Uses a spectral library to detect and exclude interfered transitions via iterative least squares fitting.
- **Algorithm**: For each peptide, fits `observed = scale * library + residuals`. Transitions with large positive residuals (interference) are iteratively removed and the model refitted.
- **Supported formats**: BLIB (Skyline BiblioSpec) and Carafe TSV (DIA-NN) spectral library formats.
- **Key insight**: Only HIGH positive residuals indicate interference. Low/zero signal indicates low abundance, not interference.
- **Dramatic improvement**: Up to 111% CV reduction for ~29% of peptides with real interference.

### Performance Optimizations
- **Vectorized least squares**: Library-assisted rollup processes all samples in parallel using BLAS matrix operations (~10x speedup on large datasets).
- **Merge-and-sort streaming**: CSV merge and sort in a single DuckDB operation, eliminating redundant sorting passes.
- **Pre-sorted optimization**: Rollup skips sorting when data is already sorted (detected automatically).
- **Implementation**: `spectral_library.py` -> `least_squares_rollup_vectorized()`, `data_io.py` -> `merge_and_sort_streaming()`

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

### New Consensus Transition Rollup Method
- **Consensus rollup**: New transition->peptide rollup method (`method: "consensus"`) that weights transitions by their cross-sample consistency. Transitions that deviate from the consensus fold-change pattern are down-weighted.
- **Algorithm**: Models `log2(T_ij) = alpha_i + beta_j + epsilon_ij` where alpha_i is the transition-specific fragmentation efficiency and beta_j is the sample abundance. Weights are computed as `w_i = 1 / (Var(epsilon_i) + lambda)`.
- **Diagnostics output**: When using `consensus` method, PRISM automatically outputs `consensus_diagnostics.csv` containing peptide, transition, sample, residual, weight, and variance information for identifying problematic transitions.

### Methods Documentation
- **New `docs/methods.md`**: Comprehensive technical documentation of all PRISM algorithms, written in manuscript methods style. Covers data formats, transition->peptide rollup, global normalization, batch correction, protein parsimony, and peptide->protein rollup.

## Bug Fixes

- **Scale Handling**: Fixed "Double Log" issues where data was redundantly logged; ensured strict Log2 (internal) vs Linear (output) handling.
- **CV Calculations**: Fixed Coefficient of Variation (CV) calculations to correctly use linear scale, ensuring accurate QC metrics.
- **Outlier Handling**: Fixed a crash in protein rollup when outlier samples were excluded (updated sample list propagation).
- **None Sample ID Handling**: Fixed crash when Sample ID column contains `None` values by filtering nulls before sorting.

## Testing

- **291 tests passing** (up from 196 in v0.1.2)
- **60% overall coverage** (up from 47%)
- **New test module**: `tests/test_spectral_library.py` with 11 tests for vectorized least squares

