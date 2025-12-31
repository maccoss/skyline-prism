# Skyline-PRISM v0.1.2rc4 Release Notes

## Overview

This release focuses on performance improvements for large datasets and fixes the adaptive rollup fallback logic. The QC validation decision logic has been clarified, and memory management for DuckDB sorting has been improved.

## Performance Improvements

### Adaptive Rollup Pre-computation (Major Speedup)
- **Efficient Data Loading**: DuckDB query now only loads columns needed for adaptive learning (8 columns instead of all 27+), significantly reducing I/O and memory for large datasets.
- **O(1) Peptide Lookup**: Converted valid peptide check from O(n) array scan to O(1) set lookup.
- **Pre-built Lookup Dictionary**: Groupby results are now stored in a dictionary for O(1) access instead of calling `.get_group()` on each iteration.
- **Reduced Optimization Parameters**: Removed `relative_intensity` from optimization (empirically always 0). Now optimizing only 2 parameters: `mz` and `shape_outlier`, reducing the optimization search space.
- **Skip QC Validation When No Improvement**: When optimization finds no improvement (all betas ≈ 0), the expensive QC validation step is skipped.
- **Proper Sum Fallback**: When falling back to sum, the CLI now uses the actual sum method instead of adaptive rollup with zero betas.
- **QC-Based Decision Logic**: QC validation (not reference improvement) is the primary criterion for using adaptive weights.

### DuckDB Memory Management
- **External Sorting for Large Datasets**: DuckDB is now configured with memory limits and a temp directory for disk spilling, enabling external sorting for very large datasets (>10GB) without running out of memory.
- **Increased Default Memory**: Default sort buffer increased from 2GB to 8GB for better handling of large files.

## Bug Fixes

- **Adaptive Fallback Logic**: Fixed issue where meaningless all-zero adaptive parameters were logged when falling back to sum method.
- **Duplicate Warnings**: Removed duplicate "Falling back to sum" warnings that appeared in both transition_rollup.py and cli.py.
- **Incorrect Pipeline Version**: Fixed issue where the QC report incorrectly listed version "0.3.0" instead of the actual installed version (0.1.2rc4).
- **Incorrect Batch Correction Method**: Fixed issue where "combat_reference_anchored" was hardcoded in metadata even when using standard "combat".
- **QC Report Layout**: Updated "Dataset Summary" table to show "Transitions" count and improved layout (Samples/Transitions/Peptides/Proteins).
- **Redundant Plot Titles**: Removed internal titles from CV distribution plots in QC report to avoid redundancy with section headers.

## Testing

- **New test file**: `tests/test_adaptive_fallback.py` - 7 tests for QC validation decision logic and fallback behavior:
  - Tests for skipping QC validation when no improvement found
  - Tests for proper fallback flags and reasons
  - Tests for min_improvement_pct threshold behavior

## Installation

```bash
pip install skyline-prism==0.1.2rc4
```

Or install from source:

```bash
git clone https://github.com/maccoss/skyline-prism.git
cd skyline-prism
pip install -e ".[viz]"
```

## Full Changelog

See commits since v0.1.2rc3: https://github.com/maccoss/skyline-prism/compare/v0.1.2rc3...v0.1.2rc4
