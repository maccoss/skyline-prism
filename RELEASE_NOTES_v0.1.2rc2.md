# Skyline-PRISM v0.1.2rc2 Release Notes

## Overview

This release focuses on bug fixes, improved robustness, and Python 3.10+ support. The adaptive rollup feature has been refined with better config handling and comprehensive scale handling tests.

## Breaking Changes

- **Python 3.10+ now required** - Dropped Python 3.9 support to enable modern type annotation syntax (`list[str] | None`). Update your environment if you're still on Python 3.9.

## Bug Fixes

- **Fixed overflow warnings during log2/linear conversions** - Large abundance values no longer cause RuntimeWarning during scale transformations
- **Fixed inf values in output** - Proper handling of zero/negative values during log2 transformation prevents infinite values in output files
- **Fixed adaptive rollup not learning** - `learn_adaptive_weights` now defaults to `true` when `method: adaptive`, ensuring weights are learned from reference samples
- **Fixed config key validation** - Added warnings for unknown/misspelled config keys (e.g., `learn_weights` instead of `learn_adaptive_weights`)
- **Fixed `shape_corr_low_threshold`** - Parameter now correctly passed from config to the learning function
- **Fixed pipeline variable reference error** - Resolved `NameError: name 'peptide_normalized_path' is not defined` at Stage 4

## Improvements

- **Streamlined output files** - Removed redundant intermediate files:
  - Renamed `peptides_rollup.2.parquet` to `peptides_rollup.parquet`
  - Eliminated duplicate `peptides_normalized.3.parquet` (data now in `corrected_peptides.parquet`)
  - Single-write optimization: `corrected_peptides.parquet` written once and reused
  
- **Improved CI testing** - Test matrix now includes Python 3.10, 3.11, 3.12, and 3.13

- **Enhanced documentation** - Updated README, SPECIFICATION.md, and AGENTS.md to reflect current output file names and config parameters

## Testing

- **196 tests passing** (up from 182)
- **New test file**: `tests/test_scale_handling.py` - Comprehensive tests for log2/linear conversions and CV calculations

## Output Files

The pipeline now produces these output files:

| File | Description |
|------|-------------|
| `corrected_peptides.parquet` | Peptide-level normalized/batch-corrected quantities (LINEAR scale) |
| `corrected_proteins.parquet` | Protein-level normalized/batch-corrected quantities (LINEAR scale) |
| `peptides_rollup.parquet` | Raw peptide abundances from transition rollup (before normalization) |
| `proteins_raw.parquet` | Raw protein abundances from peptide rollup (before normalization) |
| `protein_groups.tsv` | Protein group definitions |
| `metadata.json` | Complete processing parameters for reproducibility |
| `qc_report.html` | HTML QC report with embedded diagnostic plots |

## Installation

```bash
pip install skyline-prism==0.1.2rc1
```

Or install from source:

```bash
git clone https://github.com/maccoss/skyline-prism.git
cd skyline-prism
pip install -e ".[viz]"
```

## Full Changelog

See commits since v0.1.1rc2: https://github.com/maccoss/skyline-prism/compare/v0.1.1rc1...v0.1.2rc2
