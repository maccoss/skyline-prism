# Skyline-PRISM v26.1.0 Release Notes

## Overview

This release introduces the `consensus` transition rollup method and comprehensive methods documentation, with significant improvements to the `library_assist` configuration and several important bug fixes.

## New Features

### New Consensus Transition Rollup Method
- **Consensus rollup**: New transition→peptide rollup method (`method: "consensus"`) that weights transitions by their cross-sample consistency. Transitions that deviate from the consensus fold-change pattern are down-weighted.
- **Algorithm**: Models `log2(T_ij) = α_i + β_j + ε_ij` where α_i is the transition-specific fragmentation efficiency and β_j is the sample abundance. Weights are computed as `w_i = 1 / (Var(ε_i) + λ)`.

### Methods Documentation
- **New `docs/methods.md`**: Comprehensive technical documentation of all PRISM algorithms, written in manuscript methods style. Covers data formats, transition→peptide rollup, global normalization, batch correction, protein parsimony, and peptide→protein rollup.

### Library-Assist Configuration Improvements
- **Flexible method naming**: Both `library_assist` (underscore) and `library-assisted` (hyphen) are now accepted as valid method names.
- **Nested config format**: The `library_assist:` block now supports nested parameters (`library_path`, `mz_tolerance`, `min_matched_fragments`) in addition to the flat format.
- **Cleaner config template**: Removed unsupported `r_squared_threshold` and `mad_threshold` options from config template since these are internal defaults.

## Bug Fixes

- **Divide-by-zero warning**: Fixed RuntimeWarning in `spectral_library.py` when computing R-squared with zero total sum of squares. Added `np.errstate` context manager to suppress the already-handled edge case.
- **Undefined function**: Fixed `robust_least_squares_rollup` reference error by using `least_squares_rollup` with `remove_outliers` parameter.
- **Config parsing**: Fixed library-assist config parsing to properly read nested `library_assist.library_path` format.

## Code Quality

- **Ruff linter cleanup**: Fixed 250 lint issues across the codebase (whitespace, import sorting, unused variables).
- **All lints passing**: Project now passes `ruff check` for E/F/I/W rules.

## Testing

- **291 tests passing**
- **60% overall coverage**
