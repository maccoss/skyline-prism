# Skyline-PRISM v26.1.1 Release Notes

## Overview

Patch release to fix missing dependencies, configuration validation, and protein grouping issues.

## Bug Fixes

- **Missing dependency**: Added `statsmodels>=0.13` to required dependencies. RT-lowess normalization requires statsmodels but it was not listed in `pyproject.toml`, causing crashes on fresh pip installs.
- **Config validation false positive**: Added `library_assist` to known config keys. Previously, using the `transition_rollup.library_assist` nested configuration block triggered a spurious "unknown configuration keys" warning.
- **Protein parsimony crash**: Fixed crash when `protein_column` and `protein_name_column` config options point to the same column (e.g., both set to `"Protein"`). This caused a `ValueError: The truth value of a Series is ambiguous` error during protein grouping.

## Testing

- **291 tests passing**
- **60% overall coverage**
