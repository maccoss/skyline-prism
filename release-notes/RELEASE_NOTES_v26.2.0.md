# Skyline-PRISM v26.2.0 Release Notes

## Overview

Minor version release with significant improvements to protein rollup code architecture and behavior. This release consolidates protein rollup logic into a single source of truth and fixes the `min_peptides` threshold behavior.

## Behavioral Changes

### Protein Rollup `min_peptides` Threshold

**Previous behavior**:
- Proteins with 2 peptides used **mean** (geometric mean in log space)
- The `min_peptides` threshold was hardcoded at 3, ignoring user configuration

**New behavior**:
- Proteins with **fewer than `min_peptides`** now use **sum in linear space**
- The `min_peptides` config setting is properly respected
- Single peptide proteins continue to use the peptide value directly
- Proteins at or above `min_peptides` use the configured method (median_polish, topn, etc.)

**Why sum instead of mean?**:
- Median polish `col_effects` represent summed protein abundance
- Using sum for low-peptide proteins maintains consistent scale with median_polish results
- This provides better comparability across proteins regardless of peptide count

**Example with `min_peptides: 5`:**

| Peptide Count | Method Used | `low_confidence` |
|---------------|-------------|------------------|
| 1 | Direct peptide value | `True` |
| 2-4 | Sum (linear space) | `True` |
| 5+ | `median_polish` (or configured method) | `False` |

## Code Architecture

### Consolidated Protein Rollup Logic

Previously, there were **two separate implementations** of peptide→protein rollup:
- `rollup_to_proteins()` in `rollup.py` (public API for programmatic use)
- `_process_single_protein()` in `chunked_processing.py` (CLI pipeline)

This duplication caused maintenance burden and bugs where fixes were applied to one but not the other.

**Solution**: Created a single core function `rollup_protein_matrix()` in `rollup.py`:
- Encapsulates all protein rollup logic (min_peptides handling, method selection, etc.)
- Both implementations now call this core function
- Single source of truth for rollup behavior

### New Public API

```python
from skyline_prism import rollup_protein_matrix, ProteinMatrixRollupResult

# Core function for rolling up a peptide matrix to protein-level abundance
result = rollup_protein_matrix(
    matrix,  # Peptide x Sample DataFrame (log2 scale)
    method="median_polish",
    min_peptides=3,
    topn_n=3,
    topn_selection="median_abundance",
)

# Result contains:
# - result.abundances: pd.Series of sample abundances
# - result.residuals: dict of peptide -> sample -> residual (for median_polish)
# - result.polish_result: MedianPolishResult (for median_polish)
# - result.topn_result: TopNResult (for topn method)
```

## Technical Details

### Files Modified

- `skyline_prism/rollup.py`:
  - Added `_sum_linear()` helper function for linear-space summation
  - Added `ProteinMatrixRollupResult` dataclass
  - Added `rollup_protein_matrix()` core function
  - Updated `rollup_to_proteins()` to use core function

- `skyline_prism/chunked_processing.py`:
  - Updated `_process_single_protein()` to use `rollup_protein_matrix()` from rollup.py
  - Removed duplicate rollup logic

- `skyline_prism/__init__.py`:
  - Added exports: `rollup_protein_matrix`, `ProteinMatrixRollupResult`

## Testing

- All 53 rollup, chunked processing, and parsimony tests passing
- Verified consistent behavior between API and CLI implementations
- Test `test_two_peptide_uses_sum` validates new sum behavior

## Upgrade Notes

- Results may differ slightly for proteins with 2 peptides (now summed instead of averaged)
- If you require the previous behavior, you can access peptide-level data and compute mean manually
- All other results should be identical

## Migration from v26.1.x

No code changes required. The new behavior is automatic:
- Existing configs continue to work
- `min_peptides` setting is now properly respected
- For most workflows, results will be very similar

To verify your workflow:
1. Run PRISM with your existing config
2. Compare `corrected_proteins.parquet` output
3. Proteins with 2-4 peptides (with default `min_peptides: 3`) will show sum instead of mean
