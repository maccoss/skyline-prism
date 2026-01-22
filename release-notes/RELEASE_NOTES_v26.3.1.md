# Skyline-PRISM v26.3.1 Release Notes

## Overview

Patch release with logging improvements for better consistency and cleaner output.

## Changes

### Consistent CV Reporting Across All Normalization Methods

All normalization methods now report before/after CVs for reference and QC samples:

- **median normalization**: Now logs "Before normalization - median CV" and "After normalization - median CV"
- **quantile normalization**: Now logs before/after CVs consistently with other methods
- **VSN normalization**: Now logs before/after CVs (using arcsinh-transformed values)
- **rt_lowess normalization**: Already had this logging (unchanged)

Example output:

```text
Stage 2b: Peptide Normalization
------------------------------------------------------------
  Applying quantile normalization...
  Before normalization - median CV: Reference: 12.3%, QC: 14.1%
  Quantile normalization complete: 238 samples normalized
  After normalization - median CV: Reference: 11.8%, QC: 13.5%
```

### Configurable Protein-Level Normalization

Added new `protein_normalization` config section to control protein-level normalization:

```yaml
protein_normalization:
  # Method: median (default), none
  method: "median"
```

- **median** (default): Centers all samples to the same median protein abundance
- **none**: Skip protein-level normalization (useful if peptide-level normalization is sufficient or you prefer to handle normalization downstream)

Previously, protein-level normalization was always median with no option to skip.

### Reduced Redundant Logging

Removed duplicate batch assignment logging that was appearing twice in logs:

- Previously: Batch info was logged both in `get_batches_from_source_document()` and again in Stage 2c/4c
- Now: Batch info is only logged once in Stage 2c/4c with comprehensive details

Before (redundant):

```text
  Assigning batches from source documents: 3 batches
    Plate1.csv: 80 samples
    Plate2.csv: 78 samples
    Plate3.csv: 80 samples
------------------------------------------------------------
Stage 2c: Peptide ComBat Batch Correction
------------------------------------------------------------
  Batch source: source documents
  Batches found: ['Plate1.csv', 'Plate2.csv', 'Plate3.csv']
    Plate1.csv: 80 samples
    Plate2.csv: 78 samples
    Plate3.csv: 80 samples
```

After (cleaner):

```text
------------------------------------------------------------
Stage 2c: Peptide ComBat Batch Correction
------------------------------------------------------------
  Batch source: source documents
  Batches found: ['Plate1.csv', 'Plate2.csv', 'Plate3.csv']
    Plate1.csv: 80 samples
    Plate2.csv: 78 samples
    Plate3.csv: 80 samples
```

### New Documentation: Output Files Reference

Added comprehensive documentation for all PRISM output files in `docs/output_files.md`:

- Complete column definitions for all parquet files (peptides, proteins, merged data)
- Data types and descriptions for each column
- Scale conventions (LINEAR vs LOG2) clearly documented
- Example code for reading files in Python, R, and DuckDB
- Common analysis workflows (export to CSV, filter proteins, calculate CVs)

## Bug Fixes

None in this release.

## Testing

- All existing tests passing

## Upgrade Notes

- No breaking changes from v26.3.0
- Log output format changed - scripts parsing logs may need updates
