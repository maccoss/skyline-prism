# Skyline-PRISM v26.3.2 Release Notes

## Overview

Patch release adding native parquet format support for Skyline reports and metadata files, with robust validation to prevent common user errors.

## Changes

### Native Parquet Format Support

Added full support for parquet-format Skyline reports and metadata files:

- **Skyline report files**: Now accepts `.parquet` files in addition to CSV/TSV
- **Metadata files**: Now accepts `.parquet` files in addition to CSV/TSV
- **Multiple metadata files**: Can merge multiple metadata parquet files together (e.g., `metadata1.parquet metadata2.parquet`)
- **Mixed formats**: Can process CSV, TSV, and parquet files together in the same run
- **Automatic format detection**: File format detected from extension (`.csv`, `.tsv`, `.txt`, `.parquet`)
- **Performance**: Parquet files load faster and use less memory than CSV equivalents

**Usage examples:**

```bash
# Single parquet report
prism run -i data.parquet -o output/ -c config.yaml

# Multiple parquet reports
prism run -i plate1.parquet plate2.parquet plate3.parquet -o output/

# Mixed CSV and parquet
prism run -i plate1.csv plate2.parquet -o output/

# Parquet metadata
prism run -i data.csv -o output/ -m metadata.parquet

# Multiple metadata files (mixed formats)
prism run -i data.csv -o output/ -m metadata1.csv metadata2.parquet
```

**Implementation details:**

- Updated `merge_and_sort_streaming()` to handle both CSV and parquet inputs via DuckDB
- Updated `load_skyline_report()` to auto-detect and load parquet files
- Updated `load_sample_metadata()` and `load_sample_metadata_files()` to auto-detect and load parquet files
- Source fingerprinting works with parquet files for cache validation

### Output Files Changed from TSV to CSV

Changed the two TSV output files to CSV format for better compatibility with Excel and other tools:

- `protein_groups.tsv` -> `protein_groups.csv`
- `sample_metadata.tsv` -> `sample_metadata.csv`

**Rationale:**

- Excel auto-opens CSV files correctly with double-click
- TSV files often require manual import steps in Excel
- All PRISM input files are already CSV, so this provides consistency
- No data format changes - only the delimiter changed from tab to comma

**Updated output directory structure:**

```text
output_dir/
├── corrected_peptides.parquet
├── corrected_proteins.parquet
├── peptides_rollup.parquet
├── proteins_raw.parquet
├── merged_data.parquet
├── protein_groups.csv              # Changed from .tsv
├── sample_metadata.csv             # Changed from .tsv
├── metadata.json
├── qc_report.html
└── prism_run_YYYYMMDD_HHMMSS.log
```

### Robust Column Name Detection

Added `find_column()` helper function that handles space-separated, underscore-separated, and space-free (invariant) column name formats:

- **Problem**: Skyline exports column names differently depending on export format and locale:
  - CSV exports use spaces: `Fragment Ion`, `Protein Accession`
  - Some parquet exports use underscores: `Fragment_Ion`, `Protein_Accession`
  - Invariant-language parquet exports remove all spaces: `FragmentIon`, `ProteinAccession`
- **Solution**: `find_column()` tries four name variants in order:
  1. Exact match (e.g., `Fragment Ion`)
  2. Space-to-underscore variant (e.g., `Fragment_Ion`)
  3. Underscore-to-space variant (e.g., `Fragment Ion`)
  4. Space-removed variant (e.g., `FragmentIon`) — handles invariant export format
- **Columns affected**: All auto-detected columns now use this robust matching:
  - `peptide_col` (Peptide Modified Sequence, etc.)
  - `sample_col` (Sample ID, Replicate Name)
  - `abundance_col` (Area)
  - `transition_col` (Fragment Ion)
  - `protein_col` (Protein Accession)
  - `protein_name_col` (Protein)
  - `protein_gene_col` (Protein Gene)
  - `protein_description_col` (Protein)
  - `precursor_charge_col` (Precursor Charge) - used by transition rollup
  - `product_charge_col` (Product Charge) - used by transition rollup
  - `shape_corr_col` (Shape Correlation) - used by transition rollup
  - `rt_col` (Retention Time) - used by transition rollup
  - `mz_col` (Product Mz) - used by adaptive rollup
  - `batch_col` (Batch) - used for batch tracking

### Protection Against Common User Errors

Added validation to detect if user accidentally provides PRISM output files instead of Skyline reports:

- Detects if a parquet file is `merged_data.parquet` (PRISM output) instead of Skyline report
- Checks for PRISM-specific columns in the data
- Provides helpful error message directing users to provide the original Skyline export

Example error if user provides wrong file:

```text
Error: 'merged_data.parquet' appears to be a PRISM output file, not a Skyline report.
Please provide the original Skyline parquet report instead.
PRISM output files are generated in the output directory.
```

### Fixed: Parquet File Corruption During Sort

Fixed a critical bug that could cause parquet file corruption when processing large single parquet files:

- **Problem**: When sorting a large parquet file, the `_add_parquet_metadata()` function was reading the entire sorted file back into memory to add fingerprint metadata, then rewriting it. For files with 100M+ rows, this caused memory issues and file corruption.
- **Solution**: Now uses sidecar JSON files for fingerprints (via `_add_fingerprints_to_parquet()`), avoiding the need to read and rewrite the entire parquet file.
- **Removed**: The `_add_parquet_metadata()` function has been removed as it was unsafe for large files.

### Fixed: NaN Values in Fragment Ion Column

Fixed a crash when processing peptides with missing Fragment Ion values:

- **Problem**: When `Fragment Ion` column contained NaN values, the precursor exclusion filter (`str.startswith("precursor")`) would return NaN, and inverting with `~` caused a `TypeError: bad operand type for unary ~: 'NoneType'`.
- **Solution**: Added `.fillna(False)` to handle NaN values gracefully. Missing Fragment Ion values are now treated as non-precursor ions (kept in the data).

### Fixed: Duplicate Column Creation When Re-processing Parquet Files

Fixed a bug where re-running PRISM on already-merged parquet files would create duplicate columns:

- **Problem**: When processing a parquet file that already contained `Batch`, `Source Document`, and `Sample ID` columns (from a previous PRISM run), the code would add these columns again, creating `Batch_1`, `Source Document_1`, `Sample ID_1` duplicates.
- **Solution**: Check for existing columns before adding in three key functions:
  - `_sort_parquet_low_memory()` - Uses DuckDB DESCRIBE query to check existing columns
  - `_stream_parquet_with_columns()` - Checks PyArrow schema for existing columns
  - `merge_and_sort_streaming()` - Checks CSV/parquet header for existing columns
- **Files modified**: `skyline_prism/data_io.py`

### Fixed: CV Calculation Scale Issue in Rollup Comparison Report

Fixed incorrect CV values displayed in the rollup comparison report:

- **Problem**: The rollup comparison report showed dramatically different CVs than the main QC report (e.g., 105.9% vs 34% for sum method). This occurred because `peptide_df` was in LOG2 scale, but `compute_cv_linear()` expected LINEAR scale data.
- **Solution**: Read corrected peptides from `corrected_peptides.parquet` which is already in LINEAR scale, rather than using the internal `peptide_df` variable.
- **Files modified**: `skyline_prism/cli.py`

### Fixed: ZeroDivisionError on Empty Input Data

Fixed a crash when input data contains no peptides:

- **Problem**: Division by zero error when calculating average transitions per peptide (`n_transitions / n_peptides`) when `n_peptides = 0` due to empty or corrupted input files.
- **Solution**: Added guard to only perform division when `n_peptides > 0`, and added early exit with helpful error message if no data is found.
- **Files modified**: `skyline_prism/cli.py`

### Improved: Rollup Comparison Summary Plot Formatting

Improved the readability of summary plots in the rollup comparison report:

- **X-axis labels**: Reduced font size from 6 to 4 to prevent crowding with many samples
- **Stats box position**: Moved from upper-right to lower-right to avoid overlapping with bar charts
- **R-squared label**: Changed from "R²" to "Mean R²" to clarify that it represents the average across samples
- **Files modified**: `skyline_prism/rollup_comparison_report.py`

**Why parquet format matters:**

Skyline (version 24.1+) can now export reports directly to parquet format, which offers:

- **Faster I/O** - 2-10x faster reading than CSV
- **Better compression** - Smaller file sizes
- **Type preservation** - No string->number conversion issues
- **Memory efficient** - Columnar format uses less RAM

### Fixed: Out-of-Memory Error When Sorting Parquet Files on Network Storage

Fixed an `OutOfMemoryException` that occurred when sorting parquet files stored on NAS or network-mounted filesystems:

- **Root cause 1 - NAS temp directory**: DuckDB spills intermediate sort data to a temp directory. When the temp directory was on a NAS mount (e.g., `/mnt/UNAS-DataAnalysis/...`), DuckDB could not perform disk spilling, causing it to exhaust its memory budget.
- **Root cause 2 - Hardcoded memory cap**: The DuckDB memory limit was hardcoded to 4096 MB regardless of how much RAM was available. On machines with large RAM (e.g., 44 GB free), DuckDB was unnecessarily constrained.
- **Root cause 3 - Thread memory overhead**: Multiple sort threads each maintain independent memory buffers, multiplying peak memory usage.

**Fixes applied:**

1. **Local temp directory**: DuckDB now uses `/tmp/.duckdb_temp` (local disk) for spill files, falling back to the output directory only if `/tmp` is unavailable. This ensures disk spilling works on NAS-mounted output directories.
2. **Dynamic memory limit**: The memory limit is now set to 75% of available system RAM (read from `/proc/meminfo`) rather than a fixed 4096 MB cap. On a machine with 44 GB free, this gives DuckDB ~33 GB, allowing large files to be sorted entirely in memory with no spilling needed.
3. **Reduced thread count**: Thread count is limited to 2 during sorting to reduce per-thread buffer overhead.

- **Files modified**: `skyline_prism/data_io.py`

### Improved: Config Template Default Settings

Updated both config templates (`prism config-template` and `prism config-template --minimal`) to use the recommended analysis settings as defaults:

| Setting | Old default | New default |
| ------- | ----------- | ----------- |
| `transition_rollup.method` | `sum` | `library_assist` |
| `protein_rollup.method` | `sum` | `median_polish` |
| `batch_correction.enabled` | `false` (minimal) | `true` |

The `library_assist` transition rollup and `median_polish` protein rollup are the methods that produce the best quantification precision in typical LC-MS/MS experiments. `batch_correction` is now enabled by default and automatically skipped when only one batch is detected.

Both templates now include prominent `<<< EDIT REQUIRED` markers on the two paths users must provide before running:
- `transition_rollup.library_assist.library_path` — spectral library (`.blib` or `.tsv`)
- `parsimony.fasta_path` — FASTA database used for the original search

- **Files modified**: `skyline_prism/cli.py`

## Testing

- All tests passing (328 tests)
- Added 6 new tests for `find_column()` function
- Added tests for rollup comparison functionality
- Added 3 new tests for duplicate column prevention when re-processing parquet files
- Tested with real Skyline parquet exports (117M rows)
- Verified handling of mixed metadata file formats

## Upgrade Notes

- **Breaking change for scripts**: If you have scripts that read `protein_groups.tsv` or `sample_metadata.tsv`, update them to read the `.csv` versions instead
- Existing workflows using CSV/TSV input files continue to work unchanged
- New parquet support is opt-in (use `.parquet` files when available)
- The `--from-provenance` feature will still work, but will produce CSV outputs even when re-running from older provenance files
