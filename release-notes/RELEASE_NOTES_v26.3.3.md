# Skyline-PRISM v26.3.3 Release Notes

## Overview

Patch release fixing parquet sort failures on network storage, improving column name detection for invariant-language Skyline exports, and updating config templates to use recommended analysis defaults.

## Changes

### Fixed: Out-of-Memory Error When Sorting Parquet Files on Network Storage

Fixed an `OutOfMemoryException` that occurred when sorting parquet files stored on NAS or network-mounted filesystems:

- **Root cause 1 - NAS temp directory**: DuckDB spills intermediate sort data to a temp directory. When the temp directory was on a NAS mount (e.g., `/mnt/UNAS-DataAnalysis/...`), DuckDB could not perform disk spilling, causing it to exhaust its memory budget and crash at ~47% progress.
- **Root cause 2 - Hardcoded memory cap**: The DuckDB memory limit was hardcoded to 4096 MB. On machines with large RAM (e.g., 44 GB free), this unnecessarily constrained DuckDB to a fraction of available memory.
- **Root cause 3 - Thread memory overhead**: Multiple sort threads each maintain independent memory buffers, multiplying peak memory usage.

**Fixes applied:**

1. **Local temp directory**: DuckDB now uses `/tmp/.duckdb_temp` (local disk) for spill files, falling back to the output directory only if `/tmp` is unavailable. This ensures disk spilling works reliably on NAS-mounted output directories.
2. **Dynamic memory limit**: The memory limit is now set to 75% of available system RAM (read from `/proc/meminfo`, fallback 2 GB) rather than a fixed 4096 MB cap. On a machine with 44 GB free, this gives DuckDB ~33 GB, allowing typical files to be sorted entirely in memory.
3. **Reduced thread count**: Thread count is limited to 2 during sorting to reduce per-thread buffer overhead.

- **Files modified**: `skyline_prism/data_io.py`

### Fixed: Column Detection for Invariant-Language Parquet Exports

Extended `find_column()` to handle Skyline parquet exports generated with invariant locale settings, where all spaces are removed from column names:

- **Problem**: Skyline parquet exports created with invariant language settings remove all spaces from column names (e.g., `FragmentIon` instead of `Fragment Ion`, `ProteinAccession` instead of `Protein Accession`). The existing `find_column()` only handled space-to-underscore substitution, missing this format entirely.
- **Solution**: Added a fourth variant: spaces completely removed (e.g., `"Fragment Ion"` → `"FragmentIon"`). `find_column()` now tries four variants in order:
  1. Exact match (e.g., `Fragment Ion`)
  2. Space-to-underscore (e.g., `Fragment_Ion`)
  3. Underscore-to-space (e.g., `Fragment Ion`)
  4. Space-removed (e.g., `FragmentIon`) — new, handles invariant export format

Also fixed two places in the transition rollup adaptive weight learning code that used hardcoded column name strings (`"Product Mz"`, `"Shape Correlation"`, `"Product Charge"`, `"Precursor Charge"`, `"Replicate Name"`) instead of the already-resolved column name variables. These would silently drop the columns when the invariant format was used, preventing proper adaptive weight learning.

- **Files modified**: `skyline_prism/cli.py`

### Improved: Config Template Default Settings

Updated both config templates (`prism config-template` and `prism config-template --minimal`) to use recommended analysis settings as defaults:

| Setting | Old default | New default |
| ------- | ----------- | ----------- |
| `transition_rollup.method` | `sum` | `library_assist` |
| `protein_rollup.method` | `sum` | `median_polish` |
| `batch_correction.enabled` | `false` (minimal only) | `true` |

The `library_assist` transition rollup and `median_polish` protein rollup are the methods that produce the best quantification precision in typical LC-MS/MS experiments. `batch_correction` is now enabled by default in both templates and automatically skipped when only one batch is detected.

Both templates now include prominent `<<< EDIT REQUIRED` markers and header callouts listing the two paths users must provide before running:
- `transition_rollup.library_assist.library_path` — spectral library (`.blib` or `.tsv`)
- `parsimony.fasta_path` — FASTA database used for the original search

- **Files modified**: `skyline_prism/cli.py`

### Improved: README Documentation

Added missing documentation for parquet input format and multiple input files:

- New **Input File Formats** section explaining parquet/CSV/TSV support, invariant column name handling, and multiple input files
- Updated Quick Start examples to show parquet as the recommended format
- Expanded **Merge multiple Skyline reports** section with parquet and mixed-format examples
- Updated pipeline diagram to reflect that Stage 1 accepts CSV, TSV, and parquet

- **Files modified**: `README.md`

## Testing

- All tests passing (328 tests)
- Tested fixes with real Skyline invariant-format parquet exports (48 samples, 4.4M rows)
- Verified DuckDB sort succeeds on NAS-mounted directories

## Upgrade Notes

- No breaking changes
- Existing workflows using CSV/TSV or standard parquet inputs continue to work unchanged
- The new config template defaults (`library_assist`, `median_polish`, `combat`) only affect newly generated templates; existing config files are not changed
