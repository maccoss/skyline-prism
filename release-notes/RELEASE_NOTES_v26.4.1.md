# Skyline-PRISM v26.4.1 Release Notes

Patch release: fixes `ZSTD decompression failed: Unknown frame descriptor` when reading the Stage 2 streaming peptide output written by the v26.3.5 streaming-flush refactor.

## New Features

<!-- none -->

## Bug Fixes

### Fixed: `ZSTD decompression failed: Unknown frame descriptor` reading `peptides_rollup.parquet`

The streaming-flush writers introduced in v26.3.5 in `skyline_prism/chunked_processing.py` used `pq.ParquetWriter(..., compression="zstd")` with many sequential `write_table` calls (one row group per ~2000-peptide flush, ~84 row groups for a 168k-peptide run). On NAS-mounted output directories this pattern reproducibly produced files whose zstd frames could not be decoded by pyarrow's reader at Stage 2b. Reported in production on a 117M-row, 168k-peptide, 72-sample run writing to a NAS path; Stage 2 logged "Wrote peptide abundances" successfully, but the immediate follow-up `pd.read_parquet()` raised `OSError: ZSTD decompression failed: Unknown frame descriptor`.

This is the same failure class hit on Stage 1 zstd parquet writes earlier in this release cycle; the resolution there (and in v26.4.0 for the merged-data file via DuckDB write with explicit ROW_GROUP_SIZE) was already in place, but the v26.3.5 streaming-flush refactor introduced the bad pattern in `chunked_processing.py` and that path was not covered by the Stage 1 fix.

Fix: switched all three streaming-flush writer sites in `chunked_processing.py` from zstd to snappy compression:

- `peptides_rollup.parquet` (line ~1041)
- `peptides_rollup_residuals.parquet` (line ~1056; only written when `save_residuals=True`)
- Consensus diagnostics intermediate parquet (line ~1070; converted to CSV and deleted at end of stage)

Notes:

- This brings these intermediates in line with the codebase's *final* user-facing outputs `corrected_peptides.parquet` and `corrected_proteins.parquet`, which have always been snappy (pandas `to_parquet` default; no `compression` argument specified at `cli.py:2245` / `cli.py:2580`).
- Snappy and zstd are both standard Parquet codecs; every reader (pandas, pyarrow, DuckDB, polars, R `arrow`, Spark, ...) handles them transparently because the codec is recorded per column chunk inside each file's metadata.
- Storage overhead is small: snappy compresses to ~50-60% of uncompressed where zstd default reaches ~25-35%, so the on-disk peptide rollup is roughly 1.5-2.5x larger than it would have been. For a 168k-peptide run with 72 samples this is ~25 MB additional on disk. The much larger `merged_data.parquet` (transition-level Stage 1 output) stays zstd via the DuckDB-write path established in v26.4.0.
- Single-shot zstd writes elsewhere in the codebase (the protein-level outputs at `chunked_processing.py:1606/1615`, and the various intermediates in `data_io.py`) are left alone — they use a different write pattern (all-at-once rather than many sequential `write_table` calls) and have not exhibited this failure mode.

- **Files modified**: `skyline_prism/chunked_processing.py`

## Performance

<!-- none -->

## Breaking Changes

<!-- none yet -->
