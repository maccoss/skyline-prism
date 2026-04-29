# Skyline-PRISM v26.3.4 Release Notes

Patch release: redesigned multi-parquet Stage 1 (merge + sort) for reliable scaling to 100s of input files, hardened Stage 2 reader against multi-GB zstd parquet read failures, ported the protein parsimony razor algorithm to match Osprey's deterministic iterative-greedy spec, and fixed a NAS-mount `FileExistsError` when the output directory already exists.

## New Features

### Protein parsimony aligned with Osprey's documented algorithm

The razor-peptide assignment in `skyline_prism/parsimony.py` is now byte-for-byte compatible with Osprey's iterative-greedy set cover (described in `osprey/docs/16-protein-parsimony.md`). Three things changed:

- **Razor tiebreaker.** The previous score `(n_unique, -ord(can[0]), n_remaining)` only differentiated by the first character of the protein accession and then preferred groups with more shared peptides remaining; that was not in the spec and produced different assignments than Osprey on tie cases (most notably for cohorts where most accessions share a first letter, e.g., all UniProt `P*` IDs). The new tiebreaker is `(n_unique_count, Reverse(group_id))` — when unique counts tie, the lexicographically smallest canonical protein string wins.
- **Subsuming-protein choice.** `_find_subsumable_proteins()` previously scanned in dict-insertion order with an early `break`, so the recorded subsumer for a subset protein depended on iteration order. The function now sorts the protein list and picks the lexicographically smallest valid superset, matching the "lowest group ID wins" convention.
- **Determinism.** Shared peptides are now collected in sorted order before the greedy loop starts; canonical proteins are iterated in sorted order; claimed peptides per round are sorted alphabetically before being added to a razor set. The final result is byte-identical across repeated runs regardless of HashMap or set iteration order.

Four regression tests in `tests/test_parsimony.py` lock in the new behavior: cascading two-round razor assignment (Osprey Example 2), tiebreaker on lowest group ID, repeated-run determinism, and lexicographically-smallest subsumer.

PRISM's three shared-peptide modes (`all_groups`, `unique_only`, `razor`) continue to be applied at the rollup layer rather than as a parameter on parsimony itself; the parsimony stage always produces the unique + razor split, and `rollup.py` / `chunked_processing.py` choose which peptides to use for protein quantification.

- **Files modified**: `skyline_prism/parsimony.py`, `tests/test_parsimony.py`

## Bug Fixes

### Fixed: `FileExistsError` creating output directory on NAS / network mounts

`prism run` failed with a cryptic `FileExistsError: [Errno 17] File exists: '<output_dir>'` when the output directory was an existing, usable directory on a NAS mount (e.g., `/mnt/nas/...`). The `Path.mkdir(parents=True, exist_ok=True)` call documents itself as a no-op when the directory already exists, but its internal "is this a directory?" probe relies on `Path.is_dir()`, which misreports `False` on some NAS / CIFS / SMB directory entries. `os.path.isdir()` uses the same underlying syscall and has the same failure mode.

The fix uses `os.listdir()` as the directory-usability test: it actually opens and reads directory entries (a different syscall path), so if the directory is usable for our purposes the check succeeds. If the path really is a regular file or a broken mount entry, the new error message names the path explicitly: `FileExistsError: Output path exists but is not a usable directory: <path>`.

- **Files modified**: `skyline_prism/cli.py`

### Redesigned Stage 1 (parquet merge + sort) and hardened Stage 2 reader

Stage 1 (`merge_and_sort_streaming` in `skyline_prism/data_io.py`) and the Stage 2 chunked rollup reader had accumulated a series of fixes for symptoms — `OutOfMemoryException` during the sort, then `TProtocolException: Invalid data, Deserializing page header failed` when downstream code tried to read the merged file with `pq.ParquetFile.read_row_group(i, columns=...)`. Both paths are replaced with one coherent design:

**Stage 1: two-stage merge-then-sort.** A single-pass `COPY (UNION ALL of read_parquet ... ORDER BY peptide) TO parquet` query OOMs reproducibly on multi-file parquet inputs, even when given tens of GB of memory: DuckDB's parallel parquet reader + sort + writer pipeline cannot spill cleanly enough to stay inside any reasonable cap. The fix splits the work into two stages, each with bounded memory:

- **Stage A (pyarrow streaming concat).** Each input parquet is read sequentially, one row group at a time, with threaded decompression. Metadata columns (`Batch`, `Source Document`, `Sample ID`) are added only when missing, with vectorized null-propagating Sample ID construction via `pyarrow.compute.binary_join_element_wise(..., null_handling="emit_null")`. The output is one unsorted intermediate parquet (`<output>.unsorted.parquet`, snappy-compressed for writer-thread speed; deleted after Stage B). Memory stays bounded by one row group at a time, regardless of input file count or total size, so this scales cleanly to 100s of parquet files.
- **Stage B (DuckDB external sort, zstd output).** A single-source `COPY (SELECT * FROM read_parquet(unsorted) ORDER BY peptide) TO output (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)`. With `preserve_insertion_order=false` and `temp_directory='/tmp/.duckdb_temp'`, DuckDB's external k-way merge spills to local disk when memory pressure builds, staying within the configured 8 GB cap. Sorting one file is the proven low-memory sort path. Output is zstd-compressed with explicit `ROW_GROUP_SIZE 1000000` so the downstream chunked rollup can read it efficiently.

CSV/mixed-input cohorts continue to use the existing DuckDB UNION ALL path for Stage A; they are typically smaller datasets and the parallel-reader memory pressure is not the same.

Properties:

- **zstd compression** on the persisted output (the user's stated requirement; `merged_data.parquet` is cached and reused).
- **multithreaded where it helps**: DuckDB Stage B uses default thread count for the sort and zstd write. Stage A's pyarrow concat is sequential file-by-file but uses threaded decompression within each row group.
- **streaming with bounded chunks**: Stage A bounds memory by row group, Stage B by configured DuckDB cap (8 GB). Neither stage needs the full dataset in memory.
- **scales to 100s of parquet files**: Stage A processes inputs sequentially; adding more files extends total time linearly without raising peak memory.
- **explicit `ROW_GROUP_SIZE 1000000`** keeps the row-group count proportional to dataset size (~167 row groups for a 167M-row sort) rather than producing thousands of tiny groups.

The single-file shortcut `_process_single_parquet` is unchanged.

Schemas are pre-validated in Python before any SQL runs: if a later file is missing data columns present in the first file, a clear `ValueError` names the missing columns rather than letting DuckDB raise a `BinderException`. The check ignores the synthesized metadata columns (`Batch`, `Source Document`, `Sample ID`), which are intentionally allowed to differ across files.

**Stage 2: hardened pyarrow reader.** The `pq.ParquetFile` defaults (`pre_buffer=True`, `memory_map=True`) prefetch column data and use mmap. On multi-GB zstd parquet files that are still hot in the kernel page cache after a recent write, that combination has produced `TProtocolException: Invalid data, Deserializing page header failed` on column-pruned `read_row_group(i, columns=cols)` calls. All five `pq.ParquetFile(...)` open sites in `skyline_prism/chunked_processing.py` (lines 518, 543, 945, 1341, 1537) now pass `pre_buffer=False, memory_map=False`, which forces simple synchronous IO that round-trips cleanly through this scale of file. The `read_row_group` call in the chunked rollup loop also passes `use_threads=True` explicitly so column decompression remains multithreaded.

- **Files modified**: `skyline_prism/data_io.py`, `skyline_prism/chunked_processing.py`

## Tests

### Regression Tests for Multi-Parquet Stage 1 + Stage 2

`tests/test_data_io.py::TestMultiParquetStreamingMerge` covers the scenarios that allowed the regressions to slip in:

- **`test_merge_dozen_parquet_files_scales`**: merges 12 small parquet files end-to-end and validates total row count, per-batch sample lists, and absence of duplicate metadata columns. Headline guard against the "dozens of parquet files" regression.
- **`test_merge_parquet_output_is_sorted_by_peptide`**: writes deliberately unordered peptide names across two files and verifies the merged output is globally sorted.
- **`test_merge_parquet_with_null_replicate_names`**: verifies null replicate names propagate to null Sample IDs (vectorized null handling parity with the previous Python comprehension).
- **`test_merge_parquet_with_multiple_row_groups`**: single file written with a small row group size so iteration crosses many row group boundaries.
- **`test_merge_parquet_schema_mismatch_raises`**: confirms a clear `ValueError` when a later file is missing data columns present in the first file. The check ignores synthesized metadata columns so files-with-metadata mixed with files-without still merge cleanly.
- **`test_merge_parquet_cleans_up_unsorted_intermediate`**: confirms the `<output>.unsorted.parquet` Stage A intermediate is deleted on success so subsequent runs see a clean output directory.
- **`test_merge_parquet_with_existing_metadata_across_files`**: mixes files with and without pre-existing `Batch` / `Source Document` / `Sample ID` columns; pre-existing values are preserved and no duplicate columns are produced.
- **`test_merged_output_readable_via_read_row_group`**: opens the merged output with `pq.ParquetFile` and reads every row group via `read_row_group(i, columns=cols_to_read)` — the exact API used by `chunked_processing.rollup_transitions_sorted`.
- **`test_merged_zstd_output_is_read_row_group_safe`**: end-to-end check that the persisted output uses zstd compression, produces multiple row groups, and is readable via the hardened `pq.ParquetFile(..., pre_buffer=False, memory_map=False)` open with column-pruned `read_row_group(i, columns=cols, use_threads=True)`. Locks in the read-side hardening alongside the writer codec choice.

- **Files modified**: `tests/test_data_io.py`

## Documentation

- Added a `release-notes/README.md` adapted from the Osprey convention. Documents the `YY.feature.patch` versioning scheme, the `RELEASE_NOTES_next.md` development convention, content structure, and release process.
- Added a brief note to `docs/output_files.md` describing the on-disk properties of `merged_data.parquet` (zstd compression, ~1M-row groups) and the transient `<output>.unsorted.parquet` intermediate that appears during a Stage 1 run.

## Performance

The two-stage design adds one extra parquet write (the unsorted intermediate) compared to a hypothetical single-pass query, but it is the path that actually completes on production-scale inputs without OOM. Within that constraint:

- **Stage A reads use threaded row-group decompression** (`use_threads=True` on each `read_row_group`).
- **Stage B uses DuckDB's default thread count** for the sort and the zstd write.
- **`ROW_GROUP_SIZE 1000000`** in the Stage B output produces a parquet layout that the chunked rollup reads efficiently (~167 row groups for a 167M-row sort, instead of thousands of tiny groups).
- **Snappy on the unsorted intermediate** keeps Stage A's writer thread fast; the file is deleted right after Stage B so codec choice does not affect the persisted output.

## Breaking Changes

<!-- none yet -->
