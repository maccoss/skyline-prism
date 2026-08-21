# Skyline-PRISM (C#) dotnet-v26.12.0

A memory and throughput release. PRISM could exhaust a workstation's RAM on an ordinary two-plate
cohort - taking it from the Skyline instance holding the very documents being processed, until the
machine swapped. Every stage is now bounded, and the cost of the merge and the rollup no longer grows
with the number of documents merged.

**On a 2-plate cohort (186M transition rows, 192 samples): ~50 min and a ~38 GB peak became 3.2 min and
2.74 GB**, measured over three consecutive runs with identical results. `peptides_rollup.parquet` is
bit-identical to the previous release's and `protein_groups.csv` is byte-identical - the same answers,
arrived at differently.

> [!IMPORTANT]
> `merged_data.parquet` is now the directory `merged_data/`. See **Breaking Changes** below. Output
> directories written by earlier releases still open unchanged in the QC report and the Spectrum
> density tab.

## New Features

- **The merged data is now partitioned by peptide, so cohort size no longer sets peak memory.** Stage 1
  hashes every row into a `_pep_bucket` on the peptide column and writes `merged_data/` as a
  hive-partitioned parquet directory, sized so one partition sorts inside a fraction of the memory
  budget (~16.8M rows at the usual 8 GB), so the count grows with the cohort while each partition stays
  the same. Because all of a peptide's rows land in one bucket, the
  transition rollup sorts and streams one partition at a time instead of ordering the whole cohort.
  That was the last operator whose cost scaled with the number of documents merged: on a 100-document
  cohort (~9 billion transition rows) the old global sort would have had to spill roughly 600 GB and
  re-read it during the external merge. The partition count is capped at 256; beyond that a partition's
  sort simply spills a little, which is bounded and sequential.

## Bug Fixes

- **The transition rollup no longer loads the whole cohort into memory.** `MergedParquetReader` is the
  producer that feeds Stage 2 one peptide at a time, but the DuckDB command it read through was left at
  the client library's default, `UseStreamingMode = false` - which runs the query to completion and
  materializes *every* row before handing back the first one. The result set is client-side, so neither
  `memory_limit` nor the spill directory applied to it. On a 186-million-row cohort that was ~38 GB of
  committed memory, most of it paged to disk.
- **Every DuckDB connection is now bounded and can spill.** The rollup, parsimony and sample-list
  connections were opened at DuckDB's defaults - a buffer pool of 80% of physical RAM and, with no
  `temp_directory`, no way to spill when they hit it. They now share the merge's budget and scratch
  directory.
- **The partitioned write is bounded too, by capping its threads.** DuckDB buffers rows per writing
  thread outside the spillable buffer pool, so its footprint is threads x flush-threshold: at the
  defaults on a 32-core machine that is 16.8M rows in flight, which exhausted an 8 GB budget in seconds
  and died with "failed to pin block" rather than spilling. The merge now writes with at most 8
  threads. That is faster as well as lighter - every thread keeps a buffer per partition and rows are
  hash-assigned, so 32 threads across 155 partitions means ~5,000 live buffers flushed in slivers. On
  186M rows into 16 partitions, 32 threads -> 8 took the write from **1.62 min / 7.9 GB to
  0.84 min / 2.4 GB**.
- **Repeated strings in a peptide block are pooled.** The rollup's reader allocated a fresh string per
  transition row for the sample id, fragment ion and both charges - all drawn from tiny domains, but
  the sample id alone is a 45-character `replicate__@__document`. On a 100-document cohort that is
  ~1 GB of live strings across the in-flight blocks; pooled, it is a few MB.
- **The output write buffer no longer grows with the cohort.** `processing.peptide_batch_size` counts
  peptides, but the buffer it sizes is peptides x samples doubles - so a value chosen for two documents
  became hundreds of MB at a hundred. The effective batch is now capped to keep that product flat;
  small cohorts are unaffected.
- **The wide feature x sample matrices are read a column at a time.** The Stage 2a density diagnostic,
  the protein rollup and the QC report each loaded the whole peptide table with `ParquetTable.Load`,
  which materializes every sample column as a nullable `double?[]` (16 bytes per cell) and then copies
  it into a `double[,]` (8 more) - both live at once, ~24 bytes per cell. They now use the bounded
  column reader, so the same work costs 8. On a 100-document cohort that is ~17 GB against ~6 GB for
  the QC matrix alone.

## Performance

- **A 2-plate cohort (186M transition rows, 192 samples) went from ~50 min and a ~38 GB peak to
  3.2 min and 2.74 GB** - measured over three consecutive runs, identical results each time.
  `peptides_rollup.parquet` is bit-identical to the previous release's and `protein_groups.csv` is
  byte-identical, so this is the same answer arrived at differently.
- **Stage 1 stopped sorting the cohort twice.** The merge used to close with `ORDER BY <peptide>` over
  the whole union - a blocking sort across all ~26 columns of a transition report, most of them wide
  repeated strings. Nothing needed it: the transition rollup gets its grouping from its own `ORDER BY`
  on the 8 narrow columns it actually reads, so the rows were sorted twice and the redundant one was
  the expensive one. On that cohort the old wide sort alone took 42 min, peaked at ~35 GB and spilled
  31 GB; the merge now streams in one pass and spills nothing. Dropping it also removes the unsorted
  intermediate - a full extra write and read of the entire dataset.
- **The .NET garbage collector now reclaims instead of growing.** DuckDB's buffer pool was not the only
  thing helping itself to the machine: PRISM allocates a few enormous, short-lived feature x sample
  matrices per stage, and with nothing configured the runtime services those by committing more memory
  rather than collecting - memory that, on the workstation PRISM normally runs on, the Skyline instance
  holding these documents needs. `System.GC.ConserveMemory` is now set. On the 20-document cohort the
  QC stage went from a **22.9 GB working set to 6.5 GB at identical wall-clock** (4.1 min either way).
  A heap hard limit was tried and rejected: it risks an `OutOfMemoryException` on a cohort larger than
  any measured, and it redefines the runtime's reported "total memory" - which PRISM reads to size
  DuckDB, whose pool is native and unaffected by it.
- **The QC report's PCA no longer materializes the matrix twice.** It built a list of per-feature arrays
  and then a dense copy of that, purely to produce a Gram matrix of samples x samples - a few tens of
  MB from 5.7 GB built twice, on a 100-document cohort. The Gram is now accumulated in feature blocks,
  holding ~150 MB at a time, and the transpose each call used to make first is gone (the standardizing
  loop reads features, which is the untransposed layout). Scores are bit-identical.
- **The automatic memory budget is no longer a land grab.** It was 75% of total RAM bounded by 80% of
  free; with nothing in the merge scaling with cohort size any more, it is 25% of total, bounded by 50%
  of free, capped at 8 GB. The old value was taken from whatever else was running - typically the
  Skyline instance PRISM was launched from, holding the very documents being processed, which Windows
  then paged out. `processing.merge_memory_mb` still overrides it, and now bounds the transition
  rollup's reader as well as the merge.
- Smaller parquet row groups in the merge output (1,000,000 -> 122,880 rows). DuckDB buffers a whole row
  group per writing thread, so the old value cost gigabytes of write buffer on a many-core machine; the
  smaller groups also give the per-run density queries finer row-group skipping.

## Breaking Changes

- **`merged_data.parquet` is now the directory `merged_data/`**, hive-partitioned into
  `_pep_bucket=N/*.parquet`, and its rows are **no longer sorted by peptide**. Nothing inside PRISM
  depended on either property - every consumer aggregates or issues its own `ORDER BY` - and both the
  QC report and the Spectrum density tab still open output directories written by earlier releases.
  Code outside the pipeline should read the glob, and sort for itself if it needs an order:

  ```sql
  SELECT * FROM read_parquet('output/merged_data/**/*.parquet', hive_partitioning=false);
  ```

  `prism merge -o out.parquet` likewise now produces a directory; the `.parquet` suffix is dropped
  rather than used to name a directory, and the command prints where the data actually went. The
  Python engine is unchanged - it still writes a single sorted file, because its rollup consumes the
  file in order.
