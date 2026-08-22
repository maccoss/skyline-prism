# Skyline-PRISM (C#) dotnet-v26.13.0

A throughput release for the transition rollup. The stage that dominates a run reads about 1.5x faster
and the whole pipeline about 25% faster, with lower peak memory - and the run log now tells you where a
slow cohort spent its time, which previously could only be measured from outside the process.

## New Features

- **The run log now reports what each stage cost, not just what it did.** Every stage prints its elapsed
  time, followed by a sorted summary with percentages. Previously a stage that doubled in time was
  invisible unless someone wall-clocked the whole run from outside - every performance number in the
  previous release had to be obtained that way - so a slow cohort can now be diagnosed from the run log
  a user already has.

## Bug Fixes

- **A dataset with no detected samples now fails with an explanation** instead of a SQL syntax error
  raised from deep inside the transition rollup. It names the likely cause (sample-column detection) and
  where to check it.

## Performance

- **The transition rollup reads ~1.5x faster, and the whole pipeline ~25% faster.** The rollup's
  producer was measured before being changed: 68% of its time was managed work rather than DuckDB, and
  almost all of that was materializing a fragment ion, two charges and a ~45-character sample id as
  strings on *every transition row* - which the string pool then deduplicated and discarded. Those now
  come out of SQL: the transition id is composed there, the precursor test is a boolean, and the sample
  name is resolved to its index by a join (that index is the output column, so the per-row name lookup
  was redundant work). On a 2-plate cohort (186M transition rows, 192 samples), measured against the
  previous release's binary on the same inputs: **3.33 min -> 2.49 min, peak 2.75 GB -> 2.40 GB**, with
  `peptides_rollup.parquet`, `proteins_raw.parquet` and `corrected_proteins.parquet` bit-identical and
  all 662,339 `peptide_residuals` transition ids unchanged.

- **The transition rollup stays single-threaded, and that is now a measured result rather than an open
  question.** It is the largest stage of a run (~58% of wall clock on a 2-plate cohort), so two ways of
  replacing its reader were built and benchmarked against the shipping one on a real 11.4M-row
  partition: skipping the sort and grouping rows in memory, and writing a narrow projection to a temp
  parquet to read back in bulk. Grouping without a sort measured **2.8x slower** and used 2.3x the
  memory; the two-phase read tied with what ships. Neither justified the change, so the rollup is
  unchanged - but large-cohort users can now take its single-threadedness as a known ceiling rather
  than an oversight. `dotnet/STAGE2_THROUGHPUT.md` records the numbers.
