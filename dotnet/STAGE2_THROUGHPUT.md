# Stage 2 reader throughput

Status: unreleased, on top of dotnet-v26.12.0. Stage 2 (transition -> peptide rollup) is bounded in memory and no longer
the pathological case it was, but it is **single-threaded, and will stay that way until the DuckDB
binding can be read from concurrently**. This records what was measured, so the next person does not
re-derive it.

## What changed since v26.12.0

The producer was measured before it was touched: **68% of its time was managed work, 32% DuckDB**, so
the ceiling was never the database. Almost all of the managed cost was `GetString` materializing a
fragment ion, two charges and a ~45-character sample id on *every transition row* - which the string
pool then deduplicated and threw away.

So the reader stopped asking for strings. The transition id is composed in SQL, the precursor test is a
SQL boolean, and the sample name is resolved to its **index** by an inner join against the run's sample
list (the index IS the output column, so the lookup was redundant work).

Measured end to end on a 2-plate cohort (186M rows, 192 samples), against the v26.12.0 binary on the
same inputs and machine: **3.33 min -> 2.49 min, peak 2.75 GB -> 2.40 GB.** `peptides_rollup.parquet`,
`proteins_raw.parquet` and `corrected_proteins.parquet` bit-identical; `corrected_peptides` equal to
1e-16 (summation order); all 662,339 `peptide_residuals` transition ids identical in both directions.

Per-partition detail, and the two shapes that were tried and rejected:

| shape (15.5M-row partition) | sec | Mrow/s | allocated |
|---|---|---|---|
| drain rows, touch no column (the DuckDB floor) | 1.9 | 8.37 | 0 GB |
| as shipped in v26.12.0 | 5.0 | 3.12 | 4.48 GB |
| typed accessors only, no boxing | 5.3 | 2.91 | 3.09 GB |
| **sample index + transition id in SQL (shipped)** | **3.3** | **4.71** | **1.48 GB** |
| pep/tid as `dense_rank()` ids, zero allocation | 6.3 | 2.47 | 0.00 GB |

Two results worth keeping: the **allocation-free** shape was the **slowest** (two window functions cost
more than the strings they removed), and typed accessors alone bought almost nothing, because boxing
was never the expensive part. Neither is what the reasoning predicted, which is the argument for
measuring the next one too.

## Parallel reading: ruled out, not deferred

Reading partitions concurrently is legal in the data model - a peptide lives in exactly one partition,
so slices share nothing. It **segfaults in every configuration tried**:

| configuration | result |
|---|---|
| connection per partition, concurrent, shared `:memory:` | `AccessViolationException`, ~2 runs in 3 |
| all connections opened up front, none closed mid-stage, keepalive pinning the refcount | segfault |
| genuinely isolated **file-backed** databases, one per reader | segfault |
| DuckDB.NET **1.5.5** rather than 1.5.3 | segfault |

The first result suggested the cause was `Connection.ConnectionManager`'s refcounted instance cache
tearing an instance down under a live reader. The second and third rule that out: it fails with no
teardown possible, and with no shared instance at all. **Concurrent streaming readers are unsafe in
this binding**, and the current release does not fix it.

This is a segfault, not an exception - it cannot be caught, retried or contained. In the stage that
computes the reported quantities, the same corruption could produce wrong numbers as easily as a crash.
That is why it is not worked around.

## What is actually left

**1. Sidestep the row API - MEASURED, and it wins.** Have DuckDB `COPY` a partition's narrow
projection to a temp parquet, then read that with Parquet.Net a row group at a time (whole column
vectors, no per-cell marshalling). On one 15.5M-row partition:

| path | sec | Mrow/s |
|---|---|---|
| A: DuckDB streaming read (what ships today) | 4.7 | 3.30 |
| B1: `COPY` narrow projection to parquet | 2.7 | 5.74 |
| B2: Parquet.Net row-group read + block assembly | **1.6** | **9.57** |

B is already slightly cheaper than A end to end (4.3 s against 4.7 s) *before* any parallelism - note
that B1, which sorts AND writes to disk, beats A, which sorts and streams: the row-by-row marshalling
really is that expensive. The prize is B2: **2.9x faster than A, and pure managed code**, so it can run
on several threads with no concurrent DuckDB anywhere.

Sketch: phase 1 `COPY`s partitions one at a time (DuckDB parallelises the sort internally), phase 2
reads the temp files on K threads into the existing `BlockingCollection`. The two phases pipeline -
partition k can be read while k+1 is being written - so the floor is roughly the phase-1 cost, about
1.7x better than today on the read side. Temp files are deleted as they are consumed, so the extra disk
is one partition's narrow projection at a time, not a second copy of the cohort.

Not yet implemented; it is a real change to Stage 2's structure and deserves its own PR and its own
end-to-end verification.

**2. Re-tune partition sizing - MEASURED, no change warranted.** Full pipeline on the 2-plate cohort,
varying `processing.merge_memory_mb` (which is what sets partition size):

| budget | partitions | total | Stage 1 | Stage 2 | peak |
|---|---|---|---|---|---|
| 2048 MB | 16 | 2.46 min | 47.7 s | 1m 26s | 2.18 GB |
| **8192 MB (current default)** | **12** | **2.33 min** | 44.8 s | 1m 21s | 2.37 GB |
| 32768 MB | 3 | 2.49 min | 46.5 s | 1m 29s | 6.42 GB |

The current setting is already the optimum, and large partitions now cost 2.7x the peak memory for
slightly worse wall clock. The faster reader did not move the trade enough to justify changing it.
Re-measure after item 1, which changes the reader's cost model again. (This is a 2-document cohort; the
earlier catastrophic case - 66M-row partitions taking the rollup from 26 to 62 minutes - was at 20
documents, so the penalty for over-large partitions grows with the cohort, not shrinks.)

**3. Watch upstream - nothing to do today.** DuckDB.NET 1.5.5 is the latest published version and is
already known to segfault (see above). Re-run the concurrency harness against each new release. If one
ever passes, parallel readers are a small change: slice `dataset.Partitions`, one
`StreamPeptideBlocks` per slice into the existing `BlockingCollection`, each with its own share of the
budget. Do not ship it on one green run.

## Verification bar (learned the hard way)

- **Measure end to end, never one stage.** A change that improved Stage 1 by 2.7x regressed the
  pipeline by 11 minutes.
- **Compare against the previous release's binary on the same inputs**, not against remembered numbers.
  Use a git worktree at the release tag; it takes two minutes and removes all doubt.
- **Exercise the paths the default config skips.** The transition id only appears with
  `median_polish` + `output.include_residuals`; a default run would have "verified" a change to it
  without ever producing one.
- **Run the machine quiet.** Two measurement rounds here were invalidated by a concurrent experiment
  competing for the same disk, and one "new binary" run silently used a stale build because a leftover
  process held the DLL.
- **Repeat anything touching concurrency.** The crash reproduced 2 runs in 3, not 3 in 3.

## Instrumentation

The run log now reports per-stage elapsed time and a sorted summary at the end, so the next person can
see where a real cohort spends its time without wrapping the process in an external sampler - which is
how every number above had to be obtained.
