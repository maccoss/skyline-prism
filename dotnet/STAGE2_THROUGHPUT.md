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

**1. Watch upstream.** Re-run the concurrency harness (three configurations, several iterations each,
outputs compared to a serial baseline) against each new DuckDB.NET. If it ever passes, parallel readers
are a small change: slice `dataset.Partitions`, one `StreamPeptideBlocks` per slice into the existing
`BlockingCollection`, each with its own share of the budget. Do not ship it on one green run.

**2. Sidestep the row API (the real remaining option).** Have DuckDB `COPY` each partition's narrow
projection to a temp parquet, then read it with `ParquetColumnReader`, which reads whole column vectors
instead of cells. Parallelism then becomes a managed-code question with no concurrent DuckDB at all.
Open question that decides it: the extra write + read is of the narrow projection only, but at the
current per-partition read cost (~3.3 s for 15.5M rows) that extra I/O may cost more than the
vectorized read saves. Measure one partition both ways before building it.

**3. Re-tune partition sizing.** Partitions are pinned small (an eighth of the budget) because large
ones made the rollup dramatically slower - 66M-row partitions cut the merge from 39.4 to 14.7 min but
pushed the rollup from 26.0 to 62.2, a net loss. That trade is a property of the reader's speed, so it
should be re-measured now that the reader is ~1.5x faster, and again after any of the above. See
`MergedDataset.PartitionCountFor`.

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
