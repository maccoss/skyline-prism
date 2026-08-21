# Stage 2 reader throughput - what is left to do

Status as of dotnet-v26.12.0. Stage 2 (transition -> peptide rollup) is now **bounded in memory but is
the pipeline's wall-clock bottleneck**, and the obvious fix is blocked by a dependency bug. This is the
plan for lifting it, written down because the next person will otherwise repeat the two dead ends.

## Where it stands

Measured on a 20-document synthetic cohort (1.86 billion transition rows, 1,920 samples, 32-core
workstation), per-stage peak working set and average cores:

| Stage | peak | time | avg cores (of 32) |
|---|---|---|---|
| S1 merge | 4.09 GB | 14.7 min | 3.3 |
| **S2 transition -> peptide rollup** | **8.40 GB** | **62.2 min** | **1.2** |
| S2b/2c normalize + ComBat | 0.58 GB | 1.7 min | 1.4 |
| S3 parsimony | 0.72 GB | 0.0 min | - |
| S4 peptide -> protein | 1.97 GB | 0.2 min | 10.9 |
| S5b QC report | 6.77 GB | 4.0 min | 3.4 |

Stage 2 is ~75% of wall clock at roughly one core. The 32 rollup worker threads are not the problem -
they are **starved**. The single producer thread reads a DuckDB result set row by row through the
ADO.NET reader at roughly 500k rows/s, and everything downstream waits on it.

The prize: if the producer stopped being the bottleneck, the 20-document run would go from ~83 min to
something near 35. Extrapolated to 100 documents that is the difference between most of a working day
and a couple of hours. (Extrapolated - not measured. See "Verification bar".)

## Why the obvious fix is blocked

Partitions are independent by construction: the merge hashes on the peptide column, so every row of a
peptide is in exactly one partition and several can be read concurrently without splitting a block.
This was implemented, and it **crashes with an `AccessViolationException` within seconds**.

`DuckDB.NET.Data.Connection.ConnectionManager` holds a static cache of **refcounted database instances
keyed by connection string**. Every `"Data Source=:memory:"` in the process is therefore the *same*
database - one buffer pool, one `memory_limit` (a database-level setting, not a connection one), one
set of native state. When one reader thread closes its connection the refcount can reach zero and the
instance is torn down while another thread is still streaming from it. Use-after-free, surfacing at
whatever allocated next.

The binding offers no in-memory instance naming: `":memory:name"` is parsed as a file path. So there is
no one-line isolation available.

This was reverted rather than worked around, because in the stage that computes the reported
quantities a use-after-free produces wrong numbers as readily as a crash. See
`TransitionRollup.RunParallel` for the in-code account.

## Step 0: find out what the producer is actually bound by (do this first)

**Nothing below should be built before this is measured.** Twice in the work that produced this file, a
plausible theory about where time was going turned out to be wrong, and both times the fix made things
worse. The producer does two things per partition - a DuckDB sort + scan, and a managed row-by-row
marshal into `PeptideBlock` - and we do not currently know the split.

Cheap experiment (minutes, not hours): run the same per-partition query twice in a harness, once
draining `reader.Read()` without touching any column, once building blocks as
`MergedParquetReader.StreamPeptideBlocks` does. The difference is the managed cost; the floor is
DuckDB's.

- If it is mostly **managed**, option A is the whole answer and is safe.
- If it is mostly **DuckDB**, options B/C/D matter and A is not worth much.

## Option A - stop allocating per row (safe, no concurrency)

The inner loop currently allocates roughly this much per row:

```csharp
current.Ion.Add(pool.Get(reader, 1));                          // GetString allocates, then pooled
current.PrecursorCharge.Add(pool.GetKey(reader.GetValue(2)));  // boxes, then ToString allocates
current.ProductCharge.Add(pool.GetKey(reader.GetValue(3)));    // boxes, then ToString allocates
current.Sample.Add(pool.Get(reader, 4));                       // GetString allocates, then pooled
current.Area.Add(ToDouble(reader.GetValue(5)));                // boxes
current.RetentionTime.Add(ToDouble(reader.GetValue(6)));       // boxes
```

That is ~8-10 allocations per row - about 4 boxes, 2 charge strings that the pool immediately discards,
and 2 genuine strings. At 1.86 billion rows it is on the order of 15 billion allocations, and the
string pool added in this release removes the *retention* but not the *allocation*.

What to change:

1. Resolve each column's type **once** before the row loop (`reader.GetFieldType(i)`) and branch outside
   it, not per row.
2. Charges: read with a typed accessor (`GetInt64`/`GetInt32`) and map through a small
   `Dictionary<long, string>`. Charges are single digits, so this becomes zero-allocation after the
   first few rows. Today each one boxes *and* formats a fresh string.
3. `Area` / `RetentionTime` / `mz` / `shape`: `GetDouble(i)` when the column is `DOUBLE`. Keep the
   existing VARCHAR path - Skyline writes `#N/A` for unintegrated peaks, so these columns genuinely
   arrive as text sometimes (`MissingValueTests` covers it) - but choose the path once, not per row.
4. `Ion` / `Sample`: `GetString` allocates inside the binding and there is no span-returning API, so
   these stay. Cheap win available: compare against the previous row's value before hitting the pool
   dictionary, since both columns have long runs.

Risk: low. Behaviour-preserving; guarded by `MissingValueTests` and the rollup parity tests.

## Option B - bump DuckDB.NET (cheap to check)

We are on **1.5.3**; **1.5.5** is published. Check its changelog for connection-lifetime, streaming-
reader or `ConnectionManager` fixes before building anything else. If concurrent readers become safe
upstream, option C reduces to re-applying the reverted patch. This is an afternoon's check with a large
possible payoff and should be done alongside step 0.

## Option C - parallel readers with genuinely isolated instances

The reverted implementation is in the history of this branch and is small: slice
`dataset.Partitions` round-robin into N slices, run one `StreamPeptideBlocks` per slice into the
existing `BlockingCollection`, and give each its own share of the budget.

The only missing piece is instance isolation. Since `":memory:name"` is unavailable, the candidate is a
**file-backed database per reader slot** in the scratch directory
(`Data Source=<scratch>/reader{n}.duckdb`). Distinct connection strings mean distinct cache keys, hence
independent instances with independent buffer pools - which is also what makes the per-reader
`memory_limit` meaningful. The database file itself stays near-empty: all data is read from parquet.

Do not ship this on one green run. Required before it goes near `main`:

- A standalone harness running N concurrent file-backed instances streaming parquet, for many
  iterations, clean every time.
- At least 3 consecutive full-pipeline runs on a real cohort with identical outputs.
- `peptides_rollup.parquet` bit-identical to the serial path.
- An explicit decision that the residual risk is acceptable, recorded here.

If any of that is shaky, prefer option D.

## Option D - sidestep the row-by-row API entirely

Have DuckDB `COPY` each partition's **narrow projection** (the 8 columns the rollup reads) to a temp
parquet, then read it with `ParquetColumnReader`, which we already use elsewhere and which reads whole
column vectors rather than boxed cells.

- Costs one extra write + read, but only of the narrow projection - a fraction of the merged dataset,
  not a copy of it.
- Removes the per-row marshalling cost entirely rather than reducing it.
- Parallelism becomes a managed-code question (row groups, `Parquet.Net`), with no concurrent DuckDB
  instances involved - which is the whole reason this option is attractive despite the extra I/O.

## After any of the above: re-tune partition sizing

Partition size is currently pinned small (an eighth of the budget, ~16.8M rows) because large partitions
made the rollup dramatically slower - 66M-row partitions cut the merge from 39.4 to 14.7 min but pushed
the rollup from 26.0 to 62.2, a net loss end to end. That trade is a property of *today's* reader: a
spilling sort starves a slow serial consumer badly. A faster producer will move the optimum, probably
toward fewer, larger partitions, which would also make the merge cheaper. Re-measure both stages
together; see `MergedDataset.PartitionCountFor`.

## Verification bar (learned the hard way)

- **Measure end to end, never one stage.** The partition-sizing change improved Stage 1 by 2.7x and
  regressed the pipeline by 11 minutes.
- **Run the machine quiet.** Two measurement rounds here were invalidated by a concurrent experiment
  competing for the same disk, and one "new binary" run silently used a stale build because a leftover
  process held the DLL.
- **Repeat anything touching concurrency.** The crash reproduced 2 runs in 3, not 3 in 3.
- **Check parity, not just timing.** `peptides_rollup.parquet` bit-identical, `protein_groups.csv`
  byte-identical.
- **100 documents remains unverified.** The largest cohort ever run was 20, and it was 20 hardlinks to
  the same two files - so read parallelism against genuinely distinct inputs is untested.

## Instrumentation gap

All the numbers above came from an external PowerShell sampler wrapped around `prism.exe`. The pipeline
log reports what each stage *did* but not what it *cost*. Per-stage elapsed time and rows/s in the run
log would make this measurable by anyone hitting it, without rebuilding the harness - and would have
caught the Stage 2 regression in this release immediately rather than after a 70-minute run.
