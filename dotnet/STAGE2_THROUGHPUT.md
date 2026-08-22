# Stage 2 reader throughput

Status: unreleased, on top of dotnet-v26.12.0. Stage 2 (transition -> peptide rollup) is bounded in memory and no longer
the pathological case it was, but it is **single-threaded, and will stay that way until the DuckDB
binding can be read from concurrently**. Two candidate replacements for its reader were built and
measured, and **both lose to what ships** - see the verdict below. This records what was measured, so
the next person does not re-derive it.

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

## Verdict: leave Stage 2 as it is

Both candidate replacements were built as benchmark arms and **both lose to what ships**. Measured with
`bench/Stage2Bench` on an 11.4M-row partition, three interleaved repeats, all arms accumulating the same
11,400,000 values into per-peptide blocks:

| arm | median | spread | allocated | vs today |
|---|---|---|---|---|
| **duckdb-stream (ships today)** | **3.3 s** | 22% | **2.37 GB** | **1.00x** |
| nosort-managed | 9.1 s | 6% | 5.49 GB | 0.36x - **2.8x slower** |
| copy-then-read (the sketch) | 3.5 s | 22% | 2.45 GB | 0.94x |

Read those honestly: `copy-then-read`'s 0.94x sits inside both arms' run-to-run spread, so against what
ships it is a **tie**, not a loss - it simply does not win, which is enough to reject a structural
change. The no-sort gap is far outside the spread (and that arm had the *tightest* spread of the three,
6%), so its 2.8x penalty is real and not a contention artefact.

Why the no-sort idea fails is worth keeping, because it looks so attractive on paper: with no SQL in the
path, it has to compose the transition id **in C# per row** - precisely the cost that was moved into the
query in dotnet-v26.13.0. Bypassing DuckDB means giving up the SQL-side preprocessing that makes the
current reader fast, and the dictionary of live per-peptide blocks costs 2.3x the allocation on top.
That is structural, not a tuning problem.

> [!WARNING]
> The first run of this comparison reported the no-sort arm at **3.12x faster**. That was wrong: the arm
> stored a row-group-local index and discarded the values, so it was not doing the work the other arms
> were. Rows and peptides matched, so the correctness check passed. The harness now also counts
> **values accumulated**, which is what caught it - and any new arm must keep that figure equal or it is
> not measuring the same thing. A faster arm that does less work is the easiest benchmark mistake to
> make and the hardest to notice, because the number it produces is exactly the number you were hoping
> for.

The earlier per-stage figures below (4.7 s / 2.7 s / 1.6 s) came from that flawed comparison, where no
arm accumulated anything. They are kept for the decomposition they show - the sort costs ~2.7 s and the
marshalling ~2 s - but the conclusion drawn from them was wrong.

## The options, and why none of them are open

**1. Sidestep the row API - MEASURED, and it does not pay.** Two variants were tried. Both are described
here because both looked compelling on paper and the reasons they fail are not obvious.

*Two-phase (`copy-then-read`).* Have DuckDB `COPY` a partition's narrow projection to a temp parquet,
then read that back with Parquet.Net a row group at a time - whole column vectors, no per-cell
marshalling. Decomposed on one 15.5M-row partition:

| path | sec | Mrow/s |
|---|---|---|
| A: DuckDB streaming read (what ships today) | 4.7 | 3.30 |
| B1: `COPY` narrow projection to parquet | 2.7 | 5.74 |
| B2: Parquet.Net row-group read + block assembly | 1.6 | 9.57 |

The B2 line is what made this attractive: pure managed code, so it could run on several threads with no
concurrent DuckDB anywhere. But **B1 + B2 together measure 0.94x of A** once both arms build the same
blocks - the `COPY` gives back everything the faster read wins. Parallelising phase 2 would still help
in principle, but it would be parallelising the *cheap* half: phase 1 is serial and is now most of the
cost, so the ceiling is ~1.06x even with infinite threads. That is not worth a structural change to
Stage 2, and it doubles transient disk.

*No sort (`nosort-managed`).* Skip the `ORDER BY` entirely, read the partition as it lies and group rows
by peptide in a dictionary. This removes the sort - the single most expensive operation in the arm - so
it should be the fastest thing available. It is **2.8x slower**, for two reasons that only show up when
it is made to do the real work:

- With no SQL in the path, the transition id must be composed in C# per row. That is exactly the cost
  moved into the query in dotnet-v26.13.0, being paid again.
- Without a sort, **every** peptide's block is live until the partition finishes, where the sorted arms
  hold one. 5.49 GB allocated against 2.37 GB, and the dictionary and its 4,549 x 4 growing lists cost
  more than the sort they replaced.

Both are dead ends, and the second is a useful reminder that removing the obviously expensive operation
can cost more than it saves.

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

## Running the measurements yourself

`bench/Stage2Bench` compares the candidate read strategies on one partition of a real merged dataset:

```bash
prism merge <report1.parquet> <report2.parquet> -o D:/bench/merged     # build a dataset once
dotnet run -c Release --project dotnet/bench/Stage2Bench -- D:/bench/merged 3
```

It runs three arms - `duckdb-stream` (what ships), `nosort-managed` (no sort, group in a dictionary) and
`copy-then-read` (DuckDB `COPY` to a temp parquet, then Parquet.Net) - and reports the median of
interleaved repeats.

Two properties matter more than the numbers it prints:

- **Arms are interleaved**, not run in blocks, so machine-state drift shows up as variance *within* an
  arm rather than as a fake difference *between* arms.
- **Every run is watched for contention** and labelled, and the summary refuses to present contended
  figures as fact. This exists because an identical configuration measured 2.07 min and 3.70 min hours
  apart here; the cause was other software starting in between, and the first explanation reached for
  was page-cache warmth. Hours of analysis were built on that gap before anyone read the process list.

It also checks that the arms agree on rows, peptides **and values accumulated** before comparing their
timings. A faster arm that reads different data is not a faster arm - and neither is one that reads the
same rows but does less with them, which is the mistake the values column exists to catch. Any new arm
must build the same per-peptide blocks as the others.

The project is deliberately **not** in `SkylinePrism.sln`: CI builds the solutions by name, so the
harness stays out of the critical path and the shipped package while remaining in the repo and
runnable. Its package versions are pinned to match `SkylinePrism.Core` - keep them in step, or it stops
measuring the engines the pipeline actually uses.

## Verification bar (learned the hard way)

- **Measure end to end, never one stage.** A change that improved Stage 1 by 2.7x regressed the
  pipeline by 11 minutes.
- **Compare against the previous release's binary on the same inputs**, not against remembered numbers.
  Use a git worktree at the release tag; it takes two minutes and removes all doubt.
- **Exercise the paths the default config skips.** The transition id only appears with
  `median_polish` + `output.include_residuals`; a default run would have "verified" a change to it
  without ever producing one.
- **Run the machine quiet, and verify it rather than assuming it.** Three measurement rounds here were
  invalidated by competing load - twice by a concurrent experiment of my own, once by two Skyline
  instances the user had started - and one "new binary" run silently used a stale build because a
  leftover process held the DLL. `Stage2Bench` now records this; anything measured by hand should check
  the process list first.
- **Prefer ratios to absolute rates.** Contention hits interleaved arms roughly equally, so an A/B
  comparison survives a busy machine. An absolute MB/s quoted across sessions does not - that is how a
  13x "finding" turned out to be partly measurement conditions.
- **Repeat anything touching concurrency.** The crash reproduced 2 runs in 3, not 3 in 3.

## Instrumentation

The run log now reports per-stage elapsed time and a sorted summary at the end, so the next person can
see where a real cohort spends its time without wrapping the process in an external sampler - which is
how every number above had to be obtained.
