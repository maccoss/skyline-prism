# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

- **Reference-anchored ComBat is now a tick box in the Skyline tool**: *"Anchor ComBat on the Standard
  samples"* on the Settings tab. It estimates each batch's effect from the replicates whose Skyline
  Sample Type is **Standard** - identical material run in every batch - instead of from the
  across-batch average. Differences in that material are purely technical, so no biology is removed
  and the batches need not contain comparable samples, which is the better choice when plates differ
  in what they hold (different case/control ratios, say). Nothing to configure beyond marking those
  injections `Standard` in Skyline.

  A batch with no Standard replicates is corrected on its own average, location only, and says so in
  the log; a batch with exactly one gets a location correction but no rescaling, since one reference
  is a level, not a spread.

- **`processing.merge_memory_mb`** caps DuckDB's buffer pool during the Stage 1 merge. `0` (the
  default) sizes it from the machine. Work beyond the cap spills to the sort scratch directory, so a
  smaller value is slower, never wrong.

## Breaking Changes

- **Batch estimation from acquisition-time gaps is now OFF by default** (`batch_estimation.method`
  is `none`, was `auto`). It cannot tell a real plate boundary from an ordinary pause in a
  continuously acquired run, and when it guesses wrong ComBat "corrects" between batches that do not
  exist - which silently changes every abundance. Batches should come from a real annotation; set
  `batch_estimation.method: auto` to opt back in. Both engines.

## Bug Fixes

- **Clicking a protein in the Dynamic Range plot works again.** Two things were wrong. The click
  redrew the plot *after* reporting what it had done, and the redraw rewrites the status line - so
  every message, including the reason a selection failed, was destroyed before it could be read, and
  a failure looked exactly like a click that did nothing. And a protein group containing several
  proteins reports its accessions and gene names as one `" / "`-joined string, which matches no node
  in the document; each member is now tried separately. On an 11,320-group mouse cohort that takes
  protein resolution from 99.1% to **100%**. A failure to read the document tree at all (Skyline
  busy) now says so, is no longer cached as if the document had no proteins, and retries on the next
  click.

- **The isolation-window import could hang a run indefinitely.** Reading the real DIA windows out of a
  data file normally takes about 10 seconds, but PRISM waited on Skyline's output with no deadline and
  only checked for cancellation *after* a line of output arrived - so a Skyline that went quiet (a data
  file behind a slow or half-mounted link) stopped the run for good, and **Stop could not break it
  either**. The wait is now bounded and genuinely cancellable, and PRISM kills the Skyline it started
  when it gives up.

  It also no longer runs **before** the pipeline. It is an enrichment for one plot, so it now runs
  alongside the run and writes its catalog when it finishes; the density map reads that file when the
  tab is opened, so landing late costs nothing. Previously every second it spent was a second the run
  had not started - five minutes of it, on a data file that turned out to be unreachable.

  The import is also time-boxed at 5 minutes. It is an enrichment - without it the density map falls
  back to uniform bins and nothing else about the run changes - so it can no longer hold up a run at
  all. Raise it with `PRISM_ISOLATION_TIMEOUT_SEC` (seconds) when the data really is that slow to reach.

- **The Dynamic Range tab failed at peptide level** with `The input string 'AAAAAGAGLK' was not in a
  correct format` - a peptide sequence being parsed as an abundance. Its list of metadata column names
  had drifted from what the pipeline writes, so an unlisted column was read as a replicate. Extending
  that list could not fix it, because the peptide column keeps whatever name the Skyline export used
  and that is detected per document. Replicates are now identified by **type**: a text column is never
  a replicate whatever it is called. (This is the second time this class of bug has appeared here; the
  first was the blank protein-level plot in 26.9.0.)

- **Switching level after an error left the error on screen** until you changed tab and came back. The
  level dropdown was gated on the previous load having succeeded, which the failed load had just
  cleared - so switching level did nothing, exactly when it was the natural way out.

- **Point labels no longer pile on top of each other.** Every label went up-and-right at a fixed
  offset, which is fine for one and unreadable for twenty on a dense curve. Each label now tries a
  ring of positions around its point - **below as well as above**, left and right, then further out -
  and takes the first that clears the labels already placed. Where nothing is clear it takes the
  least-overlapping spot rather than dropping a label you asked for.

- **"Label this list's members on the plot" did nothing.** The tick was a filter on a right-click label
  mode that is off by default, so it had no effect unless that mode had also been set from a menu there
  was no reason to open. The per-list tick now turns labels on by itself; the plot's right-click menu
  keeps "Label all protein lists" and "No labels" as bulk switches over the same setting.

- **Six Python tests failed on Windows** (`test_chunked_processing.py`). The peptide rollup left its
  parquet reader open across the delete of the file it was reading, which POSIX allows and Windows
  does not - so the temp file could not be removed, the delete raised, and the caller's temporary
  directory could not be cleaned up either.

- **ComBat's reference-anchored path now uses the same estimator as the standard path**, so the fixes
  released in 26.9.0 apply to it as well. It previously had its own implementation, which invented a
  placeholder scale of `1.0` where the data supported none and fed that into the empirical-Bayes prior
  - letting one such feature perturb the shrinkage of every other feature in its batch - and reported
  none of what it could not estimate. It also now holds out a feature that some batch's references
  never observed, rather than treating the unknown offset as zero.

  **Dense data is unaffected**, which is PRISM's normal case. Measured on a 400 x 40 cohort with 3
  references per batch: no cells changed on dense input; ~25% changed where features are constant
  within a batch (worst 0.3%); with 10% missing values the median change was 0.1%.

  Reference-anchored ComBat is also now covered by cross-engine fixtures. It had none - standard ComBat
  is held to R's `sva` and the end-to-end fixtures hold the engines to each other, but nothing reached
  this method, so the two engines could have drifted apart on it indefinitely.

## Performance

- **Reference-anchored ComBat no longer forces Stage 2b/2c back into memory.** It was the last
  configuration that fell back to the in-memory implementation - the pipeline's memory wall - so a
  large cohort could not use it. It streams now, bounded by the input's row-group size rather than by
  the number of samples (measured elsewhere at 102 MB against 798 MB on a 20,000-peptide x
  600-sample cohort). Only `quantile` normalization and a CSV/TSV `output.format` still fall back,
  and the log says which implementation ran.

  Results are unchanged: the same-process regression suite compares the streaming and in-memory
  implementations cell by cell at 1e-12, and its reference-anchored cases previously asserted that
  streaming was *ineligible* - so they were comparing the in-memory path with itself. They now
  compare the two implementations for real, across every normalization method, auto-revert on and
  off, dropped all-NaN rows and multi-row-group inputs.

- **The Stage 1 merge now budgets memory from FREE memory rather than total RAM.** DuckDB runs
  in-process and its buffer pool is native memory the .NET GC cannot see, so a limit written against
  total RAM on a machine that is already busy does not spill - it pages, which is what "memory at 100%"
  during Stage 1 was. On the machine this was found on (65 GB total, 11 GB free) the old budget was
  49 GB and the new one is 9 GB. The bound is the smaller of 75% of total and 80% of free, with a 2 GB
  floor, and the log names the budget it picked.

- **A single input now skips the unsorted intermediate**, sorting in one pass and saving a full write
  plus a full read of the entire dataset - most of Stage 1's wall clock when the report sits on a
  network share. The two-stage form remains for multi-document cohorts, where it exists to keep many
  parallel parquet readers inside the memory budget.

## Breaking Changes
