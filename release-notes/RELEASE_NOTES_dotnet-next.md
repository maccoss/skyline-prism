# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

- **A re-run now only redoes the stages that actually changed.** Changing one setting used to recompute
  everything: on a real cohort, changing `protein_rollup.method` re-exported ~15 GB from Skyline and
  redid the transition rollup and the peptide normalization, none of which that setting can affect.
  Only the Stage 1 merge was cached.

  Each stage now records a fingerprint of its inputs and of the config keys it actually reads, and a
  re-run into the same output directory reuses any stage whose fingerprint is unchanged. Change
  `protein_rollup.method` and only the protein arm re-runs - `corrected_peptides.parquet` comes out
  byte-identical. Change `transition_rollup.method` and everything downstream of it rebuilds, because
  invalidation chains through the fingerprints.

  Guardrails, because the failure mode here is silently stale numbers rather than an error:
  - The fingerprint includes the **PRISM version**, so any release re-runs everything rather than
    trusting that a code change did not move a result.
  - Reuse requires every file the cache claims to still exist and be non-empty - deleting an output by
    hand rebuilds it.
  - Which config keys belong to which stage is a declared table (`StageDependencies`), and a test fails
    the build if a key is added without being classified, or if mutating a key does not change exactly
    the stages that declare it.
  - `--force-reprocess` reuses nothing (and still records, so the next run benefits).

  A **closed** `.sky` also gets its export reused when the document and the report definition are
  unchanged. A **running** Skyline does not: a live document can hold unsaved edits that the file on
  disk knows nothing about, and the RPC surface exposes no document revision to key on, so re-exporting
  it every run is the only safe answer.

- **The Dynamic Range plot now says which rollup produced its values.** The y axis reads
  `Log10 abundance (median_polish)` - on the axis, so it travels with the image when the plot is copied
  into a slide - and the status line adds what that method's numbers *are*: for median polish, "a
  typical peptide's level, not a sum - so it does not scale with peptide count".

  This matters when the plot is compared against Skyline's relative-abundance view, which is the
  natural thing to do and which **sums peak areas**. The two are different quantities, and on a real
  cohort the difference reorders the top of the plot: C4A (121 peptides) leads Skyline's summed view
  and sits below ITIH2 (44 peptides) here, because summing scales with peptide count and median polish
  does not. Neither is wrong; nothing on the tab used to say they were different questions. The method
  is read from the run's own `parameters.json`, and the plot still draws (unlabeled) when a directory
  has none.

## Bug Fixes

- **A CSV export superseded by a parquet one is now deleted.** PRISM exports the PRISM report as
  parquet and falls back to invariant CSV only when the parquet comes back invalid - it never writes
  both in one run, and the fallback already deleted the failed parquet. What it did not do was the
  reverse: a CSV left by an earlier run that fell back stayed in `skyline-reports/` forever, even once
  a later run replaced it with a parquet. On a real cohort that is **14.8 GB of dead CSV beside the
  695 MB parquet** that superseded it, and two files a user can pick between by hand, one of them
  stale. A successful parquet export now removes the `<label>.csv` beside it, saying so in the log.
  Only that file is touched, never anything else in the directory.

## Performance

## Breaking Changes
