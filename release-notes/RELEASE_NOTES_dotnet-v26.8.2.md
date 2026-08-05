# Skyline-PRISM (C#) dotnet-v26.8.2 Release Notes

Patch on top of 26.8.1, both fixes to the config the tool hands back from "Show command line": it now
names the algorithm each stage used, and writes the batch column where the Python CLI can read it.

## Bug Fixes

- **The config from "Show command line" now names the algorithm it used.** Keys that identify an
  algorithm are written for every active section even when they sit at their default:
  `library_fitting_method` (which fit sets the per-sample scale in library-assisted rollup),
  `batch_correction.method`, and `sample_outlier_detection.method`. They were previously elided as
  "same as the default", which was reasoned from the C# side alone - C# implements only
  `median_polish` and only `combat`, so the value could not vary. But the config exists to be handed
  to the `prism` CLI, and the Python engine implements `median_polish` **and** `least_squares`, so a
  config that omitted the key left the reader unable to tell which fit produced the numbers. Defaulted
  values are otherwise still elided; a typical config grew from 31 keys to 34, against ~95 for a full
  object dump.
- **The batch column is written where both engines read it.** The emitted config put it under
  `metadata:`, a section only the C# engine knows, so a config copied to the Python CLI warned that
  `metadata` was unrecognized and then auto-detected the batch column instead of using the chosen one.
  It is now written under `data:`, which C# reads first (`Data.BatchColumn ?? Metadata.BatchColumn`)
  and Python reads exclusively - same behavior on C#, correct behavior on Python. The only keys the
  emitted config still carries that Python does not implement are `batch_correction.peptide_level` and
  `batch_correction.protein_level`, both recorded as C#-only in `docs/parameters.md`.
