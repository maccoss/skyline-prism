# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

## Bug Fixes

- **The config from "Show command line" now names the algorithm it used.** Keys that identify an
  algorithm are written for every active section even when they sit at their default:
  `library_fitting_method` (which fit sets the per-sample scale in library-assisted rollup),
  `batch_correction.method`, and `sample_outlier_detection.method`. They were previously elided as
  "same as the default", which was reasoned from the C# side alone - C# implements only
  `median_polish` and only `combat`, so the value could not vary. But the config exists to be handed
  to the `prism` CLI, and the Python engine implements `median_polish` **and** `least_squares`, so a
  config that omitted the key left the reader unable to tell which fit produced the numbers. Numeric
  and boolean tuning values are still elided at their defaults; a typical config grew from 31 keys to
  34, against ~95 for a full object dump.
- **The batch column is written where both engines read it.** The emitted config put it under
  `metadata:`, a section only the C# engine knows, so a config copied to the Python CLI warned that
  `metadata` was unrecognized and then auto-detected the batch column instead of using the chosen one.
  It is now written under `data:`, which C# reads first (`Data.BatchColumn ?? Metadata.BatchColumn`)
  and Python reads exclusively - same behavior on C#, correct behavior on Python. The only keys the
  emitted config still carries that Python does not implement are `batch_correction.peptide_level` and
  `batch_correction.protein_level`, both recorded as C#-only in `docs/parameters.md`.

## Performance

## Breaking Changes
