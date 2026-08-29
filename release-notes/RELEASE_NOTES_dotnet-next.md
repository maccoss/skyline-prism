# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

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
