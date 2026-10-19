# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

## Bug Fixes

- Refused to read a `merged_data/` directory that no longer holds the rows its merge counted. Before,
  the extra rows were silently included. A copy of an output directory made over an older copy, replacing
  files but removing none, can keep partitions from an earlier merge that had more of them. The number of
  partitions is chosen partly from free memory, so it can differ between runs. On a 96-run cohort this left
  a stale `_pep_bucket=2` holding 9.9M duplicated rows beside 29.9M, and anything that read the whole
  directory counted about a third more precursors than the cohort has. Ion accounting, the Spectrum
  density tab and detection in the differential analysis now refuse such a directory, with a message
  naming the leftover partitions. A re-run into it re-merges and recomputes every stage instead of
  reusing the cached merge. The check compares the parquet footers with `TotalRows` in
  `merged_data.cache.json`. It reads no column data, and took under 0.1 s on that cohort.

- Corrected documentation that contradicted the code, with no change to any output. `docs/output_files.md`
  and `SPECIFICATION.md` listed a `Coeluting` column the bundled `Skyline-PRISM` report has never exported,
  and `docs/output_files.md` omitted the `Background` column it does export (nothing reads it).

## Performance

## Breaking Changes
