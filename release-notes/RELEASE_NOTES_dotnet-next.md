# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

- **Per-stage timing in the run log.** Each stage now reports how long it took, followed by a sorted
  summary with percentages. Until now the log said what each stage *did* but never what it *cost*, so a
  stage that doubled in time was invisible unless someone wall-clocked the whole run from outside - and
  every performance number in the previous release had to be obtained that way.

## Bug Fixes

## Performance

- **The transition rollup reads ~1.5x faster, and the whole pipeline ~25% faster.** The rollup's
  producer was measured before being changed: 68% of its time was managed work, not DuckDB, and almost
  all of that was materializing a fragment ion, two charges and a ~45-character sample id as strings on
  *every transition row* - which the string pool then deduplicated and discarded. Those now come out of
  SQL instead: the transition id is composed there, the precursor test is a boolean, and the sample name
  is resolved to its index by a join (that index is the output column, so the per-row name lookup was
  redundant). On a 2-plate cohort (186M transition rows, 192 samples), against the previous release's
  binary on the same inputs: **3.33 min -> 2.49 min, peak 2.75 GB -> 2.40 GB**, with
  `peptides_rollup.parquet`, `proteins_raw.parquet` and `corrected_proteins.parquet` bit-identical and
  all 662,339 `peptide_residuals` transition ids unchanged.

## Breaking Changes
