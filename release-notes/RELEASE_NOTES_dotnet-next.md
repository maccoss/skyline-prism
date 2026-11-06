# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

## Bug Fixes

- The QC report's PCA QC-reference distance check now measures what batch correction did, not
  what normalization did. A 264-sample run read "Validation: FAILED" above two green CV improvements
  (reference +30%, QC +6%) because the ratio was 0.43, below the 0.5 cutoff. The ratio was measured
  from the raw rollup, so any loading difference between the reference and QC injections counted
  toward it, even though normalization removes that difference by design. It is now measured from
  the normalized, pre-ComBat peptides (`peptides_log2_internal.parquet`) to the corrected output,
  and still fails below 0.5, because batch correction pulling two different materials together is
  real damage. An output directory without that file gets the raw ratio as a note only.
- A "Validation: FAILED" banner now names the reason inside the banner. Before, the reason was
  only in a WARNINGS box below the control-CV table. That box now holds only warnings that did not
  decide the verdict, such as a reference CV that rose.

## Performance

## Breaking Changes
