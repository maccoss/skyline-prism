# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

## Bug Fixes

- The QC report no longer fails validation on the PCA QC-reference distance ratio. A 264-sample
  run read "Validation: FAILED" above two green CV improvements (reference +30%, QC +6%) because
  the ratio was 0.43, below the 0.5 cutoff. That ratio is measured from the raw rollup, so any
  loading difference between the reference and QC injections counts toward it, and normalization
  removes that difference by design. It is now reported as a note. The verdict fails only when the
  QC CV gets worse or cannot be measured.
- A "Validation: FAILED" banner now names the reason inside the banner. Before, the reason was
  only in a WARNINGS box below the control-CV table. That box now holds only warnings that did not
  decide the verdict, such as a reference CV that rose.

## Performance

## Breaking Changes
