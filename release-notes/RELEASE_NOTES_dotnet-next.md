# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

- **The marker normalization can now be inspected, not just trusted.** Two entries appear in the QC
  Plots tab's plot picker when a run used `marker_normalization`, showing the two halves of the panel's
  PC1. Both are read from `marker_normalization.csv` - the numbers the run actually subtracted, not a
  recomputation that could drift from them - so Level and View are greyed out for them.

  - **Marker score** - the per-sample score, one column per Group-by value. Set Group-by to the study's
    condition: if the score separates the groups, the panel is tracking the phenotype rather than how
    much material was captured, and normalizing on it removes the finding. That judgement cannot be
    automated (the same separation is what you would see if the biology genuinely changes the marked
    material), which is why it is a plot rather than a warning.
  - **Marker loadings** - each marker's PC1 contribution, largest first, opposing markers in red. The
    title reports the largest marker's share of the axis and flags it above 50%: a panel carried by one
    protein is a single-protein normalization wearing a panel's clothes.

## Bug Fixes

## Performance

## Breaking Changes
