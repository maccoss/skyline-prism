# Skyline-PRISM (C#) dotnet-v26.20.0 Release Notes

A marker normalization can now be checked rather than taken on trust: the QC Plots tab shows the panel's
per-sample score and its per-marker loadings, which is how you tell a panel that measures capture from
one that is quietly tracking the phenotype.

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
