# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

- **A trend can be fitted against a timepoint written as text.** Study metadata usually names a
  timepoint for people (`V2_Week 8`, `Day 14`). The trend picker used to list only columns made
  entirely of numbers, so a column like this didn't appear at all. On the serum cohort this was built
  for, the only axes offered were `MS_Run_Order` and `Sample Group`, and neither is time. A column with
  numbers inside text is now offered once for each number it holds, named after the word in front:
  `Longitudinal Draw Description (Week)` reads 8 from `V2_Week 8`, and `(V)` reads 2. Both are
  offered, and nothing chooses between them for you. The visit index and the elapsed weeks are
  different quantities, and a slope fitted against the wrong one looks entirely reasonable. The pane
  shows what the chosen axis parsed (`Reading V0_Week 0 -> 0, V1_Week 2 -> 2, ...`), and it only
  selects an axis automatically when the column is plain numbers. `prism differential --trend-over`
  takes the same names, and an unknown name lists the available axes.
- **A trend can be restricted to some of the samples.** A trend pools every sample with a value on its
  axis, so two studies that share a timescale were merged into one slope. On that cohort, a Verapamil
  arm and a Liraglutide arm both drawn "at week 4" became a single point. The Differential pane's new
  **Restrict to** picker and `prism differential --restrict-to COLUMN=VALUE[,VALUE...]` (repeatable;
  restrictions narrow together) keep only the samples you name. A column or value the run doesn't have
  is refused, with the values it does have. Restricted samples are counted separately from samples with
  no value on the axis, and a quant report records the restriction in the command that reproduces it.
  Fitting the within-subject Verapamil trend now gives weeks 0 to 12 over 50 samples in 10 subjects.

## Bug Fixes

- **The Differential pane's "Group by" label now hides under a trend design, and the QC Plots pane's
  label no longer disappears.** The QC pane's label was named as if it belonged to the Differential
  pane, so choosing a trend hid the QC pane's label, while the Differential pane's own label (which had
  no name) stayed on screen with nothing next to it.
- **The Differential pane no longer has a Run button.** Every selector already reruns the contrast when
  it changes, so the button never needed pressing.

## Performance

## Breaking Changes
