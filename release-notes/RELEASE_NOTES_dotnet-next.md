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
- **The Differential and Markers settings are in a collapsible column beside the plot.** The
  Differential toolbar had grown to 15 drop-downs and 16 labels in one wrapping row, about twice the
  window's width. At the default size it wrapped to four lines and took the plot's height, and hiding
  one control reflowed everything after it. The settings are now grouped top to bottom in the order you
  decide them: the design, the level, the samples being compared, the hit rule, and the method. Choices
  of two to four options (design, level, adjusted or raw p, heatmap columns) are now buttons that show
  every option instead of drop-downs. **Method** is folded away by default with its current state shown
  in one line, so you can see what ran without opening it. The view (Volcano, Detection, Enrichment)
  is tabs above the plot. **«** hides the column to give the plot the full width. The Markers pane
  has the same layout.
- **The window opens at a size that suits the screen.** It used to open at a fixed 1040 × 740: a small
  box on a large monitor, and at that width the plot was about 600 pixels beside the new settings
  column. It now opens at 85% of the screen (up to 1600 × 1000), with a minimum of 960 × 640.

## Bug Fixes

- **The per-sample marker heatmap no longer draws sample names across the data.** On a 96-sample
  cohort, 96 rotated sample IDs were drawn over the heatmap, its title, and its row labels, and nothing
  underneath could be read. Each group is now named once, centered under its block of columns with a
  line between groups, which is what that view is read for. Group names wider than their block are
  staggered onto two lines so neighbors don't run together (two six-sample control groups had read as
  "Quality ControlStandard").
- **A large marker panel no longer labels every row in unreadable type.** Labels shrank to 6pt as a
  panel grew and still overlapped; the 158-member histone panel was a smear. Now at most 30 rows are
  named (every k-th) at a readable size. The full list is in the quant report's CSV.
- **The Markers box plot is readable.** It used font sizes meant for an exported figure in a strip
  two-fifths of the pane tall. The axis text took most of the height, the boxes were squeezed into a
  sliver, and the y-axis title was cut to "mean marker z-". It is now scaled to match the heatmap above it.
- **The Differential plot clears when the current settings can't produce a result.** Picking a design
  that still needs a subject column, or a group-by with no arms ticked, left the previous result on
  screen under a message saying nothing had run. That plot had a different design and a different axis
  label, and read as the answer to the new question.
- **The Differential pane's trend span now covers the samples the fit used.** The axis title and status
  line read the span from every value in the trend column, while the code's comment said it came from
  the fitted samples. Whenever samples were left out, the label described a range the model never saw.
  A Verapamil-only fit was labelled "-2 to 12", where -2 is a Liraglutide timepoint, on the axis whose
  numbers are a change across that span. It now reads "0 to 12", matching the command line and the CSV
  header. The axis title also uses the short name of the axis ("log2 change across Week") instead of
  the full label, which no longer fit on the plot.
- **A missing subject column is asked for in plain words.** The message named an internal property
  (`DifferentialOptions.SubjectLabels`), which a person can't act on, and the paired version pointed at
  a "Pair by" control that no longer exists. Both front ends now say which column is needed and why.
  In the pane, an empty Subject picker prompts for a choice instead of running and failing.

- **The Differential pane's "Group by" label now hides under a trend design, and the QC Plots pane's
  label no longer disappears.** The QC pane's label was named as if it belonged to the Differential
  pane, so choosing a trend hid the QC pane's label, while the Differential pane's own label (which had
  no name) stayed on screen with nothing next to it.
- **The Differential pane no longer has a Run button.** Every selector already reruns the contrast when
  it changes, so the button never needed pressing.
- **The Differential plots no longer have a legend over the data.** The "significant / not significant"
  box sat in the lower right of the Volcano and Detection plots, on top of points. Red only ever meant
  "past the lines drawn on the plot", and the status line gives the count and the rule.

## Performance

## Breaking Changes
