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
  takes the same names; a bare column name that offers two readings is refused with both named, and
  an unknown name lists every available axis. A hyphen counts as a minus sign only at the start of a
  value or after a space or punctuation, so timepoints written `Week-4` or `Day-14` read 4 and 14.
- **A trend can be restricted to some of the samples.** A trend pools every sample with a value on its
  axis, so two studies that share a timescale were merged into one slope. On that cohort, a Verapamil
  arm and a Liraglutide arm both drawn "at week 4" became a single point. The Differential pane's new
  **Restrict to** picker and `prism differential --restrict-to COLUMN=VALUE[,VALUE...]` (repeatable;
  restrictions narrow together) keep only the samples you name. A column or value the run doesn't have
  is refused, with the values it does have. Restricted samples are counted separately from samples with
  no value on the axis. A quant report records the restriction in its title, `quant_parameters`,
  `differential.csv` and the command that reproduces it, and its marker panels use the same samples.
  Fitting the within-subject Verapamil trend now gives weeks 0 to 12 over 50 samples in 10 subjects.
- **The Differential and Markers settings are in a collapsible column beside the plot.** The
  Differential toolbar had grown to 15 drop-downs and 16 labels in one wrapping row, about twice the
  window's width. At the default size it wrapped to four lines and took the plot's height, and hiding
  one control reflowed everything after it. The settings are now grouped top to bottom in the order you
  decide them: the design, the level, the samples being compared, the hit rule, and the method. Choices
  of two to four options (design, level, adjusted or raw p, heatmap columns) are now buttons that show
  every option instead of drop-downs. **Method** is folded away by default with its current state shown
  in one line, so you can see what ran without opening it. The view (Volcano, Detection, Enrichment)
  is tabs above the plot. A button at the top of the column hides it to give the plot the full width. The Markers pane
  has the same layout.
- **Publish an output directory to Panorama.** **Publish to Panorama...** in the Skyline tool, and
  `prism publish` on the command line, put the QC report and the quant report on Panorama as wiki
  pages, each in a folder you choose. They also upload the whole output directory next to the folder
  that holds the raw files (`.../@files/RawFiles` gives `.../@files/<output directory>`). A links
  page shows up on the folder's own page, in a Wiki web part right after Targeted MS Runs (above
  Files), so the results are found
  from the folder rather than from the wiki's page list. It links the QC page, every quant page
  published from the directory (one per contrast), and the uploaded files. Without folder
  administrator permission the links page is still published, and the publish says what a folder
  administrator would need to add by hand. Publishing
  again updates the same pages, and Panorama keeps the earlier versions of their text; only plots
  that changed are sent again. A page published from one output directory is not replaced by
  another output directory that happens to have the same name, unless taken over on purpose
  (`--take-over`). It uploads only the files
  that changed, checked against Panorama's own MD5 of its copy, and verifies every upload the same
  way. It refuses to overwrite a page someone edited on Panorama unless told to. Sign-in uses the
  key PanoramaBridge already saved on the computer, so a lab machine needs no setup; otherwise the
  window asks once, and `PRISM_PANORAMA_API_KEY` serves a headless run. Each output directory
  remembers its targets in `panorama.json`, so `prism publish -d <dir>` alone republishes. Panorama
  refuses HTML with a `<style>` block from ordinary users, and an uploaded HTML file with a doctype,
  both with nothing but "403". So the reports' CSS is inlined for the wiki, their plots become page
  attachments, and the report files go up in an inlined form. See `docs/panorama-publishing.md`.
- **The window opens at a size that suits the screen.** It used to open at a fixed 1040 x 740: a small
  box on a large monitor, and at that width the plot was about 600 pixels beside the new settings
  column. It now opens at 85% of the screen (up to 1600 x 1000), with a minimum of 960 x 640.

## Bug Fixes

- **The per-sample marker heatmap no longer draws sample names across the data.** On a 96-sample
  cohort, 96 rotated sample IDs were drawn over the heatmap, its title, and its row labels, and nothing
  underneath could be read. Past 15 columns each group is now named once, centered under its block of
  columns with a line between groups, which is what that view is read for; a heatmap of 15 or fewer
  still names every sample, since they fit. Group names wider than their block are
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
  A Verapamil-only fit was labeled "-2 to 12", where -2 is a Liraglutide timepoint, on the axis whose
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
- **The default variance prior was used at the wrong level, so moderated p-values were miscalibrated.**
  The intensity-trend prior takes its shape from a LOWESS of within-group variance against intensity.
  By default that is fitted on the run's QC and reference injections. The prior was then used at the
  level those groups gave, but no source of groups measures the noise a contrast is tested against.
  Pooled injections lack the biology a study sample carries, so their trend sat too low and made
  tests optimistic. Design groups under a paired or within-subject model contain the between-subject
  spread the subject block removes, so their trend sat too high and cost power. The trend is now
  multiplied by a level fitted, together with the prior degrees of freedom, to the contrast's own
  residuals: limma's `fitFDist` with the trend as a known offset. On 40 null permutations of a
  3,595-protein serum cohort, the share of p-values below 0.05 went from 6.1% to 4.6% for an
  unpaired contrast and from 7.0% to 5.8% for a paired one. On simulated nulls it went from 40% to
  4.9%, and for a paired contrast fitted on its design groups from 0.17% to 5.1%; a calibrated test
  gives 5%. The CLI, the CSV header and the quant report now say how far the trend was scaled
  (`prior: intensity trend from the controls, scaled x2.01 to these residuals; prior df 3.2`).
  `proteomics-toolkit` v26.8.0 makes the same change, and a new golden holds the two to each other
  across unpaired, paired and trend designs. See "The intensity prior: shape from the controls, level
  and weight from the design" in `docs/differential-analysis.md`.

## Performance

## Breaking Changes

- **Moderated p-values under the default intensity-trend prior changed.** A prior shaped on the
  controls became less optimistic than before. A paired or within-subject contrast with its prior
  shaped on the design groups became more powerful. An unpaired contrast on design groups barely
  moved. On the serum cohort this was checked against, the one protein that cleared FDR 0.05 in the
  within-subject trend over weeks 0-12 still did (p 5.3e-7 to 1.0e-6). The help for
  `--prior-from-groups` stopped saying to use it only to reproduce an older result, since the source
  now sets the trend's shape and not its level. To reproduce a result from dotnet-v26.27.0 or
  earlier, run that version. The peptide-count prior did not change: it stays DEqMS as published.
