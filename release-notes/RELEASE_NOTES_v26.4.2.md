# Skyline-PRISM v26.4.2 Release Notes

Patch release: surfaces which actual samples contribute to the Reference and QC median CV numbers (in the console log and in `qc_report.html`), and fixes overlapping legend / annotation in the CV-distribution plots.

## New Features

### Reference and QC sample lists shown in console and qc_report.html

Made it explicit which actual samples contribute to the "Reference median CV" and "QC median CV" numbers, since the same words can mean different things in different places (the user's metadata `Sample Type` column vs. the YAML `sample_annotations.reference_pattern` / `qc_pattern` regexes).

- **Console**: a one-time log block prints the reference and QC sample names (and their counts) once early in the run, with a note that the YAML patterns are used only when no metadata file is provided.
- **`qc_report.html`**: a collapsible `<details>` block under "Dataset Summary" lists the same samples plus a short explanation of where they come from (Skyline `Standard` → reference, `Quality Control` → QC, `Unknown` → experimental).

- **Files modified**: `skyline_prism/cli.py`, `skyline_prism/validation.py`

## Bug Fixes

### Fixed: overlapping legend and stats annotation in CV-distribution plots

The "Before / After Normalization" CV-distribution plots used a matplotlib legend AND a separate floating "Median / % under threshold" text box, both positioned independently. Matplotlib's default `loc="best"` legend placement frequently picked the same corner as the text box, producing overlapping labels in `qc_report.html` and saved plots.

Fix: removed the redundant floating text box in `plot_cv_distribution` and `plot_cv_comparison_wide` and folded its unique information (percent under threshold, percent under 10%) into the legend labels. Each subplot now has a single annotation element instead of two competing for the same screen space, so the collision is structurally impossible regardless of data distribution. The legend conveys all the same information: `Median: 33.1%` and `Threshold: 20.0% (23.6% under)` (and `Threshold: 20.0% (47.5% < 20.0%, 12.1% < 10%)` in the more detailed variant).

- **Files modified**: `skyline_prism/visualization.py`

## Performance

<!-- none yet -->

## Breaking Changes

<!-- none yet -->
