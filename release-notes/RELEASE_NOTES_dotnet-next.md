# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

- **MS2 signal accounting in the QC report: how much of the integrated MS2 signal an analysis actually
  labels.** A new section plots, per replicate, the MS2 signal the run assigns to a peptide, with a
  profile line per selected protein list showing how much of that belongs to each panel. Enable it with
  `qc_report.ms2_signal.enabled: true`, and name panels with `qc_report.ms2_signal.protein_lists` /
  `protein_list_files`.

  The quantity is a **union, not a sum**. Adding up transition areas double-counts: a DIA isolation
  window co-isolates tens of peptides, and two fragments whose extraction windows overlap are the same
  detector counts credited twice - which can push "assigned" past what the instrument acquired. Each
  distinct region of MS2 signal space - (isolation window, extraction window, integration bounds) -
  counts once. On a real cohort replicate that removes 17.7% of the naive sum: 23,245 rows where
  Skyline exported one shared peptide per protein assignment, and 5,623 genuine shares between
  co-isolated peptides, which are reported separately because they mean different things.

  Totals **nest rather than partition**: a region claimed by a list peptide counts for that list and
  for the assigned total, and two lists may both claim it - the question is what portion of the signal
  a panel accounts for, not which panel owns a peptide.

  All values are **raw**: linear Skyline peak areas from `merged_data/`, untouched by normalization,
  ComBat or marker adjustment. The pipeline outputs supply peptide identity only, never magnitude.

  Off by default because it costs one extra streaming pass over the merged table. Results are cached as
  `ms2_signal_accounting.parquet` (and `ms2_signal_lists.parquet`), so `prism qc -d` replots without
  recomputing and keeps working on a directory whose `merged_data/` was cleaned up.

  The largest bar is **not** acquired MS2 signal - that needs the instrument files, and no Skyline
  export can supply it (`TicArea` is one value per replicate and MS1 by construction). It is the signal
  Skyline integrated for the document's targets, and it is labelled that way, because reading it as the
  acquired total would turn unknown coverage into apparently complete coverage.

- **The extraction window Skyline used is read from the document.** `SkyDocumentInfo` now reports the
  `transition_full_scan` product-ion settings and evaluates them to the m/z range Skyline actually
  extracted over, transcribed from Skyline's own `GetDenominator` / `GetFilterWindow` and applied the
  way it applies them. The stated `product_res` is the **+/- tolerance**: `centroided res="10"` extracts
  +/-10 ppm, `qit res="0.7"` extracts +/-0.7 m/z. Documents that do not say enough - an Orbitrap setting
  with no calibration m/z - report the tolerance as unknown rather than a plausible default.

## Bug Fixes

- **The QC report says "Raw" and "Corrected" instead of "Before" and "After", and the CV plots no
  longer look like they are about median normalization.** "Before / After" left the reader to work out
  before and after *what*, and the answer depends on the configuration - with RT-LOWESS, ComBat and
  marker normalization all enabled, "After" is all three. The two panels have always been the raw
  rollup and the arm's final written output, so they are now named after the files they come from:
  `peptides_rollup` / `proteins_raw` is **Raw**, `corrected_peptides` / `corrected_proteins` is
  **Corrected**. Renamed in the section headings, the image captions, the plot titles and the CV
  tables. Separately, the CV histogram legend read "Before median 34.4%", which looked like a claim
  about median normalization rather than the median of the CV distribution; it now reads
  "Raw: median CV 34.4%".
- **The PCA legend no longer sits on top of samples.** It was drawn in a fixed corner, so on a real
  cohort it covered replicates - which matters on the one plot people use to find the odd sample out.
  It now goes in whichever corner holds the fewest points.

- **The "not even one export fits" warning no longer cries wolf.** dotnet-v26.23.1 made free memory a
  bound on export concurrency, and because free memory is always at or below installed, that bound is
  the one that decides - so a 64 GB machine with 33 GB free budgeted 19.8 GB against an export needing
  21.2 GB, reported that nothing fitted, and told the user to close Skyline windows for an export that
  fitted in free memory with 12 GB to spare. The warning also quoted the installed budget while doing
  it, which made its arithmetic read as false. Falling short of the budget and not fitting the machine
  are now separate questions: the budget still decides whether exports may overlap and stays
  conservative, while the warning fires only when one export genuinely does not fit what is free, with
  4 GB held back for the OS and Skyline's own baseline.

## Performance

## Breaking Changes
