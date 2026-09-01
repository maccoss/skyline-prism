# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

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
