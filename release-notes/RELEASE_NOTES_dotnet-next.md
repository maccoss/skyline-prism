# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time.

## New Features

- **Shared-peptide handling is now settable in the tool.** The Settings tab exposes
  `parsimony.shared_peptide_handling` (`all_groups` / `razor` / `unique_only`), which decides how a
  peptide matching several protein groups is quantified. It was previously fixed at the `all_groups`
  default with no way to change it from the GUI, while the checkbox label said "razor/parsimony" -
  implying a razor assignment that was not actually being applied. The checkbox is now labelled for what
  it does (group indistinguishable proteins), and the new selector is disabled when it is off, since
  nothing is shared when every accession is its own group.

## Bug Fixes

- **The intensity-distribution plot now matches between the HTML report and the tool.** The HTML QC
  report drew one arbitrary colour per sample - a rainbow carrying no information - while the tool's QC
  tab coloured the same plot by sample type. The report was in fact being handed the sample types and
  discarding them; it now colours by type with a legend, like the PCA and RT-lowess plots beside it and
  like the tool. Type colours are shared, so Skyline's "Standard"/"Quality Control" and PRISM's
  `reference`/`qc` render identically whichever view you are looking at.

- **Acquisition-time batch estimation now explains itself.** When batches are inferred from gaps between
  acquisition times, the run log records the threshold, the median gap, which of the two rules produced
  it, and the actual gaps that started each batch - matching what the Python implementation already
  logged. Previously batches simply appeared with no way to audit the decision.
- **...and warns when that estimate is probably wrong.** On an evenly spaced (continuously acquired) run
  the IQR term collapses and the threshold falls back to a floor only **10% above the typical spacing**,
  so a single slightly longer gap - a wash, a blank, a queue pause - starts a new batch. That case is now
  called out explicitly in the log. If the samples were not really run in separate batches, set
  `batch_estimation.method: none` or supply a real batch annotation; otherwise ComBat corrects between
  batches that do not exist.

- **"min peptides" in the tool implied a filter it never was.** The Settings tab showed
  `protein_rollup.min_peptides` as "min peptides" beside the protein rollup method, which reads as
  "proteins with fewer than N peptides are discarded". It is a method-switch threshold: groups below it
  are rolled up by a simple sum instead of the configured method (median polish is meaningless on one or
  two peptides) and flagged `low_confidence` - **no protein is dropped**. Relabelled with an explanation,
  and the behaviour is now pinned by tests so the wording and the code cannot drift apart.

## Performance

- **Documents are now exported in parallel.** With several inputs the tool exports more than one at a
  time instead of strictly one after another - measured **5.2 s versus ~23 s** for three documents, since
  each export is dominated by Skyline's startup and single-threaded document load. The degree is budgeted
  against installed RAM rather than core count (each headless export is a whole Skyline process with the
  document loaded; a 5 MB `.sky` with a 116 MB `.skyd` peaked at ~1.26 GB), spending at most 60% of RAM
  and capping at 4. Machines without the memory to overlap safely keep exporting one at a time.

## Breaking Changes
