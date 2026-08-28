# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

- **The QC report now states what produced it.** A new "Analysis Information" block at the top of
  `qc_report.html` gives the PRISM version, processing date, computer, and source files - read back
  from the run's own `parameters.json`, so a report regenerated later by a newer build still names the
  version that made the numbers - followed by a **Processing Parameters** table (transition -> peptide
  rollup, peptide normalization, sample outlier detection, parsimony, peptide -> protein rollup,
  protein normalization, batch correction) and the exact re-runnable config as YAML in a collapsible
  block. Only the keys the selected methods actually read are listed, so nothing reads as being in
  play when it is not. This restores - and extends - what the Python report showed.

  The settings shown always come from the output directory's own `parameters.json`, never from the
  config handed to `prism qc` - `-c` there carries report options only, and filling the sections a
  QC-only config omits with defaults would attribute the numbers to settings that never ran. A
  directory with no (or an unreadable) `parameters.json` still renders: the report says the producing
  version is unrecorded rather than naming the binary that drew the page.

## Bug Fixes

- **The QC report's peptide "after" numbers were missing batch correction.** It read
  `peptides_log2_internal.parquet`, which since dotnet-v26.15.0 is post-normalization and
  **pre-ComBat** (the protein arm branches from it). So every peptide CV, every peptide before/after
  plot and the whole validation verdict measured normalization alone - while the panels were
  labelled "normalized + corrected" and the protein half of the same report showed a fully corrected
  matrix. It now reads `corrected_peptides.parquet`, the peptide arm's actual output, exactly as the
  protein side reads `corrected_proteins.parquet`. On the `mini` fixture with ComBat on, the
  reference median CV reported for the peptide "after" changes from 117.6% to 34.1%; expect
  peptide-level CV numbers to move on any run with batch correction enabled.

- **Validation no longer FAILS because the QC and reference samples improved by different amounts.**
  The relative variance reduction (RVR = QC improvement / reference improvement) was a pass criterion,
  so an `RVR > 2` failed the run and the report called it "possible overfitting to the reference". The
  two control groups are different materials injected at different amounts - whichever started with
  more excess variance has more of it to remove - so an asymmetric improvement is ordinary and the
  ratio cannot establish a cause. RVR is now reported as a NOTE and decides nothing. The verdict fails
  only on evidence of damage: the QC CV got worse, or the QC and reference samples collapsed onto each
  other in PCA space. The report also spells out those two criteria, so a FAILED banner points at
  something actionable.

## Performance

## Breaking Changes

- **The Python implementation has been removed.** PRISM began as the `skyline-prism` Python package;
  the C# tools replaced it and reproduced its numbers to 1e-9 on the deterministic stages. The engine,
  its tests, its PyPI packaging and CI, the Python-era Skyline external tool and the helper scripts are
  all gone from this repository, and `README-python.md` with them.

  **Nothing about the `prism` CLI or the Skyline tool changes** - this removes a second implementation
  nearly everyone had already stopped using, not any feature of the one that ships.

  If you still need it, every Python release stays downloadable: `pip install skyline-prism==26.4.4`,
  or the [`v26.4.4` tag](https://github.com/maccoss/skyline-prism/releases/tag/v26.4.4) (any earlier
  `v*` tag works too). Those versions are unmaintained and frozen at that release's behaviour.

  Consequences worth knowing:
  - Config keys that only ever existed in Python (`method: adaptive`, `library_fitting_method:
    least_squares`, the per-plot `qc_report.plots.*` toggles, ...) are unchanged in their handling:
    PRISM still warns on the key or aborts on the method choice by name, so an old config tells you
    what it hit rather than silently doing something else. They are no longer listed in
    `docs/parameters.md`.
  - The cross-engine golden fixtures under `dotnet/tests/fixtures/` stay and still gate every release;
    they are now a frozen independent-implementation baseline, regenerable only from the `v26.4.4` tag.
  - `dotnet/PORTING_STATUS.md` is kept as a historical record of what was and was not carried over.

