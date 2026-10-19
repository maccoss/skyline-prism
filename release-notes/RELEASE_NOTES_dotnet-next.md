# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

- **The QC report shows peak quality in acquisition order, so a degraded or different run stands
  out across a cohort.** The new "Peak Quality by Acquisition Order" section scores Skyline's own
  shape correlation, isotope dot product, retention time and FWHM for every precursor in every run, as
  a robust z against that precursor's median across runs, and plots each run's median score and
  10-90% range in the order the runs were acquired. Runs whose median exceeds twice the cohort's are
  labeled and listed beside the cohort's typical values, and a second plot stacks each metric's
  per-run median so the one that moved can be read straight down. On a 96-run cohort it singled out
  exactly the two Internal Control injections - median shape correlation 0.66 against 0.82, eluting
  0.12 min early - the same two runs MSstats+'s anomaly scores found, with a deterministic score
  pinned to a numpy/scipy golden instead of an isolation forest. It is computed from `merged_data/`
  each time the report is made (about 20 s for 30 million transition rows) and can be turned off
  with `qc_report.peak_quality: false`.
- **`sample_metadata.csv` now carries `acquired_time` and `run_order`** for every cohort whose report
  has Skyline's Acquired Time, after the four resolved columns, which keep their positions. Run order
  is each replicate's place in the acquisition sequence - the same number the peak-quality plot gives
  it - so the injection order no longer has to be typed into Skyline as an annotation to be available
  downstream.

## Bug Fixes

- An output folder copied over an older copy of itself no longer has stray merged data read as real
  rows. A copy that adds and overwrites files but never deletes them keeps any `merged_data/` partition
  the newer run no longer wrote; a real cohort folder held three partitions where its merge had written
  two, putting 9.9 million duplicated rows (10,494 peptides) in front of every reader of `merged_data/` -
  the Spectrum density view, detection, ion accounting and the QC report's peak-quality section - with
  nothing to say so. Each merge now writes `merged_data/_manifest.json` listing its partitions and
  files, and a `merged_data/` that does not match it is not read: the error names every stray, missing
  or changed file and says how to fix it (the peak-quality section reports it as a failed section; the
  rest of the report is unaffected). `prism run` rebuilds such a folder, and every stage below it,
  instead of stopping. A `merged_data/` written by an earlier release is checked against the row count
  in `merged_data.cache.json` instead, which catches the same case (39,789,408 rows on disk against
  29,910,240 recorded, in the folder that was found). See `docs/output_files.md`.
- Corrected documentation that contradicted the code, with no change to any output. `docs/output_files.md`
  and `SPECIFICATION.md` listed a `Coeluting` column the bundled `Skyline-PRISM` report has never exported,
  and `docs/output_files.md` omitted the `Background` column it does export (nothing reads it).

## Performance

## Breaking Changes
