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

- Refused to read a `merged_data/` directory that no longer holds the rows its merge counted. Before,
  the extra rows were silently included. A copy of an output directory made over an older copy, replacing
  files but removing none, can keep partitions from an earlier merge that had more of them. The number of
  partitions is chosen partly from free memory, so it can differ between runs. On a 96-run cohort this left
  a stale `_pep_bucket=2` holding 9.9M duplicated rows beside 29.9M, and anything that read the whole
  directory counted about a third more precursors than the cohort has. The QC report's peak-quality
  section, ion accounting, the Spectrum density tab and detection in the differential analysis now
  refuse such a directory, with a message naming the leftover partitions; the QC report shows that
  message in place of the section and builds the rest. A re-run into it re-merges and recomputes every stage instead of
  reusing the cached merge. The check compares the parquet footers with `TotalRows` in
  `merged_data.cache.json`. It reads no column data, and took under a second on that cohort.

- Corrected documentation that contradicted the code, with no change to any output. `docs/output_files.md`
  and `SPECIFICATION.md` listed a `Coeluting` column the bundled `Skyline-PRISM` report has never exported,
  and `docs/output_files.md` omitted the `Background` column it does export (nothing reads it).

## Performance

## Breaking Changes
