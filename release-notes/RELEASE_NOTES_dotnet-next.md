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

- **`prism run --from-provenance <parameters.json> -c <overrides.yaml>` stopped ignoring the
  overrides.** The `-c` had been dropped without a word whenever a provenance file was given, so the
  documented way to change one setting of an earlier run re-ran its old settings unchanged. Each key
  the YAML set replaced the earlier run's value and every other key was kept; a subsection merged key
  by key and a list was replaced whole. The run log named the overridden keys, an unrecognized key
  warned as it did with a plain `-c`, and a file that changed no setting said so.
- **A config key left with no value stopped failing the run or quietly changing a setting.** YAML
  reads a blank as null, and PRISM had bound it as written: a blank `enabled:` under
  `batch_correction` turned ComBat off with nothing said, a blank number became 0, a blank method
  name or a list with every entry commented out (`reference_pattern:` alone) failed when sample types
  were assigned, and a bare `transition_rollup:` stopped the run with "Object reference not set to an
  instance of an object". A blank key was treated as left out and kept its default - or, with
  `--from-provenance`, the earlier run's value. An empty list is written `[]`, and `null` still
  cleared a key whose default is none, such as `parsimony.fasta_path`.
- Corrected documentation that contradicted the code, with no change to any output. `docs/output_files.md`
  and `SPECIFICATION.md` listed a `Coeluting` column the bundled `Skyline-PRISM` report has never exported,
  and `docs/output_files.md` omitted the `Background` column it does export (nothing reads it).

## Performance

## Breaking Changes
