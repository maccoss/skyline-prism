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

- **Protein-level differential analysis now defaults to a variance prior on intensity and peptide
  count, and gives the same p-values as `proteomics-toolkit`.** The new prior,
  `--prior limma-trend-intensity-n-peptides` (**Intensity and peptide count (limma)** in the pane), is
  limma-trend with each protein's log peptide count as a second covariate, fitted to the contrast's
  own residuals. limma-trend alone cannot see peptide counts and tilted by them: on the Verapamil serum
  cohort's random-label nulls, unpaired 5 vs 5, 6.9% of one-peptide proteins' null p-values fell below
  0.05 against 3.0% for proteins with ten or more peptides. The new prior gave 4.4% and 4.2%, stayed
  within half a point of 5% in every intensity fifth, and raised the prior df by 50-60%. In
  spike-ins it found 28.0% of two-fold effects against limma-trend's 23.0% (unpaired 5 vs 5), with the
  gain in well-covered proteins. With no `--prior`, the prior is now chosen by level: this one on the
  protein matrix, which carries `n_peptides`, and limma-trend on the peptide matrix, as before. The
  status line, the results CSV and the quant report name the prior that ran. `proteomics-toolkit`
  26.9.0 uses the same names and the same default, and both tools are pinned to one R limma 3.66.0
  golden, `limma_trend_intensity_n_peptides.json`, which their twin generators write byte for byte the
  same. On the Verapamil cohort, paired week 0 vs 12, default runs of the two tools agreed on all
  3,595 proteins to 1.1e-12 relative in p. See "At protein level: intensity and peptide count" in
  `docs/differential-analysis.md`.
- `--prior` now also accepts underscores for the hyphens (`limma_trend`), as the toolkit
  does.

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

- **Default protein-level moderated p-values changed.** A `prism differential` or Differential pane run
  at protein level with no prior named now uses `limma-trend-intensity-n-peptides` instead of
  `limma-trend`. Peptide-level results are unchanged. A command recorded by the quant report or
  "Show Command Line" names its prior, so one recorded under dotnet-v26.29.0 still reproduces as it
  ran; pass `--prior limma-trend` to get the earlier protein-level default. With too few proteins to
  fit peptide count, the default uses limma-trend and says so; named, the new prior refuses.
- **The peptide-count priors refuse input that is not protein level instead of running another
  prior.** `--prior peptide-count` without peptide counts, or with too few usable ones, used to run the
  global prior with a message; it now stops with an error, as does the new prior. Both also refuse a
  missing, fractional or sub-1 count, counts that are all the same (what a peptide table looks like),
  and repeated feature ids, as `proteomics-toolkit` does.
- **A config with a blank method key produced different numbers.** Treating a blank key as left out
  gave it its documented default instead of whatever the pipeline had made of a null:
  `global_normalization.method:` left blank had run median normalization and ran `rt_lowess`, and
  `batch_estimation.method:` left blank had estimated batches from acquisition-time gaps (`auto`) -
  then ComBat-corrected across them - and estimated none. A blank `output.format:` had written
  `corrected_peptides.` with no extension and wrote parquet. Every other blank method key already ran
  its default. To keep a previous run's numbers, write the method it used explicitly
  (`method: median`, `method: auto`).
