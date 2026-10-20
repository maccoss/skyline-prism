# Skyline-PRISM (C#) dotnet-v26.30.0 Release Notes

**Default numbers changed in this release, in three places.** Library assist
(`transition_rollup.method: library_assist`) used a new fit by default, which removed a fragment as
interference only when its excess was beyond the cohort's measured fragment noise; on FLARE-Extended it
cut protein QC CV from 30.2% to 20.8%. Protein-level differential analysis defaulted to a variance prior
on intensity and peptide count, and gave the same p-values as `proteomics-toolkit` 26.9.0. And the
moderated t fitted proteins with missing values on the samples where they were observed, as limma's
`lmFit` does, instead of dropping them; on complete data nothing changed (see Breaking Changes).

To reproduce a result from dotnet-v26.29.0, set `transition_rollup.library_fitting_method:
median_polish` (an earlier run's `parameters.json` already names its fit, so `--from-provenance`
reproduced it as it ran), and pass `--prior limma-trend` at protein level - in the Differential pane,
choose **Intensity trend (limma)** under Prior. The Skyline tool had no control for the library-assist
fit and did not keep one from a loaded `parameters.json`, so an earlier library-assist run could be
reproduced only from the command line. The missing-value fit had no switch.

This release also added a peak-quality-by-acquisition-order section to the QC report, and stopped
reading a `merged_data/` folder that no longer matched what its merge had written.

## New Features

- **Library assist kept its resistance to interference and lost most of its noise: its default fit
  removed a fragment as interference only when the excess was beyond the cohort's measured fragment
  noise, and every peptide's precision was checked against the signal the fit kept.** On
  FLARE-Extended (93 runs, 7 reference + 7 QC pool injections) protein within-subject SD fell from
  0.965 to 0.556 (sum: 0.502) and protein QC CV from 30.2% to 20.8% (sum: 19.3%). Under synthetic
  interference on one fragment of 30% of peptides, proteins with most of their peptides hit kept 88% of
  their fold change against sum's 79%, and the interfered peptides 91% against sum's 75% - while
  sum's CVs *improved*, because an interferent that is the same in every pool injection reads as
  signal. The new `library_fitting_method: noise_gated_ratio`, the default for `method:
  library_assist`, fitted each sample's scale as kept fragment signal over library intensity and
  removed a fragment only when it was above twice its prediction and more than `library_noise_sigma`
  (default 3) SDs of fragment noise at that predicted area. The noise was measured within the reference
  and QC pools before the fit - across all samples when no pool had 3 injections - and saved as
  `library_assist_noise.json` and in `parameters.json`; it took 5 s on FLARE. The earlier fit was 43%
  noisier than the signal it kept allowed: at low intensity noise alone pushed fragments past twice
  their prediction, so it removed fragments that were not interfered. The new
  `library_assist_precision.csv` and the QC report's "Library-Assist Precision" section compared each
  peptide's pool CV with what a sum of fragments achieved at the signal the fit kept: the median was
  1.02 (reference) and 0.96 (QC) on FLARE, and about 1.0 in every intensity fifth, but peptides whose
  fit kept 4 or fewer fragments ran at 1.1-1.5. Pinned to a first-principles golden,
  `library_assist/noise_gated.json`, and a new cohort digest. See "Library-Assisted Rollup" in
  `docs/methods.md`.
- **The QC report showed peak quality in acquisition order, so a degraded or different run stood
  out across a cohort.** The new "Peak Quality by Acquisition Order" section scored Skyline's own
  shape correlation, isotope dot product, retention time and FWHM for every precursor in every run, as
  a robust z against that precursor's median across runs, and plotted each run's median score and
  10-90% range in the order the runs were acquired. Runs whose median exceeded twice the cohort's were
  labeled and listed beside the cohort's typical values, and a second plot stacked each metric's
  per-run median so the one that moved could be read straight down. On a 96-run cohort it singled out
  exactly the two Internal Control injections - median shape correlation 0.66 against 0.82, eluting
  0.12 min early - the same two runs MSstats+'s anomaly scores found, with a deterministic score
  pinned to a numpy/scipy golden instead of an isolation forest. It was computed from `merged_data/`
  each time the report was made (about 20 s for 30 million transition rows) and could be turned off
  with `qc_report.peak_quality: false`.
- **`sample_metadata.csv` gained `acquired_time` and `run_order`** for every cohort whose report
  had Skyline's Acquired Time, after the four resolved columns, which kept their positions. Run order
  was each replicate's place in the acquisition sequence - the same number the peak-quality plot gave
  it - so the injection order no longer had to be typed into Skyline as an annotation to be available
  downstream.
- **Protein-level differential analysis defaulted to a variance prior on intensity and peptide
  count, and gave the same p-values as `proteomics-toolkit`.** The new prior,
  `--prior limma-trend-intensity-n-peptides` (**Intensity and peptide count (limma)** in the pane), was
  limma-trend with each protein's log peptide count as a second covariate, fitted to the contrast's
  own residuals. limma-trend alone could not see peptide counts and was tilted by them: on the
  Verapamil serum cohort's random-label nulls, unpaired 5 vs 5, 6.9% of one-peptide proteins' null
  p-values fell below 0.05 against 3.0% for proteins with ten or more peptides. The new prior gave 4.4%
  and 4.2%, stayed within half a point of 5% in every intensity fifth, and raised the prior df by
  50-60%. In spike-ins it found 28.0% of two-fold effects against limma-trend's 23.0% (unpaired 5 vs
  5), with the gain in well-covered proteins. With no `--prior`, the prior was chosen by level: this
  one on the protein matrix, which carries `n_peptides`, and limma-trend on the peptide matrix, as
  before. The status line, the results CSV and the quant report named the prior that ran.
  `proteomics-toolkit` 26.9.0 used the same names and the same default, and both tools were pinned to
  one R limma 3.66.0 golden, `limma_trend_intensity_n_peptides.json`, which their twin generators
  wrote byte for byte the same. On the Verapamil cohort, paired week 0 vs 12, default runs of the two
  tools agreed on all 3,595 proteins to 1.1e-12 relative in p. See "At protein level: intensity and
  peptide count" in `docs/differential-analysis.md`.
- `--prior` also accepted underscores in place of the hyphens (`limma_trend`), as the toolkit
  does.
- **The moderated t fitted proteins with missing values on the samples where they were observed,
  as limma's `lmFit` fits them, instead of testing only complete proteins.** Skyline exports are
  complete, but merging documents with different target lists leaves proteins absent from a plate,
  and every contrast that touched that plate used to drop them. A subject with no value for a protein
  dropped out of that protein's comparison, which was made on the remaining subjects with fewer
  residual df; a protein missing a whole arm was not tested but its residual variance still informed
  the prior, and its df the cap on the t's df (`DifferentialResult.DfPooled`), as in limma. A value
  that was not finite counted as missing. The run said how many proteins were fitted on fewer samples
  and how many had no estimable contrast. This applied to the unpaired, paired and trend designs; the
  blocked design, and the Welch, Student, Mann-Whitney, paired t and Wilcoxon tests, still tested only
  complete proteins. Pinned to R limma 3.66.0 by `limma_ebayes.json` - an arm missing, the reference
  subject missing from a paired design, whole subjects missing from a within-subject trend - which
  twin generators wrote byte for byte the same here and in `proteomics-toolkit`, whose 26.9.0 fitted
  missing values the same way. See "Missing values" in `docs/differential-analysis.md`.
- **PRISM and `proteomics-toolkit` 26.9.0 shared every convention of the moderated t.** Besides the
  missing-value fit above, the toolkit dropped its log pseudocount, adopted R's `lowess()` at its
  default interpolation distance for the peptide-count and intensity-trend priors, and floored zero
  residual variances as limma does - what PRISM already did. Both tools were pinned to the same R
  references: `limma_ebayes.json` for the global and limma-trend priors, the df cap, missing values and
  zero variances, `limma_trend_intensity_n_peptides.json` for the protein-level default, and the new
  `lowess.json` (R `lowess()`) for the LOWESS inside the peptide-count and intensity-trend priors,
  which PRISM's `Lowess.Fit` matched to 2.5e-10. On complete data a default run of either tool gave
  the same p-values with no setting at all. The toolkit-sourced goldens (`intensity_trend.json`,
  `peptide_count_prior.json`, `toolkit_end_to_end.json`) were regenerated from the released 26.9.0,
  run at its defaults, and held PRISM to the toolkit's own p and adjusted p. Where the df cap bound,
  the toolkit's p moved by up to 93%, to limma's (which PRISM already gave). Values that pass
  through a prior's LOWESS moved by at most 1.8e-8, from the toolkit's switch to R's `lowess()`,
  and were compared at a stated 5e-8. Re-pinning also found the toolkit's adjusted p counting
  untested features, fixed in proteomics-toolkit 26.9.0 before its release.

## Bug Fixes

- Stray merged data in an output folder copied over an older copy of itself stopped being read as
  real rows. A copy that adds and overwrites files but never deletes them kept any `merged_data/`
  partition the newer run no longer wrote; a real cohort folder held three partitions where its merge
  had written two, putting 9.9 million duplicated rows (10,494 peptides) in front of every reader of
  `merged_data/` - the Spectrum density view, detection, ion accounting and the QC report's
  peak-quality section - with nothing to say so. Each merge wrote `merged_data/_manifest.json` listing
  its partitions and files, and a `merged_data/` that did not match it was not read: the error named
  every stray, missing or changed file and said how to fix it (the peak-quality section reported it as
  a failed section; the rest of the report was unaffected). `prism run` rebuilt such a folder, and
  every stage below it, instead of stopping. A `merged_data/` written by an earlier release was checked
  against the row count in `merged_data.cache.json` instead, which caught the same case (39,789,408
  rows on disk against 29,910,240 recorded, in the folder that was found). See `docs/output_files.md`.
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
  `--from-provenance`, the earlier run's value. An empty list could still be written `[]`, and `null`
  still cleared a key whose default is none, such as `parsimony.fasta_path`.
- **The run log started recording what was said while the config was read.** An unrecognized
  config key's `WARNING` and the `NOTE` that a moved FASTA had been replaced by its archived copy
  were printed before `prism_run_*.log` was opened, so they had gone to stderr only; a scripted run
  that discarded stderr left a log that read as though every setting had been honored. They were
  written at the top of the log as well, in the order they were said. A run that failed had left a
  log that stopped mid-stage with no reason; it ended with the `Error:` line, and the console said
  where the log was. The log section of `docs/output_files.md`, which still showed the retired
  Python engine's time-stamped format, was rewritten from a real run.
- **The deprecated intensity-trend prior averaged in a group with one observed value, as
  `proteomics-toolkit` does.** Such a group places no point on the trend (it has no sd), but the
  toolkit - the definition of that estimator - averages the trend's value at its mean into the
  feature's prior with weight 1, and PRISM had skipped it. With missing values that moved the prior
  and, on a 2,500-protein test set, p by up to 2e-2; `intensity_trend.json` gained a missing-value
  case, generated from the toolkit. The trend's shape was also fitted over every feature, as the toolkit
  fits it, not only the features tested: one too sparse in the contrast's samples to test still has
  pool replicates that trace noise against intensity. Leaving them out moved the prior df by 2.5% on
  `toolkit_end_to_end.json`'s new `unpaired_feature_only_in_pools` case. Complete data was
  unaffected.
- **The peptide-count prior left out every residual variance at or below limma's floor**
  (1e-5 x the median), not just exact zeros, from its LOWESS on log peptide count. Replicates that
  agree exactly give 0 from one least-squares solve and 1e-29 from another; PRISM's QR left the 1e-29,
  which entered the fit as a log variance of -66 and bent the whole curve, moving p by 4e-3 against
  `proteomics-toolkit` on a test set with three exactly constant proteins. Both tools used the same
  rule.
- Corrected documentation that contradicted the code, with no change to any output. `docs/methods.md`
  described the retired Python engine's library-assist fit (least squares with MAD z-scores, an R²
  and output fields C# never had, and matching by ion type); it was rewritten to describe the two fits
  PRISM has. `README.md` and `docs/parameters.md` called median polish the default transition rollup,
  which has been `sum`.
- Corrected documentation that contradicted the code, with no change to any output. `docs/output_files.md`
  and `SPECIFICATION.md` listed a `Coeluting` column the bundled `Skyline-PRISM` report has never exported,
  and `docs/output_files.md` omitted the `Background` column it does export (nothing reads it).

## Breaking Changes

- **`method: library_assist` gave different numbers by default.** `library_fitting_method`
  defaulted to `noise_gated_ratio` (above), in the CLI and in the Skyline tool alike. Set
  `library_fitting_method: median_polish` to reproduce an earlier library-assist run. An earlier run's
  `parameters.json` names the fit it used, so `--from-provenance` on it still reproduced it as it ran.
  The Skyline tool had no control for the fit and did not keep one from a loaded `parameters.json`,
  so there an earlier library-assist run could not be reproduced; use the command line. A
  noise-gated run's transition rollup also depended on which samples were reference and QC pools, so
  re-typing a pool injection re-ran it.
- **Default protein-level moderated p-values changed.** A `prism differential` or Differential pane run
  at protein level with no prior named used `limma-trend-intensity-n-peptides` instead of
  `limma-trend`. Peptide-level results were unchanged. A command recorded by the quant report or
  "Show Command Line" names its prior, so one recorded under dotnet-v26.29.0 still reproduced as it
  ran; pass `--prior limma-trend` to get the earlier protein-level default. With too few proteins to
  fit peptide count, the default used limma-trend and said so; named, the new prior refused.
- **Moderated results changed wherever the compared samples had missing values.** Proteins with a
  missing value were tested (on their observed samples) rather than dropped, so more proteins
  appeared, the prior was fitted over more of them, and the cap on the t's df counted their reduced df.
  On complete data nothing changed: the fit is the same computation, and every earlier golden passed
  unchanged.
- **The peptide-count priors refused input that is not protein level instead of running another
  prior.** `--prior peptide-count` without peptide counts, or with too few usable ones, used to run the
  global prior with a message; it stopped with an error instead, as did the new prior. Both also
  refused a missing, fractional or sub-1 count, counts that are all the same (what a peptide table
  looks like), and repeated feature ids, as `proteomics-toolkit` does.
- **A config with a blank method key produced different numbers.** Treating a blank key as left out
  gave it its documented default instead of whatever the pipeline had made of a null:
  `global_normalization.method:` left blank had run median normalization and ran `rt_lowess`, and
  `batch_estimation.method:` left blank had estimated batches from acquisition-time gaps (`auto`) -
  then ComBat-corrected across them - and estimated none. A blank `output.format:` had written
  `corrected_peptides.` with no extension and wrote parquet. Every other blank method key already ran
  its default. To keep a previous run's numbers, write the method it used explicitly
  (`method: median`, `method: auto`).
