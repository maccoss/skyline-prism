# Skyline-PRISM (C#) dotnet-v26.29.0 Release Notes

**Default differential p-values changed in this release.** The default variance prior for the moderated
t became limma-trend. The previous default, the lab's intensity trend shaped on the QC and reference
pools, was calibrated on average but not by intensity: on null data it gave too many false positives
among bright features and too few among dim ones, and it found fewer true effects. limma-trend was
calibrated at every intensity in every design tested (see Breaking Changes). Two smaller changes also
moved numbers. The moderated t's degrees of freedom were capped as limma caps them, which mattered with
an infinite prior df or a small panel. Integer-coded label columns used as covariates, such as a
patient ID or a batch number, became categorical.

To reproduce a result from dotnet-v26.28.0, pass `--prior intensity-trend` (a quant report's recorded
command already named its prior), and `--covariate-type COLUMN=numeric` for an integer-coded label
covariate, which an earlier command did not record. In the Differential pane, choose **Intensity
trend, toolkit (deprecated)** under Prior and **Numeric** on the covariate. Two changes had no switch:
where the degrees-of-freedom cap bound, p-values still differed, and a covariate value that was blank
or written `#N/A`, `NA` or `NULL` became missing rather than a category of its own, so a covariate
holding one in a selected sample was skipped. proteomics-toolkit and the PRISM Differential Explorer
still defaulted to the intensity trend at this release, so default runs of the three tools differed.

This release also added a design for contrasts between subjects who were each sampled more than once.

## New Features

- **A contrast between subjects could account for subjects sampled more than once.** For a comparison
  such as sex or onset site, where each subject gave several samples, neither existing design was
  right. Unpaired counted every sample as independent evidence, which understated the standard error,
  and Paired could not be fitted at all, because the subject was nested in the group and its block
  absorbed the contrast. A new **Blocked by subject** design (`prism differential --design blocked
  --subject COL`, and under Compare in the Differential pane) modeled a subject's samples as
  correlated, as limma does: one intra-subject correlation shared by every feature
  (`duplicateCorrelation`), then generalized least squares (`lmFit(block=, correlation=)`) and the
  usual moderation under any variance prior. It was pinned to R's limma 3.68.5 by a new golden
  (`blocked.json`, from `generate_blocked.R`). On a 7,602-peptide, 15-patient ALS CSF cohort the
  result matched limma to 5e-12 on p, at a correlation of 0.441, and `check_blocked_cohort.py`
  was added to repeat that comparison on any output directory. On simulated null data with a
  correlation of 0.5, the false-positive rate at 0.05 was 4.3%, against 22.8% for the unpaired test.
  Results, the CSV header and the quant report named the correlation and counted subjects per arm,
  and `quant_parameters` recorded `block_correlation` and the `subject` column - the latter for the
  paired and within-subject trend designs as well. An arm with fewer than two subjects was refused,
  however many samples it held: with one subject per arm the fit would have been the unpaired one
  under another name. The detection view said that its tests did not account for repeated samples
  under this design.

## Bug Fixes

- **A column of whole numbers stopped being fitted as a number just because it parsed as one.** An
  integer-coded subject ID or batch used as a covariate had been entered as one centered column - a
  straight-line effect of an arbitrary label - with nothing said. A column of numbers became
  categorical when it held whole numbers that repeated across samples and its name contained a whole
  word such as patient, subject, participant, id, batch, plate, site, group, cycle, set or run, and
  stayed numeric otherwise - so `Patient age` in decimals, `Run order`, an integer age and a dose
  were numeric, whatever the size of the cohort. Adjust for moved out of the collapsed Method section
  into Comparison, and each ticked column got its own box with a Numeric | Categorical toggle, the
  reason for its type, and - before the run - a note when the design would drop or skip it. The type
  could also be set with `--covariate-type COLUMN=numeric|categorical`; results, `quant_parameters`
  (`covariate_types`) and recorded commands named the type each covariate was fitted as. A
  categorical covariate nested in the groups - a patient ID under a sex contrast - was dropped by name
  instead of failing with a generic rank-deficiency error, and a group-labeling column forced numeric
  was fitted with a warning. A categorical covariate partly confounded with the groups - some of its
  values in one arm only - was kept, with a note that the contrast was then estimated only from the
  samples whose value occurred in both arms. On a 63-sample ALS cohort, "adjusting" a sex contrast
  for patient IDs 1-15 as a number had reported 78 hits; the covariate was then reported as nested
  and dropped, with the 86 unadjusted hits. When a categorical covariate's levels used up the
  residual degrees of freedom, the refusal named it and its column count.

- **The Trend designs stopped vanishing from the Differential pane.** With no run loaded, or a run with
  no numeric column, both Trend entries were shown when the window opened and then hidden at the
  first click on Compare, with nothing to say they existed or how to get them back. They stayed on
  screen, grayed, with a tooltip saying what was missing - a loaded run, or a numeric column such as
  a time or a dose - and enabled as soon as the run had one.

- **The moderated t stopped claiming more degrees of freedom than the data had.** With few features
  tested and a large or infinite prior df, PRISM had read the t off `df.residual + df.prior` - a
  normal, when the prior df was infinite - where limma's `eBayes` caps it at the residual df pooled
  over the features, `pmin(df.residual + df.prior, sum(df.residual))`. On 8 features at 4 residual
  df with an infinite prior, a t of 5.49 had been given p = 4.0e-8; on limma's 32 df it became
  4.8e-6. Every design and prior got the cap. With a finite prior df, a cohort of hundreds or
  thousands of features was unaffected: its pooled df was far above `df.residual + df.prior`. With an
  infinite prior df, every result moved from a normal onto t on the pooled df - for 3,000 features at
  8 residual df, p rose by 0.1% at t = 3 and 4.5% at t = 8 - and a small targeted panel moved far
  more. It was pinned to R limma's own `eBayes` by a new golden, `ebayes.json`. proteomics-toolkit
  and the PRISM Differential Explorer still used the uncapped df at this release, so where the cap
  bound their p-values were smaller than PRISM's for the same t.

## Breaking Changes

- **The default variance prior became limma-trend, and the intensity trend was deprecated.** The
  intensity trend - proteomics-toolkit's estimator, shaped on the QC and reference pools and the
  default up to dotnet-v26.28.0 - was calibrated on average but tilted by intensity: on random-label
  nulls the brightest fifth of features reached 6-9% false positives at p < 0.05 and the dimmest only
  2-5%, because the pools carried technical variance only while study samples added biology that
  was roughly flat on the log scale. It also halved the prior's weight: in simulation it found 52% of
  true two-fold effects at BH 5% where limma-trend found 71% (unpaired, 5 vs 5). limma-trend -
  limma's `trend=TRUE`, still an intensity-dependent prior, but with its curve fitted to the
  contrast's own residuals - was calibrated at every intensity in every design tested: unpaired,
  paired, within-subject and between-subject trends, on the `cohort` fixture, the Verapamil serum
  cohort and simulation. Every default p-value changed. The pane labeled the new default **Intensity
  trend (limma)**, and the status line, CSV header and quant report named it `limma-trend`. The old
  estimator remained available as `--prior intensity-trend` (**Intensity trend, toolkit
  (deprecated)** in the pane), and every result that used it said it was deprecated. A recorded
  command always named its prior, so one recorded earlier kept that prior - but not necessarily the
  rest of its model: the degrees-of-freedom cap applied where it bound, an integer-coded label
  covariate became categorical unless `--covariate-type COLUMN=numeric` was added, and a blank or
  `#N/A` covariate value became missing (see below). `--prior-from-controls` and
  `--prior-from-groups` applied only to the old estimator: beside any other prior they were noted
  rather than silently ignored, and the pane showed **from controls** only beside it, with help that
  no longer suggested setting QC sample types in Skyline to enable it. A new test pinned the property
  the default was chosen for, calibration by intensity, on a seeded simulated null.
  `docs/differential-analysis.md` ("Why limma-trend is the default") has the measurements.

- **Covariate types changed for integer-coded columns.** A covariate column of repeating whole
  numbers whose name contained patient, subject, participant, individual, donor, animal, mouse, id,
  batch, plate, site, center, cohort, group, cycle, set or run became categorical by default where it
  had been numeric, so an adjusted contrast using one gave different numbers than before.
  `--covariate-type COLUMN=numeric` (or choosing Numeric on the column in the pane) restored the old
  model for that column.

- **A blank covariate value became missing in text columns too.** A blank, or a value written exactly
  `#N/A`, `N/A`, `n/a`, `NA`, `#NA`, `NULL`, `null`, `NaN` or `Infinity`, had been a category of its
  own (case mattered: `Na`, sodium, stayed a value); it became missing, so a covariate missing in a
  selected sample was skipped, with a message, as a numeric one always had been. A numeric column
  holding `#N/A` for its QC pools - as Skyline and Excel write an empty cell - had been read as text
  and fitted as categories; it was read as numeric.
