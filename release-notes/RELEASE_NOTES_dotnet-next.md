# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

- **A contrast between subjects could account for subjects sampled more than once.** For a comparison
  such as sex or onset site, where each subject gave several samples, neither existing design was
  right. Unpaired counted every sample as independent evidence, which understated the standard error,
  and Paired could not be fitted at all, because the subject is nested in the group and its block
  absorbed the contrast. A new **Blocked by subject** design (`prism differential --design blocked
  --subject COL`, and under Compare in the Differential pane) modeled a subject's samples as
  correlated, as limma does: one intra-subject correlation shared by every feature
  (`duplicateCorrelation`), then generalized least squares (`lmFit(block=, correlation=)`) and the
  usual moderation under any variance prior. It was pinned to R's limma 3.68.5 by a new golden
  (`blocked.json`, from `generate_blocked.R`). On a 7,602-peptide, 15-patient ALS CSF cohort the
  result matched limma to 5e-12 on p, at a correlation of 0.441, and `check_blocked_cohort.py`
  repeats that comparison on any output directory. On simulated null data with a correlation of 0.5,
  the false-positive rate at 0.05 was 4.3%, against 22.8% for the unpaired test. Results, the CSV
  header and the quant report named the correlation and counted subjects per arm, and
  `quant_parameters` recorded `block_correlation` and the `subject` column - the latter for the
  paired and within-subject trend designs as well. An arm with fewer than two subjects was refused,
  however many samples it held: with one subject per arm the fit would have been the unpaired one
  under another name. The detection view said that its tests did not account for repeated samples
  under this design.

## Bug Fixes

- **A column of whole numbers is no longer fitted as a number just because it parses as one.** An
  integer-coded subject ID or batch used as a covariate was entered as one centered column - a
  straight-line effect of an arbitrary label - with nothing said. Now a column of numbers is
  categorical when it holds whole numbers that repeat across samples and its name contains a whole
  word such as patient, subject, participant, id, batch, plate, site, group, cycle, set or run, and
  numeric otherwise - so `Patient age` in decimals, `Run order`, an integer age and a dose stay
  numeric, whatever the size of the cohort.
  Adjust for moved out of the collapsed Method section into Comparison, and each ticked column now
  gets its own box with a Numeric | Categorical toggle, the reason for its type, and - before the
  run - a note when the design will drop or skip it. The type can also be set with
  `--covariate-type COLUMN=numeric|categorical`; results, `quant_parameters` (`covariate_types`) and
  recorded commands name the type each covariate was fitted as. A categorical covariate nested in
  the groups - a patient ID under a sex contrast - is dropped by name instead of failing with a
  generic rank-deficiency error, and a group-labeling column forced numeric is fitted with a
  warning. A categorical covariate partly confounded with the groups - some of its values in one arm
  only - is kept, with a note that the contrast is then estimated only from the samples whose value
  occurs in both arms. On a 63-sample ALS cohort, "adjusting" a sex contrast for patient IDs 1-15 had been
  reporting 78 peptides as a numeric covariate; it now reports that the covariate is nested and
  dropped, and the 86 unadjusted hits. When a categorical covariate's levels use up the residual
  degrees of freedom, the refusal now names it and its column count.

- **The Trend designs no longer vanish from the Differential pane.** With no run loaded, or a run with
  no numeric column, both Trend entries were shown when the window opened and then hidden at the
  first click on Compare, with nothing to say they existed or how to get them back. They now stay on
  screen, grayed, with a tooltip saying what is missing - a loaded run, or a numeric column such as
  a time or a dose - and they enable as soon as the run has one.

- **The moderated t no longer claimed more degrees of freedom than the data had.** With few features
  tested and a large or infinite prior df, PRISM read the t off `df.residual + df.prior` - a normal,
  when the prior df was infinite - where limma's `eBayes` caps it at the residual df pooled over the
  features, `pmin(df.residual + df.prior, sum(df.residual))`. On 8 features at 4 residual df with an
  infinite prior, a t of 5.49 had been given p = 4.0e-8; on limma's 32 df it became 4.8e-6. Every design
  and prior got the cap. With a finite prior df, a cohort of hundreds or thousands of features was
  unaffected: its pooled df is far above `df.residual + df.prior`. With an infinite prior df, every
  result moved from a normal onto t on the pooled df - for 3,000 features at 8 residual df, p rose by
  0.1% at t = 3 and 4.5% at t = 8 - and a small targeted panel moved far more. It is pinned to R
  limma's own `eBayes` by a new golden, `ebayes.json`. proteomics-toolkit and the PRISM Differential
  Explorer still use the uncapped df, so where the cap binds their p-values are smaller than PRISM's
  for the same t.

## Performance

## Breaking Changes

- **Covariate types changed for integer-coded columns.** A covariate column of repeating whole
  numbers whose name contains patient, subject, participant, individual, donor, animal, mouse, id,
  batch, plate, site, center, cohort, group, cycle, set or run is now categorical by default where it
  used to be numeric, so an adjusted contrast using one gives different numbers than before.
  `--covariate-type COLUMN=numeric` (or choosing Numeric on the column in the pane) restores the old
  model for that column.
- **A blank covariate value is now missing in text columns too.** A blank, or a value written
  exactly `#N/A`, `N/A`, `n/a`, `NA`, `#NA`, `NULL`, `null`, `NaN` or `Infinity`, used to be a
  category of its own (case matters: `Na`, sodium, is still a value); it is now
  missing, so a covariate missing in a selected sample is skipped, with a message, as a numeric one
  always was. A numeric column holding `#N/A` for its QC pools - as Skyline and Excel write an empty
  cell - was read as text and fitted as categories; it is now numeric.
