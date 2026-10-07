# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

## Bug Fixes

- **A column of whole numbers is no longer fitted as a number just because it parses as one.** An
  integer-coded subject ID or batch used as a covariate was entered as one centered column - a
  straight-line effect of an arbitrary label - with nothing said. Now a column of numbers is
  categorical when it holds whole numbers that repeat across samples and its name contains a whole
  word such as patient, subject, id, batch, plate, cycle, set or run, or when it holds whole numbers
  with at most 10 distinct values; numeric otherwise (so `Patient age` in decimals and `Run order`
  stay numeric).
  Adjust for moved out of the collapsed Method section into Comparison, and each ticked column now
  gets its own box with a Numeric | Categorical toggle, the reason for its type, and - before the
  run - a note when the design will drop or skip it. The type can also be set with
  `--covariate-type COLUMN=numeric|categorical`; results, `quant_parameters` (`covariate_types`) and
  recorded commands name the type each covariate was fitted as. A categorical covariate nested in
  the groups - a patient ID under a sex contrast - is dropped by name instead of failing with a
  generic rank-deficiency error, and a group-labelling column forced numeric is fitted with a
  warning. On a 63-sample ALS cohort, "adjusting" a sex contrast for patient IDs 1-15 had been
  reporting 78 peptides as a numeric covariate; it now reports that the covariate is nested and
  dropped, and the 86 unadjusted hits. When a categorical covariate's levels use up the residual
  degrees of freedom, the refusal now names it and its column count.

## Performance

## Breaking Changes

- **Covariate types changed for integer-coded columns.** A covariate column of whole numbers with
  at most 10 distinct values, or of repeating whole numbers whose name contains patient, subject,
  donor, id, batch, plate, cycle, set or run, is now categorical by default where it used to be
  numeric, so an adjusted contrast using one gives different numbers than before.
  `--covariate-type COLUMN=numeric` (or choosing Numeric on the column in the pane) restores the old
  model for that column.
- **A blank covariate value is now missing in text columns too.** A blank, or a value written
  `#N/A`, `N/A`, `NA`, `NULL`, `NaN` or `Infinity`, used to be a category of its own; it is now
  missing, so a covariate missing in a selected sample is skipped, with a message, as a numeric one
  always was. A numeric column holding `#N/A` for its QC pools - as Skyline and Excel write an empty
  cell - was read as text and fitted as categories; it is now numeric.
