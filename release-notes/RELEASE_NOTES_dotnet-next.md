# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

- **A two-arm contrast between subjects can now account for subjects sampled more than once.** The
  new **Blocked by subject** design (`prism differential --design blocked --subject COL`) is for
  comparisons such as sex or onset site when each subject contributes several samples. Before it, the
  only choices were to treat those samples as independent (unpaired), which understates the standard
  error, or a paired design, which cannot be fitted when the subject is nested in the group. A
  subject's samples are now modelled as correlated, as limma does it: one intra-subject correlation
  shared by every feature (`duplicateCorrelation`), then generalized least squares
  (`lmFit(block=, correlation=)`) and the usual moderation under any variance prior. Pinned to R's
  limma 3.68.5 by a new golden (`blocked.json`, from `generate_blocked.R`); on a 7,602-peptide ALS CSF
  cohort the result matched limma to 5e-12 on p (`check_blocked_cohort.py` repeats that comparison
  on any output directory). On simulated null data with a correlation of 0.5 the
  false-positive rate at 0.05 was 4.3%, against 22.8% for the unpaired test. Results, the CSV header
  and the quant report name the correlation and count subjects per arm. The detection view says that
  its tests do not account for repeated samples under this design.

## Bug Fixes

## Performance

## Breaking Changes
