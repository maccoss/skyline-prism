# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

## Bug Fixes

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
