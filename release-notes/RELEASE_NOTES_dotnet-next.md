# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

## Bug Fixes

- **The QC plots mixed up their axis labels, titles and tick marks.** Every plot kind is drawn onto one
  reused plot, and clearing it removes only the points - tick generators, axis labels, the title and the
  legend all survived into whatever was drawn next. Switching to PCA after viewing the marker plots left
  EV protein names down the y axis, sample types along the x, and the marker loadings' title above a
  scatter of samples: three plots' worth of chrome over one set of points, each label confidently
  describing something else. The chrome is now reset before every render, and each plot labels both of
  its axes rather than inheriting one.
- **Marker score refused to draw an unreadable plot.** Grouping by a per-subject column gave one column
  per sample - 45 groups of two, with a legend covering the panel - which cannot show whether the study's
  groups separate. Above 12 groups it now says which kind of column to use instead, and the legend
  appears only when few enough groups make it worth the space.
- **Group-by is greyed out for Marker loadings**, which is per-marker rather than per-sample, so grouping
  and filtering replicates could never change it.

## Performance

## Breaking Changes
