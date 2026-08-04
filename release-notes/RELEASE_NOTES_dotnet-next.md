# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

- **The QC "Group by" value dropdown is now multi-select.** Each value has a tick box, so groups can be
  combined instead of choosing one at a time - the case that motivated it is comparing Quality Control
  against Standard in the control-correlation heatmap, where the useful reading is that each control type
  correlates with itself but not with the other, and the unknowns bury it. Ticking nothing (or everything)
  still shows every sample.
- **Control correlation defaults to the controls.** Selecting that plot with no filter set now ticks the
  control sample types automatically, matching what the HTML QC report has always done - it computes that
  heatmap over reference + QC only. Previously the tool included every sample, which is not a control
  correlation. An explicit selection is never overridden, and columns with no control values (a Condition
  annotation, say) are left alone rather than filtered to nothing.

## Bug Fixes

## Performance

## Breaking Changes
