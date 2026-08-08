# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

- **Two new views on the Spectrum density tab.** A **View** picker switches the same map between three
  readings: the existing **Heatmap** (isolation window x retention time), a **Load histogram** (how many
  spectra had how many precursors), and **Load over time** (mean precursors per spectrum against RT, with
  the minimum and maximum at each time as dashed bounds around a filled band). The heatmap's color scale
  is set by its busiest cell, so the tail that limits identification - the few spectra carrying many
  co-isolated precursors - is invisible on it; that tail is the right-hand end of the histogram. On the
  load curve the width of the band says whether the load is spread evenly across the m/z range or piled
  into a few windows, which the mean alone cannot show.
  Switching views is instant: all three read the binning already in memory, so there is no re-query and
  no re-bin. Hovering reads out the value under the cursor in the terms of the view being shown. Both new
  views count only spectra that were actually acquired, so a scheduled PRM/MTM window that was not firing
  does not become a spectrum that found nothing, and the load curve breaks at a gap in the schedule
  rather than drawing through zero.

## Bug Fixes

## Performance

## Breaking Changes
