# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

## Bug Fixes

- Rewrote the Skyline Tool Store description. It had led with reference-anchored ComBat, which is opt-in
  and not the default, and it listed none of the tool's panes beyond QC plots. It now describes standard
  ComBat, the Differential, Markers, Dynamic Range, Ion accounting and Spectrum density panes, `.sky.zip`
  input, and the `prism` CLI. The name is now just Skyline-PRISM, here and in `prism --help`, without the
  old acronym expansion.

## Performance

## Breaking Changes
