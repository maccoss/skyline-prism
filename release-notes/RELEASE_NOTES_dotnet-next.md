# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

## Bug Fixes

- **The Inputs tab now says it accepts `.sky.zip`.** dotnet-v26.24.0 taught the tool to take shared
  document archives and set the file picker's filter accordingly, but the button that opens the picker
  still read "Add Skyline document (.sky)..." and its tooltip still said "Pick a .sky file" - so the
  archives looked unsupported, which is exactly how it was reported. The button now reads
  "Add Skyline document (.sky / .sky.zip)...", the tooltip names the archive case and why an archive is
  extracted first, and the input table in `docs/skyline-tool.md` gains the row it was missing. Nothing
  functional changed: the filter, the extraction and the batch labelling were already right.

## Performance

## Breaking Changes
