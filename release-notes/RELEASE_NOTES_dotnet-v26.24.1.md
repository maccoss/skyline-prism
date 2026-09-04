# Skyline-PRISM (C#) dotnet-v26.24.1 Release Notes

A one-fix release, so that dotnet-v26.24.0's `.sky.zip` support is actually findable: the button that
opens the document picker still advertised `.sky` alone, which is how it was reported as missing.

Nothing else changed. If you are on 26.24.0 already, the only difference is the label and the tooltip;
if you are on anything older, this carries all of 26.24.0 with it.

## Bug Fixes

- **The Inputs tab now says it accepts `.sky.zip`.** dotnet-v26.24.0 taught the tool to take shared
  document archives and set the file picker's filter accordingly, but the button that opens the picker
  still read "Add Skyline document (.sky)..." and its tooltip still said "Pick a .sky file" - so the
  archives looked unsupported, which is exactly how it was reported. The button now reads
  "Add Skyline document (.sky / .sky.zip)...", the tooltip names the archive case and why an archive is
  extracted first, and the input table in `docs/skyline-tool.md` gains the row it was missing. Nothing
  functional changed: the filter, the extraction and the batch labelling were already right.
