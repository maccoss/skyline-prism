# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time.

## New Features

## Bug Fixes

- **Log output from concurrent exports is now attributable.** With several documents exporting at once,
  the lines forwarded from Skyline's own console ("Opening file...", "Success! Imported Reports", "2%",
  "3%") carried nothing identifying the document, so the log read as a stream of duplicated pairs. Every
  line produced while preparing an input is now tagged with its batch label - `[Plate1]     2%` - applied
  once at the boundary, so it covers open documents, closed ones, pre-exported reports, and the Skyline
  process output alike. Blank lines stay untagged so they still separate sections.

## Performance

## Breaking Changes
