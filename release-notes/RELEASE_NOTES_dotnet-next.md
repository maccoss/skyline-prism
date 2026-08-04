# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time.

## New Features

## Bug Fixes

## Performance

- **Documents are now exported in parallel.** With several inputs the tool exports more than one at a
  time instead of strictly one after another - measured **5.2 s versus ~23 s** for three documents, since
  each export is dominated by Skyline's startup and single-threaded document load. The degree is budgeted
  against installed RAM rather than core count (each headless export is a whole Skyline process with the
  document loaded; a 5 MB `.sky` with a 116 MB `.skyd` peaked at ~1.26 GB), spending at most 60% of RAM
  and capping at 4. Machines without the memory to overlap safely keep exporting one at a time.

## Breaking Changes
