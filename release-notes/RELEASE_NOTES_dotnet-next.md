# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

## Bug Fixes

- **A batch's density map could be binned on another batch's isolation windows.** The saved catalog
  (`isolation_schemes.xml`) referenced each batch's scheme by NAME, and kept one library entry per name -
  so when two input documents both named their scheme, say, `SWATH (25 m/z)` but defined different windows
  (a re-acquisition with a wider range, or the scheme edited between plates), reopening the output
  directory gave the second batch the first one's grid. A plausible-looking map of an acquisition that
  never happened. Each batch's own windows are now written inline with it, so they survive exactly;
  catalogs written by earlier releases still load through the name lookup. The scheme picker also
  deduplicates on the window layout instead of the name, so a same-named-but-different scheme is offered
  rather than dropped (labeled with its window count to tell the two apart).

- **Re-running into an existing output directory destroyed what the previous run knew about isolation
  windows.** The run wrote a freshly built catalog over the file, discarding entries for batches it had
  not just collected - and since dotnet-v26.16.0 removed the inclusion-list loader there was no way to
  put a lost scheme back. It now loads and merges: a re-collected batch overwrites its own entry, every
  other entry survives.

- **The density status line called a cell a "spectrum" even when it was not one.** On the approximate
  uniform-bin fallback a cell is a bin, and for a non-DIA acquisition it is a row of a scheme the data was
  not acquired with - the reading the tab's own warnings exist to prevent. It now says "bin" or "row"
  accordingly, and `docs/skyline-tool.md` no longer claims the tab refuses to draw a map for a non-DIA
  document: it draws one and warns.

## Performance

## Breaking Changes
