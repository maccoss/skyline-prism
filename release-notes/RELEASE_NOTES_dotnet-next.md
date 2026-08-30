# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

## Bug Fixes

## Performance

## Breaking Changes

- **PRISM now targets .NET 10, so you must install the .NET 10 runtime before this version will start.**
  The Skyline tool needs the .NET 10 **Desktop** Runtime; the `prism` CLI needs the base .NET 10 Runtime.
  The download links in the README are updated.

  The reason is support, not features: .NET 8's LTS window ended in November 2026, while .NET 10 is
  supported to November 2028. The artifacts stay framework-dependent, which is what keeps them at
  20-45 MB rather than bundling a runtime per platform.

  Verified beyond the test suite, because a runtime change moves GC and marshalling behavior and that is
  where DuckDB.NET has broken here before: a 47M-row, 93-sample cohort was run end to end through the
  streaming merge and the Stage 2 reader with no failure.
