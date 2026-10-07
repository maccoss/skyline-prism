# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

## Bug Fixes

- **The Trend designs no longer vanish from the Differential pane.** With no run loaded, or a run with
  no numeric column, both Trend entries were shown when the window opened and then hidden at the
  first click on Compare, with nothing to say they existed or how to get them back. They now stay on
  screen, grayed, with a tooltip saying what is missing - a loaded run, or a numeric column such as
  a time or a dose - and they enable as soon as the run has one.

## Performance

## Breaking Changes
