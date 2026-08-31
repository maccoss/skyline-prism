# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

## Bug Fixes

- **The "not even one export fits" warning no longer cries wolf.** dotnet-v26.23.1 made free memory a
  bound on export concurrency, and because free memory is always at or below installed, that bound is
  the one that decides - so a 64 GB machine with 33 GB free budgeted 19.8 GB against an export needing
  21.2 GB, reported that nothing fitted, and told the user to close Skyline windows for an export that
  fitted in free memory with 12 GB to spare. The warning also quoted the installed budget while doing
  it, which made its arithmetic read as false. Falling short of the budget and not fitting the machine
  are now separate questions: the budget still decides whether exports may overlap and stays
  conservative, while the warning fires only when one export genuinely does not fit what is free, with
  4 GB held back for the OS and Skyline's own baseline.

## Performance

## Breaking Changes
