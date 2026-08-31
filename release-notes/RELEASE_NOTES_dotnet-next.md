# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

## Bug Fixes

- **The export concurrency budget now counts memory that is already in use.** It sized itself against
  installed RAM alone, so a Skyline already holding one of the documents about to be exported - the
  normal case, since PRISM runs as a Skyline external tool - was invisible to it. That document costs
  roughly 2x its `.sky` resident, about 21 GB for the 10.6 GB plate in the logged run, and the old budget
  counted that memory as available twice. PRISM now takes the smaller of the installed and free bounds,
  and names the free figure in the log line, so "one document at a time on a 64 GB machine" is
  auditable rather than looking like a bug. The zero-fits warning - which explains that a starved
  Skyline never recovers and asks the user to close other Skyline windows - is now reachable on a full
  machine, which is the condition under which a Skyline was actually starved.

## Performance

## Breaking Changes
