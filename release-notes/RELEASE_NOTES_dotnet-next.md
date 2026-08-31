# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

## Bug Fixes

- **A Skyline that crashes on headless launch is now noticed in seconds instead of minutes, and the
  export is retried rather than downgraded to CSV.** Skyline intermittently dies about a second after a
  headless launch with an `UnauthorizedAccessException` inside ClickOnce's
  `IsolationInterop.CreateActContext` - raised while the CLR builds the activation context, so before any
  Skyline code runs. PRISM did not look for the process until its startup window expired, so a
  one-second crash cost the full three minutes and was then reported as `did not start within 180s
  (... Another export may be saturating the machine - try exporting one document at a time)` - advice
  that cannot help, in runs that were already exporting one document at a time. Because the parquet
  retries are budgeted in attempts rather than seconds, the crash consumed the whole budget in timeouts
  and the export fell back to CSV. PRISM now polls for the launched process every second, reports
  `started and then exited` naming the actual ClickOnce fault and pointing at the Windows Application
  event log, and spends its retries on the failure instead of on waiting - which in every observed case
  succeeded on the next attempt. A single missed process enumeration is not treated as a crash, so a
  Skyline legitimately minutes into opening a large document is still waited out.

## Performance

## Breaking Changes
