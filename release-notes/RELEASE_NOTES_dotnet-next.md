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

- **A QC report with no control samples now gives the counts.** It said only that the verdict needs
  ">=2 reference and >=2 QC samples ... not enough of both were found", which reads as a fact about the
  study design - so a cohort that HAD 16 reference and 16 QC samples and lost them produced a report
  indistinguishable from one that never had any. That is what a failed replicate-metadata export does:
  PRISM falls back to inferring sample types from replicate names, and where the names carry no type the
  inference matches nothing, so 32 controls silently became experimental and the dual-control verdict had
  nothing to validate. The report now states "0 reference and 0 QC of 192 samples", and when both are
  zero it names the replicate metadata as the thing to check.
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
