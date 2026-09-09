# Skyline-PRISM (C#) dotnet-v26.24.3 Release Notes

Two fixes for things reported from real use, neither of which ever put a number or an output file at
risk.

The one you would have noticed: a run that finished perfectly could still end with a modal
**"Skyline-PRISM error"** when you closed the tool. The fault was in WPF's own teardown, not in
PRISM, and it fired after every output had been written - so the dialog was the only part of it that
reached you. It is now logged and not shown.

The other is cosmetic but confusing: every version string carried a fourth component that PRISM's
scheme does not have, so a release tagged `dotnet-v26.24.2` reported itself as `26.24.2.0`.

No processing behavior differs from 26.24.2, and nothing else user-facing changed.

## Bug Fixes

- **A completed run no longer ends with a spurious error dialog.** Closing the tool could raise
  `System.DllNotFoundException` at
  `<CrtImplementationDetails>.ModuleUninitializer.SingletonDomainUnload`, reported as a modal
  "Skyline-PRISM error" even though the run had finished and every output was written. The reported
  case was a 224-sample cohort that produced 66,379 peptides and 6,659 proteins in 4m 54s, wrote
  every output file, and logged `Done.` - and only then, on close, raised the dialog. If you saw
  this, your results were fine.

  The fault is in WPF's own `DirectWriteForwarder.dll` resolving the Visual C++ runtime as the
  process exits - no PRISM code is on the stack, and `AppDomain.UnhandledException` cannot keep the
  process alive - so a fault arriving after shutdown has begun is now written to `prism-tool.log`
  and not shown. A background failure during a run still raises its dialog. One case changes
  meaning: if you close the tool mid-run and confirm stopping it, a real failure from the task
  still unwinding is logged rather than shown - `prism_run_*.log` still records it.

- **Version strings no longer carry a spurious fourth component.** `prism --version` printed
  `prism 26.24.2.0`, the QC report footer read `PRISM v26.24.2.0`, and `parameters.json` recorded
  `"pipeline_version": "26.24.2.0"` - for a release tagged `dotnet-v26.24.2`. PRISM versions as
  CalVer `YY.feature.patch`, so there is no fourth number; the trailing `.0` was padding added by
  .NET's `AssemblyVersion`, which is always normalized to four components. All three now report the
  version exactly as tagged - `26.24.3` for this release.

  One deliberate exception: a `parameters.json` written by an earlier release still records the
  padded form, and `prism qc -d` on that output directory still shows it in the report footer. A
  report states the version that produced its numbers, not the one re-rendering them.

## Upgrading

As with every release, PRISM's version is part of the stage-cache fingerprint and of the export
sidecar, so the first run in an **existing** output directory re-exports from Skyline and recomputes
every stage rather than reusing what is there. That is deliberate - a release can change a rollup's
arithmetic without leaving a trace in the config - but it is worth knowing before you re-run a large
cohort on the strength of a cosmetic fix. If you have `Export ion counts` enabled, budget for it:
that export is roughly 30x the cost of the standard one.

Documents already extracted from a `.sky.zip` are **not** re-extracted. Extraction reuse keys on the
archive's own size and timestamp, never on the PRISM version, because nothing about unzipping depends
on which release did it.
