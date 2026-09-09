# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

## Bug Fixes

- A completed run no longer ends with a spurious error dialog. Closing the tool could raise
  `System.DllNotFoundException` at
  `<CrtImplementationDetails>.ModuleUninitializer.SingletonDomainUnload`, reported as a modal
  "Skyline-PRISM error" even though the run had finished and every output was written. The fault is
  in WPF's own `DirectWriteForwarder.dll` resolving the Visual C++ runtime as the process exits -
  no PRISM code is on the stack, and `AppDomain.UnhandledException` cannot keep the process alive -
  so a fault arriving after shutdown has begun is now written to `prism-tool.log` and not shown. A
  background failure during a run still raises its dialog.

- Version strings no longer carry a spurious fourth component. `prism --version` printed
  `prism 26.24.2.0`, the QC report footer read `PRISM v26.24.2.0`, and `parameters.json` recorded
  `"pipeline_version": "26.24.2.0"` - for a release tagged `dotnet-v26.24.2`. PRISM versions as
  CalVer `YY.feature.patch`, so there is no fourth number; the trailing `.0` was padding added by
  .NET's `AssemblyVersion`, which is always normalized to four components. All three now print
  `26.24.2`.

## Performance

## Breaking Changes
