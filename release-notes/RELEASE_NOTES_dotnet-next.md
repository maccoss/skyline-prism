# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

## Bug Fixes

- **A PRISM window left open across a .NET update now asks to be restarted** instead of showing
  stack traces. A window open for five days through a .NET 10.0.14 to 10.0.15 update showed a
  `DllNotFoundException` dialog seven times in six seconds when Windows reported a pen or touch device:
  the update had deleted the folder WPF was loaded from, and WPF had not yet loaded its pen support.
  PRISM now notices when the .NET runtime or its own tool folder has gone from disk since it started,
  shows one message naming the missing file and asking for a restart from Skyline's Tools menu, and
  only logs later faults to `prism-tool.log`. Errors in a normal session are shown as before. See
  `dotnet/src/SkylinePrism.App/StartupFiles.cs`.

## Performance

## Breaking Changes
