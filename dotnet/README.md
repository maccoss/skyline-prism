# Skyline-PRISM (C# / .NET 8)

A C# port of the Python `skyline_prism` package, living side by side with it. Provides a
cross-platform `prism` CLI (Windows / Linux / macOS) and a Windows Skyline external tool
(WPF + JSON-RPC), reproducing the Python pipeline's numeric results.

## Projects

| Project | TFM | Role |
|---|---|---|
| `SkylinePrism.Core` | `net8.0` | algorithms, IO, QC (cross-platform) |
| `SkylinePrism.Cli` (`prism`) | `net8.0` | the CLI (cross-platform) |
| `SkylinePrism.Skyline` | `net8.0-windows` | Skyline JSON-RPC + report driver (Windows) |
| `SkylinePrism.App` (`SkylinePrism.exe`) | `net8.0-windows` | WPF external tool (Windows) |
| `SkylinePrism.Tests` | `net8.0` | unit + cross-language parity tests |
| `SkylinePrism.Tests.Windows` | `net8.0-windows` | RPC / WPF smoke tests |

`SkylinePrism.CrossPlatform.slnf` is the OS-agnostic subset (Core + Cli + Tests) built and
tested on Linux/macOS; the full `SkylinePrism.sln` adds the Windows-only projects.

## Build & test

```bash
# Cross-platform subset (Linux/macOS/Windows)
dotnet test dotnet/SkylinePrism.CrossPlatform.slnf

# Full solution (Windows)
dotnet test dotnet/SkylinePrism.sln
```

The parity tests read committed golden fixtures under `tests/fixtures/` produced by the
Python pipeline (see `tests/fixtures/README.md`). Linux hosts need `libfontconfig1` and
`libfreetype6` for the ScottPlot/SkiaSharp QC plots; DuckDB.NET ships its own native engine.

## CLI

```bash
prism run -i plate1.csv plate2.csv -o out/ -c config.yaml   # full pipeline
prism merge plate1.csv plate2.csv -o merged.parquet          # merge only
prism qc -d out/                                             # (re)generate qc_report.html
prism config-template -o config.yaml                         # emit a config template
```

## Package the Skyline external tool

Use the ship gate, which tests -> packages -> launch-verifies (always test before shipping):

```powershell
pwsh dotnet/build/package-and-verify.ps1
# runs the full suite, builds dotnet/publish/SkylinePrism.zip, then extracts and launches the exe
```

Or the individual steps:

```bash
dotnet msbuild dotnet/build/package.proj /p:Configuration=Release   # -> dotnet/publish/SkylinePrism.zip
pwsh dotnet/build/verify-tool.ps1                                   # smoke-launch the packaged zip
```

`verify-tool.ps1` extracts the zip to a clean directory and runs `SkylinePrism.exe`; the WPF
MainWindow (hence ScottPlot/SkiaSharp) loads at startup, so a missing/broken dependency shows up as an
assembly/XAML load error in `%LOCALAPPDATA%\SkylinePrism\prism-tool.log`. A failed Skyline connection
from the dummy arg is expected and ignored - only dependency/XAML load failures fail the check.

The tool zip is **framework-dependent** — install the **.NET 8 Desktop Runtime**
(`winget install Microsoft.DotNet.DesktopRuntime.8`) once, then install the zip via Skyline's
Tools > Tool Store > Install from file. When reinstalling over a running copy, **close the tool
first** - Skyline can leave a partial extraction (locked files) that drops `deps.json` and DLLs,
which then fails to load; reinstalling with the tool closed fixes it.

## CI

- `.github/workflows/dotnet-ci.yml` — builds + runs the parity suite on ubuntu/macos/windows
  (cross-platform subset) and packages the tool zip on Windows. Scoped to `dotnet/**`.
- `.github/workflows/dotnet-release.yml` — on a `dotnet-v*` tag, verifies both version sources
  (`Directory.Build.props` and the tool-inf manifest) match the tag, tests, and publishes a GitHub
  Release with the Skyline tool zip (`SkylinePrism.zip`) plus framework-dependent `prism` CLI
  archives per platform (`prism-win-x64.zip`, `prism-win-arm64.zip`, `prism-linux-x64.tar.gz`,
  `prism-linux-arm64.tar.gz`, `prism-osx-x64.tar.gz`, `prism-osx-arm64.tar.gz`). Both artifacts are
  framework-dependent — users install the .NET 8 runtime (Desktop Runtime for the Skyline tool).

The Python CI (`ci.yml`, `release.yml`) is unchanged and runs independently.
