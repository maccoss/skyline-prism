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

```bash
dotnet msbuild dotnet/build/package.proj /p:Configuration=Release
# -> dotnet/publish/SkylinePrism.zip  (SkylinePrism.exe + tool-inf/ + runtime DLLs)
```

Install the zip via Skyline's Tools > Tool Store > Install from file.

## CI

- `.github/workflows/dotnet-ci.yml` — builds + runs the parity suite on ubuntu/macos/windows
  (cross-platform subset) and packages the tool zip on Windows. Scoped to `dotnet/**`.
- `.github/workflows/dotnet-release.yml` — on a `dotnet-v*` tag, verifies the tool-inf version
  matches the tag, tests, and publishes a GitHub Release with the Skyline tool zip
  (`SkylinePrism.zip`) plus self-contained `prism` CLI archives per platform
  (`prism-win-x64.zip`, `prism-linux-x64.tar.gz`, `prism-osx-x64.tar.gz`, `prism-osx-arm64.tar.gz`).

The Python CI (`ci.yml`, `release.yml`) is unchanged and runs independently.
