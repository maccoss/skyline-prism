# Skyline-PRISM (C#) dotnet-v26.5.0 Release Notes

First public release of the C# (.NET 8) implementation of Skyline-PRISM, shipped side by side with
the Python reference package.

## New Features

### `prism` CLI (cross-platform)
- The full PRISM pipeline as a native command: merge → transition→peptide rollup → normalization →
  ComBat batch correction → protein parsimony/rollup → QC report.
- Runs on Windows, Linux, and macOS, for both **x64 and ARM64**.
- Commands: `run`, `merge`, `qc`, `compare`, `config-template` (see `README.md`).

### Skyline external tool (Windows)
- A WPF tool that drives a running Skyline over JSON-RPC to export the report, run the pipeline, and
  show interactive QC plots. Install via Skyline's **Tools → Tool Store → Install from file**.

### Parity
- Reproduces the Python reference pipeline's numeric results (exact parity on the deterministic
  stages).

## Distribution
- Published to **GitHub Releases** on a `dotnet-v*` tag — independent of the Python `v*` / PyPI
  cadence.
- Six **framework-dependent** CLI archives — `prism-win-x64`, `prism-win-arm64`, `prism-linux-x64`,
  `prism-linux-arm64`, `prism-osx-x64`, `prism-osx-arm64` — plus `SkylinePrism.zip` for the Skyline
  tool.
- **Prerequisite:** install the .NET 8 runtime once. The .NET 8 **Desktop** Runtime on Windows
  covers both the CLI and the Skyline tool; the base .NET 8 Runtime is enough for the CLI on
  Linux/macOS. See `README.md` for per-platform install commands.

## Notes
- Versioning: CalVer `YY.feature.patch`, aligned with the Python version line but tracked on the
  independent `dotnet-v*` tag counter. The version lives in `dotnet/Directory.Build.props` and
  `dotnet/src/SkylinePrism.App/tool-inf/info.properties` (kept in lockstep; verified against the tag
  at release time).
- The Python `skyline-prism` package remains the reference implementation and continues on PyPI.
