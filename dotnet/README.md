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

**The platform split is deliberate: a cross-platform CLI, a Windows-only GUI.** Skyline runs only on
Windows, so the tool's attached mode could never be portable, and the standalone case is covered by the
`prism` CLI on all three platforms. `SkylinePrism.App` therefore stays WPF and is not to be ported to
another UI framework; keep `Core` (and anything a headless user needs) platform-neutral, and let
GUI-only helpers depend freely on the Windows-only `SkylinePrism.Skyline`. See "Design Decisions to
Preserve" in [CLAUDE.md](../CLAUDE.md).

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

## The Skyline external tool (`SkylinePrism.exe`)

The WPF tool runs in two ways from the same executable:

- **From Skyline** (Tools menu): Skyline passes `$(SkylineConnection)`, and the open document is added
  to the **Inputs** list automatically.
- **Standalone**: launched with no connection argument (double-click the installed
  `SkylinePrism.exe`), it starts in standalone mode as a plain PRISM GUI - no Skyline required.

### Inputs: combining several Skyline documents

The Inputs tab holds one row per batch, and the run merges them all into a single cohort so ComBat can
correct between them. Each row's **Batch label** (editable, defaults to the document name and is
de-duplicated automatically) is used as the exported report's file stem, which is exactly what
`DuckDbMerge` reads back as that input's `Source Document` / `Batch` - and what distinguishes the
identically named reference/QC replicates that every plate's document contains.

| Add... | Source | Export mechanism | Format |
|---|---|---|---|
| `Add open document...` | any running Skyline instance | JSON-RPC `ExportReport` | parquet |
| `Add Skyline document (.sky)...` | a closed `.sky` | installed Skyline, headless (SkylineRunner); `SkylineCmd.exe` fallback | parquet via SkylineRunner, else invariant CSV |
| `Add exported report...` | a report you already have | none - read in place | as-is |

Kinds may be mixed in one run; the merge reads parquet and CSV together.

**Headless export: two runners, behind `ISkylineCommandRunner`.**

1. **`SkylineAppRunner` (preferred).** Drives the *installed* Skyline with no UI - the SkylineRunner
   mechanism, reimplemented (`SkylineAppRunner.cs`) because the official shim is a separate download built
   per channel (one binary finds only `Skyline`, another only `Skyline-daily`); ours probes both. It
   launches the ClickOnce `.appref-ms` with `CMD-<guid>`, then exchanges arguments and console output over
   `SkylineInputPipe-<guid>` / `SkylineOutputPipe-<guid>`. ⚠️ There is **no exit code** (the launching
   `cmd.exe` returns at once), so failure is detected from an `Error:` prefix in the piped output.
2. **`SkylineCmdRunner` (fallback).** `SkylineCmd.exe`, discovered under `%LOCALAPPDATA%\Apps\2.0\**` -
   only the ClickOnce *application* folder counts, i.e. the one with `Skyline.exe`/`Skyline-daily.exe`
   beside it (the copy in the sibling `…exe_…` folders fails with *"Unable to find Skyline.exe"*). Newest
   wins; set `PRISM_SKYLINECMD` to override.

Either way the document is opened with `--in` and **never saved**; `--report-add` installs the PRISM
report definitions into Skyline's saved-report list, exactly as the live RPC path does. Each closed
document is loaded twice, once per report, because `--report-name` takes a single report.

**Why the runner matters: parquet.** There is no `--report-format=parquet` (that flag takes only
`csv|tsv`) - Skyline picks parquet from a `.parquet` output extension, so PRISM asks for `.parquet` and
verifies the `PAR1` magic. That works through `SkylineAppRunner` (61 KB parquet vs 946 KB CSV on the same
report) but **fails through `SkylineCmd`**:

```text
Could not load file or assembly 'Parquet, Version=4.0.0.0, …'.
The module was expected to contain an assembly manifest.
```

`SkylineCmd.exe.config` is missing the `<assemblyBinding>` section that `Skyline.exe.config` carries for
Parquet.Net. The managed assembly ships as **`ParquetNet.dll`** (identity `Parquet, Version=4.0.0.0`) and
needs an explicit `<codeBase href="ParquetNet.dll" />`, because a **native** `parquet.dll` (Apache Arrow)
occupies the default probe path; the redirects for its dependency chain (`IronCompress`,
`Microsoft.IO.RecyclableMemoryStream`, `System.Buffers`, `System.Memory`, `System.Numerics.Vectors`,
`System.Runtime.CompilerServices.Unsafe`, `System.Threading.Tasks.Extensions`) are needed too - copying
those eight entries into `SkylineCmd.exe.config` fixes it (verified). Reported upstream; PRISM does not
depend on the fix, since `SkylineCmdRunner` reports `SupportsParquet = false` and goes straight to CSV.

**Replicate metadata.** For an open document PRISM reconstructs Skyline's built-in Replicates grid over
the RPC. For a closed one it reads the `.sky` header for the replicate-targeted Document Annotations,
generates a matching `PRISM-Replicates` view, and exports it with `SkylineCmd`. Both paths build the
view through `ReplicatesReportBuilder`, which **quotes** annotation columns (`"annotation_Plate"`):
Skyline parses `column/@name` as a databinding property path, where the `annotation_` prefix's
underscore is illegal unquoted and aborts the export with *"Invalid character _"*.

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
