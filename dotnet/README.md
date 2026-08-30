# Skyline-PRISM (C# / .NET 10)

The PRISM engine: a cross-platform `prism` CLI (Windows / Linux / macOS) and a Windows Skyline
external tool (WPF + JSON-RPC). Originally a port of the `skyline_prism` Python package, which it
reproduced numerically and has since replaced — that engine was retired after `v26.4.4`.

## Projects

| Project | TFM | Role |
|---|---|---|
| `SkylinePrism.Core` | `net10.0` | algorithms, IO, QC (cross-platform) |
| `SkylinePrism.Cli` (`prism`) | `net10.0` | the CLI (cross-platform) |
| `SkylinePrism.Skyline` | `net10.0-windows` | Skyline JSON-RPC + report driver (Windows) |
| `SkylinePrism.App` (`SkylinePrism.exe`) | `net10.0-windows` | WPF external tool (Windows) |
| `SkylinePrism.Tests` | `net10.0` | unit + cross-language parity tests |
| `SkylinePrism.Tests.Windows` | `net10.0-windows` | RPC / WPF smoke tests |

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

The parity tests read committed golden fixtures under `tests/fixtures/`, produced by the original
Python pipeline and now frozen (see `tests/fixtures/README.md`). Linux hosts need `libfontconfig1` and
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

### Dynamic Range tab

Log10 abundance against abundance rank - Skyline's Relative Abundance shape - over the **corrected**
matrices only (`corrected_proteins.parquet` / `corrected_peptides.parquet`), since the point is the
dynamic range of the result the user will actually analyse. Abundances are averaged across the selected
replicates on the **linear** scale and only then log-transformed (a mean of logs is a geometric mean, not
the plotted quantity), and a row with no measurement in the selection is dropped rather than plotted at
zero. Ticking a different replicate set **re-ranks**, because the ordering depends on what is averaged.

**Click-to-select.** Clicking a point calls `SetSelectedElement` with that element's locator, navigating
the user's Skyline document tree - at peptide level the **peptide** node, so its chromatograms come up.
Locators are read from Skyline (`GetLocations`, cached per level) rather than built by string surgery, so
protein naming and modified sequences stay Skyline's problem; `SkylineReportDriver.LocatorKeys` indexes
each element under its display name, the locator's trailing segment, `<protein>/<peptide>`, and the
`sp|…|…` components, because the plot is keyed on the PRISM matrices' identifiers. In standalone mode the
click just reports what was hit.

⚠️ **PRISM's parsimony is not Skyline's grouping.** PRISM groups proteins itself (optionally FASTA- and
enzyme-aware), so a peptide's PRISM group need not correspond to a protein node in the document. Peptide
selection therefore tries `<protein>/<peptide>` for each of the peptide's PRISM groups **in order**, then
falls back to the first occurrence of that sequence in the document tree - and the status line says when
the fallback decided it, rather than silently selecting a protein the plot did not mean. The status line
also lists **every** group a shared peptide belongs to, since membership is a fact about the peptide while
which group quantifies it is a `parsimony.shared_peptide_handling` decision.

**Protein lists** (`ProteinListSet`) highlight curated sets in their own colours, toggled on and off.
They are stored **per user** at `%LOCALAPPDATA%\SkylinePrism\protein-lists.json`, not per project, so the
same lists follow the user everywhere; `ProteinListWindow` edits a clone and only commits on OK. Matching
is deliberately forgiving - curated lists arrive keyed on whatever identifier their source used - so a
member matches against accession, gene name and protein name, with `sp|`/`tr|` prefixes, `_HUMAN` suffixes
and `-2` isoform numbers all tolerated. List order is priority when a protein appears in two.

Labels (gene name, falling back to accession; the sequence at peptide level) are toggled from the plot's
right-click menu: none, the Skyline selection, or whole lists - the same place Skyline puts its plot
options. Each label sits off the curve with a **leader line** back to its point, flipping to the left near
the right edge, because a label centred on its point buries the very point it names.

### Spectrum density tab

How many peptide precursors were detected in each DIA spectrum of a run: retention time across,
precursor m/z up, color = precursors per cell. Each map row is one **real isolation window**, so a cell
is a spectrum and its value is how many precursors that spectrum had to resolve. A precursor contributes
a count to every window containing its m/z (more than one only for a staggered/overlapping scheme, where
it genuinely was fragmented twice), for every RT bin its integrated peak `[Start Time, End Time]` spans.
The **max q-value** control (`Detection Q Value`) decides what counts as detected; hovering reads out
whatever is under the cursor, in the terms of the view being shown.

**Three views of the same map**, chosen from the **View** drop-down (all read the same binning, so
switching between them is instant - no re-query, no re-bin):

| View | Shows | Reads |
|---|---|---|
| **Heatmap** | isolation window x retention time, color = precursors per spectrum | *where* the load is |
| **Load histogram** | how many spectra had how many precursors | how the load is *distributed* |
| **Load over time** | mean precursors per spectrum against RT, with min/max as dashed bounds around a filled band | how the load moves over the *gradient* |

The map's color scale is set by its busiest cell, so the tail that actually limits identification - the
few spectra carrying many co-isolated precursors - is exactly what the heatmap cannot show; that is the
right-hand end of the histogram. On the load curve the *width of the band* is as informative as the mean:
a wide one says the load is piled into a few windows at that time, a narrow one says it is spread evenly,
and the mean alone cannot tell those apart.

Both summaries count only cells that were **acquired**. For a scheduled method a window that was not
firing is not a spectrum that found nothing, and counting it would pile a meaningless spike onto the
histogram's bin 0 and drag every mean down. `LoadOverTime` returns NaN at such a time and each series is
broken there rather than drawn through zero - a scheduled gap is not an idle instrument.

**Where the windows come from.** This is the awkward part, and worth knowing:

| Document's Full-Scan isolation scheme | What PRISM does |
|---|---|
| An explicit scheme (`SWATH (25 m/z)`, a VW scheme, a custom one) | Its `<isolation_window start= end=/>` list is read straight from the `.sky` and used. Exact, automatic, and the picker is locked - there is nothing to choose. |
| `Results only` (**the normal setting for a DIA analysis document**) | Skyline reads the windows from the raw files at import and persists them **nowhere** - not in the `.sky`, and not in any report column (`ChromatogramExtractionWidth` is the product-ion extraction width, not the isolation window). So PRISM has Skyline read them back out of a data file - see below. |
| Data files unreachable / no Skyline installed | Falls back to the user's saved Skyline isolation schemes, if any were captured; failing that, a uniform m/z grid labelled `uniform N Th bins (approximate)` so a cell is never mistaken for a spectrum. |

**Reading the windows out of a raw file.** PRISM cannot open vendor formats, but Skyline can, and its
`--full-scan-isolation-scheme` flag accepts *a data file path* in place of a scheme name - the
command-line form of Transition Settings > Full-Scan > Isolation scheme > Add > *Import from a data file*.
`SkylineIsolationImporter` uses it against a **throwaway `--new` document** in a temp directory, parses
the `<isolation_scheme>` out of it, and deletes it; the user's own document is never opened or modified.
Measured: 167 windows read from a 5.2 GB Thermo `.raw` on a network share in ~10 s (Skyline reads scan
headers, not the whole file). Recorded data-file paths go stale when data is archived or moved, so a path
that no longer resolves is retried beside the document before giving up.

This matters more than "one bin width vs another" suggests: those 167 windows are 3.0014 Th wide and
start at **400.4319**, because the edges are deliberately placed in the peptide *forbidden zones* (widths
an integer multiple of ~1.0005 m/z, the averagine spacing). A uniform 3 Th grid starting at 400 would be
offset ~14% of a window and would cut through precursor clusters the scheme was designed to keep intact.

Because the windows have to be known *later*, when the tab may be opened on an old output directory with
no Skyline running, a run writes what it learned to `isolation_schemes.xml` in the output directory: each
input's resolved scheme (imported or document-declared), plus every isolation scheme saved in the attached
Skyline (`GetSettingsListNames`/`GetSettingsListItem` on `IsolationSchemeList`) as a manual fallback.

Picking the wrong scheme is visible rather than silent: precursors that fall outside every window are
counted (never clamped into the nearest one) and the status line warns with the percentage.

**Timing: this all happens when the user clicks Run PRISM**, before the pipeline starts - deliberately, as
that is when the raw data is most likely still where the document says it is. The windows are resolved once
and written to the output directory; nothing later depends on the data files still being reachable.

### DIA only, and why

**This tab is for DIA.** Skyline's importer looks for a *repeating* isolation cycle, so it works only for
DIA; on anything else it fails with `No repeating isolation scheme found in <file>`. PRISM reads
`transition_full_scan/@acquisition_method` first and skips the import (and the ~10 s Skyline launch) for
PRM/DDA/SureQuant, recording the method in `isolation_schemes.xml` so the tab can warn that the map's
reading does not apply.

PRM and MTM were briefly supported by loading the Thermo inclusion list that went to the instrument. That
is gone. Reading what an acquisition actually did means walking the data file's scan headers - every MS2
scan's isolation center, width and RT - and **that belongs in Skyline or ProteoWizard, not in an external
tool**. There is no Skyline route to it either: `--exp-isolationlist-instrument` is the outbound direction
(what to load *onto* an instrument), and nothing in the CLI or the RPC surface enumerates acquired
isolation windows. Rather than carry a half-answer, the tool now says plainly when a document is not DIA.

**The scheduled-window model is kept**, because *dynamic DIA is DIA*
([Pino/Searle et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10517878/)): 8 x 8 m/z windows whose
positions shift along the gradient to track where peptides are eluting, ~300 m/z covered at any instant.
An isolation window is therefore **an m/z range crossed with an RT range** - `IsolationWindow.RtStart/RtStop`
are `NaN` for static DIA (the cycle repeats all gradient long) and carry a firing interval otherwise. What
that buys, once something can produce such windows:

- A precursor is credited to a window only if its peak eluted **while that window was firing**
  (`IndicesCovering`, not `IndicesContaining`).
- Cells outside a window's interval render as **not acquired** (NaN, drawn as a gap), so a `0` always means
  *acquired and nothing detected* - the window that fired and found nothing, which is worth seeing.
- The RT axis spans the **schedule**, not just the detections.
- The distinguishing property of dynamic DIA is that **the same m/z is covered by different windows at
  different times**, which forced choosing the source window **per cell rather than per m/z row** when
  rasterizing; a per-row choice shows one segment and blanks the rest.

Nothing currently produces a scheduled window - every real scheme PRISM can read today is always-on - so
these paths are exercised by tests (`ScheduledWindowTests`, `DynamicDiaTests`) rather than by the app.
That is deliberate: the primitives are subtle enough that re-deriving them later would be the expensive
part, and the tests keep them from rotting meanwhile.

The map itself is built from the merged data (`merged_data/`, or the single `merged_data.parquet` of a
pre-dotnet-v26.12.0 run - `MergedDataset` opens either) - which the pipeline leaves in place and re-uses
as its merge cache - so the tab works both right after a run and when the output box is simply pointed
at a previous run's directory. Computation lives in `PrecursorDensity` / `IsolationScheme` (Core) - including
`PrecursorsPerSpectrumHistogram()` and `LoadOverTime()`, the two summaries - and drawing in
`PlotRenderer.DrawPrecursorDensity` / `DrawPrecursorLoadHistogram` / `DrawPrecursorLoadOverTime`;
`MainWindow.Density.cs` is only the wiring. Real schemes are not
obliged to be uniform, gapless or non-overlapping, so the map keeps explicit `[Low, High)` rows and
rasterizes onto a uniform grid at draw time (heatmap cells must be equal-height). This is a port of the
m/z x RT heatmap in [Skyline-Cadenza](https://github.com/maccoss/skyline-cadenza), fed by the PRISM
report instead of a DIA-NN report.

## Performance and memory

The pipeline is built to keep peak memory flat as a cohort grows, because it normally runs on the same
workstation as the Skyline instance holding the documents. Two things carry that:

- **Stage 1 partitions instead of sorting.** `merged_data/` is hive-partitioned on a hash of the peptide
  column, so the rollup sorts and streams one partition at a time rather than ordering the whole cohort.
  See `MergedDataset` for the sizing trade — it was measured in both directions and the intuitive answer
  was wrong.
- **Every DuckDB connection is bounded, and every large read streams.** See `DuckDbTuning`; the defaults
  it overrides are individually capable of consuming the machine.

Stage 2 (the transition rollup) is single-threaded and the largest stage, because concurrent DuckDB
reads corrupt memory in this binding. Two ways of replacing its reader were built and measured, and
**both lose to what ships** — so this is a measured ceiling, not an unexplored one.
**[`STAGE2_THROUGHPUT.md`](STAGE2_THROUGHPUT.md)** has those numbers, the concurrency configurations
that were tried and crashed, and why each option is closed — read it before optimizing anything here,
and before assuming a plausible theory about where the time goes.

Runs report per-stage elapsed time and a sorted summary in the run log, so a slow cohort can be
diagnosed from the artifact a user already has.

### Measuring

`bench/Stage2Bench` compares the candidate Stage 2 read strategies on one partition of a real merged
dataset. It is intentionally outside `SkylinePrism.sln`, so it never reaches CI or the shipped package:

```bash
prism merge <report1.parquet> <report2.parquet> -o D:/bench/merged
dotnet run -c Release --project dotnet/bench/Stage2Bench -- D:/bench/merged 3
```

It interleaves its arms rather than running them in blocks, checks that they agree on rows, peptides and
**values accumulated** before comparing timings, and labels any run during which other software was
competing for the machine. The values check exists because an arm here once looked 3.12x faster by
accumulating an index instead of the values — same rows, same peptides, less work — so any new arm must
build the same per-peptide blocks as the others.
That last part is not decoration: an identical configuration measured 2.07 min and 3.70 min hours apart
here, the cause was two Skyline instances that had started in between, and several hours of analysis
were built on the gap before anyone read the process list. **Ratios survive a busy machine; absolute
throughput numbers do not.**

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

The tool zip is **framework-dependent** — install the **.NET 10 Desktop Runtime**
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
  framework-dependent — users install the .NET 10 runtime (Desktop Runtime for the Skyline tool).

