# Building a Skyline external tool — a field guide

Everything we've learned building Skyline external tools in C#/.NET across
[skyline-prism](https://github.com/maccoss/skyline-prism) and
[skyline-cadenza](https://github.com/maccoss/skyline-cadenza) — how a tool connects to a running
Skyline, exports reports, reads the document grid, changes settings, reads and *writes* `.blib`
libraries, imports results, and is packaged/installed. It is deliberately broader than what any one
tool uses. **Written for the next AI agent** who has to do this again: the gotchas below each cost real
time to discover.

Reference Skyline version for the CLI surface here: **26.1.1** (get it at runtime with
`--version` / RPC `GetVersion`).

---

## 0. Two ways to write a tool — pick one

| | **Live JSON-RPC tool** (PRISM `SkylinePrism.App`, cadenza) | **Classic report-macro tool** (`skyline-prism/skyline-external-tool`, Python) |
|---|---|---|
| Mechanism | Connects back to the *running* Skyline over a named pipe (JSON-RPC 2.0) | Skyline exports named reports to temp files and launches your program with path macros |
| Interactivity | Bidirectional — read grid, change settings, write libraries, re-import, live | One-shot — read the exported report(s), do work, write output |
| Manifest key | `Arguments=$(SkylineConnection)` | `Report=…`, `Arguments="…$(ReportTempPath)…"` |
| When to use | You need to read/write the live document or show interactive UI | You just need the report data and to emit files |

The classic tool is far simpler and needs no RPC. Reach for the live RPC tool only when you need to
*read the live document beyond a report* or *write back to it*.

Classic manifest (`skyline-external-tool/tool-inf/PRISM.properties`):
```properties
Command=$(ProgramPath(Python,3.12.8))
Arguments="$(ToolDir)/Scripts/run_prism.py" --report "$(ReportTempPath)" --metadata "$(ReportTempPath:Replicates)" --output "$(DocumentDir)/prism-output"
Report=Skyline-PRISM
AuxiliaryReport1=Replicates
InitialDirectory=$(DocumentDir)
```
Macros: `$(ProgramPath(title,version))`, `$(ToolDir)`, `$(DocumentDir)`, `$(ReportTempPath)` /
`$(ReportTempPath:<AuxReportName>)`, and `$(SkylineConnection)` (RPC tools). `info.properties` may add
`PythonVersion=3.12.8` so Skyline provisions the interpreter.

Everything from Section 1 on is about the **live JSON-RPC tool**.

---

## 1. Connecting to a running Skyline (JSON-RPC)

### The vendored RPC client
Copy four files **verbatim** from pwiz `pwiz_tools/Skyline/SkylineTool/` (Apache-2.0):
`SkylineJsonToolClient.cs`, `IJsonToolService.cs`, `JsonToolConstants.cs`, `JsonToolModels.cs`. Keep them
under `external/SkylineTool/` with a README noting the sync procedure (diff against a fresh pwiz checkout
before committing). Consume them two ways:
- **.NET Framework 4.7.2** tool → reference the prebuilt `SkylineTool.dll`.
- **.NET 8+** tool → **link-compile the four `.cs` files** (System.Text.Json is in-box).

### Transport
JSON-RPC 2.0 over a `NamedPipeClientStream`, **message mode**, snake_case JSON:
```csharp
var pipe = new NamedPipeClientStream(".", pipeName, PipeDirection.InOut);
pipe.Connect(5000);
pipe.ReadMode = PipeTransmissionMode.Message;      // MANDATORY — else the response read never completes
using var client = new SkylineJsonToolClient(pipe);
var path = client.GetDocumentPath();
```
The client serializes `{ jsonrpc:"2.0", method, params, id:1 }`, `Flush()`+`WaitForPipeDrain()`, then reads
until `IsMessageComplete`. Errors surface as a typed `JsonRpcException(Code, Message)`.

### ⚠️ The single biggest gotcha: `$(SkylineConnection)` is the *legacy* pipe name
Skyline passes your tool the **legacy ToolService pipe name** as `args[0]`. The JSON-RPC server listens on
a *derived* name. **Connect to `args[0]` directly and the first read fails with
`JsonReaderException: 0x00 is invalid start of value`** (you hit the old binary server). Transform it:
```csharp
// JsonToolConstants: JSON_PIPE_PREFIX = "SkylineMcpJson-"; GetJsonPipeName(name) = prefix + name.Replace("-","")
var jsonName = raw.StartsWith("SkylineMcpJson-") ? raw : JsonToolConstants.GetJsonPipeName(raw);
```
(`SkylineSession.FromArguments`.)

### ⚠️ Connect **per call** — never hold the pipe open
Skyline closes the pipe after each request/response. Reusing a pipe works for the first call and then fails
after idle with the same `0x00` error. Open a fresh pipe every call:
```csharp
public T Execute<T>(Func<ISkylineClient,T> action) {
    using var pipe = new NamedPipeClientStream(".", PipeName, PipeDirection.InOut);
    pipe.Connect(timeout); pipe.ReadMode = PipeTransmissionMode.Message;
    return action(new JsonClientAdapter(new SkylineJsonToolClient(pipe)));
}
```

### Standalone discovery (dev without `args[0]`)
Skyline writes `~/.skyline-mcp/connection-<pid>.json` (`pipe_name`, `process_id`, `connected_at`,
`skyline_version`). Walk them and pick the newest to attach a locally-launched `.exe` to a running Skyline.

### Recommended seam (testability)
Define `ISkylineClient` (the subset of RPC calls you use) + `ISkylineExecutor` (`Execute<T>`), have the
concrete `SkylineSession` implement the executor and forward through a private adapter to
`SkylineJsonToolClient`. Then a `FakeExecutor`/`FakeClient` drives your whole driver in unit tests **with no
live Skyline** — this is how PRISM tests the report/grid/library logic. (Cadenza skips the seam and hands
out the concrete client; add the seam if you want tests.)

---

## 2. Reports

### RPC surface (`IJsonToolService`)
- `ExportReport(name, filePath, culture)` → writes a saved report; **format is chosen by file extension**.
- `ExportReportFromDefinition(ReportDefinition, filePath, culture)` — export an *ad-hoc* report.
- `GetReportRows(name, offset, count, columns, filter, includeMaxLength, culture)` — inline windowed rows.
- `GetReportFromDefinitionRows(ReportDefinition, offset, count, includeMaxLength, culture)`.
- `AddReportFromDefinition(ReportDefinition)` — persist a report into user settings.
- `GetReportDocTopics()` / `GetReportDocTopic(name, dataSource)` — column documentation (see §3).

`culture` = `"invariant"` or `"localized"`. **Always use `"invariant"`** for machine consumption (English
headers without spaces, `.` decimal). `count = 0` returns **shape only** (TotalRows + columns, no rows) — use
it to size a pull, then fetch all rows in a second call.

### `ReportDefinition` shape
```csharp
class ReportDefinition {
  string[] Select;           // column IDs (dotted navigation, see .skyr below)
  string Name;
  ReportFilter[] Filter;     // { Column, Op, Value }
  ReportSort[] Sort;         // { Column, Direction }
  bool? PivotReplicate;      // fold per-replicate result columns into columns (vs. row explosion)
  bool? PivotIsotopeLabel;
  string Uimode;             // "proteomic" | "small_molecules" | "mixed"
  string DataSource;         // "document_grid" | "audit_log" | ...
}
```

### `.skyr` report definitions (ship them, install on demand)
A `.skyr` is a Skyline "view" XML. Ship it next to the tool and install with
`RunCommand(["--report-add=<file>.skyr", "--report-conflict-resolution=overwrite"])`. Column names use dotted
entity navigation; `Results!*` is the per-replicate sublist:
```xml
<view name="PRISM" rowsource="pwiz.Skyline.Model.Databinding.Entities.Transition" sublist="Results!*" uimode="proteomic">
  <column name="Precursor.Peptide.Protein" />
  <column name="Precursor.Peptide.ModifiedSequence.UnimodIds" />
  <column name="Results!*.Value.Area" />
  <column name="Results!*.Value.PrecursorResult.PeptideResult.ResultFile.Replicate.Name" />
</view>
```
A replicate-level view uses `rowsource=...Entities.Replicate`; the first `<column name="" />` is the row's own
identity.

### Export-to-file vs paginate (perf)
`GetReportRows`/`GetReportFromDefinitionRows` **recompute the report from scratch on every page** — paginating a
big report is O(pages × full-report). Prefer `ExportReport*` to write the whole report to disk in one
round-trip (~20× faster on a SEA-AD-class document); use a tiny `GetReportFromDefinitionRows(def,0,200,…)` only
to probe columns.

### PRISM's driver pattern
1. `EnsureReportsInstalled` → install the `.skyr` files (idempotent, overwrite).
2. **Parquet-first**: `ExportReport("PRISM","PRISM.parquet","invariant")`, then validate by checking the `PAR1`
   magic at head+tail; **CSV fallback** (`.csv`) for older Skyline that can't emit parquet.
3. Invariant everywhere.

---

## 3. Reading the document grid (Replicates, annotations)

### ⚠️ The built-in "Replicates" view is **not a named report**
`GetReportRows("Replicates", …)` fails with **"Report not found"** and it isn't in the exportable saved-report
list. Read it as a **document grid** instead:
```csharp
// Columns available on the Replicate entity (built-ins + user annotations):
var detail = client.GetReportDocTopic("Replicate", "document_grid");   // -> ColumnDefinition[] { Name, Description, Type }
var columns = detail.Columns.Select(c => c.Name);

// Rows, one per replicate:
var def = new ReportDefinition { Select = columns.ToArray(), PivotReplicate = false, DataSource = "document_grid" };
var shape = client.GetReportFromDefinitionRows(def, 0, 0, false, "invariant");   // count=0 -> TotalRows
var rows  = client.GetReportFromDefinitionRows(def, 0, shape.TotalRows, false, "invariant");
```

### Queryable document-grid topics (from `GetReportDocTopics`)
`Protein, ProteinResults, Peptides, PeptideResults, Precursors, PrecursorResults, PrecursorResultsSummary,
Transitions, TransitionResults, TransitionResultsSummary, Replicate, AuditLog` — each reports its column count.
`Replicate` carries the built-in replicate columns **plus every user-defined document annotation**.

### Built-in vs annotation columns
Maintain a hard-coded set of built-in Replicate columns (`Replicate, ReplicateName, FileName, FilePath,
SampleName, ModifiedTime, AcquiredTime, SampleType, AnalyteConcentration, BatchName, ReplicateLocator, …`).
**Any column not in that set is a user annotation** — carry those through verbatim (they're the experiment
design: Condition, Subject, Batch, …). PRISM curates the useful built-ins and always keeps the annotations.

### Annotation column naming
When requesting an annotation column in an ad-hoc `.skyr` you must prefix it: `annotation_<Name>` (e.g.
`annotation_Condition`). Because the built-in Replicates view isn't RPC-exportable, PRISM synthesizes a
Replicate `.skyr` on the fly that `<column name="annotation_Batch" />`, installs it with `--report-add`, then
exports it.

---

## 4. Changing document settings & annotations

**Golden rule (per Nick Shulman): anything `SkylineCmd` can do is reachable via JSON-RPC `RunCommand` against
the live document.** So most "change settings" operations are just `RunCommand([...SkylineCmd flags...])`
(§7 lists the flags). Examples: `--pep-min-length`, `--tran-precursor-ion-charges`,
`--tran-product-ion-types`, `--full-scan-acquisition-method`, `--full-scan-isolation-scheme`,
`--annotation-name/-targets/-type/-values`, `--integrate-all`.

RPC-native settings calls also exist: `GetDocumentSettings(file)` / `GetDefaultSettings(file)` (write settings
XML), and the settings-list family `GetSettingsListTypes` / `GetSettingsListNames(type, group)` /
`GetSettingsListItem` / `AddSettingsListItem(type, xml, overwrite)` / `GetSettingsListSelectedItems` /
`SelectSettingsListItems`. `ImportProperties(csvText)` imports annotation values (first column = ElementLocator
paths).

### ⚠️ SkylineCmd aborts the **whole batch** on the first unknown/invalid arg
Send flags **one per `RunCommand` batch** so one bad flag doesn't roll back the rest — **except** fields that
Skyline **mutually validates**, which must be sent **together** or each rejects the other (e.g. DIA
acquisition-method + isolation-scheme; MS1 isotope count + analyzer + ppm). Test empirically.

---

## 5. BLIB spectral libraries (SQLite)

A `.blib` is a BiblioSpec SQLite database. Use `Microsoft.Data.Sqlite`.

### ⚠️ Always `Pooling=False`
`Data Source=<path>;Mode=ReadOnly;Pooling=False` — otherwise the connection lingers in the pool and **locks the
user's `.blib`**. Same for writes.

### Reading fragment spectra
```sql
SELECT s.peptideSeq, s.precursorMZ, s.precursorCharge, s.peptideModSeq, p.peakMZ, p.peakIntensity
FROM RefSpectra s JOIN RefSpectraPeaks p ON p.RefSpectraID = s.id
```
- `peakMZ` = little-endian **doubles** (8 bytes each); `peakIntensity` = little-endian **floats** (4 bytes).
- **Blobs may be zlib-compressed**: if the blob starts with magic `0x78 0x9C`, inflate with `ZLibStream`;
  otherwise it's raw. (Guard on the magic — old/small BLIBs store peaks uncompressed.)
- Normalize intensities to base peak = 1.0 if you need relative intensities.
- RT / peak boundaries live in the `RetentionTimes` table (per-peptide `MIN(startTime)/MAX(endTime)` for the
  union firing window); `PRAGMA table_info(RetentionTimes)` first — older BLIBs lack start/end columns.
  `ScoreTypes.probabilityType` tells you filter direction (q-value lower-is-better vs. score higher-is-better).

### Discovering the active library
`GetSettingsListSelectedItems("Libraries")` → `GetSettingsListItem("Libraries", name)` → parse the library
XML; a BLIB is any root element whose local name starts with `bibliospec`; resolve its path against the
document directory. (Simpler: scan `*.blib` next to the `.sky`, document-named first.)

### ⚠️ Writing a `.blib` (BiblioSpec v8) + registering it
Create schema **majorVersion 1, minorVersion 8** (the version that added `RefSpectra.startTime/endTime`).
Tables: `LibInfo, IonMobilityTypes, ScoreTypes, SpectrumSourceFiles, Proteins, RefSpectraProteins, RefSpectra,
RefSpectraPeaks, Modifications, RetentionTimes`. Write everything in one transaction. Peaks = top-N by
intensity, re-sorted by m/z ascending, m/z as LE doubles + intensity as LE floats, each zlib(RFC-1950)
compressed.

- **UniMod IDs break BiblioSpec's matcher.** `LibKeyModificationMatcher` int-parses bracket contents and
  throws `FormatException: The number 'UniMod:4' is not in the correct format`. Translate `C[UniMod:4]` →
  `C[+57.021464]` (keep a UniMod→mass-delta table) before writing.
- **Write next to a *saved* `.sky`.** Writing to `%TEMP%` reports success but the registration silently drops
  on the next document save.
- **`AddSettingsListItem` can't register a Spectral Library** (the list's `T` is the abstract `LibrarySpec`
  with no static `Deserialize`). Use the CLI path, **path before name**:
  ```csharp
  client.RunCommand(new[] { "--add-library-path=" + blibPath, "--add-library-name=" + assayName });
  ```

### ⚠️ Note on modification formats generally
Skyline exports and BLIB use *different* modification notations: Skyline uses `(unimod:4)` / `[+57.02146]`
depending on the column, BLIB uses `[+57.02146]`. And **I and L are indistinguishable by MS** — decide
explicitly whether to collapse `I↔L` when matching peptides to a library (PRISM keeps them distinct for
library-assist because a detected peptide has its own exact predicted spectrum; FASTA parsimony collapses
them). Getting this wrong silently changes match rates.

---

## 6. Importing results, FASTA, libraries, building a blib (SkylineCmd surface)

These run via `RunCommand([...])` against the live document (or via `SkylineCmd`/`SkylineRunner` headless).
Verified against Skyline 26.1.1 (`--help=sections` lists the categories; `--help` a section for details).

**Import results (chromatograms):**
`--import-file=<raw>` + `--import-replicate-name=<name>`; `--import-all=<folder>` /
`--import-all-files=<folder>` with `--import-naming-pattern` / `--import-filename-pattern` /
`--import-samplename-pattern`; `--import-append`, `--import-before/-on-or-after=<date>`,
`--import-peak-boundaries=<file>`, `--import-warn-on-failure`. Performance: `--import-threads=<n>`
(parallel files, like the UI's "files to import simultaneously"), `--import-process-count=<n>` (sub-processes,
up to ~10× on NUMA servers), `--import-no-join` (write per-file `.skyd` without joining — for HPC).

**Build a document `.blib` from a peptide search** (the real "write blib" path when you have search results):
`--import-search-file=<search results>` (repeatable) builds a document-specific spectral library;
`--import-search-cutoff-score=<0..1>` (default 0.95), `--import-search-add-mods`,
`--import-search-irts=<name>`, `--import-search-include-ambiguous`, `--import-search-prefer-embedded-spectra`.
Add `--import-fasta` to also add the matched peptides as targets.

**Add an existing library:** `--add-library-path=<file>` + `--add-library-name=<name>`.

**Import FASTA / targets:** `--import-fasta=<file>` (`--keep-empty-proteins`);
`--import-transition-list=<csv>` (slow, ~10 rows/s — show a heartbeat); annotations via `--annotation-*`
(§4) or `ImportProperties` CSV.

**Document I/O (headless):** `--in`/`--open`, `--save`/`--out`/`--save-as`, `--new[=path]` (`--overwrite`,
`--discard-changes`), `--share-zip[=…]` (`--share-type=minimal|complete`), `--batch-commands=<file>` (run a
command file against one open document), `--log-file`, `--timestamp`, `--memstamp`.

**Reports (headless):** `--report-name` + `--report-file` + `--report-format=csv|tsv` + `--report-invariant`;
`--report-add=<file>.skyr` + `--report-conflict-resolution=overwrite|skip`.

---

## 7. Packaging & installing the tool

### `tool-inf/` manifest (RPC/exe tool)
- `info.properties`: `Name`, `Version` (keep in sync with the assembly version — verify at release),
  `Identifier = URN:LSID:<org>:<Tool>`, `Author`, `Description`.
- `<ToolName>.properties`: `Command=<Tool>.exe`, **`Arguments=$(SkylineConnection)`**, `Title`,
  `InitialDirectory=`, `OutputToImmediateWindow=False`. The properties filename **must match** the
  `Command`/`AssemblyName`.
- Copy both into the build output (`<None Include="tool-inf\..." CopyToOutputDirectory="PreserveNewest" />`).

### Zip
`dotnet msbuild build/package.proj` → `dotnet publish` the WPF app **framework-dependent**
(`--self-contained false /p:UseAppHost=true`, so the user installs the .NET 8 **Desktop** Runtime separately →
small zip) → `ZipDirectory` to `publish/<Tool>.zip`. Stage `tool-inf/` and the `Reports/*.skyr` into the zip.

### Ship gate (learned the hard way)
Automate **test → package → launch-verify** (`package-and-verify.ps1`). The verify step extracts the zip to a
clean dir (fresh-install simulation) and *actually launches* the exe with a dummy connection arg. Because the
WPF window (hence ScottPlot/SkiaSharp/XAML) loads at startup, a broken native dependency shows up as a load
error — grep the tool's log for `Could not load file or assembly | XamlParseException |
TypeInitializationException | DllNotFoundException | BadImageFormatException` and require a
"tool started" line. A failed *connection* from the dummy arg is expected; a failed *load* is not.

### Install
- **UI:** Tools ▸ Tool Store ▸ *Install from file* (pick the zip).
- **Headless:** `SkylineCmd --tool-add-zip=<zip> --tool-zip-conflict-resolution=overwrite|parallel`
  (`--tool-zip-overwrite-annotations`, `--tool-ignore-required-packages`). Add-by-settings without a zip:
  `--tool-add`, `--tool-command`, `--tool-arguments`, `--tool-report`.
- ⚠️ **Reinstalling over a *running* tool** can leave a **partial extraction** (locked files) that drops
  `deps.json` / DLLs → the tool then fails to load (we hit exactly this as a SkiaSharp
  `FileNotFoundException`). **Close the tool before reinstalling.**

### runtimeconfig / natives
The published `deps.json` must list the native assets that have to survive extraction
(`runtimes/win-x64/native/duckdb.dll`, SkiaSharp/HarfBuzz, `e_sqlite3`). On Linux CI, SkiaSharp needs
`libfontconfig1` + `libfreetype6`.

---

## 8. Project setup / NuGet / architecture

### Layout (cross-platform CLI + Windows tool)
| Project | TFM | Role |
|---|---|---|
| `Core` | `net8.0` | algorithms / IO / QC — **cross-platform, no pipes, no WPF** |
| `Cli` | `net8.0` | the cross-platform command-line entry point (references Core only) |
| `Skyline` | `net8.0-windows` | JSON-RPC client + report/grid/library driver |
| `App` | `net8.0-windows`, WPF | the `.exe` Skyline launches |

### ⚠️ Keep the RPC out of a cross-platform Core
The SkylineTool sources use `System.IO.Pipes` message-mode (Windows-only at runtime). **Link-compile the four
RPC files into the `net8.0-windows` project ONLY** — that's what keeps `Core` (and the CLI) cross-platform.
Set `Nullable disable` on that project (the vendored sources predate nullable refs).
*Cadenza folds the RPC into its Core, making Core Windows-only — fine if you have no cross-platform CLI, but
split it (PRISM-style) if you do.*

### `Directory.Build.props`
- `EnableWindowsTargeting=true` — so the `net8.0-windows`/WPF projects **build on Linux/macOS CI** (their
  tests are gated to the Windows runner).
- Single-source `<Version>`; verify it matches `tool-inf/info.properties` at release.
- ⚠️ **Do NOT set `InvariantGlobalization=true`** — WPF text/font rendering throws `CultureNotFoundException`
  under invariant globalization. Get numeric reproducibility by passing `CultureInfo.InvariantCulture`
  explicitly in every data-output path instead.

### Packages
`Microsoft.Data.Sqlite` (read/write `.blib`), `Parquet.Net` (typed columnar report I/O — pin a version that
decodes Skyline/DuckDB dictionary pages), `DuckDB.NET.Data.Full` (merge/sort large reports; ships the native
engine — pin the version, sort tie-breaks can shift across engine versions), `ScottPlot` (+ `ScottPlot.WPF`
for interactive; SkiaSharp under the hood), `YamlDotNet`, `MathNet.Numerics`.

---

## 9. Checklist for the next agent

1. **Connection:** transform `args[0]` with `GetJsonPipeName`; connect **per call**; `ReadMode=Message`. If you
   see `0x00 is invalid start of value`, it's one of these three.
2. **Reports:** invariant culture; export-to-file (validate `PAR1`) beats paginating; ship `.skyr` and install
   with `--report-add … --report-conflict-resolution=overwrite`.
3. **Replicates/grid:** it's **not** a report — `GetReportDocTopic("Replicate","document_grid")` +
   `GetReportFromDefinitionRows(DataSource="document_grid")`; `count=0` → shape only; anything not a built-in
   column is a user annotation (`annotation_<Name>`), keep it.
4. **Settings / import / build-blib:** almost everything is a `SkylineCmd` flag via `RunCommand`
   (§6) — one flag per batch except mutually-validated fields; `RunCommand` aborts the batch on the first bad
   arg.
5. **BLIB:** `Pooling=False`; guard the zlib `0x78 0x9C` magic; write BiblioSpec v8 in one transaction; convert
   UniMod IDs to mass deltas; register with `--add-library-path` (path before name), next to a **saved** `.sky`.
6. **Packaging:** `Arguments=$(SkylineConnection)`; framework-dependent publish; a launch-verify ship gate;
   close the tool before reinstalling.
7. **Architecture:** RPC only in the `net8.0-windows` project; `EnableWindowsTargeting`; no
   `InvariantGlobalization`.

## 10. Source map (where each capability lives)
- RPC client/models: `dotnet/external/SkylineTool/{SkylineJsonToolClient,IJsonToolService,JsonToolConstants,JsonToolModels}.cs`
- Connection + seam: `SkylinePrism.Skyline/{SkylineSession,ISkylineClient}.cs`
- Reports / grid / annotations driver: `SkylinePrism.Skyline/SkylineReportDriver.cs` + `Reports/*.skyr`
- BLIB read: `SkylinePrism.Core/Library/SpectralLibrary.cs`
- BLIB **write** + RT read + discovery + settings write-back: **cadenza** —
  `SkylineCadenza.Core/{Output/BlibAssayWriter.cs, Ingest/BlibRetentionTimeReader.cs, Ingest/SkylineBlibDiscovery.cs,
  SkylineRpc/SkylineSettingsConfigurator.cs}`, `App/ViewModels/MainViewModel.cs`, `docs/skyline-integration.md`
- Packaging: `dotnet/build/{package.proj,package-and-verify.ps1,verify-tool.ps1}`, `App/tool-inf/*`,
  `Directory.Build.props`
- Classic report-macro tool (contrast): `skyline-prism/skyline-external-tool/tool-inf/{PRISM,info}.properties`
