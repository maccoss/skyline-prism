# CLAUDE.md - AI Agent Guidelines for Skyline-PRISM

This document provides context and guidelines for AI agents working on the Skyline-PRISM project.
It is the single source of truth for agent guidance (it supersedes the former `AGENTS.md`).

> **Building or extending the Skyline external tool?** The canonical field guide (RPC, reports, `.blib`,
> chromatograms, packaging), a scaffolding skill, and a `dotnet new` template now live in
> **[uw-maccosslab/skyline-external-tools-ai](https://github.com/uw-maccosslab/skyline-external-tools-ai)**
> (`docs/skyline-external-tools.md`). PRISM is one of the example tools it draws from.
>
> **Read it from the local clone before touching anything under `dotnet/src/SkylinePrism.{Skyline,App}/`:**
> `D:\GitHub-Repo\uw-maccosslab\skyline-external-tools-ai`
> (on Linux/macOS, a sibling clone of the same repo). Start with `CRITICAL-RULES.md` (the hard-won
> gotchas: transform `args[0]`, connect-per-call, `PipeTransmissionMode.Message`, invariant culture,
> `.blib` `Pooling=False`, the launch-verify ship gate), then `TOC.md` -> `docs/skyline-external-tools.md`
> for the section you need. `git pull` it if it looks stale; **update the guide there, not here** -
> `docs/skyline-external-tools.md` in this repo is only a pointer stub.

## Project Overview

**Skyline-PRISM** (Proteomics Reference-Integrated Signal Modeling) normalizes LC-MS proteomics data exported from [Skyline](https://skyline.ms), with robust protein quantification using Tukey median polish and reference-anchored batch correction. It ships as a cross-platform `prism` CLI and a Windows Skyline external tool, both built from the .NET 10 code under `dotnet/`.

> [!NOTE]
> PRISM began as a Python package (`skyline-prism` on PyPI). That engine was **retired and removed** after its last release, `v26.4.4`; the C# code reproduced its numbers to 1e-9 on the deterministic stages. Older Python releases remain available from the `v*` tags and PyPI but are unmaintained. Do not add Python engine code back, and do not describe PRISM as a Python package.

### Key Concepts

- **Transition-level input required**: PRISM expects transition-level data from Skyline (not peptide or protein summaries)
- **Tukey median polish as default**: Both transition→peptide and peptide→protein rollups use median polish by default for robust outlier handling
- **Reference-anchored ComBat batch correction**: Uses inter-experiment reference samples for QC evaluation, with automatic fallback if correction degrades quality
- **Dual-control validation**: Uses intra-experiment QC samples to validate corrections without overfitting
- **Sample outlier detection**: Automatic detection of samples with abnormally low signal (one-sided, on LINEAR scale). Can report or exclude outliers.
- **Two-arm pipeline**: the arms split after peptide normalization and BEFORE any batch correction, so each output is ComBat-corrected exactly once, at its own reporting level (peptide or protein)
- **Optional RT correction**: RT-dependent correction is implemented but DISABLED by default (search engine RT calibration may not generalize between samples)

### Scale Conventions

> [!CAUTION]
> **PARQUET FILES MUST ALWAYS CONTAIN LINEAR VALUES (NOT LOG2)**
>
> This is a critical requirement that has been incorrectly regressed multiple times.
> Always verify that output parquet files contain linear abundance values, not log2-transformed values.

| Stage | Scale | Notes |
|-------|-------|-------|

| **Input** | LINEAR | Raw peak areas from Skyline |
| **Internal** | LOG2 | All rollup/normalization operates on log2 scale |
| **Output** | LINEAR | Final peptide/protein output matrices (parquet/CSV) are always written in LINEAR scale (values are 2^x, not log2(x)) |

**Implementation Location:**
The log2→linear conversion happens where each arm's corrected matrix is written -
`NormalizeCorrectStage` / `StreamingNormalizeCorrect` (`CorrectedLinearPath`), for the peptide arm at
Stage 2b/2c and the protein arm at Stage 4b/4c.

**Important:** Intermediate files (`peptides_rollup.parquet`, `proteins_raw.parquet`) remain in LOG2 scale for normalization and batch correction. Only the **final** output files (`corrected_peptides.parquet`, `corrected_proteins.parquet`) are converted to linear.

**Display conventions:**
- **Box plots**: Display LINEAR values (from parquet)
- **PCA plots**: Use LOG2 internally (convert from linear parquet for variance stabilization)
- **CV calculations**: Always on LINEAR values

**Do not write log2 values to final output files.**

The pipeline automatically handles transforms:
- Input linear values are log2-transformed for processing
- Output values are back-transformed to linear (2^x) before writing

When calling `SkylinePrism.Core` types directly, check the XML doc comments: every matrix-taking
method states whether it expects LOG2 or LINEAR.

### Data Density (CRITICAL - PRISM input is normally COMPLETE)

> [!IMPORTANT]
> **Skyline exports have no missing values.** Skyline integrates *imputed peak boundaries* for every
> replicate, so every transition has an area in every run - even where there is no real signal. The
> peptide x sample matrix that reaches Stage 2b/2c is therefore **dense**, and NaN is the rare
> exception, not the norm.

Do not design or justify an algorithm on the assumption that missing values are common. When
choosing between two behaviors, **the dense case is the one that matters**; the sparse case must be
*supported*, not optimized for. (This was got backwards once: a ComBat change was justified as
"matches the reference implementation on data with missing values - which is all real proteomics
data". The opposite is true.)

Supporting evidence in the repo: the `mini` golden fixtures contain 0 nulls and 0 NaNs; `#N/A`
tokens in a real export are imputed at the **transition** level (`tests/.../MissingValueTests.cs`);
`TransitionRollup` emits NaN for a peptide x sample cell only when that peptide had *no* transition
with data at all in that run.

Missing values do still occur, and must keep working:

- **Merging documents with different target lists.** PRISM takes a LIST of Skyline documents (one
  per plate/batch). A peptide present in one document but absent from another is NaN for every
  replicate of the second - which is also the "feature absent from a whole batch" case that
  `sva::ComBat` cannot express at all (it dies with a singular design).
- Peptides filtered out of one run but not another, and genuinely unintegrated data.

### CV Calculation (CRITICAL)

**CVs must ALWAYS be calculated on LINEAR scale data, NEVER on log-transformed data.**

Correct calculation (`CvMetrics`, which every CV in the QC report goes through):
```csharp
var linear = Math.Pow(2, log2Value);   // back-transform first
var cv = stdDev(linear) / mean(linear) * 100.0;   // CV as a percentage
```

Rationale: On log scale, variance is artificially compressed. A CV of 5% on log2 data would be meaningless - true biological CVs for proteomics control samples typically range from 10-30%.

### Peptide Modification Format (CRITICAL)

> [!CAUTION]
> **Skyline exports and BLIB spectral libraries use DIFFERENT modification formats.**
>
> This has caused repeated issues. Always convert between formats when matching peptides to library spectra.

| Source | Format | Example |
|--------|--------|---------|
| **Skyline export** | Unimod IDs in parentheses | `C(unimod:4)`, `M(unimod:35)` |
| **BLIB library** | Mass deltas in brackets | `C[+57.02146]`, `M[+15.99491]` |

**Common modifications:**

| Modification | Unimod Format | BLIB Format |
|--------------|---------------|-------------|
| Carbamidomethyl (Cys) | `C(unimod:4)` | `C[+57.02146]` |
| Oxidation (Met) | `M(unimod:35)` | `M[+15.99491]` |
| Phosphorylation (Ser) | `S(unimod:21)` | `S[+79.96633]` |
| Phosphorylation (Thr) | `T(unimod:21)` | `T[+79.96633]` |
| Phosphorylation (Tyr) | `Y(unimod:21)` | `Y[+79.96633]` |

**How PRISM matches** (`SpectralLibrary.GetSpectrum`): rather than translating one notation into
the other - which needs a table entry per modification and silently fails on the next one - it looks up
the **exact** `modifiedSequence_charge` key first, then falls back to a key with **all** modification
notation stripped (`StripModifications` removes `(unimod:N)`, `[+57.02146]`, `[Carbamidomethyl]` and
`(...)` alike). So any notation matches any other, with no per-modification table to maintain.

> [!CAUTION]
> **I and L are deliberately NOT collapsed.** `NormalizeForMatching` exists but the library lookup does
> not use it: I/L give different predicted spectra and RTs, and every detected peptide has its own exact
> predicted spectrum in the library (it was detected against it), so collapsing them would force a match
> to the wrong spectrum. Do not "fix" the lookup by adding I/L normalization.

### Processing Pipeline

The current implementation follows this stage structure:

```text
Stage 1: Merge CSVs (streaming, memory-efficient)
    ↓
Stage 2: Transition → Peptide rollup (Tukey median polish)
    ↓
Stage 2b: Peptide Global Normalization (median or VSN)
    ↓ [Optional: RT correction - disabled by default]
    ├──────────────────────────────┐   <- the arms split HERE, before any ComBat
    ↓                              ↓
Stage 3: Protein Parsimony    Stage 2c: Peptide ComBat Batch Correction
    ↓                              ↓
    ↓                         PEPTIDE OUTPUT
    ↓                         (corrected_peptides.parquet)
Stage 4: Peptide → Protein Rollup (median polish)
    ↓
Stage 4b: Protein Global Normalization (median)
    ↓
Stage 4c: Protein ComBat Batch Correction
    ↓
Stage 5: Output Generation
    ↓
    PROTEIN OUTPUT
    (corrected_proteins.parquet)
    ↓
Stage 5b: QC Report Generation (HTML + plots)
```

**Key implementation details:**

- **Streaming processing**: Stage 1 uses DuckDB-based streaming to handle ~47GB datasets
- **Batch correction applied ONCE PER ARM, never twice to the same number**: the peptide output is
  corrected at Stage 2c, the protein output at Stage 4c, and the protein arm branches from the
  **normalized, pre-ComBat** peptide matrix so it does not inherit the peptide correction. Before
  dotnet-v26.15.0 the protein arm consumed the ComBat-corrected peptides and was then corrected
  again, which measurably hurt: on a 4-batch AD cohort the second correction moved held-out QC CV
  the wrong way (12.7% -> 13.0%) and tripped the control-asymmetry heuristic (reference improved
  3x more than QC; that heuristic was called "overfitting" at the time and is a NOTE now). Correcting once instead gives 16.3% -> 12.4% and no warning.
- **Independent outputs**: the two arms are genuinely independent - `corrected_peptides` is
  bit-identical whether or not protein-level ComBat runs, and vice versa.
- **Log files**: Automatically saved to output directory with timestamp (`prism_run_YYYYMMDD_HHMMSS.log`)
- **Metadata columns**: Uses `sample`, `sample_type`, `batch` (with automatic normalization from Skyline formats)

## Project Structure

```
skyline-prism/
├── dotnet/
│   ├── src/
│   │   ├── SkylinePrism.Core/       # Platform-neutral engine (net10.0)
│   │   │   ├── Pipeline/            # PrismPipeline (stage orchestration), NormalizeCorrectStage,
│   │   │   │                        #   StreamingNormalizeCorrect, Provenance (parameters.json)
│   │   │   ├── Rollup/              # TukeyMedianPolish, TransitionRollup, ProteinRollup,
│   │   │   │                        #   Sum/TopN/Consensus rollups
│   │   │   ├── BatchCorrection/     # ComBat, ComBatCore, ReferenceAnchoredComBat, StreamingComBat
│   │   │   ├── Normalization/       # Normalizer (median/rt_lowess/quantile/vsn), OutlierDetector
│   │   │   ├── Parsimony/           # ParsimonyEngine, FastaParser, ProteinGroup
│   │   │   ├── Library/             # SpectralLibrary (.blib), LibraryRollup
│   │   │   ├── Qc/                  # QcReport, CvMetrics, ValidationStatus, DynamicRange,
│   │   │   │                        #   PrecursorDensity, IsolationScheme
│   │   │   ├── IO/                  # DuckDbMerge, parquet readers/writers, SkylineColumns
│   │   │   ├── Numerics/            # Stats, Lowess, Pca, LinAlg, Kde
│   │   │   ├── Config/              # PrismConfig, ConfigTemplate, ConfigWriter
│   │   │   └── Visualization/       # PlotRenderer (ScottPlot/SkiaSharp)
│   │   ├── SkylinePrism.Cli/        # `prism` CLI (net10.0, cross-platform)
│   │   ├── SkylinePrism.Skyline/    # Skyline JSON-RPC + headless export (Windows)
│   │   └── SkylinePrism.App/        # WPF Skyline external tool / standalone GUI (Windows)
│   ├── tests/
│   │   ├── SkylinePrism.Tests/          # Core + CLI (xUnit, cross-platform)
│   │   ├── SkylinePrism.Tests.Windows/  # Skyline + App
│   │   └── fixtures/                    # Committed cross-engine goldens + bit-exact digests
│   ├── build/                       # Skyline-tool packaging (package-and-verify.ps1)
│   └── Directory.Build.props        # <Version> - one of the two release version sources
├── docs/                            # Parameter reference, methods, output files, parsimony
├── release-notes/                   # One file per release + the rolling dotnet-next draft
├── SPECIFICATION.md                 # Algorithm/format specification
└── CLAUDE.md                        # This file
```

## Key Algorithms

### Tukey Median Polish (Default for Rollups)

Used for both transition→peptide and peptide→protein rollups. Decomposes a matrix into:
- Row effects (see table below)
- Column effects (sample abundance - **this is the output**)
- Residuals (noise/outliers - **preserved for biological analysis**)

| Rollup Stage | Row Effects Represent | Column Effects |
|--------------|----------------------|----------------|
| Transition → Peptide | Transition interference (co-eluting analytes) | Peptide abundance |
| Peptide → Protein | Peptide ionization efficiency | Protein abundance |

The median operation automatically downweights outliers without explicit filtering.

**Important**: Following Plubell et al. 2022 (doi:10.1021/acs.jproteome.1c00894), residuals are **preserved, not discarded**. Peptides/transitions with large residuals may indicate biologically interesting proteoform variation, PTMs, or protein processing.

**Implementation**:
- `Core/Rollup/TukeyMedianPolish.cs` → returns `MedianPolishResult` (row/column effects + residuals)
- `Core/Rollup/TransitionRollup.cs` → transition → peptide, writes `peptides_rollup_residuals.parquet`
- `Core/Rollup/ProteinRollup.cs` → peptide → protein, writes `proteins_raw_residuals.parquet`
- `Core/Pipeline/ResidualScaler.cs` → the batch-corrected copies of both residual files

### RT-LOWESS Normalization (the default)

Removes RT-dependent systematic variation (ion suppression, gradient drift) on top of the overall
loading difference. Per sample: take the log-ratio of each peptide to the cohort reference profile, fit
a LOWESS curve of those ratios against `mean_rt` on a grid, interpolate, and subtract.

**Implementation**: `Core/Normalization/Normalizer.cs` → `RtLowessNormalize()`, with the smoother in
`Core/Numerics/Lowess.cs`. Tuned by `global_normalization.rt_lowess.{frac,n_grid_points}`; set
`global_normalization.method: median` for a plain per-sample median shift.

### ComBat Batch Correction

Full empirical Bayes implementation (Johnson et al. 2007):
- Estimates additive (location) and multiplicative (scale) batch effects
- Uses empirical Bayes shrinkage for robust estimation
- Supports reference batch, parametric/non-parametric priors, mean-only correction

**Implementation**: `Core/BatchCorrection/ComBat.cs` (+ `ComBatCore` for the estimator,
`ReferenceAnchoredComBat` for the opt-in anchored variant, `StreamingComBat` for the bounded-memory
path). `Core/Qc/BatchCorrectionEvaluator.cs` decides the opt-in `auto_revert`.

## Development Guidelines

### Style Guidelines

- **No emojis**: Do not use emojis in code, documentation, comments, or output messages. Use plain text instead (e.g., "PASSED" instead of "✓", "WARNING" instead of "⚠️").
- **This is a strict requirement**: All status indicators, section headers, and documentation must use plain ASCII text. Use prefixes like "[WORKING]", "[ISSUE]", "[TODO]" instead of emoji symbols.
- Unicode arrows (→) for flow diagrams are acceptable.

### Building

Requires the [.NET 10 SDK](https://dotnet.microsoft.com/download).

> [!NOTE]
> **Why .NET 10, and what a bump costs.** .NET 10 is the current LTS (support to Nov 2028); .NET 8's LTS
> ends 10 November 2026, which is what prompted the move.
>
> The version lives in **three kinds of place**, and the third is the one that gets missed:
> 1. `global.json` at the repo root - the single SDK pin. It governs the whole tree, and both workflows
>    read it via `global-json-file:` rather than repeating a version, so CI cannot drift from it.
> 2. Every `<TargetFramework>`.
> 3. `dotnet/src/SkylinePrism.App/tool-inf/info.properties` - the Skyline tool manifest, which states the
>    required runtime **inside the shipped zip**. It is the closest thing to documentation at the point
>    of failure, and the .NET 10 migration shipped it still saying 8 until review caught it.
>
> Prose in `README.md`, `dotnet/README.md`, the workflow headers and the `<!-- Plain net8.0 ... -->`
> comments above the target lines all need it too - a mechanical rename leaves comments contradicting the
> line beneath them.
>
> The artifacts are **framework-dependent by design**, so a major bump means every user installs a new
> runtime before the tool will start. That is a coordinated upgrade for a lab, not a silent one; do not
> do it for a minor gain.
>
> **Verify DuckDB, not just the test suite.** A runtime change moves GC and marshalling behavior, which
> is precisely where DuckDB.NET has failed here before (see the caution below: `AccessViolation` and bare
> segfaults, reproduced across two DuckDB versions and four configurations). A green `dotnet test` says
> nothing about it - run a real cohort through Stage 1 and Stage 2. The .NET 10 move was checked that
> way on a 47M-row, 93-sample cohort.

```bash
cd dotnet
dotnet build                                  # whole solution (Windows: includes the WPF tool)
dotnet build SkylinePrism.CrossPlatform.slnf  # Core + CLI only (Linux/macOS)
```

### Running Tests

**Always run the tests after making changes:**

```bash
cd dotnet
dotnet test                                                  # everything
dotnet test tests/SkylinePrism.Tests/SkylinePrism.Tests.csproj   # Core + CLI only
dotnet test --filter "FullyQualifiedName~QcReport"           # one area
dotnet test --filter "FullyQualifiedName~QcReportTests.MedianCv_MatchesHandComputed"  # one test
```

**Test expectations:**
- All tests must pass before committing
- New features need tests; a bug fix starts with a failing test that reproduces it
- `dotnet/tests/fixtures/` holds committed goldens: cross-engine parity references and the
  **bit-exact quantity digests**. A digest failure is the gate working - find out which quantity moved
  and why before even considering regeneration (see `dotnet/tests/fixtures/README.md`)

### Code Style

- Follow the surrounding code: same naming, same comment density, same idiom
- `dotnet build` must be **warning-free**; do not silence a warning you can fix
- XML doc comments on public types/members, especially the scale a matrix argument expects

### Documentation Updates

**Keep README.md updated:**
- When adding new features, update the README.md to document them
- When changing CLI commands, update the usage examples
- When adding new configuration options, update every config surface (see below)

**CRITICAL: Configuration is a contract - keep the template, schema, and docs in sync**

A config-driven feature has multiple surfaces that MUST stay consistent. Whenever you **add, rename, or
remove** a configuration key, update ALL of the relevant surfaces in the SAME change:

1. The config class property in `dotnet/src/SkylinePrism.Core/Config/PrismConfig.cs`
2. `ConfigTemplate.Default()` (and `.Minimal()` if common) in `ConfigTemplate.cs` - the emitted YAML
3. The `KnownKeys` schema (`BuildSchema()`) in `PrismConfig.cs` - so `FindUnknownKeys` accepts it while
   still warning on typos/unknown keys
4. `ConfigWriter` - the minimal round-trippable YAML, which is also what the QC report's Processing
   Parameters table renders from (`QcReport.ParameterRows`), so a key added there reaches both
5. `StageDependencies.ByStage` - which pipeline stage(s) the key can change the OUTPUT of, or
   `OutputIrrelevant` with the reason. **This one is a correctness surface, not a documentation one:**
   the stage cache reuses a stage whose declared keys are unchanged, so a key missing from this table
   means a re-run silently keeps output computed with the OLD value.
   `StageDependencyCoverageTests` fails the build on an unclassified key, and separately asserts that
   mutating a key changes exactly the fingerprints of the stages that declare it - so this cannot be
   satisfied by listing the key in the wrong place.
6. `docs/parameters.md` - the parameter reference (key, default, description).

Confirmation is MANDATORY, not optional:
- **Adding a feature:** run `prism config-template` and confirm the new key appears in the generated YAML
  (active, or a commented example for method-specific keys), and add a row to `docs/parameters.md`. The
  C# test `ConfigValidationTests.ConfigTemplate_HasNoUnknownKeysAndValidates` fails if the emitted
  template contains a key the schema does not know - fix the schema/template, do not silence it.
- **Removing a feature:** delete the key from the template AND the schema AND `docs/parameters.md`. A
  removed key must never linger in the generated YAML.
- **A key PRISM does not implement** (including one left over from the retired Python engine) must
  never be silently ignored - make it warn (`FindUnknownKeys`) or abort (`PrismConfig.Validate`).

Users generate their configs via:
```bash
prism config-template -o config.yaml           # Full template
prism config-template --minimal -o config.yaml  # Minimal template
```

**SPECIFICATION.md** contains the detailed technical specification. Reference it for algorithm details but avoid modifying it unless the fundamental approach changes.

## Key Files to Understand

### SPECIFICATION.md
The authoritative technical specification. Contains:
- Input/output formats (Skyline report columns)
- Algorithm details (RT correction, median polish, parsimony)
- Processing pipeline stages (two-arm design)
- Configuration parameters

### Configuration Templates (IMPORTANT)

**The templates are generated from `ConfigTemplate.Default()` / `.Minimal()` in
`dotnet/src/SkylinePrism.Core/Config/ConfigTemplate.cs`** - there is no checked-in template file to
edit. Keep them in step with the `KnownKeys` schema in `PrismConfig.cs`; the test
`ConfigValidationTests.ConfigTemplate_HasNoUnknownKeysAndValidates` fails if the emitted template
contains a key the schema does not know. See "Configuration is a contract" above for the full
checklist.

Users generate templates via:
- `prism config-template -o config.yaml` (full template)
- `prism config-template --minimal -o config.yaml` (common options only)

Key sections:
- `transition_rollup`: Transition→peptide rollup (method: sum, median_polish, topn, consensus, library_assist)
- `global_normalization`: Peptide normalization (method: rt_lowess (default), median, quantile, vsn, none)
- `sample_outlier_detection`: Detect low-signal samples (method: iqr or fold_median, action: report or exclude)
- `batch_correction`: ComBat settings (enabled, peptide_level, protein_level, reference_anchored, auto_revert)
- `protein_rollup`: Peptide→protein rollup (method: median_polish, sum, topn, maxlfq, ibaq)
- `protein_normalization`: Protein-level normalization (method: median)
- `parsimony`: Shared peptide handling (all_groups, unique_only, razor) + enzyme / enzyme_specificity
- `qc_report`: QC report generation (enabled, save_plots)

### Core/BatchCorrection/
- `ComBat.Run()`: the entry point for a wide LOG2 matrix + batch labels
- `ComBatCore`: the empirical-Bayes estimator itself (shared by every path)
- `ReferenceAnchoredComBat`: the opt-in variant anchored on reference samples across batches
- `StreamingComBat`: the same result with bounded memory, a row group at a time
- `Qc/BatchCorrectionEvaluator`: the `auto_revert` decision (control CV worsened by >10%)

### Core/Qc/
- `QcReport.Generate()`: builds the self-contained `qc_report.html` from an output directory
- `Ms2SignalAccounting`: the MS2 signal accounting, cached as `ms2_signal_accounting.parquet`.
  **That cache is keyed on `SettingsKeyFor(measure, tolerance, isolation scheme, list names)`, and the
  key is stored in the file** - add a `qc_report.ms2_signal` setting that changes the numbers and it
  must go into that key too, or a re-run replots the previous run's numbers under the new run's
  caption. Nothing fails loudly if you forget: both plots look right, and the caption comes from the
  cache. The key records what was REQUESTED, never what was computed - asking for `ions` on an export
  with no ion column falls back to signal, and keying on the fallback would recompute forever.
- `CvMetrics`: every median CV in the report (always computed on the LINEAR scale)
- `ValidationStatus`: the dual-control pass/fail verdict, its warnings and its notes
- `DynamicRange`, `PrecursorDensity`, `IsolationScheme`: the GUI's analysis tabs
- `Visualization/PlotRenderer`: every plot (ScottPlot/SkiaSharp), rendered headlessly

### dotnet/Directory.Build.props
`<Version>` - drives every assembly version, `prism --version`, and provenance `pipeline_version`.
It must match `SkylinePrism.App/tool-inf/info.properties` `Version =` and the release tag.

## CLI Commands

The `prism` CLI is the cross-platform entry point. The primary command is `prism run`:

```bash
# Run the full PRISM pipeline (recommended)
prism run -i skyline_report.csv -o output_dir/ -c config.yaml -m metadata.csv
```

This produces:
- `corrected_peptides.parquet` - Peptide-level normalized/batch-corrected quantities
- `corrected_proteins.parquet` - Protein-level normalized/batch-corrected quantities
- `peptides_rollup.parquet` - Raw peptide abundances from transition rollup (before normalization)
- `proteins_raw.parquet` - Raw protein abundances from peptide rollup (before normalization)
- `protein_groups.csv` - Protein group definitions
- `peptides_rollup_residuals.parquet` / `proteins_raw_residuals.parquet` - median-polish residuals,
  plus `corrected_*_residuals.parquet` (batch-corrected copies), if `output.include_residuals`
- `sample_metadata.csv` - resolved sample, batch and sample type per replicate
- `parameters.json` - complete processing parameters for reproducibility (provenance)
- `qc_report.html` - HTML QC report with embedded diagnostic plots
- `qc_plots/` - Directory containing PNG plot files (if `save_plots: true`)

### Reproducibility with --from-provenance

The `parameters.json` output contains all processing parameters, enabling exact re-runs:

```bash
# Re-run with exact same parameters on new data
prism run -i new_data.csv -o output2/ --from-provenance output1/parameters.json

# Override specific settings while keeping others from provenance
prism run -i new_data.csv -o output2/ --from-provenance output1/parameters.json -c overrides.yaml
```

**Implementation**: `Core/Pipeline/Provenance.cs` -> `Write()` / `LoadConfig()`. The QC report reads
the same file for its "Analysis Information" block, so a report always names the version, date, host
and inputs of the run that produced the numbers.

Additional utility commands:

```bash
# Merge multiple Skyline reports into unified parquet
prism merge report1.csv report2.csv -o data.parquet -m metadata.tsv

# Regenerate QC report from existing output (without reprocessing)
prism qc -d output_dir/

# Compare control-sample CVs between two runs
prism compare -1 run1/ -2 run2/ -o comparison.html

# Generate annotated configuration template
prism config-template -o config.yaml

# Minimal config template (common options only)
prism config-template --minimal -o config.yaml
```

## Common Tasks

### Adding a New Feature

1. Read SPECIFICATION.md to understand the design
2. Implement it in the appropriate `SkylinePrism.Core` area
3. Add tests under `dotnet/tests/`
4. Run `dotnet test` to verify
5. Update README.md / `docs/` if user-facing, and add a release-note entry to
   `release-notes/RELEASE_NOTES_dotnet-next.md`
6. If configurable: update every config surface and confirm the generated YAML has the key - see
   "CRITICAL: Configuration is a contract" above (`PrismConfig.cs` + `ConfigTemplate.cs` +
   `ConfigWriter.cs` + `docs/parameters.md`). Run `prism config-template` and verify the key appears;
   remove it everywhere when removing a feature.
7. Commit with a descriptive message

### Fixing a Bug

1. Write a failing test that reproduces the bug
2. Fix the bug
3. Verify the test passes
4. Run full test suite
5. Commit with reference to the issue if applicable

## Important Notes

- **Skyline** is an external tool (https://skyline.ms) - we process its exports, we don't modify Skyline itself
- **Sample types**: `experimental`, `qc`, `reference` - these have specific meanings in the normalization workflow
- **Column naming**: input column names are auto-detected per document and matched ignoring case, spaces and underscores - see `Core/IO/SkylineColumns.cs` (the `data:` config section overrides it)
- **Log scale**: Most operations work on log2-transformed abundances
- **Median polish is default**: For both transition→peptide and peptide→protein rollups
- **Two-arm pipeline**: Batch correction happens at the reporting level (peptide or protein), not before rollup

## FASTA-Based Protein Parsimony

`Core/Parsimony/FastaParser.cs` provides FASTA parsing for protein parsimony:

**Key members:**
- `Parse()`: UniProt/NCBI format FASTA
- `StripModifications()`: remove modifications from peptide sequences before matching
- `BuildMap()`: peptide→protein mapping via **enzyme-aware** substring search
- `CleavageBoundaries()`: the enzyme terminus check behind it
- `Digest()`: in-silico digestion, used for the iBAQ denominator

**Configuration** (`parsimony:` in the YAML - see `docs/parameters.md`):
```yaml
parsimony:
  fasta_path: /path/to/search.fasta
  enzyme: trypsin             # trypsin | trypsin/p | lys-c | ...
  enzyme_specificity: full    # full | semi | none
  shared_peptide_handling: all_groups
```

**Enzyme-aware membership (important):** substring containment is *necessary but not sufficient* for a
peptide to originate from a protein. The map attaches a peptide to a protein only when it occurs there
with termini consistent with the digestion enzyme (`parsimony.enzyme` / `parsimony.enzyme_specificity`,
default `trypsin` / `full`, with initiator-methionine excision handled). This removes "phantom"
assignments to homologs that share the subsequence but not the flanking cleavage site — e.g.
`AKEGVVAAAEK` is a substring of beta-synuclein (SNCB) but is preceded there by `M`, not `K/R`, so
trypsin cannot liberate it; it is proteotypic to alpha-synuclein (SNCA). Set `enzyme_specificity: none`
to restore the legacy pure-substring behavior. The check only applies on the FASTA path; the Skyline
Protein Accession column is already enzyme-aware.

**Note:** iBAQ's in-silico digestion (below) and the parsimony terminus check share the same enzyme
rules but are separate code paths.

## iBAQ Support

iBAQ (Intensity-Based Absolute Quantification) is now integrated. It normalizes protein abundances
by the number of theoretical peptides, enabling cross-protein abundance comparison.

**Implementation:** `FastaParser.Digest()` counts the theoretical peptides per protein; the counts
become the iBAQ denominator in `ProteinRollup`.

**Configuration:**
```yaml
protein_rollup:
  method: "ibaq"
  ibaq:
    fasta_path: "/path/to/database.fasta"
    enzyme: "trypsin"
    missed_cleavages: 0
```

---

## Interacting with Skyline (C# external tool)

The C# Skyline external tool (`dotnet/src/SkylinePrism.App`) exports the reports the pipeline consumes
and reads document settings — from a **running** Skyline over JSON-RPC, or from a **closed** `.sky` via
`SkylineCmd`. The authoritative how-to is the
field guide in **[uw-maccosslab/skyline-external-tools-ai](https://github.com/uw-maccosslab/skyline-external-tools-ai)**
(`docs/skyline-external-tools.md`), cloned locally at
`D:\GitHub-Repo\uw-maccosslab\skyline-external-tools-ai` — read it before extending the
tool's Skyline integration. Key points (mirrored in the code under `dotnet/src/SkylinePrism.Skyline/`):

- **Transport:** JSON-RPC 2.0 over a **named pipe**. Skyline passes the pipe name as `args[0]`
  (`$(SkylineConnection)`); transform it with `JsonToolConstants.GetJsonPipeName`. **Connect per call**
  — open a fresh `NamedPipeClientStream` (message mode) for each request and close it; never hold the
  pipe open (idle reuse throws deserialization errors). `SkylineSession` implements this; the driver
  talks to it through the small `ISkylineClient` interface so tests can use a fake (no live Skyline).
- **Report export:** prefer exporting whole reports to **parquet** (typed, ~20× faster than paginating),
  falling back to invariant-culture CSV. `SkylineReportDriver.Export` does this and also reads the
  built-in "Replicates" document grid for metadata.
- **Reading document settings** (e.g. the digestion **enzyme**): use the settings-list RPCs —
  `GetSettingsListSelectedItems("Enzymes")` for the active enzyme name, then
  `GetSettingsListItem("Enzymes", name)` for its XML. An enzyme element looks like
  `<enzyme name="Trypsin" cut="KR" no_cut="P" sense="C" />` (`cut` = cleavage residues, `no_cut` =
  blocking residues, `sense` = `C` cleave-after / `N` cleave-before). `SkylineDigestion` parses that XML
  and maps it to a PRISM `parsimony.enzyme` name; `SkylineReportDriver.GetDigestionEnzyme()` wires it up.
  The same `GetSettingsListSelectedItems`/`GetSettingsListItem` pattern reads the active spectral library.
- **General rule (per Nick Shulman):** anything `SkylineCmd` can do is reachable via
  `RunCommand([...flags...])` against the live document — but SkylineCmd aborts the whole batch on the
  first bad flag, so send flags one per `RunCommand` (except fields Skyline mutually validates).

> [!CAUTION]
> **Annotation columns in a generated `.skyr` MUST be quoted:** `<column name="&quot;annotation_Plate&quot;" />`,
> never the bare `annotation_Plate`. Skyline parses `column/@name` as a databinding PropertyPath, whose
> bare-identifier syntax rejects `_` — and the `annotation_` prefix contains one. Unquoted, the export
> aborts with `Error parsing annotation_Plate at location 10: Invalid character _` and **no report is
> written**, so the annotation silently never reaches the metadata. Quoted, the exported column is headed
> with the plain annotation name (`Plate`). Both engines build the view through
> `ReplicatesReportBuilder`, which applies the quoting — go through it rather than hand-rolling XML.

> [!NOTE]
> **Two transition report definitions ship, and they must stay in lockstep.** `Reports/Skyline-PRISM.skyr`
> (view `PRISM`) is the standard export; `Reports/Skyline-PRISM-Ions.skyr` (view `PRISM-Ions`) is the same
> columns in the same order plus exactly one more, `Results!*.Value.TransitionIonMetrics.LcPeakTransitionIonCount`,
> which `qc_report.ms2_signal.measure: ions` reads. It is a separate report because Skyline is slow to compute
> that column (measured at 29x slower per row on a 46M-row document - about 4 hours instead of 9.5 minutes -
> and one column costs half of what five do, so the cost is per-transition chromatogram access) and because a report is
> installed into the user's Skyline settings by view name - two names mean the fast report is never silently
> replaced by the slow one. Both exporters choose through `PrismReport.NameFor/FileFor(includeIonCounts)`;
> the headless export stamp records which report produced a cached export. `PrismReportDefinitionTests`
> fails the build if the two files drift, so add a column to BOTH or to neither.

### Inputs: multiple documents, open or closed (`PrismInput`)

The tool takes a LIST of inputs — one per batch/plate — merged into a single cohort. `PrismInput.Prepare`
resolves each to a report + metadata file:

- **Running Skyline** (`SkylineSession`): the launching instance, or any other found by
  `SkylineSession.DiscoverRunning` (which reads `~/.skyline-mcp/connection-*.json` — Skyline only appears
  there while its JSON-RPC server runs, so treat that list as a convenience, not an inventory).
- **Closed `.sky`** (`HeadlessSkylineExporter` over `ISkylineCommandRunner`): the document is loaded once
  per report (`--report-name` takes one), always with `--in` and **never** `--save`. Two runners:
  - `SkylineAppRunner` (**preferred**) drives the installed Skyline headlessly — the SkylineRunner
    protocol, reimplemented so it finds Skyline *or* Skyline-daily. It runs the real `Skyline.exe`, so
    `Skyline.exe.config` applies and **parquet export works**. ⚠️ No exit code: failure is only visible as
    an `Error:` prefix in the piped output (`SkylineAppRunner.IsErrorLine`).
  - `SkylineCmdRunner` (fallback) uses `SkylineCmd.exe` via `SkylineCmdLocator` (ClickOnce **application**
    folder — the one with `Skyline*.exe` beside it; the sibling `…exe_…` copies fail with "Unable to find
    Skyline.exe" — newest first, overridable with `PRISM_SKYLINECMD`). It reports
    `SupportsParquet = false`: `SkylineCmd.exe.config` lacks the Parquet.Net assembly bindings, so a
    `.parquet` request dies with "Could not load file or assembly 'Parquet'".

  Format follows the output **extension** (`--report-format` only accepts `csv|tsv`), and the result is
  verified with `ParquetMagic` rather than trusted.
- **Pre-exported report**: used in place; no Skyline at all. This is what makes the window usable as a
  **standalone PRISM GUI** — `MainWindow` must never require a non-null `_session` to run.
- **Closed-document metadata** comes from `SkyDocumentInfo`, which stream-parses the `.sky` header
  (stopping at `</settings_summary>`, so a 2 GB document reads in ~1 s) for the replicate-targeted
  `<annotation targets="…replicate…">` names, the `<enzyme>` element, and the replicate list. Read-only.

**Batch labels are file stems.** Each input's label is the exported report's file name, because
`DuckDbMerge` derives `Batch` / `Source Document` from the stem and builds sample IDs as
`<replicate>__@__<batch>`. Labels must therefore be unique and file-name safe
(`PrismInput.EnsureUniqueLabels`). Metadata files are passed to the pipeline **positionally**, 1:1 with
the inputs, so `ReplicateMetadata` can key rows by source document — see the caution below.

> [!CAUTION]
> **Replicate names collide across documents.** Reference and QC injections are normally named
> identically in every plate's document. `ReplicateMetadata` therefore stores each file's rows under the
> document-qualified key `<replicate>__@__<document>` in addition to the bare name, and
> `TypeFor`/`BatchFor`/`HasBatchFor` prefer the qualified entry. Keyed by bare name alone the last file
> silently wins — collapsing two batches into one label (so ComBat is skipped without a word) and giving
> every plate the last document's sample types. Do not "simplify" these lookups back to the bare map.

**Why PRISM reads the enzyme from the document:** PRISM's FASTA-based parsimony is enzyme-aware (see
"FASTA-Based Protein Parsimony"). The **CLI** takes the enzyme from `parsimony.enzyme` (default
`trypsin`), but the **external tool** overrides it from the document's digestion settings so the
membership check matches the search that produced the data. If the document enzyme has no PRISM
equivalent (or can't be read), the tool keeps the config default.

---

## Current Implementation Status

This section tracks what's currently working, what needs attention, and what's not yet implemented.

### [ISSUE] Known Issues / Needs Attention

**Reference-anchored ComBat degrades held-out QC on a real cohort - unexplained, do not
recommend it until this is understood.**

Measured on AD-Clean-SMTG (4 batches; 62 experimental, 5 `TRPR` reference, 5 `HADT` held-out QC,
biology balanced across batches):

| ComBat mode | peptide-level QC CV | protein-level QC CV |
|---|---|---|
| standard | 23.9% -> **19.8%** (helps) | see the two-arm note below |
| reference-anchored (`TRPR`) | 20.3% -> **25.7%** (worse, auto-reverted) | 16.2% -> **18.3%** (worse, auto-reverted) |

So on this cohort the anchored estimator is worse than the unanchored one, at both levels, and
`auto_revert` backed it out both times. The leading hypothesis is that **one reference injection per
batch is too thin an anchor**: with n=1 the scale term is not estimable at all (the run reports
`UnestimableScales`), and the location term is pinned to a single injection's idiosyncrasies rather
than to the batch. That is a hypothesis, not a diagnosis - it has not been confirmed, and the
alternative (a defect in the anchored estimator) has not been ruled out. `RefAnchoredCrossEngineTests`
pins the current implementation against the goldens the Python engine produced across five
scenarios, so if it is a defect it predates the port.

To investigate: whether the degradation tracks the reference count per batch (compare against a cohort
with >= 2 references per batch), and whether `no_reference_batch` policy or the location-only path
behaves differently. Until then, prefer standard ComBat, and set `auto_revert: true` - it is what
caught both reversions above.

**`auto_revert` is implemented and works** (`BatchCorrectionEvaluator`, `batch_correction.auto_revert`,
default `false`). It reverts when the control CV worsens by >10%, and separately logs a NOTE when
the reference improved far more than the independent control. That asymmetry is an observation only -
it reverts nothing and fails nothing (reference and QC are different materials at different injection
amounts, so their CVs need not improve together). It caught both reversions above.

### [TODO] Not Yet Implemented

- Per-batch RT models with cross-validation
- Quality-weighted protein rollup
- directLFQ (see below)

## Not Yet Implemented

### directLFQ

directLFQ is a protein quantification algorithm that offers linear O(n) runtime scaling, making it suitable for very large cohorts (100s-1000s of samples). It is fundamentally different from maxLFQ - not just an optimization.

**Why it's different from maxLFQ:**
- maxLFQ uses pairwise median log-ratios between samples (O(n²) complexity)
- directLFQ uses an "intensity trace" approach with anchor alignment (O(n) complexity)

**Citation:** Ammar C, Schessner JP, Willems S, Michaelis AC, Mann M. "Accurate label-free quantification by directLFQ to compare unlimited numbers of proteomes." Molecular & Cellular Proteomics. 2023;22(7):100581. doi:10.1016/j.mcpro.2023.100581

**GitHub:** https://github.com/MannLabs/directlfq

**Status:** Not implemented in PRISM. For very large cohorts, users should use the directLFQ package directly. May be added in a future version.

## Design Decisions to Preserve

1. **RT correction from reference only**: Never learn RT effects from experimental samples
2. **Batch correction at reporting level**: each arm is corrected once, at the level it reports, and
   the protein arm branches from the normalized (pre-ComBat) peptide matrix. `peptides_log2_internal.parquet`
   is therefore post-normalization and pre-ComBat - which is what `docs/output_files.md` always said it
   was. What the two flags mean:

   | `peptide_level` | `protein_level` | `corrected_peptides` | `corrected_proteins` |
   |---|---|---|---|
   | on | on (default) | corrected once | corrected once |
   | on | **off** | corrected once | **not batch-corrected at all** |
   | off | on | not corrected | corrected once |
   | off | off | not corrected | not corrected |

   > [!CAUTION]
   > The `on / off` row changed meaning in dotnet-v26.15.0 and does so **silently** - there is
   > deliberately no warning. Before, `protein_level: false` still gave batch-corrected proteins,
   > because the protein arm inherited the peptide correction through its inputs. Now it gives
   > proteins with no batch correction whatsoever. That is the setting someone would reach for to
   > avoid double correction by hand, so it is the one most likely to be misread. Leaving it silent
   > was a deliberate call (small, mostly in-lab user base); do not "fix" it by reintroducing the
   > coupling, which is the thing being removed.
3. **Median polish as default**: Quality-weighted is an alternative, not the primary method
4. **All charge states as transitions**: Don't separate precursor→peptide rollup; treat all transitions equally
5. **Cross-platform CLI, Windows-only GUI** (decided, not an interim state): the `prism` CLI ships for
   Windows/Linux/macOS and `SkylinePrism.Core` stays platform-neutral, while the GUI stays **WPF on
   Windows**. A cross-platform GUI (Avalonia et al.) was considered and declined - Skyline itself is
   Windows-only, so the attached mode could never be portable, and the standalone case is served by the
   CLI. Consequences to respect:
   - Do **not** port `SkylinePrism.App` to another UI framework, and do not add a `net10.0` target to it.
   - GUI-only helpers may sit in `SkylinePrism.App` and depend on the Windows-only
     `SkylinePrism.Skyline` (e.g. `PrismInput`, `StandaloneShortcut`); they do **not** need to move to
     Core "for portability".
   - Anything a headless/Linux user needs must be reachable from the CLI. The GUI's "Show Command Line"
     exists to keep that honest - it emits the exact `prism run` invocation for the current settings.
   - The one thing still worth extracting from `MainWindow` is a view-model for **testability**
     (it is ~1000 lines of code-behind at 0% coverage), not for portability.
6. **C# Stage 1 partitions, and does NOT sort** (measured, not stylistic). `merged_data/` is a
   hive-partitioned parquet directory (`_pep_bucket=N/`), hashed on the peptide column and unsorted
   (the retired Python engine wrote one sorted file instead, because its rollup consumed the file
   positionally). Every row of a peptide is in exactly one partition, which is what lets the rollup sort and stream one partition
   at a time instead of ordering the whole cohort - the only operator whose cost grew with the number
   of documents merged. See `MergedDataset` for the sizing trade (it was measured in both directions,
   and the intuitive answer was wrong) and `docs/output_files.md` for the layout. Row CONTENT remains
   the cross-engine parity contract; row ORDER and file layout explicitly are not.

> [!CAUTION]
> **DuckDB.NET has three defaults that will silently cost you memory or correctness.** All three were
> hit for real; each is documented at its call site, and `DuckDbTuning` exists to centralize the fixes.
>
> - **`DuckDBCommand.UseStreamingMode` defaults to `false`.** A non-streaming `ExecuteReader` runs the
>   query to completion and materializes *every* row client-side before the first `Read()` returns -
>   outside the buffer pool, so neither `memory_limit` nor `temp_directory` applies. This made the
>   "streaming" rollup reader hold an entire 186M-row cohort in memory. Any query whose result set
>   scales with the data must set it.
> - **A connection with no `memory_limit`/`temp_directory` takes 80% of RAM and cannot spill.** Set
>   both, always together: a bounded pool with nowhere to spill fails outright instead of spilling.
> - **Never read from DuckDB on more than one thread.** Concurrent streaming readers corrupt memory -
>   an `AccessViolationException` or a bare segfault, not an exception you can catch. This was tested
>   to destruction and fails in every configuration: a connection per partition; every connection
>   opened up front with none closed mid-stage and a keepalive pinning the instance refcount;
>   genuinely isolated **file-backed** databases; and on DuckDB.NET **1.5.5** as well as 1.5.3. The
>   first failure looked like `Connection.ConnectionManager` (a static cache of *refcounted* database
>   instances keyed by connection string, so every `"Data Source=:memory:"` in the process is the same
>   database) tearing an instance down under a live reader - but the later configurations rule that
>   out, because they fail with no teardown possible and no shared instance at all. Parallel partition
>   readers were built, crashed, and were reverted; see `TransitionRollup.RunParallel`.
> - **`memory_limit` is a *database*-level setting, not a connection one.** Connections sharing an
>   instance - which, per the above, means every `":memory:"` connection in the process - share one
>   budget. Setting it per connection does not give each its own pool.
>
> Stage 2 is consequently single-threaded, and the largest stage: on a 2-plate cohort it is ~58% of
> wall clock (1m21s of 2m20s). That is a measured ceiling, not a mystery.

> [!IMPORTANT]
> **Benchmarking here: check the machine before believing a number.** Absolute throughput figures from
> this repo have been wrong more than once because other software was running - an identical
> configuration measured 2.07 min and 3.70 min hours apart, and the first explanation reached for was
> page-cache warmth rather than the process list. Two rules follow:
>
> - **Prefer ratios to absolute rates.** Contention hits interleaved A/B arms roughly equally, so a
>   comparison survives a busy machine; a MB/s quoted across sessions does not.
> - **Compare against the previous release's binary**, built from its tag in a `git worktree`, rather
>   than against remembered numbers. It takes two minutes and removes the doubt entirely.
>
> - **Make the arms do the same WORK, and assert it.** Equal row counts are not enough. An arm here
>   once looked 3.12x faster because it accumulated an index instead of the values; rows and peptides
>   matched, so the correctness check passed it. `Stage2Bench` now counts values accumulated as well -
>   any new arm must keep that figure equal. A faster arm that does less work produces exactly the
>   number you were hoping for, which is what makes this the easiest benchmark mistake to miss.
>
> `dotnet/bench/Stage2Bench` does all of this automatically (interleaved arms, load sampling, contention
> labelling, and a cross-arm rows/peptides/values check) - use it rather than hand-rolling a timer.
> **`dotnet/STAGE2_THROUGHPUT.md` carries the measurements** - including the two candidate replacements
> for the Stage 2 reader (skip the sort and group in a dictionary; `COPY` a narrow projection and read
> it back with Parquet.Net) that were built, measured, and **both rejected** - plus the verification bar
> anything touching concurrency has to clear. Do not re-propose either without reading why they lost.

## Release Process

PRISM ships as the **C# (.NET 10)** tools - the `prism` CLI and the Windows Skyline external tool -
published to GitHub Releases. There is one release track.

> [!NOTE]
> There used to be a second, Python (`skyline-prism` on PyPI) track, tagged `v{version}`. It ended at
> `v26.4.4` and the engine has been removed; those tags and PyPI versions stay downloadable but are
> unmaintained. The `dotnet-v*` namespace is kept as-is rather than renamed, so existing tags, release
> URLs and the workflow's tag→notes-file mapping keep working.

### Versioning scheme

CalVer **`YY.feature.patch`** (e.g. `26.15.0` = year 2026, feature release 15, patch 0): **YY** =
two-digit year, **feature** bumps for new features, **patch** for bug-fix-only releases. The version is
bumped **only at release time**, not during development. Tags are `dotnet-v{version}`
(e.g. `dotnet-v26.15.0`).

### Release notes

All release notes live in `release-notes/`; **`release-notes/README.md` is the canonical convention**.
One file per release, with a rolling draft renamed at release time: `RELEASE_NOTES_dotnet-next.md` ->
`RELEASE_NOTES_dotnet-v{version}.md`. Content structure:
`## New Features / ## Bug Fixes / ## Performance / ## Breaking Changes` (omit empty sections); past
tense, lead with user impact, include concrete numbers, reference config keys by name.

> [!IMPORTANT]
> **The notes file IS the GitHub Release description - write it for that audience.**
> `dotnet-release.yml` publishes `release-notes/RELEASE_NOTES_${tag}.md` verbatim as the Release body,
> so whatever lands in that file is what users read on the Releases page. Two consequences:
>
> - **Rename the draft before tagging.** The workflow resolves the path from the tag and **fails** if
>   the file is missing - after the artifacts have already built.
> - It renders as GitHub-flavoured Markdown, so keep headings/links/code fences valid; the leading `#`
>   title is redundant with the release's own heading but harmless.
>
> Backfill an old Release with
> `gh release edit <tag> --notes-file release-notes/RELEASE_NOTES_<tag>.md`.

### Making a release

Two version sources MUST stay in lockstep; `dotnet-release.yml` fails the release if either differs
from the tag:

- `dotnet/Directory.Build.props` `<Version>` - drives every C# assembly version, the CLI's
  `prism --version` (prints the 4-part `X.Y.Z.0`), and provenance `pipeline_version`.
- `dotnet/src/SkylinePrism.App/tool-inf/info.properties` `Version =` - the Skyline tool manifest.

Steps:

1. Finalize `release-notes/RELEASE_NOTES_dotnet-next.md`; `git mv` it to
   `RELEASE_NOTES_dotnet-v{version}.md`, update its heading, **delete every section heading with no
   entries under it** (the draft is seeded with all four, and this file is published verbatim as the
   GitHub Release body), and create a fresh empty `RELEASE_NOTES_dotnet-next.md`.
2. Bump the version to `{version}` in **both** `Directory.Build.props` and `info.properties` (they
   must match each other and the tag).
3. **Run the ship gate** locally:
   `pwsh -File dotnet/build/package-and-verify.ps1 -Configuration Release` (tests -> packages
   `SkylinePrism.zip` -> extracts and launch-verifies the exe). Confirm `prism --version` prints
   `{version}.0`.
4. Commit, open a PR to `main`, let CI go green (`dotnet-ci.yml`), run
   `/code-review`, then **squash-merge** it (`gh pr merge --squash --delete-branch`). Always squash,
   for every PR, not just releases - `main` keeps one commit per change. Write the squash commit
   message deliberately; it is the permanent record, and the default (a concatenation of the branch's
   commits) is not.
5. Tag the squashed commit on `main` and push the tag:
   ```bash
   git tag dotnet-v{version} origin/main
   git push origin dotnet-v{version}
   ```
   **Pushing the tag both builds the artifacts AND creates the GitHub Release**
   (`.github/workflows/dotnet-release.yml`). Do NOT hand-create the Release.

   The Release **body is the notes file**: the workflow reads
   `release-notes/RELEASE_NOTES_${tag}.md` (so tag `dotnet-v26.7.2` -> `RELEASE_NOTES_dotnet-v26.7.2.md`)
   and fails with an explicit message if it is missing. That is why step 1's rename must happen *before*
   tagging - a tag pushed without its notes file aborts the release job after the artifacts have built.
   (Releases up to and including v26.7.1 were published with an empty body, before the workflow checked
   out the repo; they have since been backfilled with `gh release edit --notes-file`.)

Artifacts (7 assets, all **framework-dependent** - the .NET 10 runtime is NOT bundled; users install
it once):

- `SkylinePrism.zip` - the Windows Skyline external tool (needs the .NET 10 **Desktop** Runtime).
- `prism-{win-x64,win-arm64,linux-x64,linux-arm64,osx-x64,osx-arm64}.{zip,tar.gz}` - the CLI, one
  native build per platform (needs the base .NET 10 Runtime).

Framework-dependent is deliberate (small downloads, one shared runtime). Archives are still ~20-45 MB
because they bundle the app's native deps (DuckDB, SkiaSharp), not the runtime. To publish a bare
`prism` CLI for a new OS/arch, add a row to the `cli` matrix in `dotnet-release.yml`.

Canonical references: `release-notes/README.md` (notes + step-by-step), `dotnet/README.md` (C#
build/package/CI), and the workflows under `.github/workflows/` (`dotnet-release.yml`, `dotnet-ci.yml`).

## Repository Information

- **GitHub**: https://github.com/maccoss/skyline-prism
- **Owner**: maccoss (MacCoss Lab, University of Washington)
- **License**: MIT
