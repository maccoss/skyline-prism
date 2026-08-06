# C# port status — Python feature coverage

Tracks which Python `skyline_prism` features are implemented in the C# port (`dotnet/`).
Legend: **[x]** done · **[~]** partial · **[ ]** not yet · **[-]** deferred (not planned for now).

## Data I/O & merge
- [x] Streaming CSV/parquet merge (DuckDB), UNION ALL of N reports with `memory_limit` + disk-spill
      `temp_directory` — handles ~200 reports (tested with 30 as a proxy) — `DuckDbMerge`
- [x] Merge memory: budget = 75% of RAM + all cores + `temp_directory` spill. A fixed 8 GB `memory_limit`
      OOM'd on a real 69M-row report because the per-thread parallel-read buffers (one thread per core)
      exceeded it in seconds, before sorting - not the sort working set. Sized to hold the parallel read
      with host headroom; spills for anything beyond. (Caught by the SEA-AD real-world run.)
- [x] Schema-only column detection (footer read) on the merged file; distinct-query streaming for
      samples / batch map / parsimony — no full materialization of the merged table
- [x] Column auto-detection (`find_column`) — `SkylineColumns`
- [x] Sample metadata generation + sample-type detection — pipeline + `ReplicateMetadata`
- [x] Tolerant numeric parsing (`#N/A` → NaN) — `MergedParquetReader`
- [x] Source-fingerprint caching (`SourceFingerprint`; reuse merged_data.parquet when inputs
      unchanged, `--force-reprocess` to rebuild)
- [x] Batch estimation from acquisition-time gaps (`BatchEstimator`, IQR-based; auto/gap/fixed) —
      used only when neither metadata nor Source Document distinguishes batches

## Transition → peptide rollup (Stage 2)
- [x] Preprocess: precursor exclusion, impute, log2 — `RollupPreprocess`
- [x] `sum` — `SumRollup`
- [x] `median_polish` (+log2(n) offset) — `MedianPolishRollup`
- [x] `topn` — `TopNRollup`; selection intensity | correlation (Shape Correlation plumbed through the
      rollup), weighting sum | sqrt. Config keys/defaults aligned to Python: `topn_count`,
      `topn_selection` (default correlation), `topn_weighting` (default sqrt) — a golden-parity run
      caught that C# had used `top_n_*` with intensity/sum defaults.
- [-] `adaptive` (learned weights, L-BFGS-B) / QuantUMS — deferred; `method: adaptive` now ABORTS with
      a clear error (was silently falling back to `sum`). `learn_adaptive_weights` + the `adaptive_rollup:`
      block are reported as unrecognized config keys. See "Config surface & parity" below.
- [x] `consensus` (two-way-median decomposition + inverse-variance transition weighting) — `ConsensusRollup`
- [x] `library_assist` (BLIB **and** Carafe/DIA-NN TSV) — `SpectralLibrary` (BLIB/SQLite reader +
      `LoadCarafeTsv` streaming TSV reader; `Load()` auto-detects by extension) + `LibraryRollup`
      (median-polish with library prior, interference removal, per-charge sum); library picker in the tool.

## Peptide normalization (Stage 2b)
- [x] `median` — `Normalizer.MedianNormalize`
- [x] `rt_lowess` (default) — `Normalizer.RtLowessNormalize` + `Lowess`
- [x] `none`
- [x] `quantile` — `Normalizer.QuantileNormalize`
- [x] `vsn` (arcsinh; param optimization off, as in the pipeline default) — `Normalizer.VsnNormalize`.
      `vsn_params.optimize_params` (Nelder-Mead tuning) is a deliberate non-port (Python default = false).
- [x] `rt_lowess` tuning exposed: `global_normalization.rt_lowess.frac` (0.3) + `.n_grid_points` (100)
- [x] Sample outlier detection (iqr / fold_median) with report **and** exclude actions — wired at Stage 2a

## Batch correction (Stage 2c / 4c)
- [x] Standard ComBat (empirical Bayes), per-level peptide/protein toggles — `ComBat`
- [x] Reference-anchored ComBat (`ReferenceAnchoredComBat`, `batch_correction.reference_anchored`) —
      estimates batch effects from ALL peptides in the reference samples across batches (not
      single-point calibration), EB-shrunk; applied to all samples. Falls back to standard ComBat
      when a batch has no references / none present.
- [x] ComBat auto-evaluate + revert-on-QC-failure (`batch_correction.auto_revert`, opt-in) —
      `BatchCorrectionEvaluator` compares the control CV (QC preferred, else reference) pre/post ComBat
      and reverts to the uncorrected data if it worsened by >10%; also warns on reference/QC overfitting.
      Off by default to match Python's production path (Python has the revert only in legacy
      `normalize_pipeline`). `BatchCorrectionEvaluatorTests`.

## Protein parsimony (Stage 3)
- [x] `compute_protein_groups` (subsumable/indistinguishable/razor) — `ParsimonyEngine`. Subsumable
      detection uses the peptide->protein index (only proteins sharing a's rarest peptide are candidate
      supersets) instead of an O(proteins^2) all-pairs scan - identical result, near-linear; the old scan
      took ~2 h single-threaded on a real 190k-peptide set.
      Order-independent (groups depend only on the peptide/protein sets, not input row order;
      proven by `ParsimonyOspreyTests.Grouping_IsOrderIndependent`). Grouping matches
      maccoss/osprey (identical-set merge + subset elimination -> same maximal groups); the
      default `all_groups` path needs no razor. Razor tiebreak aligned to Osprey (unique count,
      then largest peptide set, then lowest accession) — deterministic even where Osprey's
      group-ID order is arbitrary. Same fix applied to Python `parsimony.py` to keep them in step.
- [x] Skyline CSV-based peptide→protein map
- [x] Parsimony on/off (one group per protein) — `BuildUngroupedGroups`
- [x] FASTA-based map (substring + I/L equivalence + enzyme-aware terminus check) — `FastaParser`;
      set `parsimony.fasta_path`. Osprey reads pre-assigned protein_ids from the library (osprey-io
      library/diann.rs `split_list`), the same model as the Skyline Protein-Accession path; the FASTA
      option re-derives the same substring+I/L edges, then applies `parsimony.enzyme` /
      `parsimony.enzyme_specificity` (default `trypsin` / `full`) so a peptide is only attached to a
      protein it can enzymatically produce (removes phantom homolog assignments — the SNCA/SNCB case).
      C# and Python share the enzyme rules exactly. Verified by `ParsimonyFastaMapTests` +
      `FastaParserTests` (C#) and `test_fasta.py`/`test_parsimony.py` (Python).
- [x] `parsimony.enzyme` / `parsimony.enzyme_specificity` — both engines, same defaults/keys. The
      **Skyline external tool** overrides `enzyme` from the document's digestion settings
      (`SkylineReportDriver.GetDigestionEnzyme` -> `SkylineDigestion`, reading the selected "Enzymes"
      item's cut/no_cut/sense XML); the CLI uses the config default. This document-derived override has
      no Python equivalent (Python is CLI-only) and is not a config key — it just sets the same key.
- Osprey parity is asserted in BOTH languages: `ParsimonyOspreyTests` (C#) and
  `tests/test_parsimony_osprey.py` (Python) run the same identical-set/subset/all-mode/razor/
  determinism cases, keeping Python <-> C# <-> Osprey in lockstep.
- [x] `shared_peptide_handling`: `all_groups`, `unique_only`, `razor` — selectable in ProteinRollup

## Peptide → protein rollup (Stage 4)
- [x] `median_polish` (no offset) — `ProteinMatrixRollup`
- [x] `sum` (sum_linear); 0→NaN, 1→direct, <min→sum_linear dispatch
- [x] `topn` (top-N peptides by median abundance, then per-sample mean)
- [x] `maxlfq` (pairwise median log-ratios -> row-mean reconstruction -> re-anchored)
- [x] `ibaq` — in-silico trypsin digest (`FastaParser.GetTheoreticalPeptideCounts`) -> log2(sum
      linear peptide intensity / n_theoretical); falls back to observed count without a FASTA

## Protein normalization (Stage 4b)
- [x] `median`, `none`

## Output (Stage 5)
- [x] parquet (LINEAR), csv/tsv writers
- [x] `protein_groups.csv`, `sample_metadata.csv`
- [x] `parameters.json` provenance (embeds the full config; named to avoid clashing with
      scientific sample metadata) — `Provenance.Write`
- [x] residuals output (`peptide_residuals.parquet`, per-transition median-polish residuals)
- [x] `--from-provenance` re-run (CLI) + "Open provenance" in the tool — `Provenance.LoadConfig`

## QC report (Stage 5b)
- [x] CV metrics (median CV, linear) — `CvMetrics`
- [x] Self-contained HTML report mirroring the Python before/after layout (peptide + protein)
- [x] Before/after plots: intensity distribution, PCA, comparative CV (reference + QC)
- [x] Interactive QC tabs in the tool (PCA / CV / intensity, peptide+protein) with a raw↔corrected
      view toggle — `MainWindow` QC Plots tab
- [x] Dynamic Range tab in the tool: log10 abundance vs rank over the corrected matrices (Skyline's
      Relative Abundance shape), protein/peptide switch, replicate subset averaging (linear mean, then
      log10), click-to-select in the Skyline tree via `SetSelectedElement` + cached `GetLocations`
      locators, user-defined coloured protein lists (`ProteinListSet`, per-user JSON) and right-click
      label modes — `DynamicRange` + `PlotRenderer.DrawDynamicRange`. C#-only (no Python equivalent)
- [x] Spectrum density tab in the tool: precursors detected per DIA spectrum as an (isolation window x
      RT) map, per run, with a selectable colormap — `PrecursorDensity` +
      `PlotRenderer.DrawPrecursorDensity`. C#-only by design: the Python engine has no equivalent (this
      is a port of Skyline-Cadenza's m/z x RT heatmap, reading the merged PRISM report instead of a
      DIA-NN report)
- [x] Real DIA isolation windows for that map — `IsolationScheme` (parses both Skyline XML spellings) +
      `IsolationSchemeCatalog` (persisted to `isolation_schemes.xml` per run). Read from the document
      when it defines a scheme; for `Results only` documents Skyline keeps the windows only inside the
      raw files (no `.sky` entry, and `ChromatogramExtractionWidth` is the product-ion window, not the
      isolation window), so `SkylineIsolationImporter` has Skyline import them from a data file
      (`--full-scan-isolation-scheme=<data file>` against a throwaway `--new` document — never the user's).
      Saved schemes are the manual fallback, and precursors outside the chosen scheme are counted+warned
- [x] Scheduled acquisitions (PRM / MTM / dynamic DIA) — `IsolationWindow` carries an optional RT firing interval
      (Cadenza's `Slot` model: m/z range x RT range), so membership requires the peak to elute while the
      window fired, unscheduled time renders as "not acquired" rather than zero, and the RT axis spans the
      schedule. Windows come from the instrument's inclusion list (`ThermoInclusionList`, the columns
      Cadenza's `ThermoCsvWriter` writes) since Skyline's importer needs a repeating cycle and a schedule
      has none. Dynamic DIA (PMC10517878) is the same model with a cycle of windows per segment, so the
      display rasterizer picks its source window per CELL, not per m/z row — a per-row choice renders one
      segment and blanks the others
- [x] Control-correlation heatmap, RT-lowess curve overlay, RT-binned CV (all before/after)
- [x] Pass/fail validation status + warnings banner (dual-control: QC CV improvement, RVR overfitting,
      PCA QC-reference distance collapse) — `ValidationStatus`
- [x] RT-bin abundance boxplot (before/after)
- [-] 3-stage plot variants — N/A: our pipeline is 2-stage (raw -> corrected); the Python 3-stage
      needs a persisted post-normalization/pre-ComBat intermediate we don't keep

## CLI
- [x] `run`, `merge`, `qc`, `config-template`, `--version`, `--from-provenance`; per-stage console logging
- [x] `run` flags: `-m/--metadata` (multi-file, merged), `--force-reprocess`; still `--reference-pattern`/`--qc-pattern`
- [x] `--no-save-plots` (skip `qc_plots/*.png`) on `run` + `qc`. Non-ports (deliberate): `qc -o`
      report-name and `--no-embed` (C# hardcodes `qc_report.html` and always base64-embeds); `merge`
      `--no-partition` (C# merge doesn't partition output).
- [x] `config-template --minimal` (`ConfigTemplate.Minimal`). `-v` collision RESOLVED: `-v` is no longer
      a version alias (Python uses it for verbose); use `--version`/`version`. C# has no verbosity levels
      (always logs fully to console + `prism_run_<ts>.log`).
- [x] `compare` (control-CV comparison of two runs + ranked per-peptide CV differences) —
      `RollupComparison`. (The Python's per-peptide library-fit visualization is not ported.)
- [x] Timestamped `prism_run_<ts>.log` in the output dir, tee'd with the console — `Program.cs`
- [x] Source-fingerprint cache + `--force-reprocess` (reuse merged_data.parquet when inputs unchanged) —
      `PrismPipeline` + `SourceFingerprint`
- [x] Unknown-config-key warnings — `PrismConfig.FindUnknownKeys` reports any key the pipeline doesn't
      read (Python-only or typo) on stderr; `PrismConfig.Validate` aborts on unported method choices
      (adaptive / least_squares / non-combat batch). CLI `run` + `qc` go through `LoadValidated`.
- [x] config-template now emits the `rt_lowess`, nested `library_assist`, and `ibaq` length blocks and
      fixes `include_residuals` to match the default; a test guards template ↔ schema stay in sync.
      `protein_rollup.min_peptides` default is 3 (Python runtime is also 3; only Python's *template text*
      says 2). `data` (column map) is a deliberate non-port — see "Config surface & parity" below.

## Concurrency & memory
- [x] Streaming merge + streaming peptide-block reader (single DuckDB producer)
- [x] **Streaming parquet writer** (`StreamingWideWriter`, multi-row-group + periodic flush) — the
      transition rollup + residuals now stream to disk instead of accumulating all rows in memory
- [x] **Bounded-parallel transition rollup** — single DuckDB producer -> capped `BlockingCollection`
      (4×DOP) -> N consumers (`processing.n_workers`: 0=all cores, 1=serial, N=cap) -> single flushing
      writer thread. Per-peptide work is pure/thread-safe; library-assist runs parallel too (Python
      couldn't, only due to pickling). Serial == parallel verified bit-identical; RAM stays flat.
- [x] Parallel protein-group rollup (`Parallel.For`, DOP=processing.n_workers, order-preserving indexed writes)
- [x] `NormalizeAndCorrect` peak memory trimmed (reuse dense matrix, free dead intermediates, log2
      transpose only for the peptide stage)

## Skyline external tool (Windows)
- [x] JSON-RPC session + report driver (PRISM parquet + PRISM-Replicates). Driver now depends on an
      `ISkylineExecutor`/`ISkylineClient` seam (SkylineSession wraps the concrete JSON-RPC client),
      so `SkylineReportDriverTests` cover the parquet-first/CSV-fallback decision, metadata
      report-name resolution, report-list dedup, .blib discovery, and the dynamic report install
      against a fake - no live Skyline needed.
- [x] Dynamic annotation column for batch — emitted QUOTED (`"annotation_<Name>"`) by
      `ReplicatesReportBuilder`, shared by the live-RPC and headless paths. The bare form is rejected by
      Skyline's PropertyPath parser ("Invalid character _") and writes no report, so the requested batch
      annotation never reached the metadata before this.
- [x] Multiple inputs per run (`PrismInput`): documents open in the launching Skyline or in any other
      running instance (`SkylineSession.DiscoverRunning`), CLOSED `.sky` documents exported headlessly
      via `SkylineCmd` (`HeadlessSkylineExporter` + `SkylineCmdLocator`), and already-exported report
      files. Each input's editable batch label is the exported file stem, which is what `DuckDbMerge`
      reads back as its Source Document / Batch. Headless export is invariant CSV — Skyline's command
      line offers only `--report-format=csv|tsv`; the merge mixes CSV and parquet inputs freely.
- [x] Closed-document header read (`SkyDocumentInfo`): replicate-targeted annotation names (to shape the
      generated PRISM-Replicates report), the `<enzyme>` element, and the replicate list — stream-parsed
      and stopped at `</settings_summary>`, so a 2 GB document reads in ~1 s. Read-only; never writes.
- [x] Standalone mode: the window no longer requires a Skyline connection, so `SkylinePrism.exe` doubles
      as a plain PRISM GUI over already-exported reports.
- [-] Cross-platform GUI (Avalonia/MAUI/Uno) — DECLINED by decision, not deferred. Skyline is Windows-only,
      so the attached mode can never be portable, and headless/Linux users are served by the `prism` CLI
      (already shipped for win/linux/osx x64+arm64). `SkylinePrism.App` stays WPF; GUI-only helpers may
      depend on the Windows-only `SkylinePrism.Skyline` without needing to move to Core. See "Design
      Decisions to Preserve" in `CLAUDE.md`.
- [ ] Extract a view-model from `MainWindow` — wanted for TESTABILITY (~1000 lines of code-behind at 0%
      coverage), explicitly NOT for portability.
- [x] WPF app: Inputs tab, report/metadata pickers, Settings panel, interactive PCA, QC report
- [x] Library picker dropdown (.blib files next to the document + Browse), shown for library_assist
- [x] "Open provenance" - load a prior run's parameters.json and propagate settings into the UI
- [-] Results explorer (protein/peptide tree + per-feature boxplots) — DEFERRED by decision; a
      .NET-native results view may be designed later, not a Python-parity goal
- [x] Group-by / color-by any Replicates-report column in the interactive QC tab (Sample Type default;
      any annotation), applied to PCA, CV, and intensity plots with a standardized per-group palette

## Validation status & warnings (Python validation.py)
- [x] Pass/fail verdict + warnings banner (QC-CV-increased, RVR overfitting >2, PCA QC-reference distance
      collapse) — `ValidationStatus.Compute`, rendered in the HTML report (`QcReport.AppendValidation`)
- [x] "Batch source" line — the "Batches: N from <source>" log names where labels came from (metadata
      Batch column > per-file Source Document > acquisition-time estimation > single default label)
- [x] Pattern-based sample-type fallback — Replicates "Sample Type" column first (Standard/Quality
      Control), then `reference_pattern`/`qc_pattern` substrings on the replicate name (`ClassifySampleType`).
      Defaults broadened to Pool/Ref/Reference and QC/Control/StudyPool/Quality Control. NOTE: C# applies
      the pattern fallback with built-in defaults; Python applies patterns only when configured (opt-in).

## Cross-language parity fixtures
- [x] `mini/merge`, `mini/e2e-sum` (sum pipeline; full path incl. ComBat, tolerance-compared)
- [x] `mini/e2e-medpolish` / `e2e-maxlfq` / `e2e-topn` / `e2e-prot-topn` / `e2e-consensus` —
      method-isolation fixtures (ComBat off -> every stage exact-parity to 1e-9) covering
      median_polish, maxLFQ, transition topN, protein topN, and consensus against Python goldens
      (`PipelineMethodParityTests`)
- [x] Protein-rollup topN nested config (`protein_rollup.topn.{n,selection}`); `selection` is now honored
      (`median_abundance` default | `frequency` — swaps the primary sort key), previously parsed-but-ignored.
      median_abundance path verified exact by e2e-prot-topn.
- [x] Library rollup algorithm unit-tested (median-polish scale, interference removal, m/z match)
- [x] Real-world validation (SEA-AD-MTG-Pilot: 82 samples / 89,983 peptides / 7,700 proteins; sum /
      rt_lowess / median_polish, ComBat off): `peptides_rollup` BIT-EXACT vs Python (max 3.5e-15);
      `corrected_peptides` essentially exact (median rel ~0 - rt_lowess matches statsmodels);
      `corrected_proteins` 0.03% median / 0.13% max on the 7,696 shared proteins. The only structural
      diff (6 proteins, all `n_unique=0` in high-homology families: myosins/histone/POTE/Na-channels) is a
      stale-output skew - the Python output (2026-05-25) predates the 2026-07-02 parsimony Osprey-alignment
      applied to both languages, so current Python would match.
- [x] Real-world LIBRARY-ASSIST validation, END-TO-END (Edge-Pilot: 35 samples / 190,726 peptides /
      10,914 proteins; library_assist (438 MB BLIB) / rt_lowess / median_polish, ComBat off): 100% BLIB
      match (0 missing of 6.67M cells = Python's 225,975 matched); `peptides_rollup` BIT-EXACT (max 7.1e-15
      over 6.67M values), `corrected_peptides` essentially exact, `corrected_proteins` 0.017% median /
      0.064% max on the 10,913 shared proteins. Output sample columns carry the `__@__<source>` suffix,
      matching Python. The only structural diff is +1 group (P20671 / H2A1D histone, `n_unique=0` - all
      peptides shared; nothing Python-only) - confirmed the stale-output parsimony version-skew: it's the
      SAME zero-unique-peptide high-homology family that differed in SEA-AD, and the Python output predates
      the 2026-07-02 Osprey tiebreak alignment, so current Python would match. This run
      surfaced + fixed: the merge memory OOM, the Sample-ID output-column suffix, and TWO parsimony hot
      spots (subsumable O(proteins^2) and the razor's O(canonical x iterations) rescan) - parsimony now
      ~1 min (dominated by the parallel FASTA BuildMap), whole pipeline ~2 min on cached merge.
- [~] End-to-end `output-lib-sum` parity gate — the Carafe-TSV loader now exists; the golden fixture
      still needs to be wired as a `PipelineMethodParityTests` case

## Config surface & parity (parameter exposure)

Audited the full C# config against Python's `KNOWN_CONFIG_KEYS` + `config-template`. Policy: every Python
setting is either implemented or a **deliberate, documented non-port that surfaces at runtime** — an
unknown-key warning on stderr, or a hard error for method choices. Nothing is silently dropped. Guarded by
`ConfigValidationTests` (unknown-key detection, nested-block resolution, method-choice aborts, template ↔
schema sync).

**Newly exposed (were hardcoded, or parsed-but-ignored):**
- [x] `global_normalization.rt_lowess.frac` (0.3) + `.n_grid_points` (100) — threaded into `RtLowessNormalize`
- [x] `transition_rollup.library_assist:` nested block (Python's canonical form) — folded onto the flat
      `library_*` keys by `ResolveLibraryAssist` (nested wins); BOTH forms accepted
- [x] `library_assist.remove_outliers` / flat `library_remove_outliers` — threaded to `LibraryRollup`
- [x] `protein_rollup.topn.selection: frequency` — now honored (was silently `median_abundance`)
- [x] `protein_rollup.ibaq.{min,max}_peptide_length` — threaded into the iBAQ in-silico digest
- [x] `output.include_residuals` template default corrected to `true` (matched the code default)
- [x] `batch_correction.auto_revert` (opt-in, C#-only safety net) — revert ComBat if it worsens the
      control CV by >10%; Python has this only in its legacy `normalize_pipeline`
- [x] `data.*` column-name overrides (peptide / protein / protein_name / abundance / rt / sample /
      transition + batch / sample_type) — honored by `SkylineColumns.Detect`, winning over auto-detect;
      an override resolves if present else falls back to auto (like Python's `data.peptide_column`).
      Matching is robust to invariant (no-space) vs English (spaced) vs underscore + case, so one config
      works for `.parquet` and `.csv/.tsv` exports. batch/sample_type also accepted under `metadata.*`.
      (`ColumnDetectionTests`; verified on the real SEA-AD invariant + phospho parquets.)
- removed dead `qc_report.embed_plots` (parsed, never read)

**Deliberate non-ports (documented; surfaced at runtime):**
- [-] `transition_rollup.method: adaptive` + `adaptive_rollup.*` + `learn_adaptive_weights` — adaptive/QuantUMS
      rollup not ported; `method: adaptive` aborts, the keys warn as unrecognized
- [-] `library_assist.fitting_method: least_squares` — only median_polish ported; `least_squares` aborts
- [-] `batch_correction.method` non-combat — ComBat is the only algorithm; other values abort
- [-] `global_normalization.vsn_params.optimize_params` — VSN runs unoptimized (Python default = false)
- [-] `qc_report.plots.*` per-plot toggles, `qc_report.filename`, `embed_plots` link-mode — C# emits a
      fixed, always-base64-embedded plot set
- [-] `protein_rollup.median_polish.{max_iterations,convergence_tolerance}` — `TukeyMedianPolish`
      convergence is hardcoded to the Python defaults
- [-] niche / no-op: `sample_annotations.experimental_pattern`, `batch_estimation.{min,max}_samples_per_batch`,
      `transition_rollup.enabled`, `output.compress`
