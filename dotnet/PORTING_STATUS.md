# C# port status — Python feature coverage

Tracks which Python `skyline_prism` features are implemented in the C# port (`dotnet/`).
Legend: **[x]** done · **[~]** partial · **[ ]** not yet · **[-]** deferred (not planned for now).

## Data I/O & merge
- [x] Streaming CSV/parquet merge (DuckDB), UNION ALL of N reports with `memory_limit` + disk-spill
      `temp_directory` — handles ~200 reports (tested with 30 as a proxy) — `DuckDbMerge`
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
- [x] `library_assist` (BLIB) — `SpectralLibrary` (BLIB/SQLite reader) + `LibraryRollup`
      (median-polish with library prior, interference removal, per-charge sum); library picker in
      the tool. Carafe-TSV library loader not yet ported (BLIB only).

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
- [ ] ComBat auto-evaluate + revert-on-QC-failure — NOTE: only in Python's LEGACY `normalize_pipeline`,
      not its production CLI path; valuable safety feature but lower priority

## Protein parsimony (Stage 3)
- [x] `compute_protein_groups` (subsumable/indistinguishable/razor) — `ParsimonyEngine`.
      Order-independent (groups depend only on the peptide/protein sets, not input row order;
      proven by `ParsimonyOspreyTests.Grouping_IsOrderIndependent`). Grouping matches
      maccoss/osprey (identical-set merge + subset elimination -> same maximal groups); the
      default `all_groups` path needs no razor. Razor tiebreak aligned to Osprey (unique count,
      then largest peptide set, then lowest accession) — deterministic even where Osprey's
      group-ID order is arbitrary. Same fix applied to Python `parsimony.py` to keep them in step.
- [x] Skyline CSV-based peptide→protein map
- [x] Parsimony on/off (one group per protein) — `BuildUngroupedGroups`
- [x] FASTA-based map (substring + I/L equivalence) — `FastaParser`; set `parsimony.fasta_path`.
      Osprey reads pre-assigned protein_ids from the library (osprey-io library/diann.rs `split_list`),
      the same model as the Skyline Protein-Accession path; the FASTA option re-derives the same
      substring+I/L edges. Verified by `ParsimonyFastaMapTests`.
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
- [x] Control-correlation heatmap, RT-lowess curve overlay, RT-binned CV (all before/after)
- [x] Pass/fail validation status + warnings banner (dual-control: QC CV improvement, RVR overfitting,
      PCA QC-reference distance collapse) — `ValidationStatus`
- [x] RT-bin abundance boxplot (before/after)
- [-] 3-stage plot variants — N/A: our pipeline is 2-stage (raw -> corrected); the Python 3-stage
      needs a persisted post-normalization/pre-ComBat intermediate we don't keep

## CLI
- [x] `run`, `merge`, `qc`, `config-template`, `--version`, `--from-provenance`; per-stage console logging
- [x] `run` flags: `-m/--metadata` (multi-file, merged), `--force-reprocess`; still `--reference-pattern`/`--qc-pattern`
- [~] `qc` flag gaps: `-o` report-name, `--no-save-plots`, `--no-embed`; `merge`: `-m`, `--no-partition`
- [x] `config-template --minimal` (`ConfigTemplate.Minimal`). NOTE: `-v` = version in C# vs verbose in
      Python is an unresolved minor CLI-flag collision.
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
- [x] Dynamic annotation column (`annotation_<Name>`) for batch
- [x] WPF app: report/metadata pickers, Settings panel, interactive PCA, QC report
- [x] Library picker dropdown (.blib files next to the document + Browse), shown for library_assist
- [x] "Open provenance" - load a prior run's parameters.json and propagate settings into the UI
- [-] Results explorer (protein/peptide tree + per-feature boxplots) — DEFERRED by decision; a
      .NET-native results view may be designed later, not a Python-parity goal
- [x] Group-by / color-by any Replicates-report column in the interactive QC tab (Sample Type default;
      any annotation), applied to PCA, CV, and intensity plots with a standardized per-group palette

## Validation status & warnings (Python validation.py)
- [x] Pass/fail verdict + warnings banner (QC-CV-increased, RVR overfitting >2, PCA QC-reference distance
      collapse) — `ValidationStatus.Compute`, rendered in the HTML report (`QcReport.AppendValidation`)
- [~] "Batch source" line — logs "N batches from acquisition-time gaps" when estimating, but not an
      explicit source line for the Source-Document / metadata cases
- [ ] Pattern-based sample-type fallback (`reference_pattern`/`qc_pattern` regex when no metadata file)

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
- [ ] End-to-end `output-lib-sum` parity gate (needs the Carafe-TSV loader)

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
- removed dead `qc_report.embed_plots` (parsed, never read)

**Deliberate non-ports (documented; surfaced at runtime):**
- [-] `transition_rollup.method: adaptive` + `adaptive_rollup.*` + `learn_adaptive_weights` — adaptive/QuantUMS
      rollup not ported; `method: adaptive` aborts, the keys warn as unrecognized
- [-] `library_assist.fitting_method: least_squares` — only median_polish ported; `least_squares` aborts
- [-] `batch_correction.method` non-combat — ComBat is the only algorithm; other values abort
- [-] `global_normalization.vsn_params.optimize_params` — VSN runs unoptimized (Python default = false)
- [-] `data.*` column-name overrides — C# auto-detects columns (`SkylineColumns`); no override. (C# reads
      the batch / sample-type column names under `metadata.*`, not `data.*`.)
- [-] `qc_report.plots.*` per-plot toggles, `qc_report.filename`, `embed_plots` link-mode — C# emits a
      fixed, always-base64-embedded plot set
- [-] `protein_rollup.median_polish.{max_iterations,convergence_tolerance}` — `TukeyMedianPolish`
      convergence is hardcoded to the Python defaults
- [-] niche / no-op: `sample_annotations.experimental_pattern`, `batch_estimation.{min,max}_samples_per_batch`,
      `transition_rollup.enabled`, `output.compress`
