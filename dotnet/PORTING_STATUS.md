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
- [ ] Source-fingerprint caching (skip re-merge when inputs unchanged)
- [x] Batch estimation from acquisition-time gaps (`BatchEstimator`, IQR-based; auto/gap/fixed) —
      used only when neither metadata nor Source Document distinguishes batches

## Transition → peptide rollup (Stage 2)
- [x] Preprocess: precursor exclusion, impute, log2 — `RollupPreprocess`
- [x] `sum` — `SumRollup`
- [x] `median_polish` (+log2(n) offset) — `MedianPolishRollup`
- [x] `topn` (intensity selection + sum weighting) — `TopNRollup`. Correlation-selection / sqrt-weighting
      variants need the shape-correlation matrix (not yet plumbed).
- [-] `adaptive` (learned weights, L-BFGS-B) — deferred, not planned for now
- [ ] `consensus`
- [x] `library_assist` (BLIB) — `SpectralLibrary` (BLIB/SQLite reader) + `LibraryRollup`
      (median-polish with library prior, interference removal, per-charge sum); library picker in
      the tool. Carafe-TSV library loader not yet ported (BLIB only).

## Peptide normalization (Stage 2b)
- [x] `median` — `Normalizer.MedianNormalize`
- [x] `rt_lowess` (default) — `Normalizer.RtLowessNormalize` + `Lowess`
- [x] `none`
- [x] `quantile` — `Normalizer.QuantileNormalize`
- [x] `vsn` (arcsinh; param optimization off, as in the pipeline default) — `Normalizer.VsnNormalize`
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
- [x] `compute_protein_groups` (subsumable/indistinguishable/razor) — `ParsimonyEngine`
- [x] Skyline CSV-based peptide→protein map
- [x] Parsimony on/off (one group per protein) — `BuildUngroupedGroups`
- [x] FASTA-based map (substring + I/L equivalence) — `FastaParser`; set `parsimony.fasta_path`
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
- [x] Control-correlation heatmap (before/after) + RT-lowess curve overlay (before/after)
- [ ] RT-bin boxplot / RT-bin CV; pass/fail validation status + warnings layer

## CLI
- [x] `run`, `merge`, `qc`, `config-template`, `--version`, `--from-provenance`; per-stage console logging
- [~] `run` flag gaps: **`-m/--metadata` not wired** (engine supports `metadataPath` but CmdRun never
      passes it — high-impact small fix), `--reference-pattern` / `--qc-pattern`, `--force-reprocess`
- [~] `qc` flag gaps: `-o` report-name, `--no-save-plots`, `--no-embed`; `merge`: `-m`, `--no-partition`
- [ ] `config-template --minimal`; `-v` means version in C# but verbose in Python (collision)
- [ ] `compare` (rollup/CV comparison report) — not ported
- [ ] Timestamped `prism_run_<ts>.log` written to the output dir (CLI logs to console only)
- [ ] Source-fingerprint cache + `--force-reprocess` (C# always re-merges)
- [ ] Unknown-config-key warnings (C# silently ignores unknown keys)
- [ ] config-template omits `data` (column map), `batch_estimation`, `processing`, and rollup/norm
      sub-parameter blocks; `protein_rollup.min_peptides` default differs (Python 2 vs C# 3)

## Concurrency & memory
- [x] Streaming merge + streaming peptide-block reader (single DuckDB producer)
- [x] **Streaming parquet writer** (`StreamingWideWriter`, multi-row-group + periodic flush) — the
      transition rollup + residuals now stream to disk instead of accumulating all rows in memory
- [x] **Bounded-parallel transition rollup** — single DuckDB producer -> capped `BlockingCollection`
      (4×DOP) -> N consumers (`processing.n_workers`: 0=all cores, 1=serial, N=cap) -> single flushing
      writer thread. Per-peptide work is pure/thread-safe; library-assist runs parallel too (Python
      couldn't, only due to pickling). Serial == parallel verified bit-identical; RAM stays flat.
- [ ] Parallelize protein-group rollup (`Parallel.ForEach`, matrix read-only, indexed writes)
- [ ] `NormalizeAndCorrect` holds several full feature×sample copies at once (reducible ~half in place)

## Skyline external tool (Windows)
- [x] JSON-RPC session + report driver (PRISM parquet + PRISM-Replicates)
- [x] Dynamic annotation column (`annotation_<Name>`) for batch
- [x] WPF app: report/metadata pickers, Settings panel, interactive PCA, QC report
- [x] Library picker dropdown (.blib files next to the document + Browse), shown for library_assist
- [x] "Open provenance" - load a prior run's parameters.json and propagate settings into the UI
- [-] Results explorer (protein/peptide tree + per-feature boxplots) — DEFERRED by decision; a
      .NET-native results view may be designed later, not a Python-parity goal
- [ ] PCA color-by / group-by arbitrary metadata column (Python offers sample_type / batch / any category)

## Validation status & warnings (Python validation.py)
- [ ] Pass/fail verdict + warnings (QC-CV-increased, RVR overfitting >2, PCA QC-reference distance
      collapse) rendered as a status banner; C# report has metrics/plots but no verdict layer
- [ ] "Batch source" line (which of Source Document / metadata / acquisition-time supplied batch labels)
- [ ] Pattern-based sample-type fallback (`reference_pattern`/`qc_pattern` regex when no metadata file)

## Cross-language parity fixtures
- [x] `mini/merge`, `mini/e2e-sum` (sum pipeline)
- [x] Library rollup algorithm unit-tested (median-polish scale, interference removal, m/z match)
- [ ] `mini/e2e-medpolish`, end-to-end `output-lib-sum` parity gate (needs the Carafe-TSV loader)
