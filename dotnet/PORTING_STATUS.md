# C# port status — Python feature coverage

Tracks which Python `skyline_prism` features are implemented in the C# port (`dotnet/`).
Legend: **[x]** done · **[~]** partial · **[ ]** not yet · **[-]** deferred (not planned for now).

## Data I/O & merge
- [x] Streaming CSV/parquet merge (DuckDB) — `DuckDbMerge`
- [x] Column auto-detection (`find_column`) — `SkylineColumns`
- [x] Sample metadata generation + sample-type detection — pipeline + `ReplicateMetadata`
- [x] Tolerant numeric parsing (`#N/A` → NaN) — `MergedParquetReader`
- [ ] Source-fingerprint caching (skip re-merge when inputs unchanged)
- [ ] Batch estimation from acquisition-time gaps (batch comes from Source Document / metadata only)

## Transition → peptide rollup (Stage 2)
- [x] Preprocess: precursor exclusion, impute, log2 — `RollupPreprocess`
- [x] `sum` — `SumRollup`
- [x] `median_polish` (+log2(n) offset) — `MedianPolishRollup`
- [ ] `topn`
- [-] `adaptive` (learned weights, L-BFGS-B) — deferred, not planned for now
- [ ] `consensus`
- [x] `library_assist` (BLIB) — `SpectralLibrary` (BLIB/SQLite reader) + `LibraryRollup`
      (median-polish with library prior, interference removal, per-charge sum); library picker in
      the tool. Carafe-TSV library loader not yet ported (BLIB only).

## Peptide normalization (Stage 2b)
- [x] `median` — `Normalizer.MedianNormalize`
- [x] `rt_lowess` (default) — `Normalizer.RtLowessNormalize` + `Lowess`
- [x] `none`
- [ ] `quantile`
- [ ] `vsn`
- [~] Sample outlier detection — `OutlierDetector` detects; `exclude` action not wired into the pipeline

## Batch correction (Stage 2c / 4c)
- [x] Standard ComBat (empirical Bayes), per-level peptide/protein toggles — `ComBat`
- [ ] Reference-anchored ComBat (Pino 2020)

## Protein parsimony (Stage 3)
- [x] `compute_protein_groups` (subsumable/indistinguishable/razor) — `ParsimonyEngine`
- [x] Skyline CSV-based peptide→protein map
- [x] Parsimony on/off (one group per protein) — `BuildUngroupedGroups`
- [ ] FASTA-based map (substring + I/L equivalence)
- [~] `shared_peptide_handling`: `all_groups` done; `unique_only` / `razor` not wired into rollup

## Peptide → protein rollup (Stage 4)
- [x] `median_polish` (no offset) — `ProteinMatrixRollup`
- [x] `sum` (sum_linear); 0→NaN, 1→direct, <min→sum_linear dispatch
- [ ] `topn`
- [ ] `maxlfq`
- [ ] `ibaq`

## Protein normalization (Stage 4b)
- [x] `median`, `none`

## Output (Stage 5)
- [x] parquet (LINEAR), csv/tsv writers
- [x] `protein_groups.csv`, `sample_metadata.csv`
- [ ] `metadata.json` provenance
- [ ] residuals output (`*_residuals.parquet`)
- [ ] `--from-provenance` re-run

## QC report (Stage 5b)
- [x] CV metrics (median CV, linear) — `CvMetrics`
- [~] Self-contained HTML report — structure + CV tables + a subset of plots
- [x] Plots: CV histogram, PCA, intensity distribution
- [ ] Plots: boxplot comparison, control correlation, RT-lowess overlay, RT-bin boxplot, RT-bin CV, 3-stage variants (full 15-plot set)

## CLI
- [x] `run`, `merge`, `qc`, `config-template`
- [ ] `compare` (rollup comparison report)
- [ ] Byte-exact `config-template` YAML (currently a functional template)

## Skyline external tool (Windows)
- [x] JSON-RPC session + report driver (PRISM parquet + PRISM-Replicates)
- [x] Dynamic annotation column (`annotation_<Name>`) for batch
- [x] WPF app: report/metadata pickers, Settings panel, interactive PCA, QC report
- [x] Library picker dropdown (.blib files next to the document + Browse), shown for library_assist

## Cross-language parity fixtures
- [x] `mini/merge`, `mini/e2e-sum` (sum pipeline)
- [x] Library rollup algorithm unit-tested (median-polish scale, interference removal, m/z match)
- [ ] `mini/e2e-medpolish`, end-to-end `output-lib-sum` parity gate (needs the Carafe-TSV loader)
