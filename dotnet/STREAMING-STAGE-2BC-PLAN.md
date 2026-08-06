# Plan: stream Stage 2b/2c (peptide normalization + ComBat)

**Status:** not started. This document is the handoff for a fresh session.
**Branch to work from:** `feat/density-and-dynamic-range-tabs` (or `main` once it is merged).

## 1. Goal and why

`PrismPipeline.NormalizeAndCorrect` is the pipeline's memory wall. It materializes the whole
peptide x sample matrix in memory, several times over. Stage 1 (merge) and Stage 2 (transition ->
peptide rollup) already stream; Stage 2b/2c does not, so a cohort of ~100 Skyline documents
(thousands of replicates) OOMs there and nowhere else.

Target: peak memory O(row group x samples) + O(rows x batches) instead of O(rows x samples).

### Measured today (synthetic, `GC.GetTotalMemory`)

| Quantity | 150k peptides | 500k peptides |
|---|---|---|
| Peptide -> protein-group index (added by the grouping-columns feature) | 16.8 MB | 59.9 MB |
| The 4 joined grouping columns | 8.7 MB | 29.5 MB |
| **One** peptide x 5,000-sample matrix copy | **5.6 GB** | 18.6 GB |

The grouping columns are noise. The matrices are the problem.

### Already done (interim, do not redo)

- `matrixAll`, `matrix` and `normalized` are nulled as they die (pre-existing).
- Commit `8ab5786`: the loaded `ParquetTable`'s **sample columns are released** right after they are
  copied into the working matrix, via the new `ParquetTable.ReleaseColumns`. Parquet nullable columns
  arrive as `double?[]` (16 B/cell vs the matrix's 8), so they were ~half the stage's peak and were held
  from load until return. This roughly cut peak in half for zero arithmetic change.
- Commit `aa80e3f`: the peptide->group index is released once the peptide output is written.

Remaining live at peak (per cell of rows x samples): `normalized` (8 B) + `corrected` (8 B) + one
transpose for writing (8 B), plus ComBat's internal `sData` and `bayes` (8 B each) while it runs.

## 2. Decisions already taken (do not relitigate)

1. **Quantile normalization is NOT streamed.** It stays on the current in-memory path, with a logged
   note when selected. Rationale below (§4). It is not the default.
2. **Parity must be strict.** A silent numerical change is the main risk of this work. See §5 - the
   existing cross-language parity tests are far too loose to protect this refactor, so the FIRST task is
   building a tight C#-vs-C# harness.

## 3. The key insight: ComBat streams exactly

`ComBat.Run(data[nf, nSamples], batchLabels)` looks unstreamable because `ItSol` (the empirical-Bayes
fixed-point iteration) re-reads the standardized matrix `sData` on every iteration. It does not have to.

Per row `f` and batch `i`, `ItSol` needs `sum over batch-i samples of (sData[f,s] - gammaStar)^2`, which
expands to:

```
S2 - 2 * gammaStar * S1 + n * gammaStar^2
      where S1 = sum(sData[f,s]), S2 = sum(sData[f,s]^2) over batch i's samples
```

So **two sufficient statistics per (batch, row)** are enough to run the entire EB iteration - no access
to the full standardized matrix. Everything else ComBat needs is already per-row:

| Quantity | Shape | Streamable? |
|---|---|---|
| `grandMean[f]`, `varPooled[f]` | O(rows) | yes - from that row alone |
| `gammaHat[i,f]`, `deltaHat[i,f]` | O(batches x rows) | yes - from that row alone |
| `S1[i,f]`, `S2[i,f]` (new) | O(batches x rows) | yes - accumulate per row |
| priors (`gammaBar`, `t2`, `aPrior`, `bPrior`) | O(batches) | computed from the above across rows |
| `ItSol` / `IntEprior` | O(batches x rows) | operate on the above, not on `sData` |
| final adjustment | per row | yes, given `gammaStar`/`deltaStar` + per-row stats |

At 100k rows x 10 batches that is ~10 MB. `ReplaceZeroWithMedianOfPositive(varPooled)` needs all rows'
`varPooled` first - fine, it is O(rows).

⚠️ `ReferenceAnchoredComBat.Run` and `BatchCorrectionEvaluator.Evaluate` (auto-revert) must get the same
treatment; auto-revert compares control CVs before/after, both of which are streamable, but note it needs
the *pre-ComBat* values to revert to, so the revert decision must be made before pass 2 commits output.

## 4. Normalization wants the opposite orientation

Parquet is columnar, so reading one sample column at a time is O(rows), not O(rows x samples).

| Method | Needs | Plan |
|---|---|---|
| `median` | per-sample median over all rows | column-at-a-time pass -> per-sample offsets |
| `rt_lowess` | per-sample LOWESS vs `mean_rt`, then global median curve | column-at-a-time -> per-sample curves (grid is small) |
| `vsn` | per-sample median of linear | column-at-a-time |
| `none` | - | - |
| `quantile` | value -> rank within its own column, at apply time | **NOT streamable row-wise.** Mapping a cell needs that whole column in memory; doing it for every column is the full matrix again. Keep in-memory (decision §2.1). |

## 5. Parity strategy - DO THIS FIRST

⚠️ **The existing golden tests will not catch a regression here.** `PipelineParityTests` compares against
the Python goldens with:

- `peptides_rollup.parquet`: `absTol 1e-9, relTol 1e-9` (tight)
- `corrected_peptides.parquet`: `absTol 1e-6, **relTol 3e-2**`
- `corrected_proteins.parquet`: same

That 3% relative tolerance exists because C# ComBat and Python ComBat legitimately diverge. It would wave
through a serious streaming bug.

**Task 0 (before any refactor):** build a C#-vs-C# regression harness.

1. Keep the current implementation reachable (e.g. rename to `NormalizeAndCorrectInMemory`) so both paths
   can run in one test process.
2. On the `mini` fixtures and a synthetic wider cohort, run both paths and compare corrected outputs at
   ~`1e-12` relative - orders of magnitude tighter than the cross-language tests.
3. Cover every combination that changes the code path: each normalization method (incl. `quantile`, which
   must go down the in-memory path), ComBat on/off, reference-anchored on/off, auto-revert triggering and
   not, and a case with all-NaN rows dropped (`keep` != all rows).

**Summation order matters.** `NumpyMath.PairwiseSum` over a full column is not bit-identical to
accumulating across row groups. Where the harness demands exactness, preserve the existing reduction order
(e.g. accumulate per row group then combine with the same pairwise scheme), or accept a documented
`1e-12`-style tolerance - but decide deliberately, per statistic, rather than discovering it in a red test.

## 6. Implementation order

1. **Task 0** - the parity harness above. Nothing else starts until this is green against the unchanged
   implementation (it should trivially pass, proving the harness works).
2. **Phase A - column pass.** Compute per-sample normalization factors by reading one column at a time.
   New helper, e.g. `Normalizer.ComputeFactors(path, samples, method, ...)`. Verify factors match those
   the in-memory path computes.
3. **Phase B - row pass 1.** Stream row groups; apply factors on the fly; accumulate `grandMean`,
   `varPooled`, `gammaHat`, `deltaHat`, `S1`, `S2`, plus the before-CV metrics and the all-NaN row mask.
4. **Phase C - EB.** Priors + `ItSol`/`IntEprior` rewritten against sufficient statistics. Unit-test
   against the existing `ComBat.Run` on a small matrix: identical `gammaStar`/`deltaStar`.
5. **Phase D - row pass 2.** Stream again, re-apply, correct, and write incrementally with the existing
   `StreamingWideWriter` (`Create` + `WriteRowGroup(metaColumnData, sampleColumnData)`), which the
   transition rollup already uses. This also removes the full `linearCols`/`log2Cols` transposes.
6. **Wire up.** Route Stage 2b/2c through the streamed path; keep the in-memory path for `quantile` and as
   a fallback. Log which path ran.
7. Re-run the full gate and the new harness; measure peak on a synthetic wide cohort.

## 7. Files and landmarks

| File | What |
|---|---|
| `src/SkylinePrism.Core/Pipeline/PrismPipeline.cs` | `NormalizeAndCorrect` (~line 400+): load, drop-NaN rows, CVs, normalize, ComBat, meta columns, write. Also `PeptideGroupIndex` / `PeptideGroupColumns` (the derived grouping columns) |
| `src/SkylinePrism.Core/BatchCorrection/ComBat.cs` | `Run`, `ItSol`, `IntEprior`, `PostMean`, `ReplaceZeroWithMedianOfPositive` |
| `src/SkylinePrism.Core/BatchCorrection/ReferenceAnchoredComBat.cs` | reference-anchored variant |
| `src/SkylinePrism.Core/Qc/BatchCorrectionEvaluator.cs` | auto-revert decision |
| `src/SkylinePrism.Core/Normalization/Normalizer.cs` | `MedianNormalize`, `RtLowessNormalize`, `QuantileNormalize`, `VsnNormalize` |
| `src/SkylinePrism.Core/IO/StreamingWideWriter.cs` | row-group writer to reuse in Phase D |
| `src/SkylinePrism.Core/IO/ParquetTable.cs` | `Load`, `GetDouble` (coerces to `double?[]`), `ReleaseColumns` |
| `src/SkylinePrism.Core/Rollup/TransitionRollup.cs` | the existing streamed stage - the pattern to follow |
| `tests/SkylinePrism.Tests/Pipeline/PipelineParityTests.cs` | the loose cross-language goldens (see §5) |
| `tests/fixtures/mini/e2e-*/` | end-to-end fixtures with configs + expected output |

## 8. Gotchas that will bite

- **Scale conventions.** Internals are LOG2; `corrected_*.parquet` are LINEAR (`2^x`). See CLAUDE.md -
  this has been regressed before. The internal `peptides_log2_internal.parquet` stays LOG2.
- **Derived grouping columns go on the CORRECTED output only.** `protein_group`, `leading_protein`,
  `leading_name`, `leading_gene_name` must NOT be written to `peptides_log2_internal.parquet`: its
  readers (`QcReport.LoadMatrix`, `ProteinRollup`) treat any undeclared column as a sample and will try
  to parse `"PG0003"` as an abundance. That regression already happened once and was caught by
  `QcReportTests`.
- **`keep` (all-NaN row filtering) must be preserved**, including that meta columns and the derived
  columns are filtered by the same index set.
- **`n == nAll` aliasing.** The current code deliberately aliases `matrix = matrixAll` when no rows are
  dropped; the streamed version has no equivalent, but any test comparing the two paths must exercise
  both the dropped and non-dropped cases.
- **QC report reads `peptides_log2_internal.parquet` and `peptides_rollup.parquet`**, not
  `corrected_peptides.parquet` - do not change those schemas.
- Run the ship gate, not just `dotnet test`:
  `pwsh -File dotnet/build/package-and-verify.ps1 -Configuration Release`.
