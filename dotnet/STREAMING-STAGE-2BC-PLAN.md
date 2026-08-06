# Plan: stream Stage 2b/2c (peptide normalization + ComBat)

**Status: DONE for the default configuration.** Task 0 and Phases A-D landed on
`feat/density-and-dynamic-range-tabs`. What is still in-memory, and why, is in §9.

## 1. Goal and why

`PrismPipeline.NormalizeAndCorrect` was the pipeline's memory wall. It materialized the whole
peptide x sample matrix in memory, several times over. Stage 1 (merge) and Stage 2 (transition ->
peptide rollup) already streamed, so a cohort of ~100 Skyline documents (thousands of replicates)
OOM'd there and nowhere else.

Target: peak memory O(row group x samples) + O(rows x batches) instead of O(rows x samples).

### Measured, before and after

| Quantity | 150k peptides | 500k peptides |
|---|---|---|
| Peptide -> protein-group index (added by the grouping-columns feature) | 16.8 MB | 59.9 MB |
| The 4 joined grouping columns | 8.7 MB | 29.5 MB |
| **One** peptide x 5,000-sample matrix copy | **5.6 GB** | 18.6 GB |

The grouping columns were noise. The matrices were the problem.

**Result** (synthetic 20,000 peptides x 600 samples, median normalization + ComBat, peak managed
heap sampled during the stage):

| Implementation | Peak |
|---|---|
| In-memory | 798 MB |
| Streaming | **102 MB** |

The gap widens with sample count: the in-memory peak is a fixed multiple of rows x samples, while
what remains in the streaming path is bounded by the input's row-group size (2,000 rows), not by the
cohort.

### Interim steps taken before the refactor (do not redo)

- `matrixAll`, `matrix` and `normalized` are nulled as they die (pre-existing).
- Commit `8ab5786`: the loaded `ParquetTable`'s **sample columns are released** right after they are
  copied into the working matrix, via `ParquetTable.ReleaseColumns`.
- Commit `aa80e3f`: the peptide->group index is released once the peptide output is written.

Both still apply to the in-memory implementation, which remains the fallback.

## 2. Decisions taken (do not relitigate)

1. **Quantile normalization is NOT streamed.** Mapping a cell needs its whole column's rank
   distribution at apply time. It stays on the in-memory path, and the pipeline logs which path ran.
2. **Parity is strict.** The existing cross-language parity tests are far too loose to protect this
   work (§5), so a C#-vs-C# harness came first.

## 3. The key insight: ComBat streams exactly

`ComBat.Run(data[nf, nSamples], batchLabels)` looks unstreamable because `ItSol` (the empirical-Bayes
fixed-point iteration) re-reads the standardized matrix `sData` on every iteration. It does not have
to. Per row `f` and batch `i`, `ItSol` needs `sum over batch-i samples of (sData[f,s] - gammaStar)^2`.

⚠️ **This plan originally proposed `S2 - 2*gammaStar*S1 + n*gammaStar^2` (raw sum and sum-of-squares).
That form was NOT used** - it subtracts two large nearly-equal quantities, and on a batch whose
standardized values sit far from zero it cancels away most of its significant digits. The
implementation uses the shift-about-the-mean identity instead:

```
sum (x_k - g)^2  =  sum (x_k - m)^2  +  n * (m - g)^2      where m = mean(x) = gammaHat[i,f]
```

Both terms are the same order as the answer, so it only ever costs a few ulps. Better still, the two
quantities it needs are ones ComBat already computes: `gammaHat[i,f]` and the sum of squared
deviations that `Stats.Var(row, ddof: 1)` is built from. `StreamingComBatTests` pins both the
agreement with the in-memory `ItSol` and the margin over the expanded form.

Everything else ComBat needs is per-row:

| Quantity | Shape | Streamable? |
|---|---|---|
| `grandMean[f]`, `varPooled[f]` | O(rows) | yes - from that row alone |
| `gammaHat[i,f]`, `sumSq[i,f]` | O(batches x rows) | yes - accumulate per row |
| priors (`gammaBar`, `t2`, `aPrior`, `bPrior`) | O(batches) | computed from the above across rows |
| `ItSol` | O(batches x rows) | operates on the above, not on `sData` |
| final adjustment | per row | yes, given `gammaStar`/`deltaStar` + per-row stats |

`ReplaceZeroWithMedianOfPositive(varPooled)` needs all rows' `varPooled` first - which is why
standardization cannot start until pass 1 has finished, and why there are separate passes at all.

## 4. Normalization wants the opposite orientation

Parquet is columnar, so reading one sample column at a time is O(rows), not O(rows x samples).

| Method | Needs | Implemented as |
|---|---|---|
| `median` | per-sample median over all rows | column-at-a-time pass -> per-sample offsets |
| `rt_lowess` | per-sample LOWESS vs `mean_rt`, then global median curve | two column passes (the grid span needs the kept-row set first) -> per-sample curves |
| `vsn` | per-sample median of linear | column-at-a-time |
| `none` | - | identity |
| `quantile` | value -> rank within its own column, at apply time | **not streamable** - stays in-memory |

All-NaN rows are dropped downstream but cannot change a median, a VSN scale or a LOWESS fit (every
statistic skips NaN), so the column pass does not need the kept-row set - except for the rt_lowess
grid span, which is taken over kept rows only.

`rt_lowess` on a file with no RT column silently degrades to `median`, because that is what the
in-memory path does (its rt guard falls through to the method switch, whose default is median).

## 5. Parity strategy - done first

⚠️ **The existing golden tests would not have caught a regression here.** `PipelineParityTests`
compares against the Python goldens with `corrected_peptides.parquet` at `absTol 1e-6,
relTol 3e-2` - that 3% exists because C# and Python ComBat legitimately diverge, and it would have
waved through a serious streaming bug. The `mini` fixtures are also fully dense, so they exercise no
missing-value behaviour at all.

`NormalizeCorrectParityTests` (65 cases) runs both implementations in one process over identical
synthetic input and compares every cell of both outputs at **1e-12 relative**, plus the row count and
the report line for line (which pins the control-CV metrics and the auto-revert decision). It covers
every normalization method including quantile, ComBat on/off, reference-anchored on/off, auto-revert
triggering and not, dropped all-NaN rows and not, missing values, multi-row-group inputs, and an
entirely-dropped row group.

Two properties keep it honest:

- Each case asserts **which implementation it expects to run** (`StreamingNormalizeCorrect.CanHandle`).
  Without that, a streaming path that quietly stopped being eligible would leave every case comparing
  the in-memory implementation to itself, and still be green.
- `Comparer_DetectsADifference` perturbs one cell by 1e-9 relative and requires the comparer to fail.

**Summation order.** Preserved deliberately, not by luck: `Batching.RowMeanAndPooledVar` and
`MeanAndSumSq` use the same `NumpyMath.PairwiseSum` reductions over the same values in the same order
as the in-memory code, and `CvMetrics.TryFeatureCv` / `BatchCorrectionEvaluator.Decide` were split out
of the existing implementations so both paths go through one copy of the arithmetic rather than two.
The only intentional departure is the `ItSol` sum of squares (§3), which is why the end-to-end bar is
1e-12 rather than exact.

## 6. Implementation order - as built

1. **Task 0** - `NormalizeAndCorrect` moved out of `PrismPipeline` into `NormalizeCorrectStage` as
   `RunInMemory(request)` behind a `Run(request)` dispatcher, then the parity harness above.
2. **Phase A** - `NormalizationFactors.Compute` reads one sample column at a time and returns
   per-sample factors with an `Apply(sample, value, rt)`. `NormalizationFactorsTests` asserts they
   reproduce `Normalizer` **exactly** (not approximately) for every streamable method.
3. **Phase B** - pass 1 streams row groups, applies the factors, drops all-NaN rows, and accumulates
   the before-CVs plus `grandMean` / `varPooled` / the zero-variance mask.
4. **Phase C** - pass 2 accumulates `gammaHat` / `sumSq`; `StreamingComBat.Estimate` runs the priors
   and `ItSol` against them.
5. **Phase D** - the final pass corrects and appends to both outputs with `StreamingWideWriter`,
   which also removed the full `linearCols` / `log2Cols` transposes.
6. **Wiring** - `NormalizeCorrectStage.Run` picks the path and reports it through `PathReport`
   (deliberately separate from `Report`, which the harness compares).

## 7. Files and landmarks

| File | What |
|---|---|
| `src/SkylinePrism.Core/Pipeline/NormalizeCorrectStage.cs` | the request object, the dispatcher, and `RunInMemory` (the fallback) |
| `src/SkylinePrism.Core/Pipeline/StreamingNormalizeCorrect.cs` | the streamed implementation: eligibility, the passes, the writer |
| `src/SkylinePrism.Core/Normalization/NormalizationFactors.cs` | Phase A - per-sample factors from a column pass |
| `src/SkylinePrism.Core/BatchCorrection/StreamingComBat.cs` | Phase C - EB from sufficient statistics |
| `src/SkylinePrism.Core/IO/ParquetColumnReader.cs` | one column across all row groups, or one row group across many columns |
| `src/SkylinePrism.Core/BatchCorrection/ComBat.cs` | `Run`, `ItSol`, `PostMean`/`PostVar`, `ValidateBatchSizes` (shared) |
| `src/SkylinePrism.Core/Qc/CvMetrics.cs` | `TryFeatureCv` / `MedianOfCvs` - the row-local split both paths use |
| `src/SkylinePrism.Core/Qc/BatchCorrectionEvaluator.cs` | `Decide` - the revert thresholds, shared by both paths |
| `tests/SkylinePrism.Tests/TestSupport/SyntheticCohort.cs` | the shared fixture generator (row groups, all-NaN rows, batch spreads) |
| `tests/SkylinePrism.Tests/Pipeline/NormalizeCorrectParityTests.cs` | the C#-vs-C# harness |

## 8. Gotchas that bit, or nearly did

- **Scale conventions.** Internals are LOG2; `corrected_*.parquet` are LINEAR (`2^x`). The internal
  `peptides_log2_internal.parquet` stays LOG2.
- **Derived grouping columns go on the CORRECTED output only** (`protein_group`, `leading_protein`,
  `leading_name`, `leading_gene_name`). The internal log2 file's readers (`QcReport.LoadMatrix`,
  `ProteinRollup`) treat any undeclared column as a sample and would try to parse `"PG0003"` as an
  abundance.
- **Single-row-group fixtures prove nothing about streaming.** `ParquetWideWriter.Write` emits one
  row group, so the first version of the harness read every fixture in a single group and never
  crossed a boundary - exactly where per-feature state carried across groups goes wrong. Fixtures now
  take a `rowGroupRows`, and the boundary cases are explicit.
- **`n == nAll` aliasing.** The in-memory path deliberately aliases `matrix = matrixAll` when no rows
  are dropped; the harness exercises both branches.
- Run the ship gate, not just `dotnet test`:
  `pwsh -File dotnet/build/package-and-verify.ps1 -Configuration Release`.

## 9. What is still in-memory

All three fall back to `RunInMemory`, which is correct - only memory-hungry - and the chosen path is
logged, so this is never silent.

1. **`quantile` normalization** - by decision (§2.1), not an omission.
2. **CSV/TSV `output.format`** - `WriteDelimited` builds the whole file in a `StringBuilder`. A
   delimited output of a cohort this size is not a real scenario; making the writer incremental is
   easy if it ever is.
3. **Reference-anchored ComBat** (`batch_correction.reference_anchored: true`) - a different
   estimator: NaN-aware throughout, anchored on the reference replicates, with a pooled
   within-batch reference variance. It streams by the same argument (its `ItSol` call takes the
   batch's *reference* columns, so the same two sufficient statistics per (batch, feature) suffice),
   but it has its own standardization and prior code that would need the same treatment. Not started.
   `reference_anchored` defaults to `false`.

Adjacent, not part of this work:

- **`proteins_raw.parquet` is written as a single row group** by `ProteinRollup` (via
  `ParquetWideWriter.Write`), so Stage 4b/4c reads it all at once even on the streaming path. Proteins
  are ~50x fewer than peptides so this is not the wall, but switching `ProteinRollup` to
  `StreamingWideWriter` would make the protein stage bounded too.
- **`PRISM-BUG-combat-nan-propagation.md`** (repo root): standard ComBat returns an all-NaN matrix if
  *any* input cell is missing, in both engines. The streaming path reproduces this exactly, as parity
  requires. If it is fixed, the streamed accumulators must switch to NaN-skipping **and** carry a
  per-(batch, row) count.
