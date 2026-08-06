# BUG: one missing value makes standard ComBat return an all-NaN matrix

**Found:** 2026-08-06, while building the Stage 2b/2c streaming parity harness.
**Affects:** both engines (Python `batch_correction.combat` and the C# port `ComBat.Run`).
**Severity:** high on the default config - `batch_correction.reference_anchored` defaults to `false`,
so standard ComBat is what normally runs.

**Status: FIXED in both engines** (divergence 1 below), together with the prior contamination in
rows 3-4. Verified: one missing cell in a 20x8 matrix now yields exactly one missing cell out,
where it previously yielded 160 of 160. Outstanding: the Python goldens under
`dotnet/tests/fixtures/mini/` still encode the old behaviour and must be regenerated (see
"Regenerating" in `dotnet/tests/fixtures/README.md`), and the sva goldens
(`dotnet/tests/fixtures/sva/generate.R`) have not been generated yet.

**What changed, precisely.** Every reduction now ignores NaN by compacting a feature's observed
values and running the ORDINARY pairwise reductions on them - so a dense cohort is bit-identical to
before - and a quantity the data does not determine is no longer invented:

- a feature with no variance, or absent from a batch entirely, is held out and returned unchanged;
- a (batch, feature) with fewer than 2 observations or no spread gets **no scale correction**
  (`delta* = 1`) but keeps its location correction, and is **excluded from that batch's prior**.
  The old code fed a placeholder `1.0` into `aPrior`/`bPrior`, which are means taken across
  features - so one unestimable feature perturbed the shrinkage of every other feature in the batch.

Both are reported (`ComBat: N feature(s) held out ...`) rather than happening silently.

## What happens

`ComBat.Run` uses NaN-propagating reductions (`np.mean` / `np.var`; C# `Stats.Mean` / `Stats.Var`)
where the reference-anchored variant uses NaN-aware ones (`np.nanmean` / `np.nanvar`). A single
missing cell therefore does not stay local - it spreads to the whole matrix:

```
feature f is NaN in one sample
  -> bHat[i,f] NaN for that batch      (mean over batch members)
  -> grandMean[f] NaN                   (weighted mean over batches)
  -> sData[f,*] NaN for EVERY sample    (standardization divides by a NaN scale)
  -> gammaHat[i,f] NaN for EVERY batch
  -> gammaBar[i] = Mean(gammaHat[i,*]) NaN   <-- the prior is a mean ACROSS FEATURES
  -> gammaStar[i,*] NaN for every feature
  -> every corrected cell is NaN
```

The zero-variance holdout does not catch it: `RowVarPopulation` returns NaN for such a row and
`NaN == 0.0` is false, so the row is treated as active.

Reproduced directly (20 features x 8 samples, 2 batches, one cell set to NaN):
**160 of 160 output cells were NaN.**

## Why it reaches real data

Stage 2's transition rollup writes NaN for any peptide it saw no signal for in a sample
(`TransitionRollup.cs`: `vals[b] = hasValue[b] ? Math.Log2(final[b]) : double.NaN;`), which is the
normal case across plates/documents. Stage 2b/2c drops only rows that are **all** NaN
(`NormalizeCorrectStage`), so partially-missing rows go straight into ComBat. Nothing imputes in
between. `combat_from_long` in Python pivots with `pivot_table` and likewise does not fill.

The committed `mini` fixtures are fully dense (0 nulls, 0 NaNs in `peptides_rollup.parquet`), and
`PipelineParityTests` compares the corrected outputs at 3e-2 relative - so the golden suite cannot
see this.

## Root cause: PRISM's ComBat is not sva's ComBat

The Python docstring says the implementation "is based on the original R sva package and the pyComBat
implementation". It diverges from `sva::ComBat` in nine places, and **this bug is divergence 1**: sva
is NaN-aware throughout and PRISM is not.

Verified against the sva source (`zhangyuqing/sva-devel`, `R/ComBat.R` + `R/helper.R`), 2026-08-06.
The C# is a faithful port of the Python, so every row below applies to both engines unless stated.

| # | Aspect | `sva::ComBat` | PRISM | Consequence |
|---|---|---|---|---|
| 1 | **Missing values** | branches on `any(is.na(dat))`: `Beta.NA` per gene for `B.hat`/`gamma.hat`, `rowVars(na.rm=TRUE)` for `var.pooled`/`delta.hat`, and in `it.sol` `n <- rowSums(!is.na(sdat))` (per **row**) with `na.rm=TRUE` sums | plain `np.linalg.solve` / `np.var` / `np.mean` / `np.sum`; `n = batch column count`, fixed | **this bug** - a NaN escapes its row and destroys the whole matrix |
| 2 | **Zero-variance rows** | variance computed **within each batch**, union across batches, excluded and restored unchanged | variance across **all samples** (`np.var(data, axis=1)` / `Stats.Var(row, ddof:0)`) | PRISM's exclusion set is strictly smaller: a feature constant *within* a batch but differing *between* batches is adjusted by PRISM, refused by sva |
| 3 | `var_pooled == 0` | n/a - such rows are already excluded by 2 | replaced with the median of the positive `var_pooled` | PRISM invention (from pyComBat) papering over 2 |
| 4 | `delta_hat == 0` | n/a - same | replaced with `1.0` | same |
| 5 | `it.sol` convergence | `max(abs(g.new-g.old)/g.old, abs(d.new-d.old)/d.old) > conv`; single combined max; denominator not absolute; **no iteration cap** | separate g and d changes, both must be `< conv`; denominator `abs(old) + 1e-10`; **cap 100 iterations** | PRISM's is the more robust of the two; a deliberate improvement, not a defect |
| 6 | Single-sample batch | tolerated | rejected (`ValueError` / `InvalidOperationException`) | PRISM stricter, both engines agree |
| 7 | Covariates (`mod`) | supported, with confounding/rank checks | Python: supported (`covar_mod`). **C#: not implemented** | cross-engine gap; not exposed in either engine's config |
| 8 | `ref.batch` | supported | Python: supported (`ref_batch`). **C#: not implemented** | same |
| 9 | Non-parametric prior (`par.prior = FALSE`) | `int.eprior`: leave-one-out Monte-Carlo integration over the empirical prior | a fixed shrinkage `shrink*g_hat + (1-shrink)*g_bar`, `shrink = var/(var+1)`. The Python comment says so: *"For non-parametric, we use a simpler approach"* | **not sva's algorithm at all**; currently unreachable (`par_prior` is not exposed in config and the pipeline never passes `false`), so dormant |

Rows 5-6 are deliberate and fine. Row 1 is the bug. Rows 2-4 are a self-consistent alternative to
sva's approach rather than an error, but they are an undocumented divergence. Rows 7-9 are gaps.

`postmean`, `postvar`, `aprior` and `bprior` **do** match sva exactly:
`(t2*n*g + d*gbar)/(t2*n + d)`, `(0.5*sum2 + b)/(0.5*n + a - 1)`, `m^2/v + 2`, `m*(m^2/v + 1)`.

## Measured against sva 3.58.0 (after the fix)

`dotnet/tests/fixtures/sva/generate.R` runs the real `sva::ComBat` on five constructed cases and
commits its output; `SvaGoldenTests` holds PRISM to them. What that measurement showed:

| Case | sva | PRISM |
|---|---|---|
| `dense` (no missing, no degenerate features) | works | matches to **6.4e-7** using sva's own dense `var_pooled`; **2.6e-3** using PRISM's - see below |
| `sparse` (82 missing cells) | works, 82 missing in -> 82 out | matches to **< 1e-9**, exactly |
| `constant_in_batch` | drops the 2 features, returns them unchanged | corrects their location; ~8e-3 elsewhere, because sva's priors are over 38 features and PRISM's over 40 |
| `single_obs` (1 observation in a batch) | **ERRORS**: `while (change > conv): missing value where TRUE/FALSE needed` | handles it (location-only) |
| `absent` (feature absent from a batch) | **ERRORS**: `dgesv: system is exactly singular` | handles it (feature held out) |

Two things came out of this that were not in the original divergence table:

**`var_pooled`'s denominator - and sva is the inconsistent one.** sva computes
`sum(residual^2)/n` (ddof 0) on its dense path but `rowVars(..., na.rm = TRUE)` (ddof 1) on its
missing-value path. **PRISM uses ddof 0 throughout** (`ComBat.VarPooledDdof` / `var_pooled_ddof`).

Consequences, and they are asymmetric on purpose:

| PRISM input | frequency | PRISM vs `sva::ComBat` |
|---|---|---|
| dense | **the normal case** - Skyline integrates imputed peak boundaries for every replicate | matches to floating-point noise |
| contains missing values | rare (mainly documents with different target lists) | **~0.3% apart**, because sva switches denominator and PRISM does not |

Mirroring sva's switch was considered and rejected: it would make one peptide missing from one
document shift every corrected value in the cohort by ~0.3%, a discontinuity worse than the
divergence it removes. Note also that neither denominator is the unbiased pooled estimator, which
would divide by `n - nBatch`; sva's dense form is the MLE.

> [!NOTE]
> This was first written up backwards, claiming PRISM "matches sva on all real proteomics data"
> on the assumption that real data is sparse. It is not - see "Data Density" in CLAUDE.md. The
> choice above was made after that correction, and deliberately favours the dense case.

**The zero-variance test cannot be an exact comparison.** The original code, sva, and the first
version of this fix all asked `variance == 0`. On the mini fixture the same 82 replicates gave a
within-batch variance of **exactly 0.0** through Python and **7.99e-31** through C#, flipping the
feature between "no scale to estimate" and "a scale of 8e-31" - and the two engines then produced
protein abundances **3% apart**. Both engines now treat a spread below `1e-12` of the values' own
magnitude as rounding rather than an estimate (`ComBat.IsSpreadResolvable` /
`_spread_is_resolvable`), which is far above accumulated rounding (~1e-15) and far below any real
measurement. With that, the two engines classify identically and agree.

## Options

1. **Make standard ComBat NaN-aware**, like `ReferenceAnchoredComBat` already is (`NanMean` /
   `NanVar` throughout, plus a `NanMedian` in the prior). Changes numerical output, so it must land
   in BOTH engines together or cross-language parity breaks.
2. **Impute or drop before ComBat** (e.g. require a minimum per-feature completeness), which is a
   quantification decision, not a porting one.
3. **Fail loudly** rather than silently returning NaN - at minimum, ComBat should refuse a matrix
   containing NaN instead of emitting one.

Deliberately NOT fixed as part of the Stage 2b/2c streaming work: that refactor's contract is exact
parity with the current in-memory implementation, and changing ComBat's numerics at the same time
would make a parity failure impossible to attribute.

## Note for the streaming refactor

`dotnet/STREAMING-STAGE-2BC-PLAN.md` §3 streams the EB iteration from per-(batch, row) sufficient
statistics `S1 = sum(sData)` and `S2 = sum(sData^2)`. Plain (non-NaN-skipping) accumulators
reproduce the behaviour above exactly, which is what parity requires today. If option 1 is taken,
the streamed accumulators must switch to NaN-skipping **and** carry a per-(batch, row) count at the
same time.
