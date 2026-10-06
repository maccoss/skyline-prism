# Differential-analysis goldens

Reference values for `SkylinePrism.Core.DifferentialAnalysis`, produced by scipy, statsmodels and
inmoose. `DifferentialGoldenTests` reads them.

> [!NOTE]
> **These are NOT frozen, unlike every other fixture directory here.** The `refanchored/` and
> `sva/` goldens came from engines that no longer exist in this repository, so they can only be
> re-read, never re-made. Every reference behind *these* files is a maintained package on PyPI, so
> the generator is runnable and the fixture is a live claim rather than an archived one. Add a case
> by editing the generator and re-running it; do not hand-edit a `.json`.

```bash
# from the repository root - name the fixture(s) to rewrite
uv run dotnet/tests/fixtures/differential/generate.py toolkit_end_to_end
```

With no name, every fixture is regenerated. That is rarely what you want: it rewrites the others
under whatever library versions resolve today, and once rewrote seven of ten pre-existing goldens
purely from numpy/scipy drift.

The dependency versions are pinned in the script's PEP 723 header, so `uv run` needs nothing
installed. Without `uv`, install those exact versions and run it with `python`.

## What is the reference for what

| Fixture | PRISM type | Reference |
|---|---|---|
| `fdr.json` | `Fdr.BenjaminiHochberg` | `statsmodels` `multipletests(method="fdr_bh")` |
| `polygamma.json` | `EmpiricalBayes.Trigamma` / `.Tetragamma` / `.TrigammaInverse` | `scipy.special.polygamma`; the inverse via `scipy.optimize.brentq` |
| `squeezevar.json` | `EmpiricalBayes.SqueezeVarGlobal` / `.SqueezeVarTrend` | `inmoose.limma.squeezeVar` |
| `spline.json` | `NaturalSplineBasis.Build` | `inmoose.utils.splines.ns` |
| `lmfit.json` | `LinearModel.Fit` | `numpy.linalg.lstsq` + the textbook OLS formulas |
| `moderated_t.json` | `Differential.Run` | the four above, composed the way limma composes them |
| `fisher.json` | `Detection.FisherExact` | `scipy.stats.fisher_exact(alternative="two-sided")` |
| `firth.json` | `Detection.FirthLogit` | `scipy.optimize` on the penalized log-likelihood |
| `detection_lrt.json` | `Detection.DetectionGlm` | the above, twice, + `scipy.stats.chi2.sf(., 1)` |
| `pca.json` | `Pca.Fit` (center-only, complete-case) | `numpy.linalg.svd(full_matrices=False)` |
| `intensity_trend.json` | `VariancePriors.IntensityTrend` | `proteomics_toolkit._fit_intensity_trend_prior` |
| `peptide_count_prior.json` | `VariancePriors.PeptideCountTrend` (DEqMS) | `proteomics_toolkit._fit_count_dependent_prior` |
| `simple_tests.json` | `SimpleTests` Welch / Student / Mann-Whitney | `scipy.stats.ttest_ind`, `scipy.stats.mannwhitneyu(method="asymptotic")` |
| `paired.json` | `SimpleTests` paired t / Wilcoxon, and the paired moderated design | `scipy.stats.ttest_rel`, `scipy.stats.wilcoxon(method="asymptotic")`, lstsq + `squeezeVar` |
| `corrections.json` | `Fdr.Adjust` (BY, Holm, Bonferroni) | `statsmodels` `multipletests` |
| `mcnemar.json` | `Detection.McNemar` | `statsmodels.stats.contingency_tables.mcnemar(exact=True)` |
| `blocked.json` | `DuplicateCorrelation.Estimate`, `LinearModel.FitBlocked`, and `Differential.Run` under `BlockedBySubject` | R `limma::duplicateCorrelation` + `lmFit(block=, correlation=)` + `eBayes`, written by `generate_blocked.R` |
| `toolkit_end_to_end.json` | `Differential.Run` / `.RunTrend` with the intensity-trend prior shaped on control pools and calibrated to the design (unpaired, paired, within-subject and independent trend; the two-arm designs with both an infinite and a finite d0) | `proteomics_toolkit.run_comprehensive_statistical_analysis` with `variance_prior_group_column`, `log_pseudocount=0`; the calibration also against `inmoose.limma.squeezeVar` |

**One reference is a sibling lab tool, deliberately.** `intensity_trend.json` (and with it
`peptide_count_prior.json` and `toolkit_end_to_end.json`) is pinned to
`proteomics-toolkit`, not to a third-party library, because the estimator is not a published formula
with an independent implementation to check against - it IS that tool's
`moderation="intensity_trend"`, and reproducing it is the whole requirement. It stands to PRISM as
`inmoose` does for `squeezeVar`: the definition, not a second opinion. The rule below forbids
consulting the code under test, which this does not. Regenerating it needs the toolkit installed (the
PEP 723 header pulls it from git).

`toolkit_end_to_end.json` goes through the toolkit's top-level call rather than a private function,
because what it pins is the composition: reference and QC pools outside the design give the trend's
shape, and its level and degrees of freedom are fitted to the design's residuals. That calibration
step is limma's `fitFDist` with the trend as a known offset, so the generator also hands the toolkit's
residuals and trend to `inmoose.limma.squeezeVar` and refuses to write a case where the two disagree.
Two conventions are aligned rather than tolerated, and the generator says how. The toolkit is run with `log_pseudocount=0`, because
PRISM adds none. Its inputs keep every prior point more than 1% of the x range from the next, so
PRISM's LOWESS `delta` interpolates nothing, and the generator refuses to write a case where that
fails. On the Verapamil serum cohort the same comparison agreed to ~1e-11. Its cohort is synthetic,
because a clinical one cannot be committed.

The generator imports nothing from PRISM. A golden that was produced by consulting the code under
test cannot catch a mistake the two share, which is the only kind of mistake a golden is for.

## `blocked.json` comes from R, not from `generate.py`

The subject-blocked design - a contrast whose groups are constant within subject (sex, onset site)
when subjects contribute several samples - has no Python reference: inmoose 0.9.1 ports neither
`duplicateCorrelation` nor the blocked `lmFit`. limma is the definition, so the generator is an R
script, as `sva/generate.R` is for ComBat:

```bash
# from the repository root; needs R with BiocManager::install("limma") (statmod comes with it)
Rscript dotnet/tests/fixtures/differential/generate_blocked.R
```

It writes the same string-float encoding as `generate.py` (17 significant digits rather than
Python's shortest repr, which round-trips the same 64 bits), and records the R, limma and statmod
versions that ran in the fixture's `versions` field.

**How tightly the per-feature correlations can be held depends on how they are computed.**
`duplicateCorrelation` asks statmod's `mixedModel2Fit` for each feature's REML variance components
with `maxit = 20` and an absolute tolerance of 1e-6 on the scoring step, so limma's own values are
not the REML optimum. The fixture carries both: `atanh_correlations` (what limma returns) and
`atanh_correlations_converged` (the same fits re-solved to 1e-14 / 1000 iterations). The largest
gap between the two is about 1e-5 on the atanh scale in most cases and 2.4e-3 in
`singletons_and_covariate`, where a near-flat likelihood sits against limma's lower bound. So an
implementation that replays statmod's damped Fisher scoring step for step can be held to limma
closely; one that maximizes the REML likelihood another way agrees only to that gap, and the test
must say which it is asserting.

limma also bounds every per-feature correlation to `[1/(1 - largest block) + 0.01, 0.99]` before
the 15%-trimmed mean on the atanh scale. `correlation_bounds` hits both ends, and
`singletons_and_covariate` hits the floor. The two inputs limma answers with a correlation of 0
and a warning - every block of size 1, and a block already encoded in the design (the paired
design) - are under `degenerate`.

**The same comparison on a real cohort** is `check_blocked_cohort.py`. It hands the matrix PRISM
tested to limma (through `Rscript`) and compares the result with the `differential.csv` PRISM wrote
for the same contrast under `--prior global --correction none`. The docstring has the commands. The
cohort quoted in `docs/differential-analysis.md` is not committed, but the script runs on any PRISM
output directory, so the claim can be re-checked on any data at hand.

**Not the same method as `proteomics-toolkit`'s mixed model.** The toolkit fits a per-feature
`statsmodels` `mixedlm` with a random subject intercept: one correlation per feature, no consensus
across features, no empirical-Bayes moderation, and normal-approximation p-values. It is a
different estimator, so it is neither the reference here nor expected to agree with it.

## Three things that are deliberately not what they first appear

**Floats are JSON strings, not JSON numbers.** Each is Python's shortest round-trip `repr`. Bare
`NaN`/`Infinity` are not valid JSON and `System.Text.Json` rejects them outright; and a float
carried through both ends' number paths is only approximately preserved, which is no basis for an
assertion at 1e-15. Through a string the 64 bits survive intact. Python spells the non-finite ones
`inf`, `-inf` and `nan`, and the C# reader maps those three - the fixture reads the way the tool
that wrote it writes.

**The spline is pinned by its column span, not its entries.** R's `ns` fixes the basis only up to
the orthogonal rotation its constraint QR happens to produce, so two correct implementations can
return different matrices. `hat` is the projector onto the column span, `B (B'B)^-1 B'`, which is
invariant to that rotation and is all the trend fit depends on. Asserting the basis entries would
be asserting an arbitrary choice.

**The Firth reference is an optimizer, not a formula.** No library implements Firth logistic
regression, and pinning it to the sibling Python implementation it was ported from would only prove
the two agree. The reference instead maximizes `l(b) + 0.5*log det(X' W X)` with Nelder-Mead and
Powell - different objective formulation, different algorithm, same fixed point. A direct-search
optimum determines the *log-likelihood* far more tightly than the *coefficients*: on a flat
penalized likelihood (`all_one_class`) the coefficients are genuinely not determined beyond about
1e-4, while the log-likelihood agrees to 1e-9. The test asserts each at what it can actually
support, which is why those two tolerances differ by five orders of magnitude.

## A divergence from inmoose, recorded rather than papered over

`squeezevar.json` has no case with a spline df of 2, and `spline.json` has no `df=2` case, because
**inmoose 0.9.1's `ns()` raises on a zero-interior-knot basis**:

```
ValueError: all the input arrays must have same number of dimensions,
but the array at index 0 has 1 dimension(s) and the array at index 1 has 0 dimension(s)
```

R's `splines::ns` builds that basis without complaint, and so does PRISM. The case is reachable:
`splinedf = 1 + (n >= 3) + (n >= 6) + (n >= 30)` is exactly 2 whenever 3 to 5 features reach the
trend prior. There is therefore no inmoose value to pin it against, and
`DifferentialGoldenTests.SqueezeVarTrend_SplineDfTwo_IsSupportedWhereInmooseRaises` asserts the
part that needs no reference: PRISM returns a finite per-feature prior rather than throwing, and
does not silently fall back to the global prior - which would be a different estimator producing
different numbers under the same setting.

## What these goldens do not cover

- **`SignificanceScan`, `MarkerPanel`, `Enrichment`, `DetectionMatrix`, `DifferentialDataset`.**
  These are PRISM's own procedures, not ports of a published method, so there is nothing external
  to hold them to. They are covered by behavioral tests in the same directory.
- **`Fdr.BenjaminiHochberg`'s NaN handling.** statsmodels returns all-NaN if any input is NaN and
  counts NaN toward `m`; PRISM passes NaN through and excludes it. That divergence is deliberate
  and is pinned by `FdrTests`, not here - every case in `fdr.json` is NaN-free, where the two agree
  exactly.
- **The lab's `proteomics-toolkit`.** Its `moderated_linear_model` is a third implementation of the
  same idea and is *not* pinned to these. See `docs/differential-analysis.md` for where the two
  agree and where they do not.
