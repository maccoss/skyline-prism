#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy==2.5.3",
#     "scipy==1.18.1",
#     "statsmodels==0.15.0",
#     "inmoose==0.9.1",
#     "proteomics-toolkit @ git+https://github.com/uw-maccosslab/proteomics-toolkit@v26.8.0",
# ]
# ///
"""Generate the differential-analysis golden fixtures from the reference implementations.

Unlike the `refanchored/` and `sva/` fixtures, these are NOT frozen: every reference here is a
maintained third-party library on PyPI, so the goldens can be regenerated at any time and the
dependency versions are pinned in the PEP 723 header above.

Which library is the reference for which quantity:

| PRISM type                        | reference                                              |
|-----------------------------------|--------------------------------------------------------|
| `Fdr.BenjaminiHochberg`           | `statsmodels.stats.multitest.multipletests("fdr_bh")`   |
| `EmpiricalBayes.Trigamma`         | `scipy.special.polygamma(1, .)`                         |
| `EmpiricalBayes.Tetragamma`       | `scipy.special.polygamma(2, .)`                         |
| `EmpiricalBayes.TrigammaInverse`  | `scipy.special.polygamma(1, .)` inverted (round trip)   |
| `EmpiricalBayes.SqueezeVar*`      | `inmoose.limma.squeezeVar`                              |
| `NaturalSplineBasis.Build`        | `inmoose.utils.splines.ns`                              |
| `LinearModel.Fit`                 | `numpy.linalg.lstsq` + the textbook OLS formulas        |
| `Differential.Run`                | the four above, composed the way limma composes them    |
| `Detection.FisherExact`           | `scipy.stats.fisher_exact(alternative="two-sided")`     |
| `Detection.FirthLogit`            | `scipy.optimize` on the penalized log-likelihood        |
| `Pca.Fit` (center-only, complete-case) | `numpy.linalg.svd(full_matrices=False)`            |
| `Detection.DetectionGlm`          | penalized LRT: `scipy.optimize` twice + `scipy.stats.chi2` |
| `VariancePriors.IntensityTrend`   | `proteomics_toolkit._fit_intensity_trend_prior`          |
| `VariancePriors.PeptideCountTrend` | `proteomics_toolkit._fit_count_dependent_prior`         |
| `Differential.Run`/`RunTrend`, prior from controls | `proteomics_toolkit.run_comprehensive_statistical_analysis` |
| `SimpleTests` (Welch/Student)     | `scipy.stats.ttest_ind(equal_var=...)`                  |
| `SimpleTests` (Mann-Whitney)      | `scipy.stats.mannwhitneyu(method='asymptotic')`         |
| `SimpleTests` (paired t)          | `scipy.stats.ttest_rel`                                 |
| `SimpleTests` (Wilcoxon)          | `scipy.stats.wilcoxon(method='asymptotic')`             |
| paired moderated design           | lstsq on `[1, grp, subject dummies]` + squeezeVar        |
| `Fdr.{BenjaminiYekutieli,Bonferroni,Holm}` | `statsmodels multipletests`                    |
| `Detection.McNemar`               | `statsmodels.stats.contingency_tables.mcnemar(exact=True)` |

Nothing here imports PRISM. The point of a golden is that it was produced without reference to the
code under test, so a shared mistake cannot cancel out.

`VariancePriors.IntensityTrend` is the one entry whose reference is another MacCoss Lab tool
rather than a third-party library, and that is deliberate rather than a lapse. The estimator is
not a published formula with an independent implementation to check against - it is specifically
`proteomics-toolkit`'s `moderation="intensity_trend"`, and reproducing THAT is the whole
requirement. It stands in the same relation to PRISM as `inmoose` does for `squeezeVar`: the
definition, not a second opinion. What the rule above forbids is consulting the C# code under
test, and this does not.

`FirthLogit` is the one entry with no library implementation to call. Rather than pin it to the
sibling Python implementation it was ported from - which would only prove the two agree - the
reference maximizes the Jeffreys-penalized log-likelihood directly with a derivative-free
optimizer. Different objective formulation, different algorithm, same fixed point.

Run from the repository root:

    uv run dotnet/tests/fixtures/differential/generate.py toolkit_end_to_end

naming the fixture(s) to rewrite - with no name, every fixture is regenerated, which is rarely what
you want (see ``main``). Or, without uv, in an environment holding the pinned versions above:

    python dotnet/tests/fixtures/differential/generate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.special
import scipy.stats
from scipy import stats
from inmoose.limma import squeezeVar
from inmoose.utils.splines import ns
from statsmodels.stats.multitest import multipletests

OUT = Path("dotnet/tests/fixtures/differential")


# --------------------------------------------------------------------------------------------
# Encoding
#
# Every floating-point value is written as a STRING holding Python's shortest round-trip repr.
# Two reasons, both of which bit a golden elsewhere in this repo:
#   * bare NaN / Infinity are not valid JSON, and System.Text.Json rejects them outright;
#   * a float written through the JSON number path is at the mercy of both encoders' shortest-repr
#     rules, and a golden that is only nearly bit-exact cannot be asserted at 1e-15.
# Strings sidestep both: Python writes repr(), .NET reads double.Parse(InvariantCulture), and the
# value that comes back is the same 64 bits that went in.
# --------------------------------------------------------------------------------------------


def num(x) -> str:
    """One float as its exact round-trip string."""
    return repr(float(x))


def vec(a) -> list[str]:
    return [num(v) for v in np.asarray(a, dtype=float).ravel()]


def mat(a) -> list[list[str]]:
    a = np.asarray(a, dtype=float)
    return [[num(v) for v in row] for row in a]


def write(name: str, payload: dict) -> None:
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


class Rng:
    """A tiny reproducible LCG.

    numpy's Generator is reproducible too, but its stream is a guarantee about numpy, not about
    this fixture; a bare LCG keeps the inputs readable and pins them to this file.
    """

    def __init__(self, seed: int) -> None:
        self._s = seed | 1

    def next_double(self) -> float:
        self._s = (self._s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return ((self._s >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    def normal(self) -> float:
        # Box-Muller; one draw per call is wasteful and entirely beside the point here.
        u1 = max(self.next_double(), 1e-12)
        u2 = self.next_double()
        return float(np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2))

    def matrix(self, rows: int, cols: int, loc: float = 10.0, scale: float = 1.0) -> np.ndarray:
        return np.array(
            [[loc + scale * self.normal() for _ in range(cols)] for _ in range(rows)],
            dtype=float,
        )


# --------------------------------------------------------------------------------------------
# Benjamini-Hochberg
# --------------------------------------------------------------------------------------------


def gen_fdr() -> None:
    rng = Rng(11)
    cases = []

    def add(name: str, p, note: str) -> None:
        p = np.asarray(p, dtype=float)
        expected = multipletests(p, method="fdr_bh")[1]
        cases.append({"name": name, "note": note, "p": vec(p), "expected": vec(expected)})

    add("uniform_steps", [0.01, 0.02, 0.03, 0.04, 0.05], "every step-up value equal; all take the max")
    add("one_strong_hit", [1e-8, 0.2, 0.4, 0.6, 0.8, 0.9], "a single dominant hit")
    add("ties", [0.04, 0.04, 0.04, 0.2, 0.2, 0.9], "tied p-values must take an identical q")
    add("all_large", [0.6, 0.7, 0.8, 0.9, 0.95], "everything clips to 1")
    add("single", [0.031], "m = 1: the adjusted value is the raw one")
    add("descending", [0.9, 0.5, 0.2, 0.05, 0.001], "input order is not sorted order")
    add("many", [rng.next_double() ** 3 for _ in range(200)], "200 values, heavily skewed to small p")
    add("with_exact_zero", [0.0, 0.0, 0.3, 0.7], "exact zeros stay zero")

    write(
        "fdr.json",
        {
            "reference": "statsmodels.stats.multitest.multipletests(method='fdr_bh')",
            "note": (
                "PRISM's BenjaminiHochberg additionally passes NaN through and excludes it from m, "
                "where statsmodels returns all-NaN. That behavior is PRISM's own and is pinned by "
                "FdrTests, not here - every case below is NaN-free, where the two agree exactly."
            ),
            "cases": cases,
        },
    )


# --------------------------------------------------------------------------------------------
# Polygamma
# --------------------------------------------------------------------------------------------


def gen_polygamma() -> None:
    xs = [
        0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0,
        # Either side of the asymptotic-series threshold (30), where the recurrence hands over.
        29.0, 29.999, 30.0, 30.001, 31.0, 50.0, 100.0, 500.0, 5000.0, 1e6,
        # Half-integers are what fitFDist actually evaluates: trigamma(df/2).
        2.5, 3.5, 25.5, 60.5,
    ]
    trigamma = [scipy.special.polygamma(1, x) for x in xs]
    tetragamma = [scipy.special.polygamma(2, x) for x in xs]

    # trigammaInverse is pinned by round-trip: solve trigamma(y) = x, then check trigamma(y) == x.
    # Inverting through scipy rather than through limma's own Newton keeps the reference independent.
    inv_inputs = [1e-8, 1e-6, 1e-4, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 1e3, 1e6, 1e7, 1e8, 1e10]
    inv_expected = []
    for x in inv_inputs:
        if x > 1e7:
            inv_expected.append(1.0 / np.sqrt(x))
        elif x < 1e-6:
            inv_expected.append(1.0 / x)
        else:
            root = scipy.optimize.brentq(
                lambda y: float(scipy.special.polygamma(1, y)) - x,
                1e-12, 1e12, xtol=1e-300, rtol=8.9e-16, maxiter=500,
            )
            inv_expected.append(root)

    write(
        "polygamma.json",
        {
            "reference": "scipy.special.polygamma(1|2, .); trigammaInverse via scipy.optimize.brentq",
            "note": (
                "The two smallest/largest trigammaInverse inputs take limma's closed-form branches "
                "(1/x and 1/sqrt(x)) rather than Newton, and are reproduced here the same way."
            ),
            "x": vec(xs),
            "trigamma": vec(trigamma),
            "tetragamma": vec(tetragamma),
            "trigamma_inverse_x": vec(inv_inputs),
            "trigamma_inverse_expected": vec(inv_expected),
        },
    )


# --------------------------------------------------------------------------------------------
# squeezeVar (global and intensity-trend priors)
# --------------------------------------------------------------------------------------------


def gen_squeezevar() -> None:
    rng = Rng(23)
    cases = []

    def add(name: str, variances, df: float, covariate=None, note: str = "") -> None:
        v = np.asarray(variances, dtype=float)
        cov = None if covariate is None else np.asarray(covariate, dtype=float)
        res = squeezeVar(v, df, covariate=cov)
        cases.append(
            {
                "name": name,
                "note": note,
                "variances": vec(v),
                "df_residual": num(df),
                "covariate": None if cov is None else vec(cov),
                "df_prior": num(np.atleast_1d(res["df_prior"])[0]),
                "var_prior": vec(np.atleast_1d(res["var_prior"])),
                "var_post": vec(np.atleast_1d(res["var_post"])),
            }
        )

    # --- global prior -------------------------------------------------------------------
    add(
        "global_finite_prior",
        [0.05, 0.5, 0.1, 1.2, 0.3, 0.8, 0.02, 0.15, 2.0, 0.25,
         0.6, 0.09, 0.35, 1.5, 0.04, 0.7, 0.12, 0.9, 0.2, 0.45],
        7.0,
        note="the ordinary case: a finite, small prior df",
    )
    add(
        "global_large_prior_df",
        [0.1, 0.2, 0.4, 0.5],
        5.0,
        note="tightly clustered variances drive df_prior to ~51, the regime the polygamma "
             "asymptotic threshold has to hold 1e-9 on",
    )
    add(
        "global_uniform_variance",
        [0.25] * 12,
        6.0,
        note="identical variances: evar <= 0, so df_prior is infinite and every posterior is the "
             "pooled mean",
    )
    add(
        "global_wide_spread",
        [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0],
        4.0,
        note="seven orders of magnitude; exercises the log-F fit at its limits",
    )
    add(
        "global_many_features",
        [abs(rng.normal()) * 0.4 + 0.02 for _ in range(500)],
        9.0,
        note="500 features, the realistic size",
    )

    # --- intensity-trend prior ----------------------------------------------------------
    n = 60
    amean = np.array([6.0 + 8.0 * i / (n - 1) for i in range(n)])
    # A genuine mean-variance trend: variance falls as intensity rises, which is the whole reason
    # limma-trend exists.
    trend_var = np.array(
        [float(np.exp(-0.35 * (a - 6.0)) * (0.8 + 0.4 * rng.next_double())) for a in amean]
    )
    add("trend_decreasing", trend_var, 8.0, covariate=amean,
        note="variance decays with mean intensity - the limma-trend motivating case")

    n2 = 40
    amean2 = np.array([4.0 + 0.25 * i for i in range(n2)])
    flat_var = np.array([0.3 + 0.05 * rng.normal() for _ in range(n2)])
    flat_var = np.abs(flat_var)
    add("trend_flat", flat_var, 6.0, covariate=amean2,
        note="no real trend: the spline should flatten and land near the global prior")

    # splinedf = 1 + (n>=3) + (n>=6) + (n>=30), capped at the number of distinct covariate values.
    add("trend_three_distinct", [0.2, 0.5, 0.1, 0.8, 0.3, 0.4], 5.0,
        covariate=[1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        note="only 3 distinct covariate values: splinedf caps at 3")
    add("trend_six_even", [0.2, 0.5, 0.1, 0.8, 0.3, 0.4], 5.0,
        covariate=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        note="n=6 is exactly where the (nok >= 6) term turns splinedf from 2 to 3")

    # NOT covered here, deliberately: splinedf == 2, which happens for 3 <= n <= 5 features.
    # inmoose 0.9.1's ns() raises on a zero-interior-knot basis (see README), so there is no
    # reference value to pin against. R's splines::ns handles it, and so does PRISM; that case is
    # pinned as PRISM-only behavior in EmpiricalBayesTests instead.

    write(
        "squeezevar.json",
        {
            "reference": "inmoose.limma.squeezeVar (inmoose 0.9.1, a port of limma 3.55.1)",
            "note": (
                "var_prior is length 1 for the global prior and length n for the trend prior, "
                "exactly as inmoose returns it."
            ),
            "cases": cases,
        },
    )


# --------------------------------------------------------------------------------------------
# Natural spline basis
# --------------------------------------------------------------------------------------------


def gen_spline() -> None:
    """Pin the spline by its COLUMN SPAN, not its entries.

    R's `ns` (and inmoose's port) fixes the basis only up to the orthogonal rotation the QR of the
    natural-boundary constraint happens to produce, so two correct implementations can return
    different matrices. What the trend fit depends on - and all it depends on - is the span, and
    the orthogonal projector onto the span, H = B (B'B)^-1 B', is invariant to that rotation.
    Comparing H is therefore both the strictest test available and the only one that is not
    asserting an arbitrary choice.
    """
    rng = Rng(37)
    cases = []

    def add(name: str, x, df: int, note: str) -> None:
        x = np.asarray(x, dtype=float)
        basis = np.asarray(ns(x, df=df, include_intercept=True).basis, dtype=float)
        q, _ = np.linalg.qr(basis)
        hat = q @ q.T
        cases.append(
            {
                "name": name,
                "note": note,
                "x": vec(x),
                "df": df,
                "n_basis_columns": int(basis.shape[1]),
                "hat": mat(hat),
            }
        )

    add("even_grid_df4", [4.0 + 0.5 * i for i in range(18)], 4, "evenly spaced, the default df")
    add("even_grid_df3", [4.0 + 0.5 * i for i in range(18)], 3, "one interior knot fewer")
    add("skewed_df4", [float(10.0 * rng.next_double() ** 3) for _ in range(20)], 4,
        "clustered near the lower boundary, so the interior knots are not evenly spaced")
    add("with_duplicates_df3", [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0], 3,
        "repeated covariate values; percentile knots can coincide with data points")
    add("five_points_df3", [1.0, 2.0, 3.0, 4.0, 5.0], 3, "the smallest basis with an interior knot")

    # df=2 (zero interior knots) is absent because inmoose 0.9.1's ns() raises on it. See README.

    write(
        "spline.json",
        {
            "reference": "inmoose.utils.splines.ns(x, df, include_intercept=True), inmoose 0.9.1",
            "note": (
                "`hat` is the orthogonal projector onto the basis column span, B (B'B)^-1 B'. The "
                "basis itself is defined only up to an orthogonal rotation, so its entries are NOT "
                "a parity target; the span is."
            ),
            "cases": cases,
        },
    )


# --------------------------------------------------------------------------------------------
# lmFit
# --------------------------------------------------------------------------------------------


def gen_lmfit() -> None:
    rng = Rng(51)
    cases = []

    def add(name: str, expr, design, note: str) -> None:
        expr = np.asarray(expr, dtype=float)      # features x samples
        design = np.asarray(design, dtype=float)  # samples x coef
        n, p = design.shape
        beta, _, _, _ = np.linalg.lstsq(design, expr.T, rcond=None)
        beta = beta.T
        resid = expr - beta @ design.T
        df_res = float(n - p)
        sigma = np.sqrt((resid**2).sum(axis=1) / df_res)
        stdev_unscaled = np.sqrt(np.diag(np.linalg.inv(design.T @ design)))
        cases.append(
            {
                "name": name,
                "note": note,
                "expr": mat(expr),
                "design": mat(design),
                "df_residual": num(df_res),
                "coefficients": mat(beta),
                "sigma": vec(sigma),
                "amean": vec(expr.mean(axis=1)),
                "stdev_unscaled": vec(stdev_unscaled),
            }
        )

    # Two groups of 4, design [intercept, groupB].
    design2 = np.column_stack([np.ones(8), np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)])
    add("two_groups", rng.matrix(15, 8), design2, "the plain two-group contrast")

    # Two groups plus a centered numeric covariate and a dummy-coded categorical one.
    age = np.array([61.0, 55.0, 70.0, 48.0, 66.0, 59.0, 73.0, 52.0])
    sex = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    design_cov = np.column_stack([design2, age - age.mean(), sex])
    add("with_covariates", rng.matrix(12, 8), design_cov,
        "intercept + group + centered numeric + dummy categorical")

    # An ill-conditioned design: the covariate nearly tracks the group. This is where forming the
    # normal equations instead of a QR would visibly lose digits.
    near = np.array([0.0, 0.0, 0.0, 0.001, 1.0, 1.0, 1.0, 0.999])
    design_ill = np.column_stack([design2, near])
    add("ill_conditioned", rng.matrix(10, 8), design_ill,
        "covariate almost collinear with group; QR vs normal equations diverge here")

    # Unbalanced groups, larger n.
    grp = np.array([0.0] * 5 + [1.0] * 12)
    design_unbal = np.column_stack([np.ones(17), grp])
    add("unbalanced", rng.matrix(30, 17), design_unbal, "5 vs 12")

    write(
        "lmfit.json",
        {
            "reference": "numpy.linalg.lstsq plus the textbook OLS formulas (limma lmFit)",
            "note": (
                "`expr` is features x samples (PRISM's orientation) and `design` is samples x coef. "
                "stdev_unscaled is sqrt(diag((X'X)^-1)) and depends only on the design."
            ),
            "cases": cases,
        },
    )


# --------------------------------------------------------------------------------------------
# End-to-end moderated t
# --------------------------------------------------------------------------------------------


def moderated_t(expr: np.ndarray, design: np.ndarray, coef_idx: int, trend: bool):
    """limma's lmFit -> squeezeVar -> moderated t -> BH, composed from the reference libraries.

    This is the composition the limma vignette describes, written out rather than called through
    a wrapper so that every step is visibly one of the pinned references.
    """
    n, p = design.shape
    beta, _, _, _ = np.linalg.lstsq(design, expr.T, rcond=None)
    beta = beta.T
    resid = expr - beta @ design.T
    df_res = float(n - p)
    sigma2 = (resid**2).sum(axis=1) / df_res
    stdev_unscaled = np.sqrt(np.diag(np.linalg.inv(design.T @ design)))
    amean = expr.mean(axis=1)

    sv = squeezeVar(sigma2, df_res, covariate=amean if trend else None)
    df_prior = float(np.atleast_1d(sv["df_prior"])[0])
    var_post = np.atleast_1d(sv["var_post"])

    df_total = df_res + df_prior
    t = beta[:, coef_idx] / (stdev_unscaled[coef_idx] * np.sqrt(var_post))
    if np.isinf(df_total):
        pval = 2.0 * scipy.stats.norm.cdf(-np.abs(t))
    else:
        pval = 2.0 * scipy.stats.t.cdf(-np.abs(t), df=df_total)
    adj = multipletests(pval, method="fdr_bh")[1]
    return {
        "logfc": beta[:, coef_idx],
        "amean": amean,
        "t": t,
        "p": pval,
        "adj_p": adj,
        "df_residual": df_res,
        "df_prior": df_prior,
    }


def gen_moderated_t() -> None:
    rng = Rng(67)
    cases = []

    def add(name: str, expr, n_a: int, design, trend: bool, note: str,
            covariates=None) -> None:
        expr = np.asarray(expr, dtype=float)
        design = np.asarray(design, dtype=float)
        res = moderated_t(expr, design, coef_idx=1, trend=trend)
        n_b = expr.shape[1] - n_a
        cases.append(
            {
                "name": name,
                "note": note,
                "expr": mat(expr),
                "n_a": n_a,
                "n_b": n_b,
                "design": mat(design),
                # The covariates as PRISM receives them - RAW, uncentered, in [A..., B...] order.
                # PRISM centers numerics and dummy-codes categoricals itself, so handing it the
                # design columns directly would test the arithmetic twice and the design-building
                # code not at all.
                "covariates": covariates or [],
                "trend": trend,
                "df_residual": num(res["df_residual"]),
                "df_prior": num(res["df_prior"]),
                "logfc": vec(res["logfc"]),
                "amean": vec(res["amean"]),
                "t": vec(res["t"]),
                "p": vec(res["p"]),
                "adj_p": vec(res["adj_p"]),
            }
        )

    # --- no covariates, heterogeneous variance (finite prior df) ------------------------
    n_a, n_b = 6, 7
    grp = np.array([0.0] * n_a + [1.0] * n_b)
    design = np.column_stack([np.ones(n_a + n_b), grp])
    # Per-feature noise scales must genuinely differ, or every residual variance comes out the same,
    # evar <= 0, and the prior df is infinite - which is a branch worth testing but not the typical
    # one. Spreading the scale over an order of magnitude gives the finite-prior case.
    expr = np.array(
        [[18.0 + (0.15 + 1.5 * (i / 39.0) ** 2) * rng.normal() for _ in range(n_a + n_b)]
         for i in range(40)]
    )
    # Put a real effect into the first five features so the volcano is not all noise.
    expr[:5, n_a:] += 1.7
    add("no_covariates", expr, n_a, design, False,
        "6 vs 7, heterogeneous per-feature noise, a real effect in the first 5 features")

    # --- uniform variance drives the prior df to infinity --------------------------------
    base = np.array([[10.0, 10.5, 11.0, 12.0, 12.5, 13.0]] * 8)
    add("infinite_prior_df", base, 3, np.column_stack([np.ones(6), [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]),
        False, "every feature the same row: evar <= 0, df_prior is infinite, p comes from the normal")

    # --- intensity trend ------------------------------------------------------------------
    n_a2, n_b2 = 8, 8
    grp2 = np.array([0.0] * n_a2 + [1.0] * n_b2)
    design2 = np.column_stack([np.ones(16), grp2])
    # Abundance spans a wide dynamic range and the noise shrinks with it: limma-trend's premise.
    rows = []
    for i in range(50):
        level = 8.0 + 12.0 * i / 49.0
        noise = float(np.exp(-0.12 * (level - 8.0)))
        rows.append([level + noise * rng.normal() for _ in range(16)])
    expr_tr = np.array(rows)
    expr_tr[:6, n_a2:] += 0.9
    add("trend_prior", expr_tr, n_a2, design2, True,
        "wide dynamic range with intensity-dependent noise; the trend prior is the point")
    add("trend_prior_off", expr_tr, n_a2, design2, False,
        "the same matrix with the global prior, so the two priors can be told apart")

    # --- covariate adjustment ---------------------------------------------------------------
    n_a3, n_b3 = 7, 7
    grp3 = np.array([0.0] * n_a3 + [1.0] * n_b3)
    age3 = np.array([61.0, 55.0, 70.0, 48.0, 66.0, 59.0, 73.0,
                     52.0, 64.0, 58.0, 71.0, 49.0, 67.0, 60.0])
    # Sorted levels, first dropped: F is the reference level, so the dummy marks M.
    sex3 = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0,
                     0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    design3 = np.column_stack([np.ones(14), grp3, age3 - age3.mean(), sex3])
    expr3 = rng.matrix(35, 14, loc=15.0, scale=0.8)
    # Make age genuinely predictive, so dropping the covariate would change the answer.
    expr3 += 0.03 * (age3 - age3.mean())[None, :]
    expr3[:4, n_a3:] += 1.2
    add("with_covariates", expr3, n_a3, design3, False,
        "centered age + sex dummy; age has a real effect so the adjustment matters",
        covariates=[
            {"name": "age", "kind": "numeric", "values": vec(age3)},
            # Sorted levels are [F, M] and the first is dropped, so the dummy marks M - which is
            # exactly the 1s in sex3.
            {"name": "sex", "kind": "categorical",
             "values": ["M" if v == 1.0 else "F" for v in sex3]},
        ])

    # --- a single feature, the smallest possible run -------------------------------------
    add("single_feature", np.array([[10.0, 10.2, 9.8, 12.0, 12.4, 11.6]]), 3,
        np.column_stack([np.ones(6), [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]), False,
        "n_features = 1: fitFDist takes its nok == 1 branch (df_prior 0)")

    write(
        "moderated_t.json",
        {
            "reference": (
                "numpy.linalg.lstsq + inmoose.limma.squeezeVar + scipy.stats.t.cdf + "
                "statsmodels multipletests('fdr_bh'), composed as limma composes them"
            ),
            "note": (
                "`design` is given explicitly and its column 1 is the group indicator, so the C# "
                "side must reproduce it from (n_a, n_b) and the covariates: intercept, groupB, "
                "centered numeric covariates, then dummy-coded categoricals with the first sorted "
                "level dropped. Samples are ordered [A..., B...]."
            ),
            "cases": cases,
        },
    )


# --------------------------------------------------------------------------------------------
# Fisher exact
# --------------------------------------------------------------------------------------------


def gen_fisher() -> None:
    tables = [
        (8, 2, 1, 9), (0, 10, 10, 0), (5, 5, 5, 5), (1, 0, 0, 1),
        (0, 0, 4, 6), (12, 3, 4, 11), (30, 10, 12, 28), (1, 19, 18, 2),
        (7, 0, 0, 7), (2, 8, 8, 2), (100, 50, 60, 90), (3, 1, 1, 3),
        (0, 0, 0, 0), (25, 0, 0, 25), (14, 6, 9, 11),
    ]
    cases = []
    for a, b, c, d in tables:
        _, p = scipy.stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")
        cases.append({"a": a, "b": b, "c": c, "d": d, "p": num(p)})

    write(
        "fisher.json",
        {
            "reference": "scipy.stats.fisher_exact(alternative='two-sided')",
            "note": "Table is [[a, b], [c, d]]: detected/not-detected by group.",
            "cases": cases,
        },
    )


# --------------------------------------------------------------------------------------------
# Firth-penalized logistic regression
# --------------------------------------------------------------------------------------------


def firth_reference(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Maximize l(b) + 0.5 log det(X' W X) with a derivative-free optimizer.

    Deliberately not Newton-with-hat-matrix-adjustment, which is what the implementation under
    test does: an independent algorithm reaching the same stationary point is evidence; the same
    algorithm reaching it twice is not.
    """

    def penalized(b: np.ndarray) -> float:
        eta = x @ b
        # log(1 + exp(eta)) computed stably; eta can be large under separation, which is the case
        # Firth exists for.
        log1pexp = np.logaddexp(0.0, eta)
        ll = float(np.sum(y * eta - log1pexp))
        p = 1.0 / (1.0 + np.exp(-eta))
        w = p * (1.0 - p)
        info = x.T @ (w[:, None] * x)
        sign, logdet = np.linalg.slogdet(info)
        if sign <= 0:
            return -np.inf
        return ll + 0.5 * logdet

    def objective(b: np.ndarray) -> float:
        v = penalized(b)
        return np.inf if not np.isfinite(v) else -v

    best = np.zeros(x.shape[1])
    # Nelder-Mead first (no derivatives, no curvature assumptions), then Powell to polish. Both are
    # direct-search methods, so neither reuses the Fisher information the estimator under test
    # builds its step from.
    for method, opts in (
        ("Nelder-Mead", {"xatol": 1e-14, "fatol": 1e-14, "maxiter": 100000, "maxfev": 100000}),
        ("Powell", {"xtol": 1e-14, "ftol": 1e-14, "maxiter": 100000, "maxfev": 100000}),
        ("Nelder-Mead", {"xatol": 1e-15, "fatol": 1e-15, "maxiter": 100000, "maxfev": 100000}),
    ):
        res = scipy.optimize.minimize(objective, best, method=method, options=opts)
        best = res.x
    return best, penalized(best)


def gen_firth() -> None:
    cases = []

    def add(name: str, x, y, note: str) -> None:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        beta, ll = firth_reference(x, y)
        cases.append(
            {"name": name, "note": note, "x": mat(x), "y": vec(y),
             "beta": vec(beta), "penalized_loglik": num(ll)}
        )

    add(
        "separable",
        [[1.0, -2.0], [1.0, -1.0], [1.0, -0.5], [1.0, 0.5], [1.0, 1.0], [1.0, 2.0]],
        [0, 0, 0, 1, 1, 1],
        "perfect separation: the unpenalized MLE diverges, Firth stays finite",
    )
    add(
        "mixed",
        [[1.0, -2.0], [1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
         [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]],
        [0, 1, 0, 0, 1, 1, 0, 1],
        "overlapping classes: an ordinary well-posed fit",
    )
    add(
        "two_predictors",
        [[1.0, 0.0, 1.2], [1.0, 1.0, -0.4], [1.0, 0.0, 0.7], [1.0, 1.0, 2.1],
         [1.0, 0.0, -1.1], [1.0, 1.0, 0.3], [1.0, 0.0, 1.9], [1.0, 1.0, -0.8],
         [1.0, 0.0, 0.1], [1.0, 1.0, 1.4]],
        [0, 1, 0, 1, 0, 0, 1, 0, 1, 1],
        "a group indicator plus a continuous covariate, the adjusted-detection design",
    )
    add(
        "quasi_separation",
        [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0], [1.0, 5.0]],
        [0, 0, 0, 1, 1, 0],
        "one point on the wrong side of an otherwise clean split",
    )
    add(
        "all_one_class",
        [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0]],
        [1, 1, 1, 1],
        "no contrast at all: the intercept runs away and only the penalty holds it",
    )

    write(
        "firth.json",
        {
            "reference": (
                "scipy.optimize (Nelder-Mead then Powell) maximizing "
                "l(b) + 0.5*log det(X' diag(p(1-p)) X)"
            ),
            "note": (
                "A direct-search optimum is accurate in the OBJECTIVE to roughly the tolerance "
                "asked for, but the coefficients themselves are only as well determined as the "
                "curvature allows - a flat penalized likelihood (all_one_class) pins beta far more "
                "loosely than it pins the log-likelihood. Assert the log-likelihood tightly and the "
                "coefficients loosely; that is what the fixture is able to support."
            ),
            "cases": cases,
        },
    )


# --------------------------------------------------------------------------------------------
# Sample-space PCA
# --------------------------------------------------------------------------------------------


def gen_pca() -> None:
    rng = Rng(83)
    cases = []

    def add(name: str, expr, note: str, n_components: int = 6) -> None:
        expr = np.asarray(expr, dtype=float)  # features x samples
        n_samples = expr.shape[1]
        # Complete-case, then center each feature across the selected samples (no scaling).
        complete = expr[~np.isnan(expr).any(axis=1), :]
        centered = complete - complete.mean(axis=1, keepdims=True)
        # numpy's SVD of the sample x feature matrix; scores are U * S.
        u, s, _ = np.linalg.svd(centered.T, full_matrices=False)
        k = min(n_components, n_samples, complete.shape[0])
        scores = u[:, :k] * s[:k]
        total = float((s**2).sum())
        ratio = (s[:k] ** 2) / total
        cases.append(
            {
                "name": name,
                "note": note,
                "expr": mat(expr),
                "n_components": n_components,
                "n_features_used": int(complete.shape[0]),
                # Signs are arbitrary per component, so the fixture carries |scores| and the test
                # compares magnitudes. A sign flip is not a defect; a magnitude change is.
                "abs_scores": mat(np.abs(scores)),
                "variance_ratio": vec(ratio),
            }
        )

    add("dense", rng.matrix(60, 9, loc=14.0, scale=1.5), "no missing values")

    # A matrix with structure: two clusters of samples.
    struct = rng.matrix(80, 10, loc=12.0, scale=0.4)
    struct[:, 5:] += 2.0
    add("two_clusters", struct, "PC1 should separate the two groups cleanly")

    # Missing values: whole features drop out under the complete-case rule.
    with_nan = rng.matrix(40, 8, loc=11.0, scale=1.0)
    with_nan[3, 2] = np.nan
    with_nan[17, 0] = np.nan
    with_nan[17, 7] = np.nan
    with_nan[39, 4] = np.nan
    add("with_missing", with_nan, "4 features carry a NaN and are dropped entirely")

    # Fewer features than samples: the component count is capped by rank, not by n_components.
    add("more_samples_than_features", rng.matrix(4, 9, loc=10.0, scale=1.0),
        "only 4 features, so at most 4 components exist")

    write(
        "pca.json",
        {
            "reference": "numpy.linalg.svd(full_matrices=False) on the centered complete-case matrix",
            "note": (
                "`expr` is features x samples and every column is selected. Component signs are "
                "arbitrary, so the golden holds |scores|."
            ),
            "cases": cases,
        },
    )


# --------------------------------------------------------------------------------------------
# Detection GLM
# --------------------------------------------------------------------------------------------


def gen_detection_lrt() -> None:
    """The Firth-penalized likelihood-ratio test behind the covariate-adjusted Detection view.

    `DetectionGlm` does NOT fit an unpenalized GLM, so statsmodels' `GLM(family=Binomial())` is the
    wrong reference for it: it fits Firth twice - once on the full design, once with the group
    column removed - and refers 2*(ll_full - ll_reduced) to chi2 with 1 df. The penalized
    log-likelihood is what firth.json already pins independently; this adds the two-fit arithmetic
    on top of it, with the tail probability from scipy.
    """
    cases = []

    def add(name: str, x_full, y, note: str) -> None:
        x_full = np.asarray(x_full, dtype=float)
        y = np.asarray(y, dtype=float)
        # PRISM drops design column 1 (the group indicator) to form the reduced model.
        x_red = np.delete(x_full, 1, axis=1)
        _, ll_full = firth_reference(x_full, y)
        _, ll_red = firth_reference(x_red, y)
        lrt = 2.0 * (ll_full - ll_red)
        pval = float(scipy.stats.chi2.sf(max(lrt, 0.0), 1))
        cases.append(
            {
                "name": name,
                "note": note,
                "x_full": mat(x_full),
                "y": vec(y),
                "ll_full": num(ll_full),
                "ll_reduced": num(ll_red),
                "lrt": num(lrt),
                "p": num(pval),
            }
        )

    n = 24
    grp = np.array([0.0] * 12 + [1.0] * 12)
    age = np.array([50.0 + 1.7 * i for i in range(n)])
    age_c = age - age.mean()

    y1 = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                   1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=float)
    add("group_effect_with_covariate", np.column_stack([np.ones(n), grp, age_c]), y1,
        "detection commoner in group B, adjusted for a centered numeric covariate")

    y2 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                   1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    add("no_group_effect", np.column_stack([np.ones(n), grp, age_c]), y2,
        "detection unrelated to group; the LRT should be near zero")

    y3 = np.array([0] * 12 + [1] * 12, dtype=float)
    add("complete_separation", np.column_stack([np.ones(n), grp, age_c]), y3,
        "detected in every B sample and no A sample - the case an unpenalized GLM cannot fit at all")

    add("group_only", np.column_stack([np.ones(n), grp]), y1,
        "no covariate: the reduced model is intercept-only")

    write(
        "detection_lrt.json",
        {
            "reference": (
                "scipy.optimize on the Jeffreys-penalized log-likelihood (twice) + "
                "scipy.stats.chi2.sf(., 1)"
            ),
            "note": (
                "Design column 1 is the group indicator and is the column the reduced model drops. "
                "Assert `p` and `lrt`, which are well determined; the individual log-likelihoods "
                "are reported for diagnosis. statsmodels' unpenalized GLM is deliberately NOT the "
                "reference here - it diverges on complete_separation, which is the case this path "
                "exists to handle."
            ),
            "cases": cases,
        },
    )


def gen_intensity_trend() -> None:
    """The toolkit's intensity-trend variance prior, per feature.

    The fixture stores LOG2 values because that is what PRISM holds in memory (its parquet is linear
    and the loader log2s it); the reference is handed the raw intensities those came from, so the
    delta-method conversion back into log2 space is exercised on both sides rather than cancelling
    out.
    """
    from proteomics_toolkit.statistical_analysis import _fit_intensity_trend_prior

    class Cfg:
        analysis_type = "unpaired"
        group_column = "Group"
        group_labels = ["A", "B"]
        log_base = "log2"
        variance_prior_group_column = None
        variance_prior_groups = None
        paired_column = None
        time_column = None
        dose_column = None

    cases = []

    def add(name: str, raw, n_a: int, note: str) -> None:
        raw = np.asarray(raw, dtype=float)
        n_feat, n_samp = raw.shape
        n_b = n_samp - n_a
        samples = [f"S{i}" for i in range(n_samp)]
        features = [f"f{i}" for i in range(n_feat)]
        expected, _ = _fit_intensity_trend_prior(
            {"features": features},
            pd.DataFrame(raw, index=features, columns=samples),
            pd.DataFrame({"Sample": samples, "Group": ["A"] * n_a + ["B"] * n_b}),
            Cfg(),
        )
        cases.append(
            {
                "name": name,
                "note": note,
                "n_a": n_a,
                "n_b": n_b,
                "expr_log2": mat(np.log2(raw)),
                "expected": vec(np.asarray(expected, dtype=float)),
            }
        )

    rng = Rng(29)
    # Three decades of abundance, which is what makes a trend worth fitting at all.
    base = [1.0e4, 3.0e4, 1.0e5, 4.0e5, 9.0e5, 2.0e6, 5.0e6, 1.2e7]

    def arm(b, n, cv, mult=1.0):
        return [abs(b * mult * (1.0 + cv * rng.normal())) for _ in range(n)]

    add("eight_features_two_groups",
        [arm(b, 4, 0.15) + arm(b, 4, 0.15, 1.4) for b in base], 4,
        "8 features over three decades, 4 + 4 samples: the ordinary case")

    # Heteroscedastic on purpose - noise falling with abundance is the relationship the LOWESS is
    # there to capture, and the reason a single global prior is wrong.
    cvs = [0.45, 0.38, 0.30, 0.22, 0.17, 0.13, 0.10, 0.08]
    add("heteroscedastic",
        [arm(b, 5, cv) + arm(b, 5, cv, 1.25) for b, cv in zip(base, cvs)], 5,
        "noise falls with abundance - the trend the prior exists to follow")

    # The per-feature combination is a SAMPLE-SIZE-weighted mean over groups, so an unbalanced
    # design is the case that tells a weighted mean from a plain one.
    add("unbalanced_arms",
        [arm(b, 6, 0.2) + arm(b, 3, 0.2, 1.3) for b in base], 6,
        "6 vs 3: distinguishes the sample-size-weighted combination from an unweighted one")

    write(
        "intensity_trend.json",
        {
            "reference": (
                "proteomics_toolkit.statistical_analysis._fit_intensity_trend_prior "
                "(moderation='intensity_trend'), v26.8.0"
            ),
            "note": (
                "Per-feature trend SHAPE, in log2 space: the prior before its level and degrees of "
                "freedom are fitted to the design residuals (that step is pinned by "
                "toolkit_end_to_end.json). `expr_log2` is what PRISM holds in memory; the reference "
                "was handed 2**expr_log2."
            ),
            "cases": cases,
        },
    )


def gen_simple_tests() -> None:
    """Welch, Student and Mann-Whitney, per feature.

    Mann-Whitney is pinned to ``method="asymptotic"`` ON PURPOSE. scipy's ``method="auto"`` switches
    to the exact permutation distribution when the larger sample is 8 or fewer and there are no
    ties; PRISM implements only the normal approximation, which is what every realistic cohort size
    uses. Pinning ``auto`` here would encode a branch PRISM does not have and fail on the small
    cases for a reason that has nothing to do with a defect.
    """
    rng = Rng(53)
    cases = []

    def add(name: str, a, b, note: str) -> None:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        welch = stats.ttest_ind(b, a, equal_var=False)
        student = stats.ttest_ind(b, a, equal_var=True)
        mw = stats.mannwhitneyu(b, a, alternative="two-sided", method="asymptotic")
        cases.append(
            {
                "name": name,
                "note": note,
                "a": vec(a),
                "b": vec(b),
                "welch_t": num(float(welch.statistic)),
                "welch_p": num(float(welch.pvalue)),
                "welch_df": num(float(welch.df)),
                "student_t": num(float(student.statistic)),
                "student_p": num(float(student.pvalue)),
                "student_df": num(float(student.df)),
                "mw_u": num(float(mw.statistic)),
                "mw_p": num(float(mw.pvalue)),
                "logfc": num(float(np.mean(b) - np.mean(a))),
                "median_diff": num(float(np.median(b) - np.median(a))),
            }
        )

    add("balanced", [rng.normal() for _ in range(6)], [1.2 + rng.normal() for _ in range(6)],
        "6 vs 6, similar spread")
    add("unequal_variance",
        [rng.normal() * 0.3 for _ in range(7)], [0.8 + rng.normal() * 2.5 for _ in range(9)],
        "the case Welch and Student disagree on - unequal n AND unequal spread")
    add("unbalanced_n", [rng.normal() for _ in range(12)], [0.5 + rng.normal() for _ in range(4)],
        "12 vs 4")
    add("no_difference", [rng.normal() for _ in range(10)], [rng.normal() for _ in range(10)],
        "null case: p should be unremarkable")
    add("with_ties",
        [round(rng.normal(), 1) for _ in range(10)], [round(0.6 + rng.normal(), 1) for _ in range(10)],
        "rounded values, so Mann-Whitney's tie correction is exercised")
    add("large", [rng.normal() for _ in range(40)], [0.35 + rng.normal() for _ in range(40)],
        "40 vs 40, where the normal approximation is the only sensible test anyway")

    write(
        "simple_tests.json",
        {
            "reference": (
                "scipy.stats.ttest_ind(equal_var=False|True) and "
                "scipy.stats.mannwhitneyu(alternative='two-sided', method='asymptotic')"
            ),
            "note": (
                "Contrast direction is B - A throughout, matching the moderated path. Mann-Whitney's "
                "U is for B against A. `median_diff` is what PRISM reports as the rank test's "
                "effect: a rank test makes no claim about means."
            ),
            "cases": cases,
        },
    )


def gen_corrections() -> None:
    """The multiple-testing methods beside BH."""
    rng = Rng(67)
    cases = []

    def add(name: str, p, note: str) -> None:
        p = np.asarray(p, dtype=float)
        row = {"name": name, "note": note, "p": vec(p)}
        for key, method in (("by", "fdr_by"), ("bonferroni", "bonferroni"), ("holm", "holm")):
            row[key] = vec(multipletests(p, method=method)[1])
        cases.append(row)

    add("uniform_steps", [0.01, 0.02, 0.03, 0.04, 0.05], "every step-up value equal")
    add("one_strong_hit", [1e-8, 0.2, 0.4, 0.6, 0.8, 0.9], "a single dominant hit")
    add("ties", [0.04, 0.04, 0.04, 0.2, 0.2, 0.9], "tied p-values take an identical adjusted value")
    add("single", [0.031], "m = 1: every method returns the raw value")
    add("many", [rng.next_double() ** 3 for _ in range(200)], "200 values, skewed to small p")

    write(
        "corrections.json",
        {
            "reference": "statsmodels.stats.multitest.multipletests(method='fdr_by'|'bonferroni'|'holm')",
            "note": (
                "PRISM's NaN policy (pass through, excluded from m) is its own and differs from "
                "statsmodels, which returns all-NaN if any input is NaN - so every case here is "
                "NaN-free, where the two agree exactly. FdrTests pins the NaN behavior."
            ),
            "cases": cases,
        },
    )


def gen_paired() -> None:
    """The paired tests, and the paired moderated design's subject block.

    Wilcoxon is pinned to ``method="asymptotic"`` for the same reason Mann-Whitney is: scipy's
    ``auto`` uses the exact signed-rank distribution up to n = 50 and PRISM implements only the
    normal approximation. ``ttest_rel`` has no such branch, so it is pinned as-is.

    The moderated case exercises the WITHIN-subject design: ``[1, grp, subject dummies]`` over the
    matched pairs, which is what removes between-subject level from the residual. It is composed
    here the way limma composes it, exactly as ``gen_moderated_t`` does - no PRISM code involved.
    """
    rng = Rng(89)
    cases = []

    def add(name: str, a, b, note: str) -> None:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        d = b - a
        rel = stats.ttest_rel(b, a)
        wil = stats.wilcoxon(d, alternative="two-sided", method="asymptotic")
        cases.append(
            {
                "name": name,
                "note": note,
                "a": vec(a),
                "b": vec(b),
                "paired_t": num(float(rel.statistic)),
                "paired_p": num(float(rel.pvalue)),
                "paired_df": num(float(rel.df)),
                "mean_diff": num(float(np.mean(d))),
                "wilcoxon_w": num(float(wil.statistic)),
                "wilcoxon_p": num(float(wil.pvalue)),
                "median_diff": num(float(np.median(d))),
            }
        )

    n = 8
    base = [rng.normal() * 2 for _ in range(n)]  # subject level, which pairing removes
    add("consistent_shift",
        [x + 0.3 * rng.normal() for x in base],
        [x + 1.1 + 0.3 * rng.normal() for x in base],
        "a real within-subject shift on top of large between-subject spread - the case pairing is for")
    add("no_shift",
        [x + 0.3 * rng.normal() for x in base],
        [x + 0.3 * rng.normal() for x in base],
        "null: subject level still large, but no within-subject change")
    add("with_ties",
        [round(x + 0.3 * rng.normal(), 1) for x in base],
        [round(x + 0.7 + 0.3 * rng.normal(), 1) for x in base],
        "rounded, so Wilcoxon's tie correction is exercised")
    big = [rng.normal() * 2 for _ in range(15)]
    add("fifteen_subjects",
        [x + 0.25 * rng.normal() for x in big],
        [x + 0.6 + 0.25 * rng.normal() for x in big],
        "15 pairs")

    # --- the paired MODERATED design, composed the way limma does ---
    n_feat, n_pairs = 9, 6
    subj = [rng.normal() * 1.5 for _ in range(n_pairs)]
    expr = []
    for f in range(n_feat):
        level = 12 + f * 0.4
        row_a = [level + subj[j] + 0.35 * rng.normal() for j in range(n_pairs)]
        row_b = [level + subj[j] + (0.9 if f < 3 else 0.0) + 0.35 * rng.normal() for j in range(n_pairs)]
        expr.append(row_a + row_b)
    expr = np.asarray(expr, dtype=float)

    # [intercept, grp, subject dummies for subjects 1..k-1]
    design = np.zeros((2 * n_pairs, 2 + (n_pairs - 1)))
    design[:, 0] = 1.0
    design[n_pairs:, 1] = 1.0
    for j in range(1, n_pairs):
        design[j, 1 + j] = 1.0
        design[n_pairs + j, 1 + j] = 1.0

    beta, _, _, _ = np.linalg.lstsq(design, expr.T, rcond=None)
    resid = expr.T - design @ beta
    df_resid = design.shape[0] - np.linalg.matrix_rank(design)
    sigma2 = (resid ** 2).sum(axis=0) / df_resid
    xtx_inv = np.linalg.inv(design.T @ design)
    v_c = xtx_inv[1, 1]

    sq = squeezeVar(sigma2, df_resid)
    df_prior = float(np.atleast_1d(sq["df_prior"])[0])
    var_post = np.atleast_1d(sq["var_post"])
    df_total = df_resid + df_prior
    # stdev_unscaled[1] = sqrt(v_c); same quantity as gen_moderated_t, spelled the same way.
    t_mod = beta[1, :] / (np.sqrt(v_c) * np.sqrt(var_post))
    if np.isinf(df_total):
        p_mod = 2.0 * scipy.stats.norm.cdf(-np.abs(t_mod))
    else:
        p_mod = 2.0 * scipy.stats.t.cdf(-np.abs(t_mod), df=df_total)

    write(
        "paired.json",
        {
            "reference": (
                "scipy.stats.ttest_rel, scipy.stats.wilcoxon(method='asymptotic'), and "
                "numpy.linalg.lstsq + inmoose.limma.squeezeVar composed as limma composes them"
            ),
            "note": (
                "Differences are B - A throughout. `moderated` is the paired DESIGN: samples are "
                "[A pairs..., B pairs...] in a common subject order, and the design is "
                "[1, grp, subject dummies 1..k-1]. Wilcoxon's W is the smaller signed-rank sum, as "
                "scipy reports for a two-sided test."
            ),
            "cases": cases,
            "moderated": {
                "n_pairs": n_pairs,
                "expr_log2": mat(expr),
                "logfc": vec(beta[1, :]),
                "t": vec(t_mod),
                "p": vec(p_mod),
                "df_residual": num(float(df_resid)),
                "df_prior": num(df_prior),
            },
        },
    )


def gen_trend() -> None:
    """The linear-trend designs: a slope against a numeric column, with and without a subject block.

    Composed the way limma composes any design - numpy.linalg.lstsq for the fit, inmoose's
    squeezeVar for the moderation - so no PRISM code is involved, exactly as ``gen_paired`` and
    ``gen_moderated_t`` do it.

    Two designs are pinned:

    * ``independent`` - ``[1, x]``, one sample per subject.
    * ``within_subject`` - ``[1, x, subject dummies 1..k-1]``, the same subjects followed across x.

    x is CENTERED in both, which does not move the slope but is what PRISM fits, so the intercept
    and therefore ``amean`` line up too.

    ``logfc`` is the slope times the SPAN of x, not the slope itself: that is what PRISM reports,
    so that one effect-size threshold means the same thing on a trend as on a two-arm contrast
    whatever units x is in. The raw slope is kept beside it so a reader can check the scaling.
    """
    rng = Rng(613)

    def moderated(expr, design):
        beta, _, _, _ = np.linalg.lstsq(design, expr.T, rcond=None)
        resid = expr.T - design @ beta
        df_resid = design.shape[0] - np.linalg.matrix_rank(design)
        sigma2 = (resid ** 2).sum(axis=0) / df_resid
        xtx_inv = np.linalg.inv(design.T @ design)
        v_c = xtx_inv[1, 1]
        sq = squeezeVar(sigma2, df_resid)
        df_prior = float(np.atleast_1d(sq["df_prior"])[0])
        var_post = np.atleast_1d(sq["var_post"])
        df_total = df_resid + df_prior
        t_mod = beta[1, :] / (np.sqrt(v_c) * np.sqrt(var_post))
        if np.isinf(df_total):
            p_mod = 2.0 * scipy.stats.norm.cdf(-np.abs(t_mod))
        else:
            p_mod = 2.0 * scipy.stats.t.cdf(-np.abs(t_mod), df=df_total)
        amean = expr.mean(axis=1)
        return beta[1, :], t_mod, p_mod, df_resid, df_prior, amean

    # --- independent: 12 samples, each its own subject, x = dose ---------------------------------
    n_feat = 9
    x_ind = np.array([0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0])
    expr_ind = []
    for f in range(n_feat):
        level = 12 + f * 0.4
        slope = (0.18 if f < 3 else 0.0)   # three features really move with dose
        expr_ind.append([level + slope * xv + 0.30 * rng.normal() for xv in x_ind])
    expr_ind = np.asarray(expr_ind, dtype=float)

    xc_ind = x_ind - x_ind.mean()
    design_ind = np.column_stack([np.ones_like(xc_ind), xc_ind])
    slope_i, t_i, p_i, dfr_i, dfp_i, amean_i = moderated(expr_ind, design_ind)
    span_ind = float(x_ind.max() - x_ind.min())

    # --- within subject: 6 subjects x 4 timepoints, big between-subject spread ------------------
    n_subj, n_time = 6, 4
    times = np.array([0.0, 4.0, 8.0, 12.0])
    subj_level = [rng.normal() * 2.0 for _ in range(n_subj)]   # what the block removes
    x_rep, subj_of = [], []
    for j in range(n_subj):
        for tv in times:
            x_rep.append(tv)
            subj_of.append(j)
    x_rep = np.asarray(x_rep, dtype=float)

    expr_rep = []
    for f in range(n_feat):
        level = 11 + f * 0.5
        slope = (0.09 if f < 3 else 0.0)
        row = []
        for j in range(n_subj):
            for tv in times:
                row.append(level + subj_level[j] + slope * tv + 0.22 * rng.normal())
        expr_rep.append(row)
    expr_rep = np.asarray(expr_rep, dtype=float)

    xc_rep = x_rep - x_rep.mean()
    design_rep = np.zeros((n_subj * n_time, 2 + (n_subj - 1)))
    design_rep[:, 0] = 1.0
    design_rep[:, 1] = xc_rep
    for s, j in enumerate(subj_of):
        if j > 0:
            design_rep[s, 1 + j] = 1.0
    slope_r, t_r, p_r, dfr_r, dfp_r, amean_r = moderated(expr_rep, design_rep)
    span_rep = float(x_rep.max() - x_rep.min())

    # The same data fitted WITHOUT the subject block, to show what the block is worth. Not a
    # target PRISM has to match - it is the model the block exists to avoid - but pinning it keeps
    # the two designs from silently becoming the same thing.
    design_naive = np.column_stack([np.ones_like(xc_rep), xc_rep])
    slope_n, t_n, p_n, _, _, _ = moderated(expr_rep, design_naive)

    write(
        "trend.json",
        {
            "reference": (
                "numpy.linalg.lstsq + inmoose.limma.squeezeVar, composed as limma composes a "
                "design - the same construction gen_moderated_t and gen_paired use"
            ),
            "note": (
                "x is centered before fitting, which does not move the slope. `logfc` is the slope "
                "times `x_span` - the modeled change across the observed range, which is what PRISM "
                "reports so that one effect-size cut means the same thing here as on a two-arm "
                "contrast. `slope` is the raw coefficient, in log2 per unit of x. `within_subject` "
                "adds [subject dummies 1..k-1]; `naive_t`/`naive_p` are the SAME data fitted "
                "without that block, which is the mistake the block exists to prevent - they are "
                "recorded to keep the two designs from collapsing into one."
            ),
            "independent": {
                "x": vec(x_ind),
                "x_span": num(span_ind),
                "expr_log2": mat(expr_ind),
                "slope": vec(slope_i),
                "logfc": vec(slope_i * span_ind),
                "t": vec(t_i),
                "p": vec(p_i),
                "amean": vec(amean_i),
                "df_residual": num(float(dfr_i)),
                "df_prior": num(dfp_i),
            },
            "within_subject": {
                "x": vec(x_rep),
                "x_span": num(span_rep),
                "subject_of": [str(j) for j in subj_of],
                "n_subjects": n_subj,
                "expr_log2": mat(expr_rep),
                "slope": vec(slope_r),
                "logfc": vec(slope_r * span_rep),
                "t": vec(t_r),
                "p": vec(p_r),
                "amean": vec(amean_r),
                "df_residual": num(float(dfr_r)),
                "df_prior": num(dfp_r),
                "naive_t": vec(t_n),
                "naive_p": vec(p_n),
            },
        },
    )


def gen_toolkit_end_to_end() -> None:
    """The whole default analysis, against the toolkit's own top-level entry point.

    ``gen_intensity_trend`` pins the prior SCALE in isolation, fitted on the contrast's two arms,
    and ``gen_paired``/``gen_trend`` pin the designs under a global prior. Neither pins what a lab
    analysis actually runs: ``run_comprehensive_statistical_analysis`` with
    ``moderation="intensity_trend"`` and the prior fitted on DEDICATED control replicates
    (``variance_prior_group_column``), composed with a paired or within-subject design. That
    composition is where the two tools could disagree while every piece agreed - the prior's groups
    are not the contrast's, and the trend they give is only a SHAPE whose level and degrees of
    freedom are fitted to the DESIGN residuals - so it is pinned here, through the same call the
    lab's notebooks make.

    **The calibration step is checked independently as well.** From toolkit v26.8.0 (PRISM
    ``WithCalibratedLevel``) the trend is multiplied by a level fitted, with the prior df, to
    ``residual_s2 / trend``. That is limma's fitFDist with the trend as a known covariate offset, so
    each case also hands the toolkit's own ``residual_s2`` and ``intensity_trend_shape`` to
    ``inmoose.limma.squeezeVar`` and refuses to write the golden unless the level and df agree.
    The toolkit is the definition of the shape. The calibration has a third-party reference.

    The cohort is synthetic, shaped like the Verapamil serum study this was checked against
    (subjects drawn at weeks 0, 2, 4, 6 and 12, plus reference and QC pools): a real clinical
    cohort cannot be committed. On that cohort the two tools agreed to ~1e-11 once the two
    conventions below were aligned; these cases pin the same agreement.

    Two places where PRISM and the toolkit deliberately differ, and how each is handled:

    * **The log pseudocount.** The toolkit's dispatcher adds one before log2 - by default the
      smallest value / 100 - because a matrix it is handed may hold zeros. PRISM's matrix is log2
      of a LINEAR parquet, and a non-positive value is missing rather than shifted, so it adds
      nothing. The reference is run with ``log_pseudocount = 0`` so it computes on the same numbers;
      left at its default, it moves low-abundance log ratios by up to ~1e-3 on the real cohort.
    * **The LOWESS interpolation distance.** PRISM's prior LOWESS passes ``delta`` = 1% of the x
      range (see ``VariancePriors.SmoothTrend`` for why it cannot afford 0 on a peptide-level
      contrast); the toolkit passes 0. These inputs are built so that no two trend points lie
      within 1% of the range of each other - asserted below - so every point is fitted on both
      sides and the two smoothers are the same computation. On the real cohort, where they are
      not, the difference moved the prior scale by at most 1.4e-3 and p-values by at most 8e-4,
      relative.

    ``logfc`` is in PRISM's convention: B - A for a two-arm design, and the slope times the span of
    x for a trend. ``slope`` keeps the toolkit's own trend coefficient beside it.

    **Both branches of the d0 estimate are covered.** Where every feature's variance follows the
    trend, d0 is infinite and the posterior is the scaled trend itself (``unpaired``, ``paired``).
    The ``*_finite_d0`` cases give each feature its own departure from the trend, so d0 is finite
    and the residual variances enter every t - the regime a real cohort is in.
    """
    import contextlib
    import io

    import proteomics_toolkit.statistical_analysis as sa

    rng = Rng(1931)
    n_feat = 12
    # Four decades of abundance, noisier at the bottom - the relationship the intensity trend
    # exists to follow. The spacing is wide on purpose: a feature's two control points (reference
    # and QC, 0.6 log2 apart) and its neighbor's must stay more than 1% of the range apart after
    # the replicate noise moves their means (see assert_delta_inert).
    level = [10.0 + 1.2 * f for f in range(n_feat)]
    noise = [0.36 - 0.02 * f for f in range(n_feat)]
    moves = 3  # features 0..2 carry a real effect; the rest are null

    def controls():
        """4 reference + 4 QC replicates: two materials at two levels, technical noise only.

        Half the design samples' noise, as for real pooled injections - and what keeps the group
        means from wandering into each other's LOWESS neighborhood.
        """
        ref = [[level[f] + 0.5 * noise[f] * rng.normal() for _ in range(4)] for f in range(n_feat)]
        qc = [[level[f] - 0.6 + 0.5 * noise[f] * rng.normal() for _ in range(4)]
              for f in range(n_feat)]
        return np.asarray(ref), np.asarray(qc)

    def assert_delta_inert(expr_log2, prior_cols):
        # The trend's x is log(raw group mean), one point per (feature, prior group).
        xs = np.sort(np.concatenate([
            np.log(np.mean(2.0 ** expr_log2[:, cols], axis=1)) for cols in prior_cols]))
        gap = float(np.min(np.diff(xs)))
        delta = 0.01 * float(xs[-1] - xs[0])
        if not gap > delta:
            raise SystemExit(
                f"two prior points lie {gap:.4g} apart, inside PRISM's LOWESS delta {delta:.4g}: "
                "the golden would pin the interpolation, not the estimator")

    cases = []

    def run(name, design, note, design_block, meta_rows, cfg_fn, a_cols, b_cols, x, subject_of,
            finite_d0=False):
        """Assemble [design | ref | qc], run the toolkit, record what PRISM must reproduce."""
        ref, qc = controls()
        expr = np.column_stack([design_block, ref, qc])
        n_design = design_block.shape[1]
        ref_cols = list(range(n_design, n_design + 4))
        qc_cols = list(range(n_design + 4, n_design + 8))
        assert_delta_inert(expr, [ref_cols, qc_cols])

        samples = [f"S{j:02d}" for j in range(expr.shape[1])]
        features = [f"f{i}" for i in range(n_feat)]
        meta = {}
        for j, s in enumerate(samples):
            row = {"Replicate": s, "Category": "Experimental"}
            if j < n_design:
                row.update(meta_rows[j])
            else:
                row["Category"] = "Reference" if j in ref_cols else "QC"
            meta[s] = row

        annot = pd.DataFrame({
            "Protein": features, "Description": features, "Protein Gene": features,
            "UniProt_Accession": features, "UniProt_Entry_Name": features})
        data = pd.concat([annot, pd.DataFrame(2.0 ** expr, columns=samples)], axis=1)
        data.index = features
        ann = annot.copy()
        ann.index = features
        ann["Gene"] = features

        cfg = sa.StatisticalConfig()
        cfg.statistical_test_method = "moderated_linear_model"
        cfg.moderation = "intensity_trend"
        cfg.variance_prior_group_column = "Category"
        cfg.variance_prior_groups = ["Reference", "QC"]
        cfg.log_transform_before_stats = "auto"
        cfg.log_base = "log2"
        cfg.log_pseudocount = 0.0  # see the docstring: PRISM adds none
        cfg.normalization_method = "prism"
        cfg.correction_method = "fdr_bh"
        cfg.subject_column = None
        cfg_fn(cfg)

        with contextlib.redirect_stdout(io.StringIO()):
            res = sa.run_comprehensive_statistical_analysis(
                normalized_data=data, sample_metadata=meta, config=cfg, protein_annotations=ann)
        res = res.set_index("Protein").loc[features]

        span = float(np.nanmax(x) - np.nanmin(x)) if x is not None else 1.0
        logfc = res["logFC"].to_numpy(float) * span
        # The sign convention is the thing most easily got backwards between two tools, so it is
        # checked against the data rather than assumed: the moved features must move the way they
        # were built to.
        if x is None:
            raw_diff = expr[:, b_cols].mean(axis=1) - expr[:, a_cols].mean(axis=1)
            if not np.all(np.sign(logfc[:moves]) == np.sign(raw_diff[:moves])):
                raise SystemExit(f"{name}: toolkit logFC is not B - A")
        df_resid = res["residual_df"].to_numpy(float)
        df_prior = res["posterior_df"].to_numpy(float) - df_resid
        prior_level = float(res["intensity_trend_level"].iloc[0])

        # The calibration, against limma: fitFDist on residual / trend gives the level as its scale.
        sq = squeezeVar(res["residual_s2"].to_numpy(float) / res["intensity_trend_shape"].to_numpy(float),
                        float(df_resid[0]))
        ref_level = float(np.atleast_1d(sq["var_prior"])[0])
        ref_d0 = float(np.atleast_1d(sq["df_prior"])[0])
        level_ok = abs(prior_level - ref_level) <= 1e-9 * abs(ref_level)
        # An infinite d0 agrees only with an infinite one. Folded into the relative check it passed
        # any finite toolkit d0 against an infinite reference: |finite - inf| <= 1e-9 * inf is
        # inf <= inf, which is true.
        if np.isinf(ref_d0) or np.isinf(df_prior[0]):
            d0_ok = bool(np.isinf(ref_d0) and np.isinf(df_prior[0]))
        else:
            d0_ok = abs(df_prior[0] - ref_d0) <= 1e-9 * abs(ref_d0)
        if not (level_ok and d0_ok):
            raise SystemExit(f"{name}: toolkit level {prior_level!r} / d0 {df_prior[0]!r} disagree with "
                             f"inmoose squeezeVar on the ratio ({ref_level!r} / {ref_d0!r})")
        if finite_d0 and not np.isfinite(df_prior[0]):
            raise SystemExit(f"{name}: d0 came out infinite, so this case would not reach the finite-d0 "
                             "posterior it exists to pin")
        cases.append({
            "name": name,
            "design": design,
            "note": note,
            "expr_log2": mat(expr),
            "a_columns": a_cols,
            "b_columns": b_cols,
            "prior_groups": [ref_cols, qc_cols],
            "x": None if x is None else vec(x),
            "subject_of": subject_of,
            "logfc": vec(logfc),
            "slope": vec(res["logFC"]) if x is not None else None,
            "t": vec(res["t"]),
            "p": vec(res["P.Value"]),
            "adj_p": vec(res["adj.P.Val"]),
            "amean": vec(res["AveExpr"]),
            "prior_scale": vec(res["intensity_s0_sq"]),
            "prior_level": num(prior_level),
            "df_residual": num(df_resid[0]),
            "df_prior": num(df_prior[0]),
        })

    # --- unpaired: 5 vs 5 independent samples ----------------------------------------------------
    block = np.asarray([
        [level[f] + (0.8 if (f < moves and j >= 5) else 0.0) + (noise[f] + 0.15) * rng.normal()
         for j in range(10)] for f in range(n_feat)])
    run("unpaired", "unpaired", "5 vs 5, prior on 4 reference + 4 QC pools that take no part in the contrast",
        block, [{"Group": "Control" if j < 5 else "Treated"} for j in range(10)],
        lambda c: (setattr(c, "analysis_type", "unpaired"), setattr(c, "group_column", "Group"),
                   setattr(c, "group_labels", ["Control", "Treated"])),
        list(range(5)), list(range(5, 10)), None, None)

    # --- paired: 6 subjects before and after, large between-subject spread ----------------------
    subj = [1.5 * rng.normal() for _ in range(6)]
    block = np.asarray([
        [level[f] + subj[j % 6] + (-0.7 if (f < moves and j >= 6) else 0.0) + noise[f] * rng.normal()
         for j in range(12)] for f in range(n_feat)])
    run("paired", "paired", "6 subjects, Pre then Post; the subject block removes a ~1.5 log2 spread",
        block,
        [{"Subject": f"P{j % 6}", "Timepoint": "Pre" if j < 6 else "Post"} for j in range(12)],
        lambda c: (setattr(c, "analysis_type", "paired"), setattr(c, "subject_column", "Subject"),
                   setattr(c, "paired_column", "Timepoint"), setattr(c, "paired_label1", "Pre"),
                   setattr(c, "paired_label2", "Post"), setattr(c, "group_column", "Timepoint"),
                   setattr(c, "group_labels", ["Pre", "Post"])),
        list(range(6)), list(range(6, 12)), None, [f"P{j % 6}" for j in range(12)])

    # --- trends: the study's own schedule, weeks 0, 2, 4, 6, 12 ----------------------------------
    weeks = [0.0, 2.0, 4.0, 6.0, 12.0]
    n_subj = 6
    x = np.asarray([w for _ in range(n_subj) for w in weeks])
    subject_of = [f"P{j}" for j in range(n_subj) for _ in weeks]
    subj = [1.5 * rng.normal() for _ in range(n_subj)]
    block = np.asarray([
        [level[f] + subj[int(subject_of[k][1:])] + (-0.05 if f < moves else 0.0) * x[k]
         + noise[f] * rng.normal() for k in range(len(x))] for f in range(n_feat)])
    trend_meta = [{"Subject": subject_of[k], "Week": x[k]} for k in range(len(x))]
    run("within_subject_trend", "within_subject_trend",
        "6 subjects at weeks 0, 2, 4, 6, 12, slope -0.05 log2/week on f0-f2; [1, x, subject dummies]",
        block, trend_meta,
        lambda c: (setattr(c, "analysis_type", "linear_trend"), setattr(c, "time_column", "Week"),
                   setattr(c, "subject_column", "Subject")),
        list(range(len(x))), [], x, subject_of)

    # The SAME data without the subject block: a between-subject trend, where the residual carries
    # the whole 1.5 log2 spread between people and the control pools carry none of it. This is the
    # design the uncalibrated prior got most wrong - its d0 came out infinite on this data, so the
    # posterior WAS the pools' technical variance and null features reached p ~ 1e-8 - and so the
    # one most worth pinning now that the level is fitted to the residuals.
    run("independent_trend", "trend",
        "the within-subject data again without the subject block: [1, x], prior on the pools",
        block, [{"Week": x[k]} for k in range(len(x))],
        lambda c: (setattr(c, "analysis_type", "linear_trend"), setattr(c, "time_column", "Week")),
        list(range(len(x))), [], x, None)

    # --- two-arm designs whose d0 is FINITE ------------------------------------------------------
    # In the two-arm cases above every feature's true variance sits on the trend, so the residuals
    # scatter around it by sampling noise alone, fitFDist finds no spread beyond that, and d0 comes
    # out infinite: the posterior IS the scaled trend and the residual variances never reach t. A
    # real cohort's d0 is finite (about 2.5 on the serum cohort), so these give each feature its own
    # departure from the trend, which is what makes d0 finite and puts every residual into its t.
    # Appended after the cases above, so those keep the draws they were generated with.
    spread = [float(np.exp(0.7 * rng.normal())) for _ in range(n_feat)]
    block = np.asarray([
        [level[f] + (0.8 if (f < moves and j >= 5) else 0.0) + (noise[f] + 0.15) * spread[f] * rng.normal()
         for j in range(10)] for f in range(n_feat)])
    run("unpaired_finite_d0", "unpaired",
        "the unpaired design again, each feature's noise scaled by its own factor, so d0 is finite",
        block, [{"Group": "Control" if j < 5 else "Treated"} for j in range(10)],
        lambda c: (setattr(c, "analysis_type", "unpaired"), setattr(c, "group_column", "Group"),
                   setattr(c, "group_labels", ["Control", "Treated"])),
        list(range(5)), list(range(5, 10)), None, None, finite_d0=True)

    subj = [1.5 * rng.normal() for _ in range(6)]
    block = np.asarray([
        [level[f] + subj[j % 6] + (-0.7 if (f < moves and j >= 6) else 0.0) + noise[f] * spread[f] * rng.normal()
         for j in range(12)] for f in range(n_feat)])
    run("paired_finite_d0", "paired",
        "the paired design again, each feature's noise scaled by its own factor, so d0 is finite",
        block,
        [{"Subject": f"P{j % 6}", "Timepoint": "Pre" if j < 6 else "Post"} for j in range(12)],
        lambda c: (setattr(c, "analysis_type", "paired"), setattr(c, "subject_column", "Subject"),
                   setattr(c, "paired_column", "Timepoint"), setattr(c, "paired_label1", "Pre"),
                   setattr(c, "paired_label2", "Post"), setattr(c, "group_column", "Timepoint"),
                   setattr(c, "group_labels", ["Pre", "Post"])),
        list(range(6)), list(range(6, 12)), None, [f"P{j % 6}" for j in range(12)], finite_d0=True)

    write(
        "toolkit_end_to_end.json",
        {
            "reference": (
                "proteomics_toolkit.statistical_analysis.run_comprehensive_statistical_analysis "
                "(statistical_test_method='moderated_linear_model', moderation='intensity_trend', "
                "variance_prior_group_column, correction_method='fdr_bh', log_pseudocount=0), v26.8.0; "
                "its trend level and prior df cross-checked against inmoose.limma.squeezeVar on "
                "residual_s2 / intensity_trend_shape"
            ),
            "note": (
                "Each case's columns are [design samples | 4 reference | 4 QC]; the prior is fitted "
                "on the last eight, which take no part in the design. `logfc` is PRISM's "
                "convention (B - A, or slope x span of x); `slope` is the toolkit's own trend "
                "coefficient. The pools give the trend's SHAPE; `prior_level` is the factor it was "
                "scaled by to fit the design residuals, `prior_scale` the result, and `df_prior` "
                "(posterior_df - residual_df) the df fitted with it. The toolkit was run with "
                "log_pseudocount=0 because PRISM adds none, "
                "and the inputs keep every prior point more than 1% of the x range from the next, "
                "so PRISM's LOWESS delta interpolates nothing."
            ),
            "cases": cases,
        },
    )


def gen_peptide_count_prior() -> None:
    """The DEqMS-style prior: a LOWESS of log(residual variance) on log(peptide count).

    Referenced to the toolkit's ``_fit_count_dependent_prior`` for the same reason the intensity
    trend is referenced to its neighbour: it is that tool's estimator PRISM has to reproduce, not a
    published closed form with an independent implementation to check against.
    """
    from proteomics_toolkit.statistical_analysis import _fit_count_dependent_prior

    rng = Rng(101)
    cases = []

    def add(name: str, variances, counts, note: str) -> None:
        variances = np.asarray(variances, dtype=float)
        counts = np.asarray(counts, dtype=float)
        fit = {"features": [f"f{i}" for i in range(len(variances))], "s2": variances}
        expected = _fit_count_dependent_prior(fit, counts)
        # The toolkit returns either the array or (array, extras) depending on version; take the array.
        if isinstance(expected, tuple):
            expected = expected[0]
        cases.append(
            {
                "name": name,
                "note": note,
                "variances": vec(variances),
                "counts": vec(counts),
                "expected": vec(np.asarray(expected, dtype=float)),
            }
        )

    n = 24
    counts = [float(1 + (i * 7) % 19) for i in range(n)]
    # Variance falling with peptide count is the relationship DEqMS exists to capture: a protein
    # rolled up from many peptides is better determined than one rolled up from few.
    variances = [0.6 / c + 0.02 + 0.01 * abs(rng.normal()) for c in counts]
    add("falling_with_count", variances, counts,
        "24 proteins, variance falling with peptide count - the DEqMS relationship")

    flat = [0.2 + 0.02 * abs(rng.normal()) for _ in range(n)]
    add("flat", flat, counts, "no count dependence: the trend should be nearly a flat line")

    write(
        "peptide_count_prior.json",
        {
            "reference": (
                "proteomics_toolkit.statistical_analysis._fit_count_dependent_prior "
                "(moderation='deqms'), v26.8.0"
            ),
            "note": (
                "Per-feature prior SCALE, in the same log2 space the residual variances are already "
                "in - unlike the intensity trend there is no delta-method conversion, because "
                "nothing is fitted on the raw scale. The prior degrees of freedom stay global."
            ),
            "cases": cases,
        },
    )


def gen_mcnemar() -> None:
    """McNemar's exact test, the paired counterpart of Fisher on detection.

    statsmodels' `mcnemar` defaults to `exact=True`, which is a genuine exact binomial on the
    discordant pairs at any size - so unlike the rank tests there is no asymptotic branch to choose
    between, and nothing to caveat.
    """
    from statsmodels.stats.contingency_tables import mcnemar

    cases = []

    def add(name: str, b: int, c: int, note: str) -> None:
        # The off-diagonal is all the test uses; the diagonal is filled in only because the API
        # takes a full 2x2.
        table = [[10, b], [c, 10]]
        res = mcnemar(table, exact=True)
        cases.append(
            {"name": name, "note": note, "b": b, "c": c, "expected_p": num(float(res.pvalue))}
        )

    add("no_discordance", 0, 0, "every subject agreed with itself: no evidence either way, p = 1")
    add("all_one_way", 8, 0, "8 discordant pairs, all the same direction - the strongest this can be")
    add("balanced", 6, 6, "equal discordance both ways: p = 1")
    add("mild", 9, 3, "a lean, not a separation")
    add("single_pair", 1, 0, "one discordant pair proves nothing")
    add("large", 40, 18, "58 discordant pairs, where a chi-square approximation would also do")
    add("very_large", 300, 250, "big enough that the binomial sum must be done in log space")
    add("lopsided_large", 25, 2, "strong and large")

    write(
        "mcnemar.json",
        {
            "reference": "statsmodels.stats.contingency_tables.mcnemar(exact=True)",
            "note": (
                "b and c are the two discordant counts; concordant pairs carry no information and "
                "are not part of the test. The p is two-sided."
            ),
            "cases": cases,
        },
    )


GENERATORS = [
    gen_fdr,
    gen_polygamma,
    gen_squeezevar,
    gen_spline,
    gen_lmfit,
    gen_moderated_t,
    gen_fisher,
    gen_firth,
    gen_pca,
    gen_detection_lrt,
    gen_intensity_trend,
    gen_simple_tests,
    gen_corrections,
    gen_paired,
    gen_peptide_count_prior,
    gen_mcnemar,
    gen_trend,
    gen_toolkit_end_to_end,
]


def main(names: list[str]) -> None:
    """Regenerate the named fixtures (``trend``, ``toolkit_end_to_end``, ...), or all of them.

    Naming one is the normal case. Regenerating everything to add or settle one golden rewrites
    the others under whatever library versions happen to be installed, which once rewrote seven
    of ten pre-existing goldens purely from numpy/scipy drift.
    """
    if not OUT.is_dir():
        raise SystemExit(f"run from the repository root: {OUT} not found")
    by_name = {g.__name__.removeprefix("gen_"): g for g in GENERATORS}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise SystemExit(f"unknown fixture(s) {unknown}; choose from {sorted(by_name)}")
    for g in ([by_name[n] for n in names] if names else GENERATORS):
        g()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
