#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy==2.5.3",
# ]
# ///
"""Generate the golden for library assist's noise-gated ratio fit (noise_gated.json).

There is no published implementation of this method to call, so the golden is a FIRST-PRINCIPLES
construction (CLAUDE.md, "choosing the source", option 3): the definitions below are written out and
computed with plain Python loops and `math`. numpy is used only for the seeded random draws and for
`numpy.median`. Nothing here imports PRISM, and nothing was read from its C# code - the definitions
were fixed first, from the measurements that chose the method (see the release note), and both
implementations were written against them.

Definitions
===========

Fit, per precursor and sample j. Inputs: fragment areas o[k, j] (LINEAR; NaN -> 0, negative -> 0),
library intensities lib[k] (a row is MATCHED when lib[k] is finite and > 0), min_fragments, the
ratio threshold t, the noise multiple z, and the fragment-noise curve sd(.).

    kept = every matched row
    repeat at most max_removals (5) times:
        c = sum_{kept} o / sum_{kept} lib
        stop if removal is off, or |kept| <= min_fragments
        pred_k = c * lib_k ;  nr_k = (o_k - pred_k) / pred_k  (0 when pred_k <= 0)
        flagged_k = nr_k > t  AND  (gate off  OR  o_k - pred_k > z * sd(pred_k))
        stop if nothing is flagged; else drop the flagged row with the largest nr (first on ties)
    c = sum_{kept} o / sum_{kept} lib
    abundance = c * sum_{matched} lib
    retained  = sum_{kept} r[k, j]   (r = the raw areas, NaN -> 0, negative -> 0; o when not given)
    kept      = |kept|
The gate is off when no curve is given or z <= 0. Fewer than min_fragments matched rows -> abundance
NaN, retained 0, kept 0 for every sample.

Curves. sd(e) interpolates ln(sd) linearly in ln(e) between the curve's points and clamps at both
ends. The expected CV interpolates cv linearly in ln(signal), clamped.

Noise curve estimate. Per peptide: drop precursor rows when asked; transition ids in order of first
appearance; each (transition, sample) cell takes the FIRST non-NaN area; then NaN -> 0, negative -> 0.
For each group G of n >= 3 samples and each precursor charge (> 0) with m >= 2 fragments, X is m x n:

    mu_k = mean_j X[k, j] ;  T_j = sum_k X[k, j] ;  f_j = T_j / mean(T)      (skip when mean(T) <= 0)
    for every k with mu_k > 0 and j with e = mu_k * f_j > 0:
        v = (X[k, j] - e)^2 * n * m / ((n - 1) * (m - 1))
        fine bin b = floor(log10(e) / fine_width): n_b += 1, V_b += v, L_b += ln(e)

f_j removes the injection-to-injection factor shared by a precursor's fragments, so v measures what
is left: each fragment's own noise. n*m/((n-1)(m-1)) is the residual degrees of freedom of that
rank-1 fit, spread over its cells. Fine bins, in increasing b, are pooled into coarse bins of at least
max(min_cells_per_bin, ceil(N / target_bins)) cells (N = all cells); a short remainder joins the last
coarse bin. A coarse bin's point is (exp(sum L / sum n), sqrt(sum V / sum n)).

Expected-CV curve, pool groups only: per peptide and group, S_j = sum over every fragment row of X;
a point is (mean S, 100 * sd(S, ddof=1) / mean S) when mean S > 0. Points are sorted by (mean, cv) and
split numpy.array_split-style into B = min(target_bins, P // min_signal_per_bin) contiguous chunks
(B = 1 when that is 0 and P >= 3); a chunk's point is (median of its means, median of its cvs).

Precision row, per peptide and pool group: abundance a_j, retained r_j, kept k_j and total fragment
signal s_j over the group's samples. Observed CV = 100 * sd(a, ddof=1) / mean(a) over the finite,
positive a_j (NaN when fewer than 3). Means of r, k and s over the group. Expected CV = the
expected-CV curve at mean(r). Ratio = observed / expected.

Tolerance: 1e-12 relative. Everything is closed-form arithmetic; the C# sums run in the same order
but numpy's median and Python's float formatting are the only shared pieces, so agreement is to
rounding. Fine-bin membership is a discrete decision, so the generator refuses any cell whose
log10(e) / fine_width lies within 1e-9 of an integer - a 1-ulp difference must not move a cell.

Run from the repository root:

    uv run dotnet/tests/fixtures/library_assist/generate.py
"""

import json
import math
import os

import numpy as np

NAN = float("nan")
INF = float("inf")


def s(x):
    return repr(float(x))


def clean(v):
    return 0.0 if math.isnan(v) else max(v, 0.0)


# ---------------------------------------------------------------- curves

def sd_at(curve, e):
    lv, sd = curve["level"], curve["sd"]
    if e <= lv[0]:
        return sd[0]
    if e >= lv[-1]:
        return sd[-1]
    for i in range(1, len(lv)):
        if e <= lv[i]:
            t = (math.log(e) - math.log(lv[i - 1])) / (math.log(lv[i]) - math.log(lv[i - 1]))
            return math.exp(math.log(sd[i - 1]) + t * (math.log(sd[i]) - math.log(sd[i - 1])))
    return sd[-1]


def cv_at(curve, signal):
    lv, cv = curve["level"], curve["cv"]
    if not lv:
        return NAN
    if signal <= lv[0]:
        return cv[0]
    if signal >= lv[-1]:
        return cv[-1]
    for i in range(1, len(lv)):
        if signal <= lv[i]:
            t = (math.log(signal) - math.log(lv[i - 1])) / (math.log(lv[i]) - math.log(lv[i - 1]))
            return cv[i - 1] + t * (cv[i] - cv[i - 1])
    return cv[-1]


# ---------------------------------------------------------------- fit

def fit(obs, lib, min_fragments, threshold, sigma, curve, remove, retained_src=None, max_removals=5):
    n_rows, n_samples = len(obs), len(obs[0])
    matched = [k for k in range(n_rows) if not math.isnan(lib[k]) and lib[k] > 0]
    if len(matched) < min_fragments:
        return [NAN] * n_samples, [0.0] * n_samples, [0] * n_samples
    src = obs if retained_src is None else retained_src
    lib_sum = 0.0
    for k in matched:
        lib_sum += lib[k]
    gate = curve is not None and sigma > 0

    def scale(kept, j):
        so, sl = 0.0, 0.0
        for k in kept:
            so += clean(obs[k][j])
            sl += lib[k]
        return so / sl

    abundance, retained, kept_n = [], [], []
    for j in range(n_samples):
        kept = list(matched)
        for _ in range(max_removals):
            c = scale(kept, j)
            if not remove or len(kept) <= min_fragments:
                break
            worst, worst_nr = None, -INF
            for k in kept:
                o = clean(obs[k][j])
                pred = lib[k] * c
                nr = (o - pred) / pred if pred > 0 else 0.0
                flagged = nr > threshold and (not gate or o - pred > sigma * sd_at(curve, pred))
                if flagged and nr > worst_nr:
                    worst, worst_nr = k, nr
            if worst is None:
                break
            kept.remove(worst)
        c = scale(kept, j)
        abundance.append(c * lib_sum)
        r = 0.0
        for k in kept:
            r += clean(src[k][j])
        retained.append(r)
        kept_n.append(len(kept))
    return abundance, retained, kept_n


# ---------------------------------------------------------------- noise curve

def estimate(rows, n_samples, groups, source, exclude_precursor, opts):
    width = opts["fine_width"]
    fine = {}
    points = []
    peptides = []
    by_pep = {}
    for r in rows:
        if r[0] not in by_pep:
            by_pep[r[0]] = []
            peptides.append(r[0])
        by_pep[r[0]].append(r)

    for pep in peptides:
        rs = [r for r in by_pep[pep] if not (exclude_precursor and r[5])]
        tids, index, charge = [], {}, []
        for r in rs:
            if r[1] not in index:
                index[r[1]] = len(tids)
                tids.append(r[1])
                charge.append(r[2])
        raw = [[NAN] * n_samples for _ in tids]
        filled = [[False] * n_samples for _ in tids]
        for r in rs:
            t, j, area = index[r[1]], r[3], r[4]
            if not filled[t][j] and not math.isnan(area):
                raw[t][j] = area
                filled[t][j] = True
        raw = [[clean(v) for v in row] for row in raw]

        for g in groups:
            idx = g["indices"]
            n = len(idx)
            if n < 3:
                continue
            if source == "pool_replicates":
                totals = []
                for j in idx:
                    acc = 0.0
                    for t in range(len(tids)):
                        acc += raw[t][j]
                    totals.append(acc)
                mean = sum(totals) / n
                if mean > 0:
                    ss = 0.0
                    for x in totals:
                        ss += (x - mean) ** 2
                    points.append((mean, 100.0 * math.sqrt(ss / (n - 1)) / mean))
            for z in sorted(set(charge)):
                if z <= 0:
                    continue  # an unparseable charge, which the rollup never fits
                ks = [t for t in range(len(tids)) if charge[t] == z]
                m = len(ks)
                if m < 2:
                    continue
                mu = []
                for k in ks:
                    acc = 0.0
                    for j in idx:
                        acc += raw[k][j]
                    mu.append(acc / n)
                tj = []
                for j in idx:
                    acc = 0.0
                    for k in ks:
                        acc += raw[k][j]
                    tj.append(acc)
                tbar = sum(tj) / n
                if tbar <= 0:
                    continue
                f = [x / tbar for x in tj]
                factor = n * m / ((n - 1) * (m - 1))
                for a, k in enumerate(ks):
                    if mu[a] <= 0:
                        continue
                    for b, j in enumerate(idx):
                        e = mu[a] * f[b]
                        if e <= 0:
                            continue
                        x = math.log10(e) / width
                        if abs(x - round(x)) < 1e-9:
                            raise SystemExit(f"cell on a fine-bin edge ({pep}, e={e!r}); change the seed")
                        r = raw[k][j] - e
                        bin_ = math.floor(x)
                        cell = fine.setdefault(bin_, [0, 0.0, 0.0])
                        cell[0] += 1
                        cell[1] += r * r * factor
                        cell[2] += math.log(e)

    total = sum(c[0] for c in fine.values())
    target = max(opts["min_cells_per_bin"], math.ceil(total / opts["target_bins"]))
    coarse = []
    acc = [0, 0.0, 0.0]
    for b in sorted(fine):
        c = fine[b]
        acc = [acc[0] + c[0], acc[1] + c[1], acc[2] + c[2]]
        if acc[0] >= target:
            coarse.append(acc)
            acc = [0, 0.0, 0.0]
    if acc[0] > 0:
        if coarse:
            last = coarse[-1]
            coarse[-1] = [last[0] + acc[0], last[1] + acc[1], last[2] + acc[2]]
        else:
            coarse.append(acc)
    level = [math.exp(c[2] / c[0]) for c in coarse]
    sd = [math.sqrt(c[1] / c[0]) for c in coarse]

    points.sort()
    p = len(points)
    nbins = min(opts["target_bins"], p // opts["min_signal_per_bin"])
    if nbins == 0 and p >= 3:
        nbins = 1
    s_level, s_cv = [], []
    if nbins > 0:
        base, extra = divmod(p, nbins)
        start = 0
        for i in range(nbins):
            size = base + (1 if i < extra else 0)
            chunk = points[start:start + size]
            start += size
            s_level.append(float(np.median([q[0] for q in chunk])))
            s_cv.append(float(np.median([q[1] for q in chunk])))
    return {"cells": total, "noise_level": level, "noise_sd": sd, "signal_level": s_level, "signal_cv": s_cv}


# ---------------------------------------------------------------- precision row

def precision_row(idx, abundance, retained, kept, signal, curve):
    vals = [abundance[j] for j in idx if math.isfinite(abundance[j]) and abundance[j] > 0]
    if len(vals) >= 3:
        mean = sum(vals) / len(vals)
        ss = 0.0
        for v in vals:
            ss += (v - mean) ** 2
        observed = 100.0 * math.sqrt(ss / (len(vals) - 1)) / mean
    else:
        observed = NAN
    n = len(idx)
    r_mean = sum(retained[j] for j in idx) / n
    k_mean = sum(kept[j] for j in idx) / n
    s_mean = sum(signal[j] for j in idx) / n
    expected = cv_at(curve, r_mean)
    ratio = observed / expected if math.isfinite(observed) and math.isfinite(expected) and expected > 0 else NAN
    return {"observed_cv": observed, "expected_cv": expected, "ratio": ratio,
            "retained_mean": r_mean, "kept_mean": k_mean, "signal_mean": s_mean}


# ---------------------------------------------------------------- cases

CURVE = {"level": [10.0, 100.0, 1000.0, 10000.0, 100000.0], "sd": [30.0, 80.0, 400.0, 2000.0, 12000.0]}


def matrix(rows):
    return [[s(v) for v in row] for row in rows]


def fit_case(name, obs, lib, min_fragments=3, threshold=1.0, sigma=3.0, curve=CURVE, remove=True,
             retained_src=None):
    a, r, k = fit(obs, lib, min_fragments, threshold, sigma, curve, remove, retained_src)
    return {
        "name": name,
        "observed": matrix(obs),
        "library": [s(v) for v in lib],
        "retained_source": None if retained_src is None else matrix(retained_src),
        "min_fragments": min_fragments, "threshold": s(threshold), "sigma": s(sigma), "remove": remove,
        "curve": None if curve is None else {"level": [s(v) for v in curve["level"]],
                                              "sd": [s(v) for v in curve["sd"]]},
        "expected": {"abundance": [s(v) for v in a], "retained": [s(v) for v in r], "kept": k},
    }


def fit_cases():
    lib = [1.0, 0.6, 0.35, 0.2, 0.12, 0.05]
    clean_obs = [[4000 * l, 900 * l] for l in lib]
    cases = [fit_case("no_interference_equals_sum_of_matched", clean_obs, lib)]

    strong = [row[:] for row in clean_obs]
    strong[3][0] += 6000.0  # bright sample: far beyond 2x and beyond 3 sd
    cases.append(fit_case("strong_interference_removed", strong, lib))

    dim = [[30 * l, 25 * l, 40 * l] for l in lib]
    dim[1][0] = 5 * dim[1][0]  # ~2.5x the refitted prediction, but well inside 3 sd of dim-fragment noise
    cases.append(fit_case("dim_excess_inside_noise_kept", dim, lib))
    cases.append(fit_case("gate_off_removes_on_ratio_alone", dim, lib, sigma=0.0))
    cases.append(fit_case("no_curve_means_gate_off", dim, lib, curve=None))
    cases.append(fit_case("removal_off_keeps_everything", strong, lib, remove=False))

    two = [[1000.0], [600.0], [9000.0], [8000.0]]
    cases.append(fit_case("min_fragments_limits_removals", two, [1.0, 0.6, 0.35, 0.3]))

    # Rows 2 and 3 tie exactly and only ONE removal is allowed (5 matched, min 4). Which one went is
    # visible through the retained raw signal, where the two differ.
    tie = [[1000.0], [600.0], [5000.0], [5000.0], [300.0]]
    tie_raw = [[1000.0], [600.0], [5000.0], [1234.0], [300.0]]
    cases.append(fit_case("tie_drops_first", tie, [1.0, 0.6, 0.3, 0.3, 0.3], min_fragments=4,
                          retained_src=tie_raw))

    unmatched = [[1000.0, 2000.0], [600.0, 1100.0], [7000.0, 50.0], [350.0, 720.0], [200.0, 390.0]]
    cases.append(fit_case("unmatched_rows_excluded", unmatched, [1.0, 0.6, NAN, 0.35, 0.2]))
    cases.append(fit_case("too_few_matched_is_nan", unmatched, [1.0, NAN, NAN, 0.0, 0.2]))

    floored = [[12.0, 900.0], [12.0, 520.0], [700.0, 310.0], [12.0, 180.0]]
    raw = [[0.0, 900.0], [0.0, 520.0], [700.0, 310.0], [0.0, 180.0]]
    cases.append(fit_case("retained_counts_raw_areas", floored, [1.0, 0.6, 0.35, 0.2], retained_src=raw))

    weird = [[NAN, 800.0, 950.0], [-20.0, 470.0, 560.0], [350.0, 290.0, 330.0], [210.0, 160.0, 4000.0]]
    cases.append(fit_case("nan_and_negative_areas_clean_to_zero", weird, [1.0, 0.6, 0.35, 0.2]))

    rng = np.random.default_rng(20261019)
    for i in range(6):
        m, n = 6, 8
        lib_r = sorted((float(x) for x in rng.uniform(0.05, 1.0, size=m)), reverse=True)
        level = float(rng.choice([30.0, 300.0, 3000.0, 30000.0]))
        obs = []
        for k in range(m):
            row = []
            for j in range(n):
                v = level * lib_r[k] * float(rng.lognormal(0.0, 0.3))
                if rng.uniform() < 0.25:
                    v += level * float(rng.lognormal(0.0, 1.0))
                if rng.uniform() < 0.1:
                    v = 0.0
                row.append(v)
            obs.append(row)
        cases.append(fit_case(f"random_{i}_level_{int(level)}", obs, lib_r))
    return cases


def noise_case(name, seed, n_peptides, groups, source, opts, exclude_precursor=True):
    rng = np.random.default_rng(seed)
    n_samples = 1 + max(max(g["indices"]) for g in groups)
    rows = []
    for p in range(n_peptides):
        pep = f"PEP{p:03d}K"
        level = float(rng.lognormal(7.0, 1.6))
        charges = [2] if rng.uniform() < 0.7 else [2, 3]
        for z in charges:
            m = int(rng.integers(2, 7))
            lib = rng.uniform(0.05, 1.0, size=m)
            shared = rng.lognormal(0.0, 0.15, size=n_samples)
            rows.append([pep, f"precursor_z{z}_{z}", z, 0, float(level * 3), True])
            for k in range(m):
                tid = f"y{k + 3}_z{z}_1"
                for j in range(n_samples):
                    mu = level * float(lib[k]) * float(shared[j])
                    v = mu + float(rng.normal(0.0, math.sqrt(4.0 * mu + 400.0)))
                    if rng.uniform() < 0.03:
                        v = NAN
                    rows.append([pep, tid, z, j, v, False])
                    if rng.uniform() < 0.02:
                        rows.append([pep, tid, z, j, v * 7.0, False])  # duplicate: first non-NaN wins
    out = estimate(rows, n_samples, groups, source, exclude_precursor, opts)
    return {
        "name": name, "n_samples": n_samples, "source": source, "exclude_precursor": exclude_precursor,
        "groups": groups, "options": {k: (s(v) if isinstance(v, float) else v) for k, v in opts.items()},
        "rows": [[r[0], r[1], r[2], r[3], s(r[4]), r[5]] for r in rows],
        "expected": {
            "cells": out["cells"],
            "noise_level": [s(v) for v in out["noise_level"]], "noise_sd": [s(v) for v in out["noise_sd"]],
            "signal_level": [s(v) for v in out["signal_level"]], "signal_cv": [s(v) for v in out["signal_cv"]],
        },
    }


def noise_cases():
    opts = {"fine_width": 0.05, "target_bins": 8, "min_cells_per_bin": 10, "min_signal_per_bin": 3}
    pools = [{"name": "reference", "indices": [0, 1, 2, 3, 4]}, {"name": "qc", "indices": [5, 6, 7, 8]}]
    everyone = [{"name": "all", "indices": list(range(12))}]
    return [
        noise_case("pool_replicates", 7, 40, pools, "pool_replicates", opts),
        noise_case("all_samples", 11, 40, everyone, "all_samples", opts),
    ]


def interp_cases():
    cv = {"level": [100.0, 1000.0, 10000.0], "cv": [80.0, 40.0, 15.0]}
    sd_q = [1.0, 10.0, 31.6, 100.0, 450.0, 10000.0, 1e6, 0.0]
    cv_q = [5.0, 100.0, 316.2, 1000.0, 5000.0, 10000.0, 1e7, 0.0]
    return [{
        "name": "clamped_log_interpolation",
        "noise": {"level": [s(v) for v in CURVE["level"]], "sd": [s(v) for v in CURVE["sd"]]},
        "sd_queries": [s(v) for v in sd_q],
        "sd_expected": [s(sd_at(CURVE, v)) for v in sd_q],
        "signal": {"level": [s(v) for v in cv["level"]], "cv": [s(v) for v in cv["cv"]]},
        "cv_queries": [s(v) for v in cv_q],
        "cv_expected": [s(cv_at(cv, v)) for v in cv_q],
    }]


def precision_cases():
    curve = {"level": [100.0, 1000.0, 10000.0], "cv": [80.0, 40.0, 15.0]}
    rng = np.random.default_rng(3)
    cases = []
    for i in range(4):
        n = 10
        a = [float(x) for x in rng.lognormal(7.0, 0.3, size=n)]
        if i == 1:
            a[2] = NAN
        if i == 2:
            a = [NAN] * 8 + a[8:]  # fewer than 3 usable: CV is NaN
        r = [float(x) for x in rng.uniform(50, 5000, size=n)]
        k = [int(x) for x in rng.integers(3, 7, size=n)]
        sig = [float(x) for x in rng.uniform(500, 9000, size=n)]
        idx = [0, 2, 3, 5, 6, 9] if i != 3 else [1, 4, 7]
        exp = precision_row(idx, a, r, k, sig, curve)
        cases.append({
            "name": f"row_{i}", "indices": idx,
            "abundance": [s(v) for v in a], "retained": [s(v) for v in r], "kept": k,
            "signal": [s(v) for v in sig],
            "curve": {"level": [s(v) for v in curve["level"]], "cv": [s(v) for v in curve["cv"]]},
            "expected": {key: s(v) for key, v in exp.items()},
        })
    return cases


def main():
    golden = {
        "reference": "first-principles construction: the definitions in this generator's docstring, "
                     "computed with plain Python loops and math (numpy only for seeded draws and "
                     "numpy.median); no PRISM code, no third-party implementation of the method exists",
        "note": "fit_cases pin LibraryRollup.NoiseGatedRatio (abundance, retained raw signal, kept count); "
                "noise_cases pin FragmentNoise.Estimate (noise curve and expected-CV curve) on synthetic "
                "blocks with precursor rows, NaN, negative and duplicate cells; interp_cases pin the two "
                "curve lookups; precision_cases pin LibraryAssistPrecision.Row",
        "tolerance": {"relative": "1e-12",
                      "why": "closed-form arithmetic summed in the same order; agreement is to rounding"},
        "fit_cases": fit_cases(),
        "noise_cases": noise_cases(),
        "interp_cases": interp_cases(),
        "precision_cases": precision_cases(),
    }
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "noise_gated.json"), "w", newline="\n") as f:
        json.dump(golden, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
