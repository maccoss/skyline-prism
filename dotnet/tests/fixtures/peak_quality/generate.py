#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy==2.5.3",
#     "scipy==1.18.1",
# ]
# ///
"""Generate the peak-quality golden fixtures (PRISM's `Qc/PeakQuality`) from numpy and scipy.

Which library is the reference for which quantity:

| PRISM                                   | reference                                                 |
|-----------------------------------------|-----------------------------------------------------------|
| `PeakQuality.RobustCenter` median       | `numpy.median` of the finite values                       |
| `PeakQuality.RobustCenter` scale        | `scipy.stats.median_abs_deviation(scale='normal')`        |
| `PeakQuality.ScorePrecursor` z's, score | the formulas below, on those two                          |
| per-run score median, 10th, 90th        | `numpy.nanpercentile(method='linear')`                    |
| per-run medians of the raw metrics      | `numpy.nanmedian`                                         |
| `MergedParquetReader.StreamPrecursorQuality` | `numpy.median` of each run's finite fragment values |

Nothing here imports PRISM. The score is PRISM's own definition rather than a published method, so
the reference is the definition written out a second time, independently, on top of the two library
calls that carry the statistics:

    center    = median of the precursor's finite values across runs (needs >= 3 of them)
    scale     = MAD / Phi^-1(3/4)                       (needs scale > 0)
    z, low    = max(0, (center - x) / scale)            shape correlation, isotope dot product
    z, either = |x - center| / scale                    retention time, FWHM
    score     = sqrt(mean of z^2 over the metrics with a finite z), summed in that metric order

Run from the repository root:

    uv run dotnet/tests/fixtures/peak_quality/generate.py precursors

naming the fixture(s) to rewrite (precursors, cohort, shifted, transitions); with no name all are.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy
import scipy.special
import scipy.stats

OUT = Path(__file__).resolve().parent
MIN_RUNS = 3
REFERENCE = (
    "numpy.median and scipy.stats.median_abs_deviation(scale='normal') for each precursor's "
    "center and scale; numpy.nanpercentile(method='linear') and numpy.nanmedian for the per-run "
    "summaries; numpy.median of each run's finite fragment values for the transition rollup"
)


# Every float is a STRING holding Python's shortest round-trip repr: bare NaN is not valid JSON, and
# a float routed through two encoders' number paths is only nearly preserved. See
# ../differential/generate.py, whose convention this follows.
def num(x) -> str:
    return repr(float(x))


def vec(a) -> list[str]:
    return [num(v) for v in np.asarray(a, dtype=float).ravel()]


def write(name: str, payload: dict) -> None:
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


class Rng:
    """The LCG from ../differential/generate.py: the inputs are pinned to this file, not to numpy."""

    def __init__(self, seed: int) -> None:
        self._s = seed | 1

    def next_double(self) -> float:
        self._s = (self._s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return ((self._s >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    def normal(self) -> float:
        u1 = max(self.next_double(), 1e-12)
        u2 = self.next_double()
        return float(np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2))

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.next_double()


# --------------------------------------------------------------------------------------------
# The reference score
# --------------------------------------------------------------------------------------------


def center(x) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return math.nan, math.nan, 0
    return (
        float(np.median(finite)),
        float(scipy.stats.median_abs_deviation(finite, scale="normal")),
        int(finite.size),
    )


def has_scale(c) -> bool:
    median, scale, n = c
    return n >= MIN_RUNS and math.isfinite(scale) and scale > 0


def z_low(x, c) -> list[float]:
    median, scale, _ = c
    return [max(0.0, (median - v) / scale) if has_scale(c) and math.isfinite(v) else math.nan for v in x]


def z_either(x, c) -> list[float]:
    median, scale, _ = c
    return [abs(v - median) / scale if has_scale(c) and math.isfinite(v) else math.nan for v in x]


def score_precursor(shape, dotp, rt, fwhm) -> dict:
    centers = {k: center(v) for k, v in (("shape", shape), ("dotp", dotp), ("rt", rt), ("fwhm", fwhm))}
    zs = {
        "shape": z_low(shape, centers["shape"]),
        "dotp": z_low(dotp, centers["dotp"]),
        "rt": z_either(rt, centers["rt"]),
        "fwhm": z_either(fwhm, centers["fwhm"]),
    }
    score = []
    for r in range(len(shape)):
        total, k = 0.0, 0
        for name in ("shape", "dotp", "rt", "fwhm"):
            z = zs[name][r]
            if math.isfinite(z):
                total += z * z
                k += 1
        score.append(math.sqrt(total / k) if k > 0 else math.nan)
    return {"centers": centers, "z": zs, "score": score}


def summarize(shape: np.ndarray, dotp: np.ndarray, rt: np.ndarray, fwhm: np.ndarray) -> dict:
    """Per-run summaries of a precursor x run cohort (rows = precursors)."""
    n_prec, n_runs = shape.shape
    scores = np.full((n_prec, n_runs), np.nan)
    rt_dev = np.full((n_prec, n_runs), np.nan)
    fwhm_ratio = np.full((n_prec, n_runs), np.nan)
    for p in range(n_prec):
        s = score_precursor(shape[p], dotp[p], rt[p], fwhm[p])
        scores[p] = s["score"]
        rt_c, fwhm_c = s["centers"]["rt"], s["centers"]["fwhm"]
        if rt_c[2] >= MIN_RUNS:
            rt_dev[p] = rt[p] - rt_c[0]
        if fwhm_c[2] >= MIN_RUNS and fwhm_c[0] > 0:
            fwhm_ratio[p] = fwhm[p] / fwhm_c[0]

    runs = []
    for r in range(n_runs):
        col = scores[:, r]
        runs.append({
            "scored": int(np.sum(np.isfinite(col))),
            "score_p10": num(np.nanpercentile(col, 10, method="linear")),
            "score_median": num(np.nanpercentile(col, 50, method="linear")),
            "score_p90": num(np.nanpercentile(col, 90, method="linear")),
            "shape_correlation": num(np.nanmedian(shape[:, r])),
            "isotope_dot_product": num(np.nanmedian(dotp[:, r])),
            "rt_deviation": num(np.nanmedian(rt_dev[:, r])),
            "fwhm_ratio": num(np.nanmedian(fwhm_ratio[:, r])),
        })
    medians = np.array([float(r["score_median"]) for r in runs])
    cohort = float(np.median(medians[np.isfinite(medians)]))
    return {
        "precursors": int(np.sum(np.isfinite(scores).any(axis=1))),
        "cohort_median": num(cohort),
        # 1-based positions in run order, which is the input order here.
        "flagged": [r + 1 for r in range(n_runs) if medians[r] > 2.0 * cohort],
        "runs": runs,
    }


# --------------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------------

NAN = math.nan


def gen_precursors() -> None:
    cases = []

    def add(name: str, note: str, shape, dotp, rt, fwhm) -> None:
        s = score_precursor(shape, dotp, rt, fwhm)
        cases.append({
            "name": name,
            "note": note,
            "shape": vec(shape), "dotp": vec(dotp), "rt": vec(rt), "fwhm": vec(fwhm),
            "expected": {
                "centers": {k: {"median": num(c[0]), "scale": num(c[1]), "count": c[2]}
                            for k, c in s["centers"].items()},
                "z": {k: vec(v) for k, v in s["z"].items()},
                "score": vec(s["score"]),
            },
        })

    add("typical",
        "eight runs, run 4 degraded: low shape correlation and dot product, early, narrow",
        [0.91, 0.88, 0.93, 0.62, 0.90, 0.89, 0.92, 0.87],
        [0.81, 0.79, 0.84, 0.70, 0.80, 0.83, 0.78, 0.82],
        [21.40, 21.42, 21.39, 21.28, 21.41, 21.43, 21.40, 21.38],
        [0.110, 0.104, 0.108, 0.091, 0.112, 0.107, 0.105, 0.109])
    add("high_values_are_not_penalized",
        "shape correlation and dot product ABOVE the median score zero; RT and FWHM count either way",
        [0.50, 0.55, 0.60, 0.99, 0.52],
        [0.40, 0.45, 0.42, 0.98, 0.41],
        [10.0, 10.1, 9.9, 10.6, 10.05],
        [0.05, 0.06, 0.055, 0.09, 0.052])
    add("missing_values",
        "NaN anywhere is left out of the center and of that run's score, never imputed",
        [0.90, NAN, 0.85, 0.88, 0.70, 0.91],
        [NAN, 0.60, 0.62, NAN, 0.40, 0.61],
        [30.1, 30.2, NAN, 30.15, 30.6, 30.12],
        [0.08, 0.081, 0.079, NAN, NAN, 0.082])
    add("too_few_runs",
        "a metric with fewer than three finite runs has a median but no z; the others still score",
        [0.90, NAN, 0.80, NAN, NAN],
        [0.5, 0.52, 0.48, 0.30, 0.51],
        [12.0, 12.1, 11.9, 12.4, 12.05],
        [NAN, NAN, NAN, NAN, 0.07])
    add("zero_mad_dot_product",
        "most runs carry the SAME dot product (no MS1 isotope match), so the MAD is zero and the "
        "metric is left out rather than dividing by it",
        [0.80, 0.82, 0.79, 0.81, 0.60, 0.83],
        [0.0, 0.0, 0.0, 0.0, 0.35, 0.0],
        [8.0, 8.02, 7.99, 8.01, 8.2, 8.0],
        [0.04, 0.041, 0.039, 0.04, 0.05, 0.042])
    add("even_count",
        "an even number of finite runs: the median is the mean of the middle two, as numpy takes it",
        [0.70, 0.74, 0.78, 0.66],
        [0.55, 0.50, 0.60, 0.45],
        [15.0, 15.3, 14.9, 15.1],
        [0.06, 0.07, 0.065, 0.055])
    add("run_with_nothing",
        "a run with no finite metric has no score at all",
        [0.85, 0.86, NAN, 0.84, 0.87],
        [0.6, 0.62, NAN, 0.58, 0.61],
        [5.0, 5.05, NAN, 4.95, 5.02],
        [0.03, 0.031, NAN, 0.029, 0.032])

    write("precursors.json", {
        "reference": REFERENCE,
        "note": "One precursor's values across runs -> its centers, one-sided / two-sided z's and RMS "
                "score. Closed-form arithmetic on sorted values: agrees to ~1e-15.",
        "normal_quartile": num(scipy.special.ndtri(0.75)),
        "cases": cases,
    })


def gen_cohort() -> None:
    rng = Rng(20261018)
    n_prec, n_runs, bad = 80, 12, 6  # run 7 (0-based 6) is the degraded one
    shape = np.empty((n_prec, n_runs))
    dotp = np.empty((n_prec, n_runs))
    rt = np.empty((n_prec, n_runs))
    fwhm = np.empty((n_prec, n_runs))
    for p in range(n_prec):
        base_shape, base_dotp = rng.uniform(0.6, 0.95), rng.uniform(0.3, 0.9)
        base_rt, base_fwhm = rng.uniform(5.0, 40.0), rng.uniform(0.05, 0.15)
        for r in range(n_runs):
            shape[p, r] = min(1.0, base_shape + 0.02 * rng.normal())
            dotp[p, r] = min(1.0, max(0.0, base_dotp + 0.02 * rng.normal()))
            rt[p, r] = base_rt + 0.01 * rng.normal() + 0.002 * r  # a slow drift across the run order
            fwhm[p, r] = base_fwhm * (1 + 0.03 * rng.normal())
            if rng.next_double() < 0.05:
                shape[p, r] = NAN
            if rng.next_double() < 0.05:
                rt[p, r] = NAN
        if p % 10 == 3:
            # Every run but one at exactly zero: a precursor with no MS1 isotope match.
            dotp[p, :] = 0.0
            dotp[p, p % n_runs] = 0.4
        if p % 17 == 5:
            # Seen in too few runs to center.
            fwhm[p, 2:] = NAN
        shape[p, bad] -= 0.15
        dotp[p, bad] = max(0.0, dotp[p, bad] - 0.03)
        rt[p, bad] -= 0.10
        fwhm[p, bad] *= 0.95

    write_cohort("cohort.json",
                 "80 precursors x 12 runs, run 7 degraded. Per-run summaries are the exact numpy "
                 "quantiles; PRISM reads them back from fixed-width histograms, so the test allows one "
                 "bin width per quantity. Runs are in input order (acquisition order).",
                 shape, dotp, rt, fwhm)


def gen_shifted() -> None:
    """A run far enough off that its summaries lie OUTSIDE PRISM's histogram ranges."""
    rng = Rng(4242)
    n_prec, n_runs, bad = 30, 6, 5
    shape = np.empty((n_prec, n_runs))
    dotp = np.empty((n_prec, n_runs))
    rt = np.empty((n_prec, n_runs))
    fwhm = np.empty((n_prec, n_runs))
    for p in range(n_prec):
        base_shape, base_dotp = rng.uniform(0.6, 0.95), rng.uniform(0.3, 0.9)
        base_rt, base_fwhm = rng.uniform(5.0, 40.0), rng.uniform(0.05, 0.15)
        for r in range(n_runs):
            shape[p, r] = min(1.0, base_shape + 0.02 * rng.normal())
            dotp[p, r] = min(1.0, max(0.0, base_dotp + 0.02 * rng.normal()))
            rt[p, r] = base_rt + 0.01 * rng.normal()
            fwhm[p, r] = base_fwhm * (1 + 0.03 * rng.normal())
        # Six minutes late and five times as wide: past the RT-deviation range (+-4 min), the FWHM
        # ratio range (4) and, for every precursor, the score range (40).
        rt[p, bad] += 6.0
        fwhm[p, bad] *= 5.0
    write_cohort("shifted.json",
                 "30 precursors x 6 runs; run 6 elutes 6 min late with peaks 5x as wide, so its RT "
                 "deviation, FWHM ratio and every score lie outside PRISM's histogram ranges. Those "
                 "values are kept exactly, so the test's one-bin allowance must still hold.",
                 shape, dotp, rt, fwhm)


def write_cohort(name: str, note: str, shape, dotp, rt, fwhm) -> None:
    write(name, {
        "reference": REFERENCE,
        "note": note,
        "runs": shape.shape[1],
        "shape": [vec(row) for row in shape],
        "dotp": [vec(row) for row in dotp],
        "rt": [vec(row) for row in rt],
        "fwhm": [vec(row) for row in fwhm],
        "expected": summarize(shape, dotp, rt, fwhm),
    })


def gen_transitions() -> None:
    """Transition rows as merged_data/ holds them, and the precursor x run medians they make."""
    rng = Rng(7)
    samples = ["s1", "s2", "s3", "s4", "s5"]
    rows = []
    for pep, charge, n_frag in (("PEPTIDEA", 2, 4), ("PEPTIDEA", 3, 3), ("ELVISK", 2, 5), ("LIVESK", 2, 3)):
        base = {"shape": rng.uniform(0.6, 0.95), "rt": rng.uniform(10, 30), "fwhm": rng.uniform(0.05, 0.1),
                "dotp": rng.uniform(0.3, 0.9)}
        for si, s in enumerate(samples):
            if pep == "LIVESK" and s == "s4":
                continue  # not measured in this run at all
            dotp = base["dotp"] + 0.02 * rng.normal() - (0.2 if s == "s3" else 0.0)
            # MS1 rows: present in the data, excluded from the fragment medians.
            for ion in ("precursor", "precursor [M+1]"):
                rows.append([pep, charge, ion, s, 0.999, base["rt"] + 0.5, 0.5, dotp])
            for f in range(n_frag):
                shape = base["shape"] + 0.05 * rng.normal() - (0.25 if s == "s3" else 0.0)
                rt = base["rt"] + 0.01 * rng.normal() - (0.1 if s == "s3" else 0.0)
                fwhm = base["fwhm"] * (1 + 0.05 * rng.normal())
                if pep == "ELVISK" and f == 1:
                    shape = NAN  # Skyline writes a missing shape correlation as NaN
                if pep == "PEPTIDEA" and charge == 3 and s == "s2":
                    shape = NAN  # every fragment missing: no value for this run
                rows.append([pep, charge, f"y{f + 3}", s, shape, rt, fwhm, dotp])
    # A replicate the run's sample list does not include: dropped by the join.
    rows.append(["ELVISK", 2, "y3", "s9", 0.1, 99.0, 9.0, 0.0])

    keys = sorted({(r[0], r[1]) for r in rows})
    precursors = []
    for pep, charge in keys:
        out = {"peptide": pep, "charge": charge}
        for col, i in (("shape", 4), ("rt", 5), ("fwhm", 6)):
            vals = []
            for s in samples:
                v = [r[i] for r in rows if (r[0], r[1], r[3]) == (pep, charge, s)
                     and not r[2].startswith("precursor") and math.isfinite(r[i])]
                vals.append(float(np.median(v)) if v else NAN)
            out[col] = vec(vals)
        vals = []
        for s in samples:
            v = [r[7] for r in rows if (r[0], r[1], r[3]) == (pep, charge, s) and math.isfinite(r[7])]
            vals.append(float(np.median(v)) if v else NAN)
        out["dotp"] = vec(vals)
        precursors.append(out)

    write("transitions.json", {
        "reference": REFERENCE,
        "note": "Transition rows -> each precursor's median per run over its FRAGMENT rows (MS1 rows "
                "and non-finite values left out), the dot product over all rows. Medians of a few "
                "values: exact.",
        "samples": samples,
        "columns": ["peptide", "charge", "fragment_ion", "sample", "shape_correlation",
                    "retention_time", "fwhm", "isotope_dot_product"],
        "rows": [[r[0], r[1], r[2], r[3], num(r[4]), num(r[5]), num(r[6]), num(r[7])] for r in rows],
        "expected": precursors,
    })


FIXTURES = {
    "precursors": gen_precursors,
    "cohort": gen_cohort,
    "shifted": gen_shifted,
    "transitions": gen_transitions,
}


def main(names: list[str]) -> None:
    print(f"numpy {np.__version__}, scipy {scipy.__version__}")
    for name in names or FIXTURES:
        FIXTURES[name]()


if __name__ == "__main__":
    main(sys.argv[1:])
