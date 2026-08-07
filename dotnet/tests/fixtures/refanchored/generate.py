#!/usr/bin/env python
"""Generate cross-engine fixtures for reference-anchored ComBat.

Why this exists: standard ComBat has been pinned to R's sva since 2026-08, and the two engines
are pinned to each other by the mini end-to-end fixtures. Reference-anchored ComBat had neither -
it is PRISM's own method, so there is no external reference for it, and no fixture exercised it.
The two engines could therefore drift apart on it indefinitely without any test noticing, which
is precisely the failure mode that let the standard-path NaN bug survive.

There is no third-party implementation to check against, so these fixtures pin the PYTHON engine's
output and hold C# to it. That makes Python the reference for this one method and makes any future
divergence a visible, deliberate act rather than an accident.

Run from the repository root:
    python dotnet/tests/fixtures/refanchored/generate.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

OUT = Path("dotnet/tests/fixtures/refanchored")


def load_batch_correction():
    """Import the module directly: the package __init__ pulls in plotting/sklearn deps we
    do not need here, and this script must run in a bare numpy+pandas environment."""
    path = Path("skyline_prism/batch_correction.py")
    spec = importlib.util.spec_from_file_location("prism_batch_correction", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Rng:
    """The same LCG the C# tests use, so both sides can build identical data if they ever need
    to; here it only has to be reproducible across runs and platforms."""

    def __init__(self, seed: int) -> None:
        self._s = seed | 1

    def next_double(self) -> float:
        self._s = (self._s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return ((self._s >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    def next_gaussian(self) -> float:
        u1 = max(self.next_double(), 1e-12)
        u2 = self.next_double()
        return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def cohort(
    n_features: int,
    n_batch: int,
    n_per_batch: int,
    n_ref_per_batch: list[int],
    nan_fraction: float = 0.0,
    constant_in_batch: bool = False,
    seed: int = 20260807,
):
    rng = Rng(seed)
    batch: list[str] = []
    ref_mask: list[bool] = []
    for b in range(n_batch):
        for k in range(n_per_batch):
            batch.append(f"B{b}")
            ref_mask.append(k < n_ref_per_batch[b])

    n_samples = n_batch * n_per_batch
    data = np.zeros((n_features, n_samples))
    for f in range(n_features):
        level = 18 + 4 * rng.next_double()
        for s in range(n_samples):
            b = s // n_per_batch
            # A technical offset and a technical scale difference per batch, which is what
            # reference anchoring is supposed to find, plus noise.
            data[f, s] = level + 0.6 * b + rng.next_gaussian() * 0.3 * (1 + 0.4 * b)
            if nan_fraction > 0 and rng.next_double() < nan_fraction:
                data[f, s] = np.nan

    if constant_in_batch:
        # The case CLAUDE.md calls common in proteins_raw: flat across a whole plate.
        for f in range(0, n_features, 5):
            data[f, :n_per_batch] = 20.0

    return data, np.array(batch), np.array(ref_mask)


CASES = {
    # name: (kwargs for cohort, no_reference_batch)
    "dense": (dict(n_features=60, n_batch=4, n_per_batch=8, n_ref_per_batch=[3, 3, 3, 3]), "fallback"),
    "sparse": (
        dict(n_features=60, n_batch=4, n_per_batch=8, n_ref_per_batch=[3, 3, 3, 3], nan_fraction=0.12),
        "fallback",
    ),
    "constant_in_batch": (
        dict(n_features=60, n_batch=3, n_per_batch=8, n_ref_per_batch=[3, 3, 3], constant_in_batch=True),
        "fallback",
    ),
    # One batch with a single reference (location only) and one with none.
    "single_ref_and_fallback": (
        dict(n_features=40, n_batch=3, n_per_batch=6, n_ref_per_batch=[2, 1, 0]),
        "fallback",
    ),
    "skip_unreferenced_batch": (
        dict(n_features=40, n_batch=3, n_per_batch=6, n_ref_per_batch=[2, 0, 2]),
        "skip",
    ),
}


def main() -> None:
    bc = load_batch_correction()
    OUT.mkdir(parents=True, exist_ok=True)

    for name, (kwargs, policy) in CASES.items():
        data, batch, ref_mask = cohort(**kwargs)
        corrected = bc.combat_reference_anchored(
            data, batch, ref_mask, no_reference_batch=policy
        )

        np.savetxt(OUT / f"{name}_input.csv", data, delimiter=",", fmt="%.17g")
        np.savetxt(OUT / f"{name}_expected.csv", corrected, delimiter=",", fmt="%.17g")
        (OUT / f"{name}_batches.csv").write_text("\n".join(batch) + "\n", encoding="utf-8")
        (OUT / f"{name}_refmask.csv").write_text(
            "\n".join("1" if m else "0" for m in ref_mask) + "\n", encoding="utf-8"
        )
        (OUT / f"{name}_policy.txt").write_text(policy + "\n", encoding="utf-8")

        n_nan_in = int(np.isnan(data).sum())
        n_nan_out = int(np.isnan(np.asarray(corrected)).sum())
        print(f"{name}: {data.shape} nan in={n_nan_in} out={n_nan_out} policy={policy}")


if __name__ == "__main__":
    main()
