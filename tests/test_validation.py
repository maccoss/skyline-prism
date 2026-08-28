"""Tests for the dual-control validation verdict (skyline_prism.validation).

The verdict answers one question: did the processing DAMAGE the controls? The QC-vs-reference
improvement ratio (RVR) deliberately does not enter into it - reference and QC are different
materials injected at different amounts, so whichever started with more excess variance has more of
it to remove, and an asymmetric improvement is ordinary. These pin that, and the matching rule that
every failing condition also produces a warning, so a FAILED report always says what failed.

Mirrors the C# RvrDegenerateTests / ValidationStatusTests.
"""

import numpy as np
import pandas as pd

from skyline_prism.validation import ValidationMetrics, validate_correction


def _metrics(qc_improvement=0.10, rvr=1.0, pca_ratio=1.0):
    """A ValidationMetrics with only the fields the verdict reads varied."""
    return ValidationMetrics(
        reference_cv_before=0.20,
        reference_cv_after=0.10,
        qc_cv_before=0.20,
        qc_cv_after=0.18,
        reference_cv_improvement=0.50,
        qc_cv_improvement=qc_improvement,
        relative_variance_reduction=rvr,
        pca_qc_reference_distance_before=1.0,
        pca_qc_reference_distance_after=pca_ratio,
        pca_distance_ratio=pca_ratio,
    )


class TestVerdict:
    def test_asymmetric_improvement_passes(self):
        """QC improved 10x more than the reference - ordinary, not a failure."""
        assert _metrics(qc_improvement=0.10, rvr=10.0).passed
        assert _metrics(qc_improvement=0.10, rvr=0.05).passed

    def test_unchanged_qc_cv_passes(self):
        """Nothing was damaged, so nothing failed."""
        assert _metrics(qc_improvement=0.0).passed

    def test_worse_qc_cv_fails(self):
        assert not _metrics(qc_improvement=-0.01).passed

    def test_collapsed_controls_fail(self):
        assert not _metrics(pca_ratio=0.3).passed

    def test_unmeasured_pca_distance_is_not_a_failure(self):
        """NaN means the geometry was degenerate, not that the controls collapsed."""
        assert _metrics(pca_ratio=np.nan).passed

    def test_undefined_rvr_is_not_a_failure(self):
        assert _metrics(rvr=np.nan).passed


def _long_frame(ref_spread, qc_spread, n_peptides=40, col="abundance"):
    """Long-format frame: 2 reference + 2 QC replicates, CV within each type set by its spread.

    Values are LOG2 - `calculate_cv` back-transforms with 2**x before taking the CV (CVs are always
    computed on the linear scale). The spread is therefore a log2 offset, and the resulting linear
    CV rises monotonically with it, which is all these tests need.
    """
    rows = []
    rng = np.random.default_rng(7)
    for p in range(n_peptides):
        base = 14.0 + rng.random()
        for name, stype, spread, sign in [
            ("REF_A", "reference", ref_spread, -1),
            ("REF_B", "reference", ref_spread, +1),
            ("QC_A", "qc", qc_spread, -1),
            ("QC_B", "qc", qc_spread, +1),
        ]:
            rows.append(
                {
                    "replicate_name": name,
                    "sample_type": stype,
                    "precursor_id": f"PEP{p}",
                    col: base + sign * spread,
                }
            )
    return pd.DataFrame(rows)


class TestValidateCorrection:
    def _run(self, ref_before, ref_after, qc_before, qc_after):
        before = _long_frame(ref_before, qc_before, col="abundance")
        after = _long_frame(ref_after, qc_after, col="abundance_normalized")
        return validate_correction(before, after)

    def test_asymmetric_improvement_is_a_note_not_a_warning(self):
        # Reference barely improves, QC a lot -> a large, finite RVR.
        m = self._run(ref_before=0.20, ref_after=0.199, qc_before=0.30, qc_after=0.15)

        assert m.relative_variance_reduction > 2.0
        assert any("RVR=" in n for n in m.notes)
        assert not any("RVR" in w for w in m.warnings)
        assert not any("overfitting" in msg.lower() for msg in m.warnings + m.notes)

    def test_reference_that_does_not_improve_gives_an_undefined_rvr(self):
        # The old code forced +inf here and reported it as "QC improved much more than reference",
        # while the reference had in fact got worse.
        m = self._run(ref_before=0.20, ref_after=0.24, qc_before=0.30, qc_after=0.28)

        assert np.isnan(m.relative_variance_reduction)
        assert any("undefined" in n.lower() for n in m.notes)
        assert any("Reference CV increased" in w for w in m.warnings)

    def test_qc_degradation_is_a_warning_and_fails(self):
        m = self._run(ref_before=0.20, ref_after=0.10, qc_before=0.10, qc_after=0.30)

        assert any("QC CV increased" in w for w in m.warnings)
        assert not m.passed

    def test_a_balanced_improvement_reports_no_ratio_message(self):
        m = self._run(ref_before=0.30, ref_after=0.15, qc_before=0.30, qc_after=0.15)

        assert not any("RVR" in msg for msg in m.warnings + m.notes)
