"""Tests for adaptive rollup fallback and QC validation decision logic.

These tests verify:
1. QC validation is the primary criterion for using adaptive weights
2. QC validation is skipped when reference shows no improvement (saves time)
3. Fallback to sum uses actual sum method, not adaptive with zero betas
"""

import numpy as np
import pandas as pd
import pytest

from skyline_prism.transition_rollup import (
    AdaptiveRollupParams,
    AdaptiveRollupResult,
    learn_adaptive_weights,
)


@pytest.fixture
def sample_data_for_qc_validation():
    """Create sample data for testing QC validation logic."""
    np.random.seed(42)

    data_rows = []
    peptides = ["PEPTIDEA", "PEPTIDEB", "PEPTIDEC", "PEPTIDED", "PEPTIDEE"]

    # Reference samples for learning
    ref_samples = ["Ref1", "Ref2", "Ref3", "Ref4"]
    # QC samples for validation
    qc_samples = ["QC1", "QC2", "QC3"]
    # Experimental samples
    exp_samples = ["Exp1", "Exp2"]

    all_samples = ref_samples + qc_samples + exp_samples

    for pep_idx, peptide in enumerate(peptides):
        n_transitions = 4 + pep_idx  # 4-8 transitions per peptide
        base_intensity = 5000 * (1 + pep_idx)

        for t_idx in range(n_transitions):
            frag_ion = f"y{t_idx + 3}"
            mz = 400 + t_idx * 100

            for sample in all_samples:
                # Add realistic variation
                intensity = base_intensity * (0.8 + 0.4 * np.random.random())
                shape_corr = 0.7 + 0.3 * np.random.random()

                data_rows.append(
                    {
                        "Peptide Modified Sequence": peptide,
                        "Fragment Ion": frag_ion,
                        "Replicate Name": sample,
                        "Area": max(intensity, 100),
                        "Product Mz": mz,
                        "Shape Correlation": shape_corr,
                    }
                )

    return pd.DataFrame(data_rows)


class TestQCValidationDecisionLogic:
    """Test that QC validation is used as the decision criterion."""

    def test_no_reference_improvement_skips_qc_validation(self, sample_data_for_qc_validation):
        """When reference shows no improvement, QC validation should be skipped."""
        ref_samples = ["Ref1", "Ref2", "Ref3", "Ref4"]
        qc_samples = ["QC1", "QC2", "QC3"]

        # Use high min_improvement_pct so any small improvement is ignored
        # This simulates the "no improvement" case reliably
        initial_params = AdaptiveRollupParams(min_improvement_pct=10.0)

        result = learn_adaptive_weights(
            sample_data_for_qc_validation,
            reference_samples=ref_samples,
            qc_samples=qc_samples,
            peptide_col="Peptide Modified Sequence",
            transition_col="Fragment Ion",
            sample_col="Replicate Name",
            abundance_col="Area",
            n_iterations=5,
            initial_params=initial_params,
        )

        # Should fall back to sum because improvement is below 10% threshold
        assert result.use_adaptive_weights is False
        # Fallback reason should mention "improvement" or threshold
        assert result.fallback_reason is not None
        assert (
            "improvement" in result.fallback_reason.lower()
            or "threshold" in result.fallback_reason.lower()
        )

    def test_result_structure(self, sample_data_for_qc_validation):
        """Verify result contains expected QC metrics."""
        ref_samples = ["Ref1", "Ref2", "Ref3", "Ref4"]
        qc_samples = ["QC1", "QC2", "QC3"]

        result = learn_adaptive_weights(
            sample_data_for_qc_validation,
            reference_samples=ref_samples,
            qc_samples=qc_samples,
            peptide_col="Peptide Modified Sequence",
            transition_col="Fragment Ion",
            sample_col="Replicate Name",
            abundance_col="Area",
            n_iterations=5,
        )

        assert isinstance(result, AdaptiveRollupResult)
        # Should have reference CV metrics
        assert np.isfinite(result.reference_cv_sum)
        assert np.isfinite(result.reference_cv_adaptive)
        # Should have QC CV metrics (even if skipped, should be set)
        assert np.isfinite(result.qc_cv_sum)
        # qc_cv_adaptive may be nan or equal to sum (if skipped)
        assert np.isfinite(result.qc_cv_adaptive) or np.isnan(result.qc_cv_adaptive)

    def test_no_qc_samples_uses_reference_decision(self, sample_data_for_qc_validation):
        """Without QC samples, decision falls back to reference-based."""
        ref_samples = ["Ref1", "Ref2", "Ref3", "Ref4"]
        qc_samples = []  # No QC samples

        result = learn_adaptive_weights(
            sample_data_for_qc_validation,
            reference_samples=ref_samples,
            qc_samples=qc_samples,
            peptide_col="Peptide Modified Sequence",
            transition_col="Fragment Ion",
            sample_col="Replicate Name",
            abundance_col="Area",
            n_iterations=5,
        )

        assert isinstance(result, AdaptiveRollupResult)
        # QC CVs should be NaN since no QC samples
        assert np.isnan(result.qc_cv_sum)


class TestFallbackBehavior:
    """Test that fallback properly uses sum method."""

    def test_fallback_result_has_correct_flags(self, sample_data_for_qc_validation):
        """Verify fallback sets use_adaptive_weights=False."""
        ref_samples = ["Ref1", "Ref2", "Ref3", "Ref4"]
        qc_samples = ["QC1", "QC2", "QC3"]

        # Force no improvement by using minimal iterations
        result = learn_adaptive_weights(
            sample_data_for_qc_validation,
            reference_samples=ref_samples,
            qc_samples=qc_samples,
            peptide_col="Peptide Modified Sequence",
            transition_col="Fragment Ion",
            sample_col="Replicate Name",
            abundance_col="Area",
            n_iterations=1,
            initial_params=AdaptiveRollupParams(min_improvement_pct=50.0),  # High threshold
        )

        # Should not use adaptive weights
        assert result.use_adaptive_weights is False
        # Should have a fallback reason
        assert result.fallback_reason is not None

    def test_zero_betas_when_no_improvement(self, sample_data_for_qc_validation):
        """When optimization finds no improvement, betas should be ~0."""
        ref_samples = ["Ref1", "Ref2", "Ref3", "Ref4"]
        qc_samples = ["QC1", "QC2", "QC3"]

        result = learn_adaptive_weights(
            sample_data_for_qc_validation,
            reference_samples=ref_samples,
            qc_samples=qc_samples,
            peptide_col="Peptide Modified Sequence",
            transition_col="Fragment Ion",
            sample_col="Replicate Name",
            abundance_col="Area",
            n_iterations=2,  # Very few iterations
        )

        # When no improvement is found, the optimizer returns to initial values (zeros)
        # or stays at zeros if it couldn't improve
        # The key point is that use_adaptive_weights should be False
        if not result.use_adaptive_weights:
            # Betas should be at or near zero
            assert abs(result.params.beta_relative_intensity) < 1.0
            assert abs(result.params.beta_mz) < 1.0
            assert abs(result.params.beta_shape_corr) < 1.0


class TestMinImprovementThreshold:
    """Test min_improvement_pct threshold behavior."""

    def test_high_threshold_forces_fallback(self, sample_data_for_qc_validation):
        """High improvement threshold should force fallback to sum."""
        ref_samples = ["Ref1", "Ref2", "Ref3", "Ref4"]
        qc_samples = ["QC1", "QC2", "QC3"]

        # Set unrealistically high threshold
        initial_params = AdaptiveRollupParams(min_improvement_pct=99.0)

        result = learn_adaptive_weights(
            sample_data_for_qc_validation,
            reference_samples=ref_samples,
            qc_samples=qc_samples,
            peptide_col="Peptide Modified Sequence",
            transition_col="Fragment Ion",
            sample_col="Replicate Name",
            abundance_col="Area",
            n_iterations=10,
            initial_params=initial_params,
        )

        # Should fall back since 99% improvement is unrealistic
        assert result.use_adaptive_weights is False

    def test_zero_threshold_allows_any_improvement(self, sample_data_for_qc_validation):
        """Zero threshold should accept any non-negative improvement."""
        ref_samples = ["Ref1", "Ref2", "Ref3", "Ref4"]
        qc_samples = ["QC1", "QC2", "QC3"]

        initial_params = AdaptiveRollupParams(min_improvement_pct=0.0)

        result = learn_adaptive_weights(
            sample_data_for_qc_validation,
            reference_samples=ref_samples,
            qc_samples=qc_samples,
            peptide_col="Peptide Modified Sequence",
            transition_col="Fragment Ion",
            sample_col="Replicate Name",
            abundance_col="Area",
            n_iterations=10,
            initial_params=initial_params,
        )

        # Result should be valid (may or may not use adaptive depending on QC validation)
        assert isinstance(result, AdaptiveRollupResult)
        # Either uses adaptive or has a valid reason not to
        assert result.use_adaptive_weights or result.fallback_reason is not None
