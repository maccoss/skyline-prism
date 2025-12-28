"""Tests for transition rollup module.

Tests the quality-weighted aggregation of transitions to peptides,
median polish rollup, and variance model learning.
"""

import numpy as np
import pandas as pd
import pytest

from skyline_prism.transition_rollup import (
    TransitionRollupResult,
    VarianceModelParams,
    aggregate_transitions_weighted,
    compute_transition_variance,
    compute_transition_weights,
    learn_variance_model,
    rollup_transitions_to_peptides,
)


class TestVarianceModelParams:
    """Test VarianceModelParams dataclass."""

    def test_default_values(self):
        """Test default parameter values."""
        params = VarianceModelParams()
        assert params.alpha == 1.0
        assert params.beta == 0.01
        assert params.gamma == 100.0
        assert params.delta == 1.0
        assert params.shape_corr_exponent == 2.0
        assert params.coelution_penalty == 10.0

    def test_custom_values(self):
        """Test custom parameter initialization."""
        params = VarianceModelParams(
            alpha=2.0,
            beta=0.05,
            gamma=200.0,
            delta=1.5,
            shape_corr_exponent=3.0,
            coelution_penalty=5.0,
        )
        assert params.alpha == 2.0
        assert params.beta == 0.05
        assert params.gamma == 200.0
        assert params.delta == 1.5
        assert params.shape_corr_exponent == 3.0
        assert params.coelution_penalty == 5.0


class TestComputeTransitionVariance:
    """Test variance computation from intensity and quality metrics."""

    def test_variance_increases_with_intensity(self):
        """Higher intensity should have higher variance (for Poisson component)."""
        params = VarianceModelParams(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0)

        intensity = np.array([100.0, 1000.0, 10000.0])
        shape_corr = np.array([1.0, 1.0, 1.0])
        coeluting = np.array([True, True, True])

        variance = compute_transition_variance(intensity, shape_corr, coeluting, params)

        # With only alpha (shot noise), variance should scale linearly with intensity
        assert variance[1] > variance[0]
        assert variance[2] > variance[1]
        np.testing.assert_array_almost_equal(variance, intensity)

    def test_poor_shape_correlation_increases_variance(self):
        """Low shape correlation should increase variance."""
        params = VarianceModelParams(delta=2.0, shape_corr_exponent=2.0)

        intensity = np.array([1000.0, 1000.0, 1000.0])
        shape_corr = np.array([1.0, 0.5, 0.0])  # Perfect, medium, poor
        coeluting = np.array([True, True, True])

        variance = compute_transition_variance(intensity, shape_corr, coeluting, params)

        # Poor correlation should have highest variance
        assert variance[2] > variance[1] > variance[0]

    def test_non_coeluting_increases_variance(self):
        """Non-coeluting transitions should have higher variance."""
        params = VarianceModelParams(coelution_penalty=10.0)

        intensity = np.array([1000.0, 1000.0])
        shape_corr = np.array([1.0, 1.0])
        coeluting = np.array([True, False])

        variance = compute_transition_variance(intensity, shape_corr, coeluting, params)

        # Non-coeluting should have higher variance
        assert variance[1] > variance[0]


class TestComputeTransitionWeights:
    """Test transition weight computation."""

    def test_equal_quality_equal_weights(self):
        """Transitions with equal quality should get equal weights."""
        intensities = pd.DataFrame(
            {
                "S1": [1000.0, 1000.0, 1000.0],
                "S2": [1000.0, 1000.0, 1000.0],
            },
            index=["t1", "t2", "t3"],
        )
        shape_corrs = pd.DataFrame(1.0, index=intensities.index, columns=intensities.columns)
        coeluting = pd.DataFrame(True, index=intensities.index, columns=intensities.columns)
        params = VarianceModelParams()

        weights = compute_transition_weights(intensities, shape_corrs, coeluting, params)

        # All weights should be equal
        assert len(set(weights.values.round(6))) == 1

    def test_low_quality_lower_weight(self):
        """Transition with poor quality should get lower weight."""
        intensities = pd.DataFrame(
            {
                "S1": [1000.0, 1000.0],
                "S2": [1000.0, 1000.0],
            },
            index=["good", "poor"],
        )
        shape_corrs = pd.DataFrame(
            {"S1": [1.0, 0.3], "S2": [1.0, 0.3]},
            index=["good", "poor"],
        )
        coeluting = pd.DataFrame(True, index=intensities.index, columns=intensities.columns)
        params = VarianceModelParams(delta=2.0)

        weights = compute_transition_weights(intensities, shape_corrs, coeluting, params)

        # Poor quality transition should have lower weight
        assert weights["good"] > weights["poor"]


class TestAggregateTransitionsWeighted:
    """Test weighted aggregation of transitions."""

    def test_simple_weighted_average(self):
        """Test basic weighted average computation."""
        # Log2 intensities
        intensities = pd.DataFrame(
            {
                "S1": [10.0, 12.0, 11.0],  # Mean would be 11
                "S2": [11.0, 13.0, 12.0],  # Mean would be 12
            },
            index=["t1", "t2", "t3"],
        )
        weights = pd.Series([1.0, 1.0, 1.0], index=["t1", "t2", "t3"])

        abundances, uncertainties, n_used = aggregate_transitions_weighted(
            intensities, weights, min_transitions=2
        )

        assert n_used == 3
        np.testing.assert_almost_equal(abundances["S1"], 11.0)
        np.testing.assert_almost_equal(abundances["S2"], 12.0)

    def test_min_transitions_not_met(self):
        """Should return NaN if min_transitions not met."""
        intensities = pd.DataFrame(
            {"S1": [10.0, 12.0], "S2": [11.0, 13.0]},
            index=["t1", "t2"],
        )
        weights = pd.Series([1.0, 1.0], index=["t1", "t2"])

        abundances, uncertainties, n_used = aggregate_transitions_weighted(
            intensities, weights, min_transitions=5
        )

        assert n_used == 0
        assert np.isnan(abundances["S1"])
        assert np.isnan(abundances["S2"])


class TestRollupTransitionsToPeptides:
    """Test the main rollup function."""

    @pytest.fixture
    def sample_transition_data(self):
        """Create sample transition-level data."""
        data = pd.DataFrame(
            {
                "peptide_modified": ["PEPTIDEK"] * 6 + ["ANOTHERK"] * 4,
                "fragment_ion": ["y3", "y4", "y5", "y3", "y4", "y5", "y3", "y4", "y3", "y4"],
                "replicate_name": [
                    "S1",
                    "S1",
                    "S1",
                    "S2",
                    "S2",
                    "S2",
                    "S1",
                    "S1",
                    "S2",
                    "S2",
                ],
                "area": [
                    1000,
                    2000,
                    1500,
                    1100,
                    2100,
                    1600,  # PEPTIDEK
                    800,
                    900,
                    850,
                    950,  # ANOTHERK
                ],
                "shape_correlation": [0.95, 0.98, 0.90, 0.94, 0.97, 0.89, 0.92, 0.88, 0.91, 0.87],
                "coeluting": [True] * 10,
            }
        )
        return data

    def test_quality_weighted_rollup(self, sample_transition_data):
        """Test quality-weighted rollup method."""
        result = rollup_transitions_to_peptides(
            sample_transition_data,
            method="quality_weighted",
            min_transitions=2,
        )

        assert isinstance(result, TransitionRollupResult)
        assert "PEPTIDEK" in result.peptide_abundances.index
        assert "ANOTHERK" in result.peptide_abundances.index
        assert "S1" in result.peptide_abundances.columns
        assert "S2" in result.peptide_abundances.columns
        # Values should be present (not NaN)
        assert not np.isnan(result.peptide_abundances.loc["PEPTIDEK", "S1"])

    def test_median_polish_rollup(self, sample_transition_data):
        """Test median polish rollup method."""
        result = rollup_transitions_to_peptides(
            sample_transition_data,
            method="median_polish",
            min_transitions=2,
        )

        assert isinstance(result, TransitionRollupResult)
        # Median polish results should be available
        assert result.median_polish_results is not None
        assert "PEPTIDEK" in result.median_polish_results

    def test_sum_rollup(self, sample_transition_data):
        """Test sum rollup method."""
        result = rollup_transitions_to_peptides(
            sample_transition_data,
            method="sum",
            min_transitions=2,
        )

        assert isinstance(result, TransitionRollupResult)
        # Sum should produce values
        assert not np.isnan(result.peptide_abundances.loc["PEPTIDEK", "S1"])

    def test_unknown_method_raises(self, sample_transition_data):
        """Unknown rollup method should raise error."""
        with pytest.raises(ValueError, match="Unknown rollup method"):
            rollup_transitions_to_peptides(
                sample_transition_data,
                method="invalid_method",
            )


class TestLearnVarianceModel:
    """Test variance model learning from reference samples."""

    @pytest.fixture
    def reference_data(self):
        """Create sample data with reference samples for parameter learning."""
        np.random.seed(42)

        # Create realistic transition data with multiple peptides
        peptides = [f"PEPTIDE{i}K" for i in range(1, 11)]
        transitions = ["y3", "y4", "y5", "y6"]
        samples = ["Ref1", "Ref2", "Ref3", "Exp1", "Exp2"]

        rows = []
        for peptide in peptides:
            base_intensity = np.random.uniform(500, 5000)
            for transition in transitions:
                trans_effect = np.random.uniform(0.5, 2.0)
                for sample in samples:
                    # Add sample-to-sample variation (technical noise)
                    noise = np.random.normal(1.0, 0.1)
                    intensity = base_intensity * trans_effect * noise
                    rows.append(
                        {
                            "peptide_modified": peptide,
                            "fragment_ion": transition,
                            "replicate_name": sample,
                            "area": max(10, intensity),
                            "shape_correlation": np.random.uniform(0.8, 1.0),
                            "coeluting": np.random.random() > 0.1,
                        }
                    )

        return pd.DataFrame(rows)

    def test_learn_variance_model_basic(self, reference_data):
        """Test that variance model learning runs and returns valid parameters."""
        reference_samples = ["Ref1", "Ref2", "Ref3"]

        params = learn_variance_model(
            reference_data,
            reference_samples=reference_samples,
            n_iterations=10,  # Few iterations for speed
        )

        assert isinstance(params, VarianceModelParams)
        # All parameters should be positive
        assert params.alpha > 0
        assert params.beta > 0
        assert params.gamma > 0
        assert params.delta > 0
        assert params.shape_corr_exponent > 0
        assert params.coelution_penalty > 0

    def test_learn_variance_model_insufficient_samples(self, reference_data):
        """Should return defaults if < 2 reference samples."""
        reference_samples = ["Ref1"]  # Only one sample

        params = learn_variance_model(
            reference_data,
            reference_samples=reference_samples,
        )

        # Should return default parameters
        default = VarianceModelParams()
        assert params.alpha == default.alpha
        assert params.beta == default.beta

    def test_learn_variance_model_no_quality_columns(self):
        """Should work without quality columns (using defaults for shape/coelution)."""
        # Data without shape_correlation and coeluting columns
        data = pd.DataFrame(
            {
                "peptide_modified": ["PEPTIDEK"] * 12,
                "fragment_ion": ["y3", "y4", "y5"] * 4,
                "replicate_name": ["Ref1", "Ref1", "Ref1", "Ref2", "Ref2", "Ref2",
                                   "Ref3", "Ref3", "Ref3", "Exp1", "Exp1", "Exp1"],
                "area": np.random.uniform(1000, 5000, 12),
            }
        )

        reference_samples = ["Ref1", "Ref2", "Ref3"]

        # Should not raise, should use defaults for missing columns
        params = learn_variance_model(
            data,
            reference_samples=reference_samples,
            n_iterations=5,
        )

        assert isinstance(params, VarianceModelParams)

    def test_learned_params_improve_cv(self, reference_data):
        """Learned parameters should potentially improve (or not worsen) CV."""
        reference_samples = ["Ref1", "Ref2", "Ref3"]

        # Get CV with default parameters
        default_params = VarianceModelParams()
        default_result = rollup_transitions_to_peptides(
            reference_data[reference_data["replicate_name"].isin(reference_samples)],
            method="quality_weighted",
            params=default_params,
        )
        default_cvs = (
            default_result.peptide_abundances.std(axis=1)
            / default_result.peptide_abundances.mean(axis=1).abs()
        )
        default_median_cv = default_cvs.median()

        # Learn parameters
        learned_params = learn_variance_model(
            reference_data,
            reference_samples=reference_samples,
            n_iterations=20,
        )

        # Get CV with learned parameters
        learned_result = rollup_transitions_to_peptides(
            reference_data[reference_data["replicate_name"].isin(reference_samples)],
            method="quality_weighted",
            params=learned_params,
        )
        learned_cvs = (
            learned_result.peptide_abundances.std(axis=1)
            / learned_result.peptide_abundances.mean(axis=1).abs()
        )
        learned_median_cv = learned_cvs.median()

        # Learned CV should not be dramatically worse
        # (may not always improve due to optimization stochasticity)
        assert learned_median_cv < default_median_cv * 1.5


# =============================================================================
# Tests for new quality-weighted rollup (v2)
# =============================================================================

from skyline_prism.transition_rollup import (
    QualityWeightParams,
    QualityWeightResult,
    TransitionQualityMetrics,
    compute_quality_weights,
    compute_transition_quality_metrics,
    compute_transition_residuals,
    learn_quality_weights,
    rollup_peptide_quality_weighted,
)


class TestQualityWeightParams:
    """Test QualityWeightParams dataclass."""

    def test_default_values(self):
        """Test default parameter values."""
        params = QualityWeightParams()
        assert params.alpha == 0.5  # sqrt weighting
        assert params.beta == 1.0
        assert params.gamma == 1.0
        assert params.delta == 0.5
        assert params.shape_corr_agg == "median"
        assert params.fallback_method == "sum"
        assert params.min_improvement_pct == 5.0

    def test_custom_values(self):
        """Test custom parameter initialization."""
        params = QualityWeightParams(
            alpha=1.0,
            beta=2.0,
            gamma=0.5,
            delta=0.0,
            shape_corr_agg="mean",
            fallback_method="median_polish",
            min_improvement_pct=10.0,
        )
        assert params.alpha == 1.0
        assert params.beta == 2.0
        assert params.gamma == 0.5
        assert params.delta == 0.0
        assert params.shape_corr_agg == "mean"
        assert params.fallback_method == "median_polish"
        assert params.min_improvement_pct == 10.0


class TestComputeTransitionQualityMetrics:
    """Test quality metric computation."""

    def test_mean_intensity(self):
        """Mean intensity is computed correctly."""
        intensity = pd.DataFrame(
            {"S1": [100.0, 200.0], "S2": [150.0, 250.0]},
            index=["T1", "T2"],
        )
        shape_corr = pd.DataFrame(
            {"S1": [0.9, 0.8], "S2": [0.95, 0.85]},
            index=["T1", "T2"],
        )

        metrics = compute_transition_quality_metrics(intensity, shape_corr, "median")

        np.testing.assert_array_almost_equal(
            metrics.mean_intensity.values, [125.0, 225.0]
        )

    def test_shape_corr_median(self):
        """Median shape correlation is computed correctly."""
        intensity = pd.DataFrame(
            {"S1": [100.0, 100.0], "S2": [100.0, 100.0], "S3": [100.0, 100.0]},
            index=["T1", "T2"],
        )
        shape_corr = pd.DataFrame(
            {"S1": [0.9, 0.5], "S2": [0.8, 0.6], "S3": [0.7, 0.7]},
            index=["T1", "T2"],
        )

        metrics = compute_transition_quality_metrics(intensity, shape_corr, "median")

        np.testing.assert_array_almost_equal(
            metrics.shape_corr.values, [0.8, 0.6]
        )

    def test_shape_corr_cv(self):
        """CV of shape correlation is computed correctly."""
        intensity = pd.DataFrame(
            {"S1": [100.0, 100.0], "S2": [100.0, 100.0]},
            index=["T1", "T2"],
        )
        # T1 has low CV, T2 has high CV
        shape_corr = pd.DataFrame(
            {"S1": [0.9, 0.5], "S2": [0.9, 0.9]},
            index=["T1", "T2"],
        )

        metrics = compute_transition_quality_metrics(intensity, shape_corr, "median")

        # T1 has low CV (consistent), T2 has higher CV (variable)
        assert metrics.shape_corr_cv["T1"] < metrics.shape_corr_cv["T2"]


class TestComputeTransitionResiduals:
    """Test residual computation via median polish."""

    def test_returns_zero_for_single_transition(self):
        """Single transition should return zero residual."""
        intensity = pd.DataFrame(
            {"S1": [15.0], "S2": [16.0]},  # log2 scale
            index=["T1"],
        )

        residuals_mad = compute_transition_residuals(intensity)

        assert len(residuals_mad) == 1
        assert residuals_mad["T1"] == 0.0

    def test_outlier_has_larger_residual(self):
        """Transition with outlier value should have larger residual MAD."""
        # T1-T3 have consistent pattern across samples (perfect additive model)
        # T4 has outliers in S1 and S2 (much higher than expected from pattern)
        intensity = pd.DataFrame(
            {
                "S1": [15.0, 14.0, 13.0, 25.0],  # T4 is anomalously high
                "S2": [16.0, 15.0, 14.0, 26.0],  # T4 is anomalously high
                "S3": [17.0, 16.0, 15.0, 14.0],  # T4 is normal
                "S4": [18.0, 17.0, 16.0, 15.0],  # T4 is normal
            },
            index=["T1", "T2", "T3", "T4"],
        )

        residuals_mad = compute_transition_residuals(intensity)

        # T4 should have higher residual MAD due to the outliers
        # T1-T3 follow perfect additive model, so have zero residuals
        assert residuals_mad["T4"] > 0
        assert residuals_mad["T4"] > residuals_mad["T1"]


class TestComputeQualityWeights:
    """Test quality weight computation."""

    def test_higher_intensity_higher_weight(self):
        """Higher intensity transitions should have higher weight (alpha > 0)."""
        metrics = TransitionQualityMetrics(
            mean_intensity=pd.Series([100.0, 1000.0, 10000.0], index=["T1", "T2", "T3"]),
            shape_corr=pd.Series([1.0, 1.0, 1.0], index=["T1", "T2", "T3"]),
            shape_corr_cv=pd.Series([0.0, 0.0, 0.0], index=["T1", "T2", "T3"]),
            residuals_mad=pd.Series([0.0, 0.0, 0.0], index=["T1", "T2", "T3"]),
        )
        residuals_mad = pd.Series([0.0, 0.0, 0.0], index=["T1", "T2", "T3"])
        params = QualityWeightParams(alpha=0.5, beta=0.0, gamma=0.0, delta=0.0)

        weights = compute_quality_weights(metrics, residuals_mad, params)

        # Higher intensity = higher weight (sqrt relationship)
        assert weights["T3"] > weights["T2"] > weights["T1"]

    def test_higher_shape_corr_higher_weight(self):
        """Higher shape correlation should have higher weight."""
        metrics = TransitionQualityMetrics(
            mean_intensity=pd.Series([1000.0, 1000.0, 1000.0], index=["T1", "T2", "T3"]),
            shape_corr=pd.Series([0.5, 0.8, 1.0], index=["T1", "T2", "T3"]),
            shape_corr_cv=pd.Series([0.0, 0.0, 0.0], index=["T1", "T2", "T3"]),
            residuals_mad=pd.Series([0.0, 0.0, 0.0], index=["T1", "T2", "T3"]),
        )
        residuals_mad = pd.Series([0.0, 0.0, 0.0], index=["T1", "T2", "T3"])
        params = QualityWeightParams(alpha=0.0, beta=1.0, gamma=0.0, delta=0.0)

        weights = compute_quality_weights(metrics, residuals_mad, params)

        assert weights["T3"] > weights["T2"] > weights["T1"]

    def test_larger_residual_lower_weight(self):
        """Larger residuals should result in lower weight."""
        metrics = TransitionQualityMetrics(
            mean_intensity=pd.Series([1000.0, 1000.0, 1000.0], index=["T1", "T2", "T3"]),
            shape_corr=pd.Series([1.0, 1.0, 1.0], index=["T1", "T2", "T3"]),
            shape_corr_cv=pd.Series([0.0, 0.0, 0.0], index=["T1", "T2", "T3"]),
            residuals_mad=pd.Series([0.0, 0.0, 0.0], index=["T1", "T2", "T3"]),
        )
        residuals_mad = pd.Series([0.0, 0.5, 1.0], index=["T1", "T2", "T3"])
        params = QualityWeightParams(alpha=0.0, beta=0.0, gamma=1.0, delta=0.0)

        weights = compute_quality_weights(metrics, residuals_mad, params)

        # Larger residual = lower weight (exponential decay)
        assert weights["T1"] > weights["T2"] > weights["T3"]


class TestRollupPeptideQualityWeighted:
    """Test full quality-weighted rollup for a single peptide."""

    def test_basic_rollup(self):
        """Basic rollup produces expected outputs."""
        intensity = pd.DataFrame(
            {"S1": [15.0, 14.0, 13.0], "S2": [16.0, 15.0, 14.0]},  # log2 scale
            index=["T1", "T2", "T3"],
        )
        shape_corr = pd.DataFrame(
            {"S1": [1.0, 0.9, 0.8], "S2": [0.95, 0.85, 0.75]},
            index=["T1", "T2", "T3"],
        )
        params = QualityWeightParams()

        abundances, uncertainties, weights, n_used = rollup_peptide_quality_weighted(
            intensity, shape_corr, params, min_transitions=2
        )

        assert len(abundances) == 2  # Two samples
        assert len(weights) == 3  # Three transitions
        assert n_used == 3
        assert not abundances.isna().any()

    def test_min_transitions_not_met(self):
        """Returns NaN when minimum transitions not met."""
        intensity = pd.DataFrame(
            {"S1": [15.0, 14.0], "S2": [16.0, 15.0]},
            index=["T1", "T2"],
        )
        shape_corr = pd.DataFrame(
            {"S1": [1.0, 0.9], "S2": [0.95, 0.85]},
            index=["T1", "T2"],
        )
        params = QualityWeightParams()

        abundances, uncertainties, weights, n_used = rollup_peptide_quality_weighted(
            intensity, shape_corr, params, min_transitions=3
        )

        assert abundances.isna().all()
        assert n_used == 0


class TestLearnQualityWeights:
    """Test quality weight parameter learning."""

    @pytest.fixture
    def sample_data(self):
        """Create sample transition data with reference and QC samples."""
        np.random.seed(42)
        n_peptides = 20
        n_transitions = 5
        n_ref = 4
        n_qc = 3

        rows = []
        for pep_idx in range(n_peptides):
            peptide = f"PEP{pep_idx}"
            base_intensity = 10 ** np.random.uniform(3, 6)  # Vary peptide intensity

            for trans_idx in range(n_transitions):
                transition = f"T{trans_idx}"
                trans_factor = np.random.uniform(0.5, 1.5)  # Transition effect
                # Quality varies by transition
                base_quality = 0.7 + 0.3 * (1 - trans_idx / n_transitions)

                for rep_idx in range(n_ref):
                    sample = f"Ref{rep_idx + 1}"
                    noise = np.random.normal(1.0, 0.15)  # 15% CV
                    intensity = base_intensity * trans_factor * noise
                    shape_corr = base_quality + np.random.normal(0, 0.05)

                    rows.append({
                        "peptide_modified": peptide,
                        "fragment_ion": transition,
                        "replicate_name": sample,
                        "area": max(intensity, 1),
                        "shape_correlation": np.clip(shape_corr, 0, 1),
                    })

                for rep_idx in range(n_qc):
                    sample = f"QC{rep_idx + 1}"
                    noise = np.random.normal(1.0, 0.15)
                    intensity = base_intensity * trans_factor * noise
                    shape_corr = base_quality + np.random.normal(0, 0.05)

                    rows.append({
                        "peptide_modified": peptide,
                        "fragment_ion": transition,
                        "replicate_name": sample,
                        "area": max(intensity, 1),
                        "shape_correlation": np.clip(shape_corr, 0, 1),
                    })

        return pd.DataFrame(rows)

    def test_returns_result(self, sample_data):
        """Learning returns QualityWeightResult."""
        reference_samples = ["Ref1", "Ref2", "Ref3", "Ref4"]
        pool_samples = ["QC1", "QC2", "QC3"]

        result = learn_quality_weights(
            sample_data,
            reference_samples=reference_samples,
            pool_samples=pool_samples,
            n_iterations=5,
        )

        assert isinstance(result, QualityWeightResult)
        assert isinstance(result.params, QualityWeightParams)
        assert np.isfinite(result.reference_cv_before)
        assert np.isfinite(result.reference_cv_after)
        assert np.isfinite(result.pool_cv_before)
        assert np.isfinite(result.pool_cv_after)

    def test_insufficient_reference_samples(self, sample_data):
        """Returns fallback when insufficient reference samples."""
        reference_samples = ["Ref1"]  # Only one sample
        pool_samples = ["QC1", "QC2"]

        result = learn_quality_weights(
            sample_data,
            reference_samples=reference_samples,
            pool_samples=pool_samples,
        )

        assert result.use_quality_weights is False
        assert "Insufficient reference samples" in result.fallback_reason

    def test_learning_improves_reference_cv(self, sample_data):
        """Learned parameters should improve or maintain CV on reference."""
        reference_samples = ["Ref1", "Ref2", "Ref3", "Ref4"]
        pool_samples = ["QC1", "QC2", "QC3"]

        result = learn_quality_weights(
            sample_data,
            reference_samples=reference_samples,
            pool_samples=pool_samples,
            n_iterations=20,
        )

        # Reference CV should improve (or not worsen much)
        # Allow small degradation due to optimization stochasticity
        assert result.reference_cv_after <= result.reference_cv_before * 1.1
