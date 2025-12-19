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
