"""Tests for transition rollup module.

Tests the adaptive weighted aggregation of transitions to peptides,
median polish rollup, and adaptive weight learning.
"""

import numpy as np
import pandas as pd
import pytest

from skyline_prism.transition_rollup import (
    AdaptiveRollupParams,
    AdaptiveRollupResult,
    TransitionRollupResult,
    compute_adaptive_weights,
    learn_adaptive_weights,
    rollup_peptide_adaptive,
    rollup_transitions_to_peptides,
)


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
                "Product Mz": [400, 500, 600, 400, 500, 600, 450, 550, 450, 550],
            }
        )
        return data

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

    def test_adaptive_rollup(self, sample_transition_data):
        """Test adaptive rollup method."""
        result = rollup_transitions_to_peptides(
            sample_transition_data,
            method="adaptive",
            min_transitions=2,
        )

        assert isinstance(result, TransitionRollupResult)
        assert "PEPTIDEK" in result.peptide_abundances.index
        assert "ANOTHERK" in result.peptide_abundances.index
        assert "S1" in result.peptide_abundances.columns
        assert "S2" in result.peptide_abundances.columns
        # Values should be present (not NaN)
        assert not np.isnan(result.peptide_abundances.loc["PEPTIDEK", "S1"])

    def test_unknown_method_raises(self, sample_transition_data):
        """Unknown rollup method should raise error."""
        with pytest.raises(ValueError, match="Unknown rollup method"):
            rollup_transitions_to_peptides(
                sample_transition_data,
                method="invalid_method",
            )


# ============================================================================
# Tests for AdaptiveRollup
# ============================================================================


class TestAdaptiveRollupParams:
    """Test AdaptiveRollupParams dataclass."""

    def test_default_values(self):
        """Test default parameter values."""
        params = AdaptiveRollupParams()
        # Default betas are all 0 (simple sum baseline)
        assert params.beta_log_intensity == 0.0
        assert params.beta_mz == 0.0
        assert params.beta_shape_corr == 0.0
        assert params.mz_min == 0.0
        assert params.mz_max == 2000.0
        assert params.log_intensity_center == 15.0
        assert params.fallback_to_sum is True
        assert params.min_improvement_pct == 0.1  # Low threshold since optimization can't go negative

    def test_custom_values(self):
        """Test custom parameter initialization."""
        params = AdaptiveRollupParams(
            beta_log_intensity=1.0,
            beta_mz=-0.5,
            beta_shape_corr=2.0,
            mz_min=200.0,
            mz_max=1500.0,
            log_intensity_center=14.0,
            fallback_to_sum=False,
            min_improvement_pct=10.0,
        )
        assert params.beta_log_intensity == 1.0
        assert params.beta_mz == -0.5
        assert params.beta_shape_corr == 2.0
        assert params.mz_min == 200.0
        assert params.mz_max == 1500.0
        assert params.log_intensity_center == 14.0
        assert params.fallback_to_sum is False
        assert params.min_improvement_pct == 10.0


class TestComputeAdaptiveWeights:
    """Test adaptive weight computation."""

    def test_zero_betas_equal_weights(self):
        """When all betas are zero, all weights should be equal (=1)."""
        params = AdaptiveRollupParams(
            beta_log_intensity=0.0, beta_mz=0.0, beta_shape_corr=0.0
        )
        log_intensity = np.array([10.0, 12.0, 14.0, 16.0])
        mz_values = np.array([400.0, 600.0, 800.0, 1000.0])
        shape_corr = np.array([0.9, 0.95, 0.8, 0.99])

        weights = compute_adaptive_weights(log_intensity, mz_values, shape_corr, params)

        np.testing.assert_array_almost_equal(weights, [1.0, 1.0, 1.0, 1.0])

    def test_higher_intensity_higher_weight(self):
        """Higher intensity should have higher weight when beta_log_intensity > 0."""
        params = AdaptiveRollupParams(
            beta_log_intensity=1.0, beta_mz=0.0, beta_shape_corr=0.0,
            log_intensity_center=14.0,  # Center at 14
        )
        log_intensity = np.array([10.0, 14.0, 18.0])  # Below, at, and above center
        mz_values = np.array([500.0, 500.0, 500.0])
        shape_corr = np.array([0.9, 0.9, 0.9])

        weights = compute_adaptive_weights(log_intensity, mz_values, shape_corr, params)

        # Weights should increase with intensity
        assert weights[0] < weights[1] < weights[2]
        # Weight at center should be 1.0 (exp(0))
        np.testing.assert_almost_equal(weights[1], 1.0)

    def test_shape_corr_increases_weight(self):
        """Higher shape correlation should increase weight when beta_shape_corr > 0."""
        params = AdaptiveRollupParams(
            beta_log_intensity=0.0, beta_mz=0.0, beta_shape_corr=2.0
        )
        log_intensity = np.array([14.0, 14.0, 14.0])
        mz_values = np.array([500.0, 500.0, 500.0])
        shape_corr = np.array([0.5, 0.75, 1.0])

        weights = compute_adaptive_weights(log_intensity, mz_values, shape_corr, params)

        # Weights should increase with shape correlation
        assert weights[0] < weights[1] < weights[2]

    def test_mz_affects_weight(self):
        """M/z should affect weight when beta_mz != 0."""
        params = AdaptiveRollupParams(
            beta_log_intensity=0.0, beta_mz=1.0, beta_shape_corr=0.0,
            mz_min=400.0, mz_max=1000.0,  # Define range for normalization
        )
        log_intensity = np.array([14.0, 14.0, 14.0])
        mz_values = np.array([400.0, 700.0, 1000.0])  # Low, mid, high m/z
        shape_corr = np.array([0.9, 0.9, 0.9])

        weights = compute_adaptive_weights(log_intensity, mz_values, shape_corr, params)

        # With positive beta_mz, higher m/z should have higher weight
        assert weights[0] < weights[1] < weights[2]


class TestRollupPeptideAdaptive:
    """Test adaptive peptide rollup."""

    @pytest.fixture
    def sample_intensity_matrix(self):
        """Create sample intensity matrix in log2 scale."""
        return pd.DataFrame(
            {
                "Sample1": [14.0, 13.0, 12.0],
                "Sample2": [14.5, 13.5, 12.5],
                "Sample3": [15.0, 14.0, 13.0],
            },
            index=["Trans1", "Trans2", "Trans3"],
        )

    @pytest.fixture
    def sample_mz_values(self):
        """Create sample m/z values."""
        return pd.Series([400.0, 600.0, 800.0], index=["Trans1", "Trans2", "Trans3"])

    @pytest.fixture
    def sample_shape_corr_matrix(self):
        """Create sample shape correlation matrix."""
        return pd.DataFrame(
            {
                "Sample1": [0.95, 0.90, 0.85],
                "Sample2": [0.92, 0.88, 0.82],
                "Sample3": [0.98, 0.93, 0.88],
            },
            index=["Trans1", "Trans2", "Trans3"],
        )

    def test_basic_rollup(
        self, sample_intensity_matrix, sample_mz_values, sample_shape_corr_matrix
    ):
        """Test basic adaptive rollup returns expected structure."""
        params = AdaptiveRollupParams()

        abund, uncert, weights, n_used = rollup_peptide_adaptive(
            sample_intensity_matrix,
            sample_mz_values,
            sample_shape_corr_matrix,
            params,
        )

        assert isinstance(abund, pd.Series)
        assert isinstance(uncert, pd.Series)
        assert isinstance(weights, pd.Series)
        assert len(abund) == 3  # 3 samples
        assert n_used == 3  # 3 transitions

    def test_min_transitions_not_met(self):
        """Should return NaN when min_transitions not met."""
        intensity_matrix = pd.DataFrame(
            {"Sample1": [14.0, 13.0]},
            index=["Trans1", "Trans2"],
        )
        mz_values = pd.Series([400.0, 600.0], index=["Trans1", "Trans2"])
        shape_corr_matrix = pd.DataFrame(
            {"Sample1": [0.9, 0.85]},
            index=["Trans1", "Trans2"],
        )
        params = AdaptiveRollupParams()

        abund, _, _, n_used = rollup_peptide_adaptive(
            intensity_matrix, mz_values, shape_corr_matrix, params, min_transitions=3
        )

        assert all(np.isnan(abund))
        # n_used is 0 when min_transitions not met (function returns early)
        assert n_used == 0

    def test_zero_betas_equals_sum(
        self, sample_intensity_matrix, sample_mz_values, sample_shape_corr_matrix
    ):
        """When all betas are 0, result should equal simple sum."""
        params = AdaptiveRollupParams(
            beta_log_intensity=0.0, beta_mz=0.0, beta_shape_corr=0.0
        )

        abund_adaptive, _, _, _ = rollup_peptide_adaptive(
            sample_intensity_matrix,
            sample_mz_values,
            sample_shape_corr_matrix,
            params,
        )

        # Calculate expected sum (on linear scale, then back to log2)
        linear = 2**sample_intensity_matrix
        expected = np.log2(linear.sum(axis=0))

        np.testing.assert_array_almost_equal(
            abund_adaptive.values, expected.values, decimal=10
        )


class TestLearnAdaptiveWeights:
    """Test learning adaptive weight parameters."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with transitions, peptides, and samples."""
        np.random.seed(42)

        peptides = ["Pep1"] * 4 + ["Pep2"] * 3 + ["Pep3"] * 5
        transitions = (
            ["Pep1_y3", "Pep1_y4", "Pep1_y5", "Pep1_y6"]
            + ["Pep2_y3", "Pep2_y4", "Pep2_y5"]
            + ["Pep3_y3", "Pep3_y4", "Pep3_y5", "Pep3_y6", "Pep3_y7"]
        )

        samples = ["Ref1", "Ref2", "Ref3", "Ref4", "QC1", "QC2", "QC3", "Exp1", "Exp2"]
        mz_values = [400, 500, 600, 700, 450, 550, 650, 480, 580, 680, 780, 880]

        rows = []
        for i, (pep, trans) in enumerate(zip(peptides, transitions)):
            base_intensity = 10000 * (1 + i * 0.1)
            mz = mz_values[i]
            for j, sample in enumerate(samples):
                # Add some sample-specific variation
                intensity = base_intensity * (1 + np.random.normal(0, 0.1))
                shape_corr = 0.85 + np.random.uniform(0, 0.15)
                rows.append(
                    {
                        "Peptide Modified Sequence": pep,
                        "Fragment Ion": trans,
                        "Replicate Name": sample,
                        "Area": max(intensity, 100),
                        "Product Mz": mz,
                        "Shape Correlation": shape_corr,
                    }
                )

        return pd.DataFrame(rows)

    def test_returns_result(self, sample_data):
        """Should return AdaptiveRollupResult."""
        reference_samples = ["Ref1", "Ref2", "Ref3", "Ref4"]
        qc_samples = ["QC1", "QC2", "QC3"]

        result = learn_adaptive_weights(
            sample_data,
            reference_samples=reference_samples,
            qc_samples=qc_samples,
            peptide_col="Peptide Modified Sequence",
            transition_col="Fragment Ion",
            sample_col="Replicate Name",
            abundance_col="Area",
            n_iterations=5,
        )

        assert isinstance(result, AdaptiveRollupResult)
        assert isinstance(result.params, AdaptiveRollupParams)
        assert np.isfinite(result.reference_cv_sum)
        assert np.isfinite(result.reference_cv_adaptive)

    def test_insufficient_reference_samples(self, sample_data):
        """Returns fallback when insufficient reference samples."""
        reference_samples = ["Ref1"]  # Only one sample
        qc_samples = ["QC1", "QC2"]

        result = learn_adaptive_weights(
            sample_data,
            reference_samples=reference_samples,
            qc_samples=qc_samples,
            peptide_col="Peptide Modified Sequence",
            transition_col="Fragment Ion",
            sample_col="Replicate Name",
            abundance_col="Area",
        )

        assert result.use_adaptive_weights is False
        assert "reference samples" in result.fallback_reason.lower()

    def test_learned_params_constraints(self, sample_data):
        """Learned beta_relative_intensity should be >= 0 (constraint)."""
        reference_samples = ["Ref1", "Ref2", "Ref3", "Ref4"]
        qc_samples = ["QC1", "QC2", "QC3"]

        result = learn_adaptive_weights(
            sample_data,
            reference_samples=reference_samples,
            qc_samples=qc_samples,
            peptide_col="Peptide Modified Sequence",
            transition_col="Fragment Ion",
            sample_col="Replicate Name",
            abundance_col="Area",
            n_iterations=10,
        )

        # beta_relative_intensity should respect the >= 0 constraint
        assert result.params.beta_relative_intensity >= 0.0
        # beta_log_intensity and beta_sqrt_intensity are deprecated and always 0
        assert result.params.beta_log_intensity == 0.0
        assert result.params.beta_sqrt_intensity == 0.0


class TestAdaptiveZeroBetasEqualsSum:
    """Test that adaptive rollup with all betas=0 equals simple sum.

    This is a critical property: when all beta coefficients are zero,
    all transition weights should be 1.0, making the weighted sum
    equivalent to a simple sum. This provides a principled baseline
    and ensures the adaptive method degrades gracefully.
    """

    @pytest.fixture
    def multi_peptide_data(self):
        """Create multi-peptide transition data for comprehensive testing."""
        np.random.seed(12345)

        data_rows = []
        peptides = ["AAAPEPTIDEK", "BBBSEQUENCER", "CCCANALYZEK", "DDDDETECTOR"]
        samples = ["Sample_A", "Sample_B", "Sample_C", "Sample_D", "Sample_E"]

        for pep_idx, peptide in enumerate(peptides):
            n_transitions = np.random.randint(4, 8)  # 4-7 transitions per peptide
            base_intensity = 5000 * (1 + pep_idx)

            for t_idx in range(n_transitions):
                frag_ion = f"y{t_idx + 3}"
                mz = 350 + t_idx * 100 + pep_idx * 50

                for sample in samples:
                    # Vary intensity by sample with realistic noise
                    sample_factor = 0.8 + 0.4 * np.random.random()
                    trans_factor = 0.5 + np.random.random()  # Transition-specific
                    intensity = base_intensity * sample_factor * trans_factor
                    shape_corr = 0.7 + 0.3 * np.random.random()

                    data_rows.append({
                        "Peptide Modified Sequence": peptide,
                        "Fragment Ion": frag_ion,
                        "Replicate Name": sample,
                        "Area": intensity,
                        "Product Mz": mz,
                        "Shape Correlation": shape_corr,
                    })

        return pd.DataFrame(data_rows)

    def test_zero_betas_equals_sum_single_peptide(self, multi_peptide_data):
        """For a single randomly selected peptide, adaptive(betas=0) should equal sum."""
        # Pick a random peptide
        np.random.seed(42)
        peptide = np.random.choice(multi_peptide_data["Peptide Modified Sequence"].unique())
        pep_data = multi_peptide_data[
            multi_peptide_data["Peptide Modified Sequence"] == peptide
        ].copy()
        samples = pep_data["Replicate Name"].unique().tolist()

        # Method 1: Simple sum of transition areas per sample
        sum_by_sample = pep_data.groupby("Replicate Name")["Area"].sum()

        # Method 2: Adaptive rollup with all betas = 0
        # First, pivot to get intensity matrix (log2 scale)
        intensity_pivot = pep_data.pivot_table(
            index="Fragment Ion",
            columns="Replicate Name",
            values="Area",
            aggfunc="first",
        )
        intensity_log2 = np.log2(np.maximum(intensity_pivot.values, 1.0))
        intensity_matrix = pd.DataFrame(
            intensity_log2, index=intensity_pivot.index, columns=intensity_pivot.columns
        )

        # Get m/z values
        mz_pivot = pep_data.pivot_table(
            index="Fragment Ion", columns="Replicate Name", values="Product Mz", aggfunc="first"
        )
        mz_values = mz_pivot.apply(lambda x: x.dropna().iloc[0] if x.notna().any() else 0.0, axis=1)

        # Get shape correlation
        shape_pivot = pep_data.pivot_table(
            index="Fragment Ion",
            columns="Replicate Name",
            values="Shape Correlation",
            aggfunc="first",
        )
        shape_pivot = shape_pivot.reindex(index=intensity_matrix.index, columns=samples).fillna(1.0)

        # Adaptive rollup with zero betas
        params = AdaptiveRollupParams(
            beta_log_intensity=0.0,
            beta_mz=0.0,
            beta_shape_corr=0.0,
        )

        abund_adaptive, _, weights, _ = rollup_peptide_adaptive(
            intensity_matrix,
            mz_values,
            shape_pivot,
            params,
            min_transitions=1,
        )

        # All weights should be exactly 1.0
        np.testing.assert_array_almost_equal(
            weights.values, np.ones(len(weights)), decimal=10,
            err_msg="Weights should all be 1.0 when betas are 0",
        )

        # Adaptive result is log2(weighted sum of linear values)
        # With weights=1, this equals log2(sum of linear values)
        expected_log2 = np.log2(sum_by_sample)

        for sample in samples:
            np.testing.assert_almost_equal(
                abund_adaptive[sample],
                expected_log2[sample],
                decimal=6,
                err_msg=f"Adaptive(betas=0) should equal log2(sum) for sample {sample}",
            )

    def test_zero_betas_equals_sum_all_peptides(self, multi_peptide_data):
        """Verify across all peptides that adaptive(betas=0) equals sum."""
        samples = multi_peptide_data["Replicate Name"].unique().tolist()
        peptides = multi_peptide_data["Peptide Modified Sequence"].unique()

        for peptide in peptides:
            pep_data = multi_peptide_data[
                multi_peptide_data["Peptide Modified Sequence"] == peptide
            ].copy()

            # Simple sum
            sum_by_sample = pep_data.groupby("Replicate Name")["Area"].sum()

            # Adaptive with zero betas (via main rollup function)
            result = rollup_transitions_to_peptides(
                pep_data,
                method="adaptive",
                peptide_col="Peptide Modified Sequence",
                transition_col="Fragment Ion",
                sample_col="Replicate Name",
                abundance_col="Area",
                adaptive_params=AdaptiveRollupParams(
                    beta_log_intensity=0.0,
                    beta_mz=0.0,
                    beta_shape_corr=0.0,
                ),
                min_transitions=1,
            )

            # Result is in log2 scale
            expected_log2 = np.log2(sum_by_sample)

            for sample in samples:
                adaptive_val = result.peptide_abundances.loc[peptide, sample]
                expected_val = expected_log2[sample]
                np.testing.assert_almost_equal(
                    adaptive_val,
                    expected_val,
                    decimal=5,
                    err_msg=f"Mismatch for peptide {peptide}, sample {sample}",
                )
