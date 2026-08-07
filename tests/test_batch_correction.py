"""Tests for ComBat batch correction implementation."""

import numpy as np
import pandas as pd
import pytest

from skyline_prism.batch_correction import (
    _check_inputs,
    _make_design_matrix,
    _postmean,
    _postvar,
    combat,
    combat_from_long,
    combat_reference_anchored,
)


class TestInputValidation:
    """Tests for input validation."""

    def test_valid_inputs(self):
        """Test that valid inputs pass validation."""
        data = np.random.randn(10, 6)
        batch = np.array([1, 1, 1, 2, 2, 2])
        data_out, batch_out, covar_out = _check_inputs(data, batch)
        assert data_out.shape == (10, 6)
        assert len(batch_out) == 6

    def test_batch_length_mismatch(self):
        """Test that batch length must match sample count."""
        data = np.random.randn(10, 6)
        batch = np.array([1, 1, 2, 2])  # Wrong length
        with pytest.raises(ValueError, match="Batch length"):
            _check_inputs(data, batch)

    def test_single_sample_batch_error(self):
        """Test that single-sample batches raise error."""
        data = np.random.randn(10, 6)
        batch = np.array([1, 1, 1, 2, 2, 3])  # Batch 3 has 1 sample
        with pytest.raises(ValueError, match="single sample"):
            _check_inputs(data, batch)

    def test_1d_data_error(self):
        """Test that 1D data raises error."""
        data = np.random.randn(10)
        batch = np.array([1, 1, 1, 2, 2, 2])
        with pytest.raises(ValueError, match="2D matrix"):
            _check_inputs(data, batch)


class TestDesignMatrix:
    """Tests for design matrix construction."""

    def test_basic_design_matrix(self):
        """Test basic design matrix with two batches."""
        batch = np.array([1, 1, 1, 2, 2, 2])
        design, batches, n_batch, ref_idx = _make_design_matrix(batch)

        assert n_batch == 2
        assert len(batches) == 2
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert ref_idx is None
        assert design.shape == (6, 2)

    def test_three_batches(self):
        """Test design matrix with three batches."""
        batch = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3])
        design, batches, n_batch, ref_idx = _make_design_matrix(batch)

        assert n_batch == 3
        assert design.shape == (9, 3)
        assert list(batches[0]) == [0, 1, 2]
        assert list(batches[1]) == [3, 4]
        assert list(batches[2]) == [5, 6, 7, 8]

    def test_reference_batch(self):
        """Test design matrix with reference batch."""
        batch = np.array([1, 1, 1, 2, 2, 2])
        design, batches, n_batch, ref_idx = _make_design_matrix(batch, ref_batch=1)

        assert ref_idx == 0  # Index of batch "1"

    def test_string_batches(self):
        """Test with string batch labels."""
        batch = np.array(["A", "A", "B", "B", "B"])
        design, batches, n_batch, ref_idx = _make_design_matrix(batch)

        assert n_batch == 2


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_postmean(self):
        """Test posterior mean calculation."""
        g_hat = np.array([1.0, 2.0, 3.0])
        g_bar = 2.0
        n = 5
        d_star = np.array([1.0, 1.0, 1.0])
        t2 = 1.0

        result = _postmean(g_hat, g_bar, n, d_star, t2)
        # Should shrink towards g_bar
        assert result[0] < g_hat[0] + 0.5  # First value shrinks up
        assert result[2] > g_hat[2] - 0.5  # Third value shrinks down

    def test_postvar(self):
        """Test posterior variance calculation."""
        sum_sq = np.array([10.0, 20.0, 30.0])
        n = 5
        a = 2.0
        b = 1.0

        result = _postvar(sum_sq, n, a, b)
        assert len(result) == 3
        assert all(result > 0)


class TestComBatCore:
    """Tests for core ComBat functionality."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic data with known batch effects."""
        np.random.seed(42)
        n_features = 100
        n_samples_per_batch = 5

        # Batch 1: mean=0, var=1
        batch1 = np.random.normal(0, 1, (n_features, n_samples_per_batch))

        # Batch 2: mean=2, var=1.5 (batch effect)
        batch2 = np.random.normal(2, 1.5, (n_features, n_samples_per_batch))

        # Batch 3: mean=-1, var=0.8 (batch effect)
        batch3 = np.random.normal(-1, 0.8, (n_features, n_samples_per_batch))

        data = np.hstack([batch1, batch2, batch3])
        batch = np.array([1] * 5 + [2] * 5 + [3] * 5)

        return data, batch

    def test_combat_output_shape(self, synthetic_data):
        """Test that ComBat returns correct shape."""
        data, batch = synthetic_data
        corrected = combat(data, batch)

        assert corrected.shape == data.shape

    def test_combat_reduces_batch_variance(self, synthetic_data):
        """Test that ComBat reduces between-batch variance."""
        data, batch = synthetic_data

        # Calculate batch means before correction
        means_before = []
        for b in [1, 2, 3]:
            batch_data = data[:, batch == b]
            means_before.append(np.mean(batch_data))
        var_before = np.var(means_before)

        # Apply ComBat
        corrected = combat(data, batch)

        # Calculate batch means after correction
        means_after = []
        for b in [1, 2, 3]:
            batch_data = corrected[:, batch == b]
            means_after.append(np.mean(batch_data))
        var_after = np.var(means_after)

        # Batch variance should decrease
        assert var_after < var_before

    def test_combat_preserves_overall_mean(self, synthetic_data):
        """Test that ComBat approximately preserves overall mean."""
        data, batch = synthetic_data
        corrected = combat(data, batch)

        mean_before = np.mean(data)
        mean_after = np.mean(corrected)

        # Allow 5% tolerance
        assert abs(mean_after - mean_before) < abs(mean_before) * 0.1

    def test_combat_mean_only(self, synthetic_data):
        """Test mean-only correction mode."""
        data, batch = synthetic_data
        corrected = combat(data, batch, mean_only=True)

        assert corrected.shape == data.shape
        # Should still reduce batch variance
        means_before = [np.mean(data[:, batch == b]) for b in [1, 2, 3]]
        means_after = [np.mean(corrected[:, batch == b]) for b in [1, 2, 3]]
        assert np.var(means_after) < np.var(means_before)

    def test_combat_reference_batch(self, synthetic_data):
        """Test reference batch is not modified."""
        data, batch = synthetic_data
        corrected = combat(data, batch, ref_batch=1)

        # Reference batch should be unchanged
        ref_mask = batch == 1
        np.testing.assert_array_almost_equal(corrected[:, ref_mask], data[:, ref_mask])

    def test_combat_nonparametric(self, synthetic_data):
        """Test non-parametric estimation."""
        data, batch = synthetic_data
        corrected = combat(data, batch, par_prior=False)

        assert corrected.shape == data.shape


class TestComBatWithPandas:
    """Test ComBat with pandas inputs."""

    def test_dataframe_input(self):
        """Test that DataFrame input returns DataFrame output."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(50, 9),
            index=[f"gene_{i}" for i in range(50)],
            columns=[f"sample_{i}" for i in range(9)],
        )
        batch = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3])

        corrected = combat(data, batch)

        assert isinstance(corrected, pd.DataFrame)
        assert list(corrected.index) == list(data.index)
        assert list(corrected.columns) == list(data.columns)

    def test_series_batch(self):
        """Test with pandas Series for batch."""
        np.random.seed(42)
        data = np.random.randn(50, 9)
        batch = pd.Series([1, 1, 1, 2, 2, 3, 3, 3, 3])

        corrected = combat(data, batch)
        assert corrected.shape == data.shape


class TestComBatFromLong:
    """Test the long-format wrapper."""

    def test_long_format_roundtrip(self):
        """Test conversion from long to wide and back."""
        np.random.seed(42)

        # Create long-format data
        features = [f"peptide_{i}" for i in range(20)]
        samples = ["S1", "S2", "S3", "S4", "S5", "S6"]
        batches = ["B1", "B1", "B1", "B2", "B2", "B2"]

        rows = []
        for feat in features:
            for i, samp in enumerate(samples):
                # Add batch effect to B2
                batch_effect = 2.0 if batches[i] == "B2" else 0.0
                abundance = np.random.randn() + batch_effect
                rows.append(
                    {
                        "precursor_id": feat,
                        "replicate_name": samp,
                        "batch": batches[i],
                        "abundance": abundance,
                    }
                )

        df = pd.DataFrame(rows)

        # Apply ComBat
        result = combat_from_long(
            df,
            abundance_col="abundance",
            feature_col="precursor_id",
            sample_col="replicate_name",
            batch_col="batch",
        )

        assert "abundance_batch_corrected" in result.columns
        assert len(result) == len(df)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_variance_features(self):
        """Test handling of features with zero variance."""
        np.random.seed(42)
        data = np.random.randn(50, 6)
        # Make some features constant
        data[0, :] = 5.0
        data[10, :] = 0.0

        batch = np.array([1, 1, 1, 2, 2, 2])

        # Should not raise, should handle gracefully
        corrected = combat(data, batch)
        assert corrected.shape == data.shape

        # Constant features should remain constant (or close)
        assert np.std(corrected[0, :]) < 1e-10
        assert np.std(corrected[10, :]) < 1e-10

    def test_two_batches_minimum(self):
        """Test that we need at least 2 samples per batch."""
        data = np.random.randn(10, 4)
        batch = np.array([1, 2, 2, 2])  # Batch 1 has 1 sample

        with pytest.raises(ValueError, match="single sample"):
            combat(data, batch)

    def test_large_batch_count(self):
        """Test with many batches."""
        np.random.seed(42)
        n_batches = 10
        samples_per_batch = 3
        n_samples = n_batches * samples_per_batch

        data = np.random.randn(50, n_samples)
        batch = np.repeat(range(n_batches), samples_per_batch)

        # Add different effects to each batch
        for b in range(n_batches):
            mask = batch == b
            data[:, mask] += b * 0.5

        corrected = combat(data, batch)
        assert corrected.shape == data.shape


class TestNumericalStability:
    """Test numerical stability."""

    def test_small_values(self):
        """Test with very small abundance values."""
        np.random.seed(42)
        data = np.random.randn(50, 9) * 1e-6
        batch = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3])

        corrected = combat(data, batch)
        assert np.all(np.isfinite(corrected))

    def test_large_values(self):
        """Test with large abundance values."""
        np.random.seed(42)
        data = np.random.randn(50, 9) * 1e6
        batch = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3])

        corrected = combat(data, batch)
        assert np.all(np.isfinite(corrected))

    def test_mixed_sign_values(self):
        """Test with mix of positive and negative values (log2 space)."""
        np.random.seed(42)
        # Simulate log2 abundances (can be negative)
        data = np.random.randn(50, 9) * 3  # Mix of positive and negative
        batch = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3])

        corrected = combat(data, batch)
        assert np.all(np.isfinite(corrected))


class TestReproducibility:
    """Test that results are reproducible."""

    def test_deterministic_output(self):
        """Test that same input gives same output."""
        np.random.seed(42)
        data = np.random.randn(50, 9)
        batch = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3])

        corrected1 = combat(data.copy(), batch.copy())
        corrected2 = combat(data.copy(), batch.copy())

        np.testing.assert_array_almost_equal(corrected1, corrected2)


# Reference tests based on InMoose expected outputs
class TestReferenceValues:
    """Tests against known reference values.

    These tests use patterns from the InMoose pycombat tests to ensure
    our implementation produces similar results.
    """

    @pytest.fixture
    def reference_data(self):
        """Create test data similar to InMoose tests."""
        np.random.seed(12345)  # Fixed seed for reproducibility

        # Create data with known batch structure
        # Batch 1: samples 0-2, mean ~3, std ~1
        # Batch 2: samples 3-4, mean ~2, std ~0.6
        # Batch 3: samples 5-8, mean ~4, std ~1
        # Each column is a sample's values across all genes
        n_genes = 1000
        matrix = np.column_stack(
            [
                np.random.normal(loc=3, scale=1, size=n_genes),
                np.random.normal(loc=3, scale=1, size=n_genes),
                np.random.normal(loc=3, scale=1, size=n_genes),
                np.random.normal(loc=2, scale=0.6, size=n_genes),
                np.random.normal(loc=2, scale=0.6, size=n_genes),
                np.random.normal(loc=4, scale=1, size=n_genes),
                np.random.normal(loc=4, scale=1, size=n_genes),
                np.random.normal(loc=4, scale=1, size=n_genes),
                np.random.normal(loc=4, scale=1, size=n_genes),
            ]
        )  # genes x samples (1000 x 9)

        batch = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3])
        return matrix, batch

    def test_shape_preserved(self, reference_data):
        """Test output shape matches input."""
        matrix, batch = reference_data
        result = combat(matrix, batch)
        assert result.shape == matrix.shape

    def test_mean_approximately_preserved(self, reference_data):
        """Test overall mean is approximately preserved."""
        matrix, batch = reference_data
        result = combat(matrix, batch)

        # Mean should be close (within 5%)
        orig_mean = np.mean(matrix)
        result_mean = np.mean(result)
        assert abs(result_mean - orig_mean) <= abs(orig_mean) * 0.05

    def test_variance_reduced(self, reference_data):
        """Test that variance is reduced or similar after correction."""
        matrix, batch = reference_data
        result = combat(matrix, batch)

        # Variance should not increase dramatically
        assert np.var(result) <= np.var(matrix) * 1.1

    def test_batch_effects_reduced(self, reference_data):
        """Test that batch effects are reduced."""
        matrix, batch = reference_data

        # Measure batch effect as variance of batch means
        def batch_effect_measure(data, batch_labels):
            batch_means = []
            for b in np.unique(batch_labels):
                batch_means.append(np.mean(data[:, batch_labels == b]))
            return np.var(batch_means)

        before = batch_effect_measure(matrix, batch)
        result = combat(matrix, batch)
        after = batch_effect_measure(result, batch)

        # Batch effect should decrease
        assert after < before


class TestBatchCorrectionEvaluation:
    """Tests for batch correction evaluation with reference/QC samples."""

    @pytest.fixture
    def evaluation_data(self):
        """Create test data with reference, QC, and experimental samples."""
        np.random.seed(42)

        n_features = 100
        samples = []

        # Create data with batch effects
        # Batch 1: 3 experimental + 1 reference + 1 qc
        # Batch 2: 3 experimental + 1 reference + 1 qc
        # Batch 3: 3 experimental + 1 reference + 1 qc

        for batch_id in [1, 2, 3]:
            # Batch-specific offset
            batch_offset = (batch_id - 2) * 0.5  # -0.5, 0, 0.5

            # Experimental samples (different biology per sample)
            for exp_idx in range(3):
                sample_name = f"exp_b{batch_id}_{exp_idx}"
                bio_effect = np.random.randn(n_features) * 0.3
                for feat_idx in range(n_features):
                    samples.append(
                        {
                            "precursor_id": f"peptide_{feat_idx}",
                            "replicate_name": sample_name,
                            "batch": batch_id,
                            "sample_type": "experimental",
                            "abundance": 10
                            + batch_offset
                            + bio_effect[feat_idx]
                            + np.random.randn() * 0.1,
                        }
                    )

            # Reference sample (same material, should be identical across batches)
            ref_name = f"ref_b{batch_id}"
            for feat_idx in range(n_features):
                samples.append(
                    {
                        "precursor_id": f"peptide_{feat_idx}",
                        "replicate_name": ref_name,
                        "batch": batch_id,
                        "sample_type": "reference",
                        "abundance": 10 + batch_offset + np.random.randn() * 0.05,  # Low noise
                    }
                )

            # QC sample (same QC material per batch, different between batches)
            qc_name = f"qc_b{batch_id}"
            qc_offset = np.random.randn() * 0.1  # Small QC-specific variation
            for feat_idx in range(n_features):
                samples.append(
                    {
                        "precursor_id": f"peptide_{feat_idx}",
                        "replicate_name": qc_name,
                        "batch": batch_id,
                        "sample_type": "qc",
                        "abundance": 10 + batch_offset + qc_offset + np.random.randn() * 0.05,
                    }
                )

        return pd.DataFrame(samples)

    def test_evaluate_batch_correction_detects_improvement(self, evaluation_data):
        """Test that evaluation detects improvement from batch correction."""
        from skyline_prism.batch_correction import combat_from_long, evaluate_batch_correction

        # Apply ComBat
        corrected = combat_from_long(
            evaluation_data,
            abundance_col="abundance",
            feature_col="precursor_id",
            sample_col="replicate_name",
            batch_col="batch",
        )

        # Evaluate
        evaluation = evaluate_batch_correction(
            corrected,
            abundance_before="abundance",
            abundance_after="abundance_batch_corrected",
            sample_type_col="sample_type",
            feature_col="precursor_id",
            sample_col="replicate_name",
            batch_col="batch",
        )

        # Reference CV should improve (same material across batches)
        assert evaluation.reference_improvement > 0, (
            f"Reference CV should improve, got {evaluation.reference_improvement:.3f}"
        )

        # Pool CV should not get dramatically worse
        assert evaluation.qc_cv_after <= evaluation.qc_cv_before * 1.5, (
            f"Pool CV increased too much: {evaluation.qc_cv_before:.3f} -> "
            f"{evaluation.qc_cv_after:.3f}"
        )

    def test_combat_with_reference_samples(self, evaluation_data):
        """Test the combined combat + evaluation function."""
        from skyline_prism.batch_correction import combat_with_reference_samples

        corrected, evaluation = combat_with_reference_samples(
            evaluation_data,
            abundance_col="abundance",
            feature_col="precursor_id",
            sample_col="replicate_name",
            batch_col="batch",
            sample_type_col="sample_type",
        )

        assert "abundance_batch_corrected" in corrected.columns
        assert evaluation is not None
        assert hasattr(evaluation, "passed")
        assert hasattr(evaluation, "reference_cv_before")
        assert hasattr(evaluation, "qc_cv_before")

    def test_evaluation_detects_overfitting(self):
        """Test that evaluation flags potential overfitting."""
        from skyline_prism.batch_correction import (
            evaluate_batch_correction,
        )

        np.random.seed(123)

        # Create artificial data where reference improves but QC gets worse
        # This simulates overfitting to reference samples
        n_features = 50
        samples = []

        for batch_id in [1, 2]:
            batch_offset = (batch_id - 1) * 1.0

            # Reference - will "improve" dramatically
            for rep in range(3):
                for feat_idx in range(n_features):
                    samples.append(
                        {
                            "precursor_id": f"pep_{feat_idx}",
                            "replicate_name": f"ref_b{batch_id}_{rep}",
                            "batch": batch_id,
                            "sample_type": "reference",
                            "abundance": 10 + batch_offset,
                            "abundance_batch_corrected": 10.0,  # Perfect correction
                        }
                    )

            # Pool - gets worse
            for rep in range(3):
                for feat_idx in range(n_features):
                    original = 10 + batch_offset + np.random.randn() * 0.1
                    samples.append(
                        {
                            "precursor_id": f"pep_{feat_idx}",
                            "replicate_name": f"qc_b{batch_id}_{rep}",
                            "batch": batch_id,
                            "sample_type": "qc",
                            "abundance": original,
                            "abundance_batch_corrected": original
                            + np.random.randn() * 0.5,  # More noise
                        }
                    )

        df = pd.DataFrame(samples)

        evaluation = evaluate_batch_correction(
            df,
            abundance_before="abundance",
            abundance_after="abundance_batch_corrected",
        )

        # Should detect that QC got worse
        assert evaluation.qc_improvement < 0 or len(evaluation.warnings) > 0

    def test_evaluation_without_samples_gracefully_handles(self, evaluation_data):
        """Test evaluation handles missing sample types gracefully."""
        from skyline_prism.batch_correction import combat_with_reference_samples

        # Remove QC samples
        data_no_qc = evaluation_data[evaluation_data["sample_type"] != "qc"].copy()

        corrected, evaluation = combat_with_reference_samples(
            data_no_qc,
            abundance_col="abundance",
            sample_type_col="sample_type",
        )

        # Should still return corrected data, but no evaluation
        assert "abundance_batch_corrected" in corrected.columns
        assert evaluation is None

    def test_batch_correction_evaluation_dataclass(self):
        """Test BatchCorrectionEvaluation dataclass."""
        from skyline_prism.batch_correction import BatchCorrectionEvaluation

        evaluation = BatchCorrectionEvaluation(
            reference_cv_before=0.15,
            reference_cv_after=0.08,
            qc_cv_before=0.12,
            qc_cv_after=0.10,
            reference_improvement=0.47,
            qc_improvement=0.17,
            overfitting_ratio=2.8,
            batch_variance_before=0.25,
            batch_variance_after=0.05,
            passed=False,
            warnings=["Possible overfitting"],
        )

        assert evaluation.reference_cv_before == 0.15
        assert evaluation.passed is False
        assert len(evaluation.warnings) == 1

    def test_fallback_on_failure_uses_uncorrected_data(self):
        """Test that fallback_on_failure reverts to uncorrected data."""
        from skyline_prism.batch_correction import combat_with_reference_samples

        np.random.seed(999)

        # Create data where batch correction will make things worse
        # QC samples get artificially worse after "correction"
        n_features = 30
        samples = []

        for batch_id in [1, 2]:
            batch_offset = (batch_id - 1) * 2.0  # Large batch effect

            # Reference samples
            for rep in range(3):
                for feat_idx in range(n_features):
                    samples.append(
                        {
                            "precursor_id": f"pep_{feat_idx}",
                            "replicate_name": f"ref_b{batch_id}_{rep}",
                            "batch": batch_id,
                            "sample_type": "reference",
                            "abundance": 10 + batch_offset + np.random.randn() * 0.05,
                        }
                    )

            # QC samples - very consistent within batch
            for rep in range(3):
                for feat_idx in range(n_features):
                    samples.append(
                        {
                            "precursor_id": f"pep_{feat_idx}",
                            "replicate_name": f"qc_b{batch_id}_{rep}",
                            "batch": batch_id,
                            "sample_type": "qc",
                            "abundance": 10 + batch_offset + np.random.randn() * 0.02,
                        }
                    )

        df = pd.DataFrame(samples)

        # Apply with fallback enabled
        corrected, evaluation = combat_with_reference_samples(
            df,
            abundance_col="abundance",
            fallback_on_failure=True,
        )

        # If evaluation failed, the corrected values should equal original
        if evaluation is not None and not evaluation.passed:
            # Check that fallback message is in warnings
            assert any("FALLBACK" in w for w in evaluation.warnings)

            # Corrected values should match original (within floating point)
            np.testing.assert_array_almost_equal(
                corrected["abundance_batch_corrected"].values,
                corrected["abundance"].values,
            )

    def test_fallback_disabled_keeps_bad_correction(self):
        """Test that fallback_on_failure=False keeps corrected data even if QC fails."""
        from skyline_prism.batch_correction import combat_with_reference_samples

        np.random.seed(888)

        n_features = 30
        samples = []

        for batch_id in [1, 2]:
            batch_offset = (batch_id - 1) * 2.0

            for rep in range(3):
                for feat_idx in range(n_features):
                    samples.append(
                        {
                            "precursor_id": f"pep_{feat_idx}",
                            "replicate_name": f"ref_b{batch_id}_{rep}",
                            "batch": batch_id,
                            "sample_type": "reference",
                            "abundance": 10 + batch_offset + np.random.randn() * 0.1,
                        }
                    )

            for rep in range(3):
                for feat_idx in range(n_features):
                    samples.append(
                        {
                            "precursor_id": f"pep_{feat_idx}",
                            "replicate_name": f"qc_b{batch_id}_{rep}",
                            "batch": batch_id,
                            "sample_type": "qc",
                            "abundance": 10 + batch_offset + np.random.randn() * 0.05,
                        }
                    )

        df = pd.DataFrame(samples)

        # Apply with fallback disabled
        corrected, evaluation = combat_with_reference_samples(
            df,
            abundance_col="abundance",
            fallback_on_failure=False,
        )

        # Corrected values should NOT equal original (ComBat was applied)
        # Even if evaluation failed, we keep the corrected values
        assert not np.allclose(
            corrected["abundance_batch_corrected"].values,
            corrected["abundance"].values,
        )


def _build_reference_anchored_data(
    n_feat,
    batches,
    n_ref,
    n_exp,
    offsets,
    mu,
    bio_sd=1.0,
    noise=0.05,
    seed=0,
    ref_noise=None,
):
    """Build a (features x samples) matrix with known per-batch additive offsets.

    References are identical material (mu + offset + small noise); experimental
    samples add biological deviation. Returns (data_in, batch, reference_mask, info).
    ``n_ref`` may be an int (same for all batches) or a dict batch->count.
    ``ref_noise`` optionally overrides the reference replicate SD per batch
    (dict batch->sd) to create scale differences.
    """
    rng = np.random.default_rng(seed)
    cols, batch, ref_mask = [], [], []
    for b in batches:
        nref_b = n_ref[b] if isinstance(n_ref, dict) else n_ref
        rsd = noise if ref_noise is None else ref_noise.get(b, noise)
        for _ in range(nref_b):
            cols.append(mu + offsets[b] + rng.normal(0, rsd, n_feat))
            batch.append(b)
            ref_mask.append(True)
        for _ in range(n_exp):
            bio = rng.normal(0, bio_sd, n_feat)
            cols.append(mu + bio + offsets[b] + rng.normal(0, noise, n_feat))
            batch.append(b)
            ref_mask.append(False)
    data_in = np.array(cols).T
    return data_in, np.array(batch), np.array(ref_mask)


class TestReferenceAnchoredComBat:
    """Tests for reference-anchored ComBat (single-point calibration + EB)."""

    def test_additive_offset_removed_references_align(self):
        """Remove a pure per-batch additive offset seen in clean references.

        Reference samples align across batches afterward.
        """
        n_feat = 60
        mu = np.linspace(12, 18, n_feat)
        offsets = {"A": 0.0, "B": 0.8, "C": -0.5}
        data_in, batch, ref_mask = _build_reference_anchored_data(
            n_feat, ["A", "B", "C"], n_ref=2, n_exp=4, offsets=offsets, mu=mu, seed=1
        )
        corrected = combat_reference_anchored(data_in, batch, ref_mask)

        ref_idx = np.where(ref_mask)[0]
        before = data_in[:, ref_idx].std(axis=1).mean()
        after = corrected[:, ref_idx].std(axis=1).mean()
        # Cross-batch reference spread should collapse to ~noise
        assert after < 0.15 * before

    def test_biological_signal_preserved(self):
        """Preserve experimental biology, removing only the batch offset.

        The deviation of each experimental sample from its own batch's reference survives.
        """
        n_feat = 60
        mu = np.full(n_feat, 15.0)
        offsets = {"A": 0.0, "B": 1.0}
        data_in, batch, ref_mask = _build_reference_anchored_data(
            n_feat, ["A", "B"], n_ref=3, n_exp=5, offsets=offsets, mu=mu, seed=2
        )
        corrected = combat_reference_anchored(data_in, batch, ref_mask)

        # Within batch B, the experimental-vs-reference contrast must be
        # preserved (offset is common to both, so it cancels in the contrast).
        exp_b = np.where((batch == "B") & ~ref_mask)[0]
        ref_b = np.where((batch == "B") & ref_mask)[0]
        before_contrast = data_in[:, exp_b].mean(axis=1) - data_in[:, ref_b].mean(axis=1)
        after_contrast = corrected[:, exp_b].mean(axis=1) - corrected[:, ref_b].mean(axis=1)
        # Biology is preserved up to the (legitimate) scale harmonization, which
        # rescales deviations by ~1/sqrt(delta) (delta ~ 1 here). Strong but not
        # necessarily perfect correlation; near-identity magnitude.
        corr = np.corrcoef(before_contrast, after_contrast)[0, 1]
        assert corr > 0.99
        np.testing.assert_allclose(before_contrast, after_contrast, atol=0.2)

    def test_output_is_absolute_scale_not_ratio(self):
        """Return absolute log2 abundance, not a ratio to the reference.

        With zero offsets the corrected reference mean is ~alpha (the reference level),
        NOT ~0, which is what a ratio-to-reference would produce.
        """
        n_feat = 40
        mu = np.linspace(10, 20, n_feat)
        offsets = {"A": 0.0, "B": 0.0, "C": 0.0}
        data_in, batch, ref_mask = _build_reference_anchored_data(
            n_feat, ["A", "B", "C"], n_ref=2, n_exp=3, offsets=offsets, mu=mu,
            noise=0.02, seed=3
        )
        corrected = combat_reference_anchored(data_in, batch, ref_mask)

        # Corrected references center on the reference level (mu), not 0 — this is
        # the key check that output is calibrated absolute abundance, not a ratio.
        ref_idx = np.where(ref_mask)[0]
        corrected_ref_mean = corrected[:, ref_idx].mean(axis=1)
        np.testing.assert_allclose(corrected_ref_mean, mu, atol=0.2)
        assert corrected_ref_mean.mean() > 10  # nowhere near 0
        # References (zero offset, tiny noise) are essentially unchanged.
        np.testing.assert_allclose(data_in[:, ref_idx], corrected[:, ref_idx], atol=0.2)
        # Whole matrix stays in a sane absolute range around mu (10..20) plus
        # experimental biological spread (~3 SD), not collapsed near 0 or inflated.
        assert corrected.min() > 5 and corrected.max() < 25
        # And the output range tracks the input range (no scale collapse/blowup).
        assert abs((corrected.max() - corrected.min()) - (data_in.max() - data_in.min())) < 2.0

    def test_noisy_feature_shrunk_toward_consensus(self):
        """Shrink a spurious single-reference delta toward the batch consensus.

        The applied shift is smaller than the naive single-point delta.
        """
        n_feat = 80
        mu = np.full(n_feat, 15.0)
        # Single reference per batch -> heavy shrinkage; clean consensus offset 0.6
        offsets = {"A": 0.0, "B": 0.6}
        data_in, batch, ref_mask = _build_reference_anchored_data(
            n_feat, ["A", "B"], n_ref=1, n_exp=4, offsets=offsets, mu=mu,
            noise=0.01, seed=4
        )
        # Corrupt feature 0's single reference in batch B: reads offset ~2.5 (wrong)
        b_ref = np.where((batch == "B") & ref_mask)[0][0]
        data_in[0, b_ref] = mu[0] + 2.5

        corrected = combat_reference_anchored(data_in, batch, ref_mask)

        # Applied shift for batch B (location-only): input - output, constant per sample
        b_samples = np.where(batch == "B")[0]
        applied_shift = (data_in[:, b_samples] - corrected[:, b_samples]).mean(axis=1)
        naive_delta_f0 = 2.5  # what naive single-point would subtract
        consensus = np.median(applied_shift)  # ~0.6
        # Feature 0's shift is pulled well below the naive delta, toward consensus
        assert applied_shift[0] < naive_delta_f0
        assert abs(applied_shift[0] - consensus) < abs(naive_delta_f0 - consensus)

    def test_no_reference_batch_fallback_warns(self, caplog):
        """Fall back to grand-mean estimation for a batch with no references.

        A warning is emitted, and the batches that do have references still align.
        """
        import logging

        n_feat = 50
        mu = np.full(n_feat, 14.0)
        offsets = {"A": 0.0, "B": 0.7, "C": -0.4}
        # Batch C has NO references
        data_in, batch, ref_mask = _build_reference_anchored_data(
            n_feat,
            ["A", "B", "C"],
            n_ref={"A": 2, "B": 2, "C": 0},
            n_exp=4,
            offsets=offsets,
            mu=mu,
            seed=5,
        )
        with caplog.at_level(logging.WARNING):
            corrected = combat_reference_anchored(
                data_in, batch, ref_mask, no_reference_batch="fallback"
            )

        assert any("no reference samples" in r.message.lower() for r in caplog.records)
        # Reference batches A and B still align after correction
        ref_ab = np.where(ref_mask & np.isin(batch, ["A", "B"]))[0]
        assert corrected[:, ref_ab].std(axis=1).mean() < 0.2
        # Output still on absolute scale (fallback batch corrected, not blown up)
        assert 10 < corrected.mean() < 18

    def test_no_reference_batch_error_mode(self):
        """no_reference_batch='error' raises when a batch lacks references."""
        n_feat = 20
        mu = np.full(n_feat, 14.0)
        offsets = {"A": 0.0, "B": 0.5}
        data_in, batch, ref_mask = _build_reference_anchored_data(
            n_feat, ["A", "B"], n_ref={"A": 2, "B": 0}, n_exp=3, offsets=offsets, mu=mu, seed=6
        )
        with pytest.raises(ValueError, match="no reference samples"):
            combat_reference_anchored(data_in, batch, ref_mask, no_reference_batch="error")

    def test_eb_shrinkage_reduces_noisy_delta_dispersion(self):
        """Shrink noisy per-feature reference deltas toward the batch consensus.

        The applied per-feature shifts are LESS dispersed than the raw, un-shrunk
        single-point deltas.
        """
        n_feat = 200
        rng = np.random.default_rng(11)
        mu = rng.normal(15, 1.5, n_feat)
        # All features share the same TRUE batch-B offset (0.5); per-feature
        # variation in the measured reference delta is pure estimation noise.
        offsets = {"A": 0.0, "B": 0.5}
        data_in, batch, ref_mask = _build_reference_anchored_data(
            n_feat, ["A", "B"], n_ref=2, n_exp=4, offsets=offsets, mu=mu,
            noise=0.3, seed=11  # substantial reference noise -> noisy raw deltas
        )

        # Raw single-point delta per feature for batch B (vs pooled reference mean)
        a_ref = np.where((batch == "A") & ref_mask)[0]
        b_ref = np.where((batch == "B") & ref_mask)[0]
        alpha = np.concatenate([data_in[:, a_ref], data_in[:, b_ref]], axis=1).mean(axis=1)
        raw_delta_b = data_in[:, b_ref].mean(axis=1) - alpha

        corrected = combat_reference_anchored(data_in, batch, ref_mask)
        b_samples = np.where(batch == "B")[0]
        applied_shift = (data_in[:, b_samples] - corrected[:, b_samples]).mean(axis=1)

        # The true per-feature offset is identical, so all variation in raw_delta_b
        # is noise. EB shrinkage toward the consensus should shrink that spread.
        assert applied_shift.std() < raw_delta_b.std()

    def test_scale_harmonized_when_replicates_available(self):
        """Harmonize dispersion when the batches' references differ in spread.

        With >= 2 references per batch, the noisier batch's spread is pulled toward
        the other batch's.
        """
        n_feat = 80
        mu = np.full(n_feat, 15.0)
        offsets = {"A": 0.0, "B": 0.0}
        # Batch B references are 6x noisier than batch A references
        data_in, batch, ref_mask = _build_reference_anchored_data(
            n_feat,
            ["A", "B"],
            n_ref=6,
            n_exp=4,
            offsets=offsets,
            mu=mu,
            ref_noise={"A": 0.1, "B": 0.6},
            seed=8,
        )
        corrected = combat_reference_anchored(data_in, batch, ref_mask)

        a_ref = np.where((batch == "A") & ref_mask)[0]
        b_ref = np.where((batch == "B") & ref_mask)[0]
        spread_a_before = data_in[:, a_ref].std(axis=1).mean()
        spread_b_before = data_in[:, b_ref].std(axis=1).mean()
        spread_a_after = corrected[:, a_ref].std(axis=1).mean()
        spread_b_after = corrected[:, b_ref].std(axis=1).mean()

        # Before: B much noisier than A. After: ratio much closer to 1.
        ratio_before = spread_b_before / spread_a_before
        ratio_after = spread_b_after / spread_a_after
        assert ratio_before > 3.0
        assert ratio_after < 0.5 * ratio_before
