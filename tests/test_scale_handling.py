"""Tests for log2/linear scale handling throughout the pipeline.

These tests ensure data is correctly transformed between scales at each stage
of the pipeline to prevent overflow errors and incorrect calculations.

Scale Convention (from AGENTS.md):
- Input: LINEAR (raw peak areas from Skyline)
- Internal: LOG2 (all rollup/normalization operates on log2 scale)
- Output: LINEAR (final peptide/protein matrices are 2^x, not log2)

Critical: CVs must ALWAYS be calculated on LINEAR scale data.
"""

import numpy as np
import pandas as pd
import pytest


class TestScaleConventions:
    """Test that scale conventions are followed throughout the pipeline."""

    def test_log2_to_linear_conversion(self):
        """Test basic log2 to linear conversion."""
        log2_values = np.array([10.0, 12.0, 14.0, 16.0])
        linear_values = 2**log2_values

        expected = np.array([1024.0, 4096.0, 16384.0, 65536.0])
        np.testing.assert_array_almost_equal(linear_values, expected)

    def test_linear_to_log2_conversion(self):
        """Test basic linear to log2 conversion."""
        linear_values = np.array([1024.0, 4096.0, 16384.0, 65536.0])
        log2_values = np.log2(linear_values)

        expected = np.array([10.0, 12.0, 14.0, 16.0])
        np.testing.assert_array_almost_equal(log2_values, expected)

    def test_roundtrip_conversion(self):
        """Test that linear -> log2 -> linear roundtrip preserves values."""
        original = np.array([1000.0, 5000.0, 10000.0, 50000.0])
        log2_values = np.log2(original)
        recovered = 2**log2_values

        np.testing.assert_array_almost_equal(recovered, original)

    def test_zero_handling_in_log2(self):
        """Test that zeros are handled properly when converting to log2."""
        linear_values = np.array([0.0, 1000.0, 0.0, 5000.0])

        # Replace zeros with NaN before log2
        linear_values_safe = np.where(linear_values == 0, np.nan, linear_values)
        log2_values = np.log2(linear_values_safe)

        assert np.isnan(log2_values[0])
        assert np.isnan(log2_values[2])
        assert not np.isnan(log2_values[1])
        assert not np.isnan(log2_values[3])


class TestCVCalculation:
    """Test that CV is always calculated on linear scale."""

    def test_cv_on_linear_scale(self):
        """CV should be calculated on linear scale, not log2."""
        # Typical proteomics values in log2 scale
        log2_data = np.array([13.0, 13.5, 12.8, 13.2, 13.1])

        # CORRECT: Convert to linear, then calculate CV
        linear_data = 2**log2_data
        cv_correct = (linear_data.std() / linear_data.mean()) * 100

        # INCORRECT: Calculate CV on log2 scale (artificially compressed)
        cv_incorrect = (log2_data.std() / log2_data.mean()) * 100

        # The correct CV should be much larger (10-30% typical for proteomics)
        # The incorrect CV would be artificially small (~2%)
        assert cv_correct > 10, f"CV should be >10% for proteomics data, got {cv_correct:.1f}%"
        assert cv_incorrect < 5, f"Log2 CV should be artificially small, got {cv_incorrect:.1f}%"
        assert cv_correct > cv_incorrect * 3, "Linear CV should be much larger than log2 CV"

    def test_cv_calculation_helper(self):
        """Test CV calculation with the expected workflow."""
        # Simulate peptide abundances across replicates (log2 scale)
        log2_abundances = pd.DataFrame(
            {
                "rep1": [13.0, 14.0, 15.0],
                "rep2": [13.2, 14.1, 14.8],
                "rep3": [12.9, 13.9, 15.2],
            }
        )

        # Convert to linear for CV calculation
        linear_abundances = 2**log2_abundances

        # Calculate CV per row (per peptide)
        cv_per_peptide = (linear_abundances.std(axis=1) / linear_abundances.mean(axis=1)) * 100

        # All CVs should be in reasonable range for proteomics
        assert all(cv_per_peptide > 5), "CVs should be >5%"
        assert all(cv_per_peptide < 50), "CVs should be <50%"


class TestOverflowPrevention:
    """Test that operations don't cause overflow errors."""

    def test_large_log2_values_dont_overflow(self):
        """Test that typical log2 values don't overflow when converted to linear."""
        # Typical range for proteomics log2 values: 0-25
        log2_values = np.array([0.0, 10.0, 15.0, 20.0, 25.0])
        linear_values = 2**log2_values

        # Should not overflow
        assert np.all(np.isfinite(linear_values))

        # But very large log2 values would overflow (float64 max exponent ~1024)
        extreme_log2 = np.array([1025.0, 2000.0])
        with pytest.warns(RuntimeWarning, match="overflow encountered"):
            extreme_linear = 2**extreme_log2
        assert np.all(np.isinf(extreme_linear)), "Should overflow for extreme values"

    def test_median_normalization_on_log2_scale(self):
        """Test that median normalization works correctly on log2 scale."""
        # Simulate samples with different loading (in log2 scale)
        log2_data = pd.DataFrame(
            {
                "sample1": [13.0, 14.0, 15.0],
                "sample2": [14.0, 15.0, 16.0],  # 2x more loaded
                "sample3": [12.0, 13.0, 14.0],  # 2x less loaded
            }
        )

        # Calculate median per sample
        sample_medians = log2_data.median()
        global_median = sample_medians.median()

        # Normalization factors (additive on log2 scale)
        norm_factors = sample_medians - global_median

        # Apply normalization
        normalized = log2_data.copy()
        for col in normalized.columns:
            normalized[col] = normalized[col] - norm_factors[col]

        # After normalization, all sample medians should be equal
        normalized_medians = normalized.median()
        np.testing.assert_array_almost_equal(
            normalized_medians.values, np.full(3, global_median), decimal=10
        )


class TestPipelineScaleHandling:
    """Integration tests for scale handling in the pipeline."""

    def test_peptide_rollup_output_scale(self):
        """Test that peptide rollup outputs are in the expected scale."""
        # This would be an integration test with actual rollup code
        # For now, test the principle

        # Simulated transition data (linear scale from Skyline)
        transition_linear = np.array([1000.0, 2000.0, 1500.0])

        # Sum rollup (should happen on linear scale)
        peptide_linear = transition_linear.sum()

        # Convert to log2 for internal processing
        peptide_log2 = np.log2(peptide_linear)

        # Should be in reasonable range
        assert 10 < peptide_log2 < 20, f"Peptide log2 should be reasonable, got {peptide_log2}"

    def test_protein_normalization_doesnt_overflow(self):
        """Test protein normalization doesn't cause overflow."""
        # Simulate protein abundances in log2 scale (after rollup)
        protein_log2 = pd.DataFrame(
            {
                "prot1": [15.0, 15.2, 14.8],
                "prot2": [12.0, 12.1, 11.9],
            },
            index=["sample1", "sample2", "sample3"],
        ).T

        # Global median normalization (on log2 scale)
        sample_medians = protein_log2.median()
        global_median = sample_medians.median()
        norm_factors = sample_medians - global_median

        # Apply normalization (subtraction on log2 = division on linear)
        normalized_log2 = protein_log2.copy()
        for col in normalized_log2.columns:
            normalized_log2[col] = normalized_log2[col] - norm_factors[col]

        # Convert to linear for output
        output_linear = 2**normalized_log2

        # Should not overflow
        assert np.all(np.isfinite(output_linear.values))

        # Should be in reasonable range for proteomics
        assert output_linear.values.max() < 1e10
        assert output_linear.values.min() > 0


class TestDataFrameScaleTracking:
    """Test ideas for tracking data scale in DataFrames."""

    def test_scale_attribute_tracking(self):
        """Test using DataFrame attrs to track scale."""
        df = pd.DataFrame(
            {
                "peptide": ["pep1", "pep2"],
                "sample1": [13.0, 14.0],
                "sample2": [13.2, 14.1],
            }
        )

        # Track scale using attrs
        df.attrs["scale"] = "log2"

        assert df.attrs.get("scale") == "log2"

        # After conversion
        sample_cols = ["sample1", "sample2"]
        df_linear = df.copy()
        df_linear[sample_cols] = 2 ** df[sample_cols]
        df_linear.attrs["scale"] = "linear"

        assert df_linear.attrs.get("scale") == "linear"


class TestConfigValidation:
    """Test config validation for unknown keys."""

    def test_detect_unknown_config_keys(self):
        """Test that unknown config keys are detected."""
        import sys

        sys.path.insert(0, ".")
        from skyline_prism.cli import _find_unknown_config_keys

        # Config with typos
        user_config = {
            "transition_rollup": {
                "method": "adaptive",
                "min_transtions": 3,  # TYPO - missing 'i'
            },
            "sample_anotations": {  # TYPO - missing 'n'
                "reference_pattern": ["-Pool_"],
            },
        }

        unknown = _find_unknown_config_keys(user_config)

        assert "transition_rollup.min_transtions" in unknown
        assert "sample_anotations" in unknown

    def test_known_keys_not_flagged(self):
        """Test that known keys are not flagged as unknown."""
        import sys

        sys.path.insert(0, ".")
        from skyline_prism.cli import _find_unknown_config_keys

        # Valid config - should have no unknown keys
        user_config = {
            "transition_rollup": {
                "method": "adaptive",
                "learn_adaptive_weights": True,
                "min_transitions": 3,
            },
            "sample_annotations": {
                "reference_pattern": ["-Pool_"],
                "qc_pattern": ["-QC_"],
            },
            "batch_correction": {
                "enabled": True,
                "method": "combat",
            },
        }

        unknown = _find_unknown_config_keys(user_config)

        assert len(unknown) == 0, f"Should have no unknown keys, got: {unknown}"

    def test_deprecated_learn_weights_accepted(self):
        """Test that deprecated learn_weights is still accepted."""
        import sys

        sys.path.insert(0, ".")
        from skyline_prism.cli import _find_unknown_config_keys

        # Using deprecated learn_weights (should be learn_adaptive_weights)
        user_config = {
            "transition_rollup": {
                "method": "adaptive",
                "learn_weights": True,  # Deprecated but still accepted
            },
        }

        unknown = _find_unknown_config_keys(user_config)

        # learn_weights should NOT be flagged (backwards compatibility)
        assert "transition_rollup.learn_weights" not in unknown
