"""Tests for protein rollup module."""

import pytest
import pandas as pd
import numpy as np

from skyline_prism.rollup import (
    tukey_median_polish,
    MedianPolishResult,
)


class TestTukeyMedianPolish:
    """Tests for Tukey median polish algorithm."""

    def test_simple_matrix(self):
        """Test median polish on a simple matrix."""
        # Create a simple peptides x samples matrix
        data = pd.DataFrame({
            'Sample1': [10.0, 12.0, 11.0],
            'Sample2': [11.0, 13.0, 12.0],
            'Sample3': [9.0, 11.0, 10.0],
        }, index=['Pep1', 'Pep2', 'Pep3'])

        result = tukey_median_polish(data)

        assert isinstance(result, MedianPolishResult)
        assert len(result.col_effects) == 3  # 3 samples
        assert len(result.row_effects) == 3  # 3 peptides
        assert result.converged

    def test_convergence(self):
        """Test that median polish converges."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(10, 5) + 20,
            index=[f'Pep{i}' for i in range(10)],
            columns=[f'Sample{i}' for i in range(5)]
        )

        result = tukey_median_polish(data, max_iter=100)

        assert result.converged
        assert result.n_iterations < 100

    def test_outlier_robustness(self):
        """Test that median polish is robust to outliers."""
        # Create matrix with one outlier
        data = pd.DataFrame({
            'Sample1': [10.0, 10.0, 10.0, 100.0],  # Last is outlier
            'Sample2': [11.0, 11.0, 11.0, 11.0],
            'Sample3': [12.0, 12.0, 12.0, 12.0],
        }, index=['Pep1', 'Pep2', 'Pep3', 'PepOutlier'])

        result = tukey_median_polish(data)

        # Sample effects should be approximately 10, 11, 12
        # despite the outlier
        effects = result.col_effects
        assert abs(effects['Sample2'] - effects['Sample1'] - 1.0) < 0.5
        assert abs(effects['Sample3'] - effects['Sample2'] - 1.0) < 0.5

    def test_missing_values(self):
        """Test handling of missing values."""
        data = pd.DataFrame({
            'Sample1': [10.0, np.nan, 10.0],
            'Sample2': [11.0, 11.0, np.nan],
            'Sample3': [12.0, 12.0, 12.0],
        }, index=['Pep1', 'Pep2', 'Pep3'])

        result = tukey_median_polish(data)

        # Should still produce results
        assert len(result.col_effects) == 3
        assert not result.col_effects.isna().any()

    def test_single_peptide(self):
        """Test behavior with single peptide (degenerate case)."""
        data = pd.DataFrame({
            'Sample1': [10.0],
            'Sample2': [11.0],
            'Sample3': [12.0],
        }, index=['Pep1'])

        result = tukey_median_polish(data)

        # With single peptide, sample effects should equal the values
        # (centered around the median)
        assert len(result.col_effects) == 3

    def test_preserves_relative_quantification(self):
        """Test that relative differences between samples are preserved."""
        # Create matrix where Sample2 is consistently 2x Sample1 (log2 diff = 1)
        data = pd.DataFrame({
            'Sample1': [10.0, 12.0, 11.0, 13.0],
            'Sample2': [11.0, 13.0, 12.0, 14.0],  # +1 log2
        }, index=['Pep1', 'Pep2', 'Pep3', 'Pep4'])

        result = tukey_median_polish(data)

        diff = result.col_effects['Sample2'] - result.col_effects['Sample1']
        assert abs(diff - 1.0) < 0.1


class TestRollupMethods:
    """Tests for different rollup method implementations."""

    def test_topn_rollup(self):
        """Test Top-N rollup method."""
        # Import when implemented
        pass

    def test_ibaq_rollup(self):
        """Test iBAQ rollup method."""
        # Import when implemented
        pass


class TestQualityWeightedAggregation:
    """Tests for quality-weighted transition aggregation."""

    def test_weight_calculation(self):
        """Test variance model weight calculation."""
        # Will test VarianceModelParams when transition rollup is implemented
        pass
