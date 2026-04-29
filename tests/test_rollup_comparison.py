"""Tests for rollup comparison module."""

import numpy as np
import pandas as pd

from skyline_prism.rollup_comparison import (
    LibraryFitStep,
    PeptideLibraryComparison,
    RollupComparisonSummary,
    compute_cv_linear,
    compute_ranking_score,
    fit_library_single_sample,
    rank_peptides,
)


class TestComputeCVLinear:
    """Tests for compute_cv_linear function."""

    def test_basic_cv(self):
        """Test CV calculation on simple data."""
        abundances = pd.Series([100.0, 110.0, 90.0, 105.0, 95.0])
        cv = compute_cv_linear(abundances)
        # CV = std / mean = ~7.9 / 100 = 0.079
        assert 0.07 < cv < 0.09

    def test_zero_mean(self):
        """Test CV with zero mean returns NaN."""
        abundances = pd.Series([0.0, 0.0, 0.0])
        cv = compute_cv_linear(abundances)
        assert np.isnan(cv)

    def test_all_nan(self):
        """Test CV with all NaN returns NaN."""
        abundances = pd.Series([np.nan, np.nan])
        cv = compute_cv_linear(abundances)
        assert np.isnan(cv)

    def test_single_value(self):
        """Test CV with single value returns NaN."""
        abundances = pd.Series([100.0])
        cv = compute_cv_linear(abundances)
        assert np.isnan(cv)

    def test_identical_values(self):
        """Test CV with identical values is zero."""
        abundances = pd.Series([100.0, 100.0, 100.0])
        cv = compute_cv_linear(abundances)
        assert cv == 0.0


class TestComputeRankingScore:
    """Tests for compute_ranking_score function."""

    def test_most_improved_positive(self):
        """Test most_improved with positive improvement."""
        # library_cv < sum_cv -> positive score
        score = compute_ranking_score(sum_cv=0.20, library_cv=0.10, criterion="most_improved")
        assert score == 0.5  # (0.20 - 0.10) / 0.20

    def test_most_improved_negative(self):
        """Test most_improved with negative improvement."""
        # library_cv > sum_cv -> negative score
        score = compute_ranking_score(sum_cv=0.10, library_cv=0.20, criterion="most_improved")
        assert score == -1.0  # (0.10 - 0.20) / 0.10

    def test_most_worsened(self):
        """Test most_worsened criterion."""
        score = compute_ranking_score(sum_cv=0.10, library_cv=0.20, criterion="most_worsened")
        assert score == 1.0  # (0.20 - 0.10) / 0.10

    def test_highest_cv(self):
        """Test highest_cv criterion."""
        score = compute_ranking_score(sum_cv=0.25, library_cv=0.15, criterion="highest_cv")
        assert score == 0.25

    def test_largest_difference(self):
        """Test largest_difference criterion."""
        score = compute_ranking_score(sum_cv=0.10, library_cv=0.20, criterion="largest_difference")
        assert score == 0.10

    def test_nan_returns_negative_inf(self):
        """Test that NaN CV returns -inf for ranking."""
        score = compute_ranking_score(sum_cv=np.nan, library_cv=0.10, criterion="most_improved")
        assert score == -np.inf


class TestFitLibrarySingleSample:
    """Tests for fit_library_single_sample function."""

    def test_basic_fit(self):
        """Test basic library fitting."""
        # Create synthetic data where library matches observed pattern
        observed = pd.Series({
            "y3": 1000.0,
            "y4": 2000.0,
            "y5": 1500.0,
            "b3": 800.0,
        })
        library = pd.Series({
            "y3": 0.5,
            "y4": 1.0,
            "y5": 0.75,
            "b3": 0.4,
        })
        mz_values = pd.Series({
            "y3": 350.0,
            "y4": 450.0,
            "y5": 550.0,
            "b3": 300.0,
        })

        abundance, r_squared, steps = fit_library_single_sample(
            observed=observed,
            library=library,
            mz_values=mz_values,
            min_fragments=2,
        )

        # Should have at least raw step
        assert len(steps) >= 1
        assert steps[0].step_name == "raw"

        # Abundance should be positive
        assert abundance > 0

    def test_insufficient_fragments(self):
        """Test that insufficient fragments returns NaN."""
        observed = pd.Series({"y3": 1000.0})
        library = pd.Series({"y3": 1.0})
        mz_values = pd.Series({"y3": 350.0})

        abundance, r_squared, steps = fit_library_single_sample(
            observed=observed,
            library=library,
            mz_values=mz_values,
            min_fragments=2,
        )

        assert np.isnan(abundance)

    def test_no_matching_fragments(self):
        """Test when observed and library don't share fragments."""
        observed = pd.Series({"y3": 1000.0, "y4": 2000.0})
        library = pd.Series({"b3": 1.0, "b4": 0.5})
        mz_values = pd.Series({"y3": 350.0, "y4": 450.0, "b3": 300.0, "b4": 400.0})

        abundance, r_squared, steps = fit_library_single_sample(
            observed=observed,
            library=library,
            mz_values=mz_values,
            min_fragments=2,
        )

        assert np.isnan(abundance)


class TestRankPeptides:
    """Tests for rank_peptides function."""

    def test_ranking_order(self):
        """Test that peptides are ranked correctly."""
        # Create mock peptide results
        peptide_results = {
            "PEP1": _mock_peptide_comparison(sum_cv=0.30, library_cv=0.10),  # 67% improvement
            "PEP2": _mock_peptide_comparison(sum_cv=0.20, library_cv=0.15),  # 25% improvement
            "PEP3": _mock_peptide_comparison(sum_cv=0.25, library_cv=0.20),  # 20% improvement
        }

        top = rank_peptides(peptide_results, "most_improved", top_n=3)

        # PEP1 has highest improvement, should be first
        assert top[0] == "PEP1"
        assert top[1] == "PEP2"
        assert top[2] == "PEP3"

    def test_top_n_limit(self):
        """Test that top_n limits the results."""
        peptide_results = {
            f"PEP{i}": _mock_peptide_comparison(sum_cv=0.20, library_cv=0.10)
            for i in range(10)
        }

        top = rank_peptides(peptide_results, "most_improved", top_n=3)
        assert len(top) == 3


def _mock_peptide_comparison(sum_cv: float, library_cv: float) -> PeptideLibraryComparison:
    """Create a mock PeptideLibraryComparison for testing."""
    if sum_cv > 0:
        cv_improvement = (sum_cv - library_cv) / sum_cv
    else:
        cv_improvement = 0.0

    return PeptideLibraryComparison(
        peptide="TESTPEPTIDE",
        mz_values=pd.Series({"y3": 350.0}),
        raw_transitions={},
        library_spectrum=pd.Series(),
        fitting_steps={},
        sum_abundances_raw=pd.Series(),
        library_abundances_raw=pd.Series(),
        sum_abundances_corrected=pd.Series(),
        library_abundances_corrected=pd.Series(),
        sum_cv=sum_cv,
        library_cv=library_cv,
        cv_improvement=cv_improvement,
        ranking_score=compute_ranking_score(sum_cv, library_cv, "most_improved"),
    )


class TestDataStructures:
    """Tests for data structure creation."""

    def test_library_fit_step_creation(self):
        """Test LibraryFitStep can be created."""
        step = LibraryFitStep(
            step_name="raw",
            transition_intensities=pd.Series({"y3": 1000.0}),
            library_predicted=None,
            excluded_transitions=[],
            r_squared=None,
            scale_factor=None,
            normalized_residuals=None,
        )
        assert step.step_name == "raw"

    def test_rollup_comparison_summary_creation(self):
        """Test RollupComparisonSummary can be created."""
        summary = RollupComparisonSummary(
            n_peptides_total=100,
            n_peptides_improved=60,
            n_peptides_worsened=30,
            n_peptides_unchanged=10,
            sum_median_cv=0.18,
            library_median_cv=0.14,
            median_improvement=0.20,
        )
        assert summary.n_peptides_total == 100
        assert summary.median_improvement == 0.20
