"""Tests for the visualization module."""

import numpy as np
import pandas as pd
import pytest

# Check if visualization dependencies are available
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for tests
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn  # noqa: F401

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import sklearn  # noqa: F401

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@pytest.fixture
def sample_long_data():
    """Create sample long-format proteomics data."""
    np.random.seed(42)

    n_peptides = 50
    n_samples = 10

    peptides = [f"Peptide_{i}" for i in range(n_peptides)]
    samples = [f"Sample_{i}" for i in range(n_samples)]

    data = []
    for pep in peptides:
        base_intensity = np.random.uniform(1e4, 1e6)
        for samp_idx, samp in enumerate(samples):
            # Add sample-level systematic effect
            sample_effect = 1 + 0.2 * (samp_idx / n_samples - 0.5)
            # Add noise
            noise = np.random.lognormal(0, 0.2)
            intensity = base_intensity * sample_effect * noise

            data.append(
                {
                    "precursor_id": pep,
                    "replicate_name": samp,
                    "abundance": intensity,
                    "sample_type": (
                        "reference"
                        if samp_idx < 2
                        else ("qc" if samp_idx < 4 else "experimental")
                    ),
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def normalized_data(sample_long_data):
    """Create 'normalized' version of data with reduced variance."""
    df = sample_long_data.copy()

    # Reduce sample-level systematic effects
    sample_medians = df.groupby("replicate_name")["abundance"].median()
    global_median = sample_medians.median()

    def normalize(row):
        sample_median = sample_medians[row["replicate_name"]]
        return row["abundance"] * (global_median / sample_median)

    df["abundance"] = df.apply(normalize, axis=1)
    return df


@pytest.fixture
def sample_types(sample_long_data):
    """Return sample type mapping."""
    return (
        sample_long_data.drop_duplicates("replicate_name")
        .set_index("replicate_name")["sample_type"]
        .to_dict()
    )


class TestIntensityDistribution:
    """Tests for plot_intensity_distribution."""

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_basic_plot(self, sample_long_data):
        """Test basic intensity distribution plot."""
        from skyline_prism.visualization import plot_intensity_distribution

        fig = plot_intensity_distribution(
            sample_long_data,
            show_plot=False,
        )

        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_with_sample_types(self, sample_long_data, sample_types):
        """Test with sample type coloring."""
        from skyline_prism.visualization import plot_intensity_distribution

        fig = plot_intensity_distribution(
            sample_long_data,
            sample_types=sample_types,
            show_plot=False,
        )

        assert fig is not None
        plt.close(fig)


class TestNormalizationComparison:
    """Tests for plot_normalization_comparison."""

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_basic_comparison(self, sample_long_data, normalized_data):
        """Test before/after comparison plot."""
        from skyline_prism.visualization import plot_normalization_comparison

        fig = plot_normalization_comparison(
            sample_long_data,
            normalized_data,
            show_plot=False,
        )

        assert fig is not None
        assert len(fig.axes) == 2  # Two panels
        plt.close(fig)


class TestPCA:
    """Tests for PCA visualization functions."""

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_basic_pca(self, sample_long_data, sample_types):
        """Test basic PCA plot."""
        from skyline_prism.visualization import plot_pca

        fig, pca_df = plot_pca(
            sample_long_data,
            sample_types=sample_types,
            show_plot=False,
        )

        assert fig is not None
        assert len(pca_df) == len(sample_long_data["replicate_name"].unique())
        assert "PC1" in pca_df.columns
        assert "PC2" in pca_df.columns
        plt.close(fig)

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_comparative_pca_two_panels(self, sample_long_data, normalized_data, sample_types):
        """Test comparative PCA with two panels."""
        from skyline_prism.visualization import plot_comparative_pca

        fig = plot_comparative_pca(
            sample_long_data,
            normalized_data,
            sample_groups=sample_types,
            show_plot=False,
        )

        assert fig is not None
        assert len(fig.axes) == 2
        plt.close(fig)

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_comparative_pca_three_panels(self, sample_long_data, normalized_data, sample_types):
        """Test comparative PCA with three panels (incl. batch-corrected)."""
        from skyline_prism.visualization import plot_comparative_pca

        # Use normalized as mock batch-corrected
        fig = plot_comparative_pca(
            sample_long_data,
            normalized_data,
            normalized_data,  # Using same data as mock batch-corrected
            sample_groups=sample_types,
            show_plot=False,
        )

        assert fig is not None
        assert len(fig.axes) == 3
        plt.close(fig)


class TestCorrelationHeatmap:
    """Tests for correlation heatmap functions."""

    @pytest.mark.skipif(not HAS_SEABORN, reason="seaborn not installed")
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_control_correlation(self, sample_long_data):
        """Test control sample correlation heatmap."""
        from skyline_prism.visualization import plot_control_correlation_heatmap

        fig, corr_matrix = plot_control_correlation_heatmap(
            sample_long_data,
            sample_type_col="sample_type",
            control_types=["reference", "qc"],
            show_plot=False,
        )

        assert fig is not None
        # Should have 4 control samples (2 reference + 2 qc)
        assert corr_matrix.shape == (4, 4)
        # Diagonal should be 1.0
        np.testing.assert_array_almost_equal(
            np.diag(corr_matrix.values), np.ones(4), decimal=5
        )
        plt.close(fig)

    @pytest.mark.skipif(not HAS_SEABORN, reason="seaborn not installed")
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_sample_correlation(self, sample_long_data, sample_types):
        """Test full sample correlation matrix."""
        from skyline_prism.visualization import plot_sample_correlation_matrix

        fig, corr_matrix = plot_sample_correlation_matrix(
            sample_long_data,
            sample_types=sample_types,
            show_plot=False,
        )

        assert fig is not None
        n_samples = len(sample_long_data["replicate_name"].unique())
        assert corr_matrix.shape == (n_samples, n_samples)
        plt.close(fig)


class TestCVDistribution:
    """Tests for CV distribution plotting."""

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_cv_distribution(self, sample_long_data):
        """Test CV distribution histogram."""
        from skyline_prism.visualization import plot_cv_distribution

        fig, cv_data = plot_cv_distribution(
            sample_long_data,
            sample_type_col="sample_type",
            control_types=["reference", "qc"],
            show_plot=False,
        )

        assert fig is not None
        assert "reference" in cv_data
        assert "qc" in cv_data
        assert len(cv_data["reference"]) > 0
        assert len(cv_data["qc"]) > 0
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_comparative_cv(self, sample_long_data, normalized_data):
        """Test comparative CV distribution."""
        from skyline_prism.visualization import plot_comparative_cv

        fig = plot_comparative_cv(
            sample_long_data,
            normalized_data,
            sample_type_col="sample_type",
            control_type="reference",
            show_plot=False,
        )

        assert fig is not None
        assert len(fig.axes) == 2
        plt.close(fig)


class TestHelperFunctions:
    """Tests for helper functions."""

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_get_sample_colors(self, sample_types):
        """Test sample color assignment."""
        from skyline_prism.visualization import _get_sample_colors

        colors, legend = _get_sample_colors(
            list(sample_types.keys()),
            sample_types,
        )

        assert len(colors) == len(sample_types)
        assert len(legend) == len(set(sample_types.values()))

    def test_pivot_to_wide(self, sample_long_data):
        """Test pivoting function."""
        from skyline_prism.visualization import _pivot_to_wide

        wide = _pivot_to_wide(
            sample_long_data,
            "precursor_id",
            "replicate_name",
            "abundance",
        )

        n_peptides = len(sample_long_data["precursor_id"].unique())
        n_samples = len(sample_long_data["replicate_name"].unique())

        assert wide.shape == (n_peptides, n_samples)


class TestMissingDependencies:
    """Test graceful handling of missing dependencies."""

    def test_has_dependency_flags(self):
        """Test that module has dependency availability flags."""
        from skyline_prism import visualization as viz

        # Module should have flags indicating whether dependencies are available
        assert hasattr(viz, "HAS_MATPLOTLIB")
        assert hasattr(viz, "HAS_SEABORN")
        assert hasattr(viz, "HAS_SKLEARN")

        # Since we can run tests, matplotlib should be available
        assert viz.HAS_MATPLOTLIB is True


class TestSavingPlots:
    """Tests for plot saving functionality."""

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_save_intensity_distribution(self, sample_long_data, tmp_path):
        """Test saving intensity distribution plot."""
        from skyline_prism.visualization import plot_intensity_distribution

        save_path = tmp_path / "intensity.png"
        fig = plot_intensity_distribution(
            sample_long_data,
            save_path=str(save_path),
            show_plot=False,
        )

        assert save_path.exists()
        assert save_path.stat().st_size > 0
        plt.close(fig)

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_save_pca(self, sample_long_data, tmp_path):
        """Test saving PCA plot."""
        from skyline_prism.visualization import plot_pca

        save_path = tmp_path / "pca.png"
        fig, _ = plot_pca(
            sample_long_data,
            save_path=str(save_path),
            show_plot=False,
        )

        assert save_path.exists()
        plt.close(fig)


@pytest.fixture
def sample_data_with_rt():
    """Create sample data with retention time for RT correction tests."""
    np.random.seed(42)

    n_peptides = 100
    n_samples = 8

    peptides = [f"Peptide_{i}" for i in range(n_peptides)]
    samples = [f"Sample_{i}" for i in range(n_samples)]

    data = []
    for pep_idx, pep in enumerate(peptides):
        base_intensity = np.random.uniform(1e4, 1e6)
        # Each peptide has a consistent RT
        base_rt = 10 + pep_idx * 0.5  # RTs from 10 to 60 minutes

        for samp_idx, samp in enumerate(samples):
            # Add RT-dependent bias for some samples
            rt_bias = 0.1 * np.sin(base_rt / 10) * (samp_idx - 4)

            # Add noise
            noise = np.random.lognormal(0, 0.15)
            intensity = base_intensity * (1 + rt_bias) * noise

            # Small RT variation per sample
            rt = base_rt + np.random.normal(0, 0.1)

            data.append(
                {
                    "precursor_id": pep,
                    "replicate_name": samp,
                    "abundance": intensity,
                    "retention_time": rt,
                    "sample_type": (
                        "reference"
                        if samp_idx < 2
                        else ("qc" if samp_idx < 4 else "experimental")
                    ),
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def reference_stats(sample_data_with_rt):
    """Compute reference statistics for RT correction tests."""
    ref_data = sample_data_with_rt[sample_data_with_rt["sample_type"] == "reference"]
    stats = ref_data.groupby("precursor_id").agg({"abundance": "mean"})
    stats.columns = ["mean_abundance"]
    return stats


class TestRTResiduals:
    """Tests for RT residual visualization functions."""

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_plot_rt_residuals(self, sample_data_with_rt, reference_stats):
        """Test basic RT residuals plot."""
        from skyline_prism.visualization import plot_rt_residuals

        fig, residuals_df = plot_rt_residuals(
            sample_data_with_rt,
            reference_stats,
            n_samples_to_show=2,
            show_plot=False,
        )

        assert fig is not None
        assert len(residuals_df) > 0
        assert "residual" in residuals_df.columns
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_plot_rt_correction_comparison(self, sample_data_with_rt, reference_stats):
        """Test RT correction comparison plot."""
        from skyline_prism.visualization import plot_rt_correction_comparison

        # Use same data as before/after (mock - in real use they'd be different)
        fig = plot_rt_correction_comparison(
            sample_data_with_rt,
            sample_data_with_rt,  # Same data for test
            reference_stats,
            show_plot=False,
        )

        assert fig is not None
        assert len(fig.axes) == 4  # 2x2 grid
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_plot_rt_correction_per_sample(self, sample_data_with_rt, reference_stats):
        """Test per-sample RT correction plot."""
        from skyline_prism.visualization import plot_rt_correction_per_sample

        fig = plot_rt_correction_per_sample(
            sample_data_with_rt,
            sample_data_with_rt,  # Same data for test
            reference_stats,
            n_samples=2,
            show_plot=False,
        )

        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_rt_residuals_no_matching_precursors(self, sample_data_with_rt):
        """Test RT residuals with no matching precursors."""
        from skyline_prism.visualization import plot_rt_residuals

        # Create empty reference stats
        empty_stats = pd.DataFrame({"mean_abundance": []})
        empty_stats.index.name = "precursor_id"

        fig, df = plot_rt_residuals(
            sample_data_with_rt,
            empty_stats,
            show_plot=False,
        )

        assert fig is None
        assert len(df) == 0

