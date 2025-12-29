"""Visualization module for PRISM data analysis.

Provides plotting functions for assessing normalization effects, batch correction,
and data quality through:
- Distribution comparisons (box plots, density plots)
- PCA analysis (before/after normalization/batch correction)
- Correlation heatmaps for control samples
- CV distribution histograms

All functions work with PRISM's long-format data structure.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    mpatches = None

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    PCA = None
    StandardScaler = None


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def _check_seaborn():
    """Check if seaborn is available."""
    if not HAS_SEABORN:
        raise ImportError(
            "seaborn is required for this visualization. "
            "Install with: pip install seaborn"
        )


def _check_sklearn():
    """Check if sklearn is available."""
    if not HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for PCA visualization. "
            "Install with: pip install scikit-learn"
        )


def _pivot_to_wide(
    data: pd.DataFrame,
    index_col: str,
    columns_col: str,
    values_col: str,
) -> pd.DataFrame:
    """Pivot long-format data to wide format.

    Args:
        data: Long-format DataFrame
        index_col: Column to use as row index (e.g., 'precursor_id')
        columns_col: Column to use for columns (e.g., 'replicate_name')
        values_col: Column with values (e.g., 'abundance')

    Returns:
        Wide-format DataFrame with samples as columns

    """
    return data.pivot_table(
        index=index_col,
        columns=columns_col,
        values=values_col,
        aggfunc="first",
    )


def _get_sample_colors(
    sample_names: list[str],
    sample_types: dict[str, str] | None = None,
    palette: str = "Set2",
) -> tuple[list[str], dict[str, str]]:
    """Get colors for samples based on sample type.

    Args:
        sample_names: List of sample names
        sample_types: Dict mapping sample name to type (experimental, qc, reference)
        palette: Color palette name

    Returns:
        Tuple of (colors_list, legend_dict)

    """
    _check_matplotlib()

    # Default colors for sample types
    type_colors = {
        "experimental": "#1f77b4",  # Blue
        "qc": "#ff7f0e",  # Orange
        "reference": "#2ca02c",  # Green
        "unknown": "#7f7f7f",  # Gray
    }

    if sample_types is None:
        colors = ["#1f77b4"] * len(sample_names)
        legend = {}
    else:
        colors = []
        for sample in sample_names:
            sample_type = sample_types.get(sample, "unknown")
            colors.append(type_colors.get(sample_type, type_colors["unknown"]))

        # Build legend
        present_types = set(sample_types.values())
        legend = {t: type_colors.get(t, type_colors["unknown"]) for t in present_types}

    return colors, legend


def plot_intensity_distribution(
    data: pd.DataFrame,
    sample_col: str = "replicate_name",
    abundance_col: str = "abundance",
    sample_types: dict[str, str] | None = None,
    title: str = "Intensity Distribution",
    log_transform: bool = True,
    figsize: tuple[int, int] = (14, 6),
    show_plot: bool = True,
    save_path: str | None = None,
) -> plt.Figure | None:
    """Create box plot showing intensity distribution across samples.

    Args:
        data: Long-format DataFrame with abundance data
        sample_col: Column name with sample identifiers
        abundance_col: Column name with abundance values
        sample_types: Dict mapping sample name to type for coloring
        title: Plot title
        log_transform: Whether to log2-transform data
        figsize: Figure size (width, height)
        show_plot: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure if show_plot is False, else None

    """
    _check_matplotlib()

    # Get unique samples in order
    sample_order = data[sample_col].unique()

    # Prepare data for boxplot
    box_data = []
    for sample in sample_order:
        sample_data = data.loc[data[sample_col] == sample, abundance_col].dropna()
        if log_transform:
            sample_data = np.log2(sample_data.replace(0, np.nan)).dropna()
        box_data.append(sample_data.values)

    # Get colors
    colors, legend = _get_sample_colors(list(sample_order), sample_types)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create box plot
    bp = ax.boxplot(
        box_data,
        positions=range(len(sample_order)),
        widths=0.6,
        patch_artist=True,
        showfliers=False,  # Hide outliers for cleaner plot
    )

    # Color boxes
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Styling
    ax.set_xlabel("Sample", fontsize=12)
    ylabel = f"Log2({abundance_col})" if log_transform else abundance_col
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # X-axis labels
    if len(sample_order) <= 20:
        ax.set_xticks(range(len(sample_order)))
        ax.set_xticklabels(sample_order, rotation=45, ha="right", fontsize=8)
    else:
        # Too many samples, show every Nth label
        n = max(1, len(sample_order) // 20)
        ax.set_xticks(range(0, len(sample_order), n))
        ax.set_xticklabels(
            [sample_order[i] for i in range(0, len(sample_order), n)],
            rotation=45,
            ha="right",
            fontsize=8,
        )

    # Legend
    if legend:
        patches = [
            mpatches.Patch(color=color, label=label, alpha=0.7)
            for label, color in legend.items()
        ]
        ax.legend(handles=patches, loc="upper right")

    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show_plot:
        plt.show()
        return None
    else:
        return fig


def plot_normalization_comparison(
    data_before: pd.DataFrame,
    data_after: pd.DataFrame,
    sample_col: str = "replicate_name",
    abundance_col_before: str = "abundance",
    abundance_col_after: str = "abundance",
    title: str = "Normalization Comparison",
    figsize: tuple[int, int] = (16, 6),
    show_plot: bool = True,
    save_path: str | None = None,
) -> plt.Figure | None:
    """Create side-by-side density plots comparing before/after normalization.

    Args:
        data_before: Long-format DataFrame before normalization
        data_after: Long-format DataFrame after normalization
        sample_col: Column name with sample identifiers
        abundance_col_before: Abundance column in before data
        abundance_col_after: Abundance column in after data
        title: Plot title
        figsize: Figure size
        show_plot: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure if show_plot is False, else None

    """
    _check_matplotlib()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Get samples
    samples_before = data_before[sample_col].unique()
    samples_after = data_after[sample_col].unique()

    # Use consistent colors
    cmap = plt.colormaps.get_cmap("tab20")

    # Before normalization
    for i, sample in enumerate(samples_before):
        sample_data = data_before.loc[
            data_before[sample_col] == sample, abundance_col_before
        ].dropna()
        log_data = np.log2(sample_data.replace(0, np.nan)).dropna()
        if len(log_data) > 0:
            # Simple density estimation using histogram
            counts, bins = np.histogram(log_data, bins=50, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax1.plot(
                bin_centers,
                counts,
                alpha=0.6,
                linewidth=1,
                color=cmap(i % 20),
            )

    ax1.set_xlabel("Log2(Intensity)", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Before Normalization", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # After normalization
    for i, sample in enumerate(samples_after):
        sample_data = data_after.loc[
            data_after[sample_col] == sample, abundance_col_after
        ].dropna()
        log_data = np.log2(sample_data.replace(0, np.nan)).dropna()
        if len(log_data) > 0:
            counts, bins = np.histogram(log_data, bins=50, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax2.plot(
                bin_centers,
                counts,
                alpha=0.6,
                linewidth=1,
                color=cmap(i % 20),
            )

    ax2.set_xlabel("Log2(Intensity)", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("After Normalization", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Calculate and display median range
    def calc_median_range(df, sample_col, abundance_col):
        medians = []
        for sample in df[sample_col].unique():
            sample_data = df.loc[df[sample_col] == sample, abundance_col].dropna()
            log_data = np.log2(sample_data.replace(0, np.nan)).dropna()
            if len(log_data) > 0:
                medians.append(log_data.median())
        return max(medians) - min(medians) if medians else 0

    range_before = calc_median_range(data_before, sample_col, abundance_col_before)
    range_after = calc_median_range(data_after, sample_col, abundance_col_after)
    reduction = (range_before - range_after) / range_before * 100 if range_before > 0 else 0

    fig.suptitle(
        f"{title}\nMedian range: {range_before:.2f} → {range_after:.2f} "
        f"({reduction:.1f}% reduction)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show_plot:
        plt.show()
        return None
    else:
        return fig


def plot_pca(
    data: pd.DataFrame,
    sample_col: str = "replicate_name",
    precursor_col: str = "precursor_id",
    abundance_col: str = "abundance",
    sample_types: dict[str, str] | None = None,
    sample_groups: dict[str, str] | None = None,
    title: str = "PCA Analysis",
    n_components: int = 2,
    log_transform: bool = True,
    figsize: tuple[int, int] = (10, 8),
    show_plot: bool = True,
    save_path: str | None = None,
) -> tuple[plt.Figure | None, pd.DataFrame]:
    """Create PCA plot with sample coloring by type or group.

    Args:
        data: Long-format DataFrame with abundance data
        sample_col: Column name with sample identifiers
        precursor_col: Column name with peptide/precursor identifiers
        abundance_col: Column name with abundance values
        sample_types: Dict mapping sample name to type (for coloring)
        sample_groups: Dict mapping sample name to group (alternative coloring)
        title: Plot title
        n_components: Number of PCA components to compute
        log_transform: Whether to log2-transform data before PCA
        figsize: Figure size
        show_plot: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        Tuple of (Figure if show_plot is False, PCA results DataFrame)

    """
    _check_matplotlib()
    _check_sklearn()

    # Pivot to wide format
    wide_data = _pivot_to_wide(data, precursor_col, sample_col, abundance_col)

    # Log transform if requested
    if log_transform:
        wide_data = np.log2(wide_data.replace(0, np.nan))

    # Drop peptides with too many missing values
    wide_data = wide_data.dropna(axis=0, thresh=len(wide_data.columns) * 0.5)
    wide_data = wide_data.fillna(wide_data.median())

    if wide_data.shape[0] < n_components:
        logger.warning(f"Too few peptides for PCA: {wide_data.shape[0]}")
        return None, pd.DataFrame()

    # Transpose: samples as rows, peptides as columns
    pca_input = wide_data.T

    # Standardize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pca_input)

    # PCA
    n_comp = min(n_components, pca_input.shape[0] - 1, pca_input.shape[1])
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(scaled_data)

    # Results DataFrame
    pca_df = pd.DataFrame(
        scores[:, :2],
        columns=["PC1", "PC2"],
        index=pca_input.index,
    )
    pca_df["Sample"] = pca_df.index

    # Add grouping info
    if sample_groups:
        pca_df["Group"] = [sample_groups.get(s, "Unknown") for s in pca_df.index]
    elif sample_types:
        pca_df["Group"] = [sample_types.get(s, "Unknown") for s in pca_df.index]
    else:
        pca_df["Group"] = "All"

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique groups and colors
    unique_groups = pca_df["Group"].unique()
    cmap = plt.colormaps.get_cmap("Set1")
    group_colors = {g: cmap(i / max(1, len(unique_groups) - 1)) for i, g in enumerate(unique_groups)}

    # Plot each group
    for group in unique_groups:
        group_data = pca_df[pca_df["Group"] == group]
        ax.scatter(
            group_data["PC1"],
            group_data["PC2"],
            c=[group_colors[group]],
            label=group,
            alpha=0.7,
            s=100,
            edgecolors="black",
            linewidth=0.5,
        )

    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
        fontsize=12,
    )
    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
        fontsize=12,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if len(unique_groups) > 1:
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show_plot:
        plt.show()
        return None, pca_df
    else:
        return fig, pca_df


def plot_comparative_pca(
    data_original: pd.DataFrame,
    data_normalized: pd.DataFrame,
    data_batch_corrected: pd.DataFrame | None = None,
    sample_col: str = "replicate_name",
    precursor_col: str = "precursor_id",
    abundance_col_original: str = "abundance",
    abundance_col_normalized: str = "abundance",
    abundance_col_corrected: str = "abundance",
    sample_groups: dict[str, str] | None = None,
    figsize: tuple[int, int] = (18, 6),
    show_plot: bool = True,
    save_path: str | None = None,
) -> plt.Figure | None:
    """Create comparative PCA plots showing effects of normalization and batch correction.

    Args:
        data_original: Long-format DataFrame with original data
        data_normalized: Long-format DataFrame with normalized data
        data_batch_corrected: Optional long-format DataFrame with batch-corrected data
        sample_col: Column name with sample identifiers
        precursor_col: Column name with peptide/precursor identifiers
        abundance_col_original: Abundance column in original data
        abundance_col_normalized: Abundance column in normalized data
        abundance_col_corrected: Abundance column in batch-corrected data
        sample_groups: Dict mapping sample name to group (for coloring)
        figsize: Figure size
        show_plot: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure if show_plot is False, else None

    """
    _check_matplotlib()
    _check_sklearn()

    n_plots = 3 if data_batch_corrected is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    def perform_pca(df, abundance_col, log_transform=True):
        """Perform PCA and return results."""
        wide_data = _pivot_to_wide(df, precursor_col, sample_col, abundance_col)
        if log_transform:
            wide_data = np.log2(wide_data.replace(0, np.nan))
        wide_data = wide_data.dropna(axis=0, thresh=len(wide_data.columns) * 0.5)
        wide_data = wide_data.fillna(wide_data.median())

        if wide_data.shape[0] < 2:
            return None, None

        pca_input = wide_data.T
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pca_input)

        n_comp = min(2, pca_input.shape[0] - 1, pca_input.shape[1])
        pca = PCA(n_components=n_comp)
        scores = pca.fit_transform(scaled_data)

        pca_df = pd.DataFrame(
            scores[:, :2],
            columns=["PC1", "PC2"],
            index=pca_input.index,
        )
        return pca_df, pca

    def plot_single_pca(ax, pca_df, pca, title):
        """Plot a single PCA panel."""
        if pca_df is None or pca is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            return

        # Add group info
        if sample_groups:
            pca_df["Group"] = [sample_groups.get(s, "Unknown") for s in pca_df.index]
        else:
            pca_df["Group"] = "All"

        unique_groups = pca_df["Group"].unique()
        cmap = plt.colormaps.get_cmap("Set1")
        group_colors = {g: cmap(i / max(1, len(unique_groups) - 1)) for i, g in enumerate(unique_groups)}

        for group in unique_groups:
            group_data = pca_df[pca_df["Group"] == group]
            ax.scatter(
                group_data["PC1"],
                group_data["PC2"],
                c=[group_colors[group]],
                label=group,
                alpha=0.7,
                s=80,
                edgecolors="black",
                linewidth=0.5,
            )

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=11)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        if len(unique_groups) > 1:
            ax.legend(fontsize=8)

    # Perform PCA on each dataset
    pca_orig, pca_obj_orig = perform_pca(data_original, abundance_col_original)
    pca_norm, pca_obj_norm = perform_pca(data_normalized, abundance_col_normalized)

    # Plot
    plot_single_pca(axes[0], pca_orig, pca_obj_orig, "Original Data")
    plot_single_pca(axes[1], pca_norm, pca_obj_norm, "After Normalization")

    if data_batch_corrected is not None:
        pca_corr, pca_obj_corr = perform_pca(data_batch_corrected, abundance_col_corrected)
        plot_single_pca(axes[2], pca_corr, pca_obj_corr, "After Batch Correction")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show_plot:
        plt.show()
        return None
    else:
        return fig


def plot_control_correlation_heatmap(
    data: pd.DataFrame,
    sample_col: str = "replicate_name",
    precursor_col: str = "precursor_id",
    abundance_col: str = "abundance",
    sample_type_col: str = "sample_type",
    control_types: list[str] | None = None,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    log_transform: bool = True,
    title: str = "Control Sample Correlation",
    figsize: tuple[int, int] = (10, 8),
    show_plot: bool = True,
    save_path: str | None = None,
) -> tuple[plt.Figure | None, pd.DataFrame]:
    """Create correlation heatmap for control samples (reference, qc).

    Args:
        data: Long-format DataFrame with abundance data
        sample_col: Column name with sample identifiers
        precursor_col: Column name with peptide/precursor identifiers
        abundance_col: Column name with abundance values
        sample_type_col: Column name with sample type
        control_types: List of sample types to include (default: ['reference', 'qc'])
        method: Correlation method
        log_transform: Whether to log2-transform data
        title: Plot title
        figsize: Figure size
        show_plot: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        Tuple of (Figure if show_plot is False, correlation matrix DataFrame)

    """
    _check_matplotlib()
    _check_seaborn()

    if control_types is None:
        control_types = ["reference", "qc"]

    # Filter to control samples
    mask = data[sample_type_col].isin(control_types)
    control_data = data.loc[mask]

    if len(control_data) == 0:
        logger.warning("No control samples found in data")
        return None, pd.DataFrame()

    # Pivot to wide format
    wide_data = _pivot_to_wide(control_data, precursor_col, sample_col, abundance_col)

    if log_transform:
        wide_data = np.log2(wide_data.replace(0, np.nan))

    # Drop rows with too many missing values
    wide_data = wide_data.dropna(axis=0, thresh=len(wide_data.columns) * 0.5)

    if wide_data.shape[0] == 0:
        logger.warning("No valid data for correlation after filtering")
        return None, pd.DataFrame()

    # Calculate correlation
    corr_matrix = wide_data.corr(method=method)

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    # Use seaborn heatmap
    mask_nan = corr_matrix.isna()
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdYlBu_r",
        center=0 if corr_matrix.min().min() < 0 else None,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": f"{method.capitalize()} Correlation"},
        ax=ax,
        mask=mask_nan,
        vmin=max(-1, corr_matrix.min().min()),
        vmax=1,
        annot_kws={"size": 8},
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    plt.tight_layout()

    # Print summary statistics
    off_diagonal = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
    off_diagonal = off_diagonal[~np.isnan(off_diagonal)]

    if len(off_diagonal) > 0:
        logger.info(f"Correlation Summary ({title}):")
        logger.info(f"  Mean correlation: {np.mean(off_diagonal):.3f}")
        logger.info(f"  Median correlation: {np.median(off_diagonal):.3f}")
        logger.info(f"  Min correlation: {np.min(off_diagonal):.3f}")
        logger.info(f"  Max correlation: {np.max(off_diagonal):.3f}")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show_plot:
        plt.show()
        return None, corr_matrix
    else:
        return fig, corr_matrix


def plot_cv_distribution(
    data: pd.DataFrame,
    sample_col: str = "replicate_name",
    precursor_col: str = "precursor_id",
    abundance_col: str = "abundance",
    sample_type_col: str = "sample_type",
    control_types: list[str] | None = None,
    cv_threshold: float = 20.0,
    title: str = "CV Distribution for Control Samples",
    figsize: tuple[int, int] | None = None,
    show_plot: bool = True,
    save_path: str | None = None,
) -> tuple[plt.Figure | None, dict[str, list[float]]]:
    """Create CV distribution histograms for control samples.

    Args:
        data: Long-format DataFrame with abundance data
        sample_col: Column name with sample identifiers
        precursor_col: Column name with peptide/precursor identifiers
        abundance_col: Column name with abundance values
        sample_type_col: Column name with sample type
        control_types: List of sample types to analyze (default: ['reference', 'qc'])
        cv_threshold: CV threshold line to display (%)
        title: Plot title
        figsize: Figure size (auto-calculated if None)
        show_plot: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        Tuple of (Figure if show_plot is False, dict mapping control type to CV values)

    """
    _check_matplotlib()

    if control_types is None:
        control_types = ["reference", "qc"]

    if figsize is None:
        figsize = (6 * len(control_types), 5)

    # Calculate CV for each control type
    cv_data = {}

    for control_type in control_types:
        # Get samples of this type
        type_mask = data[sample_type_col] == control_type
        type_data = data.loc[type_mask]
        type_samples = type_data[sample_col].unique()

        if len(type_samples) < 2:
            logger.warning(f"Insufficient samples for {control_type} CV calculation")
            cv_data[control_type] = []
            continue

        # Pivot to wide format
        wide_data = _pivot_to_wide(type_data, precursor_col, sample_col, abundance_col)

        # Calculate CV per peptide (on linear scale)
        linear = wide_data.copy()  # Already linear
        cv_values = []

        for idx in linear.index:
            row = linear.loc[idx].dropna()
            if len(row) >= 2 and row.mean() > 0:
                cv = (row.std() / row.mean()) * 100
                cv_values.append(cv)

        cv_data[control_type] = cv_values

    # Create plots
    fig, axes = plt.subplots(1, len(control_types), figsize=figsize)
    if len(control_types) == 1:
        axes = [axes]

    for idx, control_type in enumerate(control_types):
        ax = axes[idx]
        cv_values = cv_data.get(control_type, [])

        if len(cv_values) > 0:
            # Histogram
            ax.hist(
                cv_values,
                bins=50,
                alpha=0.7,
                color="skyblue",
                edgecolor="black",
                linewidth=0.5,
            )

            # Statistics
            median_cv = np.median(cv_values)
            mean_cv = np.mean(cv_values)
            pct_under_threshold = np.sum(np.array(cv_values) < cv_threshold) / len(cv_values) * 100
            pct_under_10 = np.sum(np.array(cv_values) < 10) / len(cv_values) * 100

            # Median line
            ax.axvline(
                median_cv,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Median: {median_cv:.1f}%",
            )

            # Threshold line
            ax.axvline(
                cv_threshold,
                color="orange",
                linestyle=":",
                linewidth=2,
                alpha=0.8,
                label=f"Threshold: {cv_threshold}%",
            )

            # Stats text
            stats_text = f"{pct_under_threshold:.1f}% < {cv_threshold}% CV\n{pct_under_10:.1f}% < 10% CV"
            ax.text(
                0.95,
                0.95,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
            )

            ax.set_xlim(0, min(100, np.percentile(cv_values, 99)))

            # Print statistics
            logger.info(f"{control_type} CV Statistics:")
            logger.info(f"  Median CV: {median_cv:.2f}%")
            logger.info(f"  Mean CV: {mean_cv:.2f}%")
            logger.info(f"  % < {cv_threshold}%: {pct_under_threshold:.1f}%")

        else:
            ax.text(
                0.5,
                0.5,
                f"Insufficient data\nfor {control_type}",
                transform=ax.transAxes,
                fontsize=12,
                ha="center",
                va="center",
            )

        n_samples = len(data.loc[data[sample_type_col] == control_type, sample_col].unique())
        ax.set_title(
            f"{control_type.capitalize()} Controls\n({n_samples} samples)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Coefficient of Variation (%)", fontsize=11)
        ax.set_ylabel("Number of Peptides", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show_plot:
        plt.show()
        return None, cv_data
    else:
        return fig, cv_data


def plot_comparative_cv(
    data_before: pd.DataFrame,
    data_after: pd.DataFrame,
    sample_col: str = "replicate_name",
    precursor_col: str = "precursor_id",
    abundance_col_before: str = "abundance",
    abundance_col_after: str = "abundance",
    sample_type_col: str = "sample_type",
    control_type: str = "reference",
    cv_threshold: float = 20.0,
    figsize: tuple[int, int] = (12, 5),
    show_plot: bool = True,
    save_path: str | None = None,
) -> plt.Figure | None:
    """Compare CV distributions before and after normalization.

    Args:
        data_before: Long-format DataFrame before normalization
        data_after: Long-format DataFrame after normalization
        sample_col: Column name with sample identifiers
        precursor_col: Column name with peptide/precursor identifiers
        abundance_col_before: Abundance column in before data
        abundance_col_after: Abundance column in after data
        sample_type_col: Column name with sample type
        control_type: Sample type to analyze
        cv_threshold: CV threshold line to display (%)
        figsize: Figure size
        show_plot: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure if show_plot is False, else None

    """
    _check_matplotlib()

    def calc_cvs(df, abundance_col):
        """Calculate CV values for control samples."""
        type_mask = df[sample_type_col] == control_type
        type_data = df.loc[type_mask]

        if len(type_data[sample_col].unique()) < 2:
            return []

        wide_data = _pivot_to_wide(type_data, precursor_col, sample_col, abundance_col)
        cv_values = []

        for idx in wide_data.index:
            row = wide_data.loc[idx].dropna()
            if len(row) >= 2 and row.mean() > 0:
                cv = (row.std() / row.mean()) * 100
                cv_values.append(cv)

        return cv_values

    cv_before = calc_cvs(data_before, abundance_col_before)
    cv_after = calc_cvs(data_after, abundance_col_after)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for ax, cv_values, label in [
        (ax1, cv_before, "Before Normalization"),
        (ax2, cv_after, "After Normalization"),
    ]:
        if len(cv_values) > 0:
            ax.hist(cv_values, bins=50, alpha=0.7, color="skyblue", edgecolor="black")

            median_cv = np.median(cv_values)
            pct_under = np.sum(np.array(cv_values) < cv_threshold) / len(cv_values) * 100

            ax.axvline(median_cv, color="red", linestyle="--", linewidth=2)
            ax.axvline(cv_threshold, color="orange", linestyle=":", linewidth=2, alpha=0.8)

            ax.text(
                0.95,
                0.95,
                f"Median: {median_cv:.1f}%\n{pct_under:.1f}% < {cv_threshold}%",
                transform=ax.transAxes,
                fontsize=10,
                va="top",
                ha="right",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
            )

            ax.set_xlim(0, min(100, np.percentile(cv_values, 99)))
        else:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center", va="center")

        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Coefficient of Variation (%)", fontsize=11)
        ax.set_ylabel("Number of Peptides", fontsize=11)
        ax.grid(True, alpha=0.3)

    # Calculate improvement
    if cv_before and cv_after:
        median_before = np.median(cv_before)
        median_after = np.median(cv_after)
        improvement = (median_before - median_after) / median_before * 100 if median_before > 0 else 0

        fig.suptitle(
            f"CV Distribution Comparison - {control_type.capitalize()} Samples\n"
            f"Median CV: {median_before:.1f}% → {median_after:.1f}% ({improvement:.1f}% improvement)",
            fontsize=14,
            fontweight="bold",
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show_plot:
        plt.show()
        return None
    else:
        return fig


def plot_sample_correlation_matrix(
    data: pd.DataFrame,
    sample_col: str = "replicate_name",
    precursor_col: str = "precursor_id",
    abundance_col: str = "abundance",
    sample_types: dict[str, str] | None = None,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    log_transform: bool = True,
    title: str = "Sample Correlation Matrix",
    figsize: tuple[int, int] = (12, 10),
    show_plot: bool = True,
    save_path: str | None = None,
) -> tuple[plt.Figure | None, pd.DataFrame]:
    """Create correlation heatmap for all samples.

    Args:
        data: Long-format DataFrame with abundance data
        sample_col: Column name with sample identifiers
        precursor_col: Column name with peptide/precursor identifiers
        abundance_col: Column name with abundance values
        sample_types: Dict mapping sample name to type (for annotation)
        method: Correlation method
        log_transform: Whether to log2-transform data
        title: Plot title
        figsize: Figure size
        show_plot: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        Tuple of (Figure if show_plot is False, correlation matrix DataFrame)

    """
    _check_matplotlib()
    _check_seaborn()

    # Pivot to wide format
    wide_data = _pivot_to_wide(data, precursor_col, sample_col, abundance_col)

    if log_transform:
        wide_data = np.log2(wide_data.replace(0, np.nan))

    # Drop rows with too many missing values
    wide_data = wide_data.dropna(axis=0, thresh=len(wide_data.columns) * 0.5)

    # Calculate correlation
    corr_matrix = wide_data.corr(method=method)

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    # Determine if annotations are feasible
    n_samples = len(corr_matrix)
    annot = n_samples <= 30

    sns.heatmap(
        corr_matrix,
        annot=annot,
        fmt=".2f" if annot else None,
        cmap="RdYlBu_r",
        square=True,
        linewidths=0.5 if n_samples <= 50 else 0,
        cbar_kws={"label": f"{method.capitalize()} Correlation"},
        ax=ax,
        vmin=max(-1, corr_matrix.min().min()),
        vmax=1,
        annot_kws={"size": 6} if annot else None,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")

    if n_samples <= 30:
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
    else:
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show_plot:
        plt.show()
        return None, corr_matrix
    else:
        return fig, corr_matrix


def plot_rt_residuals(
    data: pd.DataFrame,
    reference_stats: pd.DataFrame,
    sample_col: str = "replicate_name",
    precursor_col: str = "precursor_id",
    abundance_col: str = "abundance",
    rt_col: str = "retention_time",
    sample_type_col: str = "sample_type",
    reference_type: str = "reference",
    qc_type: str = "qc",
    title: str = "RT-Dependent Residuals",
    n_samples_to_show: int = 4,
    figsize: tuple[int, int] | None = None,
    show_plot: bool = True,
    save_path: str | None = None,
) -> tuple[plt.Figure | None, pd.DataFrame]:
    """Plot abundance residuals vs retention time for reference and QC samples.

    Shows how abundances deviate from the reference mean as a function of RT.
    This is useful for visualizing RT-dependent bias before/after correction.

    Args:
        data: Long-format DataFrame with abundance data
        reference_stats: DataFrame with per-precursor reference statistics
            (must have 'mean_abundance' column, indexed by precursor)
        sample_col: Column name with sample identifiers
        precursor_col: Column name with peptide/precursor identifiers
        abundance_col: Column name with abundance values
        rt_col: Column name with retention times
        sample_type_col: Column name with sample type
        reference_type: Sample type value for reference samples
        qc_type: Sample type value for QC samples
        title: Plot title
        n_samples_to_show: Number of samples of each type to display
        figsize: Figure size (auto-calculated if None)
        show_plot: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        Tuple of (Figure if show_plot is False, residuals DataFrame)

    """
    _check_matplotlib()

    # Merge reference statistics to get expected abundance
    merged = data.merge(
        reference_stats[["mean_abundance"]],
        left_on=precursor_col,
        right_index=True,
        how="inner",
    )

    if len(merged) == 0:
        logger.warning("No matching precursors between data and reference_stats")
        return None, pd.DataFrame()

    # Calculate residuals (log2 scale)
    merged["log2_abundance"] = np.log2(merged[abundance_col].replace(0, np.nan))
    merged["log2_expected"] = np.log2(merged["mean_abundance"].replace(0, np.nan))
    merged["residual"] = merged["log2_abundance"] - merged["log2_expected"]

    # Filter to reference and QC samples
    ref_samples = merged.loc[
        merged[sample_type_col] == reference_type, sample_col
    ].unique()
    qc_samples = merged.loc[
        merged[sample_type_col] == qc_type, sample_col
    ].unique()

    # Select subset of samples to show
    ref_to_show = ref_samples[: min(n_samples_to_show, len(ref_samples))]
    qc_to_show = qc_samples[: min(n_samples_to_show, len(qc_samples))]

    n_ref = len(ref_to_show)
    n_qc = len(qc_to_show)
    n_cols = max(n_ref, n_qc)

    if n_cols == 0:
        logger.warning("No reference or QC samples found")
        return None, merged

    if figsize is None:
        figsize = (4 * n_cols, 8)

    fig, axes = plt.subplots(2, n_cols, figsize=figsize, squeeze=False)

    # Color settings
    ref_color = "#2ca02c"  # Green for reference
    qc_color = "#ff7f0e"  # Orange for QC

    def plot_sample_residuals(ax, sample_name, color, sample_type_label):
        """Plot residuals for a single sample."""
        sample_data = merged[merged[sample_col] == sample_name].dropna(
            subset=["residual", rt_col]
        )

        if len(sample_data) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return

        rt = sample_data[rt_col].values
        residuals = sample_data["residual"].values

        # Scatter plot
        ax.scatter(rt, residuals, alpha=0.3, s=10, c=color, edgecolors="none")

        # Add LOESS/smoothed trend line
        try:
            from scipy.ndimage import uniform_filter1d

            # Sort by RT and compute rolling median
            sort_idx = np.argsort(rt)
            rt_sorted = rt[sort_idx]
            res_sorted = residuals[sort_idx]

            # Bin and compute median for smoother line
            n_bins = min(50, len(rt_sorted) // 10)
            if n_bins >= 3:
                bin_edges = np.linspace(rt_sorted.min(), rt_sorted.max(), n_bins + 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_medians = np.zeros(n_bins)

                for i in range(n_bins):
                    mask = (rt_sorted >= bin_edges[i]) & (rt_sorted < bin_edges[i + 1])
                    if mask.sum() > 0:
                        bin_medians[i] = np.median(res_sorted[mask])

                # Smooth the binned medians
                smoothed = uniform_filter1d(bin_medians, size=3, mode="nearest")
                ax.plot(bin_centers, smoothed, color="darkred", linewidth=2, label="Trend")
        except Exception:
            pass

        # Reference line at 0
        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

        # Calculate statistics
        median_res = np.median(residuals)
        mad = np.median(np.abs(residuals - median_res))

        ax.set_title(f"{sample_name}\n(MAD={mad:.3f})", fontsize=10)
        ax.set_xlabel("Retention Time", fontsize=9)
        ax.set_ylabel("Log2 Residual", fontsize=9)
        ax.set_ylim(-3, 3)  # Standard scale for visibility
        ax.grid(True, alpha=0.3)

    # Plot reference samples (top row)
    for i in range(n_cols):
        ax = axes[0, i]
        if i < n_ref:
            plot_sample_residuals(ax, ref_to_show[i], ref_color, "Reference")
        else:
            ax.axis("off")

    axes[0, 0].set_ylabel("Reference Samples\nLog2 Residual", fontsize=10)

    # Plot QC samples (bottom row)
    for i in range(n_cols):
        ax = axes[1, i]
        if i < n_qc:
            plot_sample_residuals(ax, qc_to_show[i], qc_color, "QC")
        else:
            ax.axis("off")

    axes[1, 0].set_ylabel("QC Samples\nLog2 Residual", fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show_plot:
        plt.show()
        return None, merged
    else:
        return fig, merged


def plot_rt_correction_comparison(
    data_before: pd.DataFrame,
    data_after: pd.DataFrame,
    reference_stats: pd.DataFrame,
    sample_col: str = "replicate_name",
    precursor_col: str = "precursor_id",
    abundance_col: str = "abundance",
    rt_col: str = "retention_time",
    sample_type_col: str = "sample_type",
    reference_type: str = "reference",
    qc_type: str = "qc",
    figsize: tuple[int, int] = (14, 10),
    show_plot: bool = True,
    save_path: str | None = None,
) -> plt.Figure | None:
    """Compare RT residuals before and after RT-dependent normalization.

    Creates a 2x2 grid showing:
    - Top row: Reference samples (used for fitting) before/after
    - Bottom row: QC samples (held-out QC) before/after

    Reference samples should show improvement (used for fitting).
    QC samples show how well the correction generalizes to held-out data.

    Args:
        data_before: Long-format DataFrame before RT correction
        data_after: Long-format DataFrame after RT correction
        reference_stats: DataFrame with per-precursor reference statistics
        sample_col: Column name with sample identifiers
        precursor_col: Column name with peptide/precursor identifiers
        abundance_col: Column name with abundance values
        rt_col: Column name with retention times
        sample_type_col: Column name with sample type
        reference_type: Sample type value for reference samples
        qc_type: Sample type value for QC samples
        figsize: Figure size
        show_plot: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure if show_plot is False, else None

    """
    _check_matplotlib()

    def compute_residuals(df: pd.DataFrame) -> pd.DataFrame:
        """Compute residuals for a dataset."""
        merged = df.merge(
            reference_stats[["mean_abundance"]],
            left_on=precursor_col,
            right_index=True,
            how="inner",
        )
        merged["log2_abundance"] = np.log2(merged[abundance_col].replace(0, np.nan))
        merged["log2_expected"] = np.log2(merged["mean_abundance"].replace(0, np.nan))
        merged["residual"] = merged["log2_abundance"] - merged["log2_expected"]
        return merged

    residuals_before = compute_residuals(data_before)
    residuals_after = compute_residuals(data_after)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    def plot_aggregated_residuals(ax, df, sample_type, color, title):
        """Plot aggregated RT residuals for all samples of a type."""
        type_data = df[df[sample_type_col] == sample_type].dropna(
            subset=["residual", rt_col]
        )

        if len(type_data) == 0:
            ax.text(0.5, 0.5, f"No {sample_type} data", ha="center", va="center", transform=ax.transAxes)
            return 0, 0

        rt = type_data[rt_col].values
        residuals = type_data["residual"].values

        # Scatter with low alpha for density
        ax.scatter(rt, residuals, alpha=0.1, s=5, c=color, edgecolors="none")

        # Compute binned statistics
        n_bins = 50
        bin_edges = np.linspace(rt.min(), rt.max(), n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_medians = np.zeros(n_bins)
        bin_mads = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (rt >= bin_edges[i]) & (rt < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_res = residuals[mask]
                bin_medians[i] = np.median(bin_res)
                bin_mads[i] = np.median(np.abs(bin_res - bin_medians[i]))

        # Plot trend line
        ax.plot(bin_centers, bin_medians, color="darkred", linewidth=2, label="Median")

        # Plot MAD envelope
        ax.fill_between(
            bin_centers,
            bin_medians - bin_mads,
            bin_medians + bin_mads,
            alpha=0.3,
            color="darkred",
            label="±MAD",
        )

        # Reference line
        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

        # Statistics
        overall_mad = np.median(np.abs(residuals - np.median(residuals)))
        rt_bias = np.max(np.abs(bin_medians))

        ax.set_title(f"{title}\nMAD={overall_mad:.3f}, Max RT bias={rt_bias:.3f}", fontsize=11)
        ax.set_xlabel("Retention Time", fontsize=10)
        ax.set_ylabel("Log2 Residual", fontsize=10)
        ax.set_ylim(-2, 2)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

        return overall_mad, rt_bias

    # Reference samples
    ref_mad_before, ref_bias_before = plot_aggregated_residuals(
        axes[0, 0], residuals_before, reference_type, "#2ca02c", "Reference - Before RT Correction"
    )
    ref_mad_after, ref_bias_after = plot_aggregated_residuals(
        axes[0, 1], residuals_after, reference_type, "#2ca02c", "Reference - After RT Correction"
    )

    # QC samples (held-out QC)
    qc_mad_before, qc_bias_before = plot_aggregated_residuals(
        axes[1, 0], residuals_before, qc_type, "#ff7f0e", "QC - Before RT Correction"
    )
    qc_mad_after, qc_bias_after = plot_aggregated_residuals(
        axes[1, 1], residuals_after, qc_type, "#ff7f0e", "QC - After RT Correction"
    )

    # Calculate improvements
    ref_mad_improvement = (ref_mad_before - ref_mad_after) / ref_mad_before * 100 if ref_mad_before > 0 else 0
    qc_mad_improvement = (qc_mad_before - qc_mad_after) / qc_mad_before * 100 if qc_mad_before > 0 else 0
    ref_bias_improvement = (ref_bias_before - ref_bias_after) / ref_bias_before * 100 if ref_bias_before > 0 else 0
    qc_bias_improvement = (qc_bias_before - qc_bias_after) / qc_bias_before * 100 if qc_bias_before > 0 else 0

    fig.suptitle(
        f"RT-Dependent Normalization QC\n"
        f"Reference: MAD {ref_mad_improvement:+.1f}%, RT bias {ref_bias_improvement:+.1f}% | "
        f"QC (held-out): MAD {qc_mad_improvement:+.1f}%, RT bias {qc_bias_improvement:+.1f}%",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    # Log summary
    logger.info("RT Correction QC Summary:")
    logger.info(f"  Reference MAD: {ref_mad_before:.3f} -> {ref_mad_after:.3f} ({ref_mad_improvement:+.1f}%)")
    logger.info(f"  Reference RT bias: {ref_bias_before:.3f} -> {ref_bias_after:.3f} ({ref_bias_improvement:+.1f}%)")
    logger.info(f"  QC MAD: {qc_mad_before:.3f} -> {qc_mad_after:.3f} ({qc_mad_improvement:+.1f}%)")
    logger.info(f"  QC RT bias: {qc_bias_before:.3f} -> {qc_bias_after:.3f} ({qc_bias_improvement:+.1f}%)")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show_plot:
        plt.show()
        return None
    else:
        return fig


def plot_rt_correction_per_sample(
    data_before: pd.DataFrame,
    data_after: pd.DataFrame,
    reference_stats: pd.DataFrame,
    sample_col: str = "replicate_name",
    precursor_col: str = "precursor_id",
    abundance_col: str = "abundance",
    rt_col: str = "retention_time",
    sample_type_col: str = "sample_type",
    reference_type: str = "reference",
    qc_type: str = "qc",
    n_samples: int = 3,
    figsize: tuple[int, int] | None = None,
    show_plot: bool = True,
    save_path: str | None = None,
) -> plt.Figure | None:
    """Show RT residuals for individual samples before/after correction.

    Creates a grid showing before/after for selected reference and QC samples.

    Args:
        data_before: Long-format DataFrame before RT correction
        data_after: Long-format DataFrame after RT correction
        reference_stats: DataFrame with per-precursor reference statistics
        sample_col: Column name with sample identifiers
        precursor_col: Column name with peptide/precursor identifiers
        abundance_col: Column name with abundance values
        rt_col: Column name with retention times
        sample_type_col: Column name with sample type
        reference_type: Sample type value for reference samples
        qc_type: Sample type value for QC samples
        n_samples: Number of samples of each type to show
        figsize: Figure size (auto-calculated if None)
        show_plot: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure if show_plot is False, else None

    """
    _check_matplotlib()

    def compute_residuals(df: pd.DataFrame) -> pd.DataFrame:
        merged = df.merge(
            reference_stats[["mean_abundance"]],
            left_on=precursor_col,
            right_index=True,
            how="inner",
        )
        merged["log2_abundance"] = np.log2(merged[abundance_col].replace(0, np.nan))
        merged["log2_expected"] = np.log2(merged["mean_abundance"].replace(0, np.nan))
        merged["residual"] = merged["log2_abundance"] - merged["log2_expected"]
        return merged

    residuals_before = compute_residuals(data_before)
    residuals_after = compute_residuals(data_after)

    # Get sample names
    ref_samples = residuals_before.loc[
        residuals_before[sample_type_col] == reference_type, sample_col
    ].unique()[:n_samples]
    qc_samples = residuals_before.loc[
        residuals_before[sample_type_col] == qc_type, sample_col
    ].unique()[:n_samples]

    all_samples = list(ref_samples) + list(qc_samples)
    sample_types_list = [reference_type] * len(ref_samples) + [qc_type] * len(qc_samples)

    if len(all_samples) == 0:
        logger.warning("No reference or QC samples found")
        return None

    n_rows = len(all_samples)
    if figsize is None:
        figsize = (10, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, 2, figsize=figsize, squeeze=False)

    colors = {reference_type: "#2ca02c", qc_type: "#ff7f0e"}

    for row_idx, (sample_name, sample_type) in enumerate(zip(all_samples, sample_types_list)):
        color = colors.get(sample_type, "#1f77b4")

        for col_idx, (df, stage_label) in enumerate([
            (residuals_before, "Before"),
            (residuals_after, "After"),
        ]):
            ax = axes[row_idx, col_idx]
            sample_data = df[df[sample_col] == sample_name].dropna(subset=["residual", rt_col])

            if len(sample_data) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue

            rt = sample_data[rt_col].values
            residuals = sample_data["residual"].values

            ax.scatter(rt, residuals, alpha=0.3, s=10, c=color, edgecolors="none")

            # Binned trend
            n_bins = min(30, len(rt) // 10)
            if n_bins >= 3:
                bin_edges = np.linspace(rt.min(), rt.max(), n_bins + 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_medians = np.zeros(n_bins)

                for i in range(n_bins):
                    mask = (rt >= bin_edges[i]) & (rt < bin_edges[i + 1])
                    if mask.sum() > 0:
                        bin_medians[i] = np.median(residuals[mask])

                ax.plot(bin_centers, bin_medians, color="darkred", linewidth=2)

            ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

            mad = np.median(np.abs(residuals - np.median(residuals)))
            type_label = "Ref" if sample_type == reference_type else "QC"
            ax.set_title(f"{sample_name} ({type_label}) - {stage_label}\nMAD={mad:.3f}", fontsize=10)
            ax.set_ylim(-2, 2)
            ax.grid(True, alpha=0.3)

            if col_idx == 0:
                ax.set_ylabel("Log2 Residual", fontsize=9)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Retention Time", fontsize=9)

    fig.suptitle(
        "RT-Dependent Normalization: Per-Sample Comparison",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show_plot:
        plt.show()
        return None
    else:
        return fig


# =============================================================================
# Wide-Format Data Visualization Functions
# =============================================================================
# These functions work with wide-format DataFrames where:
# - Rows are features (peptides or proteins)
# - Columns are samples (plus optional metadata columns)
# - Values are log2-transformed abundances


def plot_intensity_distribution_wide(
    data: pd.DataFrame,
    sample_cols: list[str],
    sample_types: dict[str, str] | None = None,
    title: str = "Intensity Distribution",
    figsize: tuple[int, int] = (14, 6),
    show_plot: bool = True,
    save_path: str | None = None,
) -> plt.Figure | None:
    """Create box plot showing intensity distribution across samples (wide-format data).

    Args:
        data: Wide-format DataFrame (features x samples)
        sample_cols: List of column names containing sample abundances
        sample_types: Dict mapping sample name to type for coloring
        title: Plot title
        figsize: Figure size (width, height)
        show_plot: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure if show_plot is False, else None

    """
    _check_matplotlib()

    # Prepare data for boxplot
    box_data = []
    for sample in sample_cols:
        sample_data = data[sample].dropna()
        box_data.append(sample_data.values)

    # Get colors
    colors, legend = _get_sample_colors(sample_cols, sample_types)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create box plot
    bp = ax.boxplot(
        box_data,
        positions=range(len(sample_cols)),
        widths=0.6,
        patch_artist=True,
        showfliers=False,
    )

    # Color boxes
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Styling
    ax.set_xlabel("Sample", fontsize=12)
    ax.set_ylabel("Log2 Abundance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # X-axis labels
    if len(sample_cols) <= 20:
        ax.set_xticks(range(len(sample_cols)))
        ax.set_xticklabels(sample_cols, rotation=45, ha="right", fontsize=8)
    else:
        n = max(1, len(sample_cols) // 20)
        ax.set_xticks(range(0, len(sample_cols), n))
        ax.set_xticklabels(
            [sample_cols[i] for i in range(0, len(sample_cols), n)],
            rotation=45,
            ha="right",
            fontsize=8,
        )

    # Legend
    if legend:
        patches = [
            mpatches.Patch(color=color, label=label, alpha=0.7)
            for label, color in legend.items()
        ]
        ax.legend(handles=patches, loc="upper right")

    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show_plot:
        plt.show()
        return None
    else:
        return fig


def plot_normalization_comparison_wide(
    data_before: pd.DataFrame,
    data_after: pd.DataFrame,
    sample_cols: list[str],
    title: str = "Normalization Comparison",
    figsize: tuple[int, int] = (16, 6),
    show_plot: bool = True,
    save_path: str | None = None,
) -> plt.Figure | None:
    """Create side-by-side density plots comparing before/after (wide-format data).

    Args:
        data_before: Wide-format DataFrame before normalization
        data_after: Wide-format DataFrame after normalization
        sample_cols: List of column names containing sample abundances
        title: Plot title
        figsize: Figure size
        show_plot: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure if show_plot is False, else None

    """
    _check_matplotlib()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    cmap = plt.colormaps.get_cmap("tab20")

    # Before normalization
    for i, sample in enumerate(sample_cols):
        values = data_before[sample].dropna().values
        if len(values) > 0:
            counts, bins = np.histogram(values, bins=50, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax1.plot(bin_centers, counts, alpha=0.6, linewidth=1, color=cmap(i % 20))

    ax1.set_xlabel("Log2 Abundance", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Before Normalization", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # After normalization
    for i, sample in enumerate(sample_cols):
        values = data_after[sample].dropna().values
        if len(values) > 0:
            counts, bins = np.histogram(values, bins=50, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax2.plot(bin_centers, counts, alpha=0.6, linewidth=1, color=cmap(i % 20))

    ax2.set_xlabel("Log2 Abundance", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("After Normalization", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Calculate and display median range
    medians_before = data_before[sample_cols].median()
    medians_after = data_after[sample_cols].median()
    range_before = medians_before.max() - medians_before.min()
    range_after = medians_after.max() - medians_after.min()
    reduction = (range_before - range_after) / range_before * 100 if range_before > 0 else 0

    fig.suptitle(
        f"{title}\nMedian range: {range_before:.2f} -> {range_after:.2f} "
        f"({reduction:.1f}% reduction)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show_plot:
        plt.show()
        return None
    else:
        return fig


def plot_pca_wide(
    data: pd.DataFrame,
    sample_cols: list[str],
    sample_types: dict[str, str] | None = None,
    title: str = "PCA Analysis",
    n_components: int = 2,
    figsize: tuple[int, int] = (10, 8),
    show_plot: bool = True,
    save_path: str | None = None,
) -> tuple[plt.Figure | None, pd.DataFrame]:
    """Create PCA plot from wide-format data.

    Args:
        data: Wide-format DataFrame (features x samples)
        sample_cols: List of column names containing sample abundances
        sample_types: Dict mapping sample name to type (for coloring)
        title: Plot title
        n_components: Number of PCA components to compute
        figsize: Figure size
        show_plot: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        Tuple of (Figure if show_plot is False, PCA results DataFrame)

    """
    _check_matplotlib()
    _check_sklearn()

    # Extract sample data and transpose (samples as rows)
    sample_data = data[sample_cols].copy()
    sample_data = sample_data.dropna(axis=0, thresh=len(sample_cols) * 0.5)
    sample_data = sample_data.fillna(sample_data.median())

    if sample_data.shape[0] < n_components:
        logger.warning(f"Too few features for PCA: {sample_data.shape[0]}")
        return None, pd.DataFrame()

    # Transpose: samples as rows, features as columns
    pca_input = sample_data.T

    # Standardize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pca_input)

    # PCA
    n_comp = min(n_components, pca_input.shape[0] - 1, pca_input.shape[1])
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(scaled_data)

    # Results DataFrame
    pca_df = pd.DataFrame(
        scores[:, :2],
        columns=["PC1", "PC2"],
        index=pca_input.index,
    )
    pca_df["Sample"] = pca_df.index

    # Add grouping info
    if sample_types:
        pca_df["Group"] = [sample_types.get(s, "unknown") for s in pca_df.index]
    else:
        pca_df["Group"] = "All"

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    unique_groups = pca_df["Group"].unique()
    cmap = plt.colormaps.get_cmap("Set1")
    group_colors = {
        g: cmap(i / max(1, len(unique_groups) - 1)) for i, g in enumerate(unique_groups)
    }

    for group in unique_groups:
        group_data = pca_df[pca_df["Group"] == group]
        ax.scatter(
            group_data["PC1"],
            group_data["PC2"],
            c=[group_colors[group]],
            label=group,
            alpha=0.7,
            s=80,
            edgecolors="black",
            linewidth=0.5,
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if len(unique_groups) > 1:
        ax.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show_plot:
        plt.show()
        return None, pca_df
    else:
        return fig, pca_df


def plot_comparative_pca_wide(
    data_before: pd.DataFrame,
    data_after: pd.DataFrame,
    sample_cols: list[str],
    sample_types: dict[str, str] | None = None,
    title: str = "PCA Comparison",
    figsize: tuple[int, int] = (14, 6),
    show_plot: bool = True,
    save_path: str | None = None,
) -> plt.Figure | None:
    """Compare PCA before and after normalization (wide-format data).

    Args:
        data_before: Wide-format DataFrame before normalization
        data_after: Wide-format DataFrame after normalization
        sample_cols: List of column names containing sample abundances
        sample_types: Dict mapping sample name to type (for coloring)
        title: Plot title
        figsize: Figure size
        show_plot: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure if show_plot is False, else None

    """
    _check_matplotlib()
    _check_sklearn()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    def perform_pca_and_plot(df, ax, plot_title):
        """Perform PCA and plot on given axis."""
        sample_data = df[sample_cols].copy()
        sample_data = sample_data.dropna(axis=0, thresh=len(sample_cols) * 0.5)
        sample_data = sample_data.fillna(sample_data.median())

        if sample_data.shape[0] < 2:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(plot_title)
            return

        pca_input = sample_data.T
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pca_input)

        n_comp = min(2, pca_input.shape[0] - 1, pca_input.shape[1])
        pca = PCA(n_components=n_comp)
        scores = pca.fit_transform(scaled_data)

        pca_df = pd.DataFrame(
            scores[:, :2],
            columns=["PC1", "PC2"],
            index=pca_input.index,
        )

        if sample_types:
            pca_df["Group"] = [sample_types.get(s, "unknown") for s in pca_df.index]
        else:
            pca_df["Group"] = "All"

        unique_groups = pca_df["Group"].unique()
        cmap = plt.colormaps.get_cmap("Set1")
        group_colors = {
            g: cmap(i / max(1, len(unique_groups) - 1))
            for i, g in enumerate(unique_groups)
        }

        for group in unique_groups:
            group_data = pca_df[pca_df["Group"] == group]
            ax.scatter(
                group_data["PC1"],
                group_data["PC2"],
                c=[group_colors[group]],
                label=group,
                alpha=0.7,
                s=80,
                edgecolors="black",
                linewidth=0.5,
            )

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=11)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=11)
        ax.set_title(plot_title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        if len(unique_groups) > 1:
            ax.legend(fontsize=8)

    perform_pca_and_plot(data_before, ax1, "Before Normalization")
    perform_pca_and_plot(data_after, ax2, "After Normalization")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show_plot:
        plt.show()
        return None
    else:
        return fig


def plot_cv_comparison_wide(
    data_before: pd.DataFrame,
    data_after: pd.DataFrame,
    sample_cols: list[str],
    sample_types: dict[str, str] | None = None,
    control_type: str = "reference",
    cv_threshold: float = 20.0,
    title: str = "CV Distribution Comparison",
    figsize: tuple[int, int] = (12, 5),
    show_plot: bool = True,
    save_path: str | None = None,
) -> plt.Figure | None:
    """Compare CV distributions before and after normalization (wide-format data).

    Args:
        data_before: Wide-format DataFrame before normalization
        data_after: Wide-format DataFrame after normalization
        sample_cols: List of column names containing sample abundances
        sample_types: Dict mapping sample name to type
        control_type: Sample type to analyze ('reference' or 'qc')
        cv_threshold: CV threshold line to display (%)
        title: Plot title
        figsize: Figure size
        show_plot: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure if show_plot is False, else None

    """
    _check_matplotlib()

    # Get control sample columns
    if sample_types:
        control_cols = [c for c in sample_cols if sample_types.get(c) == control_type]
    else:
        control_cols = sample_cols

    if len(control_cols) < 2:
        logger.warning(f"Insufficient {control_type} samples for CV calculation")
        return None

    def calc_cvs(df):
        """Calculate CV for each feature across control samples."""
        control_data = df[control_cols]
        # Calculate on linear scale (exponentiate from log2)
        linear_data = 2 ** control_data
        cv_values = (linear_data.std(axis=1) / linear_data.mean(axis=1)) * 100
        return cv_values.dropna().values

    cv_before = calc_cvs(data_before)
    cv_after = calc_cvs(data_after)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for ax, cv_values, label in [
        (ax1, cv_before, "Before Normalization"),
        (ax2, cv_after, "After Normalization"),
    ]:
        if len(cv_values) > 0:
            ax.hist(cv_values, bins=50, alpha=0.7, color="skyblue", edgecolor="black")

            median_cv = np.median(cv_values)
            pct_under = np.sum(cv_values < cv_threshold) / len(cv_values) * 100

            ax.axvline(median_cv, color="red", linestyle="--", linewidth=2,
                       label=f"Median: {median_cv:.1f}%")
            ax.axvline(cv_threshold, color="orange", linestyle=":", linewidth=2,
                       alpha=0.8, label=f"Threshold: {cv_threshold}%")

            ax.text(
                0.95, 0.95,
                f"Median: {median_cv:.1f}%\n{pct_under:.1f}% < {cv_threshold}%",
                transform=ax.transAxes, fontsize=10, va="top", ha="right",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
            )

            ax.set_xlim(0, min(100, np.percentile(cv_values, 99)))
        else:
            ax.text(0.5, 0.5, "Insufficient data",
                    transform=ax.transAxes, ha="center", va="center")

        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Coefficient of Variation (%)", fontsize=11)
        ax.set_ylabel("Number of Features", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(f"{title} ({control_type.capitalize()} Samples)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show_plot:
        plt.show()
        return None
    else:
        return fig


def plot_control_correlation_wide(
    data: pd.DataFrame,
    sample_cols: list[str],
    sample_types: dict[str, str] | None = None,
    control_types: list[str] | None = None,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    title: str = "Control Sample Correlation",
    figsize: tuple[int, int] = (10, 8),
    show_plot: bool = True,
    save_path: str | None = None,
) -> tuple[plt.Figure | None, pd.DataFrame]:
    """Create correlation heatmap for control samples (wide-format data).

    Args:
        data: Wide-format DataFrame (features x samples)
        sample_cols: List of column names containing sample abundances
        sample_types: Dict mapping sample name to type
        control_types: List of sample types to include (default: ['reference', 'qc'])
        method: Correlation method
        title: Plot title
        figsize: Figure size
        show_plot: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        Tuple of (Figure if show_plot is False, correlation matrix DataFrame)

    """
    _check_matplotlib()
    _check_seaborn()

    if control_types is None:
        control_types = ["reference", "qc"]

    # Get control sample columns
    if sample_types:
        control_cols = [c for c in sample_cols if sample_types.get(c) in control_types]
    else:
        control_cols = sample_cols

    if len(control_cols) < 2:
        logger.warning("Insufficient control samples for correlation")
        return None, pd.DataFrame()

    # Get control data
    control_data = data[control_cols].dropna(axis=0, thresh=len(control_cols) * 0.5)

    if control_data.shape[0] == 0:
        logger.warning("No valid data for correlation after filtering")
        return None, pd.DataFrame()

    # Calculate correlation
    corr_matrix = control_data.corr(method=method)

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        corr_matrix,
        annot=True if len(control_cols) <= 15 else False,
        fmt=".3f",
        cmap="RdYlBu_r",
        center=0.9,
        vmin=0.5,
        vmax=1.0,
        square=True,
        ax=ax,
        linewidths=0.5,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")

    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show_plot:
        plt.show()
        return None, corr_matrix
    else:
        return fig, corr_matrix

def plot_cv_three_stage(
    data_rawsum: pd.DataFrame,
    data_medianpolish: pd.DataFrame,
    data_normalized: pd.DataFrame,
    sample_cols: list[str],
    sample_types: dict[str, str] | None = None,
    control_type: str = "reference",
    title: str = "CV Distribution: Processing Stages",
    figsize: tuple[int, int] = (15, 5),
    show_plot: bool = True,
    save_path: str | None = None,
) -> plt.Figure | None:
    """Compare CV distributions across 3 processing stages (wide-format data).

    Shows how CV improves from raw sum -> median polish -> normalized.

    Args:
        data_rawsum: Wide-format DataFrame with sum-aggregated peptides
        data_medianpolish: Wide-format DataFrame after Tukey median polish
        data_normalized: Wide-format DataFrame after normalization + batch correction
        sample_cols: List of column names containing sample abundances
        sample_types: Dict mapping sample name to type
        control_type: Sample type to analyze ('reference' or 'qc')
        title: Plot title
        figsize: Figure size
        show_plot: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure if show_plot is False, else None

    """
    _check_matplotlib()

    # Get control sample columns
    if sample_types:
        control_cols = [c for c in sample_cols if sample_types.get(c) == control_type]
    else:
        control_cols = sample_cols

    if len(control_cols) < 2:
        logger.warning(f"Insufficient {control_type} samples for CV calculation")
        return None

    def calc_cvs(df):
        """Calculate CV for each feature across control samples."""
        control_data = df[control_cols]
        # Calculate on linear scale (exponentiate from log2)
        linear_data = 2 ** control_data
        cv_values = (linear_data.std(axis=1) / linear_data.mean(axis=1)) * 100
        return cv_values.dropna().values

    datasets = [
        (data_rawsum, "Raw Sum", "#ff6b6b"),
        (data_medianpolish, "Median Polish", "#4ecdc4"),
        (data_normalized, "Normalized", "#45b7d1"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, (df, label, color) in zip(axes, datasets):
        cv_values = calc_cvs(df)

        if len(cv_values) > 0:
            ax.hist(cv_values, bins=50, alpha=0.7, color=color, edgecolor="black")

            median_cv = np.median(cv_values)
            mean_cv = np.mean(cv_values)

            # Show median and mean lines (no labels - stats in text box)
            ax.axvline(median_cv, color="darkred", linestyle="--", linewidth=2)
            ax.axvline(mean_cv, color="darkblue", linestyle=":", linewidth=2)

            ax.text(
                0.95, 0.95,
                f"Median: {median_cv:.1f}%\nMean: {mean_cv:.1f}%\nN={len(cv_values):,}",
                transform=ax.transAxes, fontsize=10, va="top", ha="right",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
            )

            ax.set_xlim(0, min(100, np.percentile(cv_values, 99)))
        else:
            ax.text(0.5, 0.5, "Insufficient data",
                    transform=ax.transAxes, ha="center", va="center")

        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Coefficient of Variation (%)", fontsize=11)
        ax.set_ylabel("Number of Features", fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{title} ({control_type.capitalize()} Samples)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show_plot:
        plt.show()
        return None
    else:
        return fig


def plot_pca_three_stage(
    data_rawsum: pd.DataFrame,
    data_medianpolish: pd.DataFrame,
    data_normalized: pd.DataFrame,
    sample_cols: list[str],
    sample_types: dict[str, str] | None = None,
    title: str = "PCA: Processing Stages",
    figsize: tuple[int, int] = (18, 5),
    show_plot: bool = True,
    save_path: str | None = None,
) -> plt.Figure | None:
    """Compare PCA across 3 processing stages (wide-format data).

    Shows how sample clustering changes from raw sum -> median polish -> normalized.

    Args:
        data_rawsum: Wide-format DataFrame with sum-aggregated peptides
        data_medianpolish: Wide-format DataFrame after Tukey median polish
        data_normalized: Wide-format DataFrame after normalization + batch correction
        sample_cols: List of column names containing sample abundances
        sample_types: Dict mapping sample name to type (for coloring)
        title: Plot title
        figsize: Figure size
        show_plot: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure if show_plot is False, else None

    """
    _check_matplotlib()
    _check_sklearn()

    datasets = [
        (data_rawsum, "Raw Sum"),
        (data_medianpolish, "Median Polish"),
        (data_normalized, "Normalized"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    def perform_pca_and_plot(df, ax, plot_title):
        """Perform PCA and plot on given axis."""
        sample_data = df[sample_cols].copy()
        sample_data = sample_data.dropna(axis=0, thresh=len(sample_cols) * 0.5)
        sample_data = sample_data.fillna(sample_data.median())

        if sample_data.shape[0] < 2:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(plot_title)
            return

        pca_input = sample_data.T
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pca_input)

        n_comp = min(2, pca_input.shape[0] - 1, pca_input.shape[1])
        pca = PCA(n_components=n_comp)
        scores = pca.fit_transform(scaled_data)

        pca_df = pd.DataFrame(
            scores[:, :2],
            columns=["PC1", "PC2"],
            index=pca_input.index,
        )

        if sample_types:
            pca_df["Group"] = [sample_types.get(s, "unknown") for s in pca_df.index]
        else:
            pca_df["Group"] = "All"

        unique_groups = pca_df["Group"].unique()
        cmap = plt.colormaps.get_cmap("Set1")
        group_colors = {
            g: cmap(i / max(1, len(unique_groups) - 1))
            for i, g in enumerate(unique_groups)
        }

        for group in unique_groups:
            group_data = pca_df[pca_df["Group"] == group]
            ax.scatter(
                group_data["PC1"],
                group_data["PC2"],
                c=[group_colors[group]],
                label=group,
                alpha=0.7,
                s=80,
                edgecolors="black",
                linewidth=0.5,
            )

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=11)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=11)
        ax.set_title(plot_title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

    for ax, (df, label) in zip(axes, datasets):
        perform_pca_and_plot(df, ax, label)

    # Add a single legend outside the plots
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=9,
                   bbox_to_anchor=(0.99, 0.95))

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show_plot:
        plt.show()
        return None
    else:
        return fig


def plot_boxplot_two_stage(
    data_before: pd.DataFrame,
    data_after: pd.DataFrame,
    sample_cols: list[str],
    sample_types: dict[str, str] | None = None,
    before_label: str = "Before Normalization",
    after_label: str = "After Normalization",
    title: str = "Intensity Distribution: Before vs After",
    figsize: tuple[int, int] = (14, 6),
    show_plot: bool = True,
    save_path: str | None = None,
) -> plt.Figure | None:
    """Compare intensity distributions across 2 processing stages (wide-format data).

    Shows box plots of sample distributions before and after normalization/correction.

    Args:
        data_before: Wide-format DataFrame before processing
        data_after: Wide-format DataFrame after processing
        sample_cols: List of column names containing sample abundances
        sample_types: Dict mapping sample name to type (for coloring)
        before_label: Label for the before stage
        after_label: Label for the after stage
        title: Plot title
        figsize: Figure size
        show_plot: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure if show_plot is False, else None

    """
    _check_matplotlib()

    datasets = [
        (data_before, before_label),
        (data_after, after_label),
    ]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Get colors for samples
    colors, legend = _get_sample_colors(sample_cols, sample_types)

    for ax, (df, stage_label) in zip(axes, datasets):
        # Prepare data for boxplot
        box_data = []
        for sample in sample_cols:
            if sample in df.columns:
                sample_data = df[sample].dropna()
                box_data.append(sample_data.values)
            else:
                box_data.append(np.array([]))

        # Create box plot
        bp = ax.boxplot(
            box_data,
            positions=range(len(sample_cols)),
            widths=0.6,
            patch_artist=True,
            showfliers=False,
        )

        # Color boxes by sample type
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Calculate median of sample medians
        sample_medians = [np.median(d) for d in box_data if len(d) > 0]
        if sample_medians:
            median_of_medians = np.median(sample_medians)
            ax.axhline(median_of_medians, color="darkred", linestyle="--",
                       linewidth=1.5, alpha=0.7)

        ax.set_title(stage_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Sample", fontsize=11)
        ax.set_ylabel("Log2 Abundance", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

        # Reduce x-tick labels for readability
        if len(sample_cols) > 20:
            n = max(1, len(sample_cols) // 10)
            ax.set_xticks(range(0, len(sample_cols), n))
            ax.set_xticklabels(
                [sample_cols[i][:15] for i in range(0, len(sample_cols), n)],
                rotation=45, ha="right", fontsize=7,
            )
        else:
            ax.set_xticks(range(len(sample_cols)))
            ax.set_xticklabels(sample_cols, rotation=45, ha="right", fontsize=7)

    # Add legend outside plots
    if legend:
        patches = [
            mpatches.Patch(color=color, label=label, alpha=0.7)
            for label, color in legend.items()
        ]
        fig.legend(handles=patches, loc="upper right", fontsize=9,
                   bbox_to_anchor=(0.99, 0.95))

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show_plot:
        plt.show()
        return None
    else:
        return fig


def plot_boxplot_three_stage(
    data_rawsum: pd.DataFrame,
    data_medianpolish: pd.DataFrame,
    data_normalized: pd.DataFrame,
    sample_cols: list[str],
    sample_types: dict[str, str] | None = None,
    title: str = "Intensity Distribution: Processing Stages",
    figsize: tuple[int, int] = (18, 6),
    show_plot: bool = True,
    save_path: str | None = None,
) -> plt.Figure | None:
    """Compare intensity distributions across 3 processing stages (wide-format data).

    Shows box plots of sample distributions from raw sum -> median polish -> normalized.

    Args:
        data_rawsum: Wide-format DataFrame with sum-aggregated peptides
        data_medianpolish: Wide-format DataFrame after Tukey median polish
        data_normalized: Wide-format DataFrame after normalization + batch correction
        sample_cols: List of column names containing sample abundances
        sample_types: Dict mapping sample name to type (for coloring)
        title: Plot title
        figsize: Figure size
        show_plot: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure if show_plot is False, else None

    """
    _check_matplotlib()

    datasets = [
        (data_rawsum, "Raw Sum"),
        (data_medianpolish, "Median Polish"),
        (data_normalized, "Normalized"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Get colors for samples
    colors, legend = _get_sample_colors(sample_cols, sample_types)

    for ax, (df, stage_label) in zip(axes, datasets):
        # Prepare data for boxplot
        box_data = []
        for sample in sample_cols:
            if sample in df.columns:
                sample_data = df[sample].dropna()
                box_data.append(sample_data.values)
            else:
                box_data.append(np.array([]))

        # Create box plot
        bp = ax.boxplot(
            box_data,
            positions=range(len(sample_cols)),
            widths=0.6,
            patch_artist=True,
            showfliers=False,
        )

        # Color boxes by sample type
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Calculate median of sample medians
        sample_medians = [np.median(d) for d in box_data if len(d) > 0]
        if sample_medians:
            median_of_medians = np.median(sample_medians)
            ax.axhline(median_of_medians, color="darkred", linestyle="--",
                       linewidth=1.5, alpha=0.7)

        ax.set_title(stage_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Sample", fontsize=11)
        ax.set_ylabel("Log2 Abundance", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

        # Reduce x-tick labels for readability
        if len(sample_cols) > 20:
            n = max(1, len(sample_cols) // 10)
            ax.set_xticks(range(0, len(sample_cols), n))
            ax.set_xticklabels(
                [sample_cols[i][:15] for i in range(0, len(sample_cols), n)],
                rotation=45, ha="right", fontsize=7,
            )
        else:
            ax.set_xticks(range(len(sample_cols)))
            ax.set_xticklabels(sample_cols, rotation=45, ha="right", fontsize=7)

    # Add legend outside plots
    if legend:
        patches = [
            mpatches.Patch(color=color, label=label, alpha=0.7)
            for label, color in legend.items()
        ]
        fig.legend(handles=patches, loc="upper right", fontsize=9,
                   bbox_to_anchor=(0.99, 0.95))

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    if show_plot:
        plt.show()
        return None
    else:
        return fig
