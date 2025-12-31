#!/usr/bin/env python3
"""Test script for RT-Lowess normalization.

This script evaluates RT-Lowess normalization:
- Fit lowess to abundance vs RT per sample
- Normalize so all samples match the global median lowess curve

Usage:
    python scripts/test_rt_normalization.py example-files/output-test/

This will generate diagnostic plots and comparison metrics.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from statsmodels.nonparametric.smoothers_lowess import lowess

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_peptide_data(output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load peptides_rollup.parquet and sample_metadata.tsv."""
    output_dir = Path(output_dir)

    peptide_path = output_dir / "peptides_rollup.parquet"
    meta_path = output_dir / "sample_metadata.tsv"

    if not peptide_path.exists():
        raise FileNotFoundError(f"Peptide file not found: {peptide_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pq.read_table(peptide_path).to_pandas()
    meta = pd.read_csv(meta_path, sep="\t")

    return df, meta


def get_sample_columns(df: pd.DataFrame) -> list[str]:
    """Get sample column names (exclude metadata columns)."""
    exclude = {"Peptide Modified Sequence Unimod Ids", "n_transitions", "mean_rt"}
    return [c for c in df.columns if c not in exclude]


def extract_batch_from_sample(sample_name: str) -> str:
    """Extract batch (plate) from sample name."""
    if "__@__" in sample_name:
        return sample_name.split("__@__")[1]
    return "unknown"


def get_sample_type(sample_name: str) -> str:
    """Determine sample type from sample name."""
    name_lower = sample_name.lower()
    if "pool" in name_lower:
        return "reference"
    elif "carl" in name_lower:
        return "qc"
    else:
        return "experimental"


def wide_to_long(df: pd.DataFrame, sample_cols: list[str]) -> pd.DataFrame:
    """Convert wide peptide matrix to long format."""
    long_df = df.melt(
        id_vars=["Peptide Modified Sequence Unimod Ids", "mean_rt"],
        value_vars=sample_cols,
        var_name="sample",
        value_name="abundance",
    )
    long_df["batch"] = long_df["sample"].apply(extract_batch_from_sample)
    long_df["sample_type"] = long_df["sample"].apply(get_sample_type)
    long_df["log2_abundance"] = np.log2(long_df["abundance"].replace(0, np.nan))
    return long_df


# =============================================================================
# RT-Lowess Normalization
# =============================================================================


def fit_all_sample_lowess(
    long_df: pd.DataFrame,
    rt_grid: np.ndarray,
    frac: float = 0.3,
    delta: float = 0.0,
) -> dict[str, np.ndarray]:
    """Fit lowess for all samples and return curves on RT grid.

    This fits lowess ONCE per sample and caches the result.
    Uses delta parameter for speed (skip points close together).
    """
    sample_curves = {}

    # Use delta = 1% of RT range for speedup
    rt_range = rt_grid[-1] - rt_grid[0]
    if delta == 0.0:
        delta = rt_range * 0.01  # 1% of range

    samples = long_df["sample"].unique()

    for sample in samples:
        sample_data = long_df[long_df["sample"] == sample].dropna(
            subset=["log2_abundance", "mean_rt"]
        )

        if len(sample_data) < 20:
            sample_curves[sample] = None
            continue

        # Fit lowess with delta for speed
        sorted_data = sample_data.sort_values("mean_rt")
        rt_vals = sorted_data["mean_rt"].values
        abund_vals = sorted_data["log2_abundance"].values

        smoothed = lowess(abund_vals, rt_vals, frac=frac, delta=delta, return_sorted=True)

        # Interpolate to grid
        curve_at_grid = np.interp(
            rt_grid, smoothed[:, 0], smoothed[:, 1], left=np.nan, right=np.nan
        )
        sample_curves[sample] = curve_at_grid

    return sample_curves


def apply_lowess_normalization_fast(
    long_df: pd.DataFrame,
    frac: float = 0.3,
    n_grid_points: int = 100,
) -> pd.DataFrame:
    """Apply RT-lowess normalization (highly optimized).

    Key optimizations:
    1. Fit lowess ONCE per sample (not twice)
    2. Use delta parameter in lowess for ~10x speedup
    3. Coarser grid (100 points is plenty for interpolation)
    4. Vectorized corrections per sample
    """
    result_df = long_df.copy()

    # Create RT grid
    rt_min = long_df["mean_rt"].min()
    rt_max = long_df["mean_rt"].max()
    rt_grid = np.linspace(rt_min, rt_max, n_grid_points)

    # Fit lowess for ALL samples ONCE
    print("  Fitting lowess for all samples...")
    t0 = time.time()
    sample_curves = fit_all_sample_lowess(long_df, rt_grid, frac=frac)
    print(f"    Done in {time.time() - t0:.1f}s")

    # Compute global median curve from cached fits
    print("  Computing global median curve...")
    valid_curves = [c for c in sample_curves.values() if c is not None]
    curves_array = np.array(valid_curves)
    global_curve = np.nanmedian(curves_array, axis=0)

    # Apply corrections per sample
    print("  Applying corrections...")
    t0 = time.time()
    corrections = np.zeros(len(long_df))
    samples = long_df["sample"].unique()

    for i, sample in enumerate(samples):
        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{len(samples)} samples...")

        sample_curve = sample_curves.get(sample)
        if sample_curve is None:
            continue

        mask = (result_df["sample"] == sample).values
        sample_rts = long_df.loc[mask, "mean_rt"].values

        # Vectorized interpolation
        sample_vals = np.interp(sample_rts, rt_grid, sample_curve)
        global_vals = np.interp(sample_rts, rt_grid, global_curve)

        corrections[mask] = sample_vals - global_vals

    print(f"    Done in {time.time() - t0:.1f}s")

    # Apply corrections
    result_df["log2_abundance_rt_norm"] = long_df["log2_abundance"] - corrections

    return result_df, sample_curves, global_curve, rt_grid


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_lowess_overlay_comparison_fast(
    sample_curves: dict[str, np.ndarray | None],
    global_curve: np.ndarray,
    rt_grid: np.ndarray,
    meta_df: pd.DataFrame,
    figsize: tuple[int, int] = (16, 7),
    save_path: str | None = None,
) -> None:
    """Two-panel plot using pre-computed lowess curves (fast)."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Create sample -> batch mapping
    sample_to_batch = meta_df.set_index("sample_id")["batch"].to_dict()
    batches = sorted(meta_df["batch"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(batches)))
    batch_to_color = dict(zip(batches, colors))

    # Left panel: Before (original sample curves)
    ax = axes[0]
    for sample, curve in sample_curves.items():
        if curve is None:
            continue
        batch = sample_to_batch.get(sample, "unknown")
        color = batch_to_color.get(batch, "gray")
        ax.plot(rt_grid, curve, color=color, alpha=0.3, linewidth=1)

    for batch, color in batch_to_color.items():
        short_batch = batch.replace("2025-IRType-Plasma-PRISM-", "").replace("_subset", "")
        ax.plot([], [], color=color, linewidth=2, label=short_batch)
    ax.legend(title="Batch", loc="upper right")
    ax.set_xlabel("Retention Time (min)")
    ax.set_ylabel("log2(abundance)")
    ax.set_title("Before RT Normalization")

    # Right panel: After (all curves aligned to global median)
    ax = axes[1]
    # After normalization, all curves should collapse to the global curve
    for sample, curve in sample_curves.items():
        if curve is None:
            continue
        batch = sample_to_batch.get(sample, "unknown")
        color = batch_to_color.get(batch, "gray")
        # After normalization: curve - (curve - global) = global
        # So we show what the corrected curves would look like
        corrected_curve = curve - (curve - global_curve)  # = global_curve
        ax.plot(rt_grid, corrected_curve, color=color, alpha=0.3, linewidth=1)

    # Plot the global curve prominently
    ax.plot(
        rt_grid, global_curve, color="black", linewidth=3, label="Global Median", linestyle="--"
    )

    for batch, color in batch_to_color.items():
        short_batch = batch.replace("2025-IRType-Plasma-PRISM-", "").replace("_subset", "")
        ax.plot([], [], color=color, linewidth=2, label=short_batch)
    ax.legend(title="Batch", loc="upper right")
    ax.set_xlabel("Retention Time (min)")
    ax.set_ylabel("log2(abundance)")
    ax.set_title("After RT Normalization (all aligned to global median)")

    plt.suptitle("Lowess Fits Overlay - All Samples", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_rt_bin_boxplots_comparison(
    long_df_before: pd.DataFrame,
    long_df_after: pd.DataFrame,
    value_col_after: str = "log2_abundance_rt_norm",
    n_bins: int = 8,
    figsize: tuple[int, int] = (16, 8),
    save_path: str | None = None,
) -> None:
    """Two-panel plot with actual boxplots per RT bin, colored by batch."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    rt_min = long_df_before["mean_rt"].min()
    rt_max = long_df_before["mean_rt"].max()
    bin_edges = np.linspace(rt_min, rt_max, n_bins + 1)

    batches = sorted(long_df_before["batch"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(batches)))
    batch_to_color = dict(zip(batches, colors))

    for ax, (df, col, title) in zip(
        axes,
        [
            (long_df_before, "log2_abundance", "Before RT Normalization"),
            (long_df_after, value_col_after, "After RT Normalization"),
        ],
    ):
        # Prepare data for boxplots
        positions = []
        data_to_plot = []
        box_colors = []
        labels = []

        width = 0.8 / len(batches)

        for i in range(n_bins):
            rt_start, rt_end = bin_edges[i], bin_edges[i + 1]

            for j, batch in enumerate(batches):
                # Filter data for this bin and batch
                mask = (
                    (df["mean_rt"] >= rt_start) & (df["mean_rt"] < rt_end) & (df["batch"] == batch)
                )
                vals = df.loc[mask, col].dropna().values

                if len(vals) > 5:
                    data_to_plot.append(vals)
                    positions.append(i + (j - len(batches) / 2 + 0.5) * width)
                    box_colors.append(batch_to_color[batch])

            if i == 0:
                labels = [f"{rt_start:.1f}" for _ in batches]

        # Create boxplots
        bp = ax.boxplot(
            data_to_plot,
            positions=positions,
            widths=width * 0.8,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
        )

        # Color the boxes
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # X-axis labels
        bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(n_bins)]
        ax.set_xticks(range(n_bins))
        ax.set_xticklabels([f"{c:.1f}" for c in bin_centers], fontsize=9)
        ax.set_xlabel("RT Bin Center (min)")
        ax.set_ylabel("log2(abundance)")
        ax.set_title(title)

        # Legend
        for batch, color in batch_to_color.items():
            short_batch = batch.replace("2025-IRType-Plasma-PRISM-", "").replace("_subset", "")
            ax.plot([], [], "s", color=color, markersize=10, label=short_batch)
        ax.legend(title="Batch", loc="upper right")

    plt.suptitle("Abundance Distribution per RT Bin by Batch", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_pca_comparison(
    long_df_before: pd.DataFrame,
    long_df_after: pd.DataFrame,
    value_col_after: str = "log2_abundance_rt_norm",
    figsize: tuple[int, int] = (14, 6),
    save_path: str | None = None,
) -> None:
    """Plot PCA before and after RT normalization, highlighting reference/QC."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Sample type styles
    type_styles = {
        "reference": {
            "color": "red",
            "marker": "^",
            "size": 100,
            "label": "Reference (Pool)",
        },
        "qc": {"color": "blue", "marker": "s", "size": 80, "label": "QC (Carl)"},
        "experimental": {
            "color": "gray",
            "marker": "o",
            "size": 30,
            "label": "Experimental",
        },
    }

    for ax, (df, col, title) in zip(
        axes,
        [
            (long_df_before, "log2_abundance", "Before RT Normalization"),
            (long_df_after, value_col_after, "After RT Normalization"),
        ],
    ):
        # Pivot to wide format
        wide = df.pivot(
            index="Peptide Modified Sequence Unimod Ids",
            columns="sample",
            values=col,
        )

        # Drop rows with too many NaNs
        wide = wide.dropna(thresh=wide.shape[1] * 0.5)

        # Impute remaining NaNs with row median
        wide = wide.T.fillna(wide.median(axis=1)).T

        # Z-score normalize rows
        wide_values = wide.values.astype(float)
        row_means = np.nanmean(wide_values, axis=1, keepdims=True)
        row_stds = np.nanstd(wide_values, axis=1, keepdims=True)
        row_stds[row_stds == 0] = 1
        wide_z = (wide_values - row_means) / row_stds
        wide_z = np.nan_to_num(wide_z, nan=0.0)

        # PCA
        pca = PCA(n_components=2)
        coords = pca.fit_transform(wide_z.T)

        # Get sample types
        sample_types = [get_sample_type(s) for s in wide.columns]

        # Plot experimental first, then QC, then reference (foreground)
        for stype in ["experimental", "qc", "reference"]:
            style = type_styles[stype]
            mask = [st == stype for st in sample_types]
            if any(mask):
                pts = coords[mask]
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    c=style["color"],
                    marker=style["marker"],
                    s=style["size"],
                    alpha=0.7,
                    label=style["label"],
                    edgecolors="black" if stype != "experimental" else "none",
                    linewidths=0.5,
                )

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)

    plt.suptitle("PCA - Sample Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def compute_cv_by_sample_type(
    long_df: pd.DataFrame,
    value_col: str = "log2_abundance",
) -> dict[str, float]:
    """Compute median CV for each sample type (reference, QC)."""
    results = {}

    for sample_type in ["reference", "qc"]:
        type_data = long_df[long_df["sample_type"] == sample_type]

        if len(type_data) == 0:
            continue

        # Pivot to wide
        wide = type_data.pivot(
            index="Peptide Modified Sequence Unimod Ids",
            columns="sample",
            values=value_col,
        )

        if wide.shape[1] < 3:
            continue

        # CV on LINEAR scale
        linear = 2**wide
        cvs = (linear.std(axis=1) / linear.mean(axis=1)) * 100
        results[sample_type] = np.nanmedian(cvs)

    return results


# =============================================================================
# Main
# =============================================================================


def main(output_dir: str):
    """Run RT-lowess normalization test and generate diagnostic plots."""
    output_dir = Path(output_dir)
    plot_dir = output_dir / "rt_normalization_plots"
    plot_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("RT-Lowess Normalization Test")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    peptide_df, meta_df = load_peptide_data(output_dir)
    sample_cols = get_sample_columns(peptide_df)

    print(f"  Peptides: {len(peptide_df)}")
    print(f"  Samples: {len(sample_cols)}")
    print(f"  RT range: {peptide_df['mean_rt'].min():.2f} - {peptide_df['mean_rt'].max():.2f} min")
    print()

    # Convert to long format
    print("Converting to long format...")
    long_df = wide_to_long(peptide_df, sample_cols)

    # Count sample types
    type_counts = long_df.groupby("sample_type")["sample"].nunique()
    print(f"  Reference samples: {type_counts.get('reference', 0)}")
    print(f"  QC samples: {type_counts.get('qc', 0)}")
    print(f"  Experimental samples: {type_counts.get('experimental', 0)}")
    print()

    # ---------------------------------------------------------------------
    # Apply RT-Lowess Normalization
    # ---------------------------------------------------------------------
    print("=" * 70)
    print("APPLYING RT-LOWESS NORMALIZATION")
    print("=" * 70)
    print()

    t_start = time.time()
    long_df_norm, sample_curves, global_curve, rt_grid = apply_lowess_normalization_fast(
        long_df, frac=0.3
    )
    t_total = time.time() - t_start
    print(f"\nTotal normalization time: {t_total:.1f}s")
    print()

    # ---------------------------------------------------------------------
    # Generate Plots
    # ---------------------------------------------------------------------
    print("=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    print()

    # Plot 1: Lowess overlay comparison (before/after) - using cached curves
    print("Plot 1: Lowess fits overlay (before/after)...")
    plot_lowess_overlay_comparison_fast(
        sample_curves,
        global_curve,
        rt_grid,
        meta_df,
        save_path=plot_dir / "01_lowess_overlay_comparison.png",
    )

    # Plot 2: RT bin boxplots (before/after)
    print("Plot 2: RT bin boxplots (before/after)...")
    plot_rt_bin_boxplots_comparison(
        long_df,
        long_df_norm,
        n_bins=8,
        save_path=plot_dir / "02_rt_bin_boxplots_comparison.png",
    )

    # Plot 3: PCA comparison
    print("Plot 3: PCA comparison...")
    plot_pca_comparison(
        long_df,
        long_df_norm,
        save_path=plot_dir / "03_pca_comparison.png",
    )

    # ---------------------------------------------------------------------
    # Summary Statistics
    # ---------------------------------------------------------------------
    print()
    print("=" * 70)
    print("CV SUMMARY")
    print("=" * 70)
    print()

    cv_before = compute_cv_by_sample_type(long_df, "log2_abundance")
    cv_after = compute_cv_by_sample_type(long_df_norm, "log2_abundance_rt_norm")

    print("Reference samples (Pool):")
    if "reference" in cv_before:
        print(f"  Before: {cv_before['reference']:.1f}%")
        print(f"  After:  {cv_after.get('reference', float('nan')):.1f}%")
        improvement = cv_before["reference"] - cv_after.get("reference", 0)
        print(f"  Change: {improvement:+.1f}%")
    print()

    print("QC samples (Carl):")
    if "qc" in cv_before:
        print(f"  Before: {cv_before['qc']:.1f}%")
        print(f"  After:  {cv_after.get('qc', float('nan')):.1f}%")
        improvement = cv_before["qc"] - cv_after.get("qc", 0)
        print(f"  Change: {improvement:+.1f}%")
    print()

    print(f"All plots saved to: {plot_dir}")
    print()
    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_rt_normalization.py <output_dir>")
        print("Example: python scripts/test_rt_normalization.py example-files/output-test/")
        sys.exit(1)

    main(sys.argv[1])
