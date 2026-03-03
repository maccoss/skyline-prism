"""Rollup comparison report generation module.

Generates a self-contained HTML report comparing library_assist vs sum
rollup methods, with detailed visualization of the library fitting process.
"""

from __future__ import annotations

import base64
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .rollup_comparison import (
        LibraryFitStep,
        PeptideLibraryComparison,
        RollupComparisonResult,
    )

logger = logging.getLogger(__name__)

# Check for matplotlib
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


# =============================================================================
# Plot Generation
# =============================================================================


def _get_transition_colors(n_transitions: int) -> list[str]:
    """Get a consistent color palette for transitions."""
    if not HAS_MATPLOTLIB:
        return ["#1f77b4"] * n_transitions

    cmap = plt.cm.get_cmap("tab10")
    return [matplotlib.colors.rgb2hex(cmap(i % 10)) for i in range(n_transitions)]


def plot_transition_bar(
    intensities: pd.Series,
    mz_values: pd.Series,
    title: str = "",
    colors: list[str] | None = None,
    highlight_outliers: list[str] | None = None,
    predicted: pd.Series | None = None,
    figsize: tuple[float, float] = (3.0, 2.5),
) -> "plt.Figure":
    """Create a bar plot of transition intensities vs m/z.

    Args:
        intensities: Series of transition -> intensity (linear scale)
        mz_values: Series of transition -> m/z
        title: Plot title
        colors: List of colors for each transition
        highlight_outliers: List of transition names to highlight as outliers
        predicted: Optional series of predicted intensities to overlay
        figsize: Figure size in inches

    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")

    fig, ax = plt.subplots(figsize=figsize)

    # Sort by m/z
    common_idx = intensities.index.intersection(mz_values.index)
    if len(common_idx) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=9)
        return fig

    intensities = intensities.loc[common_idx]
    mz_values = mz_values.loc[common_idx]
    sorted_idx = mz_values.sort_values().index
    intensities = intensities.loc[sorted_idx]
    mz_values = mz_values.loc[sorted_idx]

    x = range(len(intensities))

    # Default colors
    if colors is None:
        colors = _get_transition_colors(len(intensities))
    else:
        colors = [colors[i % len(colors)] for i in range(len(intensities))]

    # Modify colors for outliers
    bar_colors = []
    edge_colors = []
    for i, t in enumerate(intensities.index):
        if highlight_outliers and str(t) in highlight_outliers:
            bar_colors.append("#ff6b6b")  # Red for outliers
            edge_colors.append("#c92a2a")
        else:
            bar_colors.append(colors[i])
            edge_colors.append("black")

    # Plot bars
    bars = ax.bar(
        x,
        intensities.values,
        color=bar_colors,
        edgecolor=edge_colors,
        linewidth=0.5,
        alpha=0.8,
    )

    # Overlay predicted values if provided
    if predicted is not None:
        predicted = predicted.loc[sorted_idx]
        ax.scatter(
            x,
            predicted.values,
            marker="o",
            s=30,
            color="black",
            zorder=5,
            label="Library predicted",
        )
        # Connect with line
        ax.plot(x, predicted.values, "k--", alpha=0.5, linewidth=1)

    # Format x-axis with m/z values
    ax.set_xticks(x)
    ax.set_xticklabels([f"{mz:.0f}" for mz in mz_values.values], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("m/z", fontsize=8)
    ax.set_ylabel("Intensity", fontsize=8)
    ax.set_title(title, fontsize=9)

    # Format y-axis
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    ax.tick_params(axis="both", labelsize=7)

    plt.tight_layout()
    return fig


def plot_summary_stats(
    abundances: pd.Series,
    cv: float,
    method_name: str,
    sample_order: list[str] | None = None,
    r_squared: float | None = None,
    figsize: tuple[float, float] = (3.0, 2.5),
) -> "plt.Figure":
    """Create a summary plot with statistics.

    Args:
        abundances: Series of replicate -> abundance
        cv: Coefficient of variation
        method_name: Name of the method ("sum" or "library_assist")
        sample_order: List of sample names in display order (for consistent x-axis)
        r_squared: Optional R² value
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")

    fig, ax = plt.subplots(figsize=figsize)

    # Reorder abundances to match sample_order if provided
    if sample_order is not None:
        # Filter to samples that exist in abundances and maintain order
        ordered_samples = [s for s in sample_order if s in abundances.index]
        abundances = abundances.loc[ordered_samples]

    # Plot bar chart of abundances
    valid = abundances.dropna()
    if len(valid) > 0:
        x = range(len(valid))
        ax.bar(x, valid.values, color="#4c6ef5", edgecolor="black", linewidth=0.5, alpha=0.8)
        ax.set_xticks(x)
        # Use short sample names if available - smaller font to avoid crowding
        labels = [_get_short_sample_name(str(s)) for s in valid.index]
        ax.set_xticklabels(labels, fontsize=4, rotation=45, ha="right")

        # Add mean line
        mean_val = valid.mean()
        ax.axhline(y=mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean")

        # Add stats text - position in lower right to avoid overlap with bars
        stats_text = f"Mean: {mean_val:.2e}\nStd: {valid.std():.2e}\nCV: {cv*100:.1f}%"
        if r_squared is not None:
            stats_text += f"\nMean R²: {r_squared:.3f}"

        # Position stats box in lower right, below the data
        ax.text(
            0.98,
            0.02,
            stats_text,
            transform=ax.transAxes,
            fontsize=6,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
        )

    ax.set_title(method_name, fontsize=9)
    ax.set_ylabel("Abundance", fontsize=8)
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    ax.tick_params(axis="both", labelsize=6)

    plt.tight_layout()
    return fig


def _encode_figure_base64(fig: "plt.Figure", dpi: int = 100) -> str:
    """Convert matplotlib figure to base64-encoded PNG."""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{img_base64}" />'


def _save_and_embed_plot(
    fig: "plt.Figure",
    name: str,
    plots_dir: Path,
    save_plots: bool,
    embed_plots: bool = True,
    dpi: int = 100,
) -> tuple[str, Path | None]:
    """Save a matplotlib figure and/or generate HTML for embedding.

    Args:
        fig: matplotlib Figure object
        name: Base name for the plot file
        plots_dir: Directory to save plots
        save_plots: Whether to save PNG file
        embed_plots: Whether to embed as base64
        dpi: Resolution for saved images

    Returns:
        Tuple of (HTML string for embedding, path if saved)
    """
    plot_path = None
    html = ""

    if save_plots:
        plot_path = plots_dir / f"{name}.png"
        fig.savefig(plot_path, dpi=dpi, bbox_inches="tight", facecolor="white")

    if embed_plots:
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        html = f'<img src="data:image/png;base64,{img_base64}" alt="{name}" />'
    elif plot_path:
        html = f'<img src="rollup_comparison_plots/{name}.png" alt="{name}" />'

    plt.close(fig)
    return html, plot_path


# =============================================================================
# HTML Report Generation
# =============================================================================


def _get_report_css() -> str:
    """Return CSS styles for the report."""
    return """
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #1a1a2e;
            border-bottom: 3px solid #4c6ef5;
            padding-bottom: 10px;
        }
        h2 {
            color: #16213e;
            margin-top: 30px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }
        h3 {
            color: #0f3460;
            margin-top: 20px;
        }
        .config-table, .summary-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        .config-table td, .config-table th,
        .summary-table td, .summary-table th {
            padding: 10px 15px;
            border: 1px solid #ddd;
            text-align: left;
        }
        .config-table th, .summary-table th {
            background-color: #4c6ef5;
            color: white;
        }
        .summary-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .improved {
            color: #2e7d32;
            font-weight: bold;
        }
        .worsened {
            color: #c62828;
            font-weight: bold;
        }
        .peptide-section {
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #fafafa;
        }
        .peptide-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .peptide-name {
            font-family: monospace;
            font-size: 1.1em;
            background-color: #e9ecef;
            padding: 5px 10px;
            border-radius: 4px;
        }
        .cv-badge {
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }
        .cv-improved {
            background-color: #d4edda;
            color: #155724;
        }
        .cv-worsened {
            background-color: #f8d7da;
            color: #721c24;
        }
        .plot-grid {
            display: grid;
            gap: 10px;
            overflow-x: auto;
        }
        .plot-row {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .row-label {
            width: 120px;
            min-width: 120px;
            font-weight: bold;
            font-size: 0.85em;
            color: #495057;
        }
        .plot-cell {
            flex: 0 0 auto;
            min-width: 300px;
            min-height: 250px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            color: #666;
            font-size: 0.85em;
        }
        .plot-cell img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .timestamp {
            margin-top: 30px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }
        .legend {
            display: flex;
            gap: 20px;
            margin: 15px 0;
            flex-wrap: wrap;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 3px;
            border: 1px solid #333;
        }
    </style>
    """


def _get_short_sample_name(full_name: str) -> str:
    """Extract short sample name from full sample ID.

    Handles formats like:
    - "IRType-Plasma-Pool_007__@__2025-IRType-Plasma-PRISM-Plate1_subset" -> "Pool_007"
    - "Sample_A_001" -> "Sample_A_001"
    """
    # Split on __@__ if present (batch separator)
    if "__@__" in full_name:
        full_name = full_name.split("__@__")[0]

    # Try to extract meaningful part (last segment with numbers)
    parts = full_name.split("-")
    for part in reversed(parts):
        if any(c.isdigit() for c in part):
            return part

    # Fallback: return last 15 characters
    return full_name[-15:] if len(full_name) > 15 else full_name


def create_peptide_plot_grid(
    peptide_result: "PeptideLibraryComparison",
    samples: list[str],
    plots_dir: Path,
    save_plots: bool = True,
    embed_plots: bool = True,
) -> str:
    """Create the grid of plots for one peptide.

    Grid structure:
    - Row 0: Raw transitions for each replicate + summary
    - Row 1: Library scaled (initial fit)
    - Row 2: Outliers removed (if any)
    - Row 3: Final sum abundances
    - Row 4: Final library abundances

    Args:
        peptide_result: PeptideLibraryComparison with all data
        samples: List of sample names
        plots_dir: Directory to save plots
        save_plots: Whether to save PNG files
        embed_plots: Whether to embed as base64

    Returns:
        HTML string for the peptide plot grid
    """
    if not HAS_MATPLOTLIB:
        return "<p>matplotlib not available for plotting</p>"

    peptide_safe = peptide_result.peptide.replace("/", "_").replace("\\", "_")[:50]

    # Get transitions that are in the library (for consistent display)
    library_transitions = set(peptide_result.library_spectrum.index)

    # Filter mz_values to only library-matched transitions
    library_mz = peptide_result.mz_values[
        peptide_result.mz_values.index.isin(library_transitions)
    ]

    # Create color mapping by transition name for consistency
    sorted_transitions = library_mz.sort_values().index.tolist()
    all_colors = _get_transition_colors(len(sorted_transitions))
    color_map = {t: all_colors[i] for i, t in enumerate(sorted_transitions)}

    html_rows = []

    # Row 0: Raw transitions (filtered to library-matched only)
    row_html = ['<div class="plot-row">', '<div class="row-label">Raw Transitions</div>']
    for sample in samples:
        if sample in peptide_result.raw_transitions:
            obs = peptide_result.raw_transitions[sample]
            # Filter to library-matched transitions
            obs_filtered = obs[obs.index.isin(library_transitions)]
            if len(obs_filtered) > 0:
                # Get colors in the right order for this sample's transitions
                sample_colors = [color_map.get(t, "#888888") for t in obs_filtered.index]
                fig = plot_transition_bar(
                    intensities=obs_filtered,
                    mz_values=library_mz,
                    title=_get_short_sample_name(sample),
                    colors=sample_colors,
                )
                img_html, _ = _save_and_embed_plot(
                    fig, f"{peptide_safe}_raw_{sample}", plots_dir, save_plots, embed_plots
                )
                row_html.append(f'<div class="plot-cell">{img_html}</div>')
            else:
                row_html.append('<div class="plot-cell">N/A</div>')
        else:
            row_html.append('<div class="plot-cell">N/A</div>')

    # Summary for raw (sum)
    fig = plot_summary_stats(
        peptide_result.sum_abundances_corrected,
        peptide_result.sum_cv,
        "Sum",
        sample_order=samples,
    )
    img_html, _ = _save_and_embed_plot(
        fig, f"{peptide_safe}_raw_summary", plots_dir, save_plots, embed_plots
    )
    row_html.append(f'<div class="plot-cell">{img_html}</div>')
    row_html.append("</div>")
    html_rows.append("\n".join(row_html))

    # Row 1: Library scaled (initial fit)
    row_html = ['<div class="plot-row">', '<div class="row-label">Library Scaled</div>']
    for sample in samples:
        if sample in peptide_result.fitting_steps:
            steps = peptide_result.fitting_steps[sample]
            # Find the library_scaled step
            lib_step = next((s for s in steps if s.step_name == "library_scaled"), None)
            if lib_step:
                short_name = _get_short_sample_name(sample)
                title = f"{short_name} (R²={lib_step.r_squared:.2f})" if lib_step.r_squared else short_name
                # Get colors for this step's transitions
                step_colors = [color_map.get(t, "#888888") for t in lib_step.transition_intensities.index]
                fig = plot_transition_bar(
                    intensities=lib_step.transition_intensities,
                    mz_values=library_mz,
                    title=title,
                    colors=step_colors,
                    predicted=lib_step.library_predicted,
                )
                img_html, _ = _save_and_embed_plot(
                    fig, f"{peptide_safe}_libscaled_{sample}", plots_dir, save_plots, embed_plots
                )
                row_html.append(f'<div class="plot-cell">{img_html}</div>')
            else:
                row_html.append('<div class="plot-cell">N/A</div>')
        else:
            row_html.append('<div class="plot-cell">N/A</div>')

    # Summary placeholder for library scaled
    row_html.append('<div class="plot-cell"></div>')
    row_html.append("</div>")
    html_rows.append("\n".join(row_html))

    # Row 2: Outliers removed (if any outliers were found or reverted)
    has_outlier_step = any(
        any(s.step_name in ("outliers_removed", "outliers_reverted") for s in steps)
        for steps in peptide_result.fitting_steps.values()
    )

    if has_outlier_step:
        row_html = ['<div class="plot-row">', '<div class="row-label">Outlier Handling</div>']
        for sample in samples:
            if sample in peptide_result.fitting_steps:
                steps = peptide_result.fitting_steps[sample]
                # Find either outliers_removed or outliers_reverted step
                outlier_step = next(
                    (s for s in steps if s.step_name in ("outliers_removed", "outliers_reverted")),
                    None
                )
                if outlier_step:
                    short_name = _get_short_sample_name(sample)
                    # Add indicator if reverted
                    if outlier_step.step_name == "outliers_reverted":
                        title = f"{short_name} (Reverted)"
                    else:
                        title = f"{short_name} (R²={outlier_step.r_squared:.2f})" if outlier_step.r_squared else short_name
                    step_colors = [color_map.get(t, "#888888") for t in outlier_step.transition_intensities.index]
                    fig = plot_transition_bar(
                        intensities=outlier_step.transition_intensities,
                        mz_values=library_mz,
                        title=title,
                        colors=step_colors,
                        highlight_outliers=outlier_step.excluded_transitions,
                        predicted=outlier_step.library_predicted,
                    )
                    img_html, _ = _save_and_embed_plot(
                        fig, f"{peptide_safe}_outliers_{sample}", plots_dir, save_plots, embed_plots
                    )
                    row_html.append(f'<div class="plot-cell">{img_html}</div>')
                else:
                    row_html.append('<div class="plot-cell">No outliers</div>')
            else:
                row_html.append('<div class="plot-cell">N/A</div>')

        row_html.append('<div class="plot-cell"></div>')
        row_html.append("</div>")
        html_rows.append("\n".join(row_html))

    # Row 3: Final library-assist
    row_html = ['<div class="plot-row">', '<div class="row-label">Library-Assist</div>']
    for sample in samples:
        # Show empty cells for replicates (abundance shown in summary)
        row_html.append('<div class="plot-cell"></div>')

    # Get average R² across samples
    r_squared_values = []
    for steps in peptide_result.fitting_steps.values():
        final_step = steps[-1] if steps else None
        if final_step and final_step.r_squared is not None:
            r_squared_values.append(final_step.r_squared)
    avg_r2 = np.mean(r_squared_values) if r_squared_values else None

    fig = plot_summary_stats(
        peptide_result.library_abundances_corrected,
        peptide_result.library_cv,
        "Library-Assist",
        sample_order=samples,
        r_squared=avg_r2,
    )
    img_html, _ = _save_and_embed_plot(
        fig, f"{peptide_safe}_library_summary", plots_dir, save_plots, embed_plots
    )
    row_html.append(f'<div class="plot-cell">{img_html}</div>')
    row_html.append("</div>")
    html_rows.append("\n".join(row_html))

    return '<div class="plot-grid">\n' + "\n".join(html_rows) + "\n</div>"


def generate_rollup_comparison_report(
    result: "RollupComparisonResult",
    output_path: str | Path,
    save_plots: bool = True,
    embed_plots: bool = True,
) -> dict[str, Path]:
    """Generate the self-contained HTML comparison report.

    Args:
        result: RollupComparisonResult with all comparison data
        output_path: Path to save the HTML report
        save_plots: Whether to save individual PNG files
        embed_plots: Whether to embed plots in HTML (base64)

    Returns:
        Dict mapping plot names to file paths (if save_plots=True)
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, cannot generate rollup comparison report")
        return {}

    output_path = Path(output_path)
    plots_dir = output_path.parent / "rollup_comparison_plots"
    plot_paths: dict[str, Path] = {}

    if save_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    # Build HTML
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        '<meta charset="UTF-8">',
        "<title>PRISM Library-Assist Rollup QC Report</title>",
        _get_report_css(),
        "</head>",
        "<body>",
        '<div class="container">',
        "<h1>Library-Assist Rollup QC Report</h1>",
    ]

    # Configuration section
    html_parts.append("<h2>Configuration</h2>")
    html_parts.append('<table class="config-table">')
    html_parts.append("<tr><th>Setting</th><th>Value</th></tr>")
    html_parts.append(f"<tr><td>Comparison</td><td>library_assist vs sum</td></tr>")
    html_parts.append(f"<tr><td>Sample Type</td><td>{result.sample_type} ({len(result.samples)} replicates)</td></tr>")
    html_parts.append(f"<tr><td>Spectral Library</td><td>{result.library_path}</td></tr>")
    html_parts.append(f"<tr><td>Ranking Criterion</td><td>{result.ranking_criterion}</td></tr>")
    html_parts.append(f"<tr><td>Normalization Applied</td><td>{'Yes' if result.normalization_applied else 'No'}</td></tr>")
    html_parts.append(f"<tr><td>Batch Correction Applied</td><td>{'Yes' if result.batch_correction_applied else 'No'}</td></tr>")
    html_parts.append("</table>")

    # Summary section
    html_parts.append("<h2>Overall Performance Summary</h2>")
    html_parts.append('<table class="summary-table">')
    html_parts.append("<tr><th>Method</th><th>Median CV</th><th># Peptides</th><th>Improved</th><th>Worsened</th></tr>")

    sum_cv_pct = f"{result.summary.sum_median_cv * 100:.1f}%" if not np.isnan(result.summary.sum_median_cv) else "N/A"
    lib_cv_pct = f"{result.summary.library_median_cv * 100:.1f}%" if not np.isnan(result.summary.library_median_cv) else "N/A"

    html_parts.append(f"<tr><td>sum</td><td>{sum_cv_pct}</td><td>{result.summary.n_peptides_total}</td><td>-</td><td>-</td></tr>")
    html_parts.append(
        f"<tr><td>library_assist</td><td>{lib_cv_pct}</td><td>{result.summary.n_peptides_total}</td>"
        f'<td class="improved">{result.summary.n_peptides_improved}</td>'
        f'<td class="worsened">{result.summary.n_peptides_worsened}</td></tr>'
    )
    html_parts.append("</table>")

    # Median improvement
    if not np.isnan(result.summary.median_improvement):
        improvement_pct = result.summary.median_improvement * 100
        improvement_class = "improved" if improvement_pct > 0 else "worsened"
        html_parts.append(
            f'<p>Median CV improvement: <span class="{improvement_class}">{improvement_pct:+.1f}%</span></p>'
        )

    # Legend
    html_parts.append('<div class="legend">')
    html_parts.append('<div class="legend-item"><div class="legend-color" style="background-color: #4c6ef5;"></div><span>Observed intensity</span></div>')
    html_parts.append('<div class="legend-item"><div class="legend-color" style="background-color: #ff6b6b;"></div><span>Outlier transition</span></div>')
    html_parts.append('<div class="legend-item">● Library predicted</div>')
    html_parts.append("</div>")

    # Top peptides section
    html_parts.append(f"<h2>Top {len(result.top_peptides)} Peptides by {result.ranking_criterion.replace('_', ' ').title()}</h2>")

    for i, peptide in enumerate(result.top_peptides):
        pep_result = result.peptide_results[peptide]

        # Determine improvement status
        if pep_result.cv_improvement > 0.01:
            badge_class = "cv-improved"
            badge_text = f"Improved {pep_result.cv_improvement * 100:.1f}%"
        elif pep_result.cv_improvement < -0.01:
            badge_class = "cv-worsened"
            badge_text = f"Worsened {-pep_result.cv_improvement * 100:.1f}%"
        else:
            badge_class = ""
            badge_text = "Unchanged"

        html_parts.append('<div class="peptide-section">')
        html_parts.append('<div class="peptide-header">')
        html_parts.append(f'<h3>#{i+1}: <span class="peptide-name">{peptide}</span></h3>')
        html_parts.append(f'<span class="cv-badge {badge_class}">{badge_text}</span>')
        html_parts.append("</div>")

        # CV details
        sum_cv_pct = f"{pep_result.sum_cv * 100:.1f}%" if not np.isnan(pep_result.sum_cv) else "N/A"
        lib_cv_pct = f"{pep_result.library_cv * 100:.1f}%" if not np.isnan(pep_result.library_cv) else "N/A"
        html_parts.append(f"<p>CV: sum = {sum_cv_pct}, library_assist = {lib_cv_pct}</p>")

        # Plot grid
        grid_html = create_peptide_plot_grid(
            pep_result, result.samples, plots_dir, save_plots, embed_plots
        )
        html_parts.append(grid_html)

        html_parts.append("</div>")

    # Timestamp
    html_parts.append(f'<div class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>')

    # Close HTML
    html_parts.extend(["</div>", "</body>", "</html>"])

    # Write report
    html_content = "\n".join(html_parts)
    output_path.write_text(html_content, encoding="utf-8")
    logger.info(f"Rollup comparison report saved to: {output_path}")

    return plot_paths
