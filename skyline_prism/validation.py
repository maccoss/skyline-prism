"""Validation module for assessing normalization quality.

Uses the dual-control design:
- Inter-experiment reference: calibration anchor
- Intra-experiment QC: validation control
"""

import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# Check for optional visualization dependencies
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


@dataclass
class ValidationMetrics:
    """Metrics for assessing normalization quality."""

    # CV metrics
    reference_cv_before: float
    reference_cv_after: float
    qc_cv_before: float
    qc_cv_after: float

    # CV improvement
    reference_cv_improvement: float  # (before - after) / before
    qc_cv_improvement: float
    relative_variance_reduction: float  # qc improvement / reference improvement

    # PCA metrics
    pca_qc_reference_distance_before: float
    pca_qc_reference_distance_after: float
    pca_distance_ratio: float  # after / before (should be ~1, not << 1)

    # Warnings
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Check if validation passed basic criteria."""
        return (
            self.qc_cv_improvement > 0 and  # QC CV improved
            self.pca_distance_ratio > 0.5 and  # Didn't collapse QC into reference
            self.relative_variance_reduction < 2.0  # Didn't overfit to reference
        )


def calculate_cv(
    data: pd.DataFrame,
    sample_mask: pd.Series,
    abundance_col: str = 'abundance',
    precursor_col: str = 'precursor_id',
    replicate_col: str = 'replicate_name',
) -> float:
    """Calculate median coefficient of variation across peptides.

    Args:
        data: DataFrame with peptide data
        sample_mask: Boolean mask for samples to include
        abundance_col: Column with abundance values
        precursor_col: Column with precursor identifiers
        replicate_col: Column with replicate names

    Returns:
        Median CV across peptides

    """
    subset = data.loc[sample_mask]

    # Pivot to wide format
    matrix = subset.pivot_table(
        index=precursor_col,
        columns=replicate_col,
        values=abundance_col,
    )

    # Calculate CV per peptide (on linear scale)
    linear = np.power(2, matrix)
    cv_per_peptide = linear.std(axis=1) / linear.mean(axis=1)

    return cv_per_peptide.median()


def calculate_pca_distance(
    data: pd.DataFrame,
    qc_mask: pd.Series,
    reference_mask: pd.Series,
    abundance_col: str = 'abundance',
    precursor_col: str = 'precursor_id',
    replicate_col: str = 'replicate_name',
    n_components: int = 2,
) -> float:
    """Calculate distance between QC and reference centroids in PCA space.

    Args:
        data: DataFrame with peptide data
        qc_mask: Boolean mask for QC samples
        reference_mask: Boolean mask for reference samples
        abundance_col: Column with abundance values
        precursor_col: Column with precursor identifiers
        replicate_col: Column with replicate names
        n_components: Number of PCA components

    Returns:
        Euclidean distance between QC and reference centroids in PC space

    """
    # Get QC and reference samples
    control_mask = qc_mask | reference_mask
    subset = data.loc[control_mask]

    # Pivot to wide format (samples as rows, peptides as columns)
    matrix = subset.pivot_table(
        index=replicate_col,
        columns=precursor_col,
        values=abundance_col,
    )

    # Handle missing values - drop peptides with too many missing
    matrix = matrix.dropna(axis=1, thresh=len(matrix) * 0.5)
    matrix = matrix.fillna(matrix.median())

    if matrix.shape[1] < n_components:
        logger.warning(f"Too few peptides for PCA: {matrix.shape[1]}")
        return np.nan

    # Fit PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(matrix.values)
    scores_df = pd.DataFrame(
        scores,
        index=matrix.index,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )

    # Get sample names for each type
    qc_samples = data.loc[qc_mask, replicate_col].unique()
    ref_samples = data.loc[reference_mask, replicate_col].unique()

    # Calculate centroids
    qc_centroid = scores_df.loc[scores_df.index.isin(qc_samples)].mean()
    ref_centroid = scores_df.loc[scores_df.index.isin(ref_samples)].mean()

    # Euclidean distance
    distance = np.sqrt(((qc_centroid - ref_centroid) ** 2).sum())

    return distance


def validate_correction(
    data_before: pd.DataFrame,
    data_after: pd.DataFrame,
    sample_type_col: str = 'sample_type',
    abundance_col_before: str = 'abundance',
    abundance_col_after: str = 'abundance_normalized',
    precursor_col: str = 'precursor_id',
    replicate_col: str = 'replicate_name',
) -> ValidationMetrics:
    """Assess whether correction improved data quality without overcorrection.

    Uses the dual-control design:
    - Reference samples: should show CV improvement (technical variation removed)
    - QC samples: should show similar CV improvement AND remain distinct from reference

    Args:
        data_before: DataFrame with original data
        data_after: DataFrame with normalized data
        sample_type_col: Column indicating sample type
        abundance_col_before: Abundance column in before data
        abundance_col_after: Abundance column in after data
        precursor_col: Column with precursor identifiers
        replicate_col: Column with replicate names

    Returns:
        ValidationMetrics with assessment results

    """
    logger.info("Validating normalization quality")

    warnings = []

    # Create masks for sample types
    qc_mask_before = data_before[sample_type_col] == 'qc'
    ref_mask_before = data_before[sample_type_col] == 'reference'
    qc_mask_after = data_after[sample_type_col] == 'qc'
    ref_mask_after = data_after[sample_type_col] == 'reference'

    # Calculate CVs before
    ref_cv_before = calculate_cv(
        data_before, ref_mask_before,
        abundance_col_before, precursor_col, replicate_col
    )
    qc_cv_before = calculate_cv(
        data_before, qc_mask_before,
        abundance_col_before, precursor_col, replicate_col
    )

    # Calculate CVs after
    ref_cv_after = calculate_cv(
        data_after, ref_mask_after,
        abundance_col_after, precursor_col, replicate_col
    )
    qc_cv_after = calculate_cv(
        data_after, qc_mask_after,
        abundance_col_after, precursor_col, replicate_col
    )

    # CV improvements
    ref_cv_improvement = (ref_cv_before - ref_cv_after) / ref_cv_before if ref_cv_before > 0 else 0
    qc_cv_improvement = (qc_cv_before - qc_cv_after) / qc_cv_before if qc_cv_before > 0 else 0

    # Relative variance reduction
    rvr = qc_cv_improvement / ref_cv_improvement if ref_cv_improvement > 0 else np.inf

    # Check for warnings
    if qc_cv_improvement < 0:
        warnings.append("QC CV increased after normalization")
    if ref_cv_improvement < 0:
        warnings.append("Reference CV increased after normalization")
    if rvr > 2.0:
        warnings.append(f"QC improved much more than reference (RVR={rvr:.2f}) - possible overfitting")
    if rvr < 0.5:
        warnings.append(f"QC improved much less than reference (RVR={rvr:.2f}) - normalization may not generalize")

    # Calculate PCA distances
    pca_dist_before = calculate_pca_distance(
        data_before, qc_mask_before, ref_mask_before,
        abundance_col_before, precursor_col, replicate_col
    )
    pca_dist_after = calculate_pca_distance(
        data_after, qc_mask_after, ref_mask_after,
        abundance_col_after, precursor_col, replicate_col
    )

    pca_ratio = pca_dist_after / pca_dist_before if pca_dist_before > 0 else np.nan

    if pca_ratio < 0.5:
        warnings.append(f"QC-reference PCA distance decreased by {(1-pca_ratio)*100:.1f}% - "
                       "samples may be collapsing together")

    metrics = ValidationMetrics(
        reference_cv_before=ref_cv_before,
        reference_cv_after=ref_cv_after,
        qc_cv_before=qc_cv_before,
        qc_cv_after=qc_cv_after,
        reference_cv_improvement=ref_cv_improvement,
        qc_cv_improvement=qc_cv_improvement,
        relative_variance_reduction=rvr,
        pca_qc_reference_distance_before=pca_dist_before,
        pca_qc_reference_distance_after=pca_dist_after,
        pca_distance_ratio=pca_ratio,
        warnings=warnings,
    )

    # Log summary
    logger.info(f"Reference CV: {ref_cv_before:.3f} -> {ref_cv_after:.3f} "
                f"({ref_cv_improvement*100:.1f}% improvement)")
    logger.info(f"QC CV: {qc_cv_before:.3f} -> {qc_cv_after:.3f} "
                f"({qc_cv_improvement*100:.1f}% improvement)")
    logger.info(f"PCA distance ratio: {pca_ratio:.2f}")

    if warnings:
        for w in warnings:
            logger.warning(w)

    if metrics.passed:
        logger.info("Validation PASSED")
    else:
        logger.warning("Validation FAILED - review warnings")

    return metrics


def generate_qc_report(
    metrics: ValidationMetrics,
    normalization_log: list[str],
    output_path: str,
    data_before: Optional[pd.DataFrame] = None,
    data_after: Optional[pd.DataFrame] = None,
    data_batch_corrected: Optional[pd.DataFrame] = None,
    reference_stats: Optional[pd.DataFrame] = None,
    sample_type_col: str = "sample_type",
    precursor_col: str = "precursor_id",
    sample_col: str = "replicate_name",
    abundance_col: str = "abundance",
    rt_col: str = "retention_time",
    save_plots: bool = True,
    embed_plots: bool = True,
) -> dict[str, str]:
    """Generate HTML QC report with embedded or linked PNG plots.

    Args:
        metrics: ValidationMetrics from validate_correction
        normalization_log: List of processing steps applied
        output_path: Path to save HTML report
        data_before: Optional original data for plots
        data_after: Optional normalized data for plots
        data_batch_corrected: Optional batch-corrected data for plots
        reference_stats: Optional reference statistics for RT plots
        sample_type_col: Column name with sample type
        precursor_col: Column name with precursor identifiers
        sample_col: Column name with sample identifiers
        abundance_col: Column name with abundance values
        rt_col: Column name with retention times
        save_plots: Whether to save individual PNG files
        embed_plots: Whether to embed plots in HTML (base64)

    Returns:
        Dict mapping plot names to file paths (if save_plots=True)

    """
    output_path = Path(output_path)
    plots_dir = output_path.parent / "qc_plots"
    plot_paths = {}

    # Create plots directory if saving
    if save_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots if we have data and matplotlib
    plot_html_sections = []

    if HAS_MATPLOTLIB and data_before is not None:
        from . import visualization as viz

        # Get sample types mapping
        sample_types = None
        if sample_type_col in data_before.columns:
            sample_types = (
                data_before.drop_duplicates(sample_col)
                .set_index(sample_col)[sample_type_col]
                .to_dict()
            )

        # 1. Intensity Distribution Comparison
        if data_after is not None:
            try:
                fig = viz.plot_normalization_comparison(
                    data_before,
                    data_after,
                    sample_col=sample_col,
                    abundance_col_before=abundance_col,
                    abundance_col_after=abundance_col,
                    title="Intensity Distribution: Before vs After Normalization",
                    show_plot=False,
                )
                if fig is not None:
                    plot_html, plot_path = _save_and_embed_plot(
                        fig, "intensity_distribution", plots_dir, save_plots, embed_plots
                    )
                    plot_html_sections.append(("Intensity Distribution", plot_html))
                    if plot_path:
                        plot_paths["intensity_distribution"] = str(plot_path)
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to generate intensity distribution plot: {e}")

        # 2. PCA Comparison
        try:
            fig = viz.plot_comparative_pca(
                data_before,
                data_after if data_after is not None else data_before,
                data_batch_corrected,
                sample_col=sample_col,
                precursor_col=precursor_col,
                abundance_col_original=abundance_col,
                abundance_col_normalized=abundance_col,
                abundance_col_corrected=abundance_col,
                sample_groups=sample_types,
                show_plot=False,
            )
            if fig is not None:
                plot_html, plot_path = _save_and_embed_plot(
                    fig, "pca_comparison", plots_dir, save_plots, embed_plots
                )
                plot_html_sections.append(("PCA Analysis", plot_html))
                if plot_path:
                    plot_paths["pca_comparison"] = str(plot_path)
                plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to generate PCA plot: {e}")

        # 3. Control Sample Correlation
        try:
            fig, _ = viz.plot_control_correlation_heatmap(
                data_after if data_after is not None else data_before,
                sample_col=sample_col,
                precursor_col=precursor_col,
                abundance_col=abundance_col,
                sample_type_col=sample_type_col,
                control_types=["reference", "qc"],
                title="Control Sample Correlation (After Normalization)",
                show_plot=False,
            )
            if fig is not None:
                plot_html, plot_path = _save_and_embed_plot(
                    fig, "control_correlation", plots_dir, save_plots, embed_plots
                )
                plot_html_sections.append(("Control Sample Correlation", plot_html))
                if plot_path:
                    plot_paths["control_correlation"] = str(plot_path)
                plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to generate correlation heatmap: {e}")

        # 4. CV Distribution Comparison
        if data_after is not None:
            try:
                fig = viz.plot_comparative_cv(
                    data_before,
                    data_after,
                    sample_col=sample_col,
                    precursor_col=precursor_col,
                    abundance_col_before=abundance_col,
                    abundance_col_after=abundance_col,
                    sample_type_col=sample_type_col,
                    control_type="reference",
                    show_plot=False,
                )
                if fig is not None:
                    plot_html, plot_path = _save_and_embed_plot(
                        fig, "cv_distribution_reference", plots_dir, save_plots, embed_plots
                    )
                    plot_html_sections.append(("CV Distribution (Reference)", plot_html))
                    if plot_path:
                        plot_paths["cv_distribution_reference"] = str(plot_path)
                    plt.close(fig)

                # QC CV
                fig = viz.plot_comparative_cv(
                    data_before,
                    data_after,
                    sample_col=sample_col,
                    precursor_col=precursor_col,
                    abundance_col_before=abundance_col,
                    abundance_col_after=abundance_col,
                    sample_type_col=sample_type_col,
                    control_type="qc",
                    show_plot=False,
                )
                if fig is not None:
                    plot_html, plot_path = _save_and_embed_plot(
                        fig, "cv_distribution_qc", plots_dir, save_plots, embed_plots
                    )
                    plot_html_sections.append(("CV Distribution (QC)", plot_html))
                    if plot_path:
                        plot_paths["cv_distribution_qc"] = str(plot_path)
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to generate CV distribution plot: {e}")

        # 5. RT Correction Plots (if RT data and reference stats available)
        if (
            reference_stats is not None
            and rt_col in data_before.columns
            and data_after is not None
        ):
            try:
                fig = viz.plot_rt_correction_comparison(
                    data_before,
                    data_after,
                    reference_stats,
                    sample_col=sample_col,
                    precursor_col=precursor_col,
                    abundance_col=abundance_col,
                    rt_col=rt_col,
                    sample_type_col=sample_type_col,
                    show_plot=False,
                )
                if fig is not None:
                    plot_html, plot_path = _save_and_embed_plot(
                        fig, "rt_correction", plots_dir, save_plots, embed_plots
                    )
                    plot_html_sections.append(("RT-Dependent Normalization QC", plot_html))
                    if plot_path:
                        plot_paths["rt_correction"] = str(plot_path)
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to generate RT correction plot: {e}")

    # Build plots HTML
    plots_html = ""
    for title, plot_content in plot_html_sections:
        plots_html += f"""
        <div class="plot-section">
            <h3>{title}</h3>
            {plot_content}
        </div>
        """

    # Generate the HTML report
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>PRISM QC Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white;
                     padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #0066cc; padding-bottom: 10px; }}
        h2 {{ color: #444; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 8px; }}
        h3 {{ color: #555; }}
        .metric {{ margin: 10px 0; padding: 8px; background: #f9f9f9; border-radius: 4px; }}
        .metric-name {{ font-weight: 600; color: #333; }}
        .metric-value {{ color: #0066cc; font-family: monospace; font-size: 1.1em; }}
        .warning {{ color: #856404; background: #fff3cd; border: 1px solid #ffc107;
                   padding: 12px; margin: 8px 0; border-radius: 4px; }}
        .passed {{ color: #155724; background: #d4edda; border: 1px solid #28a745;
                  padding: 15px; border-radius: 4px; font-size: 1.1em; }}
        .failed {{ color: #721c24; background: #f8d7da; border: 1px solid #dc3545;
                  padding: 15px; border-radius: 4px; font-size: 1.1em; }}
        table {{ border-collapse: collapse; margin: 20px 0; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #f0f0f0; font-weight: 600; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .plot-section {{ margin: 20px 0; text-align: center; }}
        .plot-section img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
        .timestamp {{ color: #888; font-size: 0.9em; margin-top: 30px; padding-top: 20px;
                     border-top: 1px solid #ddd; }}
        ol {{ line-height: 1.8; }}
        .improvement-positive {{ color: #28a745; }}
        .improvement-negative {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 PRISM QC Report</h1>

        <h2>Validation Status</h2>
        <div class="{'passed' if metrics.passed else 'failed'}">
            <strong>{'✓ PASSED' if metrics.passed else '✗ FAILED'}</strong> —
            {'All validation criteria met' if metrics.passed else 'Review warnings below'}
        </div>

        <h2>CV Metrics</h2>
        <table>
            <tr>
                <th>Sample Type</th>
                <th>CV Before</th>
                <th>CV After</th>
                <th>Improvement</th>
            </tr>
            <tr>
                <td><strong>Reference</strong> (calibration)</td>
                <td>{metrics.reference_cv_before:.3f}</td>
                <td>{metrics.reference_cv_after:.3f}</td>
                <td class="{'improvement-positive' if metrics.reference_cv_improvement > 0 else 'improvement-negative'}">
                    {metrics.reference_cv_improvement*100:+.1f}%
                </td>
            </tr>
            <tr>
                <td><strong>QC</strong> (validation)</td>
                <td>{metrics.qc_cv_before:.3f}</td>
                <td>{metrics.qc_cv_after:.3f}</td>
                <td class="{'improvement-positive' if metrics.qc_cv_improvement > 0 else 'improvement-negative'}">
                    {metrics.qc_cv_improvement*100:+.1f}%
                </td>
            </tr>
        </table>

        <div class="metric">
            <span class="metric-name">Relative Variance Reduction (RVR):</span>
            <span class="metric-value">{metrics.relative_variance_reduction:.2f}</span>
            <br><small style="color: #666;">Target: ~1.0 | &gt;&gt;1 suggests overfitting | &lt;&lt;1 suggests poor generalization</small>
        </div>

        <h2>PCA Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Interpretation</th>
            </tr>
            <tr>
                <td>QC-Reference Distance (Before)</td>
                <td>{metrics.pca_qc_reference_distance_before:.2f}</td>
                <td>Baseline separation</td>
            </tr>
            <tr>
                <td>QC-Reference Distance (After)</td>
                <td>{metrics.pca_qc_reference_distance_after:.2f}</td>
                <td>Post-normalization separation</td>
            </tr>
            <tr>
                <td>Distance Ratio</td>
                <td><strong>{metrics.pca_distance_ratio:.2f}</strong></td>
                <td>{'Good - samples remain distinct' if metrics.pca_distance_ratio > 0.5 else 'WARNING - Samples may be collapsing'}</td>
            </tr>
        </table>

        <h2>Warnings</h2>
        {''.join(f'<div class="warning">⚠️ {w}</div>' for w in metrics.warnings) if metrics.warnings else '<p style="color: #28a745;">✓ No warnings</p>'}

        <h2>QC Plots</h2>
        {plots_html if plots_html else '<p style="color: #888;">No plots generated (data not provided or matplotlib unavailable)</p>'}

        <h2>Processing Steps</h2>
        <ol>
            {''.join(f'<li>{step}</li>' for step in normalization_log) if normalization_log else '<li>No processing steps recorded</li>'}
        </ol>

        <div class="timestamp">
            Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>"""

    # Write the HTML file
    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"QC report saved to {output_path}")
    if save_plots and plot_paths:
        logger.info(f"QC plots saved to {plots_dir}")

    return plot_paths


def _save_and_embed_plot(
    fig,
    name: str,
    plots_dir: Path,
    save_plots: bool,
    embed_plots: bool,
) -> tuple[str, Optional[Path]]:
    """Save a matplotlib figure and/or generate HTML for embedding.

    Args:
        fig: matplotlib Figure object
        name: Base name for the plot file
        plots_dir: Directory to save plots
        save_plots: Whether to save PNG file
        embed_plots: Whether to embed as base64

    Returns:
        Tuple of (HTML string for embedding, path if saved)

    """
    plot_path = None
    html = ""

    if save_plots:
        plot_path = plots_dir / f"{name}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")

    if embed_plots:
        # Convert to base64 for embedding
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor="white")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        html = f'<img src="data:image/png;base64,{img_base64}" alt="{name}" />'
    elif plot_path:
        # Link to saved file
        html = f'<img src="qc_plots/{name}.png" alt="{name}" />'

    return html, plot_path

def generate_comprehensive_qc_report(
    peptide_raw: pd.DataFrame,
    peptide_corrected: pd.DataFrame,
    protein_raw: pd.DataFrame,
    protein_corrected: pd.DataFrame,
    sample_cols: list[str],
    sample_types: dict[str, str],
    output_path: str | Path,
    method_log: list[str],
    config: dict | None = None,
    save_plots: bool = True,
    embed_plots: bool = True,
    peptide_rawsum: pd.DataFrame | None = None,
    peptide_medianpolish: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Generate comprehensive QC report with multi-stage plots for peptides and proteins.

    Creates an HTML report with diagnostic plots comparing processing stages.
    If peptide_rawsum and peptide_medianpolish are provided, generates 3-stage
    comparison plots (rawsum -> medianpolish -> normalized).

    Args:
        peptide_raw: Wide-format DataFrame with raw peptide abundances (legacy, used if
                     peptide_medianpolish not provided)
        peptide_corrected: Wide-format DataFrame with corrected peptide abundances
        protein_raw: Wide-format DataFrame with raw protein abundances
        protein_corrected: Wide-format DataFrame with corrected protein abundances
        sample_cols: List of column names containing sample abundances
        sample_types: Dict mapping sample name to type ('experimental', 'qc', 'reference')
        output_path: Path to save HTML report
        method_log: List of processing steps applied
        config: Optional config dict for QC report settings
        save_plots: Whether to save individual PNG files
        embed_plots: Whether to embed plots in HTML (base64)
        peptide_rawsum: Wide-format DataFrame with sum-aggregated peptides (stage 1)
        peptide_medianpolish: Wide-format DataFrame after median polish (stage 2)

    Returns:
        Dict mapping plot names to file paths (if save_plots=True)

    """
    output_path = Path(output_path)
    plots_dir = output_path.parent / "qc_plots"
    plot_paths = {}
    peptide_plot_sections = []
    protein_plot_sections = []

    # Determine if we have 3-stage data
    has_three_stage = peptide_rawsum is not None and peptide_medianpolish is not None

    # Create plots directory if saving
    if save_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    # Get config settings
    if config is None:
        config = {}
    qc_config = config.get('qc_report', {})
    plot_settings = qc_config.get('plots', {})

    # Check if matplotlib is available
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available - skipping QC plots")
    else:
        from . import visualization as viz

        # =====================================================================
        # PEPTIDE-LEVEL PLOTS
        # =====================================================================
        logger.info("Generating peptide-level QC plots...")

        # 1. Peptide Intensity Distribution Comparison
        if plot_settings.get('intensity_distribution', True):
            try:
                # Use median polish as "raw" for intensity comparison
                raw_for_intensity = peptide_medianpolish if has_three_stage else peptide_raw
                fig = viz.plot_normalization_comparison_wide(
                    raw_for_intensity,
                    peptide_corrected,
                    sample_cols,
                    title="Peptide Intensity Distribution",
                    show_plot=False,
                )
                if fig is not None:
                    plot_html, plot_path = _save_and_embed_plot(
                        fig, "peptide_intensity_distribution", plots_dir,
                        save_plots, embed_plots
                    )
                    peptide_plot_sections.append(
                        ("Peptide Intensity Distribution", plot_html)
                    )
                    if plot_path:
                        plot_paths["peptide_intensity_distribution"] = str(plot_path)
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to generate peptide intensity plot: {e}")

        # 1b. Peptide Box Plot Comparison (3-stage if available, 2-stage otherwise)
        if plot_settings.get('boxplot_comparison', True):
            try:
                if has_three_stage:
                    fig = viz.plot_boxplot_three_stage(
                        peptide_rawsum,
                        peptide_medianpolish,
                        peptide_corrected,
                        sample_cols,
                        sample_types=sample_types,
                        title="Peptide Intensity: Processing Stages",
                        show_plot=False,
                    )
                    plot_title = "Peptide Box Plot (3 Stages)"
                else:
                    fig = viz.plot_boxplot_two_stage(
                        peptide_raw,
                        peptide_corrected,
                        sample_cols,
                        sample_types=sample_types,
                        before_label="Raw (Sum Rollup)",
                        after_label="Normalized + Corrected",
                        title="Peptide Intensity: Before vs After",
                        show_plot=False,
                    )
                    plot_title = "Peptide Box Plot Comparison"
                if fig is not None:
                    plot_html, plot_path = _save_and_embed_plot(
                        fig, "peptide_boxplot_comparison", plots_dir,
                        save_plots, embed_plots
                    )
                    peptide_plot_sections.append(
                        (plot_title, plot_html)
                    )
                    if plot_path:
                        plot_paths["peptide_boxplot_comparison"] = str(plot_path)
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to generate peptide boxplot: {e}")

        # 2. Peptide PCA Comparison (3-stage if available)
        if plot_settings.get('pca_comparison', True):
            try:
                if has_three_stage:
                    fig = viz.plot_pca_three_stage(
                        peptide_rawsum,
                        peptide_medianpolish,
                        peptide_corrected,
                        sample_cols,
                        sample_types=sample_types,
                        title="Peptide PCA: Processing Stages",
                        show_plot=False,
                    )
                else:
                    fig = viz.plot_comparative_pca_wide(
                        peptide_raw,
                        peptide_corrected,
                        sample_cols,
                        sample_types=sample_types,
                        title="Peptide PCA: Before vs After Normalization",
                        show_plot=False,
                    )
                if fig is not None:
                    plot_html, plot_path = _save_and_embed_plot(
                        fig, "peptide_pca_comparison", plots_dir,
                        save_plots, embed_plots
                    )
                    peptide_plot_sections.append(("Peptide PCA Analysis", plot_html))
                    if plot_path:
                        plot_paths["peptide_pca_comparison"] = str(plot_path)
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to generate peptide PCA plot: {e}")

        # 3. Peptide CV Distribution - Reference Samples (3-stage if available)
        if plot_settings.get('cv_distribution', True):
            # Check if we have reference samples
            ref_cols = [c for c in sample_cols if sample_types.get(c) == 'reference']
            if len(ref_cols) >= 2:
                try:
                    if has_three_stage:
                        fig = viz.plot_cv_three_stage(
                            peptide_rawsum,
                            peptide_medianpolish,
                            peptide_corrected,
                            sample_cols,
                            sample_types=sample_types,
                            control_type="reference",
                            title="Peptide CV Distribution (Reference)",
                            show_plot=False,
                        )
                    else:
                        fig = viz.plot_cv_comparison_wide(
                            peptide_raw,
                            peptide_corrected,
                            sample_cols,
                            sample_types=sample_types,
                            control_type="reference",
                            title="Peptide CV Distribution (Reference)",
                            show_plot=False,
                        )
                    if fig is not None:
                        plot_html, plot_path = _save_and_embed_plot(
                            fig, "peptide_cv_reference", plots_dir,
                            save_plots, embed_plots
                        )
                        peptide_plot_sections.append(
                            ("Peptide CV (Reference Samples)", plot_html)
                        )
                        if plot_path:
                            plot_paths["peptide_cv_reference"] = str(plot_path)
                        plt.close(fig)
                except Exception as e:
                    logger.warning(f"Failed to generate peptide CV (ref) plot: {e}")

            # 4. Peptide CV Distribution - QC Samples (3-stage if available)
            qc_cols = [c for c in sample_cols if sample_types.get(c) == 'qc']
            if len(qc_cols) >= 2:
                try:
                    if has_three_stage:
                        fig = viz.plot_cv_three_stage(
                            peptide_rawsum,
                            peptide_medianpolish,
                            peptide_corrected,
                            sample_cols,
                            sample_types=sample_types,
                            control_type="qc",
                            title="Peptide CV Distribution (QC)",
                            show_plot=False,
                        )
                    else:
                        fig = viz.plot_cv_comparison_wide(
                            peptide_raw,
                            peptide_corrected,
                            sample_cols,
                            sample_types=sample_types,
                            control_type="qc",
                            title="Peptide CV Distribution (QC)",
                            show_plot=False,
                        )
                    if fig is not None:
                        plot_html, plot_path = _save_and_embed_plot(
                            fig, "peptide_cv_qc", plots_dir,
                            save_plots, embed_plots
                        )
                        peptide_plot_sections.append(
                            ("Peptide CV (QC Samples)", plot_html)
                        )
                        if plot_path:
                            plot_paths["peptide_cv_qc"] = str(plot_path)
                        plt.close(fig)
                except Exception as e:
                    logger.warning(f"Failed to generate peptide CV (qc) plot: {e}")

        # 5. Peptide Control Correlation
        if plot_settings.get('control_correlation', True):
            try:
                fig, _ = viz.plot_control_correlation_wide(
                    peptide_corrected,
                    sample_cols,
                    sample_types=sample_types,
                    control_types=["reference", "qc"],
                    title="Peptide Control Sample Correlation (After Correction)",
                    show_plot=False,
                )
                if fig is not None:
                    plot_html, plot_path = _save_and_embed_plot(
                        fig, "peptide_control_correlation", plots_dir,
                        save_plots, embed_plots
                    )
                    peptide_plot_sections.append(
                        ("Peptide Control Correlation", plot_html)
                    )
                    if plot_path:
                        plot_paths["peptide_control_correlation"] = str(plot_path)
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to generate peptide correlation plot: {e}")

        # =====================================================================
        # PROTEIN-LEVEL PLOTS
        # =====================================================================
        logger.info("Generating protein-level QC plots...")

        # 6. Protein Intensity Distribution Comparison
        if plot_settings.get('intensity_distribution', True):
            try:
                fig = viz.plot_normalization_comparison_wide(
                    protein_raw,
                    protein_corrected,
                    sample_cols,
                    title="Protein Intensity Distribution",
                    show_plot=False,
                )
                if fig is not None:
                    plot_html, plot_path = _save_and_embed_plot(
                        fig, "protein_intensity_distribution", plots_dir,
                        save_plots, embed_plots
                    )
                    protein_plot_sections.append(
                        ("Protein Intensity Distribution", plot_html)
                    )
                    if plot_path:
                        plot_paths["protein_intensity_distribution"] = str(plot_path)
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to generate protein intensity plot: {e}")

        # 6b. Protein Box Plot Comparison
        if plot_settings.get('boxplot_comparison', True):
            try:
                fig = viz.plot_boxplot_two_stage(
                    protein_raw,
                    protein_corrected,
                    sample_cols,
                    sample_types=sample_types,
                    before_label="Raw (Median Polish Rollup)",
                    after_label="Normalized + Corrected",
                    title="Protein Intensity: Before vs After",
                    show_plot=False,
                )
                if fig is not None:
                    plot_html, plot_path = _save_and_embed_plot(
                        fig, "protein_boxplot_comparison", plots_dir,
                        save_plots, embed_plots
                    )
                    protein_plot_sections.append(
                        ("Protein Box Plot Comparison", plot_html)
                    )
                    if plot_path:
                        plot_paths["protein_boxplot_comparison"] = str(plot_path)
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to generate protein boxplot: {e}")

        # 7. Protein PCA Comparison
        if plot_settings.get('pca_comparison', True):
            try:
                fig = viz.plot_comparative_pca_wide(
                    protein_raw,
                    protein_corrected,
                    sample_cols,
                    sample_types=sample_types,
                    title="Protein PCA: Before vs After Normalization",
                    show_plot=False,
                )
                if fig is not None:
                    plot_html, plot_path = _save_and_embed_plot(
                        fig, "protein_pca_comparison", plots_dir,
                        save_plots, embed_plots
                    )
                    protein_plot_sections.append(("Protein PCA Analysis", plot_html))
                    if plot_path:
                        plot_paths["protein_pca_comparison"] = str(plot_path)
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to generate protein PCA plot: {e}")

        # 8. Protein CV Distribution - Reference Samples
        if plot_settings.get('cv_distribution', True):
            ref_cols = [c for c in sample_cols if sample_types.get(c) == 'reference']
            if len(ref_cols) >= 2:
                try:
                    fig = viz.plot_cv_comparison_wide(
                        protein_raw,
                        protein_corrected,
                        sample_cols,
                        sample_types=sample_types,
                        control_type="reference",
                        title="Protein CV Distribution (Reference)",
                        show_plot=False,
                    )
                    if fig is not None:
                        plot_html, plot_path = _save_and_embed_plot(
                            fig, "protein_cv_reference", plots_dir,
                            save_plots, embed_plots
                        )
                        protein_plot_sections.append(
                            ("Protein CV (Reference Samples)", plot_html)
                        )
                        if plot_path:
                            plot_paths["protein_cv_reference"] = str(plot_path)
                        plt.close(fig)
                except Exception as e:
                    logger.warning(f"Failed to generate protein CV (ref) plot: {e}")

            # 9. Protein CV Distribution - QC Samples
            qc_cols = [c for c in sample_cols if sample_types.get(c) == 'qc']
            if len(qc_cols) >= 2:
                try:
                    fig = viz.plot_cv_comparison_wide(
                        protein_raw,
                        protein_corrected,
                        sample_cols,
                        sample_types=sample_types,
                        control_type="qc",
                        title="Protein CV Distribution (QC)",
                        show_plot=False,
                    )
                    if fig is not None:
                        plot_html, plot_path = _save_and_embed_plot(
                            fig, "protein_cv_qc", plots_dir,
                            save_plots, embed_plots
                        )
                        protein_plot_sections.append(
                            ("Protein CV (QC Samples)", plot_html)
                        )
                        if plot_path:
                            plot_paths["protein_cv_qc"] = str(plot_path)
                        plt.close(fig)
                except Exception as e:
                    logger.warning(f"Failed to generate protein CV (qc) plot: {e}")

        # 10. Protein Control Correlation
        if plot_settings.get('control_correlation', True):
            try:
                fig, _ = viz.plot_control_correlation_wide(
                    protein_corrected,
                    sample_cols,
                    sample_types=sample_types,
                    control_types=["reference", "qc"],
                    title="Protein Control Sample Correlation (After Correction)",
                    show_plot=False,
                )
                if fig is not None:
                    plot_html, plot_path = _save_and_embed_plot(
                        fig, "protein_control_correlation", plots_dir,
                        save_plots, embed_plots
                    )
                    protein_plot_sections.append(
                        ("Protein Control Correlation", plot_html)
                    )
                    if plot_path:
                        plot_paths["protein_control_correlation"] = str(plot_path)
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to generate protein correlation plot: {e}")

    # =========================================================================
    # Calculate Summary Metrics (now with 3-stage support)
    # =========================================================================
    def calc_median_cv(df, cols):
        """Calculate median CV across features."""
        linear_data = 2 ** df[cols]
        cv_values = (linear_data.std(axis=1) / linear_data.mean(axis=1)) * 100
        return cv_values.median()

    metrics = {}
    ref_cols = [c for c in sample_cols if sample_types.get(c) == 'reference']
    qc_cols = [c for c in sample_cols if sample_types.get(c) == 'qc']

    # Use 3-stage data if available
    peptide_raw_for_metrics = peptide_rawsum if has_three_stage else peptide_raw
    peptide_mid_for_metrics = peptide_medianpolish if has_three_stage else None

    if len(ref_cols) >= 2:
        metrics['peptide_ref_cv_rawsum'] = calc_median_cv(peptide_raw_for_metrics, ref_cols)
        if peptide_mid_for_metrics is not None:
            metrics['peptide_ref_cv_medpol'] = calc_median_cv(peptide_mid_for_metrics, ref_cols)
        metrics['peptide_ref_cv_normalized'] = calc_median_cv(peptide_corrected, ref_cols)
        metrics['protein_ref_cv_before'] = calc_median_cv(protein_raw, ref_cols)
        metrics['protein_ref_cv_after'] = calc_median_cv(protein_corrected, ref_cols)

    if len(qc_cols) >= 2:
        metrics['peptide_qc_cv_rawsum'] = calc_median_cv(peptide_raw_for_metrics, qc_cols)
        if peptide_mid_for_metrics is not None:
            metrics['peptide_qc_cv_medpol'] = calc_median_cv(peptide_mid_for_metrics, qc_cols)
        metrics['peptide_qc_cv_normalized'] = calc_median_cv(peptide_corrected, qc_cols)
        metrics['protein_qc_cv_before'] = calc_median_cv(protein_raw, qc_cols)
        metrics['protein_qc_cv_after'] = calc_median_cv(protein_corrected, qc_cols)

    # =========================================================================
    # Build peptide plots HTML
    # =========================================================================
    peptide_plots_html = ""
    for title, plot_content in peptide_plot_sections:
        peptide_plots_html += f"""
        <div class="plot-section">
            <h3>{title}</h3>
            {plot_content}
        </div>
        """

    # =========================================================================
    # Build protein plots HTML
    # =========================================================================
    protein_plots_html = ""
    for title, plot_content in protein_plot_sections:
        protein_plots_html += f"""
        <div class="plot-section">
            <h3>{title}</h3>
            {plot_content}
        </div>
        """

    # =========================================================================
    # Build metrics HTML (3-stage for peptides)
    # =========================================================================
    metrics_html = ""
    if metrics:
        metrics_html = """
        <h2>Summary Metrics (Median CV %)</h2>
        """
        # Peptide metrics table (3-stage)
        if has_three_stage:
            metrics_html += """
        <h3>Peptide-Level CV</h3>
        <table>
            <tr>
                <th>Sample Type</th>
                <th>Raw Sum</th>
                <th>Median Polish</th>
                <th>Normalized</th>
                <th>Total Improvement</th>
            </tr>
            """
            for sample_type, type_key in [('Reference', 'ref'), ('QC', 'qc')]:
                rawsum_key = f'peptide_{type_key}_cv_rawsum'
                medpol_key = f'peptide_{type_key}_cv_medpol'
                norm_key = f'peptide_{type_key}_cv_normalized'
                if rawsum_key in metrics and norm_key in metrics:
                    cv_rawsum = metrics[rawsum_key]
                    cv_medpol = metrics.get(medpol_key, float('nan'))
                    cv_norm = metrics[norm_key]
                    improvement = (cv_rawsum - cv_norm) / cv_rawsum * 100 if cv_rawsum > 0 else 0
                    improvement_class = "improvement-positive" if improvement > 0 else "improvement-negative"
                    metrics_html += f"""
            <tr>
                <td>{sample_type}</td>
                <td>{cv_rawsum:.1f}%</td>
                <td>{cv_medpol:.1f}%</td>
                <td>{cv_norm:.1f}%</td>
                <td class="{improvement_class}">{improvement:+.1f}%</td>
            </tr>
                    """
            metrics_html += "</table>"
        else:
            # Legacy 2-stage peptide table
            metrics_html += """
        <h3>Peptide-Level CV</h3>
        <table>
            <tr>
                <th>Sample Type</th>
                <th>Before</th>
                <th>After</th>
                <th>Improvement</th>
            </tr>
            """
            for sample_type, type_key in [('Reference', 'ref'), ('QC', 'qc')]:
                before_key = f'peptide_{type_key}_cv_rawsum'
                after_key = f'peptide_{type_key}_cv_normalized'
                if before_key in metrics and after_key in metrics:
                    cv_before = metrics[before_key]
                    cv_after = metrics[after_key]
                    improvement = (cv_before - cv_after) / cv_before * 100 if cv_before > 0 else 0
                    improvement_class = "improvement-positive" if improvement > 0 else "improvement-negative"
                    metrics_html += f"""
            <tr>
                <td>{sample_type}</td>
                <td>{cv_before:.1f}%</td>
                <td>{cv_after:.1f}%</td>
                <td class="{improvement_class}">{improvement:+.1f}%</td>
            </tr>
                    """
            metrics_html += "</table>"

        # Protein metrics table (always 2-stage)
        metrics_html += """
        <h3>Protein-Level CV</h3>
        <table>
            <tr>
                <th>Sample Type</th>
                <th>Before</th>
                <th>After</th>
                <th>Improvement</th>
            </tr>
        """
        for sample_type, type_key in [('Reference', 'ref'), ('QC', 'qc')]:
            before_key = f'protein_{type_key}_cv_before'
            after_key = f'protein_{type_key}_cv_after'
            if before_key in metrics and after_key in metrics:
                cv_before = metrics[before_key]
                cv_after = metrics[after_key]
                improvement = (cv_before - cv_after) / cv_before * 100 if cv_before > 0 else 0
                improvement_class = "improvement-positive" if improvement > 0 else "improvement-negative"
                metrics_html += f"""
            <tr>
                <td>{sample_type}</td>
                <td>{cv_before:.1f}%</td>
                <td>{cv_after:.1f}%</td>
                <td class="{improvement_class}">{improvement:+.1f}%</td>
            </tr>
                """
        metrics_html += "</table>"

    # =========================================================================
    # Generate the HTML report
    # =========================================================================
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>PRISM QC Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white;
                     padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #0066cc; padding-bottom: 10px; }}
        h2 {{ color: #444; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 8px; }}
        h3 {{ color: #555; }}
        table {{ border-collapse: collapse; margin: 20px 0; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #f0f0f0; font-weight: 600; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .plot-section {{ margin: 30px 0; text-align: center; }}
        .plot-section img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
        .timestamp {{ color: #888; font-size: 0.9em; margin-top: 30px; padding-top: 20px;
                     border-top: 1px solid #ddd; }}
        ol {{ line-height: 1.8; }}
        .improvement-positive {{ color: #28a745; font-weight: bold; }}
        .improvement-negative {{ color: #dc3545; font-weight: bold; }}
        .section-header {{ background: linear-gradient(to right, #0066cc, #0099cc);
                          color: white; padding: 10px 15px; margin: 30px 0 20px 0;
                          border-radius: 4px; }}
        .summary-box {{ background: #e8f4fc; border: 1px solid #b8daff;
                       padding: 15px; border-radius: 4px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>PRISM QC Report</h1>

        <div class="summary-box">
            <strong>Pipeline Summary:</strong>
            <ul>
                <li>Peptides: {len(peptide_corrected):,} features</li>
                <li>Proteins: {len(protein_corrected):,} features</li>
                <li>Samples: {len(sample_cols)}</li>
                <li>Reference samples: {len([c for c in sample_cols if sample_types.get(c) == 'reference'])}</li>
                <li>QC samples: {len([c for c in sample_cols if sample_types.get(c) == 'qc'])}</li>
            </ul>
        </div>

        {metrics_html}

        <div class="section-header">
            <h2 style="margin: 0; color: white;">Peptide-Level QC</h2>
        </div>

        {peptide_plots_html if peptide_plots_html else '<p>No peptide-level plots generated.</p>'}

        <div class="section-header">
            <h2 style="margin: 0; color: white;">Protein-Level QC</h2>
        </div>

        {protein_plots_html if protein_plots_html else '<p>No protein-level plots generated.</p>'}

        <h2>Processing Steps</h2>
        <ol>
            {''.join(f'<li>{step}</li>' for step in method_log) if method_log else '<li>No processing steps recorded</li>'}
        </ol>

        <div class="timestamp">
            Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>"""

    # Write the HTML file
    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"QC report saved to {output_path}")
    if save_plots and plot_paths:
        logger.info(f"QC plots saved to {plots_dir}")

    return plot_paths
