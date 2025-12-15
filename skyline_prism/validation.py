"""
Validation module for assessing normalization quality.

Uses the dual-control design:
- Inter-experiment reference: calibration anchor
- Intra-experiment pool: validation control
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Metrics for assessing normalization quality."""
    
    # CV metrics
    reference_cv_before: float
    reference_cv_after: float
    pool_cv_before: float
    pool_cv_after: float
    
    # CV improvement
    reference_cv_improvement: float  # (before - after) / before
    pool_cv_improvement: float
    relative_variance_reduction: float  # pool improvement / reference improvement
    
    # PCA metrics
    pca_pool_reference_distance_before: float
    pca_pool_reference_distance_after: float
    pca_distance_ratio: float  # after / before (should be ~1, not << 1)
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    @property
    def passed(self) -> bool:
        """Check if validation passed basic criteria."""
        return (
            self.pool_cv_improvement > 0 and  # Pool CV improved
            self.pca_distance_ratio > 0.5 and  # Didn't collapse pool into reference
            self.relative_variance_reduction < 2.0  # Didn't overfit to reference
        )


def calculate_cv(
    data: pd.DataFrame,
    sample_mask: pd.Series,
    abundance_col: str = 'abundance',
    precursor_col: str = 'precursor_id',
    replicate_col: str = 'replicate_name',
) -> float:
    """
    Calculate median coefficient of variation across peptides.
    
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
    pool_mask: pd.Series,
    reference_mask: pd.Series,
    abundance_col: str = 'abundance',
    precursor_col: str = 'precursor_id',
    replicate_col: str = 'replicate_name',
    n_components: int = 2,
) -> float:
    """
    Calculate distance between pool and reference centroids in PCA space.
    
    Args:
        data: DataFrame with peptide data
        pool_mask: Boolean mask for pool samples
        reference_mask: Boolean mask for reference samples
        abundance_col: Column with abundance values
        precursor_col: Column with precursor identifiers
        replicate_col: Column with replicate names
        n_components: Number of PCA components
        
    Returns:
        Euclidean distance between pool and reference centroids in PC space
    """
    # Get pool and reference samples
    control_mask = pool_mask | reference_mask
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
    pool_samples = data.loc[pool_mask, replicate_col].unique()
    ref_samples = data.loc[reference_mask, replicate_col].unique()
    
    # Calculate centroids
    pool_centroid = scores_df.loc[scores_df.index.isin(pool_samples)].mean()
    ref_centroid = scores_df.loc[scores_df.index.isin(ref_samples)].mean()
    
    # Euclidean distance
    distance = np.sqrt(((pool_centroid - ref_centroid) ** 2).sum())
    
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
    """
    Assess whether correction improved data quality without overcorrection.
    
    Uses the dual-control design:
    - Reference samples: should show CV improvement (technical variation removed)
    - Pool samples: should show similar CV improvement AND remain distinct from reference
    
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
    pool_mask_before = data_before[sample_type_col] == 'pool'
    ref_mask_before = data_before[sample_type_col] == 'reference'
    pool_mask_after = data_after[sample_type_col] == 'pool'
    ref_mask_after = data_after[sample_type_col] == 'reference'
    
    # Calculate CVs before
    ref_cv_before = calculate_cv(
        data_before, ref_mask_before, 
        abundance_col_before, precursor_col, replicate_col
    )
    pool_cv_before = calculate_cv(
        data_before, pool_mask_before,
        abundance_col_before, precursor_col, replicate_col
    )
    
    # Calculate CVs after
    ref_cv_after = calculate_cv(
        data_after, ref_mask_after,
        abundance_col_after, precursor_col, replicate_col
    )
    pool_cv_after = calculate_cv(
        data_after, pool_mask_after,
        abundance_col_after, precursor_col, replicate_col
    )
    
    # CV improvements
    ref_cv_improvement = (ref_cv_before - ref_cv_after) / ref_cv_before if ref_cv_before > 0 else 0
    pool_cv_improvement = (pool_cv_before - pool_cv_after) / pool_cv_before if pool_cv_before > 0 else 0
    
    # Relative variance reduction
    rvr = pool_cv_improvement / ref_cv_improvement if ref_cv_improvement > 0 else np.inf
    
    # Check for warnings
    if pool_cv_improvement < 0:
        warnings.append("Pool CV increased after normalization")
    if ref_cv_improvement < 0:
        warnings.append("Reference CV increased after normalization")
    if rvr > 2.0:
        warnings.append(f"Pool improved much more than reference (RVR={rvr:.2f}) - possible overfitting")
    if rvr < 0.5:
        warnings.append(f"Pool improved much less than reference (RVR={rvr:.2f}) - normalization may not generalize")
    
    # Calculate PCA distances
    pca_dist_before = calculate_pca_distance(
        data_before, pool_mask_before, ref_mask_before,
        abundance_col_before, precursor_col, replicate_col
    )
    pca_dist_after = calculate_pca_distance(
        data_after, pool_mask_after, ref_mask_after,
        abundance_col_after, precursor_col, replicate_col
    )
    
    pca_ratio = pca_dist_after / pca_dist_before if pca_dist_before > 0 else np.nan
    
    if pca_ratio < 0.5:
        warnings.append(f"Pool-reference PCA distance decreased by {(1-pca_ratio)*100:.1f}% - "
                       "samples may be collapsing together")
    
    metrics = ValidationMetrics(
        reference_cv_before=ref_cv_before,
        reference_cv_after=ref_cv_after,
        pool_cv_before=pool_cv_before,
        pool_cv_after=pool_cv_after,
        reference_cv_improvement=ref_cv_improvement,
        pool_cv_improvement=pool_cv_improvement,
        relative_variance_reduction=rvr,
        pca_pool_reference_distance_before=pca_dist_before,
        pca_pool_reference_distance_after=pca_dist_after,
        pca_distance_ratio=pca_ratio,
        warnings=warnings,
    )
    
    # Log summary
    logger.info(f"Reference CV: {ref_cv_before:.3f} -> {ref_cv_after:.3f} "
                f"({ref_cv_improvement*100:.1f}% improvement)")
    logger.info(f"Pool CV: {pool_cv_before:.3f} -> {pool_cv_after:.3f} "
                f"({pool_cv_improvement*100:.1f}% improvement)")
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
    normalization_log: List[str],
    output_path: str,
    data_before: Optional[pd.DataFrame] = None,
    data_after: Optional[pd.DataFrame] = None,
) -> None:
    """
    Generate HTML QC report.
    
    Args:
        metrics: ValidationMetrics from validate_correction
        normalization_log: List of processing steps applied
        output_path: Path to save HTML report
        data_before: Optional original data for plots
        data_after: Optional normalized data for plots
    """
    # Simple HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Normalization QC Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; border-bottom: 1px solid #ccc; }}
            .metric {{ margin: 10px 0; }}
            .metric-name {{ font-weight: bold; }}
            .metric-value {{ color: #0066cc; }}
            .warning {{ color: #cc6600; background: #fff3e0; padding: 10px; margin: 5px 0; }}
            .passed {{ color: #006600; background: #e0ffe0; padding: 10px; }}
            .failed {{ color: #cc0000; background: #ffe0e0; padding: 10px; }}
            table {{ border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
            th {{ background: #f0f0f0; }}
        </style>
    </head>
    <body>
        <h1>Normalization QC Report</h1>
        
        <h2>Validation Status</h2>
        <div class="{'passed' if metrics.passed else 'failed'}">
            {'PASSED' if metrics.passed else 'FAILED'} - 
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
                <td>Reference</td>
                <td>{metrics.reference_cv_before:.3f}</td>
                <td>{metrics.reference_cv_after:.3f}</td>
                <td>{metrics.reference_cv_improvement*100:.1f}%</td>
            </tr>
            <tr>
                <td>Pool</td>
                <td>{metrics.pool_cv_before:.3f}</td>
                <td>{metrics.pool_cv_after:.3f}</td>
                <td>{metrics.pool_cv_improvement*100:.1f}%</td>
            </tr>
        </table>
        
        <div class="metric">
            <span class="metric-name">Relative Variance Reduction:</span>
            <span class="metric-value">{metrics.relative_variance_reduction:.2f}</span>
            <br><small>(Should be close to 1.0; >>1 suggests overfitting, <<1 suggests poor generalization)</small>
        </div>
        
        <h2>PCA Metrics</h2>
        <div class="metric">
            <span class="metric-name">Pool-Reference Distance Before:</span>
            <span class="metric-value">{metrics.pca_pool_reference_distance_before:.2f}</span>
        </div>
        <div class="metric">
            <span class="metric-name">Pool-Reference Distance After:</span>
            <span class="metric-value">{metrics.pca_pool_reference_distance_after:.2f}</span>
        </div>
        <div class="metric">
            <span class="metric-name">Distance Ratio:</span>
            <span class="metric-value">{metrics.pca_distance_ratio:.2f}</span>
            <br><small>(Should be close to 1.0; <<1 suggests pool and reference are collapsing together)</small>
        </div>
        
        <h2>Warnings</h2>
        {''.join(f'<div class="warning">{w}</div>' for w in metrics.warnings) if metrics.warnings else '<p>No warnings</p>'}
        
        <h2>Processing Steps</h2>
        <ol>
            {''.join(f'<li>{step}</li>' for step in normalization_log)}
        </ol>
        
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    logger.info(f"QC report saved to {output_path}")
