"""Transition rollup module for aggregating transitions to peptides.

This module handles Stage 2 of the PRISM pipeline: rolling up individual
transition intensities to peptide-level quantities.

Key concepts:
- Default method is simple sum (fast, reliable baseline)
- Adaptive method learns optimal weights from reference samples
- Median polish provides robust aggregation with outlier handling
- When using median_polish, transition-level residuals are captured for outlier analysis

Adaptive rollup key insight: When all beta = 0, all weights = 1, giving simple sum.
This means optimization can only improve upon the sum baseline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .rollup import MedianPolishResult

logger = logging.getLogger(__name__)


@dataclass
class TransitionRollupResult:
    """Result of transition to peptide rollup.

    SCALE: peptide_abundances are on LOG2 SCALE when log_transform=True.
    To convert to linear scale: 2 ** peptide_abundances

    When using median_polish method, median_polish_results contains per-peptide
    MedianPolishResult objects, which include the residual matrix.
    Large residuals may indicate transitions with interference or biologically
    interesting variation.
    """

    peptide_abundances: pd.DataFrame  # Peptide x sample matrix (log2 if log_transform)
    peptide_uncertainties: pd.DataFrame  # Uncertainty estimates (log2 scale)
    transition_weights: pd.DataFrame  # Weights used per transition
    n_transitions_used: pd.DataFrame  # Number of transitions per peptide/sample
    # Median polish results per peptide (when method='median_polish')
    # Keys are peptide identifiers, values are MedianPolishResult objects
    median_polish_results: dict[str, MedianPolishResult] | None = None


# ============================================================================
# AdaptiveRollup: Learnable transition weights
# ============================================================================


@dataclass
class AdaptiveRollupParams:
    """Parameters for adaptive transition weighting.

    Weight function (in log space):
        log(w_t) = beta_sqrt_intensity * (sqrt(intensity) - sqrt_center) / sqrt_scale
                 + beta_mz * normalized_mz
                 + beta_shape_corr * median_shape_corr
                 + beta_shape_corr_outlier * low_fraction

        w = exp(log(w))

    Key insight: When all beta = 0, all weights are 1, giving simple sum (baseline).

    Features (4 optimized):
    - relative_intensity: Transition intensity / max intensity in the peptide [0, 1]
                         Computed per-peptide so the most intense transition = 1.0
    - mz: Product m/z, normalized to [0, 1] range
    - shape_corr: Median shape correlation across samples (0-1)
    - shape_corr_low_frac: Fraction of samples with shape correlation below threshold
                           (configurable via shape_corr_low_threshold, default 0.5)

    Not optimized (reserved for future use):
    - beta_shape_corr_max: Max shape correlation - kept at 0.0

    Constraints:
    - beta_relative_intensity >= 0 (higher intensity transitions should not be penalized)
    - Other betas are unconstrained (can be positive or negative)

    Note: beta_log_intensity and beta_sqrt_intensity are deprecated.
    """

    # DEPRECATED: beta_log_intensity is redundant - kept for backwards compatibility
    # Always set to 0.0 in new code
    beta_log_intensity: float = 0.0

    # DEPRECATED: beta_sqrt_intensity replaced by beta_relative_intensity
    # Kept for backwards compatibility, but NOT used if beta_relative_intensity is set
    beta_sqrt_intensity: float = 0.0

    # Coefficient for relative intensity (transition_mean / peptide_max)
    # Range [0, 1]: 1.0 = most intense transition in the peptide
    # >= 0 by constraint: higher relative intensity should not decrease weight
    # Default 0.0: no relative intensity weighting (baseline = simple sum)
    beta_relative_intensity: float = 0.0

    # Coefficient for normalized m/z (m/z scaled to [0, 1])
    # Default 0.0: no m/z weighting
    beta_mz: float = 0.0

    # Coefficient for median shape correlation (legacy name preserved for compatibility)
    # Default 0.0: no shape correlation weighting
    beta_shape_corr: float = 0.0

    # Coefficient for maximum shape correlation across samples
    # Default 0.0: no max shape correlation weighting
    beta_shape_corr_max: float = 0.0

    # Coefficient for low shape correlation fraction
    # Counts fraction of samples with shape_corr < shape_corr_low_threshold
    # Unconstrained: can be positive or negative
    # Negative values penalize transitions with more low-correlation samples
    # Default 0.0: no low-fraction weighting
    beta_shape_corr_outlier: float = 0.0

    # Feature normalization parameters (learned from data)
    mz_min: float = 0.0      # Minimum m/z for normalization
    mz_max: float = 2000.0   # Maximum m/z for normalization
    log_intensity_center: float = 15.0  # Centering value for log2(intensity)
    sqrt_intensity_center: float = 100.0  # Centering value for sqrt(intensity)
    sqrt_intensity_scale: float = 100.0   # Scale for sqrt(intensity) normalization

    # Fixed threshold for counting "low" shape correlations
    # Fraction of samples with shape_corr < this threshold is used as a feature
    # Default 0.5: transitions with many samples below 0.5 may have interference
    shape_corr_low_threshold: float = 0.5

    # Fallback settings
    fallback_to_sum: bool = True      # Fall back to sum if no improvement
    min_improvement_pct: float = 0.1  # Minimum CV improvement required


@dataclass
class AdaptiveRollupResult:
    """Result of learning adaptive rollup weights.

    Contains learned parameters, CV metrics for reference and QC samples,
    and decision on whether to use learned weights or fall back to sum.
    """

    params: AdaptiveRollupParams         # Learned (or default) parameters
    use_adaptive_weights: bool           # Whether to use learned weights
    reference_cv_sum: float              # CV with simple sum (baseline)
    reference_cv_adaptive: float         # CV with learned weights
    qc_cv_sum: float                     # QC CV with simple sum
    qc_cv_adaptive: float                # QC CV with learned weights
    improvement_pct: float               # Relative improvement on reference
    qc_improvement_pct: float            # Relative improvement on QC
    fallback_reason: str | None          # Reason for fallback if applicable


def compute_adaptive_weights(
    mean_log_intensity: np.ndarray,
    mz_values: np.ndarray,
    median_shape_corr: np.ndarray,
    params: AdaptiveRollupParams,
    relative_intensity: np.ndarray | None = None,
    max_shape_corr: np.ndarray | None = None,
    shape_corr_outlier_frac: np.ndarray | None = None,
) -> np.ndarray:
    """Compute adaptive transition weights from features.

    Weight function:
        log(w) = beta_log_intensity * (log_intensity - log_center)
               + beta_relative_intensity * relative_intensity
               + beta_mz * normalized_mz
               + beta_shape_corr * median_shape_corr
               + beta_shape_corr_max * max_shape_corr
               + beta_shape_corr_outlier * outlier_fraction

        w = exp(log(w))

    When all betas are 0, all weights are 1 (simple sum baseline).

    Args:
        mean_log_intensity: Mean log2 intensity per transition (n_transitions,)
        mz_values: Product m/z per transition (n_transitions,)
        median_shape_corr: Median shape correlation per transition (n_transitions,)
        params: Adaptive rollup parameters with beta coefficients
        relative_intensity: Relative intensity per transition [0, 1] where
                           1.0 = most intense transition in the peptide (optional)
        max_shape_corr: Maximum shape correlation per transition (optional)
        shape_corr_outlier_frac: Fraction of outlier samples per transition (optional)

    Returns:
        Weight per transition (n_transitions,), NOT normalized

    """
    n_trans = len(mean_log_intensity)

    # Normalize m/z to [0, 1] range
    mz_range = params.mz_max - params.mz_min
    if mz_range > 0:
        normalized_mz = (mz_values - params.mz_min) / mz_range
    else:
        normalized_mz = np.zeros_like(mz_values)
    normalized_mz = np.clip(normalized_mz, 0.0, 1.0)

    # Center log intensity (deprecated, beta should be 0)
    centered_log_intensity = mean_log_intensity - params.log_intensity_center

    # Relative intensity: already in [0, 1] range, no normalization needed
    # 1.0 = most intense transition in this peptide, 0 = no signal
    if relative_intensity is None:
        relative_intensity = np.ones(n_trans)  # Equal weights if not provided

    # Use provided max shape correlation or default to median
    if max_shape_corr is None:
        max_shape_corr = median_shape_corr

    # Use provided outlier fraction or default to zeros
    if shape_corr_outlier_frac is None:
        shape_corr_outlier_frac = np.zeros(n_trans)

    # Compute log-weight as linear combination of features
    # Note: beta_sqrt_intensity is deprecated, use beta_relative_intensity
    log_weight = (
        params.beta_log_intensity * centered_log_intensity
        + params.beta_relative_intensity * relative_intensity
        + params.beta_mz * normalized_mz
        + params.beta_shape_corr * median_shape_corr
        + params.beta_shape_corr_max * max_shape_corr
        + params.beta_shape_corr_outlier * shape_corr_outlier_frac
    )

    # Clamp log_weight to prevent overflow in exp()
    # exp(700) is close to float64 max, exp(-700) is close to 0
    log_weight = np.clip(log_weight, -50, 50)

    # Exponentiate to get weights (exp(0) = 1 when all betas = 0)
    weights = np.exp(log_weight)

    return weights


def rollup_peptide_adaptive(
    intensity_matrix: pd.DataFrame,
    mz_values: pd.Series,
    shape_corr_matrix: pd.DataFrame,
    params: AdaptiveRollupParams,
    min_transitions: int = 3,
) -> tuple[pd.Series, pd.Series, pd.Series, int]:
    """Roll up transitions to peptide using adaptive weights.

    Computes: Peptide_s = log2(sum_t w_t * 2^intensity_t,s)
    where w_t = exp(beta_log_intensity * log(I) + beta_mz * mz + beta_shape_corr * sc)

    Args:
        intensity_matrix: Transition x sample matrix (LOG2 scale)
        mz_values: Product m/z per transition (same index as intensity_matrix)
        shape_corr_matrix: Transition x sample shape correlation (0-1)
        params: Adaptive rollup parameters
        min_transitions: Minimum transitions required

    Returns:
        Tuple of (abundances, uncertainties, weights, n_transitions_used)

    """
    n_transitions = len(intensity_matrix)

    if n_transitions < min_transitions:
        return (
            pd.Series(np.nan, index=intensity_matrix.columns),
            pd.Series(np.nan, index=intensity_matrix.columns),
            pd.Series(dtype=float),
            0,
        )

    # Compute per-transition features (aggregated across samples)
    mean_log_intensity = intensity_matrix.mean(axis=1).values
    median_shape_corr = shape_corr_matrix.median(axis=1).values

    # Align m/z values with intensity_matrix index
    mz_aligned = mz_values.reindex(intensity_matrix.index).fillna(0).values

    # Compute weights
    weights = compute_adaptive_weights(
        mean_log_intensity, mz_aligned, median_shape_corr, params
    )

    # Normalize weights to sum to n_transitions (preserves sum magnitude)
    weight_sum = weights.sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        # Fallback to equal weights if sum is invalid
        normalized_weights = np.ones(n_transitions)
    else:
        normalized_weights = weights * (n_transitions / weight_sum)

    # Convert to linear space for aggregation
    linear_matrix = 2 ** intensity_matrix

    abundances = pd.Series(index=intensity_matrix.columns, dtype=float)
    uncertainties = pd.Series(index=intensity_matrix.columns, dtype=float)

    for sample in intensity_matrix.columns:
        linear_values = linear_matrix[sample].values
        valid = ~np.isnan(linear_values)

        if valid.sum() < min_transitions:
            abundances[sample] = np.nan
            uncertainties[sample] = np.nan
            continue

        valid_weights = normalized_weights[valid]
        valid_linear = linear_values[valid]

        # Weighted sum in linear space
        weighted_sum = (valid_weights * valid_linear).sum()

        # Convert back to log2
        abundances[sample] = np.log2(max(weighted_sum, 1.0))

        # Uncertainty: CV of weighted contributions
        contributions = valid_weights * valid_linear
        if len(contributions) > 1 and contributions.sum() > 0:
            uncertainties[sample] = np.std(contributions) / np.mean(contributions)
        else:
            uncertainties[sample] = np.nan

    # Return weights as Series with transition index
    weights_series = pd.Series(weights, index=intensity_matrix.index, name="weight")

    return abundances, uncertainties, weights_series, n_transitions


@dataclass
class _AdaptivePrecomputedPeptide:
    """Pre-computed data for a single peptide for fast adaptive weight learning."""

    # Intensity matrix (LINEAR scale): n_transitions x n_samples
    # Zeros are preserved as zeros (not clipped to 1)
    intensity_linear: np.ndarray
    # Mean log2 intensity per transition: n_transitions
    mean_log_intensity: np.ndarray
    # Relative intensity per transition: n_transitions
    # Computed as mean_linear / max(mean_linear) within each peptide
    # Range [0, 1] where 1.0 = most intense transition in this peptide
    relative_intensity: np.ndarray
    # Product m/z per transition: n_transitions
    mz_values: np.ndarray
    # Median shape correlation per transition: n_transitions
    median_shape_corr: np.ndarray
    # Maximum shape correlation per transition: n_transitions
    max_shape_corr: np.ndarray
    # Fraction of samples with outlier (low) shape correlation: n_transitions
    shape_corr_outlier_frac: np.ndarray
    # Number of transitions
    n_transitions: int
    # Sample names
    sample_names: list[str]


@dataclass
class _AdaptiveNormalizationParams:
    """Normalization parameters computed from data for adaptive weights."""

    mz_min: float
    mz_max: float
    log_intensity_center: float
    # Note: relative_intensity is computed per-peptide, so no global center/scale needed
    shape_corr_low_threshold: float  # Fixed threshold for "low" shape correlation


def _precompute_adaptive_metrics(
    data: pd.DataFrame,
    sample_list: list[str],
    peptide_col: str,
    transition_col: str,
    sample_col: str,
    abundance_col: str,
    mz_col: str,
    shape_corr_col: str,
    min_transitions: int = 3,
    exclude_precursor: bool = True,
    batch_col: str | None = "Batch",
    shape_corr_low_threshold: float = 0.5,
) -> tuple[dict[str, _AdaptivePrecomputedPeptide], _AdaptiveNormalizationParams]:
    """Pre-compute metrics for all peptides for adaptive weight learning.

    Returns pre-computed metrics plus feature normalization bounds.

    OPTIMIZED: Uses vectorized groupby operations instead of per-peptide loops.
    All metrics are computed in a single pass through the data.

    When batch_col is provided and samples appear in multiple batches,
    they are treated as separate replicates (e.g., Ref_029 in Plate1 and
    Ref_029 in Plate2 become two separate columns).

    Computes the following features per transition:
    - mean_log_intensity: Mean log2 intensity across samples
    - mean_sqrt_intensity: Mean sqrt intensity across samples
    - mz_values: Product m/z
    - median_shape_corr: Median shape correlation across samples
    - max_shape_corr: Maximum shape correlation across samples
    - shape_corr_low_frac: Fraction of samples with shape corr < threshold

    The fraction of "low" shape correlation samples is computed using a
    fixed threshold (shape_corr_low_threshold, default 0.5). Transitions
    where many samples have low shape correlation may have interference.

    Args:
        data: Transition-level DataFrame (LINEAR scale intensities)
        sample_list: Samples to include
        peptide_col: Column name for peptide identifier
        transition_col: Column name for transition identifier
        sample_col: Column name for sample identifier
        abundance_col: Column name for abundance values
        mz_col: Column name for product m/z
        shape_corr_col: Column name for shape correlation
        min_transitions: Minimum transitions required
        exclude_precursor: If True, filter out MS1 precursor ions
        batch_col: Column name for batch identifier (to handle duplicate names)
        shape_corr_low_threshold: Threshold for "low" shape correlation (default 0.5)

    Returns:
        Tuple of (precomputed_dict, normalization_params)

    """
    # Filter to specified samples
    filtered = data[data[sample_col].isin(sample_list)].copy()

    # Filter out MS1 precursor ions if requested (default behavior)
    if exclude_precursor and transition_col in filtered.columns:
        filtered = filtered[
            ~filtered[transition_col].astype(str).str.startswith('precursor')
        ]

    # Create composite sample identifier if batch column exists and there are duplicates
    if batch_col and batch_col in filtered.columns:
        # Check for duplicate sample names across batches
        sample_batch = filtered[[sample_col, batch_col]].drop_duplicates()
        dup_samples = sample_batch.groupby(sample_col).size()
        has_dups = (dup_samples > 1).any()

        if has_dups:
            # Create composite key: "sample::batch"
            filtered['_sample_batch'] = (
                filtered[sample_col].astype(str) + '::' + filtered[batch_col].astype(str)
            )
            composite_col = '_sample_batch'
            unique_sample_batches = filtered[composite_col].unique().tolist()
        else:
            composite_col = sample_col
            unique_sample_batches = list(dict.fromkeys(sample_list))
    else:
        composite_col = sample_col
        unique_sample_batches = list(dict.fromkeys(sample_list))

    # Use the fixed threshold for "low" shape correlation
    # Transitions with many samples below this threshold may have interference
    has_shape_corr = shape_corr_col in filtered.columns

    has_mz = mz_col in filtered.columns

    # ========================================================================
    # VECTORIZED APPROACH: Compute all per-transition metrics via groupby
    # ========================================================================

    # Create unique transition identifier including precursor and product charge
    # A transition is defined by precursor m/z -> product m/z combination
    # y19+2 from precursor 3+ and y19+2 from precursor 4+ are DIFFERENT transitions
    product_charge_col = 'Product Charge'
    precursor_charge_col = 'Precursor Charge'
    if product_charge_col in filtered.columns and precursor_charge_col in filtered.columns:
        # Full transition ID: FragmentIon_ProductCharge_PrecursorCharge
        filtered['_transition_id'] = (
            filtered[transition_col].astype(str) + '_' +
            filtered[product_charge_col].astype(str) + '_' +
            filtered[precursor_charge_col].astype(str)
        )
        trans_id_col = '_transition_id'
    elif product_charge_col in filtered.columns:
        filtered['_transition_id'] = (
            filtered[transition_col].astype(str) + '_' +
            filtered[product_charge_col].astype(str)
        )
        trans_id_col = '_transition_id'
    else:
        trans_id_col = transition_col

    # Precompute transformed values once
    # Store log2 of intensity (zeros become -inf, but that's ok for NaN detection)
    # We'll handle zeros properly in the weighted sum calculation
    intensity_vals = filtered[abundance_col].values.astype(float)
    # Clip to 1 for log2 calculation (for means/metrics), but mark zeros
    filtered['_intensity_is_zero'] = intensity_vals <= 0
    filtered['_log2_intensity'] = np.log2(np.maximum(intensity_vals, 1.0))
    filtered['_sqrt_intensity'] = np.sqrt(np.maximum(intensity_vals, 0.0))

    if has_shape_corr:
        filtered['_shape_low'] = (
            filtered[shape_corr_col].fillna(1.0) < shape_corr_low_threshold
        ).astype(float)

    # Group by peptide + unique transition (including charge) for per-transition statistics
    group_cols = [peptide_col, trans_id_col]

    # Build aggregation dict - compute all metrics in ONE groupby
    agg_dict = {
        '_log2_intensity': ['mean', 'first'],  # mean for metric, first for sample values
        '_sqrt_intensity': 'mean',
    }
    if has_mz:
        agg_dict[mz_col] = 'first'  # m/z is constant per transition
    if has_shape_corr:
        agg_dict[shape_corr_col] = ['median', 'max']
        agg_dict['_shape_low'] = 'mean'  # Mean of 0/1 = fraction below threshold

    # Aggregate per transition (across all samples)
    trans_stats = filtered.groupby(group_cols, sort=False).agg(agg_dict)
    # Flatten column names: ('_log2_intensity', 'mean') -> 'log2_intensity_mean'
    trans_stats.columns = ['_'.join(col).lstrip('_') for col in trans_stats.columns]
    trans_stats = trans_stats.reset_index()

    # Count transitions per peptide to filter by min_transitions
    trans_counts = trans_stats.groupby(peptide_col).size()
    valid_peptides = trans_counts[trans_counts >= min_transitions].index

    # Filter to valid peptides
    trans_stats = trans_stats[trans_stats[peptide_col].isin(valid_peptides)]

    # Compute normalization bounds from aggregated stats
    if has_mz:
        mz_col_agg = f'{mz_col}_first'
        valid_mz = trans_stats[mz_col_agg][trans_stats[mz_col_agg] > 0]
        mz_min = float(valid_mz.min()) if len(valid_mz) > 0 else 0.0
        mz_max = float(valid_mz.max()) if len(valid_mz) > 0 else 2000.0
    else:
        mz_min, mz_max = 0.0, 2000.0

    # Column names after flattening: 'log2_intensity_mean', 'sqrt_intensity_mean'
    log_int_col = 'log2_intensity_mean'
    valid_log_int = trans_stats[log_int_col].dropna()
    log_intensity_center = float(valid_log_int.median()) if len(valid_log_int) > 0 else 15.0

    # Note: relative_intensity is computed per-peptide below, no global params needed

    norm_params = _AdaptiveNormalizationParams(
        mz_min=mz_min,
        mz_max=mz_max,
        log_intensity_center=log_intensity_center,
        shape_corr_low_threshold=shape_corr_low_threshold,
    )

    # ========================================================================
    # Build per-peptide intensity matrices via pivot (needed for rollup)
    # This is the only part that needs per-peptide structure
    # ========================================================================

    # Pivot intensity to get peptide -> (transition x sample) matrices
    # Use multi-index pivot for efficiency
    # Store RAW linear intensity to avoid 2^log2(0-clipped) = 1 issue
    # Use trans_id_col (includes precursor and product charge) for unique transitions
    # Each (peptide, transition_id, sample) combination should now be unique
    intensity_pivot = filtered.pivot_table(
        index=[peptide_col, trans_id_col],
        columns=composite_col,
        values=abundance_col,  # Raw linear values, not log2
        aggfunc='first',
    )

    # Ensure all samples present
    for s in unique_sample_batches:
        if s not in intensity_pivot.columns:
            intensity_pivot[s] = np.nan
    intensity_pivot = intensity_pivot[unique_sample_batches]

    # ========================================================================
    # Build results dict using efficient groupby iteration
    # ========================================================================

    # Set index on trans_stats for fast lookup
    trans_stats_indexed = trans_stats.set_index(peptide_col)

    # Define column names for extraction
    shape_med_col = f'{shape_corr_col}_median' if has_shape_corr else None
    shape_max_col = f'{shape_corr_col}_max' if has_shape_corr else None
    shape_low_col = 'shape_low_mean' if has_shape_corr else None


    results = {}

    # Use groupby on trans_stats for efficient iteration
    for peptide in valid_peptides:
        try:
            # Fast index-based lookup
            pep_stats = trans_stats_indexed.loc[[peptide]]
        except KeyError:
            continue

        n_trans = len(pep_stats)

        # Get intensity matrix for this peptide (LINEAR scale)
        try:
            intensity_linear = intensity_pivot.loc[peptide].values.astype(float)
        except KeyError:
            continue

        # Handle case where only one transition (returns Series not DataFrame)
        if intensity_linear.ndim == 1:
            intensity_linear = intensity_linear.reshape(1, -1)

        # Extract pre-computed metrics (using .values for speed)
        mean_log_intensity = pep_stats[log_int_col].values

        # Compute RELATIVE intensity within this peptide:
        # relative_intensity = mean_linear / max(mean_linear)
        # Range [0, 1] where 1.0 = most intense transition in this peptide
        mean_linear = np.nanmean(intensity_linear, axis=1)  # mean across samples
        max_mean_linear = np.nanmax(mean_linear)
        if max_mean_linear > 0:
            relative_intensity = mean_linear / max_mean_linear
        else:
            relative_intensity = np.ones(n_trans)  # All zeros -> equal weights

        if has_mz:
            mz_values = pep_stats[mz_col_agg].fillna(0).values
        else:
            mz_values = np.zeros(n_trans)

        if has_shape_corr:
            median_shape_corr = pep_stats[shape_med_col].fillna(1.0).values
            max_shape_corr = pep_stats[shape_max_col].fillna(1.0).values
            shape_corr_low_frac = pep_stats[shape_low_col].fillna(0.0).values
        else:
            median_shape_corr = np.ones(n_trans)
            max_shape_corr = np.ones(n_trans)
            shape_corr_low_frac = np.zeros(n_trans)

        results[peptide] = _AdaptivePrecomputedPeptide(
            intensity_linear=intensity_linear,
            mean_log_intensity=mean_log_intensity,
            relative_intensity=relative_intensity,
            mz_values=mz_values,
            median_shape_corr=median_shape_corr,
            max_shape_corr=max_shape_corr,
            shape_corr_outlier_frac=shape_corr_low_frac,
            n_transitions=n_trans,
            sample_names=unique_sample_batches,
        )

    return results, norm_params


def _rollup_with_adaptive_params(
    precomputed: dict[str, _AdaptivePrecomputedPeptide],
    params: AdaptiveRollupParams,
    min_transitions: int = 3,
) -> pd.DataFrame:
    """Fast rollup using pre-computed metrics and adaptive parameters.

    Args:
        precomputed: Pre-computed metrics for each peptide
        params: Adaptive rollup parameters
        min_transitions: Minimum transitions required

    Returns:
        Peptide x sample matrix of abundances (LOG2 scale)

    """
    if not precomputed:
        return pd.DataFrame()

    # Get sample names from first peptide
    first = next(iter(precomputed.values()))
    sample_names = first.sample_names

    results = {}

    for peptide, metrics in precomputed.items():
        if metrics.n_transitions < min_transitions:
            continue

        # Compute weights with all features
        weights = compute_adaptive_weights(
            metrics.mean_log_intensity,
            metrics.mz_values,
            metrics.median_shape_corr,
            params,
            relative_intensity=metrics.relative_intensity,
            max_shape_corr=metrics.max_shape_corr,
            shape_corr_outlier_frac=metrics.shape_corr_outlier_frac,
        )

        # Normalize weights
        weight_sum = weights.sum()
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            # Fallback to equal weights if sum is invalid
            normalized_weights = np.ones(metrics.n_transitions)
        else:
            normalized_weights = weights * (metrics.n_transitions / weight_sum)

        # Weighted sum in linear space for each sample
        # Use raw linear values (zeros stay zero, no 2^0=1 issue)
        linear_matrix = metrics.intensity_linear
        abundances = np.zeros(len(sample_names))

        for i in range(len(sample_names)):
            col = linear_matrix[:, i]
            # Valid means not NaN and not negative
            valid = np.isfinite(col) & (col >= 0)
            if valid.sum() >= min_transitions:
                weighted_sum = (normalized_weights[valid] * col[valid]).sum()
                abundances[i] = np.log2(max(weighted_sum, 1.0))
            else:
                abundances[i] = np.nan

        results[peptide] = abundances

    return pd.DataFrame.from_dict(results, orient='index', columns=sample_names)


def _compute_median_cv_for_adaptive(abundances: pd.DataFrame) -> float:
    """Compute median CV across peptides on LINEAR scale.

    Args:
        abundances: Peptide x sample matrix (LOG2 scale)

    Returns:
        Median CV as a decimal (not percentage)

    """
    if abundances.empty:
        return np.nan

    # Convert to linear scale
    linear = 2 ** abundances

    # Calculate CV per peptide (across samples)
    means = linear.mean(axis=1)
    stds = linear.std(axis=1)

    # Filter out peptides with near-zero mean
    valid = means > 1.0
    if valid.sum() == 0:
        return np.nan

    cvs = stds[valid] / means[valid]
    return float(cvs.median())


def _rollup_all_peptides_sum_for_adaptive(
    data: pd.DataFrame,
    sample_list: list[str],
    peptide_col: str,
    transition_col: str,
    sample_col: str,
    abundance_col: str,
    min_transitions: int = 3,
    exclude_precursor: bool = True,
    batch_col: str | None = "Batch",
) -> pd.DataFrame:
    """Roll up all peptides using simple sum (for baseline comparison).

    This function mirrors the main pipeline's behavior: for each peptide,
    it pivots to transition × sample (taking first value for duplicates),
    then sums transitions to get peptide abundance.

    When batch_col is provided and samples appear in multiple batches,
    they are treated as separate replicates (e.g., Ref_029 in Plate1 and
    Ref_029 in Plate2 become two separate columns).

    Args:
        data: Transition-level DataFrame (LINEAR scale intensities)
        sample_list: List of samples to include
        peptide_col: Column name for peptide identifier
        transition_col: Column name for transition identifier
        sample_col: Column name for sample identifier
        abundance_col: Column name for abundance values
        min_transitions: Minimum transitions per peptide
        exclude_precursor: If True, filter out MS1 precursor ions
        batch_col: Column name for batch identifier (to handle duplicate names)

    Returns:
        Peptide x sample matrix of abundances (LOG2 scale)

    """
    # Filter to specified samples
    filtered_data = data[data[sample_col].isin(sample_list)].copy()

    # Filter out MS1 precursor ions if requested (default behavior)
    if exclude_precursor and transition_col in filtered_data.columns:
        filtered_data = filtered_data[
            ~filtered_data[transition_col].astype(str).str.startswith('precursor')
        ]

    # Create composite sample identifier if batch column exists and there are duplicates
    if batch_col and batch_col in filtered_data.columns:
        # Check for duplicate sample names across batches
        sample_batch = filtered_data[[sample_col, batch_col]].drop_duplicates()
        dup_samples = sample_batch.groupby(sample_col).size()
        has_dups = (dup_samples > 1).any()

        if has_dups:
            # Create composite key: "sample::batch"
            filtered_data['_sample_batch'] = (
                filtered_data[sample_col] + '::' + filtered_data[batch_col].astype(str)
            )
            composite_col = '_sample_batch'
            # Get unique sample-batch combinations for the requested samples
            unique_sample_batches = filtered_data[composite_col].unique().tolist()
        else:
            composite_col = sample_col
            unique_sample_batches = list(dict.fromkeys(sample_list))
    else:
        composite_col = sample_col
        unique_sample_batches = list(dict.fromkeys(sample_list))

    # Vectorized: sum and count transitions per peptide x sample using groupby
    # This replaces the per-peptide loop with a single groupby operation
    grouped = filtered_data.groupby([peptide_col, composite_col])[abundance_col].agg(['sum', 'count'])
    grouped = grouped.reset_index()
    grouped.columns = [peptide_col, 'sample', 'abundance_sum', 'n_transitions']

    # Apply minimum transitions filter
    grouped.loc[grouped['n_transitions'] < min_transitions, 'abundance_sum'] = np.nan

    # Pivot to peptide x sample matrix
    peptide_abundances = grouped.pivot(
        index=peptide_col,
        columns='sample',
        values='abundance_sum'
    )

    # Ensure all samples are present in correct order
    for sample in unique_sample_batches:
        if sample not in peptide_abundances.columns:
            peptide_abundances[sample] = np.nan
    peptide_abundances = peptide_abundances[unique_sample_batches]

    # Convert to log2 scale
    peptide_abundances = np.log2(peptide_abundances.clip(lower=1))

    return peptide_abundances


def learn_adaptive_weights(
    data: pd.DataFrame,
    reference_samples: list[str],
    qc_samples: list[str],
    peptide_col: str = "peptide_modified",
    transition_col: str = "fragment_ion",
    sample_col: str = "replicate_name",
    abundance_col: str = "area",
    mz_col: str = "Product Mz",
    shape_corr_col: str = "Shape Correlation",
    n_iterations: int = 500,
    initial_params: AdaptiveRollupParams | None = None,
) -> AdaptiveRollupResult:
    """Learn adaptive rollup weights from reference samples.

    Optimizes beta coefficients to minimize median CV on reference samples.
    Validates on QC samples to ensure generalization.

    Loss function: median(CV_p) over all peptides p in reference samples

    Key insight: When all betas = 0, all weights = 1, giving simple sum baseline.

    Features optimized (5 total):
    - beta_sqrt_intensity: Weight for sqrt(intensity) (>= 0)
    - beta_mz: Weight for normalized m/z (any)
    - beta_shape_corr: Weight for median shape correlation (any)
    - beta_shape_corr_max: Weight for max shape correlation (any)
    - beta_shape_corr_outlier: Weight for outlier fraction (any)

    Note: beta_log_intensity is deprecated and NOT optimized.

    Args:
        data: Transition-level DataFrame (LINEAR scale intensities)
        reference_samples: List of reference sample names (for learning)
        qc_samples: List of QC sample names (for validation)
        peptide_col: Column name for peptide identifier
        transition_col: Column name for transition identifier
        sample_col: Column name for sample identifier
        abundance_col: Column name for abundance values
        mz_col: Column with product m/z values
        shape_corr_col: Column with shape correlation values
        n_iterations: Maximum optimization iterations
        initial_params: Starting parameters (uses zeros/defaults if None)

    Returns:
        AdaptiveRollupResult with learned params and validation metrics

    """
    from scipy.optimize import minimize

    # Note: Caller (cli.py) logs the learning header; we just log progress

    if initial_params is None:
        initial_params = AdaptiveRollupParams()

    if len(reference_samples) < 2:
        logger.warning(
            f"Need at least 2 reference samples, got {len(reference_samples)}. "
            "Using default parameters (simple sum)."
        )
        return AdaptiveRollupResult(
            params=initial_params,
            use_adaptive_weights=False,
            reference_cv_sum=np.nan,
            reference_cv_adaptive=np.nan,
            qc_cv_sum=np.nan,
            qc_cv_adaptive=np.nan,
            improvement_pct=0.0,
            qc_improvement_pct=0.0,
            fallback_reason="Insufficient reference samples",
        )

    # Compute baseline CV using sum method
    logger.info("  Computing baseline CV (sum method)...")
    ref_abundances_sum = _rollup_all_peptides_sum_for_adaptive(
        data, reference_samples, peptide_col, transition_col, sample_col, abundance_col
    )
    reference_cv_sum = _compute_median_cv_for_adaptive(ref_abundances_sum)
    logger.info(f"  Reference CV (sum): {reference_cv_sum:.4f}")

    qc_cv_sum = np.nan
    if len(qc_samples) >= 2:
        qc_abundances_sum = _rollup_all_peptides_sum_for_adaptive(
            data, qc_samples, peptide_col, transition_col, sample_col, abundance_col
        )
        qc_cv_sum = _compute_median_cv_for_adaptive(qc_abundances_sum)
        logger.info(f"  QC CV (sum): {qc_cv_sum:.4f}")

    # Pre-compute metrics for reference samples
    logger.info("  Pre-computing metrics for reference samples...")
    ref_metrics, norm_params = _precompute_adaptive_metrics(
        data, reference_samples, peptide_col, transition_col,
        sample_col, abundance_col, mz_col, shape_corr_col
    )
    logger.info(f"  Pre-computed {len(ref_metrics)} peptides")
    logger.info(f"  m/z range: [{norm_params.mz_min:.1f}, {norm_params.mz_max:.1f}]")
    logger.info(f"  Log2 intensity center: {norm_params.log_intensity_center:.2f}")
    logger.info("  Relative intensity: computed per-peptide [0, 1]")
    logger.info(f"  Shape corr low threshold: {norm_params.shape_corr_low_threshold:.3f}")

    # Track optimization - keep the BEST parameters found, not just where optimizer stops
    iteration_count = [0]
    best_cv = [reference_cv_sum]
    best_params = [np.array([0.0, 0.0, 0.0])]  # Start with zeros (sum baseline)

    def objective(beta_array):
        """Objective: minimize median CV on reference samples."""
        # Optimizing 3 parameters:
        # [0] beta_relative_intensity >= 0
        # [1] beta_mz (unconstrained)
        # [2] beta_shape_corr_outlier (penalize high outlier fraction)
        # Note: beta_shape_corr (median) is NOT optimized - fixed at 0.0
        params = AdaptiveRollupParams(
            beta_log_intensity=0.0,  # Deprecated, not optimized
            beta_sqrt_intensity=0.0,  # Deprecated, not optimized
            beta_relative_intensity=max(0.0, beta_array[0]),
            beta_mz=beta_array[1],
            beta_shape_corr=0.0,  # Not optimized - fixed at 0.0
            beta_shape_corr_max=0.0,  # Not optimized - kept for future use
            beta_shape_corr_outlier=beta_array[2],
            mz_min=norm_params.mz_min,
            mz_max=norm_params.mz_max,
            log_intensity_center=norm_params.log_intensity_center,
            shape_corr_low_threshold=norm_params.shape_corr_low_threshold,
        )

        try:
            abundances = _rollup_with_adaptive_params(ref_metrics, params)
            cv = _compute_median_cv_for_adaptive(abundances)
            iteration_count[0] += 1
            # Track the BEST parameters found during optimization
            if np.isfinite(cv) and cv < best_cv[0]:
                best_cv[0] = cv
                best_params[0] = np.array(beta_array).copy()
            return cv if np.isfinite(cv) else 1.0
        except Exception:
            return 1.0

    # Initial parameters: all zeros (simple sum baseline)
    # Note: beta_log_intensity, beta_shape_corr, and beta_shape_corr_max are not optimized
    x0 = [
        initial_params.beta_relative_intensity,
        initial_params.beta_mz,
        initial_params.beta_shape_corr_outlier,
    ]

    # Bounds for 3 parameters
    bounds = [
        (0.0, 1.0),     # beta_relative_intensity (must be non-negative)
        (-1.0, 1.0),    # beta_mz (unconstrained)
        (-1.0, 0.0),    # beta_shape_corr_outlier (penalize high outlier fraction)
    ]

    logger.info("  Optimizing 3 beta coefficients...")
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": n_iterations, "ftol": 1e-6},
    )
    
    # Log convergence status
    if result.success:
        logger.info(f"  Optimization CONVERGED in {iteration_count[0]} evaluations")
    else:
        logger.warning(f"  Optimization did NOT converge: {result.message}")
        logger.warning(f"    ({iteration_count[0]} evaluations, maxiter={n_iterations})")

    # Use BEST parameters found during optimization, not where optimizer stopped
    # This guarantees we never return worse than the starting point (zeros = sum)
    opt = best_params[0]
    logger.info(f"  Best CV found: {best_cv[0]:.4f} (baseline: {reference_cv_sum:.4f})")

    learned_params = AdaptiveRollupParams(
        beta_log_intensity=0.0,  # Deprecated
        beta_sqrt_intensity=0.0,  # Deprecated
        beta_relative_intensity=max(0.0, opt[0]),
        beta_mz=opt[1],
        beta_shape_corr=0.0,  # Not optimized - fixed at 0.0
        beta_shape_corr_max=0.0,  # Not optimized
        beta_shape_corr_outlier=opt[2],
        mz_min=norm_params.mz_min,
        mz_max=norm_params.mz_max,
        log_intensity_center=norm_params.log_intensity_center,
        shape_corr_low_threshold=norm_params.shape_corr_low_threshold,
        fallback_to_sum=initial_params.fallback_to_sum,
        min_improvement_pct=initial_params.min_improvement_pct,
    )

    # Compute final CV on reference (should match best_cv[0])
    ref_abundances_adaptive = _rollup_with_adaptive_params(ref_metrics, learned_params)
    reference_cv_adaptive = _compute_median_cv_for_adaptive(ref_abundances_adaptive)

    logger.info("  Learned parameters (3 features):")
    logger.info(f"    beta_relative_intensity: {learned_params.beta_relative_intensity:.4f}")
    logger.info(f"    beta_mz: {learned_params.beta_mz:.4f}")
    logger.info(f"    beta_shape_corr_outlier: {learned_params.beta_shape_corr_outlier:.4f}")
    logger.info(f"  Reference CV: {reference_cv_sum:.4f} -> {reference_cv_adaptive:.4f}")

    # Calculate improvement on reference
    if reference_cv_sum > 0:
        improvement_pct = (reference_cv_sum - reference_cv_adaptive) / reference_cv_sum * 100
    else:
        improvement_pct = 0.0
    logger.info(f"  Reference improvement: {improvement_pct:.1f}%")

    # Validate on QC samples
    qc_cv_adaptive = np.nan
    qc_improvement_pct = 0.0
    use_adaptive_weights = True
    fallback_reason = None

    if len(qc_samples) >= 2:
        # Pre-compute metrics for QC samples
        qc_metrics, _ = _precompute_adaptive_metrics(
            data, qc_samples, peptide_col, transition_col,
            sample_col, abundance_col, mz_col, shape_corr_col
        )

        # Use learned params with reference normalization
        qc_abundances_adaptive = _rollup_with_adaptive_params(qc_metrics, learned_params)
        qc_cv_adaptive = _compute_median_cv_for_adaptive(qc_abundances_adaptive)

        logger.info(f"  QC CV: {qc_cv_sum:.4f} -> {qc_cv_adaptive:.4f}")

        if qc_cv_sum > 0:
            qc_improvement_pct = (qc_cv_sum - qc_cv_adaptive) / qc_cv_sum * 100
            logger.info(f"  QC improvement: {qc_improvement_pct:.1f}%")

        # Decision: use adaptive weights or fall back to sum?
        # Require improvement on BOTH reference and QC
        if improvement_pct < initial_params.min_improvement_pct:
            use_adaptive_weights = False
            fallback_reason = (
                f"Reference improvement ({improvement_pct:.1f}%) below threshold "
                f"({initial_params.min_improvement_pct}%)"
            )
        elif qc_cv_adaptive > qc_cv_sum * 1.05:  # QC got worse by more than 5%
            use_adaptive_weights = False
            fallback_reason = (
                f"QC CV increased from {qc_cv_sum:.4f} to {qc_cv_adaptive:.4f}"
            )
    else:
        logger.warning("  Not enough QC samples for validation")
        # Without QC validation, require higher improvement threshold
        if improvement_pct < initial_params.min_improvement_pct:
            use_adaptive_weights = False
            fallback_reason = (
                f"Reference improvement ({improvement_pct:.1f}%) below threshold "
                f"({initial_params.min_improvement_pct}%) and no QC validation"
            )

    if use_adaptive_weights:
        logger.info("  Using adaptive weights")
    else:
        logger.warning(f"  Falling back to sum: {fallback_reason}")

    return AdaptiveRollupResult(
        params=learned_params,
        use_adaptive_weights=use_adaptive_weights,
        reference_cv_sum=reference_cv_sum,
        reference_cv_adaptive=reference_cv_adaptive,
        qc_cv_sum=qc_cv_sum,
        qc_cv_adaptive=qc_cv_adaptive,
        improvement_pct=improvement_pct,
        qc_improvement_pct=qc_improvement_pct,
        fallback_reason=fallback_reason,
    )


def rollup_peptide_topn(
    intensity_matrix: pd.DataFrame,
    shape_corr_matrix: pd.DataFrame,
    n_transitions: int = 3,
    selection_method: str = "correlation",
    weighting: str = "sum",
    min_transitions: int = 3,
) -> tuple[pd.Series, pd.Series, pd.Series, int]:
    """Roll up transitions using Top-N selection.

    Selects the same N transitions for ALL replicates based on either:
    - correlation: transitions with highest median shape correlation
    - intensity: transitions with highest mean intensity

    Then aggregates using either:
    - sum: simple sum of selected transitions
    - sqrt: sqrt(intensity)-weighted sum

    Args:
        intensity_matrix: Transition x sample matrix (LOG2 scale)
        shape_corr_matrix: Transition x sample shape correlation (0-1)
        n_transitions: Number of transitions to select (default: 3)
        selection_method: How to select transitions - "correlation" or "intensity"
        weighting: How to weight selected transitions - "sum" or "sqrt"
        min_transitions: Minimum transitions required

    Returns:
        Tuple of (abundances, uncertainties, weights, n_transitions_used)
    """
    n_available = len(intensity_matrix)

    if n_available < min_transitions:
        return (
            pd.Series(np.nan, index=intensity_matrix.columns),
            pd.Series(np.nan, index=intensity_matrix.columns),
            pd.Series(dtype=float),
            0,
        )

    # Convert to linear for intensity calculations
    linear_intensity = 2 ** intensity_matrix

    # Step 1: Score and rank transitions
    if selection_method == "correlation":
        # Score by median shape correlation across samples
        scores = shape_corr_matrix.median(axis=1)
    else:  # intensity
        # Score by mean intensity across samples
        scores = linear_intensity.mean(axis=1)

    # Step 2: Select top N transitions (same for ALL samples)
    n_select = min(n_transitions, n_available)
    selected_transitions = scores.nlargest(n_select).index.tolist()

    # Subset to selected transitions only
    selected_intensity = intensity_matrix.loc[selected_transitions]
    selected_linear = linear_intensity.loc[selected_transitions]

    # Step 3: Compute weights
    if weighting == "sum":
        # Equal weights = simple sum
        weights = pd.Series(1.0, index=selected_transitions)
    else:  # sqrt
        # Weight by sqrt of mean intensity
        mean_intensity = selected_linear.mean(axis=1)
        weights = np.sqrt(np.maximum(mean_intensity.values, 1.0))
        weights = pd.Series(weights, index=selected_transitions)

    # Step 4: Aggregate - weighted sum in linear space
    # Normalize weights to sum to n_select (preserve sum magnitude)
    weight_sum = weights.sum()
    if np.isfinite(weight_sum) and weight_sum > 0:
        normalized_weights = weights * (n_select / weight_sum)
    else:
        normalized_weights = pd.Series(1.0, index=selected_transitions)

    abundances = pd.Series(index=intensity_matrix.columns, dtype=float)
    uncertainties = pd.Series(index=intensity_matrix.columns, dtype=float)

    for sample in intensity_matrix.columns:
        linear_values = selected_linear[sample]
        valid = ~linear_values.isna()

        if valid.sum() < min_transitions:
            abundances[sample] = np.nan
            uncertainties[sample] = np.nan
            continue

        valid_weights = normalized_weights[valid]
        valid_linear = linear_values[valid]

        # Weighted sum in linear space
        weighted_sum = (valid_weights * valid_linear).sum()

        # Convert back to log2
        abundances[sample] = np.log2(max(weighted_sum, 1.0))

        # Uncertainty: CV of contributions
        contributions = valid_weights * valid_linear
        if len(contributions) > 1 and contributions.sum() > 0:
            uncertainties[sample] = contributions.std() / contributions.mean()
        else:
            uncertainties[sample] = np.nan

    # Return full weight vector (zeros for non-selected)
    full_weights = pd.Series(0.0, index=intensity_matrix.index)
    full_weights.loc[selected_transitions] = weights.values

    return abundances, uncertainties, full_weights, n_select


def rollup_transitions_to_peptides(
    data: pd.DataFrame,
    peptide_col: str = "peptide_modified",
    transition_col: str = "fragment_ion",
    sample_col: str = "replicate_name",
    abundance_col: str = "area",
    shape_corr_col: str = "shape_correlation",
    mz_col: str = "Product Mz",
    method: str = "sum",
    adaptive_params: AdaptiveRollupParams | None = None,
    min_transitions: int = 3,
    log_transform: bool = True,
) -> TransitionRollupResult:
    """Roll up transition-level data to peptide-level quantities.

    SCALE CONVENTIONS:
    - Input (abundance_col): LINEAR scale (raw areas from Skyline)
    - Internal processing: LOG2 scale (if log_transform=True)
    - Output (peptide_abundances): LOG2 scale (if log_transform=True)

    To get linear-scale output: 2 ** result.peptide_abundances

    Args:
        data: DataFrame with transition-level Skyline data (LINEAR scale)
        peptide_col: Column identifying peptides
        transition_col: Column identifying transitions
        sample_col: Column identifying samples/replicates
        abundance_col: Column with transition intensities (LINEAR scale)
        shape_corr_col: Column with shape correlation values
        mz_col: Column with product m/z values (for adaptive method)
        method: Rollup method ('sum', 'median_polish', 'adaptive', 'topn')
        adaptive_params: AdaptiveRollupParams for adaptive method
        min_transitions: Minimum transitions required per peptide
        log_transform: Whether to log2 transform intensities (default: True)

    Returns:
        TransitionRollupResult with peptide abundances (LOG2 scale if log_transform)

    """
    logger.info(f"Rolling up transitions to peptides using method: {method}")

    # Get unique peptides and samples
    peptides = data[peptide_col].unique()
    samples = data[sample_col].unique()

    logger.info(f"  {len(peptides)} peptides, {len(samples)} samples")

    # Initialize output matrices
    peptide_abundances = pd.DataFrame(index=peptides, columns=samples, dtype=float)
    peptide_uncertainties = pd.DataFrame(index=peptides, columns=samples, dtype=float)
    n_transitions_used = pd.DataFrame(index=peptides, columns=samples, dtype=int)
    all_weights = {}
    all_median_polish_results = {}  # Store median polish results for residual output

    for peptide in peptides:
        pep_data = data[data[peptide_col] == peptide]

        # Pivot to get transition × sample matrices
        intensity_matrix = pep_data.pivot_table(
            index=transition_col, columns=sample_col, values=abundance_col, aggfunc="first"
        )

        # Fill missing samples with NaN
        for sample in samples:
            if sample not in intensity_matrix.columns:
                intensity_matrix[sample] = np.nan
        intensity_matrix = intensity_matrix[samples]  # Reorder

        # Log transform if needed
        if log_transform:
            intensity_matrix = np.log2(intensity_matrix.clip(lower=1))

        if method == "median_polish":
            # Use median polish for robust aggregation
            from .rollup import tukey_median_polish

            if len(intensity_matrix) >= min_transitions:
                result = tukey_median_polish(intensity_matrix)
                abundances = result.col_effects
                # Uncertainty from residual variance
                residual_var = result.residuals.var()
                uncertainties = pd.Series(
                    np.sqrt(residual_var.mean()), index=intensity_matrix.columns
                )
                n_used = len(intensity_matrix)
                # Store the full result for residual analysis
                all_median_polish_results[peptide] = result
            else:
                abundances = pd.Series(np.nan, index=samples)
                uncertainties = pd.Series(np.nan, index=samples)
                n_used = 0

        elif method == "sum":
            # Simple sum (convert to linear, sum, back to log)
            linear = 2**intensity_matrix if log_transform else intensity_matrix
            summed = linear.sum(axis=0)
            abundances = np.log2(summed.clip(lower=1)) if log_transform else summed
            uncertainties = pd.Series(np.nan, index=samples)
            n_used = (~intensity_matrix.isna()).sum().min()

        elif method == "adaptive":
            # Adaptive weights using learned beta coefficients
            if adaptive_params is None:
                adaptive_params = AdaptiveRollupParams()

            # Get shape correlation matrix
            if shape_corr_col in pep_data.columns:
                shape_corr_matrix = pep_data.pivot_table(
                    index=transition_col,
                    columns=sample_col,
                    values=shape_corr_col,
                    aggfunc="first",
                )
                for sample in samples:
                    if sample not in shape_corr_matrix.columns:
                        shape_corr_matrix[sample] = 1.0
                shape_corr_matrix = shape_corr_matrix[samples].fillna(1.0)
            else:
                shape_corr_matrix = pd.DataFrame(
                    1.0, index=intensity_matrix.index, columns=intensity_matrix.columns
                )

            # Get m/z values per transition
            if mz_col in pep_data.columns:
                mz_pivot = pep_data.pivot_table(
                    index=transition_col,
                    columns=sample_col,
                    values=mz_col,
                    aggfunc="first",
                )
                mz_values = mz_pivot.apply(
                    lambda x: x.dropna().iloc[0] if x.notna().any() else 0.0, axis=1
                )
            else:
                mz_values = pd.Series(0.0, index=intensity_matrix.index)

            # Roll up using adaptive weights
            abundances, uncertainties, weights, n_used = rollup_peptide_adaptive(
                intensity_matrix, mz_values, shape_corr_matrix,
                adaptive_params, min_transitions
            )
            all_weights[peptide] = weights

        else:
            raise ValueError(f"Unknown rollup method: {method}")

        peptide_abundances.loc[peptide] = abundances
        peptide_uncertainties.loc[peptide] = uncertainties
        n_transitions_used.loc[peptide] = n_used

    # Compile weights into DataFrame
    if all_weights:
        weights_df = pd.DataFrame(all_weights).T
    else:
        weights_df = pd.DataFrame()

    logger.info(
        f"  Rolled up to {(~peptide_abundances.isna().all(axis=1)).sum()} peptides with data"
    )

    return TransitionRollupResult(
        peptide_abundances=peptide_abundances,
        peptide_uncertainties=peptide_uncertainties,
        transition_weights=weights_df,
        n_transitions_used=n_transitions_used,
        median_polish_results=all_median_polish_results if all_median_polish_results else None,
    )

