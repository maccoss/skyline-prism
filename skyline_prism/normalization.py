"""Normalization module with RT-aware correction."""

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import interpolate

from .batch_correction import combat_with_reference_samples

logger = logging.getLogger(__name__)


@dataclass
class RTCorrectionResult:
    """Result of RT-aware correction."""

    corrected_data: pd.DataFrame
    correction_factors: pd.DataFrame   # RT × sample correction values
    rt_model: dict                     # Model parameters per sample
    reference_stats: pd.DataFrame      # Per-peptide stats from reference


@dataclass
class NormalizationResult:
    """Result of full normalization pipeline."""

    normalized_data: pd.DataFrame
    rt_correction: Optional[RTCorrectionResult]
    global_factors: pd.Series          # Per-sample normalization factors
    method_log: list[str] = field(default_factory=list)


def compute_reference_statistics(
    data: pd.DataFrame,
    reference_mask: pd.Series,
    precursor_col: str = 'precursor_id',
    abundance_col: str = 'abundance',
    rt_col: str = 'retention_time',
    replicate_col: str = 'replicate_name',
) -> pd.DataFrame:
    """Compute per-peptide statistics from reference replicates.

    Args:
        data: DataFrame with peptide data
        reference_mask: Boolean mask for reference samples
        precursor_col: Column with precursor identifiers
        abundance_col: Column with abundance values
        rt_col: Column with retention times
        replicate_col: Column with replicate names

    Returns:
        DataFrame with per-precursor statistics:
        - mean_abundance
        - cv (coefficient of variation)
        - median_rt
        - n_observations

    """
    ref_data = data.loc[reference_mask]

    stats = ref_data.groupby(precursor_col).agg({
        abundance_col: ['mean', 'std', 'count'],
        rt_col: 'median',
    })

    # Flatten column names
    stats.columns = ['mean_abundance', 'std_abundance', 'n_observations', 'median_rt']

    # Calculate CV
    stats['cv'] = stats['std_abundance'] / stats['mean_abundance']

    # Replace inf/nan CV with 0
    stats['cv'] = stats['cv'].replace([np.inf, -np.inf], np.nan).fillna(0)

    logger.info(f"Computed statistics for {len(stats)} precursors from reference")
    logger.info(f"Median CV across reference: {stats['cv'].median():.3f}")

    return stats


def fit_rt_spline(
    rt: np.ndarray,
    residuals: np.ndarray,
    df: int = 5,
    smooth: Optional[float] = None,
) -> Callable:
    """Fit a smoothing spline to RT vs residuals.

    Args:
        rt: Retention times
        residuals: Abundance residuals (observed - expected)
        df: Degrees of freedom for spline
        smooth: Smoothing parameter (if None, use df)

    Returns:
        Callable that predicts residual for a given RT

    """
    # Remove NaN values
    valid = ~(np.isnan(rt) | np.isnan(residuals))
    rt_valid = rt[valid]
    res_valid = residuals[valid]

    if len(rt_valid) < df + 1:
        # Not enough points, return zero function
        logger.warning(f"Only {len(rt_valid)} valid points, returning zero correction")
        return lambda x: np.zeros_like(x)

    # Sort by RT
    sort_idx = np.argsort(rt_valid)
    rt_sorted = rt_valid[sort_idx]
    res_sorted = res_valid[sort_idx]

    # Fit smoothing spline
    try:
        if smooth is not None:
            spline = interpolate.UnivariateSpline(rt_sorted, res_sorted, s=smooth)
        else:
            # Use degrees of freedom to set smoothing
            # Higher df = more flexible, less smoothing
            spline = interpolate.UnivariateSpline(rt_sorted, res_sorted, k=3)
            # Adjust smoothing to achieve approximately df effective parameters
            spline.set_smoothing_factor(len(rt_sorted) - df)
    except Exception as e:
        logger.warning(f"Spline fitting failed: {e}, using LOESS instead")
        # Fallback to simple binned median
        return _fit_binned_correction(rt_sorted, res_sorted)

    return spline


def _fit_binned_correction(
    rt: np.ndarray,
    residuals: np.ndarray,
    n_bins: int = 20,
) -> Callable:
    """Fallback: fit binned median correction."""
    bins = np.linspace(rt.min(), rt.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_medians = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (rt >= bins[i]) & (rt < bins[i+1])
        if mask.sum() > 0:
            bin_medians[i] = np.median(residuals[mask])

    # Interpolate between bin centers
    return interpolate.interp1d(
        bin_centers, bin_medians,
        kind='linear',
        bounds_error=False,
        fill_value=(bin_medians[0], bin_medians[-1])
    )


def rt_correction_from_reference(
    data: pd.DataFrame,
    reference_stats: pd.DataFrame,
    reference_mask: pd.Series,
    precursor_col: str = 'precursor_id',
    abundance_col: str = 'abundance',
    rt_col: str = 'retention_time',
    replicate_col: str = 'replicate_name',
    batch_col: Optional[str] = 'batch',
    spline_df: int = 5,
    per_batch: bool = True,
) -> RTCorrectionResult:
    """Compute RT-dependent correction factors from reference samples.

    For each sample, model how its abundances deviate from reference mean
    as a function of RT, using a smoothing spline.

    Args:
        data: DataFrame with peptide data
        reference_stats: Statistics from compute_reference_statistics
        reference_mask: Boolean mask for reference samples
        precursor_col: Column with precursor identifiers
        abundance_col: Column with abundance values
        rt_col: Column with retention times
        replicate_col: Column with replicate names
        batch_col: Column with batch identifiers (optional)
        spline_df: Degrees of freedom for RT spline
        per_batch: Whether to fit separate models per batch

    Returns:
        RTCorrectionResult with corrected data and model details

    """
    logger.info("Computing RT-dependent correction from reference")

    data = data.copy()

    # Get reference mean abundance per precursor
    data = data.merge(
        reference_stats[['mean_abundance', 'median_rt']],
        left_on=precursor_col,
        right_index=True,
        how='left'
    )

    # Calculate residuals (observed - reference mean)
    data['residual'] = data[abundance_col] - data['mean_abundance']

    # Use reference RT for consistency
    data['rt_for_model'] = data['median_rt']

    # Get samples and batches
    samples = data[replicate_col].unique()
    rt_models = {}
    correction_factors = {}

    if per_batch and batch_col in data.columns:
        batches = data[batch_col].unique()

        for batch in batches:
            batch_mask = data[batch_col] == batch
            batch_ref_mask = batch_mask & reference_mask

            # Fit RT model from reference samples in this batch
            ref_data = data.loc[batch_ref_mask]

            if len(ref_data) == 0:
                logger.warning(f"No reference samples in batch {batch}")
                continue

            rt_values = ref_data['rt_for_model'].values
            residuals = ref_data['residual'].values

            # Fit spline
            spline = fit_rt_spline(rt_values, residuals, df=spline_df)

            # Store model
            rt_models[f'batch_{batch}'] = {
                'spline': spline,
                'rt_range': (rt_values.min(), rt_values.max()),
            }

            # Apply to all samples in this batch
            batch_samples = data.loc[batch_mask, replicate_col].unique()
            for sample in batch_samples:
                sample_mask = data[replicate_col] == sample
                sample_rts = data.loc[sample_mask, 'rt_for_model'].values
                correction_factors[sample] = spline(sample_rts)

    else:
        # Global model from all reference samples
        ref_data = data.loc[reference_mask]

        rt_values = ref_data['rt_for_model'].values
        residuals = ref_data['residual'].values

        spline = fit_rt_spline(rt_values, residuals, df=spline_df)
        rt_models['global'] = {
            'spline': spline,
            'rt_range': (rt_values.min(), rt_values.max()),
        }

        # Apply to all samples
        for sample in samples:
            sample_mask = data[replicate_col] == sample
            sample_rts = data.loc[sample_mask, 'rt_for_model'].values
            correction_factors[sample] = spline(sample_rts)

    # Apply corrections
    data['rt_correction'] = 0.0
    for sample, corrections in correction_factors.items():
        sample_mask = data[replicate_col] == sample
        data.loc[sample_mask, 'rt_correction'] = corrections

    data['abundance_rt_corrected'] = data[abundance_col] - data['rt_correction']

    # Clean up temporary columns
    corrected_data = data.drop(
        columns=['residual', 'rt_for_model', 'mean_abundance', 'median_rt'],
        errors='ignore'
    )

    # Build correction factors DataFrame
    cf_df = pd.DataFrame({
        sample: data.loc[data[replicate_col] == sample, 'rt_correction'].values
        for sample in samples
    })

    return RTCorrectionResult(
        corrected_data=corrected_data,
        correction_factors=cf_df,
        rt_model=rt_models,
        reference_stats=reference_stats,
    )


def median_normalize(
    data: pd.DataFrame,
    abundance_col: str = 'abundance',
    replicate_col: str = 'replicate_name',
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply median centering normalization.

    Subtract sample median so all samples have median = 0 (on log scale).

    Args:
        data: DataFrame with peptide data
        abundance_col: Column with abundance values
        replicate_col: Column with replicate names

    Returns:
        Tuple of:
        - Normalized DataFrame
        - Series of normalization factors per sample

    """
    data = data.copy()

    # Calculate median per sample
    sample_medians = data.groupby(replicate_col)[abundance_col].median()

    # Global median
    global_median = sample_medians.median()

    # Normalization factors: shift each sample to global median
    norm_factors = sample_medians - global_median

    # Apply
    data['norm_factor'] = data[replicate_col].map(norm_factors)
    data[f'{abundance_col}_normalized'] = data[abundance_col] - data['norm_factor']

    logger.info(f"Median normalization: max shift = {norm_factors.abs().max():.3f}")

    return data.drop(columns=['norm_factor']), norm_factors


def vsn_normalize(
    data: pd.DataFrame,
    abundance_col: str = 'abundance',
    replicate_col: str = 'replicate_name',
    optimize_params: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply Variance Stabilizing Normalization (VSN) using arcsinh transformation.

    This method applies an arcsinh transformation to stabilize variance across
    intensity ranges, making variance independent of the mean.

    The transformation is: arcsinh(a * x + b) where:
    - a is a scaling factor (1/median by default, or optimized)
    - b is an offset (typically 0)

    Args:
        data: DataFrame with peptide data (log2 scale)
        abundance_col: Column with abundance values
        replicate_col: Column with replicate names
        optimize_params: Whether to optimize VSN parameters per sample
                        (slower but may give better variance stabilization)

    Returns:
        Tuple of:
        - Normalized DataFrame with new column '{abundance_col}_normalized'
        - Series of VSN parameters (a values) per sample

    """
    data = data.copy()

    def vsn_transform_sample(
        values: np.ndarray,
        optimize: bool = False,
    ) -> tuple[np.ndarray, float]:
        """Apply VSN transformation to a single sample's values."""
        # Convert from log2 to linear scale for VSN
        linear_values = np.power(2, values)

        # Remove zeros and negative values for parameter estimation
        valid_mask = linear_values > 0
        data_clean = linear_values[valid_mask]

        if len(data_clean) == 0:
            return np.zeros_like(values), 1.0

        if optimize:
            # Optimize VSN parameters to minimize variance heterogeneity
            def variance_heterogeneity(params):
                a = params[0]
                b = params[1] if len(params) > 1 else 0.0

                if a <= 0:
                    return 1e6

                transformed = np.arcsinh(a * data_clean + b)

                # Sort by original intensity and compute rolling variances
                sorted_idx = np.argsort(data_clean)
                sorted_transformed = transformed[sorted_idx]

                window_size = max(100, len(data_clean) // 20)
                variances = []

                for i in range(0, len(sorted_transformed) - window_size, window_size // 4):
                    window_data = sorted_transformed[i : i + window_size]
                    if len(window_data) > 10:
                        variances.append(np.var(window_data))

                # Return CV of variances (want this small for homoscedasticity)
                if len(variances) > 1:
                    mean_var = np.mean(variances)
                    if mean_var > 0:
                        return np.std(variances) / mean_var
                return 1e6

            # Try multiple starting points
            best_a, best_score = 1.0 / np.median(data_clean), float('inf')

            starting_points = [
                [1.0, 0.0],
                [0.1, 0.0],
                [10.0, 0.0],
                [1.0 / np.median(data_clean), 0.0],
                [1.0 / np.quantile(data_clean, 0.75), 0.0],
            ]

            for start in starting_points:
                try:
                    result = optimize.minimize(
                        variance_heterogeneity,
                        start,
                        method='Nelder-Mead',
                        options={'maxiter': 500},
                    )
                    if result.success and result.fun < best_score:
                        best_a = result.x[0]
                        best_score = result.fun
                except (ValueError, RuntimeError):
                    continue

            a_opt = best_a
        else:
            # Fast mode: use median-based scaling
            a_opt = 1.0 / np.median(data_clean)

        # Apply transformation
        # Handle zeros with small offset
        linear_for_transform = np.where(linear_values == 0, 1e-6, linear_values)
        transformed = np.arcsinh(a_opt * linear_for_transform)

        return transformed, a_opt

    # Apply to each sample
    samples = data[replicate_col].unique()
    vsn_params = {}

    for sample in samples:
        sample_mask = data[replicate_col] == sample
        values = data.loc[sample_mask, abundance_col].values

        transformed, a_param = vsn_transform_sample(values, optimize=optimize_params)

        data.loc[sample_mask, f'{abundance_col}_normalized'] = transformed
        vsn_params[sample] = a_param

    logger.info(f"VSN normalization applied (optimize_params={optimize_params})")

    return data, pd.Series(vsn_params)


def quantile_normalize(
    data: pd.DataFrame,
    abundance_col: str = 'abundance',
    replicate_col: str = 'replicate_name',
    precursor_col: str = 'precursor_id',
) -> pd.DataFrame:
    """Apply quantile normalization.

    Forces all samples to have identical distributions.

    Args:
        data: DataFrame with peptide data
        abundance_col: Column with abundance values
        replicate_col: Column with replicate names
        precursor_col: Column with precursor identifiers

    Returns:
        Normalized DataFrame

    """
    # Pivot to wide format
    matrix = data.pivot_table(
        index=precursor_col,
        columns=replicate_col,
        values=abundance_col,
    )

    # Rank each column
    ranks = matrix.rank(method='average')

    # Get sorted values per column, compute row means
    sorted_vals = np.sort(matrix.values, axis=0)
    rank_means = np.nanmean(sorted_vals, axis=1)

    # Map ranks back to mean values
    n_rows = len(rank_means)
    normalized = ranks.apply(
        lambda col: np.interp(
            col.values,
            np.arange(1, n_rows + 1),
            rank_means
        )
    )

    # Melt back to long format
    normalized = normalized.reset_index().melt(
        id_vars=[precursor_col],
        var_name=replicate_col,
        value_name=f'{abundance_col}_normalized'
    )

    # Merge back
    data = data.merge(
        normalized,
        on=[precursor_col, replicate_col],
        how='left'
    )

    return data


def normalize_pipeline(
    data: pd.DataFrame,
    sample_type_col: str = 'sample_type',
    precursor_col: str = 'precursor_id',
    abundance_col: str = 'abundance',
    rt_col: str = 'retention_time',
    replicate_col: str = 'replicate_name',
    batch_col: str = 'batch',
    rt_correction: bool = True,
    global_method: str = 'median',
    spline_df: int = 5,
    per_batch: bool = True,
    batch_correction: bool = False,
    batch_correction_params: Optional[dict] = None,
) -> NormalizationResult:
    """Run full normalization pipeline.

    Steps:
    1. Compute reference statistics
    2. Apply RT-dependent correction (optional)
    3. Apply global normalization
    4. Apply batch correction via ComBat (optional)

    Args:
        data: DataFrame with peptide data
        sample_type_col: Column indicating sample type (experimental, qc, reference)
        precursor_col: Column with precursor identifiers
        abundance_col: Column with abundance values
        rt_col: Column with retention times
        replicate_col: Column with replicate names
        batch_col: Column with batch identifiers
        rt_correction: Whether to apply RT-dependent correction
        global_method: Global normalization method ('median', 'quantile', 'none')
        spline_df: Degrees of freedom for RT spline
        per_batch: Whether to fit RT models per batch
        batch_correction: Whether to apply ComBat batch correction
        batch_correction_params: Additional parameters for ComBat (par_prior, mean_only, ref_batch)

    Returns:
        NormalizationResult with normalized data and diagnostics

    """
    method_log = []
    data = data.copy()

    # Identify reference samples
    reference_mask = data[sample_type_col] == 'reference'
    n_reference = data.loc[reference_mask, replicate_col].nunique()
    logger.info(f"Found {n_reference} reference replicates")

    if n_reference < 2:
        logger.warning("Fewer than 2 reference replicates - RT correction may be unreliable")

    # Step 1: Compute reference statistics
    ref_stats = compute_reference_statistics(
        data, reference_mask,
        precursor_col=precursor_col,
        abundance_col=abundance_col,
        rt_col=rt_col,
        replicate_col=replicate_col,
    )
    method_log.append(f"Computed reference statistics from {n_reference} replicates")

    # Step 2: RT correction
    rt_result = None
    working_abundance = abundance_col

    if rt_correction:
        rt_result = rt_correction_from_reference(
            data, ref_stats, reference_mask,
            precursor_col=precursor_col,
            abundance_col=abundance_col,
            rt_col=rt_col,
            replicate_col=replicate_col,
            batch_col=batch_col,
            spline_df=spline_df,
            per_batch=per_batch,
        )
        data = rt_result.corrected_data
        working_abundance = 'abundance_rt_corrected'
        method_log.append(f"Applied RT correction (df={spline_df}, per_batch={per_batch})")

    # Step 3: Global normalization
    if global_method == 'median':
        data, norm_factors = median_normalize(
            data,
            abundance_col=working_abundance,
            replicate_col=replicate_col,
        )
        method_log.append("Applied median normalization")

    elif global_method == 'vsn':
        data, norm_factors = vsn_normalize(
            data,
            abundance_col=working_abundance,
            replicate_col=replicate_col,
            optimize_params=False,  # Use fast mode by default
        )
        method_log.append("Applied VSN (variance stabilizing) normalization")

    elif global_method == 'quantile':
        data = quantile_normalize(
            data,
            abundance_col=working_abundance,
            replicate_col=replicate_col,
            precursor_col=precursor_col,
        )
        norm_factors = pd.Series()  # Quantile norm doesn't have simple factors
        method_log.append("Applied quantile normalization")

    elif global_method == 'none':
        norm_factors = pd.Series()
        method_log.append("No global normalization applied")

    else:
        raise ValueError(f"Unknown global normalization method: {global_method}")

    # Step 4: Batch correction (ComBat)
    if batch_correction:
        # Check if batch column exists and has multiple batches
        if batch_col not in data.columns:
            logger.warning(f"Batch column '{batch_col}' not found - skipping batch correction")
        else:
            n_batches = data[batch_col].nunique()
            if n_batches < 2:
                logger.warning(f"Only {n_batches} batch(es) found - skipping batch correction")
            else:
                # Determine which abundance column to use
                if 'abundance_normalized' in data.columns:
                    input_abundance = 'abundance_normalized'
                elif 'abundance_rt_corrected' in data.columns:
                    input_abundance = 'abundance_rt_corrected'
                else:
                    input_abundance = abundance_col

                # Get ComBat parameters
                combat_params = batch_correction_params or {}
                par_prior = combat_params.get('par_prior', True)
                mean_only = combat_params.get('mean_only', False)
                evaluate = combat_params.get('evaluate', True)
                fallback_on_failure = combat_params.get('fallback_on_failure', True)

                logger.info(f"Applying ComBat batch correction across {n_batches} batches")

                try:
                    # Use combat_with_reference_samples for automatic QC evaluation
                    combat_result, evaluation = combat_with_reference_samples(
                        data,
                        abundance_col=input_abundance,
                        feature_col=precursor_col,
                        sample_col=replicate_col,
                        batch_col=batch_col,
                        sample_type_col=sample_type_col,
                        par_prior=par_prior,
                        mean_only=mean_only,
                        evaluate=evaluate,
                        fallback_on_failure=fallback_on_failure,
                    )

                    # Store batch-corrected values
                    corrected_col = f'{input_abundance}_batch_corrected'
                    data['abundance_batch_corrected'] = combat_result[corrected_col]

                    # Log results
                    if evaluation is not None:
                        if evaluation.passed:
                            method_log.append(
                                f"Applied ComBat batch correction "
                                f"(ref CV: {evaluation.reference_cv_before:.3f} -> "
                                f"{evaluation.reference_cv_after:.3f}, "
                                f"QC CV: {evaluation.qc_cv_before:.3f} -> "
                                f"{evaluation.qc_cv_after:.3f})"
                            )
                        else:
                            if fallback_on_failure:
                                method_log.append(
                                    f"ComBat batch correction FAILED QC - using uncorrected data "
                                    f"(ref CV improvement: {evaluation.reference_improvement:.1%}, "
                                    f"QC CV improvement: {evaluation.qc_improvement:.1%})"
                                )
                            else:
                                method_log.append(
                                    f"ComBat batch correction applied but FAILED QC "
                                    f"({'; '.join(evaluation.warnings)})"
                                )
                    else:
                        method_log.append(
                            f"Applied ComBat batch correction (par_prior={par_prior}, "
                            f"mean_only={mean_only}) - no QC evaluation"
                        )

                    logger.info("ComBat batch correction completed")

                except Exception as e:
                    logger.error(f"Batch correction failed: {e}")
                    method_log.append(f"Batch correction FAILED: {e}")

    return NormalizationResult(
        normalized_data=data,
        rt_correction=rt_result,
        global_factors=norm_factors,
        method_log=method_log,
    )
