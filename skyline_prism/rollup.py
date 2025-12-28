"""Protein rollup module with Tukey median polish and other aggregation methods.

Supports:
- Transition → Peptide rollup with optional quality weighting
- Peptide → Protein rollup
- Multiple methods: median_polish, topn, ibaq, maxlfq
- Flexible shared peptide handling: all_groups, unique_only, razor
- Uncertainty propagation through rollup steps
- Protein-level batch correction (ComBat)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .batch_correction import (
    BatchCorrectionEvaluation,
    combat,
)

logger = logging.getLogger(__name__)


@dataclass
class MedianPolishResult:
    """Result of Tukey median polish.

    SCALE: All values are on LOG2 SCALE. The additive model on log2 scale
    corresponds to a multiplicative model on linear scale:
        log2(Y_ij) = μ + α_i + β_j + ε_ij

    The residuals matrix captures deviations from the additive model.

    Large residuals may indicate:
    - Technical outliers (interference, poor peak picking)
    - Biologically interesting variation (proteoforms, PTMs, protein processing)

    Following Plubell et al. 2022 (doi:10.1021/acs.jproteome.1c00894), outlier
    peptides should not be discarded but flagged for potential biological interest.

    To convert col_effects to linear scale: 2 ** col_effects
    """

    overall: float                    # Grand effect (μ) - log2 scale
    row_effects: pd.Series            # Peptide/transition effects (α) - log2 scale
    col_effects: pd.Series            # Sample abundances (μ + β) - log2 scale
    residuals: pd.DataFrame           # Residual matrix (rows × samples) - log2 scale
    n_iterations: int
    converged: bool

    def get_row_residual_summary(self) -> pd.DataFrame:
        """Compute per-row (peptide/transition) residual summary statistics.

        Returns a DataFrame with one row per peptide/transition containing:
        - residual_mean: Mean residual across samples
        - residual_std: Standard deviation of residuals
        - residual_mad: Median absolute deviation (robust)
        - residual_max_abs: Maximum absolute residual

        Large values indicate the row deviates from the additive model,
        potentially indicating biological interest (proteoforms, PTMs).
        """
        residuals_arr = self.residuals.values

        return pd.DataFrame({
            'residual_mean': np.nanmean(residuals_arr, axis=1),
            'residual_std': np.nanstd(residuals_arr, axis=1),
            'residual_mad': np.nanmedian(np.abs(residuals_arr - np.nanmedian(residuals_arr, axis=1, keepdims=True)), axis=1),
            'residual_max_abs': np.nanmax(np.abs(residuals_arr), axis=1),
        }, index=self.residuals.index)


@dataclass
class AggregationResult:
    """Result of quality-weighted aggregation."""

    abundances: pd.Series             # Aggregated abundances per sample
    uncertainties: pd.Series          # Uncertainty estimates per sample (log-scale std)
    n_signals_used: pd.Series         # Number of signals contributing per sample
    weights_used: Optional[pd.DataFrame] = None  # Weights applied (for diagnostics)


@dataclass
class TopNResult:
    """Result of top-N peptide rollup."""

    abundances: pd.Series             # Protein abundances per sample
    selected_peptides: list[str]      # List of peptide IDs that were selected
    n_available: int                  # How many peptides were available
    selection_method: str             # 'median_abundance' or 'frequency'


# ============================================================================
# Quality-Weighted Aggregation
# ============================================================================

@dataclass
class VarianceModelParams:
    """Parameters for the signal variance model.

    Variance model based on mass spectrometry noise sources:
        var(signal) = α·signal + β·signal² + γ + δ·quality_penalty

    Where:
    - α·signal: Shot noise (Poisson counting statistics)
    - β·signal²: Multiplicative noise (ionization efficiency variation)
    - γ: Additive noise (electronic/detector noise)
    - δ·quality_penalty: Additional variance from poor quality signals

    On log scale, this becomes approximately:
        var(log(signal)) ≈ α/signal + β + γ/signal² + δ·quality_penalty/signal²

    For high-intensity signals, β dominates (constant CV).
    For low-intensity signals, α/signal dominates (CV increases).

    Reference: Finney 2012 dissertation, section on noise models.
    """

    # Shot noise coefficient (Poisson): var ∝ signal
    alpha: float = 1.0

    # Multiplicative noise coefficient: var ∝ signal²
    beta: float = 0.01

    # Additive noise floor
    gamma: float = 100.0

    # Quality penalty coefficient
    delta: float = 1.0

    # Shape correlation exponent (how strongly correlation affects variance)
    shape_corr_exponent: float = 2.0

    # Coelution penalty multiplier
    coelution_penalty: float = 10.0


def estimate_signal_variance(
    intensity: np.ndarray,
    shape_correlation: Optional[np.ndarray] = None,
    coeluting: Optional[np.ndarray] = None,
    params: Optional[VarianceModelParams] = None,
) -> np.ndarray:
    """Estimate variance for each signal based on intensity and quality metrics.

    Uses a physically-motivated noise model:
    - Shot noise: var ∝ intensity (Poisson counting)
    - Multiplicative noise: var ∝ intensity² (ionization efficiency)
    - Additive noise: constant floor (electronics)
    - Quality penalties: from shape correlation and coelution

    Args:
        intensity: Signal intensities (linear scale)
        shape_correlation: Skyline shape correlation scores (0-1, higher is better)
        coeluting: Boolean indicating apex within integration boundaries
        params: Variance model parameters (uses defaults if None)

    Returns:
        Estimated variance for each signal (linear scale)

    """
    if params is None:
        params = VarianceModelParams()

    # Ensure positive intensity
    intensity = np.maximum(intensity, 1.0)

    # Base variance from noise model
    # var = α·I + β·I² + γ
    variance = (
        params.alpha * intensity +
        params.beta * np.square(intensity) +
        params.gamma
    )

    # Quality penalty from shape correlation
    if shape_correlation is not None:
        # Low correlation -> high penalty
        # Transform: corr=1 -> penalty=1, corr=0 -> penalty=large
        corr_safe = np.maximum(shape_correlation, 0.01)
        quality_penalty = np.power(1.0 / corr_safe, params.shape_corr_exponent)
        variance = variance * (1.0 + params.delta * (quality_penalty - 1.0))

    # Coelution penalty
    if coeluting is not None:
        coelution_mult = np.where(coeluting, 1.0, params.coelution_penalty)
        variance = variance * coelution_mult

    return variance


def estimate_log_variance(
    intensity: np.ndarray,
    shape_correlation: Optional[np.ndarray] = None,
    coeluting: Optional[np.ndarray] = None,
    params: Optional[VarianceModelParams] = None,
) -> np.ndarray:
    """Estimate variance of log-transformed signal.

    For signal S with variance var(S), the variance of log(S) is approximately:
        var(log(S)) ≈ var(S) / S²

    This is the delta method approximation.

    Args:
        intensity: Signal intensities (linear scale)
        shape_correlation: Skyline shape correlation scores
        coeluting: Boolean coelution flags
        params: Variance model parameters

    Returns:
        Estimated variance of log2(signal)

    """
    variance = estimate_signal_variance(intensity, shape_correlation, coeluting, params)

    # Delta method: var(log(S)) ≈ var(S) / S²
    # For log2: var(log2(S)) = var(ln(S)) / ln(2)² = var(S) / (S² · ln(2)²)
    log2_variance = variance / (np.square(intensity) * np.log(2)**2)

    return log2_variance


def learn_variance_params_from_reference(
    data: pd.DataFrame,
    reference_mask: pd.Series,
    precursor_col: str = 'precursor_id',
    transition_col: str = 'fragment_ion',
    abundance_col: str = 'abundance',
    sample_col: str = 'replicate_name',
    shape_correlation_col: Optional[str] = 'shape_correlation',
    coeluting_col: Optional[str] = 'coeluting',
    n_iterations: int = 50,
) -> VarianceModelParams:
    """Learn optimal variance model parameters by minimizing CV on reference samples.

    Uses the inter-experiment reference samples (which should have only technical
    variation) to optimize the weighting parameters. The objective is to minimize
    the median CV across peptides when using quality-weighted aggregation.

    Quality metrics (shape correlation) are aggregated using intensity-weighted
    averaging. This gives more importance to quality assessments from high-abundance
    samples where poor correlation is a stronger indicator of real interference.

    Args:
        data: DataFrame with transition-level data (long format)
        reference_mask: Boolean Series indicating reference sample rows
        precursor_col: Column identifying precursors
        transition_col: Column identifying transitions
        abundance_col: Column with abundance values (log2)
        sample_col: Column with sample names
        shape_correlation_col: Column with shape correlation (or None)
        coeluting_col: Column with coelution flags (or None)
        n_iterations: Number of optimization iterations

    Returns:
        Optimized VarianceModelParams

    """
    from scipy.optimize import minimize

    logger.info("Learning variance model parameters from reference samples")

    # Extract reference data
    ref_data = data.loc[reference_mask].copy()
    ref_samples = ref_data[sample_col].unique()

    if len(ref_samples) < 2:
        logger.warning("Need at least 2 reference replicates to learn parameters")
        return VarianceModelParams()

    # Check for quality columns
    has_shape_corr = shape_correlation_col and shape_correlation_col in ref_data.columns
    has_coeluting = coeluting_col and coeluting_col in ref_data.columns

    precursors = ref_data[precursor_col].unique()

    # Build transition summary for each precursor using intensity-weighted quality
    transition_summaries = {}
    for prec in precursors:
        prec_data = ref_data[ref_data[precursor_col] == prec]
        transitions = prec_data[transition_col].unique()

        if len(transitions) < 3:
            continue

        summary = {}
        for trans in transitions:
            trans_data = prec_data[prec_data[transition_col] == trans]

            # Linear intensities for this transition across reference replicates
            linear_int = np.power(2, trans_data[abundance_col].values)
            median_int = np.nanmedian(linear_int)

            # Intensity weights for averaging quality metrics
            int_weights = linear_int / np.nansum(linear_int)

            # Intensity-weighted shape correlation
            if has_shape_corr:
                corr_values = trans_data[shape_correlation_col].values
                valid = np.isfinite(corr_values) & np.isfinite(int_weights)
                if valid.any():
                    w = int_weights[valid] / int_weights[valid].sum()
                    weighted_corr = np.sum(w * corr_values[valid])
                else:
                    weighted_corr = 0.5
            else:
                weighted_corr = None

            # Intensity-weighted coelution fraction
            if has_coeluting:
                coel_values = trans_data[coeluting_col].values.astype(float)
                valid = np.isfinite(coel_values) & np.isfinite(int_weights)
                if valid.any():
                    w = int_weights[valid] / int_weights[valid].sum()
                    weighted_coel = np.sum(w * coel_values[valid])
                else:
                    weighted_coel = 0.5
            else:
                weighted_coel = None

            summary[trans] = {
                'median_intensity': median_int,
                'weighted_shape_corr': weighted_corr,
                'weighted_coeluting': weighted_coel,
            }

        transition_summaries[prec] = summary

    def compute_weighted_cv(params_array):
        """Compute median CV using given variance parameters."""
        # Unpack parameters (in log space for positivity)
        params = VarianceModelParams(
            alpha=np.exp(params_array[0]),
            beta=np.exp(params_array[1]),
            gamma=np.exp(params_array[2]),
            delta=np.exp(params_array[3]) if has_shape_corr else 1.0,
            shape_corr_exponent=params_array[4] if has_shape_corr else 2.0,
            coelution_penalty=np.exp(params_array[5]) if has_coeluting else 10.0,
        )

        cvs = []
        for prec, trans_summary in transition_summaries.items():
            transitions = list(trans_summary.keys())
            n_trans = len(transitions)

            if n_trans < 3:
                continue

            # Compute variance and weights for each transition
            intensities = np.array([trans_summary[t]['median_intensity'] for t in transitions])

            if has_shape_corr:
                corrs = np.array([trans_summary[t]['weighted_shape_corr'] for t in transitions])
            else:
                corrs = None

            if has_coeluting:
                coeluting_arr = np.array([trans_summary[t]['weighted_coeluting'] > 0.5 for t in transitions])
            else:
                coeluting_arr = None

            # Estimate variance per transition
            variances = estimate_signal_variance(intensities, corrs, coeluting_arr, params)

            # Convert to log-scale variance
            log_var = variances / (np.square(intensities) * np.log(2)**2)

            # Weights (inverse variance, normalized)
            weights = 1.0 / log_var
            weights = weights / weights.sum()

            # Get the transition matrix for this precursor in reference samples
            prec_data = ref_data[ref_data[precursor_col] == prec]
            matrix = prec_data.pivot_table(
                index=transition_col,
                columns=sample_col,
                values=abundance_col,
                aggfunc='first'
            )

            # Ensure transition order matches
            matrix = matrix.reindex(index=transitions)

            # Weighted mean across transitions for each reference sample
            peptide_values = np.sum(weights[:, np.newaxis] * matrix.values, axis=0)

            # CV of peptide values across reference samples
            if len(peptide_values) >= 2 and np.nanstd(peptide_values) > 0:
                cv = np.nanstd(peptide_values) / np.abs(np.nanmean(peptide_values))
                cvs.append(cv)

        if not cvs:
            return 1.0

        return np.median(cvs)

    # Initial parameters (log scale)
    x0 = np.array([
        0.0,   # log(alpha) = log(1)
        -4.6,  # log(beta) = log(0.01)
        4.6,   # log(gamma) = log(100)
        0.0,   # log(delta) = log(1)
        2.0,   # shape_corr_exponent
        2.3,   # log(coelution_penalty) = log(10)
    ])

    # Optimize
    result = minimize(
        compute_weighted_cv,
        x0,
        method='Nelder-Mead',
        options={'maxiter': n_iterations, 'disp': False}
    )

    # Extract optimized parameters
    opt = result.x
    learned_params = VarianceModelParams(
        alpha=np.exp(opt[0]),
        beta=np.exp(opt[1]),
        gamma=np.exp(opt[2]),
        delta=np.exp(opt[3]) if has_shape_corr else 1.0,
        shape_corr_exponent=opt[4] if has_shape_corr else 2.0,
        coelution_penalty=np.exp(opt[5]) if has_coeluting else 10.0,
    )

    initial_cv = compute_weighted_cv(x0)
    final_cv = compute_weighted_cv(opt)

    logger.info(f"Learned variance parameters: α={learned_params.alpha:.3f}, "
                f"β={learned_params.beta:.4f}, γ={learned_params.gamma:.1f}")
    logger.info(f"Reference CV: {initial_cv:.4f} → {final_cv:.4f}")
    if has_shape_corr:
        logger.info(f"  Shape correlation: δ={learned_params.delta:.3f}, "
                    f"exponent={learned_params.shape_corr_exponent:.2f}")
    if has_coeluting:
        logger.info(f"  Coelution penalty: {learned_params.coelution_penalty:.1f}x")

    return learned_params


def quality_weighted_aggregate(
    log_intensities: np.ndarray,
    variances: np.ndarray,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate signals using inverse-variance weighting.

    For uncorrelated signals with known variances, the minimum-variance
    unbiased estimator is the inverse-variance weighted mean:

        μ̂ = Σ(x_i / var_i) / Σ(1 / var_i)
        var(μ̂) = 1 / Σ(1 / var_i)

    Args:
        log_intensities: Log2 intensities to aggregate
        variances: Estimated variances for each signal
        axis: Axis along which to aggregate

    Returns:
        Tuple of (aggregated_values, aggregated_variances)

    """
    # Inverse variance weights
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_var = 1.0 / variances
        inv_var = np.where(np.isfinite(inv_var), inv_var, 0)

    # Mask NaN values
    valid = np.isfinite(log_intensities) & np.isfinite(variances)
    inv_var = np.where(valid, inv_var, 0)
    weighted_vals = np.where(valid, log_intensities * inv_var, 0)

    # Weighted sum
    sum_weights = np.nansum(inv_var, axis=axis)
    sum_weighted = np.nansum(weighted_vals, axis=axis)

    # Aggregated value and variance
    with np.errstate(divide='ignore', invalid='ignore'):
        aggregated = sum_weighted / sum_weights
        aggregated_var = 1.0 / sum_weights

    # Handle cases with no valid signals
    aggregated = np.where(sum_weights > 0, aggregated, np.nan)
    aggregated_var = np.where(sum_weights > 0, aggregated_var, np.nan)

    return aggregated, aggregated_var


def aggregate_transitions_quality_weighted(
    transition_matrix: pd.DataFrame,
    shape_correlation: Optional[pd.DataFrame] = None,
    coeluting: Optional[pd.DataFrame] = None,
    params: Optional[VarianceModelParams] = None,
    min_transitions: int = 3,
    max_variance_ratio: float = 100.0,
) -> AggregationResult:
    """Aggregate transitions to peptide level using quality-weighted combination.

    IMPORTANT: Weights are computed per-transition (not per-transition-per-replicate).
    This ensures consistent treatment across the experiment - the same weights are
    applied to all replicates, preserving relative abundances.

    Quality metrics (shape correlation) are aggregated across replicates using
    intensity-weighted averaging. This gives more importance to quality assessments
    from high-abundance samples where:
    - We expect clean signal if there's no interference
    - Poor correlation is a stronger indicator of real interference
    - Low-abundance samples have more noise, so poor correlation is less informative

    Args:
        transition_matrix: Transitions × samples matrix (log2 intensities)
        shape_correlation: Transitions × samples matrix of shape correlations
        coeluting: Transitions × samples matrix of coelution flags
        params: Variance model parameters (uses defaults if None)
        min_transitions: Minimum transitions required for aggregation
        max_variance_ratio: Cap on variance ratio to prevent extreme weights

    Returns:
        AggregationResult with abundances and uncertainties

    """
    n_transitions, n_samples = transition_matrix.shape

    if n_transitions < min_transitions:
        # Not enough transitions - return simple median
        return AggregationResult(
            abundances=transition_matrix.median(axis=0),
            uncertainties=pd.Series(np.nan, index=transition_matrix.columns),
            n_signals_used=pd.Series(n_transitions, index=transition_matrix.columns),
        )

    # Convert to linear for variance estimation
    linear_intensity = np.power(2, transition_matrix.values)

    # Compute median intensity per transition (for variance model)
    median_intensity_per_transition = np.nanmedian(linear_intensity, axis=1)

    # Get quality metrics using intensity-weighted averaging across replicates
    # Higher intensity replicates contribute more to the quality assessment
    if shape_correlation is not None:
        # Intensity-weighted average of shape correlation
        # Weight by linear intensity - high abundance samples inform quality more
        intensity_weights = linear_intensity / np.nansum(linear_intensity, axis=1, keepdims=True)

        # Handle NaN in shape correlation
        corr_values = shape_correlation.values.copy()
        valid_mask = np.isfinite(corr_values) & np.isfinite(intensity_weights)

        # Compute intensity-weighted mean correlation per transition
        shape_corr_per_transition = np.zeros(n_transitions)
        for i in range(n_transitions):
            valid = valid_mask[i, :]
            if valid.any():
                w = intensity_weights[i, valid]
                w = w / w.sum()  # Renormalize for valid entries
                shape_corr_per_transition[i] = np.sum(w * corr_values[i, valid])
            else:
                shape_corr_per_transition[i] = 0.5  # Default if no valid data
    else:
        shape_corr_per_transition = None

    if coeluting is not None:
        # Intensity-weighted fraction coeluting
        intensity_weights = linear_intensity / np.nansum(linear_intensity, axis=1, keepdims=True)
        coeluting_values = coeluting.values.astype(float)

        coeluting_weighted = np.zeros(n_transitions)
        for i in range(n_transitions):
            valid = np.isfinite(coeluting_values[i, :]) & np.isfinite(intensity_weights[i, :])
            if valid.any():
                w = intensity_weights[i, valid]
                w = w / w.sum()
                coeluting_weighted[i] = np.sum(w * coeluting_values[i, valid])
            else:
                coeluting_weighted[i] = 0.5

        coeluting_fraction = coeluting_weighted
    else:
        coeluting_fraction = None

    # Estimate variance per transition (single value, not per-replicate)
    transition_variance = estimate_signal_variance(
        median_intensity_per_transition,
        shape_correlation=shape_corr_per_transition,
        coeluting=coeluting_fraction > 0.5 if coeluting_fraction is not None else None,
        params=params,
    )

    # Convert to log-scale variance (per transition)
    log_variance_per_transition = transition_variance / (
        np.square(median_intensity_per_transition) * np.log(2)**2
    )

    # Cap extreme variance ratios
    min_var = np.nanmin(log_variance_per_transition)
    log_variance_per_transition = np.minimum(
        log_variance_per_transition,
        min_var * max_variance_ratio
    )

    # Inverse variance weights (one per transition, applied to all replicates)
    weights = 1.0 / log_variance_per_transition
    weights = weights / np.sum(weights)  # Normalize

    # Apply weights: weighted mean across transitions for each sample
    weighted_sum = np.sum(weights[:, np.newaxis] * transition_matrix.values, axis=0)
    aggregated = weighted_sum  # Already weighted mean since weights sum to 1

    # Propagated variance
    aggregated_var = np.sum(np.square(weights) * log_variance_per_transition)

    # Count contributing signals
    n_used = pd.Series(n_transitions, index=transition_matrix.columns)

    # Create weights DataFrame for diagnostics
    weights_df = pd.DataFrame(
        np.tile(weights[:, np.newaxis], (1, n_samples)),
        index=transition_matrix.index,
        columns=transition_matrix.columns
    )

    return AggregationResult(
        abundances=pd.Series(aggregated, index=transition_matrix.columns),
        uncertainties=pd.Series(np.sqrt(aggregated_var), index=transition_matrix.columns),
        n_signals_used=n_used,
        weights_used=weights_df,
    )


def tukey_median_polish(
    matrix: pd.DataFrame,
    max_iter: int = 20,
    tol: float = 1e-4,
) -> MedianPolishResult:
    """Apply Tukey's median polish to a peptide × sample matrix.

    SCALE: Input and output are on LOG2 SCALE.

    Model: y_ij = μ + α_i + β_j + ε_ij

    Where:
        - μ = overall effect (grand median)
        - α_i = row/peptide effect (relative to grand median)
        - β_j = column/sample effect (relative to grand median)
        - ε_ij = residual

    The returned col_effects are on the ORIGINAL LOG2 SCALE (μ + β_j),
    not centered deviations. To get linear-scale abundances: 2 ** col_effects

    Args:
        matrix: DataFrame with peptides as rows, samples as columns.
                Values MUST be log2 transformed.
        max_iter: Maximum number of iterations
        tol: Convergence tolerance (max absolute change in residuals)

    Returns:
        MedianPolishResult with effects and residuals (all log2 scale)

    """
    # Work with numpy for speed, but keep track of indices
    row_idx = matrix.index
    col_idx = matrix.columns

    # Initialize
    residuals = matrix.values.copy().astype(float)
    overall = 0.0
    row_effects = np.zeros(len(row_idx))
    col_effects = np.zeros(len(col_idx))

    converged = False

    for iteration in range(max_iter):
        old_residuals = residuals.copy()

        # Step 1: Row sweep (subtract row medians)
        row_medians = np.nanmedian(residuals, axis=1)
        residuals = residuals - row_medians[:, np.newaxis]

        # Update row effects
        row_effect_update = row_medians - np.nanmedian(row_medians)
        row_effects += row_effect_update
        overall += np.nanmedian(row_medians)

        # Step 2: Column sweep (subtract column medians)
        col_medians = np.nanmedian(residuals, axis=0)
        residuals = residuals - col_medians[np.newaxis, :]

        # Update column effects
        col_effect_update = col_medians - np.nanmedian(col_medians)
        col_effects += col_effect_update
        overall += np.nanmedian(col_medians)

        # Check convergence
        max_change = np.nanmax(np.abs(residuals - old_residuals))
        if max_change < tol:
            converged = True
            break

    # Return col_effects on original scale (overall + sample effects)
    # This gives actual protein/peptide abundances, not centered deviations.
    # The additive model is: y_ij = overall + row_effects[i] + col_effects[j] + residuals
    # For protein abundance per sample, we want: overall + col_effects[j]
    col_effects_original_scale = col_effects + overall

    # Wrap results
    result = MedianPolishResult(
        overall=overall,
        row_effects=pd.Series(row_effects, index=row_idx, name='peptide_effect'),
        col_effects=pd.Series(col_effects_original_scale, index=col_idx, name='protein_abundance'),
        residuals=pd.DataFrame(residuals, index=row_idx, columns=col_idx),
        n_iterations=iteration + 1,
        converged=converged,
    )

    # Non-convergence is common and acceptable - the algorithm still produces
    # robust estimates. Large residuals from interfered transitions prevent
    # convergence, which is expected behavior. Log at debug level.
    if not converged:
        logger.debug(f"Median polish did not converge after {max_iter} iterations")

    return result


def rollup_top_n(
    matrix: pd.DataFrame,
    n: int = 3,
    selection: str = 'median_abundance',
) -> TopNResult:
    """Rollup using average of top N peptides selected GLOBALLY across all samples.

    CRITICAL: The same N peptides must be used for ALL samples to ensure
    protein abundances are comparable. Per-sample selection causes bias
    because different peptides may have different ionization efficiencies.

    Selection methods:
    - 'median_abundance': Select peptides with highest median log2 abundance
      across all samples (robust to outliers)
    - 'frequency': Select peptides detected in the most samples, with ties
      broken by median abundance (maximizes data completeness)

    Args:
        matrix: Peptide × sample matrix (log2 values), peptides as rows
        n: Number of top peptides to select and average
        selection: Peptide selection criterion ('median_abundance' or 'frequency')

    Returns:
        TopNResult containing:
        - abundances: Series of protein abundances per sample
        - selected_peptides: List of the peptide IDs that were selected
        - n_available: How many peptides were available
        - selection_method: The selection method used

    """
    n_available = len(matrix)

    if n_available == 0:
        return TopNResult(
            abundances=pd.Series(np.nan, index=matrix.columns),
            selected_peptides=[],
            n_available=0,
            selection_method=selection,
        )

    # Calculate ranking metrics for each peptide
    peptide_stats = pd.DataFrame({
        'median_abundance': matrix.median(axis=1),
        'frequency': matrix.notna().sum(axis=1),
    })

    # Select top N peptides based on selection method
    if selection == 'frequency':
        # Primary: frequency (more samples = better)
        # Secondary: median abundance (tie-breaker)
        peptide_stats = peptide_stats.sort_values(
            by=['frequency', 'median_abundance'],
            ascending=[False, False]
        )
    else:  # median_abundance (default)
        # Primary: median abundance (higher = more reliably quantified)
        # Secondary: frequency (tie-breaker)
        peptide_stats = peptide_stats.sort_values(
            by=['median_abundance', 'frequency'],
            ascending=[False, False]
        )

    # Take top N peptides
    n_to_use = min(n, n_available)
    selected_peptides = peptide_stats.head(n_to_use).index.tolist()

    # Subset matrix to selected peptides only
    selected_matrix = matrix.loc[selected_peptides]

    # Calculate mean of selected peptides for each sample
    abundances = selected_matrix.mean(axis=0)

    return TopNResult(
        abundances=abundances,
        selected_peptides=selected_peptides,
        n_available=n_available,
        selection_method=selection,
    )


def rollup_ibaq(
    matrix: pd.DataFrame,
    n_theoretical_peptides: int,
) -> pd.Series:
    """iBAQ: Intensity-Based Absolute Quantification.

    Sum of intensities divided by theoretical peptide count.
    Useful for cross-protein abundance comparison.

    Args:
        matrix: Peptide × sample matrix (log2 values)
        n_theoretical_peptides: Number of theoretical tryptic peptides

    Returns:
        Series of iBAQ values per sample (log2 scale)

    """
    # Convert from log2 to linear
    linear = np.power(2, matrix)

    # Sum across peptides
    total_intensity = linear.sum(axis=0, skipna=True)

    # Divide by theoretical peptide count
    ibaq = total_intensity / n_theoretical_peptides

    # Back to log2
    return np.log2(ibaq)


def rollup_maxlfq(
    matrix: pd.DataFrame,
) -> pd.Series:
    """maxLFQ-style rollup using maximum peptide ratio extraction.

    For each pair of samples, find the median peptide log-ratio.
    Solve for protein abundances that best explain these ratios.

    Args:
        matrix: Peptide × sample matrix (log2 values)

    Returns:
        Series of protein abundances per sample

    """
    samples = matrix.columns
    n_samples = len(samples)

    if n_samples < 2:
        return matrix.median(axis=0)

    # Calculate pairwise median log-ratios
    ratio_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                # Log-ratio for each peptide
                ratios = matrix.iloc[:, i] - matrix.iloc[:, j]
                valid_ratios = ratios.dropna()
                if len(valid_ratios) > 0:
                    ratio_matrix[i, j] = np.median(valid_ratios)

    # Solve for abundances
    # The ratio_matrix[i,j] ≈ abundance[i] - abundance[j]
    # Use column means as initial estimate (equivalent to least squares solution)
    abundances = ratio_matrix.mean(axis=1)

    # Center to median
    abundances = abundances - np.median(abundances)

    # Add back the overall level from original data
    overall_median = matrix.median().median()
    abundances = abundances + overall_median

    return pd.Series(abundances, index=samples, name='protein_abundance')


def rollup_directlfq(
    matrix: pd.DataFrame,
) -> pd.Series:
    """DirectLFQ-style rollup - PLACEHOLDER, NOT FULLY IMPLEMENTED.

    NOTE: This is a simplified placeholder that does NOT implement the true
    directLFQ algorithm. The real directLFQ uses an "intensity trace" approach
    with linear O(n) runtime scaling, suitable for very large cohorts.

    For the actual directLFQ algorithm, see:
    - Paper: Ammar et al. 2023, MCP. doi:10.1016/j.mcpro.2023.100581
    - GitHub: https://github.com/MannLabs/directlfq

    This placeholder simply centers peptides and takes medians, which is
    NOT equivalent to directLFQ. Use maxlfq or median_polish instead.

    Args:
        matrix: Peptide × sample matrix (log2 values)

    Returns:
        Series of protein abundances per sample

    Raises:
        NotImplementedError: Always raises - use maxlfq or median_polish instead

    """
    raise NotImplementedError(
        "directLFQ is not implemented in PRISM. The directLFQ algorithm uses a "
        "fundamentally different 'intensity trace' approach with O(n) scaling. "
        "For large cohorts, use the directLFQ package directly: "
        "https://github.com/MannLabs/directlfq. "
        "For PRISM, use 'median_polish' (recommended) or 'maxlfq' instead."
    )


def rollup_sum(matrix: pd.DataFrame) -> pd.Series:
    """Rollup by summing all peptide abundances.

    Note: Works with linear values, not log2!

    Args:
        matrix: Peptide × sample matrix (linear intensities)

    Returns:
        Series of protein abundances per sample

    """
    return matrix.sum(axis=0, skipna=True)


# ============================================================================
# Transition to Peptide Rollup
# ============================================================================

def rollup_transitions_to_peptides(
    data: pd.DataFrame,
    transition_col: str = 'FragmentIon',
    precursor_col: str = 'precursor_id',
    abundance_col: str = 'Area',
    sample_col: str = 'replicate_name',
    method: str = 'median_polish',
    shape_correlation_col: Optional[str] = 'ShapeCorrelation',
    coeluting_col: Optional[str] = 'Coeluting',
    use_ms1: bool = False,
    ms1_col: Optional[str] = 'Ms1Area',
    min_transitions: int = 3,
    variance_params: Optional[VarianceModelParams] = None,
    reference_mask: Optional[pd.Series] = None,
    learn_params: bool = False,
) -> tuple[pd.DataFrame, dict, Optional[VarianceModelParams]]:
    """Combine transitions into peptide-level quantities.

    Methods:
    - 'median_polish': Tukey median polish (robust to outlier transitions)
    - 'sum': Simple sum of transitions
    - 'quality_weighted': Inverse-variance weighted combination using
      Skyline quality metrics (shape correlation, coelution)

    Args:
        data: DataFrame with transition-level data
        transition_col: Column identifying transitions
        precursor_col: Column identifying precursors (peptide + charge)
        abundance_col: Column with abundance values (log2)
        sample_col: Column with sample names
        method: 'median_polish', 'sum', or 'quality_weighted'
        shape_correlation_col: Column with Skyline shape correlation scores
        coeluting_col: Column with Skyline coelution flags
        use_ms1: Whether to include MS1 signal (if False, MS1 is ignored)
        ms1_col: Column with MS1 area (only used if use_ms1=True)
        min_transitions: Minimum transitions required per precursor
        variance_params: Pre-specified variance model parameters
        reference_mask: Boolean mask for reference samples (for learning params)
        learn_params: Whether to learn variance params from reference samples

    Returns:
        Tuple of:
        - DataFrame with peptide-level abundances (long format)
        - Dict of rollup details per precursor
        - Learned VarianceModelParams (if learn_params=True, else None)

    """
    logger.info(f"Rolling up transitions to peptides using {method}")
    if not use_ms1:
        logger.info("MS1 data will not be used for quantification")

    learned_params = None

    # Learn variance parameters if requested
    if method == 'quality_weighted' and learn_params and reference_mask is not None:
        learned_params = learn_variance_params_from_reference(
            data,
            reference_mask,
            precursor_col=precursor_col,
            abundance_col=abundance_col,
            sample_col=sample_col,
            shape_correlation_col=shape_correlation_col if shape_correlation_col in data.columns else None,
            coeluting_col=coeluting_col if coeluting_col in data.columns else None,
        )
        variance_params = learned_params

    precursors = data[precursor_col].unique()
    samples = data[sample_col].unique()

    peptide_results = []
    rollup_details = {}

    for precursor in precursors:
        prec_data = data[data[precursor_col] == precursor]

        # Exclude MS1 if not using it
        if not use_ms1 and ms1_col in prec_data.columns:
            # Filter out MS1 rows if they're in the transition data
            # (depends on how Skyline exports - MS1 might be separate or marked)
            if 'precursor' in prec_data[transition_col].str.lower().values:
                prec_data = prec_data[~prec_data[transition_col].str.lower().str.contains('precursor')]

        # Pivot to matrix: transitions × samples
        matrix = prec_data.pivot_table(
            index=transition_col,
            columns=sample_col,
            values=abundance_col,
            aggfunc='first'
        )

        # Ensure all samples present
        for s in samples:
            if s not in matrix.columns:
                matrix[s] = np.nan
        matrix = matrix[list(samples)]

        # Get quality metrics if available and method needs them
        shape_corr_matrix = None
        coeluting_matrix = None

        if method == 'quality_weighted':
            if shape_correlation_col and shape_correlation_col in prec_data.columns:
                shape_corr_matrix = prec_data.pivot_table(
                    index=transition_col,
                    columns=sample_col,
                    values=shape_correlation_col,
                    aggfunc='first'
                ).reindex(index=matrix.index, columns=matrix.columns)

            if coeluting_col and coeluting_col in prec_data.columns:
                coeluting_matrix = prec_data.pivot_table(
                    index=transition_col,
                    columns=sample_col,
                    values=coeluting_col,
                    aggfunc='first'
                ).reindex(index=matrix.index, columns=matrix.columns)

        # Apply rollup method
        if len(matrix) < min_transitions:
            # Too few transitions - use simple median
            abundances = matrix.median(axis=0)
            uncertainties = pd.Series(np.nan, index=samples)

        elif method == 'median_polish':
            if len(matrix) >= 2:
                result = tukey_median_polish(matrix)
                abundances = result.col_effects
                # Estimate uncertainty from residual variance
                residual_var = result.residuals.var(axis=0)
                uncertainties = np.sqrt(residual_var / len(matrix))
                rollup_details[precursor] = result
            else:
                abundances = matrix.iloc[0]
                uncertainties = pd.Series(np.nan, index=samples)

        elif method == 'sum':
            # Sum transitions (linear scale)
            linear = np.power(2, matrix)
            summed = linear.sum(axis=0, skipna=True)
            abundances = np.log2(summed)
            uncertainties = pd.Series(np.nan, index=samples)

        elif method == 'quality_weighted':
            result = aggregate_transitions_quality_weighted(
                matrix,
                shape_correlation=shape_corr_matrix,
                coeluting=coeluting_matrix,
                params=variance_params,
                min_transitions=min_transitions,
            )
            abundances = result.abundances
            uncertainties = result.uncertainties
            rollup_details[precursor] = result

        else:
            raise ValueError(f"Unknown transition rollup method: {method}")

        # Store results
        for sample in samples:
            peptide_results.append({
                precursor_col: precursor,
                sample_col: sample,
                'abundance': abundances.get(sample, np.nan),
                'uncertainty': uncertainties.get(sample, np.nan),
            })

    result_df = pd.DataFrame(peptide_results)

    logger.info(f"Rolled up {len(precursors)} precursors from transitions")
    if method == 'quality_weighted':
        n_with_quality = sum(1 for r in rollup_details.values()
                           if isinstance(r, AggregationResult) and r.weights_used is not None)
        logger.info(f"Quality weighting applied to {n_with_quality} precursors")

    return result_df, rollup_details, learned_params


# ============================================================================
# Peptide to Protein Rollup
# ============================================================================

def rollup_to_proteins(
    peptide_data: pd.DataFrame,
    protein_groups: list,  # List[ProteinGroup]
    abundance_col: str = 'abundance',
    sample_col: str = 'replicate_name',
    peptide_col: str = 'peptide_modified',
    method: str = 'median_polish',
    shared_peptide_handling: str = 'all_groups',
    n_theoretical_peptides: Optional[dict[str, int]] = None,  # For iBAQ
    topn_n: int = 3,
    topn_selection: str = 'median_abundance',
    min_peptides: int = 3,
) -> tuple[pd.DataFrame, dict[str, MedianPolishResult], dict[str, TopNResult]]:
    """Aggregate peptide abundances to protein group level.

    All proteins are quantified regardless of peptide count, using appropriate
    methods based on peptide count:
    - 1 peptide: Use peptide abundance directly
    - 2 peptides: Use mean of peptides
    - ≥3 peptides: Use requested method (median_polish, topn, etc.)

    Proteins with fewer than min_peptides are flagged as low_confidence=True
    in the output but are still quantified. The n_peptides column in output
    allows users to filter by peptide count if desired.

    Args:
        peptide_data: DataFrame with peptide abundances in long format
        protein_groups: List of ProteinGroup objects from parsimony
        abundance_col: Column with abundance values
        sample_col: Column with sample identifiers
        peptide_col: Column with peptide identifiers
        method: Rollup method (for proteins with ≥3 peptides):
            - 'median_polish': Tukey median polish (default, robust)
            - 'topn': Average of top N peptides
            - 'ibaq': Sum / theoretical peptide count
            - 'maxlfq': Maximum LFQ (pairwise ratio extraction)

            Note: directLFQ is NOT implemented. For large cohorts requiring
            O(n) scaling, use the directLFQ package directly:
            https://github.com/MannLabs/directlfq
        shared_peptide_handling: How to handle shared peptides:
            - 'all_groups': Apply to ALL groups (default, recommended)
            - 'unique_only': Only use unique peptides
            - 'razor': Assign to group with most peptides
        n_theoretical_peptides: Dict of group_id -> count (required for iBAQ)
        topn_n: N for top-N method
        topn_selection: Peptide selection for top-N ('median_abundance' or 'frequency')
        min_peptides: Minimum peptides for full-confidence quantification.
            Proteins with fewer peptides are still quantified using fallback
            methods but are flagged as low_confidence=True.

    Returns:
        Tuple of:
        - DataFrame with protein group × sample abundances. Includes columns:
            - Sample columns with protein abundances
            - leading_protein: Representative protein accession
            - leading_name: Protein name/description
            - n_peptides: Number of peptides used
            - n_unique_peptides: Number of unique peptides
            - low_confidence: True if n_peptides < min_peptides
        - Dict of MedianPolishResult per protein (if method='median_polish')
        - Dict of TopNResult per protein (if method='topn')

    """
    logger.info(f"Rolling up to proteins using {method}, "
                f"shared peptide handling: {shared_peptide_handling}")

    # Build peptide -> groups mapping based on handling strategy
    peptide_to_groups: dict[str, set[str]] = {}
    group_to_peptides: dict[str, set[str]] = {}

    for group in protein_groups:
        group_id = group.group_id

        if shared_peptide_handling == 'unique_only':
            peptides = group.unique_peptides
        elif shared_peptide_handling == 'razor':
            peptides = group.peptides  # Parsimony already assigned razor peptides
        else:  # 'all_groups' - default
            # Include ALL peptides that originally mapped to this protein
            # (shared peptides contribute to ALL groups they map to)
            peptides = group.all_mapped_peptides

        group_to_peptides[group_id] = peptides

        for pep in peptides:
            if pep not in peptide_to_groups:
                peptide_to_groups[pep] = set()
            peptide_to_groups[pep].add(group_id)

    # Get unique samples
    samples = peptide_data[sample_col].unique()

    # Results containers
    protein_abundances = {}
    protein_cv_peptides = {}  # CV across peptides per sample
    protein_uncertainties = {}  # Uncertainty (std of peptides / sqrt(n))
    protein_qc_flags = {}  # QC warning flags
    polish_results = {}
    topn_results = {}
    skipped_groups = []  # Track groups with 0 peptides

    for group in protein_groups:
        group_id = group.group_id
        peptides = group_to_peptides.get(group_id, set())
        n_peptides = len(peptides)

        # Skip groups with no peptides at all
        if n_peptides == 0:
            skipped_groups.append((group_id, 0))
            continue

        # Filter to this group's peptides
        mask = peptide_data[peptide_col].isin(peptides)
        group_data = peptide_data.loc[mask, [peptide_col, sample_col, abundance_col]]

        # Pivot to matrix form
        matrix = group_data.pivot_table(
            index=peptide_col,
            columns=sample_col,
            values=abundance_col,
            aggfunc='first'
        )

        # Ensure all samples present
        for s in samples:
            if s not in matrix.columns:
                matrix[s] = np.nan
        matrix = matrix[list(samples)]

        # Compute CV across peptides per sample (quality metric)
        # CV = std / mean (on linear scale, but we're on log scale, so use std directly)
        # On log scale, std approximates CV
        peptide_std = matrix.std(axis=0, skipna=True)
        matrix.mean(axis=0, skipna=True)
        n_valid = matrix.notna().sum(axis=0)

        # CV on log scale: std / |mean| (but mean could be ~0, so we use std as proxy)
        # For log2 data, std in log units ~ CV in linear units
        protein_cv_peptides[group_id] = peptide_std

        # Uncertainty: standard error of the mean (std / sqrt(n))
        protein_uncertainties[group_id] = peptide_std / np.sqrt(n_valid.replace(0, np.nan))

        # Build QC flags
        qc_flags = []
        if n_peptides < min_peptides:
            qc_flags.append(f"low_peptide_count({n_peptides})")
        if (n_valid < 2).any():
            qc_flags.append("single_peptide_in_sample")
        protein_qc_flags[group_id] = "; ".join(qc_flags) if qc_flags else None

        # Handle proteins with few peptides using fallback methods
        # (Per SPECIFICATION.md: always quantify, use appropriate method)
        if n_peptides == 1:
            # 1 peptide: Use peptide abundance directly
            protein_abundances[group_id] = matrix.iloc[0]
            # No polish result for single peptide

        elif n_peptides == 2:
            # 2 peptides: Use mean (no robust advantage with only 2)
            protein_abundances[group_id] = matrix.mean(axis=0)
            # No polish result for 2 peptides

        else:
            # >= 3 peptides: Apply requested rollup method
            if method == 'median_polish':
                result = tukey_median_polish(matrix)
                protein_abundances[group_id] = result.col_effects
                polish_results[group_id] = result

            elif method == 'topn':
                result = rollup_top_n(matrix, n=topn_n, selection=topn_selection)
                protein_abundances[group_id] = result.abundances
                topn_results[group_id] = result

            elif method == 'ibaq':
                if n_theoretical_peptides is None:
                    raise ValueError("n_theoretical_peptides required for iBAQ method")
                n_theor = n_theoretical_peptides.get(group_id, n_peptides)
                protein_abundances[group_id] = rollup_ibaq(matrix, n_theor)

            elif method == 'maxlfq':
                protein_abundances[group_id] = rollup_maxlfq(matrix)

            elif method == 'directlfq':
                protein_abundances[group_id] = rollup_directlfq(matrix)

            else:
                raise ValueError(f"Unknown rollup method: {method}")

    if skipped_groups:
        logger.info(f"Skipped {len(skipped_groups)} groups with 0 peptides")
        for gid, n in skipped_groups[:5]:
            logger.debug(f"  {gid}: {n} peptides")
        if len(skipped_groups) > 5:
            logger.debug(f"  ... and {len(skipped_groups) - 5} more")

    # Combine into DataFrame
    protein_df = pd.DataFrame(protein_abundances).T
    protein_df.index.name = 'protein_group_id'

    # Build CV peptides DataFrame (per-sample values)
    cv_df = pd.DataFrame(protein_cv_peptides).T
    cv_df.index.name = 'protein_group_id'

    # Build uncertainty DataFrame (per-sample values)
    unc_df = pd.DataFrame(protein_uncertainties).T
    unc_df.index.name = 'protein_group_id'

    # Add protein metadata
    group_metadata = {
        g.group_id: {
            'leading_protein': g.leading_protein,
            'leading_name': g.leading_protein_name,
            'n_peptides': g.n_peptides,
            'n_unique_peptides': g.n_unique_peptides,
            'qc_flag': protein_qc_flags.get(g.group_id),
        }
        for g in protein_groups
    }

    for col in ['leading_protein', 'leading_name', 'n_peptides', 'n_unique_peptides', 'qc_flag']:
        protein_df[col] = protein_df.index.map(
            lambda x, c=col: group_metadata.get(x, {}).get(c, np.nan)
        )

    # Add low_confidence flag for proteins with < min_peptides
    protein_df['low_confidence'] = protein_df['n_peptides'] < min_peptides

    # Log summary of low-confidence proteins
    n_low_conf = protein_df['low_confidence'].sum()
    if n_low_conf > 0:
        logger.info(
            f"  {n_low_conf} proteins have < {min_peptides} peptides "
            f"(flagged as low_confidence)"
        )

    # Attach CV and uncertainty DataFrames as attributes for downstream use
    # These can be merged into long format by the caller when constructing output
    protein_df.attrs['cv_peptides'] = cv_df
    protein_df.attrs['uncertainties'] = unc_df

    logger.info(f"Rolled up to {len(protein_df)} proteins")

    return protein_df, polish_results, topn_results


def flag_outlier_peptides(
    polish_results: dict[str, MedianPolishResult],
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Identify peptides that are consistent outliers in median polish residuals.

    A peptide is flagged if its residuals are frequently large across samples.

    Args:
        polish_results: Dict of MedianPolishResult per protein group
        threshold: Number of MADs to consider outlier

    Returns:
        DataFrame with:
        - protein_group_id
        - peptide
        - mad_score: Median absolute deviation score
        - outlier_fraction: Fraction of samples where peptide is outlier
        - is_outlier: Boolean flag

    """
    rows = []

    for group_id, result in polish_results.items():
        residuals = result.residuals

        # Calculate MAD per peptide across samples
        peptide_mads = residuals.apply(
            lambda row: np.nanmedian(np.abs(row - np.nanmedian(row))),
            axis=1
        )

        # Overall MAD for this protein
        all_residuals = residuals.values.flatten()
        all_residuals = all_residuals[~np.isnan(all_residuals)]
        if len(all_residuals) == 0:
            continue

        overall_mad = np.median(np.abs(all_residuals - np.median(all_residuals)))
        if overall_mad == 0:
            overall_mad = 1.0  # Avoid division by zero

        for peptide in residuals.index:
            pep_residuals = residuals.loc[peptide].dropna()
            if len(pep_residuals) == 0:
                continue

            # How many samples is this peptide an outlier?
            outlier_count = (np.abs(pep_residuals) > threshold * overall_mad).sum()
            outlier_fraction = outlier_count / len(pep_residuals)

            # MAD score for this peptide
            mad_score = peptide_mads[peptide] / overall_mad

            rows.append({
                'protein_group_id': group_id,
                'peptide': peptide,
                'mad_score': mad_score,
                'outlier_fraction': outlier_fraction,
                'is_outlier': outlier_fraction > 0.5 or mad_score > threshold,
            })

    return pd.DataFrame(rows)


def extract_peptide_residuals(
    polish_results: dict[str, MedianPolishResult],
) -> pd.DataFrame:
    """Extract peptide residuals from median polish results in long format.

    This function extracts the raw residuals from protein-level median polish
    for output to parquet files. Users can apply their own outlier thresholds
    downstream.

    Following Plubell et al. 2022 (doi:10.1021/acs.jproteome.1c00894), peptides
    with large residuals should not be discarded - they may indicate biologically
    interesting proteoform variation.

    Args:
        polish_results: Dict of protein_group_id -> MedianPolishResult

    Returns:
        DataFrame in long format with columns:
        - protein_group_id: Protein group identifier
        - peptide: Peptide identifier
        - replicate_name: Sample identifier
        - residual: Raw residual from median polish
        - row_effect: Peptide ionization effect (α_i)

        Plus summary statistics per peptide across samples:
        - residual_mean: Mean residual for this peptide
        - residual_std: Standard deviation of residuals
        - residual_mad: Median absolute deviation (robust)
        - residual_max_abs: Maximum absolute residual

    """
    rows = []

    for group_id, result in polish_results.items():
        # Get per-row summary statistics
        row_summary = result.get_row_residual_summary()

        # Melt residuals to long format
        residuals_long = result.residuals.reset_index().melt(
            id_vars=[result.residuals.index.name or 'index'],
            var_name='replicate_name',
            value_name='residual'
        )
        residuals_long.rename(columns={result.residuals.index.name or 'index': 'peptide'}, inplace=True)

        # Add protein group and row effects
        residuals_long['protein_group_id'] = group_id
        residuals_long['row_effect'] = residuals_long['peptide'].map(result.row_effects)

        # Add summary statistics
        for col in ['residual_mean', 'residual_std', 'residual_mad', 'residual_max_abs']:
            residuals_long[col] = residuals_long['peptide'].map(row_summary[col])

        rows.append(residuals_long)

    if not rows:
        return pd.DataFrame(columns=[
            'protein_group_id', 'peptide', 'replicate_name', 'residual',
            'row_effect', 'residual_mean', 'residual_std', 'residual_mad', 'residual_max_abs'
        ])

    return pd.concat(rows, ignore_index=True)


def extract_transition_residuals(
    transition_rollup_result,  # TransitionRollupResult
) -> Optional[pd.DataFrame]:
    """Extract transition residuals from transition rollup results in long format.

    This function extracts the raw residuals from transition-to-peptide median
    polish for output to parquet files.

    Args:
        transition_rollup_result: TransitionRollupResult with median_polish_results

    Returns:
        DataFrame in long format with columns:
        - peptide: Peptide identifier
        - transition: Transition identifier
        - replicate_name: Sample identifier
        - residual: Raw residual from median polish
        - row_effect: Transition ionization effect
        - residual_mean, residual_std, residual_mad, residual_max_abs: Summary stats

        Returns None if median_polish_results is not available (e.g., when
        quality_weighted method was used).

    """
    if transition_rollup_result.median_polish_results is None:
        return None

    rows = []

    for peptide, result in transition_rollup_result.median_polish_results.items():
        # Get per-row summary statistics
        row_summary = result.get_row_residual_summary()

        # Melt residuals to long format
        residuals_long = result.residuals.reset_index().melt(
            id_vars=[result.residuals.index.name or 'index'],
            var_name='replicate_name',
            value_name='residual'
        )
        residuals_long.rename(columns={result.residuals.index.name or 'index': 'transition'}, inplace=True)

        # Add peptide and row effects
        residuals_long['peptide'] = peptide
        residuals_long['row_effect'] = residuals_long['transition'].map(result.row_effects)

        # Add summary statistics
        for col in ['residual_mean', 'residual_std', 'residual_mad', 'residual_max_abs']:
            residuals_long[col] = residuals_long['transition'].map(row_summary[col])

        rows.append(residuals_long)

    if not rows:
        return None

    return pd.concat(rows, ignore_index=True)


def extract_quality_weights(
    rollup_details: dict[str, AggregationResult],
) -> Optional[pd.DataFrame]:
    """Extract quality weights from transition rollup results in long format.

    When using quality_weighted transition rollup, this function extracts
    the weights applied to each transition for each sample. Useful for
    understanding which transitions contributed most to the peptide abundance.

    Args:
        rollup_details: Dict of precursor_id -> AggregationResult from
                       rollup_transitions_to_peptides() when method='quality_weighted'

    Returns:
        DataFrame in long format with columns:
        - peptide: Peptide/precursor identifier
        - transition: Transition identifier
        - replicate_name: Sample identifier
        - weight: The weight applied to this transition for this sample
        - n_signals_used: Total number of transitions used for this sample

        Returns None if no weights are available.

    """
    rows = []

    for precursor, result in rollup_details.items():
        if not isinstance(result, AggregationResult):
            continue
        if result.weights_used is None:
            continue

        # weights_used is a DataFrame: transitions x samples
        weights_df = result.weights_used

        # Melt to long format
        weights_long = weights_df.reset_index().melt(
            id_vars=[weights_df.index.name or 'index'],
            var_name='replicate_name',
            value_name='weight'
        )
        weights_long.rename(
            columns={weights_df.index.name or 'index': 'transition'},
            inplace=True
        )

        # Add peptide and n_signals_used
        weights_long['peptide'] = precursor
        weights_long['n_signals_used'] = weights_long['replicate_name'].map(result.n_signals_used)

        rows.append(weights_long)

    if not rows:
        return None

    return pd.concat(rows, ignore_index=True)


def extract_topn_selections(
    topn_results: dict[str, TopNResult],
) -> pd.DataFrame:
    """Extract top-N peptide selections for each protein group.

    When using top-N protein rollup, this function extracts which peptides
    were selected for each protein group. This is important for reproducibility
    and understanding which peptides drove the protein quantification.

    Args:
        topn_results: Dict of protein_group_id -> TopNResult from
                     rollup_to_proteins() when method='topn'

    Returns:
        DataFrame with columns:
        - protein_group_id: Protein group identifier
        - peptide: Peptide identifier that was selected
        - selection_rank: Rank of this peptide (1 = first selected)
        - selection_method: Method used ('median_abundance' or 'frequency')
        - n_available: Total peptides available for this protein
        - n_selected: Number of peptides actually selected

    """
    rows = []

    for group_id, result in topn_results.items():
        for rank, peptide in enumerate(result.selected_peptides, start=1):
            rows.append({
                'protein_group_id': group_id,
                'peptide': peptide,
                'selection_rank': rank,
                'selection_method': result.selection_method,
                'n_available': result.n_available,
                'n_selected': len(result.selected_peptides),
            })

    if not rows:
        return pd.DataFrame(columns=[
            'protein_group_id', 'peptide', 'selection_rank',
            'selection_method', 'n_available', 'n_selected'
        ])

    return pd.DataFrame(rows)


# ============================================================================
# Protein-Level Batch Correction
# ============================================================================

@dataclass
class ProteinBatchCorrectionResult:
    """Result of protein-level batch correction.

    Attributes:
        corrected_data: DataFrame with batch-corrected protein abundances
        evaluation: BatchCorrectionEvaluation metrics (if reference/QC available)
        used_fallback: Whether fallback to uncorrected data was used
        method_log: List of processing steps

    """

    corrected_data: pd.DataFrame
    evaluation: Optional[BatchCorrectionEvaluation]
    used_fallback: bool
    method_log: list[str]


def batch_correct_proteins(
    protein_data: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    sample_col: str = 'replicate_name',
    batch_col: str = 'batch',
    sample_type_col: str = 'sample_type',
    reference_type: str = 'reference',
    qc_type: str = 'qc',
    par_prior: bool = True,
    mean_only: bool = False,
    evaluate: bool = True,
    fallback_on_failure: bool = True,
) -> ProteinBatchCorrectionResult:
    """Apply ComBat batch correction to protein-level abundances.

    This implements Step 5b from the PRISM specification: batch correction
    at the protein level AFTER peptide→protein rollup.

    The protein-level batch correction operates on the protein × sample matrix
    produced by rollup_to_proteins(). It uses reference and QC samples for
    QC evaluation, with automatic fallback if correction degrades data quality.

    Args:
        protein_data: DataFrame with proteins as rows, samples as columns
            (wide format from rollup_to_proteins)
        sample_metadata: DataFrame with sample annotations including batch
        sample_col: Column in metadata with sample identifiers
        batch_col: Column in metadata with batch labels
        sample_type_col: Column in metadata with sample types
        reference_type: Value indicating reference samples
        qc_type: Value indicating QC samples
        par_prior: Use parametric empirical Bayes (recommended)
        mean_only: Only correct location effects, not scale
        evaluate: Whether to evaluate correction using reference/QC
        fallback_on_failure: If True, revert to uncorrected data on QC failure

    Returns:
        ProteinBatchCorrectionResult with corrected data and diagnostics

    Example:
        >>> # After rollup_to_proteins
        >>> protein_df, polish_results = rollup_to_proteins(peptide_data, groups)
        >>>
        >>> # Apply protein-level batch correction
        >>> result = batch_correct_proteins(
        ...     protein_df,
        ...     sample_metadata,
        ...     batch_col='batch'
        ... )
        >>> corrected_proteins = result.corrected_data

    """
    method_log = []

    # Extract abundance columns (exclude metadata columns)
    metadata_cols = ['leading_protein', 'leading_name', 'n_peptides',
                     'n_unique_peptides', 'protein_group_id']
    sample_cols = [c for c in protein_data.columns if c not in metadata_cols]

    # Get abundance matrix (proteins × samples)
    abundance_matrix = protein_data[sample_cols].copy()

    # Map samples to batches
    sample_to_batch = dict(zip(
        sample_metadata[sample_col],
        sample_metadata[batch_col]
    ))
    batch_labels = [sample_to_batch.get(s) for s in sample_cols]

    # Check we have valid batches
    if None in batch_labels:
        missing = [s for s, b in zip(sample_cols, batch_labels) if b is None]
        logger.warning(f"Samples missing batch info: {missing[:5]}...")
        method_log.append(f"WARNING: {len(missing)} samples missing batch info")

    n_batches = len(set(b for b in batch_labels if b is not None))
    logger.info(f"Applying protein-level batch correction across {n_batches} batches")
    method_log.append(f"Protein-level batch correction: {n_batches} batches")

    if n_batches < 2:
        logger.warning("Only one batch - skipping batch correction")
        method_log.append("Skipped: only one batch")
        return ProteinBatchCorrectionResult(
            corrected_data=protein_data.copy(),
            evaluation=None,
            used_fallback=False,
            method_log=method_log,
        )

    # Apply ComBat to protein matrix (proteins × samples)
    try:
        corrected_matrix = combat(
            abundance_matrix.values,
            np.array(batch_labels),
            par_prior=par_prior,
            mean_only=mean_only,
        )
        corrected_df = pd.DataFrame(
            corrected_matrix,
            index=abundance_matrix.index,
            columns=abundance_matrix.columns,
        )
        method_log.append(f"ComBat applied (par_prior={par_prior}, mean_only={mean_only})")

    except Exception as e:
        logger.error(f"Protein-level batch correction failed: {e}")
        method_log.append(f"ComBat FAILED: {e}")
        return ProteinBatchCorrectionResult(
            corrected_data=protein_data.copy(),
            evaluation=None,
            used_fallback=True,
            method_log=method_log,
        )

    # Evaluate using reference and QC samples
    evaluation = None
    used_fallback = False

    if evaluate:
        # Get sample type mapping
        sample_to_type = dict(zip(
            sample_metadata[sample_col],
            sample_metadata[sample_type_col]
        ))

        reference_cols = [s for s in sample_cols
                         if sample_to_type.get(s) == reference_type]
        qc_cols = [s for s in sample_cols
                    if sample_to_type.get(s) == qc_type]

        if len(reference_cols) >= 2 and len(qc_cols) >= 2:
            # Calculate CVs before and after
            def calc_cv(df, cols):
                subset = df[cols]
                linear = np.power(2, subset)  # Convert from log2
                cv_per_protein = linear.std(axis=1) / linear.mean(axis=1)
                return float(cv_per_protein.median())

            ref_cv_before = calc_cv(abundance_matrix, reference_cols)
            ref_cv_after = calc_cv(corrected_df, reference_cols)
            qc_cv_before = calc_cv(abundance_matrix, qc_cols)
            qc_cv_after = calc_cv(corrected_df, qc_cols)

            ref_improvement = (ref_cv_before - ref_cv_after) / ref_cv_before
            qc_improvement = (qc_cv_before - qc_cv_after) / qc_cv_before

            if qc_improvement > 0:
                overfitting_ratio = ref_improvement / qc_improvement
            elif ref_improvement > 0:
                overfitting_ratio = float('inf')
            else:
                overfitting_ratio = 1.0

            # Calculate batch variance
            batch_var_before = np.var([
                abundance_matrix[sample_cols].mean().values
            ])
            batch_var_after = np.var([
                corrected_df[sample_cols].mean().values
            ])

            # Determine pass/fail
            warnings = []
            passed = True

            if qc_cv_after > qc_cv_before * 1.1:
                passed = False
                warnings.append(
                    f"QC CV increased: {qc_cv_before:.3f} -> {qc_cv_after:.3f}"
                )

            if overfitting_ratio > 2.0:
                passed = False
                warnings.append(
                    f"Possible overfitting: ref improved {ref_improvement:.1%}, "
                    f"QC only {qc_improvement:.1%}"
                )

            evaluation = BatchCorrectionEvaluation(
                reference_cv_before=ref_cv_before,
                reference_cv_after=ref_cv_after,
                qc_cv_before=qc_cv_before,
                qc_cv_after=qc_cv_after,
                reference_improvement=ref_improvement,
                qc_improvement=qc_improvement,
                overfitting_ratio=overfitting_ratio,
                batch_variance_before=batch_var_before,
                batch_variance_after=batch_var_after,
                passed=passed,
                warnings=warnings,
            )

            method_log.append(
                f"Evaluation: ref CV {ref_cv_before:.3f} -> {ref_cv_after:.3f}, "
                f"QC CV {qc_cv_before:.3f} -> {qc_cv_after:.3f}"
            )

            # Handle fallback
            if fallback_on_failure and not passed:
                logger.warning(
                    "Protein-level batch correction failed QC - using uncorrected data"
                )
                for w in warnings:
                    logger.warning(f"  - {w}")

                corrected_df = abundance_matrix.copy()
                used_fallback = True
                evaluation.warnings.append(
                    "FALLBACK: Using uncorrected protein data due to QC failure"
                )
                method_log.append("FALLBACK to uncorrected data")
            elif passed:
                logger.info("Protein-level batch correction passed QC")
                method_log.append("QC PASSED")
            else:
                logger.warning("Protein-level batch correction failed QC but keeping corrected data")
                method_log.append("QC FAILED (keeping corrected data)")
        else:
            logger.warning(
                f"Cannot evaluate: {len(reference_cols)} reference, {len(qc_cols)} QC samples"
            )
            method_log.append(
                "Evaluation skipped: need >=2 reference and QC samples"
            )

    # Reconstruct full DataFrame with metadata
    result_df = protein_data.copy()
    for col in sample_cols:
        result_df[col] = corrected_df[col]

    return ProteinBatchCorrectionResult(
        corrected_data=result_df,
        evaluation=evaluation,
        used_fallback=used_fallback,
        method_log=method_log,
    )


def protein_output_pipeline(
    peptide_data: pd.DataFrame,
    protein_groups: list,
    sample_metadata: pd.DataFrame,
    abundance_col: str = 'abundance',
    sample_col: str = 'replicate_name',
    peptide_col: str = 'peptide_modified',
    batch_col: str = 'batch',
    sample_type_col: str = 'sample_type',
    rollup_method: str = 'median_polish',
    shared_peptide_handling: str = 'all_groups',
    batch_correction: bool = True,
    batch_correction_params: Optional[dict] = None,
    min_peptides: int = 3,
) -> tuple[pd.DataFrame, dict, ProteinBatchCorrectionResult]:
    """Complete protein output pipeline: rollup then batch correction.

    This implements the Protein Output Arm from the PRISM specification:
    1. Peptide → Protein rollup (median polish or other method)
    2. Protein-level batch correction (ComBat with QC evaluation)

    Args:
        peptide_data: Normalized peptide data in long format
        protein_groups: List of ProteinGroup objects from parsimony
        sample_metadata: Sample annotations including batch info
        abundance_col: Column with peptide abundances
        sample_col: Column with sample identifiers
        peptide_col: Column with peptide identifiers
        batch_col: Column with batch labels
        sample_type_col: Column with sample types (reference/QC/experimental)
        rollup_method: Method for peptide→protein rollup
        shared_peptide_handling: How to handle shared peptides
        batch_correction: Whether to apply protein-level batch correction
        batch_correction_params: Parameters for ComBat
        min_peptides: Minimum peptides for full-confidence quantification.
            Proteins with fewer peptides are flagged as low_confidence=True.

    Returns:
        Tuple of:
        - DataFrame with final protein abundances
        - Dict of MedianPolishResult per protein (if method='median_polish')
        - ProteinBatchCorrectionResult (or None if batch_correction=False)

    """
    # Step 1: Peptide → Protein rollup
    logger.info("Step 1: Peptide to protein rollup")
    protein_df, polish_results, topn_results = rollup_to_proteins(
        peptide_data,
        protein_groups,
        abundance_col=abundance_col,
        sample_col=sample_col,
        peptide_col=peptide_col,
        method=rollup_method,
        shared_peptide_handling=shared_peptide_handling,
        min_peptides=min_peptides,
    )

    # Step 2: Protein-level batch correction
    batch_result = None
    if batch_correction:
        logger.info("Step 2: Protein-level batch correction")
        params = batch_correction_params or {}

        batch_result = batch_correct_proteins(
            protein_df,
            sample_metadata,
            sample_col=sample_col,
            batch_col=batch_col,
            sample_type_col=sample_type_col,
            par_prior=params.get('par_prior', True),
            mean_only=params.get('mean_only', False),
            evaluate=params.get('evaluate', True),
            fallback_on_failure=params.get('fallback_on_failure', True),
        )
        protein_df = batch_result.corrected_data

    return protein_df, polish_results, batch_result

