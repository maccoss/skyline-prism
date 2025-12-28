"""Transition rollup module for quality-weighted aggregation.

This module handles Stage 0 of the PRISM pipeline: rolling up individual
transition intensities to peptide-level quantities using quality-weighted
aggregation based on Skyline's per-transition quality metrics.

Key concepts:
- Each transition gets a single weight based on intensity-weighted quality
- Poor-quality transitions (low shape correlation, not coeluting) are downweighted
- Variance model parameters can be learned from reference samples
- When using median_polish, transition-level residuals are captured for outlier analysis
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
class VarianceModelParams:
    """Parameters for the signal variance model (legacy).

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

    NOTE: This is the legacy variance model. For the new quality-weighted rollup,
    use QualityWeightParams instead.
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

    # Coelution penalty (added to variance when not coeluting)
    coelution_penalty: float = 10.0


@dataclass
class QualityWeightParams:
    """Parameters for quality-weighted transition aggregation.

    Weight model based on three quality signals:
        w_t = (intensity_t)^α × (shape_corr_t)^β × (1/shape_corr_cv_t)^δ × exp(-γ × residual_t²)

    Where:
    - intensity_t: Mean intensity of transition across samples (sqrt for default)
    - shape_corr_t: Mean or median shape correlation across samples
    - shape_corr_cv_t: CV of shape correlation (high CV = unreliable)
    - residual_t: Residual from preliminary median polish (large = outlier)

    The exponents control how much each factor influences the weight:
    - α = 0.5: sqrt(intensity) weighting (default)
    - β > 0: Higher shape correlation = higher weight
    - δ > 0: Lower CV of shape correlation = higher weight
    - γ > 0: Larger residuals = lower weight (exponential decay)

    Parameters are learned from reference samples by minimizing CV.
    Validated on pool samples to ensure generalization.
    Falls back to raw sum if quality-weighted rollup doesn't improve.
    """

    # Intensity exponent: w ∝ intensity^alpha
    # 0.5 = sqrt weighting (default), 0 = ignore intensity, 1 = linear
    alpha: float = 0.5

    # Shape correlation exponent: w ∝ shape_corr^beta
    # Higher = more weight on high-correlation transitions
    beta: float = 1.0

    # Residual penalty: w ∝ exp(-gamma * residual^2)
    # Higher = more penalty for outlier transitions
    gamma: float = 1.0

    # Shape correlation CV exponent: w ∝ (1/shape_corr_cv)^delta
    # Higher = more penalty for variable shape correlation
    delta: float = 0.5

    # Aggregation method for shape correlation: 'mean' or 'median'
    shape_corr_agg: str = "median"

    # Fallback method if quality-weighted doesn't improve: 'sum' or 'median_polish'
    fallback_method: str = "sum"

    # Minimum improvement in pool CV required to use learned weights (percent)
    # If improvement < this threshold, fall back to fallback_method
    min_improvement_pct: float = 5.0


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

    peptide_abundances: pd.DataFrame  # Peptide × sample matrix (log2 if log_transform)
    peptide_uncertainties: pd.DataFrame  # Uncertainty estimates (log2 scale)
    transition_weights: pd.DataFrame  # Weights used per transition (quality_weighted)
    n_transitions_used: pd.DataFrame  # Number of transitions per peptide/sample
    variance_model: VarianceModelParams  # Model parameters used
    # Median polish results per peptide (when method='median_polish')
    # Keys are peptide identifiers, values are MedianPolishResult objects
    median_polish_results: dict[str, MedianPolishResult] | None = None


def compute_transition_variance(
    intensity: np.ndarray,
    shape_correlation: np.ndarray,
    coeluting: np.ndarray,
    params: VarianceModelParams,
) -> np.ndarray:
    """Compute variance estimate for each transition based on intensity and quality.

    Args:
        intensity: Transition intensities (linear scale)
        shape_correlation: Correlation with median transition profile (0-1)
        coeluting: Boolean indicating apex within integration boundaries
        params: Variance model parameters

    Returns:
        Estimated variance for each transition

    """
    # Base variance from intensity model
    variance = params.alpha * intensity + params.beta * intensity**2 + params.gamma

    # Quality penalty based on shape correlation
    # Poor correlation (low value) increases variance
    # Use (1 - corr)^exponent as penalty multiplier
    shape_penalty = (1.0 - np.clip(shape_correlation, 0, 1)) ** params.shape_corr_exponent
    variance += params.delta * shape_penalty * intensity**2

    # Coelution penalty
    coelution_penalty = np.where(coeluting, 0.0, params.coelution_penalty * intensity)
    variance += coelution_penalty

    return variance


def compute_transition_weights(
    intensities: pd.DataFrame,
    shape_correlations: pd.DataFrame,
    coeluting: pd.DataFrame,
    params: VarianceModelParams,
) -> pd.DataFrame:
    """Compute per-transition weights using intensity-weighted quality averaging.

    Key principle: A single weight per transition (not per replicate) using
    intensity-weighted averaging of quality metrics. This ensures consistent
    treatment across all replicates.

    Args:
        intensities: Transition × replicate intensity matrix
        shape_correlations: Transition × replicate shape correlation matrix
        coeluting: Transition × replicate coelution boolean matrix
        params: Variance model parameters

    Returns:
        DataFrame with one weight per transition

    """
    # For each transition, compute intensity-weighted average quality
    # High-abundance samples contribute more to quality assessment

    # Normalize intensities for weighting (per transition across replicates)
    intensity_weights = intensities.div(intensities.sum(axis=1), axis=0)
    intensity_weights = intensity_weights.fillna(0)

    # Compute weighted average shape correlation per transition
    weighted_corr = (shape_correlations * intensity_weights).sum(axis=1)

    # Compute weighted coelution rate per transition
    weighted_coelution = (coeluting.astype(float) * intensity_weights).sum(axis=1)

    # Mean intensity per transition (for variance model)
    mean_intensity = intensities.mean(axis=1)

    # Compute variance for each transition using aggregated quality metrics
    variance = compute_transition_variance(
        intensity=mean_intensity.values,
        shape_correlation=weighted_corr.values,
        coeluting=weighted_coelution.values > 0.5,  # Majority coeluting
        params=params,
    )

    # Weight = inverse variance
    weights = 1.0 / np.maximum(variance, 1e-10)

    # Normalize weights to sum to 1 per peptide
    # (normalization happens during aggregation)

    return pd.Series(weights, index=intensities.index, name="weight")


# ============================================================================
# New Quality-Weighted Rollup (v2) - uses intensity, shape_corr, and residuals
# ============================================================================

@dataclass
class TransitionQualityMetrics:
    """Per-transition quality metrics for weight calculation.

    Computed once per peptide, then used for weight calculation.
    """

    # Mean intensity across samples (linear scale)
    mean_intensity: pd.Series

    # Shape correlation aggregated across samples (mean or median)
    shape_corr: pd.Series

    # CV of shape correlation across samples (higher = less reliable)
    shape_corr_cv: pd.Series

    # Residuals from preliminary median polish (large = outlier)
    residuals_mad: pd.Series  # Median absolute deviation of residuals per transition


def compute_transition_quality_metrics(
    intensity_matrix: pd.DataFrame,
    shape_corr_matrix: pd.DataFrame,
    shape_corr_agg: str = "median",
) -> TransitionQualityMetrics:
    """Compute quality metrics for each transition in a peptide.

    Args:
        intensity_matrix: Transition x sample intensity matrix (LINEAR scale)
        shape_corr_matrix: Transition x sample shape correlation matrix (0-1)
        shape_corr_agg: How to aggregate shape correlation ('mean' or 'median')

    Returns:
        TransitionQualityMetrics with per-transition quality measures
    """
    # Mean intensity per transition (linear scale)
    mean_intensity = intensity_matrix.mean(axis=1)

    # Shape correlation aggregated across samples
    if shape_corr_agg == "median":
        shape_corr = shape_corr_matrix.median(axis=1)
    else:
        shape_corr = shape_corr_matrix.mean(axis=1)

    # CV of shape correlation (std/mean)
    # High CV = shape correlation varies across samples = unreliable
    shape_corr_mean = shape_corr_matrix.mean(axis=1)
    shape_corr_std = shape_corr_matrix.std(axis=1)
    # Avoid division by zero; if mean is 0, CV is 0
    shape_corr_cv = np.where(
        shape_corr_mean.abs() > 0.001,
        shape_corr_std / shape_corr_mean.abs(),
        0.0
    )
    shape_corr_cv = pd.Series(shape_corr_cv, index=intensity_matrix.index)

    # Residuals will be computed separately after median polish
    # Initialize with zeros (will be updated after median polish)
    residuals_mad = pd.Series(0.0, index=intensity_matrix.index)

    return TransitionQualityMetrics(
        mean_intensity=mean_intensity,
        shape_corr=shape_corr,
        shape_corr_cv=shape_corr_cv,
        residuals_mad=residuals_mad,
    )


def compute_transition_residuals(
    intensity_matrix: pd.DataFrame,
) -> pd.Series:
    """Compute transition residuals using median polish.

    Runs a single median polish on the transition x sample matrix
    and returns the median absolute deviation (MAD) of residuals
    per transition. Large MAD = outlier transition to downweight.

    Args:
        intensity_matrix: Transition x sample matrix (LOG2 scale)

    Returns:
        Series with MAD of residuals per transition
    """
    from .rollup import tukey_median_polish

    if len(intensity_matrix) < 2:
        return pd.Series(0.0, index=intensity_matrix.index)

    try:
        result = tukey_median_polish(intensity_matrix)
        # Compute MAD of residuals per transition (across samples)
        residuals_mad = result.residuals.abs().median(axis=1)
        return residuals_mad
    except Exception:
        # If median polish fails, return zeros (no penalty)
        return pd.Series(0.0, index=intensity_matrix.index)


def compute_quality_weights(
    metrics: TransitionQualityMetrics,
    residuals_mad: pd.Series,
    params: QualityWeightParams,
) -> pd.Series:
    """Compute quality-based weights for transitions.

    Weight formula:
        w_t = intensity^alpha * shape_corr^beta * cv_penalty^delta * exp(-gamma * residual^2)

    Args:
        metrics: Quality metrics per transition
        residuals_mad: MAD of residuals from median polish
        params: Weight model parameters

    Returns:
        Series of weights per transition (not normalized)
    """
    # Intensity contribution: intensity^alpha
    # Use max(intensity, 1) to avoid zero weights
    intensity_contrib = np.power(
        np.maximum(metrics.mean_intensity.values, 1.0),
        params.alpha
    )

    # Shape correlation contribution: shape_corr^beta
    # Clip to [0, 1] range
    shape_corr_contrib = np.power(
        np.clip(metrics.shape_corr.values, 0.001, 1.0),
        params.beta
    )

    # Shape correlation CV penalty: (1/(1+cv))^delta
    # Higher CV = lower weight
    cv_penalty = np.power(
        1.0 / (1.0 + metrics.shape_corr_cv.values),
        params.delta
    )

    # Residual penalty: exp(-gamma * residual_mad^2)
    # Large residuals = exponential decay of weight
    residual_penalty = np.exp(-params.gamma * np.power(residuals_mad.values, 2))

    # Combined weight
    weights = intensity_contrib * shape_corr_contrib * cv_penalty * residual_penalty

    return pd.Series(weights, index=metrics.mean_intensity.index, name="quality_weight")


def aggregate_transitions_quality_weighted(
    intensity_matrix: pd.DataFrame,
    weights: pd.Series,
    min_transitions: int = 3,
) -> tuple[pd.Series, pd.Series, int]:
    """Aggregate transition intensities using quality-weighted sum.

    Performs a WEIGHTED SUM in linear space (like sum method), not a weighted mean.
    The weights modulate each transition's contribution to the total signal.

    Algorithm:
    1. Convert log2 intensities to linear
    2. Multiply each transition's intensity by its weight
    3. Sum weighted intensities
    4. Convert back to log2

    This is equivalent to: log2(sum(w_i * 2^x_i))
    where w_i are quality weights and x_i are log2 intensities.

    Args:
        intensity_matrix: Transition x sample matrix (LOG2 scale)
        weights: Per-transition quality weights (NOT normalized)
        min_transitions: Minimum transitions required

    Returns:
        Tuple of (peptide abundances in LOG2, uncertainties, n_transitions_used)
    """
    n_transitions = len(intensity_matrix)

    if n_transitions < min_transitions:
        return (
            pd.Series(np.nan, index=intensity_matrix.columns),
            pd.Series(np.nan, index=intensity_matrix.columns),
            0,
        )

    # Normalize weights to sum to n_transitions
    # This preserves the sum behavior: if all weights equal, result = sum
    weight_sum = weights.sum()
    if weight_sum <= 0:
        # All weights are zero - fall back to equal weights
        normalized_weights = pd.Series(1.0, index=weights.index)
    else:
        # Scale so weights sum to n_transitions (preserves sum magnitude)
        normalized_weights = weights * (n_transitions / weight_sum)

    # Convert to linear space
    linear_matrix = 2 ** intensity_matrix

    abundances = pd.Series(index=intensity_matrix.columns, dtype=float)
    uncertainties = pd.Series(index=intensity_matrix.columns, dtype=float)

    for sample in intensity_matrix.columns:
        linear_values = linear_matrix[sample]
        valid = ~linear_values.isna()

        if valid.sum() < min_transitions:
            abundances[sample] = np.nan
            uncertainties[sample] = np.nan
            continue

        valid_weights = normalized_weights[valid]
        valid_linear = linear_values[valid]

        # Weighted sum in linear space
        weighted_sum = (valid_weights * valid_linear).sum()

        # Convert back to log2 (clip to avoid log(0))
        abundances[sample] = np.log2(max(weighted_sum, 1.0))

        # Uncertainty: use CV of weighted contributions
        contributions = valid_weights * valid_linear
        if len(contributions) > 1 and contributions.sum() > 0:
            cv = contributions.std() / contributions.mean()
            uncertainties[sample] = cv
        else:
            uncertainties[sample] = np.nan

    return abundances, uncertainties, n_transitions


def rollup_peptide_quality_weighted(
    intensity_matrix: pd.DataFrame,
    shape_corr_matrix: pd.DataFrame,
    params: QualityWeightParams,
    min_transitions: int = 3,
) -> tuple[pd.Series, pd.Series, pd.Series, int]:
    """Roll up transitions to peptide using quality weights.

    This is the complete quality-weighted rollup for a single peptide.
    Steps:
    1. Compute quality metrics (intensity, shape_corr, shape_corr_cv)
    2. If gamma > 0, run median polish to get residuals (expensive)
    3. Compute quality weights
    4. Aggregate transitions with weights

    Args:
        intensity_matrix: Transition x sample matrix (LOG2 scale)
        shape_corr_matrix: Transition x sample shape correlation (0-1)
        params: Quality weight parameters
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

    # Step 1: Compute quality metrics using LINEAR intensities
    linear_intensity = 2 ** intensity_matrix  # Convert back to linear
    metrics = compute_transition_quality_metrics(
        linear_intensity,
        shape_corr_matrix,
        shape_corr_agg=params.shape_corr_agg,
    )

    # Step 2: Compute residuals from median polish (only if gamma > 0)
    # This is the expensive step - skip if not needed
    if params.gamma > 0:
        residuals_mad = compute_transition_residuals(intensity_matrix)
    else:
        # No residual penalty - use zeros
        residuals_mad = pd.Series(0.0, index=intensity_matrix.index)

    # Step 3: Compute quality weights
    weights = compute_quality_weights(metrics, residuals_mad, params)

    # Step 4: Aggregate with weights
    abundances, uncertainties, n_used = aggregate_transitions_quality_weighted(
        intensity_matrix, weights, min_transitions
    )

    return abundances, uncertainties, weights, n_used


def rollup_peptide_topn(
    intensity_matrix: pd.DataFrame,
    shape_corr_matrix: pd.DataFrame,
    n_transitions: int = 3,
    selection_method: str = "correlation",
    weighting: str = "sqrt",
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
    if weight_sum > 0:
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


def aggregate_transitions_weighted(
    intensities: pd.DataFrame,
    weights: pd.Series,
    min_transitions: int = 3,
) -> tuple[pd.Series, pd.Series, int]:
    """Aggregate transition intensities using quality-based weights.

    Args:
        intensities: Transition x replicate matrix (log2 scale)
        weights: Per-transition weights
        min_transitions: Minimum transitions required

    Returns:
        Tuple of (peptide abundances, uncertainties, n_transitions_used)

    """
    n_transitions = len(intensities)

    if n_transitions < min_transitions:
        # Not enough transitions
        return (
            pd.Series(np.nan, index=intensities.columns),
            pd.Series(np.nan, index=intensities.columns),
            0,
        )

    # Normalize weights
    weight_sum = weights.sum()
    if weight_sum <= 0:
        normalized_weights = pd.Series(1.0 / n_transitions, index=weights.index)
    else:
        normalized_weights = weights / weight_sum

    # Weighted average per sample
    abundances = pd.Series(index=intensities.columns, dtype=float)
    uncertainties = pd.Series(index=intensities.columns, dtype=float)

    for sample in intensities.columns:
        values = intensities[sample]
        valid = ~values.isna()

        if valid.sum() < min_transitions:
            abundances[sample] = np.nan
            uncertainties[sample] = np.nan
            continue

        valid_weights = normalized_weights[valid]
        valid_values = values[valid]

        # Re-normalize weights for valid values
        w = valid_weights / valid_weights.sum()

        # Weighted mean
        abundances[sample] = (w * valid_values).sum()

        # Uncertainty: weighted standard error
        # Using weighted variance formula
        weighted_var = (w * (valid_values - abundances[sample]) ** 2).sum()
        # Effective sample size
        n_eff = 1.0 / (w**2).sum()
        uncertainties[sample] = np.sqrt(weighted_var / n_eff)

    return abundances, uncertainties, n_transitions


def rollup_transitions_to_peptides(
    data: pd.DataFrame,
    peptide_col: str = "peptide_modified",
    transition_col: str = "fragment_ion",
    sample_col: str = "replicate_name",
    abundance_col: str = "area",
    shape_corr_col: str = "shape_correlation",
    coeluting_col: str = "coeluting",
    method: str = "quality_weighted",
    params: VarianceModelParams | None = None,
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
        coeluting_col: Column with coelution boolean
        method: Rollup method ('quality_weighted', 'median_polish', 'sum')
        params: Variance model parameters (uses defaults if None)
        min_transitions: Minimum transitions required per peptide
        log_transform: Whether to log2 transform intensities (default: True)

    Returns:
        TransitionRollupResult with peptide abundances (LOG2 scale if log_transform)

    """
    if params is None:
        params = VarianceModelParams()

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

        if method == "quality_weighted":
            # Get quality metrics
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

            if coeluting_col in pep_data.columns:
                coeluting_matrix = pep_data.pivot_table(
                    index=transition_col,
                    columns=sample_col,
                    values=coeluting_col,
                    aggfunc="first",
                )
                for sample in samples:
                    if sample not in coeluting_matrix.columns:
                        coeluting_matrix[sample] = True
                coeluting_matrix = coeluting_matrix[samples].fillna(True)
            else:
                coeluting_matrix = pd.DataFrame(
                    True, index=intensity_matrix.index, columns=intensity_matrix.columns
                )

            # Compute weights (one per transition)
            weights = compute_transition_weights(
                2**intensity_matrix if log_transform else intensity_matrix,  # Linear for variance
                shape_corr_matrix,
                coeluting_matrix,
                params,
            )
            all_weights[peptide] = weights

            # Aggregate with weights
            abundances, uncertainties, n_used = aggregate_transitions_weighted(
                intensity_matrix, weights, min_transitions
            )

        elif method == "median_polish":
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
        variance_model=params,
        median_polish_results=all_median_polish_results if all_median_polish_results else None,
    )


def learn_variance_model(
    data: pd.DataFrame,
    reference_samples: list[str],
    peptide_col: str = "peptide_modified",
    transition_col: str = "fragment_ion",
    sample_col: str = "replicate_name",
    abundance_col: str = "area",
    shape_corr_col: str = "shape_correlation",
    coeluting_col: str = "coeluting",
    n_iterations: int = 100,
) -> VarianceModelParams:
    """Learn variance model parameters from reference samples.

    Uses reference replicates to optimize variance model parameters
    by minimizing CV across peptides. Reference samples should have
    only technical variation (no biological differences), making them
    ideal for learning the noise model.

    The learned parameters can then be used for quality-weighted
    aggregation of transitions to peptides.

    Args:
        data: DataFrame with transition-level data
        reference_samples: List of sample names that are reference samples
        peptide_col: Column identifying peptides
        transition_col: Column identifying transitions
        sample_col: Column identifying samples/replicates
        abundance_col: Column with transition intensities
        shape_corr_col: Column with shape correlation values
        coeluting_col: Column with coelution boolean
        n_iterations: Maximum optimization iterations

    Returns:
        Optimized VarianceModelParams

    """
    from scipy.optimize import minimize

    logger.info("Learning variance model parameters from reference samples")

    # Filter to reference samples only
    ref_data = data[data[sample_col].isin(reference_samples)].copy()
    n_ref_samples = ref_data[sample_col].nunique()

    if n_ref_samples < 2:
        logger.warning(
            f"Need at least 2 reference samples to learn parameters, got {n_ref_samples}. "
            "Using default parameters."
        )
        return VarianceModelParams()

    logger.info(f"  Using {n_ref_samples} reference samples for parameter learning")

    # Check which quality columns are available
    has_shape_corr = shape_corr_col in ref_data.columns
    has_coeluting = coeluting_col in ref_data.columns

    if not has_shape_corr:
        logger.warning(f"Shape correlation column '{shape_corr_col}' not found")
    if not has_coeluting:
        logger.warning(f"Coeluting column '{coeluting_col}' not found")

    def objective(params_array):
        """Objective: minimize median CV across peptides in reference."""
        # Ensure positive parameters via softplus or exp
        params = VarianceModelParams(
            alpha=max(0.01, params_array[0]),
            beta=max(0.0001, params_array[1]),
            gamma=max(1.0, params_array[2]),
            delta=max(0.01, params_array[3]) if has_shape_corr else 1.0,
            shape_corr_exponent=max(0.5, min(5.0, params_array[4])) if has_shape_corr else 2.0,
            coelution_penalty=max(1.0, params_array[5]) if has_coeluting else 10.0,
        )

        try:
            result = rollup_transitions_to_peptides(
                ref_data,
                peptide_col=peptide_col,
                transition_col=transition_col,
                sample_col=sample_col,
                abundance_col=abundance_col,
                shape_corr_col=shape_corr_col if has_shape_corr else None,
                coeluting_col=coeluting_col if has_coeluting else None,
                method="quality_weighted",
                params=params,
                min_transitions=3,
            )

            # Calculate CV across reference replicates for each peptide
            abundances = result.peptide_abundances
            means = abundances.mean(axis=1)
            stds = abundances.std(axis=1)

            # Filter out peptides with zero or near-zero mean
            valid = means.abs() > 0.01
            cvs = stds[valid] / means[valid].abs()

            median_cv = cvs.median()
            return median_cv if np.isfinite(median_cv) else 1.0

        except Exception as e:
            logger.debug(f"Optimization step failed: {e}")
            return 1.0

    # Initial parameters
    n_params = 6 if has_coeluting else (5 if has_shape_corr else 3)
    x0 = [1.0, 0.01, 100.0, 1.0, 2.0, 10.0][:n_params]

    # Bounds
    bounds = [
        (0.01, 10.0),     # alpha
        (0.0001, 0.5),    # beta
        (1.0, 10000.0),   # gamma
        (0.01, 10.0),     # delta
        (0.5, 5.0),       # shape_corr_exponent
        (1.0, 100.0),     # coelution_penalty
    ][:n_params]

    # Calculate initial CV
    initial_cv = objective(x0)
    logger.info(f"  Initial median CV: {initial_cv:.4f}")

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": n_iterations},
    )

    # Extract optimized parameters
    opt = result.x
    optimized = VarianceModelParams(
        alpha=max(0.01, opt[0]),
        beta=max(0.0001, opt[1]),
        gamma=max(1.0, opt[2]),
        delta=max(0.01, opt[3]) if len(opt) > 3 else 1.0,
        shape_corr_exponent=max(0.5, min(5.0, opt[4])) if len(opt) > 4 else 2.0,
        coelution_penalty=max(1.0, opt[5]) if len(opt) > 5 else 10.0,
    )

    final_cv = objective(list(opt) + [10.0] * (6 - len(opt)))  # Pad if needed

    logger.info("  Optimized parameters:")
    logger.info(f"    α (shot noise) = {optimized.alpha:.3f}")
    logger.info(f"    β (multiplicative) = {optimized.beta:.4f}")
    logger.info(f"    γ (additive floor) = {optimized.gamma:.1f}")
    if has_shape_corr:
        logger.info(f"    δ (quality penalty) = {optimized.delta:.3f}")
        logger.info(f"    shape_corr_exponent = {optimized.shape_corr_exponent:.2f}")
    if has_coeluting:
        logger.info(f"    coelution_penalty = {optimized.coelution_penalty:.1f}x")
    logger.info(f"  Median CV: {initial_cv:.4f} → {final_cv:.4f}")

    return optimized


# ============================================================================
# New Quality Weight Learning (v2)
# ============================================================================

@dataclass
class QualityWeightResult:
    """Result of quality weight learning and validation.

    Contains learned parameters, CV metrics, and decision on whether to use
    quality-weighted rollup or fall back.
    """

    params: QualityWeightParams
    reference_cv_before: float  # CV with equal weights (sum)
    reference_cv_after: float   # CV with learned weights
    pool_cv_before: float       # Validation CV with sum
    pool_cv_after: float        # Validation CV with learned weights
    use_quality_weights: bool   # Whether to use quality weights or fall back
    fallback_reason: str | None  # Reason for fallback if not using quality weights


def _compute_median_cv_linear(
    abundances: pd.DataFrame,
) -> float:
    """Compute median CV across features (peptides/proteins) on LINEAR scale.

    CRITICAL: CVs must always be calculated on linear scale data.
    Input is on LOG2 scale, so we convert to linear first.

    Args:
        abundances: Feature x sample matrix (LOG2 scale)

    Returns:
        Median CV as a decimal (not percentage)
    """
    # Convert to linear scale
    linear = 2 ** abundances

    # Calculate CV per feature (across samples)
    means = linear.mean(axis=1)
    stds = linear.std(axis=1)

    # Filter out features with near-zero mean
    valid = means > 1.0  # At least 1 in linear scale
    if valid.sum() == 0:
        return np.nan

    cvs = stds[valid] / means[valid]
    return cvs.median()


@dataclass
class _PrecomputedPeptideMetrics:
    """Pre-computed metrics for a single peptide for fast weight learning."""
    # Intensity matrix (log2 scale): n_transitions x n_samples
    intensity_log2: np.ndarray
    # Mean intensity per transition (linear scale): n_transitions
    mean_intensity_linear: np.ndarray
    # Shape correlation per transition (aggregated): n_transitions
    shape_corr: np.ndarray
    # CV of shape correlation per transition: n_transitions
    shape_corr_cv: np.ndarray
    # Residual MAD per transition: n_transitions
    residuals_mad: np.ndarray
    # Number of valid transitions
    n_transitions: int


def _precompute_peptide_metrics(
    data: pd.DataFrame,
    sample_list: list[str],
    peptide_col: str,
    transition_col: str,
    sample_col: str,
    abundance_col: str,
    shape_corr_col: str,
    shape_corr_agg: str = "median",
    min_transitions: int = 3,
) -> dict[str, _PrecomputedPeptideMetrics]:
    """Pre-compute quality metrics for all peptides (one-time cost).

    This enables fast weight learning by computing expensive operations once.

    Args:
        data: Transition-level DataFrame (LINEAR scale intensities)
        sample_list: Samples to include
        peptide_col, transition_col, sample_col, abundance_col, shape_corr_col: Column names
        shape_corr_agg: How to aggregate shape correlation ('mean' or 'median')
        min_transitions: Minimum transitions required

    Returns:
        Dict mapping peptide name to pre-computed metrics
    """
    # Filter to specified samples
    filtered = data[data[sample_col].isin(sample_list)].copy()
    peptides = filtered[peptide_col].unique()

    results = {}

    for peptide in peptides:
        pep_data = filtered[filtered[peptide_col] == peptide]

        # Pivot to transition x sample (intensity)
        intensity_pivot = pep_data.pivot_table(
            index=transition_col, columns=sample_col, values=abundance_col, aggfunc="first"
        )

        # Fill missing samples
        for s in sample_list:
            if s not in intensity_pivot.columns:
                intensity_pivot[s] = np.nan
        intensity_pivot = intensity_pivot[sample_list]

        n_trans = len(intensity_pivot)
        if n_trans < min_transitions:
            continue

        # Log2 transform
        intensity_log2 = np.log2(np.maximum(intensity_pivot.values, 1.0))

        # Mean intensity per transition (linear scale)
        linear_intensity = intensity_pivot.values
        mean_intensity_linear = np.nanmean(linear_intensity, axis=1)

        # Shape correlation
        if shape_corr_col in pep_data.columns:
            shape_pivot = pep_data.pivot_table(
                index=transition_col, columns=sample_col, values=shape_corr_col, aggfunc="first"
            )
            shape_pivot = shape_pivot.reindex(
                index=intensity_pivot.index, columns=sample_list
            ).fillna(1.0)
            shape_vals = shape_pivot.values

            if shape_corr_agg == "median":
                shape_corr = np.nanmedian(shape_vals, axis=1)
            else:
                shape_corr = np.nanmean(shape_vals, axis=1)

            # CV of shape correlation
            shape_std = np.nanstd(shape_vals, axis=1)
            shape_mean = np.nanmean(shape_vals, axis=1)
            shape_corr_cv = np.divide(
                shape_std, shape_mean,
                out=np.zeros_like(shape_std),
                where=shape_mean > 0.01
            )
        else:
            shape_corr = np.ones(n_trans)
            shape_corr_cv = np.zeros(n_trans)

        # Compute residuals from median polish
        try:
            from .rollup import tukey_median_polish
            polish_result = tukey_median_polish(
                pd.DataFrame(intensity_log2, columns=sample_list)
            )
            # MAD of residuals per transition (row)
            residuals_mad = np.median(np.abs(polish_result.residuals.values), axis=1)
        except Exception:
            residuals_mad = np.zeros(n_trans)

        results[peptide] = _PrecomputedPeptideMetrics(
            intensity_log2=intensity_log2,
            mean_intensity_linear=mean_intensity_linear,
            shape_corr=np.clip(shape_corr, 0.001, 1.0),
            shape_corr_cv=shape_corr_cv,
            residuals_mad=residuals_mad,
            n_transitions=n_trans,
        )

    return results


def _rollup_with_params_fast(
    precomputed: dict[str, _PrecomputedPeptideMetrics],
    params: QualityWeightParams,
) -> pd.DataFrame:
    """Fast rollup using pre-computed metrics.

    Only recomputes weights and aggregation - no pivoting or median polish.

    Args:
        precomputed: Pre-computed metrics for each peptide
        params: Quality weight parameters

    Returns:
        Peptide x sample matrix of abundances (LOG2 scale)
    """
    if not precomputed:
        return pd.DataFrame()

    # Get sample count from first peptide
    first = next(iter(precomputed.values()))
    n_samples = first.intensity_log2.shape[1]

    results = {}

    for peptide, metrics in precomputed.items():
        # Compute weights from pre-computed metrics
        intensity_contrib = np.power(
            np.maximum(metrics.mean_intensity_linear, 1.0),
            params.alpha
        )
        shape_corr_contrib = np.power(metrics.shape_corr, params.beta)
        cv_penalty = np.power(1.0 / (1.0 + metrics.shape_corr_cv), params.delta)
        residual_penalty = np.exp(-params.gamma * np.power(metrics.residuals_mad, 2))

        weights = intensity_contrib * shape_corr_contrib * cv_penalty * residual_penalty

        # Normalize weights to sum to n_transitions (preserves sum magnitude)
        weight_sum = weights.sum()
        if weight_sum <= 0:
            normalized_weights = np.ones_like(weights)
        else:
            normalized_weights = weights * (metrics.n_transitions / weight_sum)

        # Weighted sum in linear space for each sample
        linear_matrix = 2 ** metrics.intensity_log2
        abundances = np.zeros(n_samples)

        for i in range(n_samples):
            col = linear_matrix[:, i]
            valid = ~np.isnan(col)
            if valid.sum() >= 3:
                weighted_sum = (normalized_weights[valid] * col[valid]).sum()
                abundances[i] = np.log2(max(weighted_sum, 1.0))
            else:
                abundances[i] = np.nan

        results[peptide] = abundances

    return pd.DataFrame.from_dict(results, orient='index')


def _rollup_all_peptides_quality_weighted(
    data: pd.DataFrame,
    sample_list: list[str],
    params: QualityWeightParams,
    peptide_col: str = "peptide_modified",
    transition_col: str = "fragment_ion",
    sample_col: str = "replicate_name",
    abundance_col: str = "area",
    shape_corr_col: str = "shape_correlation",
    min_transitions: int = 3,
) -> pd.DataFrame:
    """Roll up all peptides using quality weights.

    Args:
        data: Transition-level DataFrame (LINEAR scale intensities)
        sample_list: List of samples to include
        params: Quality weight parameters
        peptide_col, transition_col, sample_col, abundance_col, shape_corr_col: Column names
        min_transitions: Minimum transitions per peptide

    Returns:
        Peptide x sample matrix of abundances (LOG2 scale)
    """
    # Filter to specified samples
    filtered_data = data[data[sample_col].isin(sample_list)].copy()

    peptides = filtered_data[peptide_col].unique()
    peptide_abundances = pd.DataFrame(index=peptides, columns=sample_list, dtype=float)

    for peptide in peptides:
        pep_data = filtered_data[filtered_data[peptide_col] == peptide]

        # Pivot to transition x sample
        intensity_matrix = pep_data.pivot_table(
            index=transition_col, columns=sample_col, values=abundance_col, aggfunc="first"
        )

        # Fill missing samples
        for sample in sample_list:
            if sample not in intensity_matrix.columns:
                intensity_matrix[sample] = np.nan
        intensity_matrix = intensity_matrix[sample_list]

        # Log transform
        intensity_matrix = np.log2(intensity_matrix.clip(lower=1))

        # Shape correlation matrix
        if shape_corr_col in pep_data.columns:
            shape_corr_matrix = pep_data.pivot_table(
                index=transition_col, columns=sample_col, values=shape_corr_col, aggfunc="first"
            )
            # Align with intensity_matrix (same rows and columns)
            shape_corr_matrix = shape_corr_matrix.reindex(
                index=intensity_matrix.index, columns=sample_list
            ).fillna(1.0)
        else:
            shape_corr_matrix = pd.DataFrame(
                1.0, index=intensity_matrix.index, columns=intensity_matrix.columns
            )

        # Roll up with quality weights
        abundances, _, _, _ = rollup_peptide_quality_weighted(
            intensity_matrix, shape_corr_matrix, params, min_transitions
        )
        peptide_abundances.loc[peptide] = abundances

    return peptide_abundances


def _rollup_all_peptides_sum(
    data: pd.DataFrame,
    sample_list: list[str],
    peptide_col: str = "peptide_modified",
    transition_col: str = "fragment_ion",
    sample_col: str = "replicate_name",
    abundance_col: str = "area",
    min_transitions: int = 3,
) -> pd.DataFrame:
    """Roll up all peptides using simple sum (vectorized).

    Args:
        data: Transition-level DataFrame (LINEAR scale intensities)
        sample_list: List of samples to include
        peptide_col, transition_col, sample_col, abundance_col: Column names
        min_transitions: Minimum transitions per peptide

    Returns:
        Peptide x sample matrix of abundances (LOG2 scale)

    """
    # Filter to specified samples
    filtered_data = data[data[sample_col].isin(sample_list)].copy()

    # Vectorized: sum by peptide and sample
    summed = filtered_data.groupby([peptide_col, sample_col])[abundance_col].sum()

    # Count transitions per peptide-sample
    n_trans = filtered_data.groupby([peptide_col, sample_col])[transition_col].count()

    # Pivot to peptide x sample matrix
    peptide_abundances = summed.unstack(level=sample_col)

    # Ensure all samples are present
    for sample in sample_list:
        if sample not in peptide_abundances.columns:
            peptide_abundances[sample] = np.nan
    peptide_abundances = peptide_abundances[sample_list]

    # Apply minimum transitions filter
    n_trans_matrix = n_trans.unstack(level=sample_col).reindex(
        index=peptide_abundances.index, columns=sample_list
    ).fillna(0)
    peptide_abundances = peptide_abundances.where(n_trans_matrix >= min_transitions)

    # Convert to log2 scale
    peptide_abundances = np.log2(peptide_abundances.clip(lower=1))

    return peptide_abundances


def learn_quality_weights(
    data: pd.DataFrame,
    reference_samples: list[str],
    pool_samples: list[str],
    peptide_col: str = "peptide_modified",
    transition_col: str = "fragment_ion",
    sample_col: str = "replicate_name",
    abundance_col: str = "area",
    shape_corr_col: str = "shape_correlation",
    n_iterations: int = 100,
    initial_params: QualityWeightParams | None = None,
) -> QualityWeightResult:
    """Learn quality weight parameters from reference samples, validate on pool.

    EFFICIENT IMPLEMENTATION: Pre-computes quality metrics once, then uses
    fast vectorized rollup during optimization.

    Uses reference samples (technical replicates) to optimize weight parameters
    by minimizing CV. Then validates on pool samples (independent technical
    replicates) to ensure generalization.

    If improvement on pool samples is less than min_improvement_pct, falls back
    to the fallback_method (default: sum).

    Args:
        data: Transition-level DataFrame (LINEAR scale intensities)
        reference_samples: List of reference sample names (for learning)
        pool_samples: List of pool sample names (for validation)
        peptide_col, transition_col, sample_col, abundance_col, shape_corr_col: Column names
        n_iterations: Maximum optimization iterations
        initial_params: Starting parameters (uses defaults if None)

    Returns:
        QualityWeightResult with learned params and validation metrics
    """
    from scipy.optimize import minimize

    logger.info("Learning quality weight parameters (efficient)")
    logger.info(f"  Reference samples: {len(reference_samples)}")
    logger.info(f"  Pool samples: {len(pool_samples)}")

    if initial_params is None:
        initial_params = QualityWeightParams()

    if len(reference_samples) < 2:
        logger.warning(
            f"Need at least 2 reference samples, got {len(reference_samples)}. "
            "Using default parameters."
        )
        return QualityWeightResult(
            params=initial_params,
            reference_cv_before=np.nan,
            reference_cv_after=np.nan,
            pool_cv_before=np.nan,
            pool_cv_after=np.nan,
            use_quality_weights=False,
            fallback_reason="Insufficient reference samples",
        )

    # Calculate baseline CV using sum (fallback method)
    logger.info("  Computing baseline CV (sum method)...")
    ref_abundances_sum = _rollup_all_peptides_sum(
        data, reference_samples, peptide_col, transition_col, sample_col, abundance_col
    )
    reference_cv_before = _compute_median_cv_linear(ref_abundances_sum)
    logger.info(f"  Reference CV (sum): {reference_cv_before:.4f}")

    pool_cv_before = np.nan
    if len(pool_samples) >= 2:
        pool_abundances_sum = _rollup_all_peptides_sum(
            data, pool_samples, peptide_col, transition_col, sample_col, abundance_col
        )
        pool_cv_before = _compute_median_cv_linear(pool_abundances_sum)
        logger.info(f"  Pool CV (sum): {pool_cv_before:.4f}")

    # PRE-COMPUTE metrics for reference samples (one-time cost)
    logger.info("  Pre-computing quality metrics for reference samples...")
    ref_metrics = _precompute_peptide_metrics(
        data, reference_samples, peptide_col, transition_col,
        sample_col, abundance_col, shape_corr_col,
        shape_corr_agg=initial_params.shape_corr_agg,
    )
    logger.info(f"  Pre-computed metrics for {len(ref_metrics)} peptides")

    # Track optimization progress
    iteration_count = [0]

    def objective(params_array):
        """Objective: minimize median CV on reference samples."""
        params = QualityWeightParams(
            alpha=max(0.0, min(2.0, params_array[0])),
            beta=max(0.0, min(5.0, params_array[1])),
            gamma=max(0.0, min(10.0, params_array[2])),
            delta=max(0.0, min(5.0, params_array[3])),
            shape_corr_agg=initial_params.shape_corr_agg,
            fallback_method=initial_params.fallback_method,
            min_improvement_pct=initial_params.min_improvement_pct,
        )

        try:
            # FAST rollup using pre-computed metrics
            abundances = _rollup_with_params_fast(ref_metrics, params)
            cv = _compute_median_cv_linear(abundances)
            iteration_count[0] += 1
            return cv if np.isfinite(cv) else 1.0
        except Exception:
            return 1.0

    # Initial parameters
    x0 = [initial_params.alpha, initial_params.beta, initial_params.gamma, initial_params.delta]

    # Bounds
    bounds = [
        (0.0, 2.0),   # alpha (intensity exponent)
        (0.0, 5.0),   # beta (shape_corr exponent)
        (0.0, 10.0),  # gamma (residual penalty)
        (0.0, 5.0),   # delta (shape_corr_cv penalty)
    ]

    logger.info("  Optimizing quality weight parameters...")
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": n_iterations},
    )
    logger.info(f"  Optimization completed in {iteration_count[0]} evaluations")

    # Extract optimized parameters
    opt = result.x
    optimized_params = QualityWeightParams(
        alpha=max(0.0, min(2.0, opt[0])),
        beta=max(0.0, min(5.0, opt[1])),
        gamma=max(0.0, min(10.0, opt[2])),
        delta=max(0.0, min(5.0, opt[3])),
        shape_corr_agg=initial_params.shape_corr_agg,
        fallback_method=initial_params.fallback_method,
        min_improvement_pct=initial_params.min_improvement_pct,
    )

    # Calculate final CV on reference (use fast rollup)
    ref_abundances_qw = _rollup_with_params_fast(ref_metrics, optimized_params)
    reference_cv_after = _compute_median_cv_linear(ref_abundances_qw)

    logger.info("  Optimized parameters:")
    logger.info(f"    alpha (intensity): {optimized_params.alpha:.3f}")
    logger.info(f"    beta (shape_corr): {optimized_params.beta:.3f}")
    logger.info(f"    gamma (residual): {optimized_params.gamma:.3f}")
    logger.info(f"    delta (shape_corr_cv): {optimized_params.delta:.3f}")
    logger.info(f"  Reference CV: {reference_cv_before:.4f} -> {reference_cv_after:.4f}")

    # Validate on pool samples
    pool_cv_after = np.nan
    use_quality_weights = True
    fallback_reason = None

    if len(pool_samples) >= 2:
        # Pre-compute metrics for pool samples
        pool_metrics = _precompute_peptide_metrics(
            data, pool_samples, peptide_col, transition_col,
            sample_col, abundance_col, shape_corr_col,
            shape_corr_agg=initial_params.shape_corr_agg,
        )
        pool_abundances_qw = _rollup_with_params_fast(pool_metrics, optimized_params)
        pool_cv_after = _compute_median_cv_linear(pool_abundances_qw)
        logger.info(f"  Pool CV: {pool_cv_before:.4f} -> {pool_cv_after:.4f}")

        # Check if improvement is sufficient
        if pool_cv_before > 0 and pool_cv_after > 0:
            improvement_pct = (pool_cv_before - pool_cv_after) / pool_cv_before * 100
            logger.info(f"  Pool improvement: {improvement_pct:.1f}%")

            if improvement_pct < optimized_params.min_improvement_pct:
                use_quality_weights = False
                fallback_reason = (
                    f"Pool improvement ({improvement_pct:.1f}%) below threshold "
                    f"({optimized_params.min_improvement_pct}%)"
                )
                logger.warning(f"  {fallback_reason}")
                logger.warning(f"  Falling back to: {optimized_params.fallback_method}")
        elif pool_cv_after >= pool_cv_before:
            use_quality_weights = False
            fallback_reason = "Quality-weighted rollup did not improve pool CV"
            logger.warning(f"  {fallback_reason}")
    else:
        logger.warning(
            "  Not enough pool samples for validation - proceeding without validation"
        )

    return QualityWeightResult(
        params=optimized_params,
        reference_cv_before=reference_cv_before,
        reference_cv_after=reference_cv_after,
        pool_cv_before=pool_cv_before,
        pool_cv_after=pool_cv_after,
        use_quality_weights=use_quality_weights,
        fallback_reason=fallback_reason,
    )

