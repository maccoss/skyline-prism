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
class TransitionRollupResult:
    """Result of transition to peptide rollup.

    When using median_polish method, transition_residuals contains per-peptide
    dictionaries of MedianPolishResult objects, which include the residual matrix.
    Large residuals may indicate transitions with interference or biologically
    interesting variation.
    """

    peptide_abundances: pd.DataFrame  # Peptide × sample matrix
    peptide_uncertainties: pd.DataFrame  # Uncertainty estimates
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


def aggregate_transitions_weighted(
    intensities: pd.DataFrame,
    weights: pd.Series,
    min_transitions: int = 3,
) -> tuple[pd.Series, pd.Series, int]:
    """Aggregate transition intensities using quality-based weights.

    Args:
        intensities: Transition × replicate matrix (log2 scale)
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

    Args:
        data: DataFrame with transition-level Skyline data
        peptide_col: Column identifying peptides
        transition_col: Column identifying transitions
        sample_col: Column identifying samples/replicates
        abundance_col: Column with transition intensities
        shape_corr_col: Column with shape correlation values
        coeluting_col: Column with coelution boolean
        method: Rollup method ('quality_weighted', 'median_polish', 'sum')
        params: Variance model parameters (uses defaults if None)
        min_transitions: Minimum transitions required per peptide
        log_transform: Whether to log2 transform intensities

    Returns:
        TransitionRollupResult with peptide abundances and diagnostics

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
