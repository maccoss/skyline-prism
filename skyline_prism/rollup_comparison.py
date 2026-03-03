"""Rollup method comparison module for QC visualization.

This module provides functionality to compare different transition-to-peptide
rollup methods, specifically focusing on library_assist vs sum comparison.
It captures intermediate fitting steps for visualization in the QC report.

The comparison shows:
- Raw transition intensities
- Library spectrum scaled to observed
- Initial fit before outlier removal
- Fit after outlier removal
- Final abundances (sum vs library_assist)
- CV calculated on corrected (normalized + batch corrected) abundances
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .spectral_library import FragmentSpectrum, SpectralLibraryRollup

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class LibraryFitStep:
    """One step in the library fitting process.

    Captures the state at each stage of the library-assisted fitting algorithm
    for visualization purposes.
    """

    step_name: str  # "raw", "library_scaled", "initial_fit", "outlier_removed"
    transition_intensities: pd.Series  # Transition -> intensity (linear scale)
    library_predicted: pd.Series | None  # Transition -> predicted intensity
    excluded_transitions: list[str]  # Transitions marked as outliers
    r_squared: float | None  # R² of fit at this step
    scale_factor: float | None  # exp(beta_s) scale factor
    normalized_residuals: pd.Series | None  # Transition -> normalized residual


@dataclass
class PeptideLibraryComparison:
    """Comparison of sum vs library_assist for one peptide.

    Contains all data needed to visualize the library fitting process
    and compare the two rollup methods.
    """

    peptide: str
    mz_values: pd.Series  # Transition -> m/z

    # Per-replicate raw data (for visualization)
    raw_transitions: dict[str, pd.Series]  # replicate -> (transition -> intensity)
    library_spectrum: pd.Series  # Transition -> library intensity (relative)

    # Library fitting steps per replicate (for visualization)
    fitting_steps: dict[str, list[LibraryFitStep]]  # replicate -> steps

    # Raw rollup abundances (before normalization)
    sum_abundances_raw: pd.Series  # replicate -> abundance
    library_abundances_raw: pd.Series  # replicate -> abundance

    # Corrected abundances (after normalization + batch correction)
    sum_abundances_corrected: pd.Series  # replicate -> abundance
    library_abundances_corrected: pd.Series  # replicate -> abundance

    # Metrics (calculated on CORRECTED abundances)
    sum_cv: float  # CV of corrected sum abundances
    library_cv: float  # CV of corrected library abundances
    cv_improvement: float  # (sum_cv - library_cv) / sum_cv
    ranking_score: float


@dataclass
class RollupComparisonSummary:
    """Summary statistics for the comparison."""

    n_peptides_total: int
    n_peptides_improved: int
    n_peptides_worsened: int
    n_peptides_unchanged: int  # CV difference < 1%

    sum_median_cv: float
    library_median_cv: float
    median_improvement: float  # Median of individual peptide improvements


@dataclass
class RollupComparisonResult:
    """Full comparison results across all peptides."""

    sample_type: str  # "reference" or "qc"
    samples: list[str]  # Sample names used
    library_path: str  # Path to spectral library

    # Processing flags
    normalization_applied: bool
    batch_correction_applied: bool

    # Per-peptide results
    peptide_results: dict[str, PeptideLibraryComparison]

    # Top N peptides by ranking criterion
    top_peptides: list[str]
    ranking_criterion: str

    # Summary statistics
    summary: RollupComparisonSummary


# =============================================================================
# Core Functions
# =============================================================================


def compute_cv_linear(abundances: pd.Series) -> float:
    """Compute coefficient of variation on linear scale.

    CV = std / mean, computed on linear (not log2) values.

    Args:
        abundances: Series of abundances (can be log2 or linear)
                   Assumes LINEAR scale input

    Returns:
        CV as a decimal (e.g., 0.15 for 15% CV)
    """
    if abundances.isna().all():
        return np.nan

    valid = abundances.dropna()
    if len(valid) < 2:
        return np.nan

    mean_val = valid.mean()
    if mean_val == 0:
        return np.nan

    return float(valid.std() / mean_val)


def compute_ranking_score(
    sum_cv: float,
    library_cv: float,
    criterion: str,
) -> float:
    """Compute ranking score for a peptide based on the criterion.

    Higher scores = more interesting peptides (shown first in report).

    Args:
        sum_cv: CV from sum rollup method
        library_cv: CV from library_assist rollup method
        criterion: One of "most_improved", "most_worsened", "highest_cv", "largest_difference"

    Returns:
        Ranking score (higher = more interesting)
    """
    if np.isnan(sum_cv) or np.isnan(library_cv):
        return -np.inf  # Put NaN peptides at the end

    if criterion == "most_improved":
        # Higher = library_assist improved CV more
        if sum_cv == 0:
            return 0.0
        return (sum_cv - library_cv) / sum_cv

    elif criterion == "most_worsened":
        # Higher = library_assist worsened CV more
        if sum_cv == 0:
            return 0.0
        return (library_cv - sum_cv) / sum_cv

    elif criterion == "highest_cv":
        # Higher = worse baseline (sum) CV
        return sum_cv

    elif criterion == "largest_difference":
        # Higher = largest absolute difference between methods
        return abs(sum_cv - library_cv)

    else:
        raise ValueError(f"Unknown ranking criterion: {criterion}")


def rank_peptides(
    peptide_results: dict[str, PeptideLibraryComparison],
    criterion: str,
    top_n: int,
) -> list[str]:
    """Rank peptides by the specified criterion and return top N.

    Args:
        peptide_results: Dict of peptide -> PeptideLibraryComparison
        criterion: Ranking criterion
        top_n: Number of top peptides to return

    Returns:
        List of top N peptide names, ordered by rank (best first)
    """
    scores = []
    for peptide, result in peptide_results.items():
        score = compute_ranking_score(result.sum_cv, result.library_cv, criterion)
        scores.append((peptide, score))

    # Sort by score descending (higher = better)
    scores.sort(key=lambda x: x[1], reverse=True)

    return [peptide for peptide, _ in scores[:top_n]]


# =============================================================================
# Library Fitting with Step Capture
# =============================================================================


def fit_library_single_sample(
    observed: pd.Series,
    library: pd.Series,
    mz_values: pd.Series,
    min_fragments: int = 2,
    outlier_threshold: float = 1.0,
    max_iterations: int = 5,
) -> tuple[float, float, list[LibraryFitStep]]:
    """Fit library spectrum to observed intensities, capturing intermediate steps.

    This is a non-vectorized version of library_median_polish_rollup_vectorized
    that captures intermediate steps for visualization.

    Algorithm (same as library_median_polish_rollup_vectorized):
    1. Model: log(Observed[t]) = log(Library[t]) + beta + epsilon[t]
    2. Estimate beta = MEDIAN across transitions of [log(obs) - log(lib)]
    3. Compute normalized residuals: (obs - pred) / pred
    4. Detect outliers: only HIGH positive residuals indicate interference
    5. Iterate: exclude worst outlier, refit, until convergence
    6. Final abundance = exp(beta) × sum(ALL library intensities)

    Args:
        observed: Series of observed intensities (Transition -> intensity, LINEAR)
        library: Series of library intensities (Transition -> intensity, relative)
        mz_values: Series of m/z values (Transition -> m/z)
        min_fragments: Minimum fragments required
        outlier_threshold: Normalized residual threshold for outlier detection
        max_iterations: Maximum outlier removal iterations

    Returns:
        Tuple of:
        - abundance: Final peptide abundance (LINEAR scale)
        - r_squared: Final R² of fit
        - steps: List of LibraryFitStep objects capturing each stage
    """
    steps = []

    # Align indices
    common_idx = observed.index.intersection(library.index)
    obs = observed.loc[common_idx].astype(float)
    lib = library.loc[common_idx].astype(float)
    mz = mz_values.loc[common_idx] if mz_values is not None else None

    # Step 0: Raw transitions
    steps.append(
        LibraryFitStep(
            step_name="raw",
            transition_intensities=obs.copy(),
            library_predicted=None,
            excluded_transitions=[],
            r_squared=None,
            scale_factor=None,
            normalized_residuals=None,
        )
    )

    # Check valid library entries
    lib_valid = (lib > 0) & ~lib.isna()
    n_valid = lib_valid.sum()

    if n_valid < min_fragments:
        # Not enough fragments - return NaN
        return np.nan, np.nan, steps

    # Filter to valid fragments
    obs_v = obs[lib_valid].copy()
    lib_v = lib[lib_valid].copy()
    lib_sum = lib_v.sum()

    # Replace zeros with NaN for log transform
    obs_safe = obs_v.replace(0, np.nan)

    # Work in log space
    log_obs = np.log(obs_safe)
    log_lib = np.log(lib_v)

    # Initialize inclusion mask
    included = ~log_obs.isna()
    excluded_transitions: list[str] = []

    # Initial fit (before outlier removal)
    diff = log_obs - log_lib
    diff_masked = diff.where(included, np.nan)

    # Suppress warning when all values are NaN (handled by returning NaN below)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        beta = np.nanmedian(diff_masked)

    if np.isnan(beta):
        return np.nan, np.nan, steps

    scale = np.exp(beta)
    predicted = lib_v * scale

    # Step 1: Library scaled (initial fit)
    norm_residuals = pd.Series(index=obs_v.index, dtype=float)
    valid_pred = predicted > 0
    norm_residuals[valid_pred] = (obs_v[valid_pred] - predicted[valid_pred]) / predicted[
        valid_pred
    ]

    # Compute initial R²
    obs_for_r2 = obs_v.fillna(0)
    residuals = obs_for_r2[included] - predicted[included]
    ss_res = (residuals**2).sum()
    obs_mean = obs_for_r2[included].mean()
    ss_tot = ((obs_for_r2[included] - obs_mean) ** 2).sum()
    r_squared_initial = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    steps.append(
        LibraryFitStep(
            step_name="library_scaled",
            transition_intensities=obs_v.copy(),
            library_predicted=predicted.copy(),
            excluded_transitions=[],
            r_squared=r_squared_initial,
            scale_factor=scale,
            normalized_residuals=norm_residuals.copy(),
        )
    )

    # Iterative outlier removal
    for iteration in range(max_iterations):
        # Find worst outlier
        worst_outlier = None
        worst_residual = outlier_threshold

        for t in included[included].index:
            if norm_residuals.get(t, 0) > worst_residual:
                worst_residual = norm_residuals[t]
                worst_outlier = t

        if worst_outlier is None:
            # No outliers exceed threshold - converged
            break

        # Check if we have enough fragments remaining
        n_included = included.sum()
        if n_included <= min_fragments:
            break

        # Exclude the outlier
        included[worst_outlier] = False
        excluded_transitions.append(str(worst_outlier))

        # Refit
        diff_masked = diff.where(included, np.nan)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            beta = np.nanmedian(diff_masked)

        if np.isnan(beta):
            break

        scale = np.exp(beta)
        predicted = lib_v * scale

        # Recompute residuals
        norm_residuals = pd.Series(index=obs_v.index, dtype=float)
        valid_pred = predicted > 0
        norm_residuals[valid_pred] = (obs_v[valid_pred] - predicted[valid_pred]) / predicted[
            valid_pred
        ]

    # Step 2: After outlier removal (if any outliers were removed)
    if excluded_transitions:
        # Compute final R²
        obs_for_r2 = obs_v.fillna(0)
        residuals = (obs_for_r2[included] - predicted[included]) if included.any() else pd.Series()
        ss_res = (residuals**2).sum() if len(residuals) > 0 else 0
        obs_mean = obs_for_r2[included].mean() if included.any() else 0
        ss_tot = ((obs_for_r2[included] - obs_mean) ** 2).sum() if included.any() else 0
        r_squared_after_outlier = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Check if outlier removal actually improved the fit
        # Keep if R² improved (higher value = better, regardless of sign)
        if r_squared_after_outlier >= r_squared_initial:
            # Outlier removal helped - use the new fit
            r_squared_final = r_squared_after_outlier
            steps.append(
                LibraryFitStep(
                    step_name="outliers_removed",
                    transition_intensities=obs_v.copy(),
                    library_predicted=predicted.copy(),
                    excluded_transitions=excluded_transitions.copy(),
                    r_squared=r_squared_final,
                    scale_factor=scale,
                    normalized_residuals=norm_residuals.copy(),
                )
            )
        else:
            # Outlier removal made fit worse - revert to initial fit
            r_squared_final = r_squared_initial
            # Recompute scale from initial fit (all fragments included)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="All-NaN slice encountered")
                beta = np.nanmedian(diff)  # Use original diff without masking
            scale = np.exp(beta)
            predicted = lib_v * scale
            excluded_transitions = []  # Clear excluded list
            # Add step showing we reverted
            steps.append(
                LibraryFitStep(
                    step_name="outliers_reverted",
                    transition_intensities=obs_v.copy(),
                    library_predicted=predicted.copy(),
                    excluded_transitions=[],
                    r_squared=r_squared_final,
                    scale_factor=scale,
                    normalized_residuals=norm_residuals.copy(),
                )
            )
    else:
        r_squared_final = r_squared_initial

    # Final abundance = scale * sum(ALL library intensities)
    abundance = scale * lib_sum

    return abundance, r_squared_final, steps


def compare_library_vs_sum(
    transition_data: pd.DataFrame,
    corrected_peptides_sum: pd.DataFrame,
    corrected_peptides_library: pd.DataFrame,
    samples: list[str],
    library: "SpectralLibraryRollup",
    library_path: str,
    peptide_col: str = "Peptide Modified Sequence",
    precursor_charge_col: str = "Precursor Charge",
    fragment_col: str = "Fragment Ion",
    product_mz_col: str = "Product Mz",
    sample_col: str = "Replicate Name",
    abundance_col: str = "TotalAreaFragment",
    library_config: dict | None = None,
    normalization_applied: bool = False,
    batch_correction_applied: bool = False,
    top_n: int = 20,
    ranking_criterion: str = "most_improved",
) -> RollupComparisonResult:
    """Compare library_assist vs sum rollup methods across all peptides.

    Args:
        transition_data: DataFrame with transition-level data (raw, LINEAR scale)
        corrected_peptides_sum: DataFrame with corrected peptide abundances from sum rollup
        corrected_peptides_library: DataFrame with corrected peptide abundances from library rollup
        samples: List of sample names to include in comparison
        library: Loaded SpectralLibraryRollup object
        library_path: Path to spectral library (for metadata)
        peptide_col: Column name for peptide sequence
        precursor_charge_col: Column name for precursor charge
        fragment_col: Column name for fragment ion
        product_mz_col: Column name for product m/z
        sample_col: Column name for sample/replicate
        abundance_col: Column name for abundance values
        library_config: Library-assist configuration parameters
        normalization_applied: Whether normalization was applied
        batch_correction_applied: Whether batch correction was applied
        top_n: Number of top peptides to include in report
        ranking_criterion: How to rank peptides

    Returns:
        RollupComparisonResult with all comparison data
    """
    if library_config is None:
        library_config = {}

    min_fragments = library_config.get("min_matched_fragments", 3)
    outlier_threshold = library_config.get("outlier_threshold", 1.0)

    peptide_results: dict[str, PeptideLibraryComparison] = {}

    # Get unique peptide + charge combinations
    if precursor_charge_col in transition_data.columns:
        transition_data["_peptide_key"] = (
            transition_data[peptide_col].astype(str) + "_" +
            transition_data[precursor_charge_col].astype(str)
        )
    else:
        transition_data["_peptide_key"] = transition_data[peptide_col].astype(str)

    peptide_keys = transition_data["_peptide_key"].unique()
    logger.info(f"Comparing rollup methods for {len(peptide_keys)} peptides across {len(samples)} samples")

    n_with_library = 0
    n_processed = 0
    n_no_replicates = 0
    n_empty_lib_series = 0

    for peptide_key in peptide_keys:
        # Get transition data for this peptide
        pep_data = transition_data[transition_data["_peptide_key"] == peptide_key]

        # Extract peptide sequence and charge
        peptide = pep_data[peptide_col].iloc[0]
        charge = int(pep_data[precursor_charge_col].iloc[0]) if precursor_charge_col in pep_data.columns else 2

        # Get library spectrum using correct API
        lib_spectrum = library.get_spectrum(peptide, charge)
        if lib_spectrum is None:
            continue

        n_with_library += 1

        # Build transition -> m/z mapping
        mz_values = pep_data.groupby(fragment_col)[product_mz_col].first()

        # Get raw transitions per replicate
        # Use groupby to handle duplicate fragment annotations (sum them)
        raw_transitions: dict[str, pd.Series] = {}
        for sample in samples:
            sample_data = pep_data[pep_data[sample_col] == sample]
            if len(sample_data) > 0:
                # Sum duplicate fragment annotations to avoid duplicate index issues
                raw_transitions[sample] = sample_data.groupby(fragment_col)[abundance_col].sum()

        if len(raw_transitions) < 2:
            n_no_replicates += 1
            continue  # Need at least 2 replicates for CV

        # Build library spectrum as Series from FragmentSpectrum
        # BLIB files use fragments_by_mz (m/z-based lookup), not fragments (annotation-based)
        # We need to match observed transitions to library by product m/z
        lib_series = pd.Series(dtype=float)

        # First try fragments dict (for TSV/Carafe libraries with annotations)
        if hasattr(lib_spectrum, 'fragments') and lib_spectrum.fragments:
            # fragments is a dict: (type, num, charge, loss) -> intensity
            for frag_key, intensity in lib_spectrum.fragments.items():
                # Convert fragment key to annotation string like "y7" or "b3"
                frag_type, frag_num, frag_charge, loss = frag_key
                annotation = f"{frag_type}{frag_num}"
                if frag_charge > 1:
                    annotation += f"^{frag_charge}"
                lib_series[annotation] = intensity

        # For BLIB files, use fragments_by_mz to match by product m/z
        if lib_series.empty and hasattr(lib_spectrum, 'fragments_by_mz') and lib_spectrum.fragments_by_mz:
            # Match each observed transition by its product m/z
            mz_tolerance = 0.02  # Da tolerance for matching
            for frag_annotation, product_mz in mz_values.items():
                mz_rounded = round(float(product_mz), 2)
                # Find best matching library m/z
                best_intensity = None
                best_diff = mz_tolerance + 1
                for lib_mz, intensity in lib_spectrum.fragments_by_mz.items():
                    diff = abs(lib_mz - mz_rounded)
                    if diff < best_diff and diff <= mz_tolerance:
                        best_diff = diff
                        best_intensity = intensity
                if best_intensity is not None:
                    lib_series[frag_annotation] = best_intensity

        if lib_series.empty:
            n_empty_lib_series += 1
            continue

        # Fit library for each replicate and capture steps
        fitting_steps: dict[str, list[LibraryFitStep]] = {}
        library_abundances_raw: dict[str, float] = {}

        for sample, obs in raw_transitions.items():
            abundance, r_squared, steps = fit_library_single_sample(
                observed=obs,
                library=lib_series,
                mz_values=mz_values,
                min_fragments=min_fragments,
                outlier_threshold=outlier_threshold,
            )
            fitting_steps[sample] = steps
            library_abundances_raw[sample] = abundance

        # Compute sum abundances (simple sum of transitions)
        sum_abundances_raw: dict[str, float] = {}
        for sample, obs in raw_transitions.items():
            sum_abundances_raw[sample] = obs.sum()

        # For CV calculation, use raw rollup abundances
        # (The corrected peptides may not have the same peptide key format)
        sum_abundances_corrected = pd.Series(sum_abundances_raw)
        library_abundances_corrected = pd.Series(library_abundances_raw)

        # Try to get corrected abundances if available
        for corrected_df, target_series, source_dict in [
            (corrected_peptides_sum, "sum", sum_abundances_raw),
            (corrected_peptides_library, "library", library_abundances_raw),
        ]:
            if peptide in corrected_df.index:
                row = corrected_df.loc[peptide]
                available_samples = [s for s in samples if s in row.index]
                if available_samples:
                    if target_series == "sum":
                        sum_abundances_corrected = row[available_samples].astype(float)
                    else:
                        library_abundances_corrected = row[available_samples].astype(float)

        # Compute CVs
        sum_cv = compute_cv_linear(sum_abundances_corrected)
        library_cv = compute_cv_linear(library_abundances_corrected)

        # Compute improvement
        if sum_cv > 0 and not np.isnan(sum_cv) and not np.isnan(library_cv):
            cv_improvement = (sum_cv - library_cv) / sum_cv
        else:
            cv_improvement = 0.0

        # Compute ranking score
        ranking_score = compute_ranking_score(sum_cv, library_cv, ranking_criterion)

        peptide_results[peptide] = PeptideLibraryComparison(
            peptide=peptide,
            mz_values=mz_values,
            raw_transitions=raw_transitions,
            library_spectrum=lib_series,
            fitting_steps=fitting_steps,
            sum_abundances_raw=pd.Series(sum_abundances_raw),
            library_abundances_raw=pd.Series(library_abundances_raw),
            sum_abundances_corrected=sum_abundances_corrected,
            library_abundances_corrected=library_abundances_corrected,
            sum_cv=sum_cv,
            library_cv=library_cv,
            cv_improvement=cv_improvement,
            ranking_score=ranking_score,
        )
        n_processed += 1

    logger.info(f"  Found library spectra for {n_with_library} peptides, processed {n_processed}")
    logger.info(f"  Filter counts: no_replicates={n_no_replicates}, empty_lib_series={n_empty_lib_series}")

    # Rank peptides and get top N
    top_peptides = rank_peptides(peptide_results, ranking_criterion, top_n)

    # Compute summary statistics
    all_sum_cvs = [r.sum_cv for r in peptide_results.values() if not np.isnan(r.sum_cv)]
    all_lib_cvs = [r.library_cv for r in peptide_results.values() if not np.isnan(r.library_cv)]
    all_improvements = [
        r.cv_improvement
        for r in peptide_results.values()
        if not np.isnan(r.cv_improvement)
    ]

    n_improved = sum(1 for r in peptide_results.values() if r.cv_improvement > 0.01)
    n_worsened = sum(1 for r in peptide_results.values() if r.cv_improvement < -0.01)
    n_unchanged = len(peptide_results) - n_improved - n_worsened

    summary = RollupComparisonSummary(
        n_peptides_total=len(peptide_results),
        n_peptides_improved=n_improved,
        n_peptides_worsened=n_worsened,
        n_peptides_unchanged=n_unchanged,
        sum_median_cv=float(np.median(all_sum_cvs)) if all_sum_cvs else np.nan,
        library_median_cv=float(np.median(all_lib_cvs)) if all_lib_cvs else np.nan,
        median_improvement=float(np.median(all_improvements)) if all_improvements else np.nan,
    )

    return RollupComparisonResult(
        sample_type="reference",  # Will be set by caller
        samples=samples,
        library_path=library_path,
        normalization_applied=normalization_applied,
        batch_correction_applied=batch_correction_applied,
        peptide_results=peptide_results,
        top_peptides=top_peptides,
        ranking_criterion=ranking_criterion,
        summary=summary,
    )
