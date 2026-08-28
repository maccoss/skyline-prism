"""ComBat batch effect correction for proteomics data.

This module implements the ComBat algorithm (Johnson et al. 2007) for removing
batch effects from high-throughput data using empirical Bayes methods.

The implementation is based on the original R sva package and the pyComBat
Python implementation, adapted for proteomics workflows.

References:
    Johnson WE, Li C, Rabinovic A. (2007) Adjusting batch effects in microarray
    expression data using empirical Bayes methods. Biostatistics, 8(1), 118-127.

"""

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ComBatResult:
    """Results from ComBat batch correction.

    Attributes:
        corrected_data: Batch-corrected data matrix (features x samples)
        gamma_star: Estimated additive batch effects after EB shrinkage
        delta_star: Estimated multiplicative batch effects after EB shrinkage
        gamma_hat: Raw additive batch effect estimates
        delta_hat: Raw multiplicative batch effect estimates
        var_pooled: Pooled variance estimates per feature
        batch_info: Dictionary with batch composition information

    """

    corrected_data: np.ndarray
    gamma_star: np.ndarray
    delta_star: np.ndarray
    gamma_hat: np.ndarray
    delta_hat: np.ndarray
    var_pooled: np.ndarray
    batch_info: dict


def _check_inputs(
    data: np.ndarray,
    batch: np.ndarray,
    covar_mod: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Validate and prepare inputs for ComBat.

    Args:
        data: Expression matrix (features x samples)
        batch: Batch labels for each sample
        covar_mod: Optional covariate model matrix

    Returns:
        Validated data, batch, and covar_mod arrays

    Raises:
        ValueError: If inputs are invalid

    """
    data = np.asarray(data, dtype=np.float64)
    batch = np.asarray(batch)

    if data.ndim != 2:
        raise ValueError("Data must be a 2D matrix (features x samples)")

    n_features, n_samples = data.shape

    if len(batch) != n_samples:
        raise ValueError(f"Batch length ({len(batch)}) must match number of samples ({n_samples})")

    # Check for batches with single samples
    unique_batches, batch_counts = np.unique(batch, return_counts=True)
    single_sample_batches = unique_batches[batch_counts == 1]
    if len(single_sample_batches) > 0:
        raise ValueError(
            f"Batches {list(single_sample_batches)} contain a single sample, "
            "which is not supported for batch effect correction. "
            "Please review your inputs."
        )

    if covar_mod is not None:
        covar_mod = np.asarray(covar_mod, dtype=np.float64)
        if covar_mod.shape[0] != n_samples:
            raise ValueError(
                f"Covariate matrix rows ({covar_mod.shape[0]}) must match "
                f"number of samples ({n_samples})"
            )

    return data, batch, covar_mod


def _make_design_matrix(
    batch: np.ndarray,
    covar_mod: np.ndarray | None = None,
    ref_batch: int | str | None = None,
) -> tuple[np.ndarray, list[np.ndarray], int, int | None]:
    """Construct the design matrix for ComBat.

    Args:
        batch: Batch labels for each sample
        covar_mod: Optional covariate model matrix
        ref_batch: Optional reference batch (won't be adjusted)

    Returns:
        design: Full design matrix
        batches: List of sample indices for each batch
        n_batch: Number of batches
        ref_idx: Index of reference batch (None if not specified)

    """
    # Get unique batches and create batch indicator matrix
    unique_batches = np.unique(batch)
    n_batch = len(unique_batches)
    n_samples = len(batch)

    # Create batch design matrix (one-hot encoding)
    batch_design = np.zeros((n_samples, n_batch), dtype=np.float64)
    batches = []
    ref_idx = None

    for i, b in enumerate(unique_batches):
        idx = np.where(batch == b)[0]
        batches.append(idx)
        batch_design[idx, i] = 1
        if ref_batch is not None and b == ref_batch:
            ref_idx = i

    # Handle reference batch in design matrix
    if ref_idx is not None:
        # Set reference batch column to 1 for all samples
        batch_design[:, ref_idx] = 1

    # Combine with covariates
    if covar_mod is not None:
        # Check for intercept column and remove if present
        if covar_mod.ndim == 1:
            covar_mod = covar_mod.reshape(-1, 1)

        # Remove intercept column if present (all ones)
        intercept_cols = np.all(covar_mod == 1, axis=0)
        if np.any(intercept_cols) and ref_idx is None:
            covar_mod = covar_mod[:, ~intercept_cols]

        if covar_mod.shape[1] > 0:
            design = np.hstack([batch_design, covar_mod])
        else:
            design = batch_design
    else:
        design = batch_design

    # Check for confounding
    rank = np.linalg.matrix_rank(design)
    if rank < design.shape[1]:
        logger.warning("Design matrix is rank deficient. Covariates may be confounded with batch.")

    return design, batches, n_batch, ref_idx


def _spread_is_resolvable(variance: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Whether a within-batch variance is a scale estimate or just floating-point rounding.

    Testing ``variance == 0`` exactly - which is what sva and every previous version of this code
    did - is knife-edge. The same 82 replicates gave a within-batch variance of exactly 0.0 here
    and 7.99e-31 in the C# port, which flipped the feature between "no scale to estimate" and "a
    scale of 8e-31"; the two engines then produced answers 3% apart. The floor sits far above
    accumulated rounding (~1e-15 relative) and far below any real measurement (a standardized
    variance of 1e-6 is values agreeing to a part in a thousand), so nothing genuine is caught.

    Args:
        variance: Per-feature within-batch variance (ddof=1); NaN allowed.
        mean: The corresponding per-feature batch mean, used as the magnitude reference.

    Returns:
        Boolean array, True where the spread is a real estimate.

    """
    relative_floor = 1e-12
    scale = np.maximum(np.abs(mean), 1.0)
    with np.errstate(invalid="ignore"):
        resolvable: np.ndarray = np.isfinite(variance) & (
            np.sqrt(np.abs(variance)) > relative_floor * scale
        )
    return resolvable


def _beta_na(y: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Least-squares fit using only the samples where ``y`` was observed.

    Ports sva's ``Beta.NA``. ComBat must not let a missing value escape its own feature: the
    priors are means taken ACROSS features, so a NaN reaching them turns the whole batch's
    output into NaN. Fitting on the observed rows keeps it local.

    Args:
        y: One feature's values across samples; may contain NaN.
        design: Design matrix (samples x coefficients).

    Returns:
        Coefficient vector.

    """
    mask = ~np.isnan(y)
    X = design[mask]  # noqa: N806 - sva/ComBat notation: X is the design matrix
    return np.linalg.solve(X.T @ X, X.T @ y[mask])


def _fit_coefficients(data: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Solve the design for every feature, ignoring missing values.

    With no missing values this is the single matrix solve (identical results, much faster);
    with missing values it drops to sva's per-feature ``Beta.NA``.
    """
    if not np.isnan(data).any():
        coefficients: np.ndarray = np.linalg.solve(design.T @ design, design.T @ data.T)
        return coefficients
    return np.column_stack([_beta_na(data[g, :], design) for g in range(data.shape[0])])


def _calculate_mean_var(
    data: np.ndarray,
    design: np.ndarray,
    batches: list[np.ndarray],
    n_batch: int,
    ref_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate B_hat, grand mean, and pooled variance.

    Args:
        data: Expression matrix (features x samples)
        design: Design matrix
        batches: List of sample indices for each batch
        n_batch: Number of batches
        ref_idx: Index of reference batch

    Returns:
        B_hat: Coefficient estimates from linear model
        grand_mean: Grand mean per feature
        var_pooled: Pooled variance per feature

    """
    n_features, n_samples = data.shape
    n_batches = [len(b) for b in batches]

    # Solve for B_hat: (X'X)^-1 X'Y, per feature where values are missing (sva's Beta.NA).
    B_hat = _fit_coefficients(data, design)  # noqa: N806 - sva's B.hat, kept comparable

    # Calculate grand mean
    if ref_idx is not None:
        # Use reference batch mean as grand mean
        grand_mean = B_hat[ref_idx, :]
    else:
        # Weighted average of batch means
        weights = np.array(n_batches) / n_samples
        grand_mean = weights @ B_hat[:n_batch, :]

    # Calculate pooled variance
    # Residuals from the model
    predicted = design @ B_hat
    residuals = data.T - predicted

    # nanvar so a missing value does not spread, with ddof=0 to match what sva::ComBat computes on
    # a DENSE input - the normal case for PRISM, since Skyline integrates imputed peak boundaries
    # for every replicate (see "Data Density" in CLAUDE.md).
    #
    # sva itself switches here: sum(residual^2)/n when the input is dense, rowVars(na.rm=TRUE)
    # (ddof=1) when it is not. PRISM deliberately does NOT switch, so it matches sva on dense input
    # and differs by ~0.3% on input with missing values. Following sva's switch would mean one
    # peptide missing from one document shifts every corrected value in the cohort by ~0.3%.
    # Keep in lockstep with ComBat.VarPooledDdof in the C# port.
    var_pooled_ddof = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN / <2 observations -> NaN
        if ref_idx is not None:
            # Use only reference batch for variance estimation
            ref_residuals = residuals[batches[ref_idx], :]
            var_pooled = np.nanvar(ref_residuals, axis=0, ddof=var_pooled_ddof)
        else:
            var_pooled = np.nanvar(residuals, axis=0, ddof=var_pooled_ddof)

    # Handle zero variance
    positive = var_pooled[var_pooled > 0]
    if positive.size > 0:
        var_pooled[~(var_pooled > 0)] = np.median(positive)

    return B_hat, grand_mean, var_pooled


def _standardize_data(
    data: np.ndarray,
    design: np.ndarray,
    B_hat: np.ndarray,  # noqa: N803 - sva's B.hat, kept comparable
    grand_mean: np.ndarray,
    var_pooled: np.ndarray,
    n_batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize data to have zero mean and unit variance.

    Args:
        data: Expression matrix (features x samples)
        design: Design matrix
        B_hat: Coefficient estimates
        grand_mean: Grand mean per feature
        var_pooled: Pooled variance per feature
        n_batch: Number of batches

    Returns:
        s_data: Standardized data
        stand_mean: Standardized mean matrix

    """
    n_features, n_samples = data.shape

    # Calculate stand_mean: grand_mean + covariate effects
    # Zero out batch effects in design
    design_mod = design.copy()
    design_mod[:, :n_batch] = 0

    stand_mean = grand_mean.reshape(-1, 1) + (design_mod @ B_hat).T

    # Standardize
    std_pooled = np.sqrt(var_pooled).reshape(-1, 1)
    s_data = (data - stand_mean) / std_pooled

    return s_data, stand_mean


def _fit_batch_effects(
    s_data: np.ndarray,
    design: np.ndarray,
    batches: list[np.ndarray],
    n_batch: int,
    mean_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate batch effect parameters (gamma_hat and delta_hat).

    Args:
        s_data: Standardized data (features x samples)
        design: Design matrix
        batches: List of sample indices for each batch
        n_batch: Number of batches
        mean_only: If True, only estimate mean effects (no scale)

    Returns:
        gamma_hat: Additive batch effects (n_batch x n_features)
        delta_hat: Multiplicative batch effects (n_batch x n_features)

    """
    n_features = s_data.shape[0]

    # Get batch portion of design matrix
    batch_design = design[:, :n_batch]

    # Solve for gamma_hat (additive effects), ignoring missing values
    gamma_hat = _fit_coefficients(s_data, batch_design)  # n_batch x n_features

    # Per-(batch, feature) observation count: sva's per-row `n <- rowSums(!is.na(sdat))`.
    counts = np.zeros((n_batch, n_features), dtype=int)
    for i, batch_idx in enumerate(batches):
        counts[i, :] = np.sum(~np.isnan(s_data[:, batch_idx]), axis=1)

    # Estimate delta_hat (variance/scale effects), and record which ones the data supports.
    # Fewer than 2 observations, or no spread among them, means there is no scale to estimate:
    # 1.0 is a placeholder meaning "do not rescale", NOT an estimate, so it is flagged and kept
    # out of the prior - feeding it in would bias the shrinkage of every feature in the batch.
    delta_hat = np.ones((n_batch, n_features))
    estimable = np.zeros((n_batch, n_features), dtype=bool)
    if not mean_only:
        for i, batch_idx in enumerate(batches):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                v = np.nanvar(s_data[:, batch_idx], axis=1, ddof=1)
            ok = (counts[i, :] >= 2) & _spread_is_resolvable(v, gamma_hat[i, :])
            delta_hat[i, ok] = v[ok]
            estimable[i, :] = ok

    return gamma_hat, delta_hat, counts, estimable


def _compute_priors(
    gamma_hat: np.ndarray,
    delta_hat: np.ndarray,
    mean_only: bool = False,
    estimable: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute empirical Bayes priors for batch effect parameters.

    Uses method of moments to estimate hyperparameters.

    Args:
        gamma_hat: Additive batch effects (n_batch x n_features)
        delta_hat: Multiplicative batch effects (n_batch x n_features)
        mean_only: If True, skip variance prior estimation
        estimable: Boolean mask of the (batch, feature) scales the data supports; the rest hold a
            placeholder 1.0 and are excluded from the prior. None means treat all as estimable.

    Returns:
        gamma_bar: Prior mean for gamma (n_batch,)
        t2: Prior variance for gamma (n_batch,)
        a_prior: Prior shape for delta (n_batch,)
        b_prior: Prior rate for delta (n_batch,)

    """
    n_batch = gamma_hat.shape[0]

    # Priors for gamma (normal distribution)
    gamma_bar = np.mean(gamma_hat, axis=1)
    t2 = np.var(gamma_hat, axis=1, ddof=1)

    # Priors for delta (inverse gamma distribution)
    if mean_only:
        a_prior = np.ones(n_batch)
        b_prior = np.ones(n_batch)
    else:
        a_prior = np.zeros(n_batch)
        b_prior = np.zeros(n_batch)

        for i in range(n_batch):
            # Method of moments for inverse gamma, over the deltas the data actually supports.
            delta_i = delta_hat[i, estimable[i, :]] if estimable is not None else delta_hat[i, :]
            m = np.mean(delta_i) if delta_i.size > 0 else np.nan
            v = np.var(delta_i, ddof=1) if delta_i.size > 1 else np.nan

            if v > 0 and m > 0:
                # a = (m^2 / v) + 2
                # b = m * ((m^2 / v) + 1)
                a_prior[i] = (m * m / v) + 2
                b_prior[i] = m * ((m * m / v) + 1)
            else:
                a_prior[i] = 1.0
                b_prior[i] = 1.0

    return gamma_bar, t2, a_prior, b_prior


def _postmean(
    g_hat: np.ndarray,
    g_bar: float,
    n: int | np.ndarray,
    d_star: np.ndarray,
    t2: float,
) -> np.ndarray:
    """Calculate posterior mean for gamma.

    Args:
        g_hat: Estimated gamma values
        g_bar: Prior mean
        n: Number of samples in batch
        d_star: Estimated delta values
        t2: Prior variance

    Returns:
        Posterior mean for gamma

    """
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def _postvar(
    sum_sq: np.ndarray,
    n: int | np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    """Calculate posterior parameters for delta.

    Args:
        sum_sq: Sum of squared residuals
        n: Number of samples in batch
        a: Prior shape
        b: Prior rate

    Returns:
        Posterior mean for delta (inverse gamma)

    """
    return (0.5 * sum_sq + b) / (0.5 * n + a - 1)


def _it_sol(
    s_data_batch: np.ndarray,
    g_hat: np.ndarray,
    d_hat: np.ndarray,
    g_bar: float,
    t2: float,
    a: float,
    b: float,
    conv: float = 0.0001,
    max_iter: int = 100,
    counts: np.ndarray | None = None,
    estimable: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve iteratively for the parametric empirical-Bayes estimates.

    Iterates between updating gamma and delta estimates until convergence.

    Args:
        s_data_batch: Standardized data for this batch (features x samples_in_batch)
        g_hat: Initial gamma estimate
        d_hat: Initial delta estimate
        g_bar: Prior mean for gamma
        t2: Prior variance for gamma
        a: Prior shape for delta
        b: Prior rate for delta
        conv: Convergence threshold
        max_iter: Maximum iterations
        counts: Per-(batch, feature) count of OBSERVED values, sva's
            ``n <- rowSums(!is.na(sdat))``. None means "every feature was seen in every
            sample of the batch", i.e. the scalar batch size.
        estimable: Per-feature mask of the scales the data supports. Where False the scale
            is left at 1.0 (not rescaled) instead of being estimated from too few points.

    Returns:
        gamma_star: EB-adjusted gamma
        delta_star: EB-adjusted delta

    """
    # Per-FEATURE observation count, as sva's `n <- rowSums(!is.na(sdat))`. With nothing missing
    # this is the batch size for every feature, i.e. the scalar this used to be.
    n = counts if counts is not None else np.full(s_data_batch.shape[0], s_data_batch.shape[1])
    g_old = g_hat.copy()
    d_old = d_hat.copy()

    for _ in range(max_iter):
        # Update gamma
        g_new = _postmean(g_hat, g_bar, n, d_old, t2)

        # Update delta
        # Sum of squared residuals after removing gamma effect (missing values skipped)
        residuals = s_data_batch - g_new.reshape(-1, 1)
        sum_sq = np.nansum(residuals**2, axis=1)
        d_new = _postvar(sum_sq, n, a, b)
        if estimable is not None:
            # No spread to estimate from -> leave this (batch, feature) unscaled.
            d_new = np.where(estimable, d_new, 1.0)

        # Check convergence
        g_change = np.max(np.abs(g_new - g_old) / (np.abs(g_old) + 1e-10))
        d_change = np.max(np.abs(d_new - d_old) / (np.abs(d_old) + 1e-10))

        if g_change < conv and d_change < conv:
            break

        g_old = g_new
        d_old = d_new

    return g_new, d_new


def _int_eprior(
    s_data_batch: np.ndarray,
    g_hat: np.ndarray,
    d_hat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Non-parametric empirical Bayes estimation.

    Uses kernel density estimation for the priors instead of
    parametric assumptions.

    Args:
        s_data_batch: Standardized data for this batch
        g_hat: Initial gamma estimate
        d_hat: Initial delta estimate

    Returns:
        gamma_star: EB-adjusted gamma
        delta_star: EB-adjusted delta

    """
    # For non-parametric, we use a simpler approach:
    # weighted average between the estimate and the overall mean
    g_bar = np.mean(g_hat)
    d_bar = np.mean(d_hat)

    # Shrinkage based on relative variance
    g_var = np.var(g_hat, ddof=1)
    d_var = np.var(d_hat, ddof=1)

    if g_var > 0:
        shrink_g = g_var / (g_var + 1)
    else:
        shrink_g = 0.5

    if d_var > 0:
        shrink_d = d_var / (d_var + 1)
    else:
        shrink_d = 0.5

    gamma_star = shrink_g * g_hat + (1 - shrink_g) * g_bar
    delta_star = shrink_d * d_hat + (1 - shrink_d) * d_bar

    return gamma_star, delta_star


def _adjust_data(
    s_data: np.ndarray,
    gamma_star: np.ndarray,
    delta_star: np.ndarray,
    batches: list[np.ndarray],
    var_pooled: np.ndarray,
    stand_mean: np.ndarray,
    ref_idx: int | None = None,
    data_orig: np.ndarray | None = None,
) -> np.ndarray:
    """Apply batch correction to standardized data.

    Args:
        s_data: Standardized data (features x samples)
        gamma_star: EB-adjusted additive effects
        delta_star: EB-adjusted multiplicative effects
        batches: List of sample indices for each batch
        var_pooled: Pooled variance per feature
        stand_mean: Standardized mean matrix
        ref_idx: Reference batch index (won't be adjusted)
        data_orig: Original data (for reference batch)

    Returns:
        Batch-corrected data in original scale

    """
    n_features, n_samples = s_data.shape
    bayes_data = s_data.copy()

    # Adjust each batch
    for i, batch_idx in enumerate(batches):
        if ref_idx is not None and i == ref_idx:
            # Don't adjust reference batch
            continue

        # Remove additive effect and scale by multiplicative effect
        gamma_i = gamma_star[i, :].reshape(-1, 1)
        delta_i = np.sqrt(delta_star[i, :]).reshape(-1, 1)

        bayes_data[:, batch_idx] = (bayes_data[:, batch_idx] - gamma_i) / delta_i

    # Transform back to original scale
    std_pooled = np.sqrt(var_pooled).reshape(-1, 1)
    bayes_data = bayes_data * std_pooled + stand_mean

    # Restore reference batch to original values
    if ref_idx is not None and data_orig is not None:
        bayes_data[:, batches[ref_idx]] = data_orig[:, batches[ref_idx]]

    return bayes_data


def combat(
    data: np.ndarray | pd.DataFrame,
    batch: np.ndarray | list | pd.Series,
    covar_mod: np.ndarray | pd.DataFrame | None = None,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: int | str | None = None,
) -> np.ndarray | pd.DataFrame:
    """Adjust for batch effects using ComBat (empirical Bayes).

    ComBat removes batch effects while preserving biological variation.
    It estimates additive (location) and multiplicative (scale) batch
    effects using empirical Bayes shrinkage for robust estimation.

    Args:
        data: Expression matrix with features as rows and samples as columns.
            Can be a numpy array or pandas DataFrame.
        batch: Batch labels for each sample. Must have same length as number
            of columns in data.
        covar_mod: Optional model matrix for biological covariates to preserve.
            Should have samples as rows and covariates as columns.
        par_prior: If True, use parametric empirical Bayes. If False, use
            non-parametric estimation (slower but more flexible).
        mean_only: If True, only correct mean (location) effects, not
            variance (scale) effects. Useful when batches have different
            sample sizes.
        ref_batch: Optional reference batch. This batch will not be adjusted,
            and other batches will be adjusted to match it.

    Returns:
        Batch-corrected data in the same format as input (array or DataFrame).

    Raises:
        ValueError: If inputs are invalid (wrong dimensions, single-sample batches, etc.)

    Example:
        >>> import numpy as np
        >>> # Simulated data with batch effects
        >>> data = np.random.randn(100, 10)  # 100 features, 10 samples
        >>> data[:, 5:] += 2  # Add batch effect to second batch
        >>> batch = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> corrected = combat(data, batch)

    References:
        Johnson WE, Li C, Rabinovic A. (2007) Adjusting batch effects in
        microarray expression data using empirical Bayes methods.
        Biostatistics, 8(1), 118-127.

    """
    # Handle pandas input
    is_dataframe = isinstance(data, pd.DataFrame)
    if is_dataframe:
        feature_names = data.index
        sample_names = data.columns
        data_array = data.values
    else:
        feature_names = None
        sample_names = None
        data_array = data

    if isinstance(batch, pd.Series):
        batch = batch.values
    batch = np.asarray(batch)

    if isinstance(covar_mod, pd.DataFrame):
        covar_mod = covar_mod.values

    # Validate inputs
    data_array, batch, covar_mod = _check_inputs(data_array, batch, covar_mod)

    n_features, n_samples = data_array.shape
    logger.info(f"ComBat: Processing {n_features} features and {n_samples} samples")

    # Build design matrix
    design, batches, n_batch, ref_idx = _make_design_matrix(batch, covar_mod, ref_batch)
    logger.info(f"Found {n_batch} batches")

    # Hold out features ComBat cannot estimate anything for: no variance at all, or no
    # observation in some batch (that batch's effect on them is then undefined). They are
    # returned untouched, as sva returns its zero-variance rows.
    observed = ~np.isnan(data_array)
    in_every_batch = np.all(
        np.stack([observed[:, bi].any(axis=1) for bi in batches], axis=0), axis=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows
        row_vars = np.nanvar(data_array, axis=1)
    zero_var_mask = ~(in_every_batch & np.isfinite(row_vars) & (row_vars != 0))
    if np.any(zero_var_mask):
        n_zero = int(np.sum(zero_var_mask))
        logger.warning(
            f"ComBat: {n_zero} feature(s) held out (no variance, or absent from a batch); "
            "passed through uncorrected."
        )
        # Keep original data for held-out features
        data_orig_zero = data_array[zero_var_mask, :].copy()
        data_array = data_array[~zero_var_mask, :]

    n_batches = [len(b) for b in batches]

    # Check for mean_only condition
    if any(n == 1 for n in n_batches):
        logger.info("Batch with single sample found, using mean_only=True")
        mean_only = True

    # Calculate mean and variance
    B_hat, grand_mean, var_pooled = _calculate_mean_var(  # noqa: N806 - sva's B.hat
        data_array, design, batches, n_batch, ref_idx
    )

    # Standardize data
    s_data, stand_mean = _standardize_data(
        data_array, design, B_hat, grand_mean, var_pooled, n_batch
    )

    # Fit batch effects
    logger.info("Fitting L/S model and finding priors")
    gamma_hat, delta_hat, counts, estimable = _fit_batch_effects(
        s_data, design, batches, n_batch, mean_only
    )
    if not mean_only:
        n_unestimable = int(np.sum(~estimable))
        if n_unestimable > 0:
            logger.info(
                f"ComBat: {n_unestimable} (batch, feature) scale(s) not estimable "
                "(< 2 observations, or no spread); location corrected only."
            )

    # Compute priors
    gamma_bar, t2, a_prior, b_prior = _compute_priors(gamma_hat, delta_hat, mean_only, estimable)

    # EB estimation
    gamma_star = np.zeros_like(gamma_hat)
    delta_star = np.zeros_like(delta_hat)

    if par_prior:
        logger.info("Finding parametric adjustments")
        for i, batch_idx in enumerate(batches):
            if mean_only:
                # Simple posterior mean for gamma
                n_i = len(batch_idx)
                gamma_star[i, :] = _postmean(
                    gamma_hat[i, :], gamma_bar[i], n_i, np.ones(gamma_hat.shape[1]), t2[i]
                )
                delta_star[i, :] = np.ones(delta_hat.shape[1])
            else:
                # Iterative EB solution
                gamma_star[i, :], delta_star[i, :] = _it_sol(
                    s_data[:, batch_idx],
                    gamma_hat[i, :],
                    delta_hat[i, :],
                    gamma_bar[i],
                    t2[i],
                    a_prior[i],
                    b_prior[i],
                    counts=counts[i, :],
                    estimable=estimable[i, :],
                )
    else:
        logger.info("Finding non-parametric adjustments")
        for i, batch_idx in enumerate(batches):
            if mean_only:
                delta_hat[i, :] = 1
            gamma_star[i, :], delta_star[i, :] = _int_eprior(
                s_data[:, batch_idx],
                gamma_hat[i, :],
                delta_hat[i, :],
            )

    # Handle reference batch
    if ref_idx is not None:
        gamma_star[ref_idx, :] = 0
        delta_star[ref_idx, :] = 1

    # Adjust data
    logger.info("Adjusting the data")
    bayes_data = _adjust_data(
        s_data, gamma_star, delta_star, batches, var_pooled, stand_mean, ref_idx, data_array
    )

    # Restore zero-variance features
    if np.any(zero_var_mask):
        full_data = np.zeros((len(zero_var_mask), n_samples))
        full_data[~zero_var_mask, :] = bayes_data
        full_data[zero_var_mask, :] = data_orig_zero
        bayes_data = full_data

    # Return in original format
    if is_dataframe:
        return pd.DataFrame(bayes_data, index=feature_names, columns=sample_names)
    else:
        return bayes_data


def combat_reference_anchored(
    data: np.ndarray | pd.DataFrame,
    batch: np.ndarray | list | pd.Series,
    reference_mask: np.ndarray | list | pd.Series,
    *,
    par_prior: bool = True,
    no_reference_batch: str = "fallback",
) -> np.ndarray | pd.DataFrame:
    """Reference-anchored ComBat: single-point calibration with EB shrinkage.

    Performs per-analyte single-point external reference calibration (Pino et al.,
    PMC6854904) stabilized by ComBat's empirical-Bayes shrinkage (Johnson et al.
    2007). Unlike standard ``combat``, which aligns every batch to the across-batch
    grand mean (assuming equal biological composition per batch), this estimates the
    per-batch technical effect from the **reference samples only** (identical material
    run in every batch) and applies the correction to all samples. Because the
    reference is identical material, any per-batch difference in it is purely
    technical, so no biology is removed.

    Model per feature ``g``, batch ``i`` (log2 abundance ``Y``):
        ``alpha[g]``     = pooled mean of Y[g] over all reference samples (all batches)
        ``gamma[i,g]``   = additive offset: mean over refs in batch i minus alpha[g]
        ``delta[i,g]``   = multiplicative scale: spread of the standardized reference
                           replicates in batch i (estimated only when batch i has >=2
                           references; otherwise 1, i.e. location-only for that batch)

    Both effects are empirical-Bayes shrunk **across features within a batch** so a
    feature poorly measured in the references borrows strength from the batch-wide
    consensus. The location shrinkage weight scales with the number of reference
    replicates in the batch (heavy shrinkage with one reference, lighter as replicates
    grow). The correction is applied to all samples; output is calibrated absolute
    log2 abundance on the input scale (not a ratio to the reference).

    Args:
        data: Expression matrix (features x samples), log2 scale. Array or DataFrame.
        batch: Batch label per sample (length = n_samples).
        reference_mask: Boolean per sample, True for inter-batch reference samples.
        par_prior: Use parametric empirical Bayes (recommended).
        no_reference_batch: Behavior for a batch with zero reference samples:
            "fallback" (default) corrects it with standard grand-mean estimation and
            warns; "skip" leaves it uncorrected and warns; "error" raises.

    Returns:
        Batch-corrected matrix in the same format as the input, on the input log2 scale.

    Raises:
        ValueError: If inputs are inconsistent, no reference samples are present, or a
            batch lacks references and ``no_reference_batch="error"``.

    References:
        Pino LK, et al. (2020) Calibration using a single-point external reference
            material harmonizes quantitative mass spectrometry proteomics data between
            platforms and laboratories. (PMC6854904)
        Johnson WE, Li C, Rabinovic A. (2007) Adjusting batch effects in microarray
            expression data using empirical Bayes methods. Biostatistics 8(1):118-127.

    """
    is_dataframe = isinstance(data, pd.DataFrame)
    if is_dataframe:
        feature_names = data.index
        sample_names = data.columns
        data_array = data.values.astype(np.float64)
    else:
        feature_names = None
        sample_names = None
        data_array = np.asarray(data, dtype=np.float64)

    if isinstance(batch, pd.Series):
        batch = batch.values
    batch = np.asarray(batch)
    if isinstance(reference_mask, pd.Series):
        reference_mask = reference_mask.values
    reference_mask = np.asarray(reference_mask, dtype=bool)

    if data_array.ndim != 2:
        raise ValueError("Data must be a 2D matrix (features x samples)")
    n_features, n_samples = data_array.shape
    if len(batch) != n_samples:
        raise ValueError(f"Batch length ({len(batch)}) must match number of samples ({n_samples})")
    if len(reference_mask) != n_samples:
        raise ValueError(
            f"reference_mask length ({len(reference_mask)}) must match "
            f"number of samples ({n_samples})"
        )
    if no_reference_batch not in ("fallback", "skip", "error"):
        raise ValueError(
            f"no_reference_batch must be 'fallback', 'skip', or 'error', "
            f"got {no_reference_batch!r}"
        )

    def _return(arr: np.ndarray) -> np.ndarray | pd.DataFrame:
        if is_dataframe:
            return pd.DataFrame(arr, index=feature_names, columns=sample_names)
        return arr

    if not reference_mask.any():
        raise ValueError(
            "Reference-anchored ComBat requires at least one reference sample "
            "(reference_mask is all False)."
        )

    # Sorted unique labels, as standard combat uses (np.unique order). Batch ORDER is not
    # statistically meaningful here - every effect is estimated per batch - but it fixes the
    # summation order of the pooled center below, so both engines must agree on it.
    unique_batches = sorted(set(batch.tolist()))
    n_batch = len(unique_batches)
    if n_batch < 2:
        logger.warning("Only one batch found - returning data unchanged")
        return _return(data_array)

    ref_batch_labels = set(batch[reference_mask].tolist())
    if len(ref_batch_labels) < 2:
        logger.warning(
            "Reference samples are present in only one batch; reference-anchored "
            "calibration cannot align batches and is effectively a no-op for other batches."
        )

    logger.info(
        f"Reference-anchored ComBat: {n_features} features, {n_samples} samples, "
        f"{n_batch} batches, {int(reference_mask.sum())} reference samples"
    )

    # Per-batch sample / reference index lists
    batch_indices = [np.where(batch == b)[0] for b in unique_batches]
    batch_ref_indices = [np.where((batch == b) & reference_mask)[0] for b in unique_batches]
    n_ref_per_batch = np.array([len(r) for r in batch_ref_indices], dtype=int)

    if np.any(n_ref_per_batch == 0):
        missing = [unique_batches[i] for i in range(n_batch) if n_ref_per_batch[i] == 0]
        if no_reference_batch == "error":
            raise ValueError(
                f"Batches {missing} have no reference samples; cannot reference-anchor. "
                "Provide references in every batch or set no_reference_batch='fallback'/'skip'."
            )
        logger.warning(
            f"Batches {missing} have no reference samples. "
            + (
                "Correcting them with standard grand-mean estimation (not reference-anchored)."
                if no_reference_batch == "fallback"
                else "Leaving them uncorrected."
            )
        )

    # The FIT SET: which samples estimate each batch's effect. This, and not a different
    # estimator, is the whole of the difference from standard ComBat - see the C# ComBatPlan,
    # which both engines' reference-anchored and standard paths are now expressed in terms of.
    #   >= 1 reference   -> the batch's references (location only when there is just one, since
    #                       a single reference is a level, not a spread)
    #   0, "fallback"    -> the batch's own samples, LOCATION ONLY: their spread is biological,
    #                       and estimating a scale from it would shrink real signal
    #   0, "skip"        -> empty; the batch is left exactly as it came in
    fit_indices: list[np.ndarray] = []
    location_only = np.zeros(n_batch, dtype=bool)
    for bi in range(n_batch):
        if n_ref_per_batch[bi] >= 1:
            fit_indices.append(batch_ref_indices[bi])
            location_only[bi] = n_ref_per_batch[bi] < 2
        elif no_reference_batch == "fallback":
            fit_indices.append(batch_indices[bi])
            location_only[bi] = True
        else:
            fit_indices.append(np.empty(0, dtype=int))
    fitted = [bi for bi in range(n_batch) if fit_indices[bi].size > 0]

    # Hold out what the data does not determine: a feature that never varies, or that some
    # FITTED batch never observed in its fit set - that batch's effect on it is then undefined,
    # which is not the same as it being zero. Batches we are not fitting ("skip") get no vote.
    row_vars = np.nanvar(data_array, axis=1)
    held_out_mask = (row_vars == 0) | np.isnan(row_vars)
    for bi in fitted:
        held_out_mask |= np.all(np.isnan(data_array[:, fit_indices[bi]]), axis=1)
    if np.any(held_out_mask):
        logger.warning(
            f"Found {int(held_out_mask.sum())} features with no variance or no observation in "
            "some batch's reference samples; passing them through uncorrected."
        )
    work = data_array[~held_out_mask, :]
    n_work = work.shape[0]

    # alpha[g]: the level everything is measured against - the fit sets' pooled mean, weighted
    # by fit-set size. With references in every batch this is the pooled reference level.
    total_fit = sum(fit_indices[bi].size for bi in fitted)
    alpha = np.zeros(n_work)
    fit_means = np.zeros((n_batch, n_work))
    for bi in fitted:
        fit_means[bi, :] = np.nanmean(work[:, fit_indices[bi]], axis=1)
        alpha += (fit_indices[bi].size / total_fit) * fit_means[bi, :]

    # var_pooled[g]: the standardization scale. For location+scale to be coherent it
    # must be the TECHNICAL variance, i.e. the pooled within-batch variance of the
    # reference replicates (mirrors standard ComBat, where var_pooled is the pooled
    # within-batch residual variance). Estimated from batches with >=2 references.
    # Standardizing by this makes delta_hat (reference-replicate variance / var_pooled)
    # average ~1 across batches, so dividing by sqrt(delta) harmonizes technical
    # dispersion rather than inflating the data.
    sumsq = np.zeros(n_work)
    pooled_df = 0
    for bi in fitted:
        if location_only[bi] or fit_indices[bi].size < 2:
            continue  # no replicates here to pool
        cols = work[:, fit_indices[bi]]
        resid = cols - fit_means[bi, :][:, None]
        sumsq += np.nansum(resid**2, axis=1)
        # Nominal, not per-feature-observed: one scalar df for the whole matrix, so a feature
        # missing from one replicate does not get a different denominator from its neighbours.
        pooled_df += fit_indices[bi].size - 1

    if pooled_df > 0:
        var_pooled = sumsq / pooled_df
    else:
        # No fit set has replicates: technical variance is not estimable, so scale
        # correction is impossible (delta stays 1 below). Use the all-sample residual
        # variance purely as a homoscedastic scale for the location EB prior; it cancels
        # on back-transform and only affects shrinkage weighting.
        residual = work - alpha[:, None]
        for bi in range(n_batch):
            idx = batch_indices[bi]
            gamma_raw = fit_means[bi, :] - alpha if fit_indices[bi].size > 0 else 0.0
            residual[:, idx] = residual[:, idx] - np.reshape(gamma_raw, (-1, 1))
        var_pooled = np.nanvar(residual, axis=1, ddof=1)

    usable = (var_pooled > 0) & np.isfinite(var_pooled)
    var_pooled[~usable] = np.median(var_pooled[usable]) if usable.any() else 1.0
    std_pooled = np.sqrt(var_pooled)

    # Standardized data, and the batch effects estimated from the fit sets only.
    s_data = (work - alpha[:, None]) / std_pooled[:, None]

    gamma_hat = np.zeros((n_batch, n_work))
    delta_hat = np.ones((n_batch, n_work))
    counts = np.zeros((n_batch, n_work), dtype=int)
    estimable = np.zeros((n_batch, n_work), dtype=bool)
    for bi in fitted:
        fit_cols = s_data[:, fit_indices[bi]]
        gamma_hat[bi, :] = np.nanmean(fit_cols, axis=1)
        counts[bi, :] = np.sum(~np.isnan(fit_cols), axis=1)
        if location_only[bi]:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            v = np.nanvar(fit_cols, axis=1, ddof=1)
        # 1.0 means "do not rescale", NOT an estimate: a (batch, feature) with fewer than two
        # observations or no resolvable spread is flagged and kept out of the prior, because a
        # placeholder inside a mean taken ACROSS features biases every feature in the batch.
        ok = (counts[bi, :] >= 2) & _spread_is_resolvable(v, gamma_hat[bi, :])
        delta_hat[bi, ok] = v[ok]
        estimable[bi, :] = ok

    # Empirical-Bayes priors. _compute_priors derives gamma_bar/t2 (and a/b for delta)
    # PER BATCH from that batch's own features, so anchored and fallback batches never
    # share a prior pool even though they are computed in one call.
    gamma_bar, t2, a_prior, b_prior = _compute_priors(
        gamma_hat, delta_hat, mean_only=False, estimable=estimable
    )

    gamma_star = np.zeros_like(gamma_hat)
    delta_star = np.ones_like(delta_hat)

    for bi in range(n_batch):
        if fit_indices[bi].size == 0:
            continue  # "skip": gamma* = 0, delta* = 1
        if location_only[bi]:
            # No spread to estimate a scale from. n is the fit set's size, which is what
            # decides how far this batch's own estimate is trusted against the prior.
            gamma_star[bi, :] = _postmean(
                gamma_hat[bi, :], gamma_bar[bi], fit_indices[bi].size, np.ones(n_work), t2[bi]
            )
            continue
        # Joint location+scale EB on the fit columns only, so the iterative delta update
        # sees reference replicate spread, not experimental biology.
        if par_prior:
            g_star, d_star = _it_sol(
                s_data[:, fit_indices[bi]],
                gamma_hat[bi, :],
                delta_hat[bi, :],
                gamma_bar[bi],
                t2[bi],
                a_prior[bi],
                b_prior[bi],
                counts=counts[bi, :],
                estimable=estimable[bi, :],
            )
        else:
            g_star, d_star = _int_eprior(
                s_data[:, fit_indices[bi]],
                gamma_hat[bi, :],
                delta_hat[bi, :],
            )
        gamma_star[bi, :] = g_star
        delta_star[bi, :] = d_star

    # Apply correction to ALL samples (stand_mean = alpha, broadcast across samples)
    bayes_work = _adjust_data(
        s_data,
        gamma_star,
        delta_star,
        batch_indices,
        var_pooled,
        alpha.reshape(-1, 1),
    )

    # Restore held-out features
    if np.any(held_out_mask):
        full = np.empty((n_features, n_samples), dtype=np.float64)
        full[~held_out_mask, :] = bayes_work
        full[held_out_mask, :] = data_array[held_out_mask, :]
        bayes_work = full

    return _return(bayes_work)


def combat_from_long(
    data: pd.DataFrame,
    abundance_col: str = "abundance",
    feature_col: str = "precursor_id",
    sample_col: str = "replicate_name",
    batch_col: str = "batch",
    covar_cols: list[str] | None = None,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: int | str | None = None,
) -> pd.DataFrame:
    """Apply ComBat to data in long format (as used in skyline-prism).

    This is a convenience wrapper that pivots long-format data to wide format,
    applies ComBat, and returns the corrected data in long format.

    Args:
        data: Long-format DataFrame with columns for abundance, features,
            samples, and batch.
        abundance_col: Column name for abundance values
        feature_col: Column name for feature identifiers (e.g., precursor_id)
        sample_col: Column name for sample identifiers
        batch_col: Column name for batch labels
        covar_cols: Optional list of covariate column names to preserve
        par_prior: Use parametric empirical Bayes
        mean_only: Only correct location effects
        ref_batch: Reference batch (won't be adjusted)

    Returns:
        Long-format DataFrame with batch-corrected abundances

    """
    # Pivot to wide format (features x samples)
    wide = data.pivot_table(
        index=feature_col,
        columns=sample_col,
        values=abundance_col,
        aggfunc="first",  # Should be unique
    )

    # Get batch labels in same order as columns
    sample_info = data[[sample_col, batch_col]].drop_duplicates()
    sample_to_batch = dict(zip(sample_info[sample_col], sample_info[batch_col]))
    batch = [sample_to_batch[s] for s in wide.columns]

    # Get covariates if specified
    covar_mod = None
    if covar_cols:
        covar_data = data[[sample_col] + covar_cols].drop_duplicates()
        covar_mod = pd.get_dummies(covar_data.set_index(sample_col)[covar_cols], drop_first=True)
        covar_mod = covar_mod.loc[wide.columns].values

    # Apply ComBat
    corrected_wide = combat(
        wide,
        batch,
        covar_mod=covar_mod,
        par_prior=par_prior,
        mean_only=mean_only,
        ref_batch=ref_batch,
    )

    # Melt back to long format
    corrected_long = corrected_wide.reset_index().melt(
        id_vars=[feature_col], var_name=sample_col, value_name=f"{abundance_col}_batch_corrected"
    )

    # Merge with original data
    result = data.merge(corrected_long, on=[feature_col, sample_col], how="left")

    return result


@dataclass
class BatchCorrectionEvaluation:
    """Evaluation metrics for batch correction quality.

    Uses reference samples (inter-batch QC) and QC samples (intra-batch QC)
    to assess whether batch correction improves data quality without overfitting.

    Attributes:
        reference_cv_before: Median CV of reference samples before correction
        reference_cv_after: Median CV of reference samples after correction
        qc_cv_before: Median CV of QC samples before correction
        qc_cv_after: Median CV of QC samples after correction
        reference_improvement: Fractional reduction in reference CV
        qc_improvement: Fractional reduction in QC CV
        overfitting_ratio: Ratio of reference to QC improvement (should be ~1)
        batch_variance_before: Variance of batch means before correction
        batch_variance_after: Variance of batch means after correction
        passed: Whether correction meets quality thresholds
        warnings: List of warning messages

    """

    reference_cv_before: float
    reference_cv_after: float
    qc_cv_before: float
    qc_cv_after: float
    reference_improvement: float
    qc_improvement: float
    overfitting_ratio: float
    batch_variance_before: float
    batch_variance_after: float
    passed: bool
    warnings: list[str]


def _calculate_sample_cv(
    data: pd.DataFrame,
    sample_mask: pd.Series,
    abundance_col: str,
    feature_col: str,
    sample_col: str,
) -> float:
    """Calculate median CV across features for a subset of samples.

    Args:
        data: Long-format DataFrame
        sample_mask: Boolean mask for samples to include
        abundance_col: Column with abundance values
        feature_col: Column with feature identifiers
        sample_col: Column with sample identifiers

    Returns:
        Median coefficient of variation

    """
    subset = data.loc[sample_mask]

    if len(subset) == 0:
        return np.nan

    # Pivot to wide format (features x samples)
    matrix = subset.pivot_table(
        index=feature_col,
        columns=sample_col,
        values=abundance_col,
    )

    if matrix.shape[1] < 2:
        return np.nan

    # Calculate CV per feature (on linear scale if log2)
    # Assume data is log2-transformed
    linear = np.power(2, matrix)
    cv_per_feature = linear.std(axis=1) / linear.mean(axis=1)

    return float(cv_per_feature.median())


def _calculate_batch_variance(
    data: pd.DataFrame,
    abundance_col: str,
    batch_col: str,
) -> float:
    """Calculate variance of batch means.

    A measure of batch effect magnitude - lower is better.

    Args:
        data: Long-format DataFrame
        abundance_col: Column with abundance values
        batch_col: Column with batch labels

    Returns:
        Variance of per-batch mean abundances

    """
    batch_means = data.groupby(batch_col)[abundance_col].mean()
    return float(np.var(batch_means))


def evaluate_batch_correction(
    data: pd.DataFrame,
    abundance_before: str,
    abundance_after: str,
    sample_type_col: str = "sample_type",
    feature_col: str = "precursor_id",
    sample_col: str = "replicate_name",
    batch_col: str = "batch",
    reference_type: str = "reference",
    qc_type: str = "qc",
    max_overfitting_ratio: float = 2.0,
    min_qc_improvement: float = 0.0,
) -> BatchCorrectionEvaluation:
    """Evaluate batch correction quality using reference and QC samples.

    This function compares the coefficient of variation (CV) of reference
    and QC samples before and after batch correction. A good batch
    correction should:

    1. Reduce CV of reference samples (they're the same material across batches)
    2. Reduce or maintain CV of QC samples (independent QC)
    3. Not improve reference much more than QC (would indicate overfitting)

    Args:
        data: Long-format DataFrame with before and after abundances
        abundance_before: Column name for abundances before correction
        abundance_after: Column name for abundances after correction
        sample_type_col: Column indicating sample type
        feature_col: Column with feature identifiers
        sample_col: Column with sample identifiers
        batch_col: Column with batch labels
        reference_type: Value in sample_type_col for reference samples
        qc_type: Value in sample_type_col for QC samples
        max_overfitting_ratio: Maximum allowed ratio of reference/QC improvement
        min_qc_improvement: Minimum required improvement in QC CV

    Returns:
        BatchCorrectionEvaluation with metrics and pass/fail status

    Example:
        >>> result = combat_from_long(data, ...)
        >>> eval_result = evaluate_batch_correction(
        ...     result,
        ...     abundance_before='abundance',
        ...     abundance_after='abundance_batch_corrected'
        ... )
        >>> if not eval_result.passed:
        ...     print("Warning:", eval_result.warnings)

    """
    warnings = []

    # Create masks for sample types
    reference_mask = data[sample_type_col] == reference_type
    qc_mask = data[sample_type_col] == qc_type

    n_reference = data.loc[reference_mask, sample_col].nunique()
    n_qc = data.loc[qc_mask, sample_col].nunique()

    logger.info(f"Evaluating batch correction with {n_reference} reference and {n_qc} QC samples")

    if n_reference < 2:
        warnings.append(f"Only {n_reference} reference samples - cannot calculate CV")
    if n_qc < 2:
        warnings.append(f"Only {n_qc} QC samples - cannot calculate CV")

    # Calculate CVs before and after
    ref_cv_before = _calculate_sample_cv(
        data, reference_mask, abundance_before, feature_col, sample_col
    )
    ref_cv_after = _calculate_sample_cv(
        data, reference_mask, abundance_after, feature_col, sample_col
    )
    qc_cv_before = _calculate_sample_cv(data, qc_mask, abundance_before, feature_col, sample_col)
    qc_cv_after = _calculate_sample_cv(data, qc_mask, abundance_after, feature_col, sample_col)

    # Calculate improvements (positive = better)
    ref_improvement = (ref_cv_before - ref_cv_after) / ref_cv_before if ref_cv_before > 0 else 0
    qc_improvement = (qc_cv_before - qc_cv_after) / qc_cv_before if qc_cv_before > 0 else 0

    # Calculate overfitting ratio
    if qc_improvement > 0:
        overfitting_ratio = ref_improvement / qc_improvement
    elif ref_improvement > 0:
        overfitting_ratio = np.inf  # Reference improved but QC didn't
        warnings.append("Reference CV improved but QC CV did not - possible overfitting")
    else:
        overfitting_ratio = 1.0  # Neither improved

    # Calculate batch variance
    batch_var_before = _calculate_batch_variance(data, abundance_before, batch_col)
    batch_var_after = _calculate_batch_variance(data, abundance_after, batch_col)

    # Determine pass/fail
    passed = True

    if qc_improvement < min_qc_improvement:
        passed = False
        warnings.append(
            f"QC CV did not improve ({qc_improvement:.1%} vs required {min_qc_improvement:.1%})"
        )

    # Observation only - it must NOT fail the evaluation, because failing it makes
    # combat_with_reference_samples(fallback_on_failure=True) throw the whole correction away.
    # Reference and QC are different materials injected at different amounts, so whichever started
    # with more excess variance has more of it to remove; the ratio cannot establish overfitting,
    # and discarding a correction over it silently changes every abundance.
    if np.isfinite(overfitting_ratio) and overfitting_ratio > max_overfitting_ratio:
        warnings.append(
            f"NOTE (not a failure): reference improved {ref_improvement:.1%} but QC only "
            f"{qc_improvement:.1%} (ratio {overfitting_ratio:.1f}) - the controls improved by "
            "very different amounts"
        )

    if qc_cv_after > qc_cv_before * 1.1:  # QC got worse by >10%
        passed = False
        warnings.append(f"QC CV increased from {qc_cv_before:.3f} to {qc_cv_after:.3f}")

    logger.info(f"Reference CV: {ref_cv_before:.3f} -> {ref_cv_after:.3f} ({ref_improvement:+.1%})")
    logger.info(f"QC CV: {qc_cv_before:.3f} -> {qc_cv_after:.3f} ({qc_improvement:+.1%})")
    logger.info(f"Batch variance: {batch_var_before:.4f} -> {batch_var_after:.4f}")
    logger.info(f"Evaluation {'PASSED' if passed else 'FAILED'}")

    return BatchCorrectionEvaluation(
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


def combat_with_reference_samples(
    data: pd.DataFrame,
    abundance_col: str = "abundance",
    feature_col: str = "precursor_id",
    sample_col: str = "replicate_name",
    batch_col: str = "batch",
    sample_type_col: str = "sample_type",
    reference_type: str = "reference",
    qc_type: str = "qc",
    par_prior: bool = True,
    mean_only: bool = False,
    evaluate: bool = True,
    fallback_on_failure: bool = True,
) -> tuple[pd.DataFrame, BatchCorrectionEvaluation | None]:
    """Apply ComBat with automatic evaluation using reference/QC samples.

    This is the recommended entry point for batch correction in PRISM workflows.
    It applies ComBat and automatically evaluates the results using reference
    samples (inter-batch QC) and QC samples (intra-batch QC).

    The dual-control evaluation ensures:
    - Reference samples (same material across batches) show reduced variance
    - QC samples (independent QC) also benefit, not just references
    - No overfitting to reference samples

    If evaluation fails (overfitting detected or QC CV increases), the function
    will fall back to using the original uncorrected abundances (when
    fallback_on_failure=True).

    Args:
        data: Long-format DataFrame with peptide/protein data
        abundance_col: Column with abundance values (log2-transformed)
        feature_col: Column with feature identifiers (precursor_id, protein, etc.)
        sample_col: Column with sample identifiers
        batch_col: Column with batch labels
        sample_type_col: Column indicating sample type
        reference_type: Value for inter-batch reference samples
        qc_type: Value for intra-batch QC samples
        par_prior: Use parametric empirical Bayes (recommended)
        mean_only: Only correct location effects, not scale
        evaluate: Whether to run evaluation (requires reference and QC samples)
        fallback_on_failure: If True and evaluation fails, use uncorrected data

    Returns:
        Tuple of:
        - DataFrame with '{abundance_col}_batch_corrected' column (may be copy of
          original abundance if evaluation failed and fallback_on_failure=True)
        - BatchCorrectionEvaluation (or None if evaluate=False)

    Example:
        >>> corrected, evaluation = combat_with_reference_samples(
        ...     data,
        ...     abundance_col='abundance',
        ...     batch_col='batch'
        ... )
        >>> if evaluation and not evaluation.passed:
        ...     print("Batch correction failed QC - using uncorrected data")

    """
    # Apply ComBat
    corrected = combat_from_long(
        data,
        abundance_col=abundance_col,
        feature_col=feature_col,
        sample_col=sample_col,
        batch_col=batch_col,
        par_prior=par_prior,
        mean_only=mean_only,
    )

    evaluation = None
    if evaluate:
        # Check if we have reference and QC samples
        has_reference = (corrected[sample_type_col] == reference_type).any()
        has_qc = (corrected[sample_type_col] == qc_type).any()

        if has_reference and has_qc:
            evaluation = evaluate_batch_correction(
                corrected,
                abundance_before=abundance_col,
                abundance_after=f"{abundance_col}_batch_corrected",
                sample_type_col=sample_type_col,
                feature_col=feature_col,
                sample_col=sample_col,
                batch_col=batch_col,
                reference_type=reference_type,
                qc_type=qc_type,
            )
        else:
            if not has_reference:
                logger.warning(
                    f"No samples with {sample_type_col}='{reference_type}' found - "
                    "skipping evaluation"
                )
            if not has_qc:
                logger.warning(
                    f"No samples with {sample_type_col}='{qc_type}' found - skipping evaluation"
                )

    # Handle fallback if evaluation failed
    corrected_col = f"{abundance_col}_batch_corrected"
    if fallback_on_failure and evaluation is not None and not evaluation.passed:
        logger.warning("Batch correction failed QC validation - falling back to uncorrected data")
        for warning in evaluation.warnings:
            logger.warning(f"  - {warning}")

        # Replace batch-corrected values with original values
        corrected[corrected_col] = corrected[abundance_col]

        # Add a flag indicating fallback was used
        evaluation.warnings.append("FALLBACK: Using uncorrected data due to QC failure")

    return corrected, evaluation
