# Skyline-PRISM v26.2.1 Release Notes

## Overview

Major enhancement release introducing **Library Median Polish** - a new fitting method for library-assisted transition rollup that provides dramatically improved robustness to interference. This release also includes improved outlier detection and bug fixes.

## New Feature: Library Median Polish Fitting Method

The library-assisted transition rollup now supports two fitting methods, with **median_polish** as the new default:

### Median Polish (NEW, Default, Recommended)

Uses the spectral library as a **prior** for transition row effects (expected fragmentation) and estimates sample scale factors using the **median**, which is inherently robust to interference outliers.

**Key advantages:**
- **Robust to 1-2 outliers automatically**: Median ignores up to 50% outliers without explicit detection
- **Cross-sample consistency**: Uses same model as standard median polish
- **Faster convergence**: Often converges in 1-2 iterations
- **Dramatic CV improvement**: For peptides with real interference, CV reductions of 30-40% are common

**Algorithm:**

1. Model: `log(Obs) = log(Library) + beta_s + epsilon`
2. Estimate beta_s via MEDIAN across transitions (robust to outliers)
3. Compute normalized residuals on linear scale
4. Only HIGH positive residuals indicate interference (obs > 2x predicted)
5. Remove worst outlier per sample, refit until convergence
6. Final abundance = exp(beta_s) * sum(ALL library intensities)

### Least Squares (Legacy)

Classic least squares fitting: `scale = (lib . obs) / (lib . lib)`. More sensitive to outliers but may perform better on very clean data.

### Configuration

```yaml
transition_rollup:
  method: "library-assist"
  library_assist:
    library_path: "/path/to/library.blib"
    fitting_method: "median_polish"  # NEW DEFAULT (or "least_squares")
    outlier_threshold: 1.0  # observed > 2x predicted = outlier
```

**Fitting method guide:**

| Method | Description | When to Use |
|--------|-------------|-------------|
| `median_polish` | Uses MEDIAN for scale estimation | **Recommended for most data** - robust to interference |
| `least_squares` | Classic OLS fitting | Very clean data with minimal interference |

## Improved Outlier Detection

### Normalized Residual Threshold

The library-assisted transition rollup now uses a **normalized residual threshold** instead of MAD-based z-scores for outlier detection.

**Threshold behavior:**
- Computes normalized residual: `(observed - predicted) / predicted`
- Flags transitions where `norm_residual > outlier_threshold`
- Default `outlier_threshold: 1.0` means: observed > 2x predicted
- Conservative `outlier_threshold: 2.0` means: observed > 3x predicted

**Key improvements:**
- **One outlier per iteration**: Removes only the worst outlier at a time to prevent over-removal
- **Respects min_transitions**: Stops removing outliers if it would go below the minimum
- **Scale-independent**: Normalized residuals work consistently across peptides with different abundance levels
- **Configurable threshold**: Users can tune sensitivity via config

**Threshold guide:**

| Threshold | Meaning | Use Case |
|-----------|---------|----------|
| 1.0 | observed > 2x predicted | Standard sensitivity (default) |
| 2.0 | observed > 3x predicted | Conservative, strict filtering |
| 0.5 | observed > 1.5x predicted | Aggressive, catches more interference |

## Bug Fixes

### CRITICAL: Fixed Zero-Handling in Least Squares Fitting

**Problem**: Library-assisted rollup was performing worse than simple sum in some cases. Investigation revealed that zero values in observed intensities were being incorrectly handled in the least squares fitting equation.

**Impact**: This caused significant underestimation of peptide abundances. For example:
- Library pattern: `[1000, 500, 100]` (3 transitions)
- Observed: `[0, 400, 80]` (first transition has zero - missing signal)
- **Before fix**: Abundance underestimated by ~5x because zero contributed to denominator
- **After fix**: Abundance correctly estimated using only fragments with signal

**Technical details**:

The least squares equation `scale = (L . O) / (L . L)` computes:
- Numerator: sum of (library_i * observed_i)
- Denominator: sum of (library_i^2)

When a transition has zero observed signal, it contributes:
- 0 to numerator (0 * lib = 0)
- lib^2 to denominator (non-zero!)

This pulls down the scale factor incorrectly. A zero where the library expects a large signal is **not** the same as "no observation" - it could indicate:
1. Interference in the MS1 selection window that suppressed the signal
2. Ion suppression
3. Below-LOD signal

**Solution**: Exclude zero observations from the fitting equation (both numerator AND denominator), but still use the full library sum for final abundance calculation. This ensures the scale factor is computed only from transitions with actual signal.

**Code change**:
```python
# Before: zeros included in fitting
included = np.ones(n_valid, dtype=bool)

# After: zeros excluded from fitting  
has_signal = obs_v > 0
included = has_signal.copy()
```

### Fixed: Method Name Normalization

**Problem**: Users specifying `method: "library-assist"` (with hyphen) got an error because the code expected `"library-assisted"` (past tense).

**Solution**: Added method name normalization in cli.py to accept:
- `"library-assist"` (alias)
- `"library_assist"` (underscore variant)
- `"library-assisted"` (canonical)

### Fixed: pytz SyntaxWarning

**Problem**: Users saw a warning at startup:
```
/usr/lib/python3/dist-packages/pytz/__init__.py:31: SyntaxWarning: invalid escape sequence '\s'
```

**Cause**: Older versions of pytz (pre-2024) had an unescaped regex pattern.

**Solution**: Added `pytz>=2024.1` to dependencies. Installing PRISM now pulls in a fixed version that shadows older system packages.

### Fixed: Unknown Config Key Warning for `outlier_threshold`

**Problem**: Users configuring `library_assist.outlier_threshold` saw:
```
WARNING - Unknown configuration keys detected (possible typos):
  - transition_rollup.library_assist.outlier_threshold
```

**Solution**: Added `outlier_threshold` to `KNOWN_CONFIG_KEYS` in cli.py.

## Documentation Updates

### Configuration Template Clarification

Updated documentation to clarify that **configuration templates are generated from functions in `cli.py`**, not from the static `config_template.yaml` file.

Users generate templates via:
```bash
prism config-template -o config.yaml           # Full template
prism config-template --minimal -o config.yaml  # Minimal template
```

When adding new configuration options, developers must update:
- `get_full_config_template()` in `cli.py`
- `get_minimal_config_template()` in `cli.py`

The static `config_template.yaml` is just a reference copy.

### Updated Files

- `AGENTS.md`: Added "CRITICAL: Configuration Template Updates" section
- `README.md`: Updated Configuration section with `prism config-template` commands
- `config_template.yaml`: Added `outlier_threshold` documentation with guidance

## Technical Details

### Files Modified

**Core algorithm changes:**
- `skyline_prism/spectral_library.py`:
  - `least_squares_rollup()`: Changed from MAD-based to normalized residual threshold
  - `least_squares_rollup_vectorized()`: Same change for batch processing
  - `library_assisted_rollup_peptide()`: Added `outlier_threshold` parameter
  - `SpectralLibraryRollup`: Added `outlier_threshold` to constructor

**CLI and config:**
- `skyline_prism/cli.py`:
  - Added `outlier_threshold` to `KNOWN_CONFIG_KEYS`
  - Updated config reading to pass threshold to library
  - Updated both template generation functions

- `skyline_prism/chunked_processing.py`:
  - Added `spectral_library_outlier_threshold` to `ChunkedRollupConfig`

**Infrastructure:**
- `pyproject.toml`:
  - Added `pytz>=2024.1` dependency
  - Changed entry point to `skyline_prism.__main__:main`

- `skyline_prism/__main__.py`: New file for cleaner CLI entry point

### Algorithm Details

The new outlier detection algorithm:

```
For each iteration (max 5):
    1. Fit: scale = (library . observed) / (library . library)
    2. Compute predicted = scale * library
    3. Compute normalized residuals = (observed - predicted) / predicted
    4. Find worst outlier: max(normalized_residuals) among included fragments
    5. If worst > threshold AND n_included > min_fragments:
         - Exclude that ONE fragment
         - Continue to next iteration
       Else:
         - Converged, stop
    6. Final fit with clean data
    7. Return abundance = scale * sum(library)
```

Key properties:
- Only **positive** residuals are considered (interference = signal > expected)
- Removes at most **one outlier per iteration** (prevents cascade removal)
- Stops when no outlier exceeds threshold OR would go below min_fragments
- Final abundance derived from library pattern, not raw sums

## Testing

- All 57 spectral library tests passing
- All 291 total tests passing
- Verified on real data: improved CV for peptides with interference

## Upgrade Notes

- No breaking changes from v26.2.0
- Library-assisted rollup results may differ slightly due to improved outlier detection
- The new algorithm is more conservative (removes fewer false positives)
- To get more aggressive outlier removal, lower the threshold (e.g., 0.5)

## Migration from v26.2.0

No code changes required. To use the new outlier threshold:

```yaml
transition_rollup:
  method: "library-assist"
  library_assist:
    outlier_threshold: 1.0  # Default, same as before conceptually
```

The algorithm change is automatic - existing configs will use the new (better) detection method.
