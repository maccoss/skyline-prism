# Skyline-PRISM v26.3.0 Release Notes

## Overview

Feature release adding **quantile normalization** and **VSN (Variance Stabilizing Normalization)** to the global normalization options, plus a new `remove_outliers` option for library-assisted rollup debugging.

This release also includes an important **batch correction fix** that ensures batches are correctly assigned from input files.

## New Feature: Quantile Normalization

Quantile normalization forces all samples to have identical distributions, which can be useful when you expect samples to have similar overall abundance profiles.

**How it works:**
1. Rank values within each sample
2. Compute mean value at each rank across all samples (the "quantile reference")
3. Replace each value with the quantile reference value corresponding to its rank

**When to use:**
- When you expect all samples to have similar overall distributions
- When you want aggressive normalization to remove systematic differences
- Useful for comparing samples with very different loading amounts

**Configuration:**
```yaml
global_normalization:
  method: "quantile"
```

**Output scale:** Log2 (same as median and RT-lowess)

## New Feature: VSN (Variance Stabilizing Normalization)

VSN uses an arcsinh transformation to stabilize variance across intensity ranges, making variance independent of the mean. This is particularly useful for heteroscedastic data where low-abundance measurements have different variance than high-abundance measurements.

**How it works:**
1. Convert from log2 to linear scale
2. Estimate scaling parameter `a` (default: 1/median)
3. Apply transformation: `arcsinh(a * linear_value)`
4. Optionally optimize `a` to minimize variance heterogeneity

**When to use:**
- When you observe heteroscedastic data (variance depends on intensity)
- When low-abundance peptides/proteins have unreliable measurements
- For data with wide dynamic range where log transformation alone is insufficient

**Configuration:**
```yaml
global_normalization:
  method: "vsn"
  vsn_params:
    optimize_params: false  # Set true for better variance stabilization (slower)
```

**Important notes about VSN output:**
- Output is on **arcsinh scale**, NOT log2
- Negative values are **valid and expected** for low-abundance peptides/proteins
- Do not interpret values as log2 - the scale is different
- Final parquet files will contain 2^(arcsinh_value) which may produce unexpected results

## Bug Fix: Batch Correction Now Uses Input Files as Batches

Fixed a critical issue where batch correction was incorrectly determining batches, especially at the protein level.

**Problem:**
- Protein-level ComBat was showing "Only 0 batch - skipping batch correction"
- Peptide-level was sometimes finding fewer batches than expected
- Batches were being assigned from metadata filenames instead of data filenames

**Solution:**
Batch assignment now follows this priority order:

1. **Source Document** (highest priority) - Each input CSV file (`-i file1.csv file2.csv...`) is automatically treated as a separate batch. This is the most reliable method.

2. **Metadata batch column** - If Source Document doesn't yield multiple batches, fall back to batch info from metadata files.

3. **Acquisition time estimation** - If neither of the above work, estimate batches from gaps in acquisition times.

**New logging output:**
```
Stage 2c: Peptide ComBat Batch Correction
  Assigning batches from source documents: 4 batches
    TRX-HDisc-Plate1-prism: 95 samples
    TRX-HDisc-Plate2-prism: 95 samples
    TRX-HDisc-Plate3-prism: 95 samples
    TRX-HDisc-Plate4-prism: 99 samples
  Batch source: source documents
  Batches found: ['TRX-HDisc-Plate1-prism', 'TRX-HDisc-Plate2-prism', ...]
  Applying ComBat across 4 batches...
```

**Key benefit:** When you run with 4 input files, you will now reliably get 4 batches at both peptide and protein levels, regardless of what the metadata files contain.

## New Feature: Disable Outlier Removal for Library-Assisted Rollup

Added a `remove_outliers` option to the library-assisted rollup configuration. This is primarily useful for debugging when you want to see the effect of the median estimation alone without iterative outlier removal.

**Background:**
The library median polish method has two components:
1. **Median estimation**: Robust scale estimation using median of log-ratios
2. **Iterative outlier removal**: Removes worst outlier per iteration until convergence

In testing, we found that for most peptides, the median estimation alone (without outlier removal) provides excellent results because the median is inherently robust to up to 50% outliers.

**Configuration:**
```yaml
transition_rollup:
  method: "library_assist"
  library_assist:
    library_path: "/path/to/library.blib"
    fitting_method: "median_polish"
    remove_outliers: true   # Default: enable outlier removal
    # remove_outliers: false  # Disable for debugging or when median alone is sufficient
```

**When to disable outlier removal:**
- For debugging to isolate the contribution of each algorithm component
- When you observe that outlier removal is removing valid signal
- When the median estimation alone gives better CVs than median + outlier removal

## Summary of Normalization Methods

| Method | Scale | Description | Best For |
|--------|-------|-------------|----------|
| `rt_lowess` | log2 | RT-dependent correction | **Recommended default** - corrects RT-dependent systematic effects |
| `median` | log2 | Global median centering | Simple cases, quick normalization |
| `quantile` | log2 | Force identical distributions | Aggressive normalization, similar samples |
| `vsn` | arcsinh | Variance stabilization | Heteroscedastic data, wide dynamic range |
| `none` | log2 | No normalization | When normalization is unwanted |

## Configuration Updates

### Updated Config Template

The full config template now includes detailed documentation for all normalization methods:

```yaml
global_normalization:
  # Method:
  #   median    - Subtract sample median (recommended for simple cases)
  #   rt_lowess - RT-dependent Lowess normalization (RECOMMENDED default)
  #   quantile  - Force identical distributions across all samples (aggressive)
  #   vsn       - Variance Stabilizing Normalization (arcsinh transformation)
  #               NOTE: Output is on arcsinh scale, NOT log2. Negative values expected.
  #   none      - Skip normalization
  method: "rt_lowess"

  # VSN parameters (only used if method: vsn)
  vsn_params:
    optimize_params: false  # Set true for better variance stabilization (slower)
```

### Updated KNOWN_CONFIG_KEYS

Added `remove_outliers` to the `transition_rollup.library_assist` config keys.

## Technical Details

### Files Modified

**Core implementation:**
- `skyline_prism/cli.py`:
  - Added quantile normalization implementation
  - Added VSN normalization implementation
  - Added `remove_outliers` config key to `KNOWN_CONFIG_KEYS`
  - Added `remove_outliers` parameter reading and logging
  - Updated full config template with detailed normalization documentation
  - Added `get_batches_from_source_document()` function to extract batches from input files
  - Refactored peptide and protein ComBat to prioritize Source Document for batch assignment
  - Added detailed batch logging showing source, batches found, and sample counts per batch
  - Fixed protein-level ComBat to use helper functions for Sample ID / Replicate Name matching

### Algorithm Details

**Quantile Normalization:**
```python
# For each sample, rank values (handling NaN)
ranks = rankdata(sample_values, method='average')

# Sort all samples, compute row means = quantile reference
sorted_data = np.sort(sample_data, axis=0)
quantile_reference = np.nanmean(sorted_data, axis=1)

# Map ranks back to quantile reference via interpolation
normalized = np.interp(ranks, range(1, n+1), quantile_reference)
```

**VSN Normalization:**
```python
# Convert from log2 to linear
linear_values = 2 ** log2_values

# Estimate scaling parameter
a_param = 1.0 / np.median(linear_values)

# Apply arcsinh transformation
normalized = np.arcsinh(a_param * linear_values)
```

## Testing

- All 301 existing tests passing
- Manual verification of quantile normalization (identical sorted values across samples)
- Manual verification of VSN normalization (variance stabilization across intensity range)
- Manual verification of batch correction with 4 input files producing 4 batches at both peptide and protein levels

## Upgrade Notes

- No breaking changes from v26.2.1
- New normalization methods are additive - existing configs continue to work
- VSN output is on a different scale than other methods - be aware when comparing

## Migration from v26.2.1

No code changes required. To use new features:

```yaml
# For quantile normalization
global_normalization:
  method: "quantile"

# For VSN normalization
global_normalization:
  method: "vsn"
  vsn_params:
    optimize_params: false

# To disable outlier removal in library-assisted rollup
transition_rollup:
  method: "library_assist"
  library_assist:
    remove_outliers: false
```
