# Skyline-PRISM

**PRISM** (Proteomics Reference-Integrated Signal Modeling) is a Python package for retention time-aware normalization of LC-MS proteomics data exported from [Skyline](https://skyline.ms), with robust protein quantification using Tukey median polish.

## Key Features

- **Robust quantification with Tukey median polish**: Default method for both transition→peptide and peptide→protein rollups - automatically handles outliers without pre-identification
- **Reference-anchored ComBat batch correction**: Full implementation of empirical Bayes batch correction with automatic QC evaluation using reference/QC samples
- **Dual-control validation**: Uses intra-experiment QC samples to validate that corrections work without overfitting
- **Sample outlier detection**: Automatic detection of samples with abnormally low signal (failed injections, degradation) with optional exclusion
- **Flexible protein inference**: Multiple strategies for handling shared peptides (all_groups, unique_only, razor)
- **Two-arm normalization pipeline**: Separate paths for peptide-level and protein-level output, with batch correction applied at the appropriate level
- **Optional RT correction**: Reference-anchored RT correction is available but disabled by default (search engine RT calibration may not generalize between samples)

## Installation

```bash
pip install skyline-prism
```

Or for development:

```bash
git clone https://github.com/maccoss/skyline-prism
cd skyline-prism
pip install -e ".[dev,viz]"
```

## Quick Start

### Run the full pipeline

```bash
# Run complete PRISM pipeline (recommended)
prism run -i skyline_report.csv -o output_dir/ -c config.yaml -m sample_metadata.tsv
```

This produces:

- `corrected_peptides.parquet` - Peptide-level quantities (batch-corrected)
- `corrected_proteins.parquet` - Protein-level quantities (batch-corrected)  
- `protein_groups.tsv` - Protein group definitions
- `peptide_residuals.parquet` - Median polish residuals (for outlier analysis)
- `metadata.json` - Processing metadata and provenance
- `prism_run_YYYYMMDD_HHMMSS.log` - Detailed log file with all processing steps and timings
- `qc_plots/` - Directory with QC plots (if QC reporting enabled)

**Scale:** Output files contain **linear-scale** abundances (raw peak area units). The pipeline operates on log2 scale internally but back-transforms to linear before writing output.

**Key output columns** in protein/peptide parquet files:

- `abundance` - Linear-scale abundance (normalized, batch-corrected)
- `abundance_raw` - Original linear-scale abundance from Skyline
- `uncertainty` - Propagated uncertainty (log2 scale, for downstream analysis)
- `cv_peptides` - CV across peptides (protein-level quality metric)
- `n_peptides` - Number of peptides used in rollup
- `qc_flag` - QC warnings (e.g., `low_peptide_count(n)`, `single_peptide_in_sample`)

See [SPECIFICATION.md](SPECIFICATION.md) for complete column definitions.

### Reproducibility: Re-run from provenance

The `metadata.json` output contains the complete processing configuration, enabling reproducible re-runs:

```bash
# Re-run with exact same parameters on new data
prism run -i new_data.csv -o output2/ --from-provenance output1/metadata.json

# Override specific settings while keeping others from provenance
prism run -i new_data.csv -o output2/ --from-provenance output1/metadata.json -c overrides.yaml
```

### Regenerate QC reports

To regenerate QC reports from an existing PRISM output directory (useful after visualization updates):

```bash
prism qc -d output_dir/
```

This reads the processed parquet files and regenerates the QC report without re-running the full pipeline.

### Generate a configuration template

Generate an annotated configuration file template:

```bash
# Full template with all options documented
prism config-template -o config.yaml

# Minimal template with only common options
prism config-template --minimal -o config.yaml
```

### Merge multiple Skyline reports

```bash
prism merge report1.csv report2.csv -o unified_data.parquet -m sample_metadata.tsv
```

### QC report contents

The QC report includes:

- **Intensity distribution plots**: Before/after normalization comparison
- **PCA analysis**: Visualize batch effects and sample clustering across processing stages
- **Control sample correlation**: Heatmaps showing reproducibility of reference and QC samples
- **CV distributions**: Technical variation in control samples before/after normalization
- **RT correction QC**: If RT-dependent normalization is enabled, shows residuals for reference (fitted) and QC (held-out validation) samples

All plots are saved as PNGs in `output_dir/qc_plots/` and embedded in the HTML report.

## Processing Pipeline

The pipeline follows a two-arm design with batch correction applied at the reporting level:

```text
Stage 1: Merge CSVs (streaming)
        │
        ▼
Stage 2: Transition → Peptide rollup (Tukey median polish)
        │
        ▼
Stage 2b: Peptide Global Normalization (median or VSN)
        │  [Optional: RT-aware correction - disabled by default]
        ▼
Stage 2c: Peptide ComBat Batch Correction
        │
        ├─────────────────────────────────┐
        │                                 │
        ▼                                 ▼
Stage 3: Protein Parsimony      PEPTIDE OUTPUT ARM
        │                       (corrected_peptides.parquet)
        ▼
Stage 4: Peptide → Protein Rollup (median polish)
        │
        ▼
Stage 4b: Protein Global Normalization (median)
        │
        ▼
Stage 4c: Protein ComBat Batch Correction
        │
        ▼
Stage 5: Output Generation
        │
        ▼
    PROTEIN OUTPUT ARM
(corrected_proteins.parquet)
        │
        ▼
Stage 5b: QC Report Generation (HTML + plots)
```

**Key Implementation Notes:**

- **Streaming CSV merge** handles large datasets (~47GB tested) without loading into memory
- **Both peptide and protein outputs** receive independent batch correction
- **Automatic log files** saved to output directory with timestamp for reproducibility
- **RT correction disabled by default** - search engine calibration may not generalize between samples
- **Protein parsimony** applied before protein rollup to avoid double-counting shared peptides

## Experimental Design Requirements

For optimal results, include these controls in each batch:

| Control Type               | Description                               | Replicates/Batch |
| -------------------------- | ----------------------------------------- | ---------------- |
| Inter-experiment reference | Commercial pool (e.g., Golden West CSF)   | 1-8              |
| Intra-experiment QC      | Pooled experimental samples               | 1-8              |

**Note:** In 96-well plate formats, controls are typically placed once per row (8 replicates per batch). Smaller experiments may have as few as 1 replicate per batch.

Plus internal QCs in all samples:

- **Protein QC**: Yeast enolase (16 ng/μg sample)
- **Peptide QC**: PRTC (30-150 fmol/injection)

## Configuration

See `config_template.yaml` for all options. Key settings:

```yaml
# Sample type detection patterns (for automatic sample type assignment)
sample_annotations:
  reference_pattern:  # Inter-experiment reference (e.g., commercial pools)
    - "-Pool_"
    - "-Pool"
  qc_pattern:       # Intra-experiment QC (e.g., pooled experimental samples)
    - "-Carl_"
    - "-Carl"
  # Everything else is treated as experimental

# Transition to peptide rollup (if using transition-level data)
transition_rollup:
  enabled: true
  method: "median_polish"  # default; alternatives: adaptive, sum
  min_transitions: 3
  learn_variance_model: false  # Set true for adaptive method

# Sample outlier detection (one-sided, low signal only)
sample_outlier_detection:
  enabled: true
  action: "report"  # "report" to log only, "exclude" to remove from analysis
  method: "iqr"     # IQR-based on LINEAR scale (not log)
  iqr_multiplier: 1.5

# Global normalization (applied after peptide rollup)
global_normalization:
  method: "median"  # median (recommended) or vsn

# RT correction (optional, applied together with global normalization)
rt_correction:
  enabled: false  # DISABLED BY DEFAULT - search engine RT calibration may not generalize
  method: "spline"
  spline_df: 5
  per_batch: true

# Batch correction (applied at both peptide and protein levels)
batch_correction:
  enabled: true
  method: "combat"  # Full empirical Bayes implementation

# Protein parsimony (applied before protein rollup)
parsimony:
  enabled: true
  fasta_path: "/path/to/database.fasta"  # FASTA used in search
  shared_peptide_handling: "all_groups"  # recommended

# Protein rollup (peptide → protein aggregation)
protein_rollup:
  method: "sum"  # default; alternatives: median_polish, topn, ibaq, maxlfq

# QC report generation
qc_report:
  enabled: true
  save_plots: true    # Save individual PNG files
  embed_plots: true   # Embed plots in HTML report
```

## Automatic Batch Estimation

If batch information is not provided in the metadata file, PRISM will automatically estimate batch assignments using the following priority:

1. **Source documents**: Different Skyline CSV/TSV files are treated as different batches
2. **Acquisition time gaps**: Large gaps between runs (>3× median gap) indicate batch boundaries
3. **Equal division**: Samples are divided into approximately equal batches

To enable time-based batch estimation, include `Result File > Acquired Time` in your Skyline report. See [SPECIFICATION.md](SPECIFICATION.md#batch-estimation) for details.

## Sample Outlier Detection

PRISM automatically detects samples with abnormally low signal intensity, which may indicate:

- Failed injections
- Sample degradation
- Instrument issues
- Empty wells

**Key features:**

- **One-sided detection**: Only flags samples with too *little* signal (not high outliers)
- **Linear scale statistics**: Detection uses linear scale (not log2) to avoid scale compression
- **Two detection methods**:
  - `iqr`: Flag samples below Q1 - 1.5×IQR (default)
  - `fold_median`: Flag samples with median < X% of overall median
- **Configurable action**: Report only or exclude from analysis

```yaml
sample_outlier_detection:
  enabled: true
  action: "report"     # "report" or "exclude"
  method: "iqr"        # "iqr" or "fold_median"
  iqr_multiplier: 1.5  # Samples below Q1 - 1.5*IQR flagged
```

Example log output:
```
Sample outlier detection (IQR method, linear scale):
  Q1=1234567, Q3=2345678, IQR=1111111
  Lower bound: 567890 (Q1 - 1.5*IQR)
  Overall median: 1789012
  OUTLIER: Sample_042 - median=24 (0.0% of overall median)
```

## Tukey Median Polish (Default Rollup Method)

Both transition→peptide and peptide→protein rollups use Tukey median polish by default. This robust method:

- **Automatically handles outliers**: Interfered transitions or problematic peptides are downweighted through the median operation
- **No pre-filtering required**: Unlike quality-weighted methods, doesn't require explicit quality metrics
- **Produces interpretable effects**: Separates row effects from column effects (sample abundance)
- **Handles missing values naturally**: Works with incomplete matrices

**What the row effects represent:**

| Rollup Stage         | Row Effects Represent                                                        | Column Effects Represent     |
| -------------------- | ---------------------------------------------------------------------------- | ---------------------------- |
| Transition → Peptide | Transition interference (co-eluting analytes affecting specific transitions) | Peptide abundance per sample |
| Peptide → Protein    | Peptide ionization efficiency (some peptides ionize better than others)      | Protein abundance per sample |

### Alternative Rollup Methods

While median polish is recommended, these alternative methods are available:

**Transition → Peptide Rollup** (when using transition-level data):

| Method             | Description                     | Use Case                                                           |
| ------------------ | ------------------------------- | ------------------------------------------------------------------ |
| `median_polish`    | Tukey median polish (default)   | General use, robust to outliers                                    |
| `adaptive`         | Learned weighted average        | When quality metrics (intensity, m/z, ShapeCorrelation) are available |
| `sum`              | Simple sum of intensities       | Fast but sensitive to outliers                                     |

**Peptide → Protein Rollup**:

| Method          | Description                                  | Use Case                                                   |
| --------------- | -------------------------------------------- | ---------------------------------------------------------- |
| `median_polish` | Tukey median polish (default)                | General use, robust to outliers                            |
| `topn`          | Mean of top N most intense peptides          | Simple, when top peptides are reliable                     |
| `ibaq`          | iBAQ (intensity / theoretical peptide count) | Absolute quantification across proteins                    |
| `maxlfq`        | Maximum LFQ (MaxQuant-style)                 | When peptide ratios are more reliable than absolute values |
| `sum`           | Simple sum of intensities                    | Fast but sensitive to outliers                             |

### iBAQ (Intensity-Based Absolute Quantification)

iBAQ normalizes protein abundances by the number of theoretical peptides, enabling comparison of absolute abundance across proteins:

```yaml
protein_rollup:
  method: "ibaq"
  ibaq:
    fasta_path: "/path/to/database.fasta"  # Required for iBAQ
    enzyme: "trypsin"                       # Must match search settings
    missed_cleavages: 0                     # Typically 0 for counting
```

```python
from skyline_prism.fasta import get_theoretical_peptide_counts

# Calculate theoretical peptide counts for iBAQ
counts = get_theoretical_peptide_counts(
    "/path/to/database.fasta",
    enzyme="trypsin",
    missed_cleavages=0,
)
```

### Adaptive Rollup with Learned Weights

For transition→peptide rollup, the `adaptive` method learns optimal weighting parameters from reference samples:

```yaml
transition_rollup:
  method: "adaptive"
  learn_variance_model: true  # Learns from reference samples
  adaptive_rollup:
    beta_log_intensity: 0.5   # Weight for log2 intensity (must be >= 0)
    beta_mz: 0.0              # Weight for normalized m/z
    beta_shape_corr: 1.0      # Weight for shape correlation
    min_improvement_pct: 5.0  # Required improvement over sum to use adaptive weights
```

The weight formula is: `w_t = exp(beta_intensity * (log2(I) - center) + beta_mz * mz_norm + beta_shape * shape_corr)`

Key insight: When all betas are 0, weights equal 1 for all transitions, equivalent to simple sum. This provides a principled baseline where the optimizer can choose "no weighting" if it doesn't help.

**Learning process:**
1. Parameters are optimized on reference samples by minimizing median CV
2. Results are validated on QC samples to prevent overfitting
3. Automatic fallback to simple sum if adaptive doesn't improve CV by `min_improvement_pct`

**Note:** For very large cohorts (hundreds to thousands of samples), consider [directLFQ](https://github.com/MannLabs/directlfq) which uses a different "intensity trace" algorithm with linear runtime scaling. Our `maxlfq` implementation uses the original pairwise ratio approach which scales quadratically with sample count.

## ComBat Batch Correction

PRISM includes a full implementation of the ComBat algorithm (Johnson et al. 2007) for empirical Bayes batch correction:

```python
from skyline_prism import combat, combat_from_long, combat_with_reference_samples

# For wide-format data (features × samples)
corrected = combat(data, batch, covar_mod=covariates)

# For long-format data (as used in PRISM pipeline)
corrected_df = combat_from_long(
    data, 
    feature_col='peptide_modified',
    sample_col='replicate_name',
    batch_col='batch',
    abundance_col='abundance'
)

# With automatic evaluation using reference/QC samples
result = combat_with_reference_samples(
    data,
    sample_type_col='sample_type',
    # ... other parameters
)
```

**Key features:**

- Parametric and non-parametric prior options
- Reference batch support (adjust other batches to match reference)
- Mean-only correction option (for unequal batch sizes)
- Automatic evaluation using reference and QC sample CVs

## Residual Analysis for Proteoform Discovery

Median polish produces residuals that capture deviations from the additive model. Following [Plubell et al. 2022](https://doi.org/10.1021/acs.jproteome.1c00894), peptides with large residuals should **not** be automatically discarded - they may indicate biologically interesting variation:

- **Proteoform differences**: Different forms of the same protein (splice variants, truncations)
- **Post-translational modifications**: PTMs affecting specific peptides
- **Protein processing**: Cleavage products (e.g., amyloid-beta from APP)
- **Technical outliers**: Interference or poor peak picking

**Accessing residuals:**

```python
from skyline_prism import rollup_to_proteins, extract_peptide_residuals

# Protein rollup returns median polish results and topn results
protein_df, polish_results, topn_results = rollup_to_proteins(data, protein_groups)

# Extract residuals in long format for output
peptide_residuals = extract_peptide_residuals(polish_results)

# Columns include:
# - protein_group_id, peptide, replicate_name
# - residual: raw residual for each peptide/sample
# - residual_mad: robust measure of peptide's overall deviation
# - residual_max_abs: maximum deviation across samples

# Find peptides that consistently deviate
outliers = peptide_residuals.groupby(['protein_group_id', 'peptide']).agg({
    'residual_mad': 'first'
}).reset_index()
outliers = outliers[outliers['residual_mad'] > 0.5]  # Threshold of your choice
```

For transition-level residuals:

```python
from skyline_prism import rollup_transitions_to_peptides, extract_transition_residuals

# Use median_polish method for transition rollup
result = rollup_transitions_to_peptides(data, method='median_polish')

# Extract transition residuals
transition_residuals = extract_transition_residuals(result)
```

## Shared Peptide Handling

Three strategies available:

- **`all_groups`** (default, recommended): Apply shared peptides to ALL protein groups. Acknowledges proteoform complexity; avoids assumptions based on FASTA annotations.

- **`unique_only`**: Only use peptides unique to a single protein group. Most conservative.

- **`razor`**: Assign shared peptides to group with most peptides (MaxQuant-style).

## Python API

```python
from skyline_prism import (
    # Data I/O
    load_skyline_report,
    merge_skyline_reports,
    load_sample_metadata,
    
    # Normalization
    normalize_pipeline,
    rt_correction_from_reference,
    
    # Rollup
    tukey_median_polish,
    rollup_to_proteins,
    rollup_transitions_to_peptides,
    
    # Batch correction
    combat,
    combat_from_long,
    combat_with_reference_samples,
    
    # Protein inference
    compute_protein_groups,
    
    # Validation
    validate_correction,
    generate_qc_report,
    
    # Visualization
    plot_intensity_distribution,
    plot_normalization_comparison,
    plot_pca,
    plot_comparative_pca,
    plot_control_correlation_heatmap,
    plot_cv_distribution,
    plot_comparative_cv,
    plot_sample_correlation_matrix,
)
```

## Data Visualization

PRISM provides visualization functions for QC assessment and normalization evaluation. Install visualization dependencies with:

```bash
pip install skyline-prism[viz]
```

### Intensity Distribution

Compare sample intensity distributions before/after normalization:

```python
from skyline_prism import plot_intensity_distribution, plot_normalization_comparison

# Box plot of intensity distributions
plot_intensity_distribution(
    data,
    sample_types={"Sample_1": "reference", "Sample_2": "qc", ...},
    title="Intensity Distribution"
)

# Side-by-side comparison of normalization effect
plot_normalization_comparison(
    data_before=raw_data,
    data_after=normalized_data,
    title="Normalization Effect"
)
```

### PCA Analysis

Visualize batch effects and normalization impact with PCA:

```python
from skyline_prism import plot_pca, plot_comparative_pca

# Single PCA plot with sample grouping
fig, pca_df = plot_pca(
    data,
    sample_groups={"Sample_1": "Batch_A", "Sample_2": "Batch_B", ...}
)

# Comparative PCA: Original → Normalized → Batch Corrected
plot_comparative_pca(
    data_original=raw_data,
    data_normalized=normalized_data,
    data_batch_corrected=corrected_data,  # Optional
    sample_groups=sample_batches,
    figsize=(18, 6)
)
```

### Control Sample Correlation

Assess reproducibility using correlation heatmaps for control samples:

```python
from skyline_prism import plot_control_correlation_heatmap, plot_sample_correlation_matrix

# Correlation heatmap for reference and QC samples only
fig, corr_matrix = plot_control_correlation_heatmap(
    data,
    sample_type_col="sample_type",
    control_types=["reference", "qc"],
    method="pearson"
)

# Full sample correlation matrix
fig, corr_matrix = plot_sample_correlation_matrix(
    data,
    sample_types=sample_type_dict
)
```

### CV Distribution

Evaluate precision using CV distributions for control samples:

```python
from skyline_prism import plot_cv_distribution, plot_comparative_cv

# CV distribution histogram for controls
fig, cv_data = plot_cv_distribution(
    data,
    sample_type_col="sample_type",
    control_types=["reference", "qc"],
    cv_threshold=20.0
)

# Compare CV before/after normalization
plot_comparative_cv(
    data_before=raw_data,
    data_after=normalized_data,
    sample_type_col="sample_type",
    control_type="reference"
)
```

### RT Correction Quality Assessment

Visualize RT-dependent correction effectiveness. This is critical for validating that corrections learned from reference samples generalize to held-out QC samples:

```python
from skyline_prism import plot_rt_correction_comparison, plot_rt_correction_per_sample

# 2×2 comparison showing reference (fitted) vs QC (held-out validation)
# before and after RT correction
fig, axes = plot_rt_correction_comparison(
    data_before=data_before_correction,
    data_after=data_after_correction,
    sample_type_col="sample_type",
    reference_mean=reference_mean_df,  # Mean abundance per peptide from reference
    figsize=(14, 10)
)

# Per-sample before/after comparison
fig, axes = plot_rt_correction_per_sample(
    data_before=data_before_correction,
    data_after=data_after_correction,
    sample_type_col="sample_type",
    reference_mean=reference_mean_df,
    samples_per_type=3,  # Number of samples to show per type
    figsize=(16, 12)
)
```

The RT correction plots help assess:

- Whether the spline model captures RT-dependent variation in reference samples
- Whether corrections generalize to QC samples (held-out validation)
- Whether any samples have unusually large residuals after correction

All visualization functions support:

- `show_plot=False` to return the figure object for further customization
- `save_path="/path/to/figure.png"` to save directly to file

## Citation

If you use Skyline-PRISM, please cite:

Tsantilas KA et al. "A framework for quality control in quantitative proteomics."
J Proteome Res. 2024. DOI: 10.1021/acs.jproteome.4c00363

## Related Projects

- [Skyline](https://skyline.ms) - Targeted mass spectrometry environment
- [Panorama](https://panoramaweb.org) - Repository for Skyline documents

## License

MIT License - see LICENSE file.
