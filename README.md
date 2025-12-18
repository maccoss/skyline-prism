# Skyline-PRISM

**PRISM** (Proteomics Reference-Integrated Signal Modeling) is a Python package for retention time-aware normalization of LC-MS proteomics data exported from [Skyline](https://skyline.ms), with robust protein quantification using Tukey median polish.

## Key Features

- **Robust quantification with Tukey median polish**: Default method for both transition→peptide and peptide→protein rollups - automatically handles outliers without pre-identification
- **Reference-anchored ComBat batch correction**: Full implementation of empirical Bayes batch correction with automatic QC evaluation using reference/pool samples
- **Dual-control validation**: Uses intra-experiment pool samples to validate that corrections work without overfitting
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

### 1. Merge Skyline reports

```bash
prism merge report1.csv report2.csv -o unified_data.parquet -m sample_metadata.tsv
```

### 2. Run normalization

```bash
prism normalize -i unified_data.parquet -o normalized_peptides.parquet -c config.yaml
```

### 3. Roll up to proteins

```bash
prism rollup -i normalized_peptides.parquet -o proteins.parquet -g protein_groups.tsv
```

### 4. Validate results

```bash
prism validate --before unified_data.parquet --after normalized_peptides.parquet --report qc_report.html
```

## Processing Pipeline

The pipeline follows a two-arm design:

```
Raw transition abundances
        │
        ▼
┌─────────────────────────────────────┐
│  Transition → Peptide rollup        │
│  (Tukey median polish - default)    │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  [Optional] RT-aware normalization  │
│  (Disabled by default)              │
└─────────────────────────────────────┘
        │
        ├──────────────────────────────────┐
        │                                  │
        ▼                                  ▼
┌─────────────────────┐      ┌─────────────────────────┐
│  PEPTIDE OUTPUT     │      │  PROTEIN OUTPUT         │
│  Batch correction   │      │  Peptide → Protein      │
│  (ComBat)           │      │  rollup (median polish) │
│         │           │      │           │             │
│         ▼           │      │           ▼             │
│  Normalized         │      │  Batch correction       │
│  peptides           │      │  (ComBat)               │
└─────────────────────┘      │           │             │
                             │           ▼             │
                             │  Normalized proteins    │
                             └─────────────────────────┘
```

## Experimental Design Requirements

For optimal results, include these controls in each batch:

| Control Type | Description | Replicates/Batch |
|--------------|-------------|------------------|
| Inter-experiment reference | Commercial pool (e.g., Golden West CSF) | 1-8 |
| Intra-experiment pool | Pooled experimental samples | 1-8 |

**Note:** In 96-well plate formats, controls are typically placed once per row (8 replicates per batch). Smaller experiments may have as few as 1 replicate per batch.

Plus internal QCs in all samples:
- **Protein QC**: Yeast enolase (16 ng/μg sample)
- **Peptide QC**: PRTC (30-150 fmol/injection)

## Configuration

See `config_template.yaml` for all options. Key settings:

```yaml
# Transition to peptide rollup (if using transition-level data)
transition_rollup:
  enabled: true
  method: "median_polish"  # default, recommended
  min_transitions: 3

# RT correction (disabled by default - see docs)
rt_correction:
  enabled: false  # DIA-NN RT calibration may not generalize between samples
  method: "spline"
  spline_df: 5
  per_batch: true

# Batch correction (applied at reporting level)
batch_correction:
  enabled: true
  method: "combat"

# Protein rollup
protein_rollup:
  method: "median_polish"  # default, recommended
  min_peptides: 3

# Shared peptide handling
parsimony:
  shared_peptide_handling: "all_groups"  # recommended
```

## Tukey Median Polish (Default Rollup Method)

Both transition→peptide and peptide→protein rollups use Tukey median polish by default. This robust method:

- **Automatically handles outliers**: Interfered transitions or problematic peptides are downweighted through the median operation
- **No pre-filtering required**: Unlike quality-weighted methods, doesn't require explicit quality metrics
- **Produces interpretable effects**: Separates row effects (transition/peptide ionization) from column effects (sample abundance)
- **Handles missing values naturally**: Works with incomplete matrices

### Alternative Rollup Methods

While median polish is recommended, these alternative methods are available:

| Method | Description | Use Case |
|--------|-------------|----------|
| `median_polish` | Tukey median polish (default) | General use, robust to outliers |
| `topn` | Mean of top N most intense peptides | Simple, when top peptides are reliable |
| `ibaq` | Intensity-Based Absolute Quantification | Cross-protein abundance comparison |
| `maxlfq` | Maximum LFQ (MaxQuant-style) | When peptide ratios are more reliable than absolute values |
| `quality_weighted` | Variance-model weighted average | When quality metrics (correlation, coelution) are available |
| `sum` | Simple sum of intensities | Spectral counting style |

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

# With automatic evaluation using reference/pool samples
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
- Automatic evaluation using reference and pool sample CVs

## Residual Analysis for Proteoform Discovery

Median polish produces residuals that capture deviations from the additive model. Following [Plubell et al. 2022](https://doi.org/10.1021/acs.jproteome.1c00894), peptides with large residuals should **not** be automatically discarded - they may indicate biologically interesting variation:

- **Proteoform differences**: Different forms of the same protein (splice variants, truncations)
- **Post-translational modifications**: PTMs affecting specific peptides
- **Protein processing**: Cleavage products (e.g., amyloid-beta from APP)
- **Technical outliers**: Interference or poor peak picking

**Accessing residuals:**

```python
from skyline_prism import rollup_to_proteins, extract_peptide_residuals

# Protein rollup returns median polish results
protein_df, polish_results = rollup_to_proteins(data, protein_groups)

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
)
```

## Citation

If you use Skyline-PRISM, please cite:

Tsantilas KA et al. "A framework for quality control in quantitative proteomics." 
J Proteome Res. 2024. DOI: 10.1021/acs.jproteome.4c00363

## Related Projects

- [Skyline](https://skyline.ms) - Targeted mass spectrometry environment
- [Panorama](https://panoramaweb.org) - Repository for Skyline documents

## License

MIT License - see LICENSE file.
