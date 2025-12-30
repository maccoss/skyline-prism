# Skyline-PRISM Specification

**PRISM**: Proteomics Reference-Integrated Signal Modeling

## Specification Document for Implementation

### Overview

This document specifies the PRISM approach to retention time (RT)-aware normalization that borrows ideas from batch correction methods (SVA, ComBat, RUV) while using the dual-control experimental design from the MacCoss lab QC framework. The key insight is to use the **inter-experiment reference** (e.g., commercial plasma/CSF pool) as the calibration standard for deriving normalization factors, and the **intra-experiment QC** (pooled experimental samples) as validation to assess whether the normalization worked without overfitting.

PRISM is designed to work with data exported from [Skyline](https://skyline.ms), the widely-used targeted mass spectrometry environment.

### Design Principles

1. **Robust protein quantification via Tukey median polish**: Use Tukey median polish for both transition→peptide and peptide→protein rollups, minimizing the influence of outliers without explicit filtering.

2. **Reference-anchored ComBat batch correction**: Use inter-experiment reference samples for QC evaluation, with automatic fallback if correction degrades quality.

3. **Validation with held-out control**: Use the intra-experiment QC to validate that corrections improved data quality without overfitting.

4. **Proper protein inference**: Handle protein parsimony before peptide-to-protein rollup to avoid double-counting shared peptides.

5. **Optional RT-dependent correction**: RT-dependent spline correction is implemented but **disabled by default**. Modern search engines (e.g., DIA-NN) apply per-file RT calibration that may not generalize between reference and experimental samples, making this correction less reliable. Enable only if RT-dependent technical variation is clearly observed in your data.

### Scale Conventions

PRISM uses different scales at different stages of the pipeline:

| Stage | Scale | Rationale |
|-------|-------|-----------|
| **Input** | LINEAR | Skyline exports raw peak areas (linear scale) |
| **Internal processing** | LOG2 | Additive operations (median polish, ComBat) work on log scale |
| **Final output** | LINEAR | Output files contain linear-scale abundances |

**Why log2 internally?**

On log scale, multiplicative effects become additive:
- A 2-fold change = +1 on log2 scale
- Median polish decomposes: `log2(Y_ij) = μ + α_i + β_j + ε_ij`
- ComBat batch correction operates on log-transformed data

**Back-transform for output:**

The CLI pipeline automatically back-transforms to linear scale before writing output files:
```python
linear_abundance = 2 ** log2_abundance
```

**Function-level documentation:**

Each function's docstring specifies the expected input and output scale. When using
the Python API directly, pay attention to scale requirements:
- `tukey_median_polish()`: Input and output are LOG2 scale
- `rollup_transitions_to_peptides()`: Input LINEAR, output LOG2 (when log_transform=True)
- `combat()`: Input and output are LOG2 scale

---

## Input and Output Specification

### Overview

This pipeline takes **transition-level** data from Skyline and produces protein-level quantification. We perform our own transition→peptide→protein rollup with quality weighting and learned parameters, rather than using Skyline's aggregated quantities.

### Input: Skyline Transition-Level Report

Export a **transition-level report** from Skyline (not pivoted by replicate). One row per transition per replicate.

#### Required Columns

| Skyline Field | Internal Name | Description |
|---------------|---------------|-------------|
| Protein | protein_names | Protein identifier(s) |
| Protein Accession | protein_ids | UniProt accession(s) |
| Protein Gene | protein_gene | Gene name(s) |
| Peptide | peptide_sequence | Unmodified sequence |
| Peptide Modified Sequence Unimod Ids | peptide_modified | Modified sequence with Unimod IDs (unique peptide ID) |
| Precursor Charge | precursor_charge | Charge state |
| Precursor Mz | precursor_mz | Precursor m/z |
| Fragment Ion | fragment_ion | Fragment ion identifier (e.g., y7, b5, precursor) |
| Product Charge | product_charge | Fragment charge |
| Product Mz | product_mz | Fragment m/z |
| Replicate Name | replicate_name | Sample identifier |
| Area | area | Integrated transition area |
| Retention Time | retention_time | Apex retention time |

**Note:** If `Fragment Ion` is not available, PRISM also accepts separate `Fragment Ion Type` and `Fragment Ion Ordinal` columns.

#### Required Quality Columns

| Skyline Field | Internal Name | Description |
|---------------|---------------|-------------|
| Shape Correlation | shape_correlation | Correlation with median transition profile (0-1). Low values indicate interference. |
| Coeluting | coeluting | Boolean: apex within integration boundaries |
| Detection Q Value | detection_qvalue | mProphet q-value (DIA) - for confidence filtering |

#### Recommended Columns

| Skyline Field | Internal Name | Description |
|---------------|---------------|-------------|
| Start Time | start_time | Integration start |
| End Time | end_time | Integration end |
| Fwhm | fwhm | Full width at half max |
| Truncated | truncated | Peak truncation flag |
| Isotope Dot Product | idotp | MS1 isotope pattern quality (precursor level) |
| File Name | file_name | Source file name |
| Total Ion Current Area | tic_area | Total ion current for TIC normalization |
| Acquired Time | acquired_time | Result file acquisition timestamp (for batch estimation) |

#### Skyline Report Configuration

In Skyline's Edit > Report/Results Grid, create a custom report with these fields:

```
Proteins > Protein
Proteins > Protein Accession
Proteins > Protein Gene
Peptides > Peptide
Peptides > Peptide Modified Sequence Unimod Ids
Precursors > Precursor Charge
Precursors > Precursor Mz
Precursors > Isotope Dot Product
Precursors > Detection Q Value
Transitions > Fragment Ion
Transitions > Product Charge
Transitions > Product Mz
Transition Results > Area
Transition Results > Retention Time
Transition Results > Start Time
Transition Results > End Time
Transition Results > Fwhm
Transition Results > Shape Correlation
Transition Results > Coeluting
Transition Results > Truncated
Replicates > Replicate Name
Replicates > File Name
Replicates > Total Ion Current Area
Result File > Acquired Time
```

**Important:** Do NOT check "Pivot Replicate Name". Export as long format.

**Note:** Including `Result File > Acquired Time` enables automatic batch estimation when batch metadata is not provided. See [Batch Estimation](#batch-estimation) for details.

---

### Input: Sample Metadata

A separate CSV/TSV file mapping replicate names to experimental metadata. PRISM supports direct export from Skyline's Replicates report with automatic column name normalization.

| Column | Required | Description |
|--------|----------|-------------|
| replicate_name | Yes | Must match Skyline replicate names exactly. Also accepts `Replicate Name` or `File Name`. |
| sample_type | Yes | One of: `experimental`, `qc`, `reference`, `blank`. Also accepts `Sample Type` with Skyline values (see below). |
| batch | No | Batch identifier. Also accepts `Batch Name` (Skyline convention). If not provided, will be estimated. |
| run_order | No | Acquisition order (integer). If not provided, calculated from Acquired Time in Skyline report. |
| subject_id | No | For paired/longitudinal designs |
| condition | No | Treatment group |
| timepoint | No | For longitudinal studies |
| ... | No | Additional annotations as needed |

**Notes:**
- If `batch` (or `Batch Name`) is not provided in metadata, PRISM will automatically estimate batch assignments. If only one batch is detected, batch correction is skipped. See [Batch Estimation](#batch-estimation).
- If `run_order` is not provided, it will be calculated automatically from the `Acquired Time` column in the Skyline report.

**Sample types (PRISM):**
- `reference`: Inter-experiment reference samples for RT correction and parameter learning
- `qc`: Intra-experiment QC (pooled samples for validation
- `experimental`: Actual experimental samples
- `blank`: Solvent/blank samples (excluded from analysis)

**Skyline Sample Type mapping:**

When using Skyline's `Sample Type` column, values are automatically mapped:

| Skyline Sample Type | PRISM sample_type | Usage |
|---------------------|-------------------|-------|
| Unknown | experimental | Experimental samples |
| Standard | reference | Inter-experiment reference (e.g., commercial plasma) |
| Quality Control | qc | Intra-experiment QC for validation |
| Solvent | blank | Excluded from analysis |
| Blank | blank | Excluded from analysis |
| Double Blank | blank | Excluded from analysis |

Example (Skyline Replicates report format):
```csv
Replicate Name,Sample Type,Batch Name
Sample_001,Unknown,batch1
Sample_002,Unknown,batch1
Pool_01,Quality Control,batch1
Reference_01,Standard,batch1
Sample_003,Unknown,batch1
```

Example (PRISM format with optional columns):
```csv
replicate_name,sample_type,batch,run_order,condition,subject_id
Sample_001,experimental,batch1,1,Treatment,P001
Sample_002,experimental,batch1,2,Control,P002
Pool_01,qc,batch1,3,,
Reference_01,reference,batch1,4,,
Sample_003,experimental,batch1,5,Treatment,P003
```

---

### Output Files

The pipeline produces multiple output files:

#### 1. `{name}_proteins.parquet` - Primary Output

The main output for downstream analysis. Contains normalized, batch-corrected protein-level quantities.

| Column | Type | Description |
|--------|------|-------------|
| protein_group_id | string | Unique protein group identifier |
| leading_protein | string | Representative protein accession |
| protein_accessions | string | All member accessions (semicolon-separated) |
| gene_names | string | Gene names if available (semicolon-separated) |
| description | string | Protein description |
| replicate_name | string | Sample identifier |
| sample_type | string | experimental/qc/reference |
| batch | string | Batch identifier |
| run_order | int | Acquisition order |
| abundance | float | Log2 abundance (normalized, batch-corrected) |
| abundance_raw | float | Log2 abundance before corrections |
| uncertainty | float | Propagated uncertainty (log2 scale std) |
| n_peptides | int | Number of peptides used in rollup |
| n_unique_peptides | int | Number of unique (non-shared) peptides |
| cv_peptides | float | CV across peptides (quality metric) |
| qc_flag | string | Any QC warnings (nullable) |

**Note:** To investigate which peptides are outliers for a given protein, use the peptide-level parquet file and filter by `protein_group_id`. Peptide residuals from the protein rollup median polish are available in the peptides file.

**Usage:**
```python
import pandas as pd
proteins = pd.read_parquet('experiment_proteins.parquet')

# Pivot to wide format for analysis
wide = proteins.pivot_table(
    index=['protein_group_id', 'leading_protein', 'gene_names'],
    columns='replicate_name', 
    values='abundance'
)
```

#### 2. `{name}_peptides.parquet` - Peptide-Level Data

For drilling down into protein quantification or peptide-level analysis.

| Column | Type | Description |
|--------|------|-------------|
| peptide_id | string | Unique peptide identifier |
| peptide_modified | string | Modified sequence |
| peptide_sequence | string | Unmodified sequence |
| protein_group_id | string | Assigned protein group |
| is_shared | bool | Maps to multiple protein groups |
| is_razor | bool | Assigned via razor logic (if applicable) |
| replicate_name | string | Sample identifier |
| sample_type | string | experimental/qc/reference |
| batch | string | Batch identifier |
| run_order | int | Acquisition order |
| abundance | float | Log2 abundance (normalized) |
| abundance_raw | float | Log2 abundance before RT correction |
| uncertainty | float | Propagated from transitions |
| retention_time | float | Best retention time |
| n_transitions | int | Number of transitions used |
| mean_shape_correlation | float | Average shape correlation of transitions |
| min_shape_correlation | float | Minimum (worst) shape correlation |
| idotp | float | Isotope dot product (if available) |
| library_dotp | float | Library dot product (if available) |
| qc_flag | string | Any QC warnings (nullable) |
| residual | float | Median polish residual (peptide→protein rollup) |
| row_effect | float | Peptide ionization effect from median polish (α_i) |
| residual_mean | float | Mean residual for this peptide across samples |
| residual_std | float | Standard deviation of residuals |
| residual_mad | float | Median absolute deviation (robust measure) |
| residual_max_abs | float | Maximum absolute residual |

**Note on residuals:** Following Plubell et al. 2022 ([doi:10.1021/acs.jproteome.1c00894](https://doi.org/10.1021/acs.jproteome.1c00894)), peptides with large residuals should not be automatically discarded - they may indicate biologically interesting proteoform variation, post-translational modifications, or protein processing. The residual columns allow users to identify and investigate these peptides for potential biological significance.

**Method-specific output columns:**

| Column | Rollup Method | Description |
|--------|--------------|-------------|
| residual | median_polish | Deviation from expected (row + column effects) |
| row_effect | median_polish | Peptide ionization efficiency (α_i) |
| topn_selected | topn (protein rollup) | Boolean - was this peptide used for protein quant? |
| transition_weight | adaptive (transition rollup) | Weight assigned to each transition (0-1) |

For **top-N rollup** (peptide→protein), the `topn_selected` column allows users to see exactly which peptides contributed to each protein's abundance. The same peptides are selected across all samples for comparability.

For **adaptive rollup** (transition→peptide only), the `transition_weight` column shows how much each transition contributed, based on learned weights from intensity, m/z, and ShapeCorrelation.

#### 3. `{name}_metadata.json` - Processing Metadata

Complete provenance and parameters for reproducibility.

```json
{
  "pipeline_version": "0.1.0",
  "processing_date": "2024-01-15T10:30:00Z",
  "source_files": ["experiment_transitions.csv"],
  
  "sample_metadata": {
    "n_samples": 48,
    "n_reference": 6,
    "n_qc": 6,
    "n_experimental": 36,
    "batches": ["batch1", "batch2"],
    "samples": [...]
  },
  
  "protein_groups": {
    "n_groups": 2500,
    "n_proteins": 3200,
    "shared_peptide_handling": "all_groups",
    "groups_summary": [...]
  },
  
  "processing_parameters": {
    "transition_rollup": {
      "method": "adaptive",
      "min_transitions": 3,
      "use_ms1": false,
      "adaptive_params": {
        "beta_log_intensity": 0.5,
        "beta_mz": 0.0,
        "beta_shape_corr": 1.0,
        "mz_min": 350.0,
        "mz_max": 1200.0,
        "learned_from": "reference_samples",
        "n_reference_samples": 6
      }
    },
    "rt_correction": {
      "enabled": true,
      "method": "spline",
      "spline_df": 5,
      "per_batch": true
    },
    "global_normalization": {
      "method": "median"
    },
    "batch_correction": {
      "enabled": true,
      "method": "combat"
    },
    "protein_rollup": {
      "method": "sum",
      "min_peptides": 3
    }
  },
  
  "validation_metrics": {
    "reference_cv_before": 0.15,
    "reference_cv_after": 0.08,
    "qc_cv_before": 0.18,
    "qc_cv_after": 0.10,
    "relative_variance_reduction": 1.12,
    "pca_distance_ratio": 0.85,
    "passed_validation": true
  },
  
  "warnings": []
}
```

#### 4. `qc_report.html` - QC Report

An HTML report summarizing normalization quality with embedded or linked diagnostic plots.

**Report Contents:**
- **Header**: Processing summary with timestamp, sample counts, batch information
- **Intensity Distribution**: Box plots showing abundance distributions per sample
- **PCA Analysis**: Principal component analysis before/after normalization to visualize batch effects
- **Control Correlation**: Correlation heatmaps for reference and QC samples
- **CV Distribution**: Coefficient of variation histograms for precision assessment
- **RT Correction**: Before/after comparison showing residuals for reference (fitted) vs QC (held-out validation)

**Configuration:**
```yaml
qc_report:
  enabled: true
  filename: "qc_report.html"
  save_plots: true        # Save individual PNG files
  embed_plots: true       # Embed plots as base64 in HTML
  plots:
    intensity_distribution: true
    pca_comparison: true
    control_correlation: true
    cv_distribution: true
    rt_correction: true
```

When `save_plots: true`, a `qc_plots/` directory is created containing individual PNG files for each plot. When `embed_plots: true`, plots are base64-encoded and embedded directly in the HTML for self-contained reports.

---

### File Naming Convention

**Input:**
- Skyline report: `{experiment}_transitions.csv`
- Sample metadata: `{experiment}_samples.csv`

**Output:**
- `{experiment}_proteins.parquet` - Primary protein-level data
- `{experiment}_peptides.parquet` - Peptide-level data  
- `{experiment}_metadata.json` - Processing parameters and provenance

---

## Batch Estimation

When batch information is not provided in the sample metadata, PRISM can automatically estimate batch assignments. This is essential for ComBat batch correction to function properly.

**Single Batch Handling:** If only one batch is detected (single Skyline document, no acquisition time gaps, and no forced division), PRISM will skip batch correction entirely. This is the expected behavior when all samples were acquired in a single run.

### Priority Order

Batch assignments are determined using the following priority:

1. **Metadata file**: If the metadata file contains a `Batch` or `Batch Name` column (Skyline uses "Batch Name"), those assignments are used directly.

2. **Source documents**: When multiple Skyline report files are loaded, each file is treated as a separate batch. This is common when each batch is exported from a separate Skyline document.

3. **Acquisition time gaps**: If `Acquired Time` is available in the data, PRISM uses IQR-based outlier detection to identify batch breaks. A gap is considered a batch boundary if it exceeds `Q3 + (iqr_multiplier × IQR)`.

   **Example:** If LC-MS runs are typically 65 min apart (±2 min), the IQR would be ~4 min. With the default multiplier of 1.5, the threshold would be ~73 min. An overnight break (e.g., 90 min gap) would be clearly detected as a batch boundary.

4. **Equal division fallback**: If `n_batches` is explicitly configured and no other method found multiple batches, samples are divided into the specified number of batches based on acquisition order.

**Note:** If none of the above methods identify multiple batches, all samples are assigned to a single batch and batch correction is skipped.

### Configuration

```yaml
batch_estimation:
  min_samples_per_batch: 12   # Minimum expected samples per batch
  max_samples_per_batch: 100  # Maximum expected samples per batch
  gap_iqr_multiplier: 1.5     # IQR multiplier for outlier gap detection
  n_batches: null             # Force specific number of batches (fallback only)
```

### Acquired Time Column

To enable time-based batch estimation, include `Result File > Acquired Time` in your Skyline report. This column contains the acquisition timestamp for each result file.

**Important:** The typical batch size in 96-well plate formats is 12-100 samples. PRISM uses this range to validate detected batches and warn about unusual batch sizes.

### API Usage

```python
from skyline_prism import estimate_batches, apply_batch_estimation

# Estimate batches from a DataFrame
result = estimate_batches(
    df,
    metadata=None,
    min_samples_per_batch=12,
    max_samples_per_batch=100,
    gap_iqr_multiplier=1.5,
)
print(f"Detected {result.n_batches} batches via '{result.method}'")

# Apply estimation and add batch column to DataFrame
df, result = apply_batch_estimation(df, metadata=metadata_df)
```

---

## Protein Parsimony and Grouping

### Overview

Proper protein parsimony requires the **search FASTA database** to determine which proteins each detected peptide could have originated from. The peptide-to-protein mapping in Skyline reports is typically limited to the "best" protein match, which is insufficient for:

1. Identifying truly unique vs shared peptides
2. Correctly assigning razor peptides
3. Detecting subsumable proteins

### FASTA Requirement

The parsimony module requires the same FASTA database used for the original search:

```yaml
parsimony:
  fasta_path: "/path/to/search.fasta"
  shared_peptide_handling: "all_groups"  # all_groups, unique_only, razor
```

**Peptide-Protein Mapping:** PRISM uses direct substring matching to map detected peptides to proteins. Each peptide is searched for in all protein sequences, so no enzyme parameters are needed for parsimony. This approach is simpler and works regardless of digestion specificity.

**Note:** Enzyme parameters are only needed for iBAQ quantification (to count theoretical peptides). See the iBAQ section for details.

### The Problem

Peptides can map to multiple proteins due to:
1. **Shared peptides**: Identical sequences in homologous proteins
2. **Protein isoforms**: Alternative splicing variants
3. **Protein families**: Conserved domains across paralogs
4. **Subsumable proteins**: All peptides of protein A are contained in protein B

If we don't handle this properly:
- Shared peptides get counted multiple times in protein rollup
- Protein abundance estimates are inflated/biased
- Statistical testing has inflated degrees of freedom

### Parsimony Strategy

We implement a parsimony algorithm to create **protein groups** where:
1. Each peptide maps to exactly one protein group
2. Protein groups represent the minimal set that explains all peptides
3. Shared peptides go to the group with the most unique peptides (or are distributed)

### Algorithm: Greedy Set Cover with Protein Groups

```
Input: 
  - peptides: set of all peptide sequences
  - protein_to_peptides: dict mapping protein -> set of peptides

Output:
  - protein_groups: list of ProteinGroup objects
  - peptide_to_group: dict mapping peptide -> protein group

Algorithm:

1. REMOVE SUBSET PROTEINS (subsumable)
   For each protein A:
       For each protein B where B != A:
           If peptides(A) ⊆ peptides(B):
               Mark A as subsumable by B
               Remove A from consideration
               Add A to B's "subsumed" list

2. IDENTIFY INDISTINGUISHABLE PROTEINS  
   For proteins with identical peptide sets:
       Group them together as indistinguishable
       Create single ProteinGroup with all member proteins

3. GREEDY ASSIGNMENT OF SHARED PEPTIDES
   remaining_peptides = all peptides
   protein_groups = []
   
   While remaining_peptides not empty:
       # Find protein(s) with most remaining peptides
       best_protein = argmax(|peptides(p) ∩ remaining_peptides|)
       
       # Create protein group
       group = ProteinGroup(
           leading_protein = best_protein,
           member_proteins = [best_protein] + subsumed[best_protein],
           peptides = peptides(best_protein) ∩ remaining_peptides
       )
       protein_groups.append(group)
       
       # Mark these peptides as assigned
       remaining_peptides -= group.peptides

4. CLASSIFY PEPTIDES
   For each protein_group:
       unique_peptides = peptides only in this group
       shared_peptides = peptides also in other groups (before assignment)
       
       # Store both for analysis
       group.unique_peptides = unique_peptides
       group.shared_peptides = shared_peptides
```

### ProteinGroup Data Structure

```python
@dataclass
class ProteinGroup:
    group_id: str                    # Unique identifier
    leading_protein: str             # Representative accession
    leading_protein_name: str        # Gene name or description
    member_proteins: List[str]       # All indistinguishable proteins
    subsumed_proteins: List[str]     # Proteins whose peptides are subset
    
    peptides: Set[str]               # All peptides assigned to this group
    unique_peptides: Set[str]        # Peptides only in this group
    razor_peptides: Set[str]         # Shared peptides assigned here by parsimony
    
    # For quantification decisions
    n_peptides: int
    n_unique_peptides: int
    sequence_coverage: float         # If sequence available
```

### Output: Protein Groups File

```tsv
GroupID	LeadingProtein	LeadingName	MemberProteins	SubsumedProteins	NPeptides	NUniquePeptides	PeptideList
PG0001	P04406	GAPDH_HUMAN	P04406	P04406-2;A0A384	12	8	GALQNIIPASTGAAK;VGVNGFGR;...
PG0002	P68363;P68366	TBA1A_HUMAN;TBA1B_HUMAN	P68363;P68366		8	2	AVFVDLEPTVIDEVR;...
```

### Handling Shared Peptides in Quantification

Three strategies, configurable:

| Strategy | Description | When to use |
|----------|-------------|-------------|
| `all_groups` | Apply shared peptides to ALL protein groups they map to | **Default**. Acknowledges proteoform complexity; avoids assumptions based on FASTA annotations |
| `unique_only` | Only use peptides unique to a single protein group | Most conservative, may lose proteins with few unique peptides |
| `razor` | Assign shared peptides to group with most peptides (MaxQuant-style) | Least preferred; makes strong assumptions about protein presence |

**Rationale for `all_groups` as default:**
Complex proteoforms exist in biology, and we don't know enough to confidently exclude peptides based on protein annotations in a FASTA file. A peptide that maps to multiple proteins may genuinely be present in multiple forms. The downstream analysis (differential expression, etc.) can handle this redundancy better than arbitrary exclusion.

**Implementation for `all_groups`:**
```python
# Each peptide contributes to ALL groups it maps to
for group in protein_groups:
    # group.peptides includes all peptides, shared or unique
    # No filtering based on sharing status
    peptide_matrix = get_peptides_for_group(group, include_shared=True)
    protein_abundance = tukey_median_polish(peptide_matrix)
```

**Note:** When using `all_groups`, the same peptide abundance contributes to multiple protein estimates. This is intentional - it reflects our uncertainty about protein assignment. Downstream statistical methods should be aware of this (e.g., don't treat protein estimates as fully independent).

### Integration with Median Polish

When rolling up peptides to proteins:

```python
def rollup_with_parsimony(peptide_data, protein_groups, method='razor'):
    """
    Roll up peptide abundances to protein groups.
    
    Args:
        peptide_data: DataFrame with peptide abundances (samples as columns)
        protein_groups: ProteinGroup objects from parsimony
        method: 'unique_only', 'razor', or 'distributed'
    
    Returns:
        protein_data: DataFrame with protein group abundances
    """
    results = {}
    
    for group in protein_groups:
        if method == 'unique_only':
            peptides = group.unique_peptides
            if len(peptides) < min_peptides:
                continue  # Skip groups without enough unique peptides
                
        elif method == 'razor':
            peptides = group.peptides  # All assigned peptides
            
        elif method == 'distributed':
            # Will need to handle weighting in median polish
            peptides = group.peptides
            weights = compute_peptide_weights(group, protein_groups)
        
        # Extract peptide subset
        pep_matrix = peptide_data.loc[peptide_data.index.isin(peptides)]
        
        # Apply median polish (or other rollup)
        protein_abundance = tukey_median_polish(pep_matrix)
        
        results[group.group_id] = protein_abundance
    
    return pd.DataFrame(results).T
```

---

## Problem Statement

### Current Approaches and Their Limitations

**DIA-NN RT-windowed normalization:**
- Assumes peptides at each RT window should have equal medians across samples
- Dangerous assumption: biological changes may correlate with RT (e.g., membrane proteins elute late, hydrophobic proteins cluster)
- Can remove real biological signal

**Global normalization (median, quantile, etc.):**
- Ignores RT-dependent technical variation (suppression, spray instability, gradient issues)
- May leave systematic technical artifacts uncorrected

**The fundamental tension:** Any RT-based correction risks removing biology, but ignoring RT leaves technical artifacts.

---

## Proposed Solution: Reference-Anchored RT Normalization

### Core Principle

Use the inter-experiment reference to **learn** what RT-dependent technical variation looks like, then apply corrections anchored to that reference. The reference is independent of the biology being studied, so any RT-dependent variation observed in the reference replicates is purely technical.

### Why This Works

1. **Inter-experiment reference replicates should be identical** - any variation is technical by definition
2. **Reference is matrix-matched** but biologically independent of experimental conditions
3. **Enables expressing quantities relative to a stable anchor** across experiments
4. **Intra-experiment QC validates** that correction didn't collapse biological differences

---

## Experimental Design Requirements

### Control Samples (per batch/plate)

| Control Type | Composition | Purpose | Replicates per Batch |
|--------------|-------------|---------|---------------------|
| Inter-experiment reference | Commercial pool (e.g., Golden West CSF, pooled plasma) | Calibration anchor, RT correction derivation | 1-8 |
| Intra-experiment QC | Pooled experimental samples from current study | Validation, assess prep consistency | 1-8 |

**Note:** In 96-well plate formats, controls are typically placed once per row (8 replicates per batch). Smaller experiments may have as few as 1 replicate per batch.

### Internal QCs (in all samples including controls)

| QC Type | Example | Added When | Purpose |
|---------|---------|------------|---------|
| Protein internal QC | Yeast enolase (16 ng/µg sample) | Before digestion | Digestion efficiency, prep consistency |
| Peptide internal QC | PRTC (30-150 fmol/injection) | Before LC-MS | LC-MS performance, injection consistency |

#### Important Considerations for Internal QCs

**Observed variability in PRTC and enolase peptides** may arise from:

1. **Co-elution suppression**: Internal QC peptides that happen to co-elute with abundant endogenous peptides in the sample matrix will show sample-dependent suppression. This means PRTC may not behave identically between reference and experimental samples.

2. **Non-linear instrument response**: At high abundance, detector saturation and ion competition effects can compress response. At low abundance, noise floor effects inflate variance. This suggests abundance-dependent weighting may be needed.

3. **Matrix-specific effects**: The reference (commercial pool) and experimental samples may have different suppression profiles even at the same RT.

**Implications for RT correction:**
- Do not assume internal QCs can serve as perfect RT anchors
- Model RT-dependent effects from the full peptide distribution in reference, not just QC peptides
- Consider abundance-stratified analysis to detect non-linearity

---

## Normalization Strategy: Peptide-First with Robust Protein Rollup

### Rationale

When normalizing at the protein level first, you implicitly average over RT before addressing RT-dependent biases. If systematic suppression exists at certain RTs, that bias gets baked into protein estimates before correction.

**Peptide-first normalization allows:**
1. RT-dependent effects to be addressed where they occur
2. Abundance-dependent effects to be modeled at the observed level
3. Peptides to be on comparable scales before protein rollup
4. Robust methods to handle outlier peptides without pre-identification

### Processing Order

The pipeline splits into two arms after RT-aware normalization, depending on the desired output level:

```
Raw transition abundances
        │
        ▼
┌───────────────────────────────────────┐
│  STEP 1: Transition → Peptide rollup  │
│  - Combine all transitions per peptide│
│  - Median polish or quality-weighted  │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  STEP 2: Log2 transformation          │
│  - Handle zeros (imputation or +1)    │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  STEP 3: Global normalization         │
│  - Median (default) or VSN            │
│  - [Optional] RT-aware correction     │
└───────────────────────────────────────┘
        │
        ├─────────────────────────────────────────────┐
        │                                             │
        ▼                                             ▼
┌─────────────────────────────┐         ┌─────────────────────────────┐
│  PEPTIDE OUTPUT ARM         │         │  PROTEIN OUTPUT ARM         │
├─────────────────────────────┤         ├─────────────────────────────┤
│                             │         │                             │
│  STEP 4a: Batch correction  │         │  STEP 4b: Protein rollup    │
│  - ComBat or similar        │         │  - Tukey median polish      │
│  - Correct at peptide level │         │  - Robust to outlier peps   │
│                             │         │                             │
│            │                │         │            │                │
│            ▼                │         │            ▼                │
│                             │         │                             │
│  Normalized peptide         │         │  STEP 5b: Batch correction  │
│  abundances                 │         │  - ComBat or similar        │
│                             │         │  - Correct at protein level │
│                             │         │                             │
│                             │         │            │                │
│                             │         │            ▼                │
│                             │         │                             │
│                             │         │  Normalized protein         │
│                             │         │  abundances                 │
└─────────────────────────────┘         └─────────────────────────────┘
```

**Rationale for two-arm design:**

1. **Batch correction should match reporting level**: 
   - If reporting peptide-level data, batch correct peptides directly
   - If reporting protein-level data, roll up first, then batch correct proteins
   - Batch correcting peptides then rolling up can introduce artifacts from the rollup process

2. **Protein rollup before batch correction**: For protein output, rolling up normalized peptides to proteins first allows batch correction to operate on the final quantities of interest. This avoids having batch effects "averaged out" unevenly across peptides during rollup.

### Tukey Median Polish for Protein Quantification

**Model:**
$$y_{ij} = \mu + \alpha_i + \beta_j + \epsilon_{ij}$$

Where:
- $y_{ij}$ = log2 abundance of peptide $i$ in sample $j$
- $\mu$ = overall level (grand effect)
- $\alpha_i$ = peptide effect (ionization efficiency - some peptides ionize better than others)
- $\beta_j$ = sample effect (**this is the protein abundance estimate**)
- $\epsilon_{ij}$ = residual

**Row effects at each rollup stage:**

| Rollup Stage | Row Effects ($\alpha_i$) | Column Effects ($\beta_j$) |
|--------------|--------------------------|---------------------------|
| Transition → Peptide | Transition interference (co-eluting analytes) | Peptide abundance |
| Peptide → Protein | Peptide ionization efficiency | Protein abundance |

**Algorithm:**
```
Initialize: residuals = y_ij

Repeat until convergence:
    1. Subtract row medians (peptide effects):
       α_i = median_j(residuals_ij)
       residuals_ij = residuals_ij - α_i
       
    2. Subtract column medians (sample effects):
       β_j = median_i(residuals_ij)
       residuals_ij = residuals_ij - β_j
       
    3. Update grand effect:
       μ += median(α) + median(β)
       α = α - median(α)
       β = β - median(β)

Output: β_j as protein abundance estimates
```

**Advantages:**
- Robust to outlier peptides (misintegrations, interferences)
- Handles missing peptides naturally
- No need to pre-identify "bad" peptides
- Preserves relative quantification across samples

**Handling proteins with few peptides:**

Median polish requires ≥2 peptides to decompose row/column effects. For proteins with fewer peptides:

| Peptides | Behavior | Output |
|----------|----------|--------|
| 1 peptide | Use peptide abundance directly | Single peptide's log2 abundance |
| 2 peptides | Mean of peptides (no robust advantage) | Mean of two peptide abundances |
| ≥3 peptides | Full median polish | Robust column effects (β_j) |

All proteins are quantified regardless of peptide count. The `n_peptides` column in output allows users to filter by peptide count if desired.

### Alternative Protein Rollup Methods

| Method | Description | When to use |
|--------|-------------|-------------|
| Tukey median polish | Iterative median subtraction, robust to outliers | Default choice, robust |
| Top-N | Average of N most intense peptides | Simple, interpretable |
| maxLFQ | Maximum peptide ratio extraction | When peptide ratios are reliable |
| iBAQ | Sum intensity / theoretical peptide count | Absolute quantification |

**Future consideration: directLFQ**

For very large cohorts (hundreds to thousands of samples), directLFQ offers linear runtime scaling vs. the quadratic scaling of maxLFQ. directLFQ uses an "intensity trace" approach rather than pairwise ratio comparisons.

- **Citation:** Ammar C, Schessner JP, Willems S, Michaelis AC, Mann M. "Accurate label-free quantification by directLFQ to compare unlimited numbers of proteomes." Molecular & Cellular Proteomics. 2023;22(7):100581. doi:10.1016/j.mcpro.2023.100581
- **GitHub:** https://github.com/MannLabs/directlfq
- **Status:** Not yet implemented in PRISM. Consider for future versions when processing very large sample cohorts.

---

## Rollup Hierarchy: Transitions → Peptides → Proteins

The data flows through multiple rollup stages. Each stage can use median polish or other robust methods.

### Key Design Principles

1. **Tukey median polish as default**: Both transition→peptide and peptide→protein rollups use Tukey median polish by default. This provides robust estimation that automatically downweights outliers (interfered transitions, problematic peptides) without requiring explicit quality metrics or pre-filtering.

2. **Complete data matrix**: Skyline imputes integration boundaries for peptides not detected in specific replicates, so we have actual measurements (including zeros) everywhere. No missing value handling is needed.

3. **Median polish doesn't remove signals**: It decomposes the matrix into row effects + column effects + residuals. Outlier signals contribute to residuals, not to the final abundance estimate. This is more robust than filtering.

### Stage 1: Transition to Peptide Rollup (Default: Median Polish)

Skyline reports can include individual transition intensities. These should be combined into peptide-level quantities before normalization.

**Why rollup transitions first?**
- Individual transitions can have interferences (detected via low shape correlation)
- Some transitions may be truncated or poorly integrated  
- Robust combination reduces impact of problematic transitions

**Default method: Tukey Median Polish** - This is the recommended approach because it automatically handles interference without requiring quality metrics from Skyline.

**Available methods:**

#### 1. Tukey Median Polish (Default, Recommended)

Robust iterative algorithm that removes row (transition) and column (sample) effects:
- Automatically downweights outlier transitions through the median operation
- Works on the full transitions × samples matrix
- Produces interpretable transition effects (some transitions consistently fly better)

**Model:**
$$y_{ij} = \mu + \alpha_i + \beta_j + \epsilon_{ij}$$

Where:
- $y_{ij}$ = log2 intensity of transition $i$ in sample $j$
- $\alpha_i$ = transition effect (consistent across samples)
- $\beta_j$ = sample effect = **peptide abundance estimate**
- $\epsilon_{ij}$ = residuals (captures noise and outliers)

#### 2. Quality-Weighted Aggregation (Alternative)

Uses Skyline's per-transition quality metrics to weight the combination. This is an alternative to median polish when you want to explicitly incorporate Skyline's quality scores.

**Key principle: Per-transition weights using intensity-weighted quality**

Shape correlation varies per-transition-per-replicate. To derive a single weight per transition:
- Quality metrics are aggregated using **intensity-weighted averaging**
- High-abundance samples contribute more to the quality assessment
- Rationale: When abundance is high, we expect clean signal; poor correlation is a strong indicator of real interference. When abundance is low, poor correlation could just be noise.

This ensures:
- Consistent treatment across the experiment (same weights for all replicates)
- Transitions with interference in high-abundance samples are heavily downweighted
- Transitions with poor correlation only in low-abundance samples get less penalty

**Required Skyline columns:**
- `Shape Correlation`: Correlation of each transition's elution profile with the median. Low values indicate interference from co-eluting analytes at that precursor→product transition.
- `Coeluting`: Boolean indicating apex within integration boundaries

**Variance model (based on Finney 2012):**
$$\text{var}(signal) = \alpha \cdot I + \beta \cdot I^2 + \gamma + \text{quality\_penalty}$$

Where:
- $\alpha \cdot I$ = shot noise (Poisson counting statistics)
- $\beta \cdot I^2$ = multiplicative noise (ionization efficiency)
- $\gamma$ = additive noise (electronic)
- quality_penalty = function of shape correlation and coelution

**Parameter learning:**
Parameters (α, β, γ, quality penalty terms) are learned by minimizing CV across peptides in reference samples. The optimization uses the same intensity-weighted quality metrics to ensure consistency between learning and application.

#### 3. Sum

Simple sum of transition intensities (converted to linear scale, then back to log2).
- Fast but not robust to outliers
- May be appropriate when transitions are well-curated

**Note on MS1 data:** By default, MS1 signal is **not** used for quantification even if present in the output. Fragment-based quantification is typically more specific. This can be enabled via configuration.

**Handling multiple charge states:** When a peptide has multiple precursor charge states (e.g., +2 and +3), the transitions from all charge states are treated as additional transitions for that peptide. They are combined together in the same transition→peptide rollup step using median polish or quality-weighted aggregation. This approach:
- Treats all transitions equivalently regardless of precursor charge state
- Allows median polish to naturally handle charge state-specific effects as "transition effects"
- Avoids arbitrary decisions about which charge state is "best"
- Maximizes information use when peptides are observed at multiple charge states

### Stage 2: Peptide to Protein Rollup (Default: Median Polish)

After normalization (RT correction and batch correction), combine peptides into protein-level quantities.

**Default method: Tukey Median Polish** - Consistent with the transition→peptide rollup, median polish is the default because it:
- Automatically downweights peptides that behave inconsistently across samples
- Requires no explicit quality metrics or pre-filtering
- Produces interpretable peptide effects (ionization efficiency differences)
- Is robust to outliers from misintegrations or interferences that weren't caught earlier

### Rollup Method Details

#### Tukey Median Polish (Default, Recommended)

See mathematical details section. Key properties:
- Robust to outlier peptides
- Handles missing values naturally
- Produces interpretable peptide effects (ionization efficiency)

#### Top-N

**Critical:** The same N peptides must be used across ALL samples for comparability. Selecting different peptides per sample would introduce bias.

**Peptide selection algorithm:**
```python
def select_top_n_peptides(peptide_matrix, n=3, method='median_abundance'):
    """
    Select N peptides to use for ALL samples.
    
    Selection criteria:
    - median_abundance: Peptides with highest median abundance across samples
    - frequency: Peptides detected in most samples, ties broken by median abundance
    
    Returns list of selected peptide identifiers.
    """
    if method == 'median_abundance':
        median_per_peptide = peptide_matrix.median(axis=1)  # median across samples
        selected = median_per_peptide.nlargest(n).index.tolist()
    elif method == 'frequency':
        # Count non-missing values per peptide
        freq = peptide_matrix.notna().sum(axis=1)
        # Sort by frequency, then by median abundance
        median_per_peptide = peptide_matrix.median(axis=1)
        ranking = pd.DataFrame({'freq': freq, 'median': median_per_peptide})
        ranking = ranking.sort_values(['freq', 'median'], ascending=[False, False])
        selected = ranking.head(n).index.tolist()
    return selected

def rollup_top_n(peptide_matrix, n=3, method='median_abundance'):
    """
    Average of N peptides, using the SAME peptides for all samples.
    """
    selected_peptides = select_top_n_peptides(peptide_matrix, n, method)
    selected_matrix = peptide_matrix.loc[selected_peptides]
    return selected_matrix.mean(axis=0), selected_peptides  # Return both!
```

**Output requirements:**
- The list of selected peptides MUST be included in output (e.g., as `topn_peptides` column)
- This allows users to verify which peptides drove each protein's quantification

**Parameters:**
- `n`: Number of top peptides (default: 3)
- `selection`: How to rank peptides - 'median_abundance' (default) or 'frequency'

#### iBAQ (Intensity-Based Absolute Quantification)

iBAQ normalizes protein abundances by the number of theoretical peptides, enabling comparison of absolute abundance across proteins.

**Algorithm:**
```python
def rollup_ibaq(peptide_matrix, n_theoretical_peptides):
    """Sum of intensities divided by theoretical peptide count."""
    # Convert from log2 to linear
    linear = 2 ** peptide_matrix
    # Sum across peptides
    total_intensity = linear.sum(axis=0)
    # Normalize by theoretical peptides
    ibaq = total_intensity / n_theoretical_peptides
    # Back to log2
    return np.log2(ibaq)
```

**Configuration for iBAQ:**

iBAQ requires enzyme parameters to calculate theoretical peptide counts:

```yaml
protein_rollup:
  method: "ibaq"
  
  # iBAQ-specific settings (required when method=ibaq)
  ibaq:
    fasta_path: "/path/to/search.fasta"
    enzyme: "trypsin"           # Must match search settings
    missed_cleavages: 0         # Typically 0 for counting
    min_peptide_length: 6
    max_peptide_length: 30
```

**FASTA infrastructure (`skyline_prism.fasta`):**

- `parse_fasta()`: Parse UniProt/NCBI format FASTA files
- `digest_protein()`: In-silico digestion with enzyme rules
- `get_theoretical_peptide_counts()`: Get peptide count per protein for iBAQ

**Example usage:**
```python
from skyline_prism.fasta import get_theoretical_peptide_counts

# Get theoretical peptide counts for iBAQ
counts = get_theoretical_peptide_counts(
    "/path/to/search.fasta",
    enzyme="trypsin",
    missed_cleavages=0,  # Strict for iBAQ
)
print(f"GAPDH has {counts['P04406']} theoretical tryptic peptides")
```

#### maxLFQ

```python
def rollup_maxlfq(peptide_matrix):
    """
    Maximum peptide ratio extraction.
    
    For each pair of samples, find the median peptide ratio.
    Solve the system of equations to get protein abundances.
    """
    n_samples = peptide_matrix.shape[1]
    
    # Calculate pairwise median ratios
    ratio_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            ratios = peptide_matrix.iloc[:, i] - peptide_matrix.iloc[:, j]
            ratio_matrix[i, j] = np.nanmedian(ratios)
    
    # Solve for abundances (least squares on ratio constraints)
    # This is simplified - full maxLFQ uses delayed normalization
    abundances = ratio_matrix.mean(axis=1)
    return abundances - abundances.mean()  # Center
```

#### directLFQ (NOT YET IMPLEMENTED)

DirectLFQ is a fundamentally different algorithm from maxLFQ, not just an optimization. While both preserve peptide ratios, directLFQ uses an "intensity trace" approach with linear O(n) runtime scaling, making it suitable for very large cohorts (100s-1000s of samples) where maxLFQ's O(n²) pairwise comparisons become prohibitive.

**Key algorithmic differences from maxLFQ:**
1. Creates an "anchor intensity trace" from a subset of ions/samples
2. Aligns all ion traces to this anchor via normalization
3. Extracts protein intensities from aligned traces
4. Runtime scales linearly with sample count

**Citation:** Ammar C, Schessner JP, Willems S, Michaelis AC, Mann M. "Accurate label-free quantification by directLFQ to compare unlimited numbers of proteomes." Molecular & Cellular Proteomics. 2023;22(7):100581. doi:10.1016/j.mcpro.2023.100581

**GitHub:** https://github.com/MannLabs/directlfq

**Status:** Not implemented in PRISM. The current `maxlfq` method uses the original pairwise ratio approach. For very large cohorts, users should consider using the directLFQ package directly or this may be added in a future version.

---

## Algorithm Specification

### Phase 1: Reference Characterization

**Input:** All inter-experiment reference injections across all batches

**Goal:** Build a model of RT-dependent AND abundance-dependent technical variation from reference replicates

```
For each peptide p:
    1. Extract abundance values from all reference replicates
    2. Calculate peptide-specific metrics:
       - Mean abundance across references
       - CV across references  
       - RT-dependent residual pattern
       - Abundance level (low/medium/high)
    3. Flag peptides with high technical variance (CV > threshold)
    4. Model RT-dependent systematic effects using reference replicates
    5. Check for abundance-dependent variance (heteroscedasticity)
```

**Outputs:**
- Per-peptide technical variance estimates
- RT-dependent correction factors (derived from reference only)
- Abundance-dependent variance model
- Peptide reliability weights

#### Modeling Abundance-Dependent Effects

Non-linearity may manifest as:
- **Compression at high abundance**: Saturating detector response
- **Inflated variance at low abundance**: Noise floor effects

**Detection approach:**
```
1. Bin peptides by mean abundance (e.g., quartiles)
2. For each bin, calculate CV across reference replicates
3. If CV varies systematically with abundance → heteroscedasticity present
4. Consider VSN or variance-stabilizing transformation
```

**VSN (Variance Stabilizing Normalization):**
- Applies asinh transformation calibrated to make variance independent of mean
- Appropriate when you observe abundance-dependent variance
- Can be applied before or instead of log2 transformation

### Phase 1b: Batch and Run Order Assessment

**Batch effects** and **run order effects** are common and should be characterized before correction.

```
For reference replicates:
    1. Plot abundance vs. run order (injection sequence)
    2. Plot abundance vs. batch
    3. Test for systematic trends:
       - Linear drift with run order?
       - Step changes between batches?
       - Interaction between RT and run order?
    
For internal QCs (PRTC, ENO):
    1. Track across all samples, not just reference
    2. Levey-Jennings plots to identify drift
    3. Correlate with run order and batch
```

**Decision tree:**
- If strong run order effect → consider run order as covariate in correction
- If batch effects dominate → fit per-batch RT models
- If both → hierarchical model with batch and run order

### Phase 2: RT-Dependent Correction Factor Estimation

#### Spline-based RT Modeling (Implemented)

The primary approach uses smoothing splines to model RT-dependent technical variation:

```
For each sample s:
    1. Calculate residuals: For each peptide p, compute
       residual_p = log2(sample_abundance_p) - log2(reference_mean_p)
    
    2. Fit smooth spline f(RT) to residuals vs retention time
       - Uses reference samples from the same batch
       - Spline captures systematic RT-dependent deviation
       - Degrees of freedom controls smoothness (default: 5)
    
    3. Correction factor for peptide p:
       correction_p = f(RT_p)
    
    4. Apply correction:
       corrected_abundance_p = raw_abundance_p - correction_p
```

**Implementation details:**
- Uses `scipy.interpolate.UnivariateSpline` for smooth fitting
- Falls back to binned median correction if spline fitting fails
- Can fit per-batch models (recommended) or a single global model
- With multiple reference replicates, uses median RT per peptide for stability

#### Alternative Approaches (Future Work)

The following approaches may be implemented in future versions:

**RUV-style factor analysis:**
- Use SVD/factor analysis on reference replicates to identify unwanted variation factors
- Regress these factors out of experimental samples
- Advantage: Can capture complex, non-RT-dependent technical variation

**ComBat-like RT-window adjustment:**
- Bin peptides by RT and apply empirical Bayes adjustment per window
- Advantage: More robust estimates when peptide counts are low

### Phase 3: Apply Correction to Experimental Samples

```
For each experimental sample s:
    For each peptide p:
        corrected_abundance_ps = raw_abundance_ps - correction_ps
        
        # Or ratio-based:
        normalized_abundance_ps = raw_abundance_ps / reference_abundance_p
```

### Phase 4: Validation Using Intra-Experiment QC

**Success criteria:**

1. **QC variance decreases:** CV of intra-experiment QC replicates should decrease after correction
2. **QC remains distinct from reference:** QC and reference should not collapse together in PCA
3. **Biological signal preserved:** Known biological differences (if any between conditions) should remain
4. **Comparable variance reduction:** Variance reduction in QC should be similar to reference (not much less)

```
Validation metrics:
    - CV_qc_before vs CV_qc_after (should decrease)
    - CV_reference_before vs CV_reference_after (should decrease)
    - Ratio of variance reductions (should be similar)
    - PCA: QC and reference should remain separated
    - If known positives: fold changes should be preserved
```

### Phase 5: ComBat Batch Correction

After RT-aware normalization (if enabled), systematic batch effects are removed using the ComBat algorithm (Johnson et al. 2007). ComBat uses empirical Bayes shrinkage to robustly estimate and remove additive and multiplicative batch effects.

#### ComBat Algorithm Overview

The model assumes:
$$Y_{ijg} = \alpha_g + X\beta_g + \gamma_{ig} + \delta_{ig}\epsilon_{ijg}$$

Where:
- $Y_{ijg}$ = expression of feature $g$ in sample $j$ of batch $i$
- $\alpha_g$ = overall mean for feature $g$
- $X\beta_g$ = biological covariates (treatment, disease state, etc.)
- $\gamma_{ig}$ = additive batch effect (location shift)
- $\delta_{ig}$ = multiplicative batch effect (scale shift)
- $\epsilon_{ijg}$ = error term

**Empirical Bayes shrinkage** borrows information across features to obtain more stable batch effect estimates, especially when sample sizes are small.

#### Implementation

PRISM implements the full parametric ComBat algorithm:

```python
from skyline_prism import combat, combat_from_long, combat_with_reference_samples

# Wide-format API (features x samples)
corrected = combat(data, batch, reference_batch="Batch1")

# Long-format API (PRISM pipeline format)
corrected_df = combat_from_long(
    df, 
    abundance_col="abundance",
    batch_col="batch",
    sample_col="sample_id",
    feature_col="peptide_modified"
)

# With automatic QC evaluation and fallback
corrected_df, eval_result = combat_with_reference_samples(
    df,
    reference_samples=["Ref1", "Ref2"],
    qc_samples=["QC1", "QC2"],
    fallback_on_failure=True  # Revert if correction degrades quality
)
```

#### Preserving Biological Factors (TODO)

**Status: NOT YET IMPLEMENTED**

ComBat can preserve known biological factors during batch correction by including them as covariates in the model matrix. This prevents biological signal from being removed along with batch effects.

**Planned configuration interface:**

```yaml
batch_correction:
  enabled: true
  method: "combat"
  
  # Biological factors to preserve during batch correction
  # These columns must exist in the sample metadata
  preserve_factors:
    - "treatment"      # e.g., "control", "drug_a", "drug_b"
    - "disease_state"  # e.g., "healthy", "disease"
    - "sex"            # e.g., "M", "F"
```

**Implementation requirements:**

1. **Metadata column validation**: Verify that specified factor columns exist in sample metadata
2. **Design matrix construction**: Build covariate matrix from factor columns
3. **Categorical encoding**: Convert categorical factors to dummy variables
4. **Numerical factors**: Pass through as-is (e.g., age, BMI)
5. **Interaction handling**: Consider whether to support interaction terms

**Python API (current, already supports covariates):**

```python
# The underlying combat() function already supports covariates via covar_mod
import numpy as np
from skyline_prism import combat

# Create design matrix for biological factors
# Example: treatment factor with 3 levels
treatment = ["control", "drug_a", "drug_b", "control", "drug_a", "drug_b"]
covar_mod = pd.get_dummies(treatment, drop_first=True).values

# Apply ComBat while preserving treatment effect
corrected = combat(data, batch, covar_mod=covar_mod)
```

**Why this matters:**

Without specifying biological factors, ComBat may remove real biological signal if it happens to be confounded with batch. For example:
- If all disease samples were processed in Batch 1 and all controls in Batch 2
- ComBat would remove the batch effect, but also the disease signal
- Specifying `preserve_factors: ["disease_state"]` prevents this

---

## Implementation Modules

### Module 1: Data Ingestion and Merging

**Functions:**
```python
def validate_skyline_report(filepath: Path) -> ValidationResult:
    """
    Validate that a Skyline report has required columns.
    
    Returns:
        ValidationResult with:
        - is_valid: bool
        - missing_columns: list
        - extra_columns: list
        - warnings: list (e.g., "No isotope dot product column")
    """

def load_skyline_report(filepath: Path, 
                        source_name: str = None) -> pd.DataFrame:
    """
    Load a single Skyline report with standardized column names.
    
    Args:
        filepath: Path to CSV/TSV report
        source_name: Identifier for this document (defaults to filename)
    
    Returns:
        DataFrame with standardized column names
    """

def merge_skyline_reports(report_paths: List[Path],
                          output_path: Path,
                          sample_metadata: pd.DataFrame = None) -> MergeResult:
    """
    Merge multiple Skyline reports into unified parquet.
    
    Steps:
        1. Validate each report
        2. Standardize column names
        3. Check for replicate name collisions
        4. Concatenate with source tracking
        5. Join sample metadata if provided
        6. Write partitioned parquet
    
    Returns:
        MergeResult with:
        - output_path: Path to parquet
        - n_reports: int
        - n_replicates: int
        - n_precursors: int
        - warnings: list
    """

def load_sample_metadata(filepath: Path) -> pd.DataFrame:
    """
    Load and validate sample metadata file.
    
    Validates:
        - Required columns present (ReplicateName, SampleType)
        - SampleType values are valid
        - RunOrder is numeric (if provided)
        - No duplicate ReplicateNames
    
    Notes:
        - Batch column is optional (will be estimated if missing)
        - RunOrder is optional (calculated from Acquired Time if missing)
    """

# Column name mapping from common Skyline exports
SKYLINE_COLUMN_MAP = {
    # Standard Skyline
    'Protein Name': 'protein_names',
    'Protein Accession': 'protein_ids', 
    'Peptide Sequence': 'peptide_sequence',
    'Peptide Modified Sequence': 'peptide_modified',
    'Precursor Charge': 'precursor_charge',
    'Precursor Mz': 'precursor_mz',
    'Best Retention Time': 'retention_time',
    'Total Area Fragment': 'abundance_fragment',
    'Total Area MS1': 'abundance_ms1',
    'Replicate Name': 'replicate_name',
    'Isotope Dot Product': 'idotp',
    'Average Mass Error PPM': 'mass_error_ppm',
    'Library Dot Product': 'library_dotp',
    'Detection Q Value': 'detection_qvalue',
    
    # EncyclopeDIA via Skyline
    'Normalized Area': 'abundance_fragment',
    
    # Alternative naming
    'ProteinName': 'protein_names',
    'ModifiedSequence': 'peptide_modified',
    'PrecursorCharge': 'precursor_charge',
    'RetentionTime': 'retention_time',
}
```

### Module 2: Protein Parsimony

**FASTA-Based Peptide-Protein Mapping**

The `fasta.py` module provides FASTA parsing and peptide-protein mapping for proper parsimony analysis.

**Implementation:**

1. **FASTA Parser** (`parse_fasta`): Parse protein sequences from FASTA file
   - Handles UniProt and NCBI formats
   - Extracts accession, gene name (GN=), description
   - Supports gzipped files

2. **Peptide-Protein Map** (`build_peptide_protein_map_from_fasta`): Build complete mapping
   - Uses direct substring search (no enzyme parameters needed)
   - Maps each detected peptide to ALL proteins whose sequence contains it
   - Handles I/L ambiguity (isoleucine/leucine indistinguishable by MS)

3. **Modified Sequence Handling** (`strip_modifications`): 
   - Strips modifications from detected sequences for FASTA matching
   - Supports Skyline, MaxQuant, UniMod, ProForma formats
   - Handles terminal modifications (n[+42], c[-17])

4. **In-silico Digestion** (for iBAQ only):
   - `digest_protein()`: Generate theoretical peptides with enzyme rules
   - `get_theoretical_peptide_counts()`: Count peptides per protein for iBAQ
   - Supported enzymes: trypsin, trypsin/p, lysc, lysn, argc, aspn, gluc, chymotrypsin

**Usage:**
```python
from skyline_prism.parsimony import build_peptide_protein_map_from_fasta

# No enzyme parameters needed - uses substring matching
pep_to_prot, prot_to_pep, prot_names = build_peptide_protein_map_from_fasta(
    df,
    fasta_path="/path/to/search.fasta",
)
```

**Functions:**
```python
def parse_fasta(fasta_path: str) -> Dict[str, str]:
    """
    Parse FASTA file and return protein_id -> sequence mapping.
    """

def in_silico_digest(sequences: Dict[str, str], 
                     enzyme: str = 'trypsin',
                     missed_cleavages: int = 2,
                     min_length: int = 6,
                     max_length: int = 30) -> Dict[str, Set[str]]:
    """
    Digest protein sequences and return protein_id -> set of peptides.
    """

def build_peptide_protein_map_from_fasta(
    fasta_path: str,
    detected_peptides: Set[str],
    **digest_params
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Build peptide-protein mapping from FASTA for detected peptides only.
    
    Returns:
        - peptide_to_proteins: dict[peptide] -> set of protein IDs
        - protein_to_peptides: dict[protein] -> set of peptides
    """

def build_peptide_protein_map(data: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Build bidirectional mapping between peptides and proteins.
    
    Returns:
        - peptide_to_proteins: dict[peptide] -> set of protein IDs
        - protein_to_peptides: dict[protein] -> set of peptides
    """

def compute_protein_groups(protein_to_peptides: Dict[str, Set[str]],
                           peptide_to_proteins: Dict[str, Set[str]]) -> List[ProteinGroup]:
    """
    Apply parsimony algorithm to create protein groups.
    
    Returns:
        List of ProteinGroup objects with peptide assignments
    """

def export_protein_groups(groups: List[ProteinGroup], 
                          output_path: Path) -> None:
    """
    Export protein groups to TSV file.
    """

def annotate_peptides_with_groups(data: pd.DataFrame,
                                   groups: List[ProteinGroup]) -> pd.DataFrame:
    """
    Add protein group assignment to peptide data.
    
    Adds columns:
        - protein_group_id: assigned group
        - peptide_type: 'unique', 'razor', or 'shared'
        - razor_protein: protein this peptide is assigned to
    """
```

### Module 3: Reference Analysis

**Functions:**
```python
def characterize_reference(data, reference_samples):
    """
    Analyze reference replicates to establish baseline behavior.
    
    Returns:
        - peptide_stats: per-peptide mean, CV, RT
        - rt_model: fitted model of RT-dependent variation
        - reliability_weights: inverse-variance weights for each peptide
    """

def estimate_rt_correction_factors(data, reference_samples, method='spline'):
    """
    Estimate RT-dependent correction factors from reference.
    
    Methods: 'spline', 'loess', 'ruv', 'combat_rt'
    
    Returns:
        - correction_factors: per-sample, per-peptide corrections
    """
```

### Module 3: Correction Application

**Functions:**
```python
def apply_correction(data, correction_factors, method='subtract'):
    """
    Apply correction factors to all samples.
    
    Methods: 'subtract' (additive on log scale), 'ratio', 'regression'
    
    Returns:
        - corrected_data: normalized abundance matrix
    """

def normalize_to_reference(data, reference_samples, method='median_ratio'):
    """
    Express abundances relative to reference.
    
    Returns:
        - normalized_data: abundances as ratios to reference
    """
```

### Module 4: Validation

**Functions:**
```python
def validate_correction(data_before, data_after, qc_samples, reference_samples):
    """
    Assess whether correction improved data quality without overcorrection.
    
    Returns:
        - metrics: dict with CV changes, PCA distances, etc.
        - plots: diagnostic visualizations
        - warnings: flags if overcorrection suspected
    """

def generate_qc_report(validation_results, output_path):
    """
    Create HTML/PDF report summarizing QC metrics.
    """
```

### Module 5: Protein Rollup

**Functions:**
```python
def tukey_median_polish(peptide_matrix, max_iter=20, tol=1e-4):
    """
    Apply Tukey's median polish to peptide × sample matrix.
    
    Args:
        peptide_matrix: DataFrame with peptides as rows, samples as columns
        max_iter: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        - protein_abundances: Series of sample effects (protein estimates)
        - peptide_effects: Series of peptide effects
        - residuals: DataFrame of residuals for diagnostics
        - converged: bool
    """

def rollup_to_protein(peptide_data, protein_column, method='median_polish', 
                      min_peptides=2):
    """
    Aggregate peptides to protein-level quantities.
    
    Methods: 'median_polish', 'maxlfq', 'topn', 'sum', 'mean'
    
    Returns:
        - protein_data: protein × sample abundance matrix
        - peptide_counts: number of peptides per protein
        - rollup_diagnostics: method-specific diagnostics
    """

def flag_outlier_peptides(residuals, threshold=3):
    """
    Identify peptides that are consistent outliers in median polish residuals.
    
    Returns:
        - outlier_flags: DataFrame indicating outlier status
        - outlier_summary: Summary statistics
    """
```

### Module 6: Visualization

**Required plots:**

*RT-dependent effects:*
- RT vs abundance before/after correction (per sample type)
- RT vs CV (binned) for reference replicates
- Heatmap of RT-dependent correction factors by sample

*Variance assessment:*
- CV distributions before/after (QC, reference, experimental)
- MA plot: CV vs mean abundance (check for heteroscedasticity)
- Peptide-level CV vs abundance (check abundance-dependence)

*Batch and run order:*
- PCA colored by sample type, batch, and run order
- Internal QC (PRTC, ENO) Levey-Jennings plots
- Abundance vs run order for reference samples

*Protein rollup diagnostics:*
- Residual distributions from median polish
- Flagged outlier peptides per protein
- Peptide count distribution

*Validation summary:*
- Side-by-side: QC vs reference CV improvement
- PCA before/after showing QC-reference separation maintained

---

## Configuration Parameters

```yaml
# prism_config.yaml

data:
  abundance_column: "TotalAreaFragment"  # or "TotalAreaMs1"
  rt_column: "BestRetentionTime"         # Skyline's aligned RT
  peptide_column: "PeptideModifiedSequence"  # Modified forms are separate peptides
  precursor_column: "PrecursorCharge"    # For precursor-level data
  protein_column: "ProteinAccession"
  sample_column: "ReplicateName"
  batch_column: "Batch"
  run_order_column: "RunOrder"

sample_annotations:
  reference_pattern: "GoldenWest|CommercialPool|InterExpRef"
  qc_pattern: "StudyPool|IntraPool|ExpPool"
  experimental_pattern: "^(?!.*(Pool|Ref)).*"

# Stage 0: Transition to peptide rollup (if needed)
transition_rollup:
  enabled: false  # Set true if input has transition-level data
  method: "median_polish"  # options: median_polish, sum, adaptive
  use_ms1: false  # Whether to include MS1 in quantification (default: no)
  min_transitions: 3  # Minimum transitions required
  
  # For adaptive method - uses learned weights from intensity, m/z, shape_corr
  # Weight formula: w_t = exp(beta_intensity * (log2(I) - center) + beta_mz * mz_norm + beta_shape * shape_corr)
  adaptive_rollup:
    beta_log_intensity: 0.5  # Weight for log2 intensity (must be >= 0)
    beta_mz: 0.0             # Weight for normalized m/z
    beta_shape_corr: 1.0     # Weight for shape correlation
    min_improvement_pct: 5.0 # Required improvement over sum to use adaptive

# Sample outlier detection (one-sided, low signal only)
# Detects samples with abnormally low signal (failed injections, degradation)
# Detection uses LINEAR scale to avoid log scale compression
sample_outlier_detection:
  enabled: true
  action: "report"  # options: report, exclude
  method: "iqr"     # options: iqr, fold_median
  iqr_multiplier: 1.5  # for IQR method: flag if median < Q1 - 1.5*IQR
  fold_threshold: 0.1  # for fold_median: flag if median < 10% of overall

preprocessing:
  log_transform: true
  log_base: 2
  zero_handling: "min_positive"  # options: min_positive, half_min, impute
  variance_stabilization: "none"  # options: none, vsn
  # VSN available as alternative to log2 + median

rt_correction:
  enabled: false  # Disabled by default (search engine RT calibration may not generalize)
  method: "spline"  # options: spline, loess
  spline_df: 5      # degrees of freedom
  loess_span: 0.3   # span parameter if using loess
  per_batch: true   # fit separate models per batch
  
global_normalization:
  method: "median"  # options: median, vsn, quantile, none
  # Default: log2 + median. VSN as alternative.

batch_correction:
  enabled: true
  method: "combat"  # options: combat, none
  # preserve_factors: []  # TODO: biological covariates to preserve (not yet implemented)

# Protein inference
parsimony:
  enabled: true
  shared_peptide_handling: "all_groups"  # options: all_groups, unique_only, razor
  # all_groups: Apply shared peptides to ALL groups (default, recommended)
  # unique_only: Only use peptides unique to one group
  # razor: Assign to group with most peptides (MaxQuant style)

# Peptide to protein rollup
protein_rollup:
  method: "sum"  # options: sum, median_polish, topn, maxlfq, ibaq
  
  # Method-specific parameters
  topn_n: 3                # for topn method
  # iBAQ requires fasta_path in parsimony section
  
validation:
  cv_improvement_threshold: 0.05
  max_pca_collapse: 0.2
  check_abundance_dependence: true
  
filtering:
  min_observations: 0.5      # fraction of samples peptide must be observed in
  max_reference_cv: 0.5      # exclude highly variable peptides from RT model
  min_intensity: 0           # minimum intensity threshold
  quality_filters:
    min_dotp: null           # minimum isotope dot product (optional)
    max_mass_error_ppm: null # maximum mass error (optional)

output:
  transitions_rolled: "transitions_to_peptides.parquet"  # if transition rollup enabled
  transition_residuals: "transition_residuals.parquet"   # median polish residuals per transition
  corrected_peptides: "corrected_peptides.parquet"
  corrected_proteins: "corrected_proteins.parquet"
  peptide_residuals: "peptide_residuals.parquet"         # median polish residuals per peptide
  protein_groups: "protein_groups.tsv"
  qc_report: "normalization_qc_report.html"
  diagnostic_plots: "plots/"
```

---

## Workflow Summary

The PRISM pipeline uses a **two-arm design**: batch correction is applied at the
reporting level (peptide or protein), not before protein rollup. This ensures
that the robust median polish rollup operates on data that has not been
over-smoothed by batch correction.

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT DATA                              │
│  - Skyline report (CSV/TSV) with transition-level data         │
│  - Sample metadata (sample types, batches)                      │
│  - Reference replicates: inter-experiment QC (for RT model)    │
│  - QC replicates (intra-experiment) (for validation)       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│         STEP 1: TRANSITION → PEPTIDE ROLLUP (if enabled)       │
│  - Tukey median polish (default) or quality-weighted sum       │
│  - Aggregate transitions to peptide-level abundances           │
│  - Preserve transition residuals for diagnostics               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│      STEP 2: GLOBAL NORMALIZATION + RT CORRECTION (optional)   │
│  - Log2 transform with median centering                        │
│  - RT-aware spline correction (DISABLED by default)            │
│    * If enabled: learns from reference, validates on QC      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 3: PROTEIN PARSIMONY                          │
│  - Build peptide-protein mappings                               │
│  - Compute protein groups (minimum set cover)                  │
│  - Handle shared peptides (all_groups, razor, unique_only)     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
          ┌───────────────────┴───────────────────┐
          │                                       │
          ▼                                       ▼
┌─────────────────────────┐         ┌─────────────────────────────┐
│   PEPTIDE ARM (4a)      │         │     PROTEIN ARM (4b)        │
│                         │         │                             │
│  ComBat batch           │         │  Peptide → Protein rollup   │
│  correction on          │         │  (Tukey median polish)      │
│  peptide abundances     │         │           │                 │
│                         │         │           ▼                 │
│                         │         │  ComBat batch correction    │
│                         │         │  on protein abundances      │
└───────────┬─────────────┘         └─────────────┬───────────────┘
            │                                     │
            ▼                                     ▼
┌─────────────────────────┐         ┌─────────────────────────────┐
│  corrected_peptides     │         │  corrected_proteins         │
│  (parquet/csv)          │         │  (parquet/csv)              │
│                         │         │                             │
│  peptide_residuals      │         │  protein_groups.tsv         │
│  (from protein rollup)  │         │                             │
└─────────────────────────┘         └─────────────────────────────┘
          │                                       │
          └───────────────────┬───────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 5: QC VALIDATION & REPORTING                  │
│  - CV comparison: before vs after correction                   │
│  - PCA: check batch effect removal                             │
│  - Reference vs QC: ensure correction generalizes            │
│  - Generate HTML QC report with embedded plots                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUTS                                 │
│  - corrected_peptides.parquet   Batch-corrected peptides       │
│  - corrected_proteins.parquet   Batch-corrected proteins       │
│  - protein_groups.tsv           Protein group definitions      │
│  - peptide_residuals.parquet    Median polish residuals        │
│  - transition_residuals.parquet (if transition rollup)         │
│  - qc_report.html               QC metrics and plots           │
│  - metadata.json                Pipeline provenance            │
└─────────────────────────────────────────────────────────────────┘
```

**Key design decisions:**

1. **RT correction is DISABLED by default**: Search engine RT calibration may
   not generalize between samples. Enable only if you have evidence of
   RT-dependent technical variation.

2. **Batch correction at reporting level**: Applied AFTER rollup for proteins,
   ensuring median polish sees the original (non-smoothed) data.

3. **Tukey median polish as default**: For both transition→peptide and
   peptide→protein rollups. Robust to outliers without explicit filtering.

4. **Residuals are preserved**: Outlier transitions/peptides may indicate
   biologically interesting variation (PTMs, proteoforms), not just noise.

---

## Mathematical Details

### Tukey Median Polish: Detailed Algorithm

For a protein with $m$ peptides measured across $n$ samples, we have matrix $Y_{m \times n}$.

**Initialization:**
$$R^{(0)} = Y$$
$$\mu^{(0)} = 0, \quad \alpha^{(0)} = \mathbf{0}_m, \quad \beta^{(0)} = \mathbf{0}_n$$

**Iteration $k$:**

Step 1 - Row (peptide) sweep:
$$\tilde{\alpha}_i = \text{median}_j(R^{(k-1)}_{ij})$$
$$R^{(k-0.5)}_{ij} = R^{(k-1)}_{ij} - \tilde{\alpha}_i$$
$$\alpha^{(k)} = \alpha^{(k-1)} + \tilde{\alpha} - \text{median}(\tilde{\alpha})$$
$$\mu^{(k-0.5)} = \mu^{(k-1)} + \text{median}(\tilde{\alpha})$$

Step 2 - Column (sample) sweep:
$$\tilde{\beta}_j = \text{median}_i(R^{(k-0.5)}_{ij})$$
$$R^{(k)}_{ij} = R^{(k-0.5)}_{ij} - \tilde{\beta}_j$$
$$\beta^{(k)} = \beta^{(k-1)} + \tilde{\beta} - \text{median}(\tilde{\beta})$$
$$\mu^{(k)} = \mu^{(k-0.5)} + \text{median}(\tilde{\beta})$$

**Convergence:** Stop when $\max|R^{(k)} - R^{(k-1)}| < \epsilon$

**Output:** $\beta$ values are the protein abundance estimates (sample effects)

**Handling missing values:** Median polish naturally handles missing data - medians are computed over available values only. This is a key advantage over mean-based methods.

### Spline-Based RT Correction

For sample $s$ and peptide $p$ with retention time $t_p$:

$$y_{sp} = \mu_p + f_s(t_p) + \epsilon_{sp}$$

Where:
- $y_{sp}$ = log2 abundance
- $\mu_p$ = true peptide abundance
- $f_s(t_p)$ = RT-dependent technical effect for sample $s$
- $\epsilon_{sp}$ = random error

**Estimation from reference:**
For reference replicate $r$, since all replicates should be identical:

$$y_{rp} = \mu_p^{ref} + f_r(t_p) + \epsilon_{rp}$$

We estimate $f_r(t_p)$ by fitting a spline to the residuals from the peptide mean:

$$\hat{f}_r(t_p) = \text{spline}(\{t_p, y_{rp} - \bar{y}_p^{ref}\})$$

**Application to experimental samples:**
For each experimental sample $s$ in the same batch as reference $r$:

$$\hat{y}_{sp}^{corrected} = y_{sp} - \hat{f}_r(t_p)$$

### RUV-III Style Approach

Given negative control samples (reference replicates) that should show no variation:

1. Form matrix $Y^{nc}$ of negative control abundances
2. Estimate unwanted factors: $\hat{W} = \text{SVD}(Y^{nc}, k)$
3. For experimental samples: $\hat{Y}^{corrected} = Y - Y \hat{W}(\hat{W}^T\hat{W})^{-1}\hat{W}^T$

### Validation Metric: Relative Variance Reduction

$$RVR = \frac{CV_{qc}^{after} / CV_{qc}^{before}}{CV_{ref}^{after} / CV_{ref}^{before}}$$

- $RVR \approx 1$: Good - similar improvement in QC and reference
- $RVR >> 1$: Warning - QC improved less than reference (possible undercorrection)
- $RVR << 1$: Warning - QC improved more than reference (possible overcorrection/overfitting)

---

## Edge Cases and Considerations

### When RT correction may be inappropriate

1. **Very short gradients** (< 30 min): May not have enough RT resolution
2. **Highly variable chromatography**: If RT shifts substantially between runs, alignment needed first
3. **Matrix-specific suppression**: Reference may not capture sample-specific effects

### No Missing Data in Skyline-Based Workflow

Unlike traditional proteomics pipelines, the Skyline-based workflow preceding PRISM produces complete data matrices:

- **Skyline imputes RT boundaries**: Using information from replicates where each peptide was detected, Skyline determines integration boundaries for all samples
- **Actual measurements everywhere**: Even when a peptide is not detected in a sample, Skyline integrates the signal at the imputed RT, producing an actual measured value (which may be zero or near-zero)
- **No missing value handling needed**: This eliminates the need for imputation strategies that can introduce bias
- **Zero values are real**: Measured zero intensities reflect actual absence of signal, not missing data

This is a unique advantage of the Skyline-based DIA workflow and a key assumption of PRISM.

### Multiple batches

- Fit separate RT models per batch (reference behavior may differ)
- Or fit global model with batch as covariate
- Validate that batch effects are reduced after correction

---

## Dependencies

**Python:**
- numpy, pandas (data manipulation)
- scipy (splines, statistics)
- scikit-learn (PCA, factor analysis)
- statsmodels (regression, LOESS)
- plotnine or matplotlib (visualization)
- pyarrow (efficient I/O)

---

## Testing Strategy

### Unit tests
- Spline fitting with known functions
- Correction factor calculation with synthetic data
- Validation metric calculations

### Integration tests
- Full pipeline on simulated data with known ground truth
- Recovery of spiked-in fold changes after correction

### Validation on real data
- Apply to existing datasets with dual controls
- Compare to global normalization and DIA-NN normalization
- Assess impact on downstream differential expression

---

## Key Design Decisions

This section documents important design decisions made during PRISM development.

### Data Filtering and Quality Control

1. **No transition-level quality filtering**: Peptides are filtered by DIA-NN q-value prior to import into Skyline. PRISM does not perform additional transition-level filtering. Tukey median polish naturally downweights problematic transitions via residuals.

2. **Single-peptide proteins are valid**: Proteins quantified by a single peptide that passed FDR thresholds are reported. No minimum peptide count is enforced - the FDR control happens upstream in DIA-NN.

3. **No missing data handling needed**: Skyline imputes RT boundaries using detected peptides, producing actual measurements (including zeros) for all peptide-sample combinations. This eliminates the need for imputation.

### Normalization Approach

4. **Peptide-level correction first**: All normalization (RT correction, batch correction) is applied at the peptide level before protein rollup. This addresses biases where they occur.

5. **Log2 + median centering as default**: Simple and effective. VSN is available as a configurable alternative for heteroscedastic data.

6. **Per-batch RT models**: Intra-batch variance is expected to be less than inter-batch variance, so separate RT models are fitted per batch.

7. **Internal QCs (PRTC) not used for RT model**: Due to observed variability from co-elution suppression and matrix effects, RT correction uses the full peptide distribution from reference samples rather than internal QC peptides.

### Skyline Integration

8. **Skyline handles RT alignment**: Run-to-run RT drift is already handled by Skyline's peak boundary imputation. PRISM uses Skyline's aligned/imputed RTs.

9. **Charge states merged by Skyline**: The same peptide at different charge states has identical RT. Skyline already merges this information. Modified forms (different RTs) are treated as different peptides.

---

## Implementation Status

This section documents the current implementation status of PRISM features as of December 2024.

### [WORKING] Fully Implemented and Tested

**Data Processing:**

- **Streaming CSV merge**: Memory-efficient merging of multiple Skyline reports (~47GB datasets tested)
- **Automatic column detection**: Handles different Skyline export formats with column name normalization
- **Metadata handling**: Support for both PRISM format (`sample`, `sample_type`, `batch`) and Skyline format (`Replicate Name`, `Sample Type`, `Batch Name`)
- **Sample type detection**: Pattern-based automatic assignment of reference/qc/experimental samples
- **Batch estimation**: Automatic batch detection from source files or acquisition timestamps

**Peptide Quantification:**

- **Transition → Peptide rollup**: Tukey median polish implementation with residual preservation
- **Quality-weighted aggregation**: Alternative rollup method with learned variance models
- **Global normalization**: Median-based normalization (default) applied at peptide level
- **Peptide batch correction**: Full ComBat implementation with empirical Bayes shrinkage

**Protein Quantification:**

- **Protein parsimony**: FASTA-based protein grouping with shared peptide handling
- **Peptide → Protein rollup**: Tukey median polish with multiple peptide handling strategies
- **Protein global normalization**: Median-based normalization applied at protein level
- **Protein batch correction**: Full ComBat implementation applied after protein rollup

**Output and Logging:**

- **Parquet output**: Efficient storage with metadata preservation
- **Log file generation**: Timestamped log files with all processing steps and timings
- **Provenance tracking**: Complete processing metadata in JSON format for reproducibility
- **Residual preservation**: Both peptide and transition residuals saved for downstream analysis

### [DISABLED] Implemented but Disabled by Default

**RT Correction:**

- **Status**: Fully implemented but **disabled by default**
- **Rationale**: Modern search engines (DIA-NN) apply per-file RT calibration that may not generalize between reference and experimental samples
- **Configuration**: Can be enabled via `rt_correction.enabled: true` in config
- **Methods**: Spline-based correction with reference sample fitting

### [ISSUE] Partially Implemented / Known Issues

**QC Reporting:**

- **QC plot generation**: Implemented (intensity distributions, PCA, correlation heatmaps)
- **HTML report generation**: Implemented with embedded plots
- **Known issues**:
  - CV calculation for reference/QC samples not matching sample types correctly (shows NaN)
  - Global median calculations occasionally show NaN for protein data
  - QC report generation warning: "list index out of range" in some edge cases

**ComBat Evaluation:**

- **Traditional ComBat**: Fully implemented and tested
- **Reference-anchored ComBat**: Method implemented but automatic QC-based evaluation not yet active
- **Current behavior**: Always applies ComBat when enabled; fallback logic needs implementation

### [TODO] Not Yet Implemented

**Advanced Features:**

- **VSN normalization**: Placeholder in config but not yet implemented
- **Per-batch RT models with cross-validation**: RT correction uses all data; per-batch fitting needs implementation
- **Automatic ComBat fallback**: QC-based decision to revert correction if quality degrades
- **Missing data imputation**: Not needed for Skyline data (uses boundary imputation)

### [COVERAGE] Test Coverage

- **Overall coverage**: 43% (164 tests passing)
- **High coverage modules**:
  - `fasta.py`: 95% (protein parsimony)
  - `transition_rollup.py`: 93% (peptide aggregation)
  - `batch_correction.py`: 89% (ComBat implementation)
  - `parsimony.py`: 78% (protein grouping)
- **Low coverage modules**:
  - `cli.py`: 13% (command-line interface - mainly integration code)
  - `normalization.py`: 12% (RT correction - disabled by default)
  - `data_io.py`: 28% (file I/O - tested via integration)

**Testing philosophy**: Core algorithms (median polish, ComBat, parsimony) have extensive unit tests. Integration code (CLI, file I/O) is tested via real-world usage on large datasets.

---

## References

1. Tsantilas KA et al. "A framework for quality control in quantitative proteomics." J Proteome Res. 2024. DOI: 10.1021/acs.jproteome.4c00363

2. Gagnon-Bartsch JA, Speed TP. "Using control genes to correct for unwanted variation in microarray data." Biostatistics. 2012.

3. Johnson WE et al. "Adjusting batch effects in microarray expression data using empirical Bayes methods." Biostatistics. 2007.

4. Leek JT, Storey JD. "Capturing heterogeneity in gene expression studies by surrogate variable analysis." PLoS Genet. 2007.

5. Demichev V et al. "DIA-NN: neural networks and interference correction enable deep proteome coverage." Nat Methods. 2020.
