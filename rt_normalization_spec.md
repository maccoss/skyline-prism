# RT-Aware Normalization for Proteomics Data

## Specification Document for Implementation

### Overview

This document specifies an approach to retention time (RT)-aware normalization that borrows ideas from batch correction methods (SVA, ComBat, RUV) while using the dual-control experimental design from the MacCoss lab QC framework. The key insight is to use the **inter-experiment reference** (e.g., commercial plasma/CSF pool) as the calibration standard for deriving normalization factors, and the **intra-experiment pool** (pooled experimental samples) as validation to assess whether the normalization worked without overfitting.

### Design Principles

1. **Peptide-first normalization**: Always normalize at the peptide level to address RT-dependent and abundance-dependent biases where they actually occur, before rolling up to protein level.

2. **Robust protein quantification**: Use Tukey median polish (or similar robust method) to combine peptides into protein-level quantities, minimizing the influence of outlier peptides.

3. **Reference-anchored correction**: Derive RT-dependent correction factors from the inter-experiment reference only, since any variation in reference replicates is purely technical.

4. **Validation with held-out control**: Use the intra-experiment pool to validate that corrections improved data quality without overfitting.

5. **Proper protein inference**: Handle protein parsimony before peptide-to-protein rollup to avoid double-counting shared peptides.

---

## Input and Output Specification

### Overview

This pipeline takes **transition-level** data from Skyline and produces protein-level quantification. We perform our own transition→peptide→protein rollup with quality weighting and learned parameters, rather than using Skyline's aggregated quantities.

### Input: Skyline Transition-Level Report

Export a **transition-level report** from Skyline (not pivoted by replicate). One row per transition per replicate.

#### Required Columns

| Skyline Field | Internal Name | Description |
|---------------|---------------|-------------|
| Protein Name | protein_name | Protein identifier(s) |
| Protein Accession | protein_accession | UniProt accession(s) |
| Peptide Sequence | peptide_sequence | Unmodified sequence |
| Peptide Modified Sequence | peptide_modified | Modified sequence (unique peptide ID) |
| Precursor Charge | precursor_charge | Charge state |
| Precursor Mz | precursor_mz | Precursor m/z |
| Fragment Ion | fragment_ion | e.g., y7, b5, precursor |
| Product Charge | product_charge | Fragment charge |
| Product Mz | product_mz | Fragment m/z |
| Replicate Name | replicate_name | Sample identifier |
| Area | area | Integrated transition area |
| Retention Time | retention_time | Apex retention time |

#### Required Quality Columns

| Skyline Field | Internal Name | Description |
|---------------|---------------|-------------|
| Shape Correlation | shape_correlation | Correlation with median transition profile (0-1). Low values indicate interference. |
| Coeluting | coeluting | Boolean: apex within integration boundaries |

#### Recommended Columns

| Skyline Field | Internal Name | Description |
|---------------|---------------|-------------|
| Start Time | start_time | Integration start |
| End Time | end_time | Integration end |
| Fwhm | fwhm | Full width at half max |
| Background | background | Background/noise estimate |
| Truncated | truncated | Peak truncation flag |
| Isotope Dot Product | idotp | MS1 isotope pattern quality (precursor level) |
| Library Dot Product | library_dotp | Spectral library match |
| Detection Q Value | detection_qvalue | mProphet q-value (DIA) |
| Average Mass Error PPM | mass_error_ppm | Mass accuracy |
| Is Decoy | is_decoy | Decoy flag |

#### Skyline Report Configuration

In Skyline's Edit > Report/Results Grid, create a custom report with these fields:

```
Proteins > Protein Name
Proteins > Protein Accession
Peptides > Peptide Sequence  
Peptides > Peptide Modified Sequence
Precursors > Charge
Precursors > Mz
Precursors > Isotope Dot Product
Transitions > Fragment Ion
Transitions > Product Charge
Transitions > Product Mz
Transition Results > Area
Transition Results > Background
Transition Results > Retention Time
Transition Results > Start Time
Transition Results > End Time
Transition Results > Fwhm
Transition Results > Shape Correlation
Transition Results > Coeluting
Transition Results > Truncated
Replicates > Replicate Name
```

**Important:** Do NOT check "Pivot Replicate Name". Export as long format.

---

### Input: Sample Metadata

A separate CSV/TSV file mapping replicate names to experimental metadata:

| Column | Required | Description |
|--------|----------|-------------|
| replicate_name | Yes | Must match Skyline replicate names exactly |
| sample_type | Yes | One of: `experimental`, `pool`, `reference` |
| batch | Yes | Batch identifier |
| run_order | Yes | Acquisition order within batch (integer) |
| subject_id | No | For paired/longitudinal designs |
| condition | No | Treatment group |
| timepoint | No | For longitudinal studies |
| ... | No | Additional annotations as needed |

**Sample types:**
- `reference`: Inter-experiment reference samples for RT correction and parameter learning
- `pool`: Intra-experiment pooled samples for validation
- `experimental`: Actual experimental samples

Example:
```csv
replicate_name,sample_type,batch,run_order,condition,subject_id
Sample_001,experimental,batch1,1,Treatment,P001
Sample_002,experimental,batch1,2,Control,P002
Pool_01,pool,batch1,3,,
Reference_01,reference,batch1,4,,
Sample_003,experimental,batch1,5,Treatment,P003
```

---

### Output Files

The pipeline produces three output files:

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
| sample_type | string | experimental/pool/reference |
| batch | string | Batch identifier |
| run_order | int | Acquisition order |
| abundance | float | Log2 abundance (normalized, batch-corrected) |
| abundance_raw | float | Log2 abundance before corrections |
| uncertainty | float | Propagated uncertainty (log2 scale std) |
| n_peptides | int | Number of peptides used in rollup |
| n_unique_peptides | int | Number of unique (non-shared) peptides |
| cv_peptides | float | CV across peptides (quality metric) |
| qc_flag | string | Any QC warnings (nullable) |

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
| sample_type | string | experimental/pool/reference |
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
    "n_pool": 6,
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
      "method": "quality_weighted",
      "min_transitions": 3,
      "use_ms1": false,
      "variance_model": {
        "alpha": 1.23,
        "beta": 0.015,
        "gamma": 85.2,
        "delta": 0.89,
        "shape_corr_exponent": 2.1,
        "coelution_penalty": 8.5,
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
      "method": "median_polish",
      "min_peptides": 3
    }
  },
  
  "validation_metrics": {
    "reference_cv_before": 0.15,
    "reference_cv_after": 0.08,
    "pool_cv_before": 0.18,
    "pool_cv_after": 0.10,
    "relative_variance_reduction": 1.12,
    "pca_distance_ratio": 0.85,
    "passed_validation": true
  },
  
  "warnings": []
}
```

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

## Protein Parsimony and Grouping

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
4. **Intra-experiment pool validates** that correction didn't collapse biological differences

---

## Experimental Design Requirements

### Control Samples (per batch/plate)

| Control Type | Composition | Purpose | Replicates per Batch |
|--------------|-------------|---------|---------------------|
| Inter-experiment reference | Commercial pool (e.g., Golden West CSF, pooled plasma) | Calibration anchor, RT correction derivation | 2-4 |
| Intra-experiment pool | Pool of experimental samples from current study | Validation, assess prep consistency | 2-4 |

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

```
Raw peptide abundances
        │
        ▼
┌───────────────────────────────────────┐
│  STEP 1: Log2 transformation          │
│  - Handle zeros (imputation or +1)    │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  STEP 2: RT-aware normalization       │
│  - Reference-anchored correction      │
│  - Address RT-dependent biases        │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  STEP 3: Global normalization         │
│  - Median centering or VSN            │
│  - Address sample loading differences │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  STEP 4: Batch correction             │
│  - ComBat or similar                  │
│  - Address batch effects              │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  STEP 5: Protein rollup               │
│  - Tukey median polish                │
│  - Robust to outlier peptides         │
└───────────────────────────────────────┘
        │
        ▼
Normalized protein abundances
```

### Tukey Median Polish for Protein Quantification

**Model:**
$$y_{ij} = \mu + \alpha_i + \beta_j + \epsilon_{ij}$$

Where:
- $y_{ij}$ = log2 abundance of peptide $i$ in sample $j$
- $\mu$ = overall level (grand effect)
- $\alpha_i$ = peptide effect (ionization efficiency, some peptides fly better)
- $\beta_j$ = sample effect (**this is the protein abundance estimate**)
- $\epsilon_{ij}$ = residual

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

**Implementation note:** For proteins with only 1-2 peptides, median polish degenerates to simple median or single value. Consider flagging these proteins or requiring minimum peptide count.

### Alternative Protein Rollup Methods

| Method | Description | When to use |
|--------|-------------|-------------|
| Tukey median polish | Iterative median subtraction, robust to outliers | Default choice, robust |
| Top-N | Average of N most intense peptides | Simple, interpretable |
| iBAQ | Sum intensity / theoretical peptide count | For absolute/cross-protein comparison |
| maxLFQ | Maximum peptide ratio extraction | When peptide ratios are reliable |
| directLFQ | Direct estimation from peptide ratios | Alternative to maxLFQ, faster |

---

## Rollup Hierarchy: Transitions → Peptides → Proteins

The data flows through multiple rollup stages. Each stage can use median polish or other robust methods.

### Key Design Principles

1. **Complete data matrix**: Skyline imputes integration boundaries for peptides not detected in specific replicates, so we have actual measurements (including zeros) everywhere. No missing value handling is needed.

2. **Per-transition weighting (not per-replicate)**: When using quality-weighted aggregation, a single weight is computed for each transition based on its average quality across all replicates. If a transition has interference in the experiment, it's downweighted everywhere consistently.

3. **Median polish doesn't remove transitions**: It decomposes the matrix into transition effects + sample effects + residuals. Outlier transitions contribute to residuals, not to the final peptide estimate.

### Stage 1: Transition to Peptide Rollup

Skyline reports can include individual transition intensities. These should be combined into peptide-level quantities before normalization.

**Why rollup transitions first?**
- Individual transitions can have interferences (detected via low shape correlation)
- Some transitions may be truncated or poorly integrated  
- Robust combination reduces impact of problematic transitions

**Available methods:**

#### 1. Tukey Median Polish (Default)

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

#### 2. Quality-Weighted Aggregation

Uses Skyline's per-transition quality metrics to weight the combination:

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

### Stage 2: Precursor to Peptide Rollup

If multiple charge states exist for the same modified peptide sequence, combine them.

**Note:** Skyline typically already handles this - the "Peptide" level in Skyline aggregates across charge states. If working with precursor-level data:

```
For each modified peptide sequence:
    precursors (charge states) × samples matrix
           │
           ▼  
    Take maximum or median across charge states
           │
           ▼
    peptide abundance
```

**Rationale for maximum:** The charge state with best ionization/detection should give the most reliable quantification.

### Stage 3: Peptide to Protein Rollup

After normalization (RT correction, global normalization, batch correction), combine peptides into protein-level quantities.

**Important:** Tukey median polish at this level provides similar robustness benefits to quality-weighted aggregation - it naturally downweights peptides that are inconsistent with the majority. The main difference is:
- Quality-weighted: Uses explicit quality metrics to set weights a priori
- Median polish: Uses the data itself to identify and downweight outliers

Both approaches address the same problem (outlier signals) from different angles.

### Rollup Method Details

#### Tukey Median Polish (Default)

See mathematical details section. Key properties:
- Robust to outlier peptides
- Handles missing values naturally
- Produces interpretable peptide effects (ionization efficiency)

#### Top-N

```python
def rollup_top_n(peptide_matrix, n=3):
    """Average of N most intense peptides per sample."""
    for sample in samples:
        sorted_peptides = peptide_matrix[sample].nlargest(n)
        protein_abundance[sample] = sorted_peptides.mean()
```

**Parameters:**
- `n`: Number of top peptides (default: 3)
- Ties: Include all tied peptides at the Nth position

#### iBAQ (Intensity-Based Absolute Quantification)

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

**Note:** Requires theoretical peptide count from in silico digestion of protein sequence.

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

#### directLFQ

Similar principle to maxLFQ but uses direct estimation without iterative optimization. See Mund et al. 2022 for details.

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

**Approach A: Spline-based RT modeling**

```
For each sample s:
    For each reference replicate r in same batch:
        1. Calculate log-ratio: ratio_pr = log2(sample_abundance_p / reference_abundance_p)
        2. Fit smooth function f(RT) to ratios using splines or LOESS
        3. This captures RT-dependent deviation from reference
    
    Correction factor for peptide p in sample s:
        correction_ps = median(f_r(RT_p)) across reference replicates r
```

**Approach B: RUV-style factor analysis**

```
1. Construct matrix Y of reference replicate abundances (peptides × reference injections)
2. Since reference replicates should be identical, perform SVD/factor analysis
3. Factors capturing variance = unwanted technical variation
4. Model: Y = W*α + ε  where W = unwanted factors
5. Apply learned W to experimental samples, regressing out unwanted variation
```

**Approach C: ComBat-like RT-window adjustment**

```
1. Bin RT into windows (e.g., 5-minute bins)
2. For each RT bin:
    a. Calculate location (mean) and scale (variance) from reference replicates
    b. Use empirical Bayes shrinkage to stabilize estimates
    c. Adjust experimental samples to match reference distribution
3. Preserve known biological covariates by including them in the model
```

### Phase 3: Apply Correction to Experimental Samples

```
For each experimental sample s:
    For each peptide p:
        corrected_abundance_ps = raw_abundance_ps - correction_ps
        
        # Or ratio-based:
        normalized_abundance_ps = raw_abundance_ps / reference_abundance_p
```

### Phase 4: Validation Using Intra-Experiment Pool

**Success criteria:**

1. **Pool variance decreases:** CV of intra-experiment pool replicates should decrease after correction
2. **Pool remains distinct from reference:** Pool and reference should not collapse together in PCA
3. **Biological signal preserved:** Known biological differences (if any between conditions) should remain
4. **Comparable variance reduction:** Variance reduction in pool should be similar to reference (not much less)

```
Validation metrics:
    - CV_pool_before vs CV_pool_after (should decrease)
    - CV_reference_before vs CV_reference_after (should decrease)
    - Ratio of variance reductions (should be similar)
    - PCA: pool and reference should remain separated
    - If known positives: fold changes should be preserved
```

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
        - Required columns present
        - SampleType values are valid
        - RunOrder is numeric
        - No duplicate ReplicateNames
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

**Functions:**
```python
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
def validate_correction(data_before, data_after, pool_samples, reference_samples):
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
def tukey_median_polish(peptide_matrix, max_iter=10, tol=1e-4):
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
- CV distributions before/after (pool, reference, experimental)
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
- Side-by-side: pool vs reference CV improvement
- PCA before/after showing pool-reference separation maintained

---

## Configuration Parameters

```yaml
# rt_normalization_config.yaml

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
  pool_pattern: "StudyPool|IntraPool|ExpPool"
  experimental_pattern: "^(?!.*(Pool|Ref)).*"

# Stage 0: Transition to peptide rollup (if needed)
transition_rollup:
  enabled: false  # Set true if input has transition-level data
  method: "median_polish"  # options: median_polish, sum, quality_weighted
  use_ms1: false  # Whether to include MS1 in quantification (default: no)
  min_transitions: 3  # Minimum transitions required
  
  # For quality_weighted method - Skyline column names
  shape_correlation_col: "ShapeCorrelation"  # Correlation with median transition
  coeluting_col: "Coeluting"  # Boolean: apex within integration boundaries

preprocessing:
  log_transform: true
  log_base: 2
  zero_handling: "min_positive"  # options: min_positive, half_min, impute
  variance_stabilization: "none"  # options: none, vsn
  # VSN available as alternative to log2 + median

rt_correction:
  enabled: true
  method: "spline"  # options: spline, loess
  spline_df: 5      # degrees of freedom
  loess_span: 0.3   # span parameter if using loess
  per_batch: true   # fit separate models per batch
  
global_normalization:
  method: "median"  # options: median, vsn, quantile, none
  # Default: log2 + median. VSN as alternative.

batch_correction:
  enabled: true
  method: "combat"  # options: combat, limma, none
  covariates: []    # biological covariates to preserve

# Protein inference
parsimony:
  enabled: true
  shared_peptide_handling: "all_groups"  # options: all_groups, unique_only, razor
  # all_groups: Apply shared peptides to ALL groups (default, recommended)
  # unique_only: Only use peptides unique to one group
  # razor: Assign to group with most peptides (MaxQuant style)

# Peptide to protein rollup
protein_rollup:
  method: "median_polish"  # options: median_polish, topn, ibaq, maxlfq, directlfq
  min_peptides: 3          # minimum peptides per protein
  
  # Method-specific parameters
  topn_n: 3                # for topn method
  ibaq_fasta: null         # path to FASTA for theoretical peptide count
  
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
  corrected_peptides: "corrected_peptides.parquet"
  corrected_proteins: "corrected_proteins.parquet"
  protein_groups: "protein_groups.tsv"
  qc_report: "normalization_qc_report.html"
  diagnostic_plots: "plots/"
```

---

## Workflow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT DATA                              │
│  - Experimental samples with internal QCs (ENO, PRTC)          │
│  - Inter-experiment reference replicates (2-4 per batch)       │
│  - Intra-experiment pool replicates (2-4 per batch)            │
│  - Batch and run order annotations                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1: LOG TRANSFORM + VARIANCE STABILIZATION     │
│  - Log2 transform (handle zeros appropriately)                  │
│  - Check for abundance-dependent variance                       │
│  - Apply VSN if heteroscedasticity detected                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 2: REFERENCE CHARACTERIZATION                 │
│  - Calculate per-peptide statistics from reference              │
│  - Identify RT-dependent technical patterns                     │
│  - Assess batch and run order effects                           │
│  - Estimate peptide reliability weights                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│           STEP 3: RT-AWARE CORRECTION (PEPTIDE LEVEL)           │
│  - Model RT-dependent deviation from reference                  │
│  - Use spline/LOESS/RUV/ComBat-style approach                  │
│  - Correction factors derived ONLY from reference               │
│  - Apply per-batch if batch effects present                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 4: GLOBAL NORMALIZATION (PEPTIDE LEVEL)       │
│  - Median centering or additional VSN                          │
│  - Address remaining sample loading differences                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 5: BATCH CORRECTION (PEPTIDE LEVEL)           │
│  - ComBat or limma removeBatchEffect                           │
│  - Preserve biological covariates                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 6: VALIDATION (PEPTIDE LEVEL)                 │
│  - Check: Pool CV decreased?                                    │
│  - Check: Pool distinct from reference?                         │
│  - Check: Variance reduction comparable?                        │
│  - Generate peptide-level QC report                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 7: PROTEIN ROLLUP                             │
│  - Tukey median polish (robust to outlier peptides)            │
│  - Minimum peptide filter                                       │
│  - Propagate uncertainty estimates                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 8: PROTEIN-LEVEL VALIDATION                   │
│  - Check protein-level CVs                                      │
│  - Compare to peptide-level results                            │
│  - Final QC report                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUTS                                    │
│  - Corrected peptide abundance matrix                          │
│  - Corrected protein abundance matrix                          │
│  - QC report with validation metrics                           │
│  - Diagnostic plots                                            │
└─────────────────────────────────────────────────────────────────┘
```

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

$$RVR = \frac{CV_{pool}^{after} / CV_{pool}^{before}}{CV_{ref}^{after} / CV_{ref}^{before}}$$

- $RVR \approx 1$: Good - similar improvement in pool and reference
- $RVR >> 1$: Warning - pool improved less than reference (possible undercorrection)
- $RVR << 1$: Warning - pool improved more than reference (possible overcorrection/overfitting)

---

## Edge Cases and Considerations

### When RT correction may be inappropriate

1. **Very short gradients** (< 30 min): May not have enough RT resolution
2. **Highly variable chromatography**: If RT shifts substantially between runs, alignment needed first
3. **Matrix-specific suppression**: Reference may not capture sample-specific effects

### Handling missing data

- Peptides missing in reference cannot be corrected this way
- Consider imputation or falling back to global normalization for these peptides
- Report fraction of peptides corrected vs. globally normalized

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

**R alternative:**
- limma (for ComBat-style correction)
- sva (surrogate variable analysis)
- proBatch (proteomics batch correction)

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

## Open Questions for Discussion

### Resolved Questions

1. **~~Should internal QCs (PRTC) be used in the RT model?~~** *Resolved: No, due to observed variability from suppression and matrix effects. Use full peptide distribution from reference instead.*

2. **~~Protein-level vs peptide-level correction?~~** *Resolved: Always peptide-level first, then robust rollup to protein.*

3. **~~How to handle RT drift between injections?~~** *Resolved: Skyline already handles run-to-run RT alignment for peak boundary imputation. The same peptide signal is aligned between replicates. Raw RTs may differ, but Skyline's aligned RTs should be used.*

4. **~~How does this interact with match-between-runs?~~** *Resolved: Skyline integrates signal at the RT determined from other replicates where the peptide was detected. Alignment is already done.*

5. **~~VSN vs log2 + median normalization?~~** *Resolved: Use log2 + median normalization as default, with VSN as a configurable option.*

6. **~~Per-batch RT models vs global?~~** *Resolved: Expect intra-batch variance < inter-batch variance, so per-batch models are appropriate. Can share information across batches if reference replicates are limited.*

7. **~~How many peptides minimum for median polish?~~** *Resolved: Require 3+ peptides for robust estimates. Also consider transition-to-peptide rollup using median polish (similar to MSstats approach).*

8. **~~Should we model RT × abundance interaction?~~** *Resolved: Backburner for v2. Focus on 1D RT correction first.*

9. **~~Run order correction: linear vs spline?~~** *Resolved: Use LOESS or spline fit, not linear. Effects are unlikely to be globally linear but may be locally linear.*

10. **~~Handling charge states and modifications?~~** *Resolved: Same peptide at different charge states has identical RT - Skyline already merges this into peptide information. Modified forms have different RTs and should be treated as different peptides.*

### Remaining Open Questions

1. **Transition quality filtering before rollup?** Should we filter transitions by quality metrics (e.g., dotp) before median polish to peptide?

2. **Minimum observations threshold?** What fraction of samples must a peptide be observed in to be included?

3. **How to handle proteins with only 1-2 peptides after filtering?** Report with warning, or exclude?

---

## References

1. Tsantilas KA et al. "A framework for quality control in quantitative proteomics." J Proteome Res. 2024. DOI: 10.1021/acs.jproteome.4c00363

2. Gagnon-Bartsch JA, Speed TP. "Using control genes to correct for unwanted variation in microarray data." Biostatistics. 2012.

3. Johnson WE et al. "Adjusting batch effects in microarray expression data using empirical Bayes methods." Biostatistics. 2007.

4. Leek JT, Storey JD. "Capturing heterogeneity in gene expression studies by surrogate variable analysis." PLoS Genet. 2007.

5. Demichev V et al. "DIA-NN: neural networks and interference correction enable deep proteome coverage." Nat Methods. 2020.
