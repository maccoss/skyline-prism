# PRISM Methods Documentation

This document provides detailed descriptions of all computational methods implemented in PRISM (Proteomics Robust Integrated Skyline Methods). It is intended to serve as a reference for manuscript methods sections and as technical documentation for users.

## Table of Contents

1. [Data Formats and Schemas](#data-formats-and-schemas)
2. [Transition → Peptide Rollup](#transition--peptide-rollup)
3. [Global Normalization](#global-normalization)
4. [Batch Correction](#batch-correction)
5. [Protein Parsimony](#protein-parsimony)
6. [Peptide → Protein Rollup](#peptide--protein-rollup)
7. [Quality Control and Outlier Detection](#quality-control-and-outlier-detection)

---

## Data Formats and Schemas

### Input Data

PRISM accepts Skyline transition-level exports in CSV format. Required columns include:

| Column | Description |
| ------ | ----------- |
| `Peptide Modified Sequence` | Modified peptide sequence (e.g., `C[+57]PEPTIDEK`) |
| `Protein Accession` | UniProt or other protein identifier |
| `Replicate Name` | Sample/replicate identifier |
| `Fragment Ion` | Transition identifier (e.g., `y6`, `b3`) |
| `Area` | Integrated peak area (linear scale) |
| `Retention Time` | Chromatographic retention time (minutes) |

Optional quality columns:

| Column | Description |
| ------ | ----------- |
| `Shape Correlation` | R^2^ of the transition XIC trace to the median trace (0-1) |
| `Product Mz` | Fragment ion m/z |
| `Precursor Charge` | Precursor ion charge state |
| `Product Charge` | Fragment ion charge state |
| `Batch` | Batch identifier for batch correction |

### Output Parquet Schemas

#### Peptide Abundances (`corrected_peptides.parquet`)

| Column | Type | Description |
| ------ | ---- | ----------- |
| `peptide` | string | Modified peptide sequence |
| `protein_accession` | string | Associated protein(s) |
| `mean_rt` | float64 | Mean retention time across samples |
| `<sample_1>` ... `<sample_n>` | float64 | Linear-scale abundance per sample |

#### Protein Abundances (`corrected_proteins.parquet`)

| Column | Type | Description |
| ------ | ---- | ----------- |
| `protein_group` | string | Protein group identifier |
| `leading_protein` | string | Representative protein accession |
| `leading_uniprot_id` | string | UniProt accession for the leading protein |
| `leading_gene_name` | string | Gene symbol for the leading protein |
| `leading_description` | string | Full protein description/name for the leading protein |
| `n_peptides` | int64 | Number of peptides in group |
| `<sample_1>` ... `<sample_n>` | float64 | Linear-scale abundance per sample |

**Leading metadata semantics:** The `leading_` prefix indicates that these fields describe the canonical representative (leading protein) of the parsimony group, not all member proteins. Values are populated from Skyline export columns (`Protein Accession`, `Protein Gene`, `Protein`).

---

## Transition → Peptide Rollup

Transitions (fragment ion peaks) from the same peptide are aggregated to produce peptide-level quantification. PRISM implements several methods.

### Sum (Default)

The simplest and most widely used approach. For each peptide, the linear-scale intensities of all fragment ions are summed:

$$
I_{peptide,s} = \sum_{t=1}^{T} I_{t,s}
$$

Where $I_{t,s}$ is the intensity of transition $t$ in sample $s$.

**Strengths:** Simple, interpretable, preserves total signal.

**Weaknesses:** All transitions weighted equally regardless of quality.

### Consensus (Inverse-Variance Weighted)

A novel approach that weights transitions by their consistency across samples. The key assumption is that all transitions from the same peptide should show identical fold-changes across samples.

**Algorithm:**

1. **Model:** For peptide P with transitions $T_1, ..., T_n$ across samples $S_1, ..., S_m$:
   $$\log_2(I_{ij}) = \alpha_i + \beta_j + \epsilon_{ij}$$
   
   Where:
   - $\alpha_i$ = transition-specific offset (fragmentation efficiency)
   - $\beta_j$ = sample-specific abundance (peptide quantity)
   - $\epsilon_{ij}$ = residual (measurement error, interference)

2. **Estimate offsets:** $\hat{\alpha}_i = \text{median}_j(\log_2(I_{ij}))$

3. **Estimate sample effects:** $\hat{\beta}_j = \text{median}_i(\log_2(I_{ij}) - \hat{\alpha}_i)$

4. **Calculate residuals:** $\hat{\epsilon}_{ij} = \log_2(I_{ij}) - \hat{\alpha}_i - \hat{\beta}_j$

5. **Compute weights:** $w_i = \frac{1}{\text{Var}(\hat{\epsilon}_{i,\cdot}) + \lambda}$
   
   Where $\lambda$ is a regularization parameter (default: 0.1).

6. **Weighted aggregation:** 
   $$I_{peptide,s} = \sum_{t=1}^{T} w_t \cdot I_{t,s}$$
   
   With weights normalized to preserve the scale of the sum method.

**Strengths:** Down-weights transitions with interference or inconsistent behavior; learns from the data rather than relying on external quality metrics.

**Weaknesses:** Requires sufficient samples to estimate transition variance.

### Median Polish (Tukey)

Applies Tukey's median polish algorithm to the transition × sample matrix, fitting the additive model:

$$\log_2(I_{ij}) = \mu + \alpha_i + \beta_j + \epsilon_{ij}$$

The column effects ($\beta_j$) represent sample-specific peptide abundance.

**Algorithm:**

1. Initialize residuals $R = \log_2(I)$
2. Iterate until convergence:
   - Sweep row medians: $\alpha_i \leftarrow \alpha_i + \text{median}_j(R_{i,\cdot}); R \leftarrow R - \text{median}_j(R_{i,\cdot})$
   - Sweep column medians: $\beta_j \leftarrow \beta_j + \text{median}_i(R_{\cdot,j}); R \leftarrow R - \text{median}_i(R_{\cdot,j})$
3. Return $\beta_j$ as peptide abundances

**Strengths:** Robust to outliers; provides residuals for detecting sample × transition interactions.

**Weaknesses:** Requires log-transformation; slower than sum.

### Adaptive (Learned Weights)

Optimizes transition weights to minimize coefficient of variation (CV) on reference samples.

**Features used:**
- Product m/z
- Shape correlation outlier frequency

**Algorithm:**

1. Pre-compute per-transition metrics across reference samples
2. Optimize weight function: $\log(w_t) = \beta_{mz} \cdot \text{norm\_mz} + \beta_{outlier} \cdot f_{outlier}$
3. Minimize median CV on reference samples
4. Validate on QC samples
5. Fall back to sum if no improvement

### Top-N

Selects the N best transitions for each peptide and uses only those.

**Selection methods:**
- `correlation`: Highest median shape correlation
- `intensity`: Highest mean intensity

### Library-Assisted Rollup

Uses a spectral library to inform transition weighting and detect interference. The core idea is that DIA spectra contain signal from 10-30 co-fragmenting peptides, so interference is the norm rather than the exception. Library knowledge of expected fragment ratios can help identify and correct for interfered transitions.

**Inputs:**
- Observed transition intensities (linear scale)
- Library spectrum with expected relative fragment intensities (normalized to base peak = 1.0)

**Library Matching:**
Transitions from Skyline are matched to library fragments by:
1. Product m/z (within tolerance, default 0.02 Da)
2. Fragment ion type (e.g., `y6`, `b3`, `precursor [M+1]`) when available

**Supported Library Formats:**
- **BLIB (Skyline):** SQLite-based format with zlib-compressed peak arrays
- **Carafe TSV (DIA-NN):** Tab-separated format with fragment annotations

**Algorithm: Iterative Least Squares with Outlier Removal**

The core equation solved is:

$$\text{observed} = s \times \text{library} + \text{residuals}$$

Where $s$ is the scale factor that best fits the observed intensities to the expected library pattern.

**Closed-form solution:**
$$s^* = \frac{\vec{L} \cdot \vec{O}}{\vec{L} \cdot \vec{L}}$$

Where $\vec{L}$ is the library intensity vector and $\vec{O}$ is the observed intensity vector.

**Key Design Principles:**

| Principle | Rationale |
|-----------|-----------|
| **Zeros are valid** | A low-abundance peptide may only have signal in the top 1-2 most intense library fragments. Zeros in minor fragments confirm absence of interference. |
| **Only HIGH residuals are outliers** | Interference adds signal (observed > expected). Low/zero signal is not interference, it's low abundance or noise. |
| **Iterative outlier removal** | Fragments with large positive residuals are removed and the model is refit to get a cleaner scale estimate. |
| **Abundance from scaled library** | Final abundance = $s \times \sum L_i$ uses the library sum, not the observed sum, for consistent quantification. |
| **Flag poor fits** | Peptides with R² < 0.5 across all replicates may indicate false positive identifications. |

**Detailed Algorithm:**

1. **Match fragments:** Map observed transitions to library m/z values (tolerance: 0.02 Da)

2. **Initial fit:** Compute closed-form least squares scale factor:
   $$s = \frac{\sum_i L_i \cdot O_i}{\sum_i L_i^2}$$

3. **Compute residuals:** $r_i = O_i - s \times L_i$

4. **Identify outliers:** Fragments with large positive residuals (signal > expected) indicate interference:
   - Compute MAD of positive residuals
   - Flag fragments with z-score > 3.0 (MAD-scaled)

5. **Refit:** Exclude outlier fragments and recalculate scale factor

6. **Iterate:** Repeat steps 3-5 until no new outliers (max 5 iterations)

7. **Calculate abundance:**
   $$I_{peptide} = s \times \sum_{t} L_t$$

   This imputes what the total signal *should* have been based on the library pattern.

8. **Quality assessment:**
   - R² of final fit (goodness of fit)
   - Number of fragments used vs. total
   - List of outlier fragments (potential interference)

**Output Structure:**

| Field | Description |
|-------|-------------|
| `scale` | Fitted scaling factor |
| `abundance` | Peptide abundance (scale * sum of library) |
| `r_squared` | Goodness of fit (0-1) |
| `n_matched` | Number of fragments used in final fit |
| `outlier_indices` | Indices of fragments flagged as interfered |
| `is_reliable` | True if R-squared >= 0.5 |
| `quality_warning` | "poor_fit" or "many_outliers" if applicable |

**Performance Optimizations:**

The library-assisted rollup uses vectorized least squares to process all samples in parallel:

- **Vectorized matrix operations**: Uses BLAS matrix operations via NumPy for O(T * S) complexity where T=transitions, S=samples
- **Single-pass fitting**: All samples fitted simultaneously rather than per-sample loops
- **Speedup**: ~10x faster than per-sample iteration on large datasets (500+ samples)

**Implementation:** `spectral_library.py` -> `least_squares_rollup_vectorized()`

**Performance Characteristics:**

From validation on 238 samples across 3 batches:

| Metric | Library-Assisted | Sum |
|--------|------------------|-----|
| Reference CV (median) | 38.1% | 35.8% |
| QC CV (median) | 41.9% | 39.1% |
| Peptides with lower CV | 29% | 71% |

The library-assisted method shows:
- **Dramatic improvement** (up to 111% CV reduction) for ~29% of peptides with real interference
- **Slightly worse CV** for peptides where library pattern doesn't match data (false discovery, very low signal)

**When Library-Assisted Helps:**
- Peptides with variable interference across replicates
- Cases where 1-2 fragments have consistent co-eluting interference
- High-intensity peptides where interference is detectable above noise

**When Library-Assisted May Hurt:**
- Library not fine-tuned to a specific instrument/collision energy
- Suspect peptide detections (library pattern doesn't match)
- Very low-abundance peptides (noise dominates)

**Diagnostic Applications:**

Peptides with consistently poor R-squared across all replicates should be flagged as suspect identifications. If the observed fragmentation pattern never matches the library, the peptide may be:
1. A false positive identification from DIA-NN/search engine
2. Correctly identified but with systematic fragmentation differences
3. Subject to overwhelming interference in all samples

---

## Global Normalization

Corrects for systematic differences in sample loading.

### RT-Lowess (Recommended)

Retention time-dependent normalization using locally weighted scatterplot smoothing (LOWESS). Corrects for RT-dependent systematic effects such as ion suppression gradients.

**Algorithm:**

1. **Define RT grid:** Create uniform grid of N points (default: 100) across the RT range
2. **Fit per-sample curves:** For each sample, fit LOWESS to $\log_2(\text{abundance})$ vs. RT
3. **Compute global curve:** Take median of all sample curves at each RT point
4. **Calculate corrections:** $\text{correction}_{s,rt} = \text{global}_{rt} - \text{sample}_{s,rt}$
5. **Apply corrections:** $\log_2(I'_{p,s}) = \log_2(I_{p,s}) + \text{correction}_{s,RT_p}$

**Parameters:**
- `frac`: Fraction of data used for local regression (default: 0.3)
- `n_grid_points`: Number of RT grid points (default: 100)

### Median Normalization

Simple global shift to align sample medians:

$$\log_2(I'_{p,s}) = \log_2(I_{p,s}) - \text{median}_p(\log_2(I_{p,s})) + \text{global\_median}$$

### Variance Stabilizing Normalization (VSN)

Applies arcsinh transformation to stabilize variance across intensity ranges:

$$I'_{p,s} = \text{arcsinh}(a \cdot I_{p,s})$$

Where $a$ is optimized to minimize heteroscedasticity.

### Quantile Normalization

Forces all samples to have identical intensity distributions by ranking and replacing with average quantiles.

---

## Batch Correction

Removes systematic differences between experimental batches while preserving biological variation.

### ComBat (Empirical Bayes)

Implementation of the ComBat algorithm (Johnson et al., 2007).

**Model:**
$$Y_{ijg} = \alpha_g + X\beta_g + \gamma_{ig} + \delta_{ig}\epsilon_{ijg}$$

Where:
- $\alpha_g$ = overall mean for feature g
- $X\beta_g$ = covariates (biological groups)
- $\gamma_{ig}$ = additive batch effect for batch i, feature g
- $\delta_{ig}$ = multiplicative batch effect

**Empirical Bayes shrinkage:**

Batch effect parameters are shrunken toward their prior distributions:
- $\gamma \sim N(\bar{\gamma}, \tau^2)$
- $\delta^2 \sim \text{InverseGamma}(a, b)$

This "borrows strength" across features to improve estimation when batches have few samples.

**Reference batch option:** One batch can be designated as reference and left unadjusted.

### ComBat with Reference Sample Preservation

Modified ComBat that preserves the variance structure of reference samples (pooled QC). Uses reference samples to estimate the target variance, then applies correction to align all batches.

---

## Protein Parsimony

Assigns peptides to protein groups using a minimal set cover approach.

### Algorithm

1. **Build peptide-protein mappings** from Skyline export or FASTA database

2. **Identify subsumable proteins:** Proteins whose peptides are a strict subset of another protein's peptides are marked as "subsumable" and merged with the subsuming protein.

3. **Identify indistinguishable proteins:** Proteins with identical peptide sets are grouped together.

4. **Greedy assignment of shared peptides:**
   - Peptides mapping to a single protein/group are "unique"
   - Shared peptides are assigned to the group with the most unique peptides ("razor" peptides)

### Protein Groups

Each protein group contains:
- **Leading protein:** Representative protein (sorted alphabetically or by evidence)
- **Leading metadata:** Group-level identifiers and names are taken from the leading protein and exposed as `leading_uniprot_id`, `leading_gene_name`, and `leading_description` in outputs.
- **Member proteins:** Indistinguishable proteins with identical peptides
- **Subsumed proteins:** Proteins whose peptides are a subset
- **Unique peptides:** Map exclusively to this group
- **Razor peptides:** Shared peptides assigned to this group

### Shared Peptide Handling Options

| Option | Description |
|--------|-------------|
| `all_groups` | Include shared peptides in ALL groups they map to (full intensity, no splitting) |
| `razor` | Assign each shared peptide to one group only (the one with most unique peptides) |
| `unique_only` | Use only unique peptides for quantification |

---

## Peptide → Protein Rollup

Aggregates peptide-level abundance to protein-level.

### Sum

Simple sum of peptide abundances:
$$I_{protein,s} = \sum_{p \in \text{group}} I_{p,s}$$

### MaxLFQ

Maximum label-free quantification algorithm (Cox et al., 2014).

**Algorithm:**

1. For each pair of samples (i, j), compute median peptide log-ratio:
   $$r_{ij} = \text{median}_p(\log_2(I_{p,i}) - \log_2(I_{p,j}))$$

2. Solve for protein abundances that best explain these ratios:
   $$\hat{\beta}_s = \text{mean}_j(r_{sj})$$

3. Center and scale to preserve absolute level

**Strengths:** Robust to missing peptides; uses only peptides present in both samples being compared.

### Median Polish

Same algorithm as transition→peptide rollup, applied to peptide × sample matrix.

### Top-N

Uses only the N most abundant peptides per protein:
- Selection by median intensity
- Selection by number of transitions

### iBAQ (Intensity-Based Absolute Quantification)

Normalizes protein intensity by the number of theoretically observable peptides:

$$\text{iBAQ}_g = \frac{\sum_{p \in g} I_p}{N_{\text{theoretical}}}$$

Where $N_{\text{theoretical}}$ is computed by *in silico* trypsin digestion of the protein sequence.

---

## Quality Control and Outlier Detection

### Sample Outlier Detection

Methods for identifying problematic samples:

| Method | Description |
|--------|-------------|
| `iqr` | Flag samples with median intensity outside 1.5×IQR |
| `zscore` | Flag samples with z-score > threshold |
| `pca` | Flag samples that are PCA outliers |

### Sample Outlier Actions

| Action | Description |
|--------|-------------|
| `flag` | Mark but include in analysis |
| `exclude` | Remove from analysis |
| `review` | Require manual review |

### QC Metrics Computed

- **Reference CV:** Coefficient of variation across reference samples per peptide
- **QC CV:** Coefficient of variation across QC samples per peptide
- **Median intensity:** Per-sample and per-peptide
- **Missing rate:** Fraction of zero/missing values
- **PCA coordinates:** First N principal components for visualization

---

## References

1. Johnson WE, Li C, Rabinovic A (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics* 8(1):118-127.

2. Cox J, Hein MY, Luber CA, Paron I, Nagaraj N, Mann M (2014). Accurate proteome-wide label-free quantification by delayed normalization and maximal peptide ratio extraction, termed MaxLFQ. *Molecular & Cellular Proteomics* 13(9):2513-2526.

3. Tukey JW (1977). Exploratory Data Analysis. Addison-Wesley.

4. Schwämmle V, León IR, Jensen ON (2013). Assessment and improvement of statistical tools for comparative proteomics analysis of sparse data sets with few experimental replicates. *Journal of Proteome Research* 12(9):4215-4224.
