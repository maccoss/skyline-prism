# PRISM Methods Documentation

This document provides detailed descriptions of all computational methods implemented in PRISM (Proteomics Robust Integrated Skyline Methods). It is intended to serve as a reference for manuscript methods sections and as technical documentation for users.

## Table of Contents

1. [Data Formats and Schemas](#data-formats-and-schemas)
2. [Transition → Peptide Rollup](#transition--peptide-rollup)
3. [Global Normalization](#global-normalization)
4. [Batch Correction](#batch-correction)
5. [Protein Parsimony](#protein-parsimony)
6. [Peptide → Protein Rollup](#peptide--protein-rollup)
7. [Marker-Protein Normalization](#marker-protein-normalization)
8. [Quality Control and Outlier Detection](#quality-control-and-outlier-detection)

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
| `Shape Correlation` | R² of the transition XIC trace to the median trace (0-1) |
| `Product Mz` | Fragment ion m/z |
| `Precursor Charge` | Precursor ion charge state |
| `Product Charge` | Fragment ion charge state |
| `Batch` | Batch identifier for batch correction |

### Output Parquet Schemas

**[`docs/output_files.md`](output_files.md) is the authoritative column-by-column reference** for every file PRISM writes, including the intermediates. It is not repeated here — a second copy is a second thing to go stale, and these tables had already drifted from what the pipeline writes.

The shape of both matrices is the same: a few **metadata** columns, then one **sample** column per replicate named `<replicate>__@__<source document>`, holding LINEAR abundances.

| File | Metadata columns |
| ---- | ---------------- |
| `corrected_peptides.parquet` | the peptide column (**named as the Skyline export named it**), `n_transitions`, `mean_rt`, and — C# engine only — `protein_group`, `leading_protein`, `leading_name`, `leading_gene_name` |
| `corrected_proteins.parquet` | `protein_group`, `leading_protein`, `leading_name`, `leading_uniprot_id`, `leading_gene_name`, `leading_description`, `n_peptides`, `n_unique_peptides`, `low_confidence` |

**Leading metadata semantics:** the `leading_` prefix means these fields describe the canonical representative of the parsimony group, not all member proteins. Values come from the Skyline export columns (`Protein Accession`, `Protein Gene`, `Protein`).

> [!IMPORTANT]
> Identify sample columns **by type**, not by subtracting a list of expected metadata names. The peptide column's name is whatever the export used, so no such list can be complete. Every text column is metadata; the only numeric metadata are `n_transitions` and `mean_rt` (peptides) and `n_peptides` / `n_unique_peptides` (proteins).

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

2. **Estimate offsets:**

   $$\hat\alpha_i = \mathrm{median_j}(\log_2(I_{ij}))$$

3. **Estimate sample effects:**

   $$\hat\beta_j = \mathrm{median_i}(\log_2(I_{ij}) - \hat\alpha_i)$$

4. **Calculate residuals:**

   $$\hat\epsilon_{ij} = \log_2(I_{ij}) - \hat\alpha_i - \hat\beta_j$$

5. **Compute weights:**

   $$w_i = \frac{1}{\mathrm{Var}(\hat\epsilon_{i,\cdot}) + \lambda}$$

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
   - Sweep row medians: $\alpha_i \leftarrow \alpha_i + \mathrm{median_j}(R_{i,\cdot});\; R \leftarrow R - \mathrm{median_j}(R_{i,\cdot})$
   - Sweep column medians: $\beta_j \leftarrow \beta_j + \mathrm{median_i}(R_{\cdot,j});\; R \leftarrow R - \mathrm{median_i}(R_{\cdot,j})$
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
2. Optimize weight function:

   $$\log(w_t) = \beta_{mz} \cdot \mathrm{norm_{mz}} + \beta_{outlier} \cdot f_{outlier}$$
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

#### Fitting Methods

Two fitting methods are available, controlled by the `fitting_method` configuration option:

##### Library Median Polish (Default, Recommended)

Uses the spectral library as a **prior** for transition row effects (fragmentation differences) and estimates sample scale factors using the **median**, which is inherently robust to interference outliers.

**Model:**
$$\log(O_{t,s}) = \log(L_t) + \beta_s + \epsilon_{t,s}$$

Where:
- $O_{t,s}$ = observed intensity for transition $t$ in sample $s$
- $L_t$ = library intensity for transition $t$ (provides the row effect)
- $\beta_s$ = sample-specific scale factor (log-scale)
- $\epsilon_{t,s}$ = residual (noise + interference)

**Algorithm:**

1. **Estimate scale via median:** For each sample $s$:

   $$\hat\beta_s = \mathrm{median_t}\left(\log(O_{t,s}) - \log(L_t)\right)$$
   
   The median automatically ignores up to 50% outliers, making this robust to 1-2 interfered transitions out of 4-6 total.

2. **Compute predicted values:** $\hat O_{t,s} = L_t \times e^{\hat\beta_s}$

3. **Identify outliers:** Compute normalized residuals:
   $$r_{t,s} = \frac{O_{t,s} - \hat O_{t,s}}{\hat O_{t,s}}$$
   
   Only HIGH positive residuals indicate interference (signal > expected). Interference can only **add** signal, never remove it.

4. **Iterative removal:** Remove the worst outlier (highest $r_{t,s}$) if it exceeds threshold (default 1.0 = observed > 2x predicted). Repeat until convergence.

5. **Calculate final abundance:**
   $$I_{peptide,s} = e^{\hat\beta_s} \times \sum_{t} L_t$$
   
   Uses the **full library sum** so all samples quantify the same total signal.

**Key advantages of median polish:**
- **Inherent robustness:** Median ignores outliers without needing to detect them first
- **Cross-sample consistency:** Same transitions tend to be interfered across samples
- **Faster convergence:** Often converges in 1-2 iterations vs. 3-5 for least squares

##### Least Squares Fitting

Classic least squares fitting, more sensitive to outliers but may perform better on very clean data.

**Closed-form solution:**
$$s^* = \frac{\vec{L} \cdot \vec{O}}{\vec{L} \cdot \vec{L}}$$

Where $\vec{L}$ is the library intensity vector and $\vec{O}$ is the observed intensity vector.

#### Handling Multiple Charge States

Peptides are often detected in multiple precursor charge states (e.g., +2 and +3). Each charge state produces a **different fragmentation pattern**, so the library stores separate spectra for each peptide+charge combination.

**Processing workflow:**

1. **Identify charge states:** For each peptide, identify all unique precursor charge states present in the data.

2. **Process independently:** Each charge state is processed separately using its own library spectrum:
   - +2 ions are matched to library spectrum `PEPTIDEK_2`
   - +3 ions are matched to library spectrum `PEPTIDEK_3`
   - Each gets its own library-assisted rollup to estimate abundance

3. **Sum on linear scale:** The final peptide abundance is the sum of abundances from all charge states:
   $$I_{peptide,s} = \sum_{z} I_{peptide,z,s}$$
   
   Where $z$ indexes the charge states.

**Example:**

For peptide `SAMPLE(unimod:21)PK` detected in +2 and +3 charge states:

| Charge | Library Key | Estimated Abundance |
|--------|-------------|---------------------|
| +2 | `SAMPLE(unimod:21)PK_2` | 1000 |
| +3 | `SAMPLE(unimod:21)PK_3` | 500 |
| **Total** | — | **1500** |

**Rationale:**

- Each precursor charge state ionizes independently in the electrospray source
- Fragmentation patterns differ between charge states (different y/b ion intensity ratios)
- The library provides expected fragment ratios **per charge state**
- Total peptide signal is the sum of all ionized precursor forms

**Implementation:** See `chunked_processing.py` -> `_process_single_peptide()`, lines 363-436.

**Key Design Principles:**

| Principle | Rationale |
|-----------|-----------|
| **Zeros are valid** | A low-abundance peptide may only have signal in the top 1-2 most intense library fragments. Zeros in minor fragments confirm absence of interference. |
| **Only HIGH residuals are outliers** | Interference adds signal (observed > expected). Low/zero signal is not interference, it's low abundance or noise. |
| **Iterative outlier removal** | Fragments with large positive residuals are removed and the model is refit to get a cleaner scale estimate. |
| **Abundance from scaled library** | Final abundance is the scale factor times the library sum, ensuring consistent quantification across samples. |
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
4. **Calculate corrections:**

   $$C_{s,rt} = G_{rt} - S_{s,rt}$$

   Where $G$ is the global (median) curve and $S$ is the per-sample fitted curve.

5. **Apply corrections:**

   $$\log_2(I_{p,s}') = \log_2(I_{p,s}) + C_{s,RT_p}$$

**Parameters:**
- `frac`: Fraction of data used for local regression (default: 0.3)
- `n_grid_points`: Number of RT grid points (default: 100)

### Median Normalization

Simple global shift to align sample medians:

$$\log_2(I_{p,s}') = \log_2(I_{p,s}) - \mathrm{median_p}(\log_2(I_{p,s})) + M_{global}$$

Where $M_{global}$ is the global median across all samples.

### Variance Stabilizing Normalization (VSN)

Applies arcsinh transformation to stabilize variance across intensity ranges:

$$I_{p,s}' = \mathrm{arcsinh}(a \cdot I_{p,s})$$

Where $a$ is optimized to minimize heteroscedasticity.

### Quantile Normalization

Forces all samples to have identical intensity distributions by ranking and replacing with average quantiles.

---

## Batch Correction

Removes systematic differences between experimental batches while preserving biological variation.

### Where do batch labels come from?

ComBat needs to know which batch each sample belongs to. PRISM resolves this automatically by trying three sources in a fixed priority order, and the source it ended up using is printed in the console log (`Batch source: ...`) and shown in `qc_report.html`.

| Priority | Source | When it is used |
|---|---|---|
| 1 | `Source Document` column inside the merged transition parquet (one input file = one batch) | Only when there are **two or more** distinct source documents. Single-input-file runs skip this. |
| 2 | The `batch` column of your sample metadata CSV | Whenever Priority 1 produces nothing and a metadata file is provided. |
| 3 | Estimation from acquisition-time gaps | Fallback when the first two produce nothing. Controlled by the `batch_estimation:` block of the YAML. |

**Using the metadata `Batch` column.** This is the path most multi-batch single-file runs take. The metadata column is auto-detected by name. Any of `batch`, `Batch`, or `Batch Name` works (Skyline's default header is `Batch Name`); they are all renamed internally to `batch`. No YAML change is needed. When the metadata column is recognized, the console logs `Renamed 'Batch' column to 'batch'` during loading, and Stage 2c later logs `Batch source: metadata`.

**Verifying after a run.** Look for these three log lines:

```text
Renamed 'Batch' column to 'batch'     # metadata column was found
Batch source: metadata                # Priority 2 (your CSV) is what's driving ComBat
Batches found: [batch_a, batch_b, ...]  # the unique batch labels found
```

If `Batch source:` says `source documents` (Priority 1) or `acquisition time estimation` (Priority 3), then your metadata `Batch` column is not what is being used.

**Edge cases.**

- Samples missing a batch in metadata are reported with a warning (`N samples without batch assignment`) but the rest of the cohort is corrected normally.
- If every sample ends up in the same batch (`n_batches < 2`), batch correction is skipped automatically with a benign log line.
- For multi-input-file runs where you want metadata to override the Source Document split, no YAML option exists today; the workaround is to merge upstream into a single parquet so Priority 1 does not fire. Tell us if you would like a config flag for this.

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

By default PRISM applies **standard grand-mean ComBat**: $\gamma_{ig}$ and $\delta_{ig}$ are estimated from all samples in batch $i$, aligning every batch to the across-batch grand mean. This assumes each batch has a comparable biological composition.

#### One implementation, two ways of pointing it

Standard and reference-anchored ComBat are the **same estimator**. Both build a *plan* and hand it to one core, and the plan is the whole of the difference between them:

| The plan says | Standard | Reference-anchored |
|---|---|---|
| Which samples a batch's effect is estimated **from** (the *fit set*) | every sample in the batch | the batch's reference replicates |
| Which samples it is applied **to** | every sample in the batch | every sample in the batch |
| Whether a batch gets a scale correction at all | always | only where the fit set has ≥ 2 replicates |
| What the pooled per-feature scale is | residual variance about the batch means | pooled *within-batch* variance of the replicates |

Everything else — the empirical-Bayes shrinkage, the missing-value handling, the refusal to invent quantities the data does not determine — is literally shared code. A fix to one is a fix to both.

#### Missing values

Every reduction ignores NaN, as `sva::ComBat` does. It is done by compacting a feature's observed values and running the ordinary reductions on them, so a **dense** cohort produces bit-identical results to an implementation that could not handle NaN at all. This matters because Skyline integrates imputed peak boundaries for every replicate, so PRISM's input is normally dense and NaN is the exception (see "Data Density" in `CLAUDE.md`).

> Until 26.9.0 a single missing value turned **every** corrected value in the cohort into NaN, because the empirical-Bayes priors are means taken *across* features and one NaN reached them.

#### What is not invented

A quantity the data does not determine is never given a placeholder value:

- **A feature is held out entirely** — returned unchanged — when it has no variance at all, or when some fitted batch never observed it. That batch's effect on it is *undefined*, which is not the same as zero.
- **A (batch, feature) scale is skipped** when the fit set has fewer than 2 observations or no resolvable spread. Its *location* effect is still estimable, so that correction is applied; only the rescaling is dropped. Crucially such a scale is also **excluded from that batch's prior** — feeding a placeholder `1.0` into a mean taken across features lets one unestimable feature perturb the shrinkage of every other feature in the batch.

Both are counted and reported in the run log rather than happening silently. "No resolvable spread" means a standard deviation below $10^{-12}$ of the values' own magnitude. An exact `variance == 0` test is knife-edge: the same 82 replicates once produced exactly `0.0` in Python and `7.99e-31` in C#, which flipped the feature between "no scale to estimate" and "a scale of 8e-31" — and left the two engines' protein abundances 3% apart.

#### Relationship to R's `sva::ComBat`

PRISM's ComBat is validated against `sva::ComBat` 3.58.0 by golden fixtures (`dotnet/tests/fixtures/sva/`), which hold **both** engines to an external reference rather than only to each other. **On dense input — the normal case — PRISM reproduces sva to floating-point noise.** Three differences remain, all deliberate:

- **`var_pooled`'s denominator.** sva uses $\sum r^2/n$ when the input is dense but `rowVars(na.rm = TRUE)` when it is not. PRISM uses the former throughout, so it matches sva on dense input and differs by ~0.3% on input with missing values. Following sva's switch would mean one peptide missing from one document shifts every corrected value in the cohort by ~0.3%.
- **Features constant within a batch** keep the location correction the data supports; sva drops them entirely.
- **Two inputs sva errors on outright**, PRISM handles: a feature observed once in a batch, and a feature absent from a batch.

#### Memory

Stage 2b/2c never holds the feature × sample matrix in memory. Normalization factors come from a column-at-a-time pass, the empirical-Bayes step is driven from two summary numbers per (batch, feature) instead of the standardized matrix, and both outputs are written one row group at a time. Peak memory is bounded by the input's row-group size rather than by the number of samples — measured at 102 MB against 798 MB on a 20,000-peptide × 600-sample cohort. This applies to standard **and** reference-anchored correction; only `quantile` normalization and a CSV/TSV `output.format` fall back to the in-memory implementation, and the log says which one ran.

### Reference-anchored ComBat (single-point calibration with empirical-Bayes shrinkage)

Enabled with `batch_correction.reference_anchored: true`, or in the Skyline tool by ticking **"Anchor ComBat on the Standard samples"** on the Settings tab. This is PRISM's recommended correction when inter-experiment **reference samples** (identical material run in every batch) are available. It combines single-point external reference calibration (Pino et al., 2020) with ComBat's empirical-Bayes shrinkage.

**Which samples are the references.** The samples whose PRISM sample type equals `batch_correction.reference_type` (default `reference`), which is what Skyline's **`Standard`** Sample Type maps to. So in practice: mark your inter-batch reference injections as `Standard` in Skyline and PRISM will anchor on them, with no metadata editing needed.

**Concept.** Rather than estimating the batch effect from all samples (which conflates technical and biological variation when batches differ biologically), PRISM estimates each batch's technical effect from the **reference samples only**. Because the reference is identical material, any per-batch difference in it is purely technical, so the correction removes no biology.

**Model, per feature $g$, batch $i$ (log2 abundance $Y$):**
- $\alpha_g$ = pooled mean of $Y_g$ over all reference samples across all batches (the reference material's level)
- $\gamma_{ig}$ = additive offset = (mean of $Y_g$ over references in batch $i$) $- \alpha_g$
- $\delta_{ig}$ = multiplicative scale = dispersion of the standardized reference replicates in batch $i$, estimated only when batch $i$ has $\geq 2$ reference replicates (otherwise $\delta_{ig}=1$, location-only for that batch)

**Empirical-Bayes shrinkage across features within a batch** stabilizes the per-analyte calibration: an analyte poorly measured in the reference borrows strength from the batch-wide consensus shift rather than being dictated by one noisy reference measurement. The additive shrinkage weight is $\tau^2 n / (\tau^2 n + 1)$ where $n$ is the number of reference replicates in the batch, so a single reference per batch produces heavy shrinkage and additional replicates progressively trust the directly measured per-analyte offset (converging to raw single-point calibration).

**Standardization scale.** The standardization variance is the pooled within-batch reference-replicate variance (the technical variance), mirroring standard ComBat's use of the within-batch pooled variance. This makes $\delta$ average $\approx 1$ across batches so dividing by $\sqrt{\delta}$ harmonizes technical dispersion rather than rescaling biology.

**Output.** Calibrated absolute log2 abundance on the input scale, applied to all samples (experimental, QC, and reference). The result is **not** a ratio to the reference; it is each sample's own abundance with the reference-derived technical offset (and, where estimable, dispersion) removed.

**Batches with few or no references.** The fit set decides what can be estimated, and the shortfall is handled per batch rather than by failing the run:

| References in the batch | What happens |
|---|---|
| ≥ 2 | Full location + scale correction, empirical-Bayes shrunk. |
| exactly 1 | **Location only.** One reference is a level, not a spread, so there is nothing to estimate a scale from. |
| 0, `no_reference_batch="fallback"` (default) | **Location only, from the batch's own average** — i.e. the grand-mean assumption, taken deliberately and only for that batch, and logged. A scale is *not* estimated here: that fit set's spread is biological, and rescaling by it would shrink real signal. |
| 0, `no_reference_batch="skip"` | Left exactly as it came in. It also gets no vote in the hold-out screening — a batch that is not being fitted cannot veto a feature. |
| 0, `no_reference_batch="error"` | The run stops. |

A feature that a batch's *references* never observed is held out and returned unchanged, for the same reason as in standard ComBat: the offset is unknown, not zero.

**Implementation:** shares one estimator with standard ComBat — `ComBatCore` +
`ComBatPlan.ReferenceAnchored` (`dotnet/src/SkylinePrism.Core/BatchCorrection/`).
Because there is no third-party implementation of this method to check against, the fixtures in
`dotnet/tests/fixtures/refanchored/` hold it to the output the original Python engine produced — they
agreed to $10^{-10}$, including on sparse input and both no-reference policies, and the goldens remain
the regression baseline.

---

## Protein Parsimony

Assigns peptides to protein groups using a minimal set cover approach. The
algorithm is aligned with the Osprey reference design documented in
`osprey/docs/16-protein-parsimony.md`, so PRISM and Osprey produce
deterministic, byte-identical razor assignments on the same peptide-protein
graph.

### Algorithm

1. **Build peptide-protein mappings** from Skyline export or FASTA database. A
   peptide can map to multiple proteins (paralogs, isoforms, homologs).

2. **Identify subsumable proteins.** A protein whose peptide set is a *strict*
   subset of another protein's peptide set provides no additional evidence and
   is removed. When a subsumed protein has multiple valid supersets, the
   lexicographically smallest accession is recorded as its subsumer so the
   `subsumed_proteins` list reported per group is deterministic.

3. **Identify indistinguishable proteins.** Proteins with *identical* peptide
   sets are collapsed into a single group with multiple member accessions. The
   alphabetically-first accession is selected as the canonical (leading)
   protein for the group.

4. **Classify peptides** as **unique** (map to exactly one canonical protein)
   or **shared** (map to two or more).

5. **Razor: iterative greedy set cover.** Shared peptides are assigned to a
   single group each:

   ```text
   while any shared peptides remain unassigned:
     pick the group G with the MOST unique peptides that still has at
       least one unassigned shared peptide (tiebreak: lowest group ID,
       i.e. lexicographically smallest canonical accession)
     claim ALL of G's unassigned shared peptides in one batch
   ```

   The iterative form looks at the **global state** at every step, so when a
   group claims peptides in one round, the updated unique counts are used to
   pick the winner of the next round. A naive single-pass approach can
   produce different (path-dependent) assignments.

   **Tiebreaker.** When two candidate groups have the same unique-peptide
   count, the group with the lexicographically smallest canonical accession
   wins. This matches Osprey's `max_by_key((unique_count, Reverse(group_id)))`.
   Earlier versions of PRISM used a coarser tiebreaker that only inspected the
   first character of the accession and then preferred the group with more
   shared peptides remaining, which produced different (and non-deterministic)
   assignments on tie cases.

   **Determinism.** Shared peptides are collected in sorted order at the start
   of the loop; canonical proteins are iterated in sorted order; claimed
   peptides per round are sorted alphabetically before being added to the
   razor set. Repeated runs on the same input produce identical razor
   assignments regardless of hash-table iteration order.

### Protein Groups

Each protein group contains:
- **Leading protein:** Canonical (alphabetically-first) accession of the group.
- **Leading metadata:** Group-level identifiers and names are taken from the
  leading protein and exposed as `leading_uniprot_id`, `leading_gene_name`,
  and `leading_description` in outputs.
- **Member proteins:** Indistinguishable proteins with identical peptide sets.
- **Subsumed proteins:** Proteins whose peptide sets are strict subsets of
  this group's; recorded against the lexicographically smallest valid superset.
- **Unique peptides:** Map exclusively to this group's canonical protein.
- **Razor peptides:** Shared peptides assigned to this group by the iterative
  greedy algorithm.

### Shared Peptide Handling Options

PRISM splits the three shared-peptide modes between two layers: parsimony
always produces the unique + razor split, and the rollup stage decides which
peptides to use for protein quantification.

| Option | Description |
|--------|-------------|
| `all_groups` | Include shared peptides in ALL groups they map to (full intensity, no splitting). Maximum sensitivity. |
| `razor` | Use only each group's unique peptides plus its razor-assigned shared peptides. Matches MaxQuant's razor logic. |
| `unique_only` | Use only unique peptides for quantification. Most conservative. |

The `razor` mode is where the iterative-greedy algorithm above directly
controls quantification; `all_groups` and `unique_only` are alternative views
on the same parsimony output.

### Example: cascading razor

```text
Group P1 peptides: {A, B, C, X, Y}    ← 3 unique (A, B, C), 2 shared (X, Y)
Group P2 peptides: {D, X, Z}          ← 1 unique (D),       2 shared (X, Z)
Group P3 peptides: {E, Y, Z}          ← 1 unique (E),       2 shared (Y, Z)

Round 1: P1 has 3 unique (most) → claims X and Y in one batch
         Remaining shared: {Z}
Round 2: P2 and P3 tie at 1 unique → P2 wins on lower group ID → claims Z
         Done.
```

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
   $$r_{ij} = \mathrm{median_p}(\log_2(I_{p,i}) - \log_2(I_{p,j}))$$

2. Solve for protein abundances that best explain these ratios:

   $$\hat\beta_s = \mathrm{mean_j}(r_{sj})$$

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

## Marker-Protein Normalization

Enabled with `marker_normalization.enabled: true`, or in the Skyline tool by ticking **"Normalize to a
protein list"** on the Settings tab. Off by default: it changes every reported abundance.

### The question it answers

A capture-based experiment measures whatever the capture caught. If the amount of the thing you care
about varies between samples — extracellular vesicles pulled from plasma, a hand-dissected glomerulus,
a punch of tissue — then every protein's share of the signal moves with it. A loading normalization
cannot separate that from biology, because it makes total signal equal *by construction*: after median
or RT-LOWESS normalization, a sample with half the EVs and a sample with twice the EVs look the same
size, and the difference has been pushed into the composition of every protein.

Residualizing on a marker score turns **"what changed in what was captured"** into **"what changed per
unit of the marked material"**. Those are different questions, and only the second one is usually the
one being asked.

This is not a replacement for the ordinary normalization; it runs after it (see *Where it runs*).

### Algorithm

Given a marker set $M$ (rows of the protein matrix matched to a protein list) on the LOG2 scale, with
$m = |M|$ markers and $n$ samples:

**1. z-score each marker across samples**, using the sample standard deviation ($\text{ddof}=1$):

$$z_{ij} = \frac{y_{ij} - \bar{y}_i}{s_i}, \qquad s_i = \sqrt{\frac{1}{n-1}\sum_j (y_{ij} - \bar{y}_i)^2}$$

Without this a high-abundance marker would dominate the axis purely by scale. A marker with $s_i = 0$
is left at zero rather than dividing by zero: it carries no information about the axis, but stays in
the block.

**2. Take PC1 by SVD** of the $m \times n$ z-scored block $Z = U S V^{\mathsf{T}}$:

$$\text{score}_j = V^{\mathsf{T}}_{0j} \cdot S_0, \qquad \text{loading}_i = U_{i0},
\qquad \text{variance explained} = \frac{S_0^2}{\sum_k S_k^2}$$

This is the decomposition `numpy.linalg.svd(Z, full_matrices=False)` gives. A dense SVD is appropriate
here and nowhere else in PRISM: the marker block is a couple of dozen rows, not the tens of thousands
of features `Pca` must avoid materializing.

**3. Orient the sign.** The sign of a principal component is arbitrary. The score and loadings are
flipped, if needed, so the score correlates **positively** with the mean z-scored marker profile.
Without this, "higher score" would mean more marked material or less, at random, from run to run.

**4. Residualize every feature** on the score. Per feature, an ordinary least-squares fit of its LOG2
profile on $[1, \text{score}]$, keeping the residual **with the intercept added back**:

$$y'_{gj} = y_{gj} - \hat{\beta}_g \cdot \text{score}_j, \qquad
\hat{\beta}_g = \frac{n\sum x y - \sum x \sum y}{n \sum x^2 - (\sum x)^2}$$

Only $\hat{\beta}_g \cdot \text{score}_j$ is removed. The plain residual $y - \hat{\alpha} - \hat{\beta}x$
would put every feature at zero in log space — an abundance of 1 — which is not a quantity anything
downstream can use. Keeping the intercept means protein X stays 100× more abundant than protein Y
while acquiring the same fold change between conditions that the residuals have.

Sums run over the samples a feature was actually observed in. Fewer than 3 observations leaves the
feature untouched: a two-point fit through a two-parameter model has no residual to speak of, and
zeroing it would fabricate a result. A score that is constant across a feature's samples is likewise
skipped.

**Alternative score.** `method: mean` replaces step 2 with the plain mean of the z-scored markers.
Offered for comparison; see *Why PC1* below.

### Where it runs, and why there

Stage **5a** — after both arms are finished, after normalization *and* batch correction, and the score
is computed at the **protein** level then applied to **both** the peptide and the protein output.

- **After the loading normalization, never instead of it.** The score has to come from data whose
  per-sample loading is already removed. Computed on raw abundances, PC1 loads on injection volume,
  and residualizing then quietly re-does the loading step using a couple of dozen proteins' worth of
  noise instead of the whole matrix.
- **One score, from the protein level.** How much marked material a sample contributed is a property
  of *the sample*, not of the table being analyzed. Re-estimating it from the peptide matrix would
  mostly re-measure the same quantity with more noise, and would let the two outputs disagree about a
  fact that has one answer.
- `peptides_log2_internal.parquet` is deliberately **not** adjusted. It is what the protein rollup
  consumed, produced before the score existed; adjusting it would describe a rollup that never happened.

### Why PC1 rather than the mean of the markers

Markers do not have to move as one block. On the cohort the shipped `EV markers` panel comes from, PC1
explains **70.4%** of marker variance and **4 of the 18** markers (`SDCBP`, `ANXA2`, `ANXA6`, `CD81`)
load with the *opposite* sign to the other 14. A mean therefore partially cancels and blunts the
estimate. PC1 weights each marker by its contribution and handles the sign structure, and it transfers
to a panel whose dominant axis is driven by different members.

On that cohort the two scores correlate at $r = +0.951$, with PC1 giving the more conservative answer.

### Diagnostics and failure modes

`marker_normalization.csv` records the per-sample score and the per-marker loadings, so a run can be
judged rather than trusted. What to look at:

| Signal | Meaning |
|---|---|
| **Fewer than 3 quantified markers** | Hard error, not a silent fallback. A score cannot be defined. |
| **PC1 variance explained < 40%** | Warning. The markers are not moving together, so the score is a weak summary of them. |
| **Score correlates with total ion current / loaded material** | Expected to a degree, and ambiguous: compatible with a residual technical effect *or* with the capture biology. Worth reporting, not worth panicking about. |
| **Score separates cases from controls** | The danger sign. That is pathology, not capture, and residualizing on it removes the finding. See below. |

**The markers themselves stay in the outputs**, flagged `normalization_marker`. Their residual is near
zero by construction, so **exclude them from any result read off these files** — a test among them is
circular.

**Choosing a panel is the whole game.** The markers must be proportional to the amount of material
captured and *not* to the phenotype. This is why the shipped `Glomerulus` panel is weighted toward
structural proteins and deliberately excludes `NPHS1`/`NPHS2` (podocyte loss *is* the phenotype in most
glomerular disease) and `COL4A1`/`COL4A2` (ubiquitous basement membrane, so the score would track any
BM rather than the GBM). It is also why `Tubular contamination` ships as a **readout** for the Dynamic
Range plot and never as a normalizer: its abundance *is* the contamination, so normalizing to it would
remove the thing being measured.

### Relationship to published methods

The mechanism — factor-analyze a *designated* subset of features, then regress the leading factor out
of everything — is well precedented. The **intent** is where PRISM's use differs from the closest
published family, and that difference is the thing to keep in mind when reading those papers.

**Closest formal precedent: RUV (Remove Unwanted Variation).** RUV-2 restricts a factor analysis to
negative control genes and removes the resulting factors from the whole matrix (Gagnon-Bartsch & Speed,
2012); RUVSeq/RUVg is the widely used implementation (Risso et al., 2014), with an unsupervised variant
for when the factor of interest is unobserved (Jacob et al., 2016) and RUV-III adding technical
replicates to the control set (Molania et al., 2019). RUV has been benchmarked favorably on large
clinical MS proteomics cohorts (Dubois et al., 2022).

> **The key difference.** RUV's control genes are chosen because they are *unaffected by the biology* —
> spike-ins, negative controls, housekeepers — so the factor they estimate is meant to be purely
> technical. PRISM's markers are the opposite: chosen **because** they are biological, and specifically
> because their abundance is proportional to how much of the material of interest was captured. The
> axis removed here is therefore a biological-but-nuisance quantity (input amount), which puts the
> intent closer to tumor-purity or cell-composition adjustment than to RUV proper.
>
> RUV's own safeguard states PRISM's main risk exactly: control features must be *a priori* not
> differentially expressed with respect to the factor of interest. Any marker that also tracks the
> phenotype gets the finding regressed out along with the capture.

**PC1 of a feature set as a summary quantity.** "Eigengenes" — SVD components of an expression matrix,
with the ones judged to be artifact filtered out — go back to Alter, Brown & Botstein (2000). WGCNA's
*module eigengene* is literally PC1 of a gene set (Langfelder & Horvath, 2007, 2008), though it is used
as a summary and a covariate rather than to residualize the matrix.

**Unsupervised cousins** estimate the factor from *all* features rather than a chosen set: SVA (Leek &
Storey, 2007; Leek et al., 2012) and, in our own field, **EigenMS** (Karpievitch et al., 2009), which
adapts SVA to LC-MS proteomics with SVD on the residuals. scLVM (Buettner et al., 2015) infers a latent
factor from a known cell-cycle gene set and regresses it out — a latent-variable model rather than PCA,
but the same shape of idea, and the closest published match to "known marker set → one factor → remove".

**Two things PRISM does differently from RUV**, worth knowing if the single-factor version turns out
to be too blunt:

- RUV treats the number of removed factors $k$ as a tuning parameter with its own diagnostics. PRISM
  removes exactly one.
- RUV-III uses technical replicates as part of the control set. PRISM already tracks reference and QC
  injections and could do the same, but does not today.

**What was not found.** A literature search on **2026-08-29** (PubMed, via the `search_articles` /
`get_article_metadata` tools) did not turn up a paper doing precisely this: PC1 of a *curated EV or
glomerular marker panel*, OLS-residualized out of an LC-MS proteomics matrix. Searches covering EV
marker normalization, glomerular/LCM proteomics normalization, and podocyte-marker normalization
returned nothing of that shape. So the machinery is well precedented and this particular application of
it may not be — which is either a gap in the search or a small novelty. Treat it as "not found", not as
"does not exist", and re-check before claiming novelty in print.

### Implementation

| Piece | Location |
|---|---|
| Score and residualization | `dotnet/src/SkylinePrism.Core/Normalization/MarkerNormalization.cs` |
| Stage 5a driver (matching, flagging, CSV) | `dotnet/src/SkylinePrism.Core/Pipeline/MarkerNormalizeStage.cs` |
| Marker panels and matching | `dotnet/src/SkylinePrism.Core/Qc/ProteinListSet.cs` |
| Config keys | [`parameters.md`](parameters.md#marker_normalization--normalize-to-a-set-of-proteins-stage-5a) |

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

### Marker-protein normalization: prior art

Retrieved from PubMed on 2026-08-29. See
[Relationship to published methods](#relationship-to-published-methods) for what each one contributes.

5. Gagnon-Bartsch JA, Speed TP (2012). Using control genes to correct for unwanted variation in microarray data. *Biostatistics* 13(3):539-552. [doi:10.1093/biostatistics/kxr034](https://doi.org/10.1093/biostatistics/kxr034)

6. Risso D, Ngai J, Speed TP, Dudoit S (2014). Normalization of RNA-seq data using factor analysis of control genes or samples. *Nature Biotechnology* 32(9):896-902. [doi:10.1038/nbt.2931](https://doi.org/10.1038/nbt.2931)

7. Jacob L, Gagnon-Bartsch JA, Speed TP (2016). Correcting gene expression data when neither the unwanted variation nor the factor of interest are observed. *Biostatistics* 17(1):16-28. [doi:10.1093/biostatistics/kxv026](https://doi.org/10.1093/biostatistics/kxv026)

8. Molania R, Gagnon-Bartsch JA, Dobrovic A, Speed TP (2019). A new normalization for Nanostring nCounter gene expression data. *Nucleic Acids Research* 47(12):6073-6083. [doi:10.1093/nar/gkz433](https://doi.org/10.1093/nar/gkz433)

9. Alter O, Brown PO, Botstein D (2000). Singular value decomposition for genome-wide expression data processing and modeling. *PNAS* 97(18):10101-10106. [doi:10.1073/pnas.97.18.10101](https://doi.org/10.1073/pnas.97.18.10101)

10. Langfelder P, Horvath S (2007). Eigengene networks for studying the relationships between co-expression modules. *BMC Systems Biology* 1:54. [doi:10.1186/1752-0509-1-54](https://doi.org/10.1186/1752-0509-1-54)

11. Langfelder P, Horvath S (2008). WGCNA: an R package for weighted correlation network analysis. *BMC Bioinformatics* 9:559. [doi:10.1186/1471-2105-9-559](https://doi.org/10.1186/1471-2105-9-559)

12. Leek JT, Storey JD (2007). Capturing heterogeneity in gene expression studies by surrogate variable analysis. *PLoS Genetics* 3(9):1724-1735. [doi:10.1371/journal.pgen.0030161](https://doi.org/10.1371/journal.pgen.0030161)

13. Leek JT, Johnson WE, Parker HS, Jaffe AE, Storey JD (2012). The sva package for removing batch effects and other unwanted variation in high-throughput experiments. *Bioinformatics* 28(6):882-883. [doi:10.1093/bioinformatics/bts034](https://doi.org/10.1093/bioinformatics/bts034)

14. Karpievitch YV, Taverner T, Adkins JN, Callister SJ, Anderson GA, Smith RD, Dabney AR (2009). Normalization of peak intensities in bottom-up MS-based proteomics using singular value decomposition. *Bioinformatics* 25(19):2573-2580. [doi:10.1093/bioinformatics/btp426](https://doi.org/10.1093/bioinformatics/btp426)

15. Buettner F, Natarajan KN, Casale FP, Proserpio V, Scialdone A, Theis FJ, Teichmann SA, Marioni JC, Stegle O (2015). Computational analysis of cell-to-cell heterogeneity in single-cell RNA-sequencing data reveals hidden subpopulations of cells. *Nature Biotechnology* 33(2):155-160. [doi:10.1038/nbt.3102](https://doi.org/10.1038/nbt.3102)

16. Dubois E, Núñez Galindo A, Dayon L, Cominetti O (2022). Assessing normalization methods in mass spectrometry-based proteome profiling of clinical samples. *Biosystems* 215-216:104661. [doi:10.1016/j.biosystems.2022.104661](https://doi.org/10.1016/j.biosystems.2022.104661)
