# AGENTS.md - AI Agent Guidelines for Skyline-PRISM

This document provides context and guidelines for AI agents working on the Skyline-PRISM project.

## Project Overview

**Skyline-PRISM** (Proteomics Reference-Integrated Signal Modeling) is a Python package for normalization of LC-MS proteomics data exported from [Skyline](https://skyline.ms), with robust protein quantification using Tukey median polish and reference-anchored batch correction.

### Key Concepts

- **Transition-level input required**: PRISM expects transition-level data from Skyline (not peptide or protein summaries)
- **Tukey median polish as default**: Both transition→peptide and peptide→protein rollups use median polish by default for robust outlier handling
- **Reference-anchored ComBat batch correction**: Uses inter-experiment reference samples for QC evaluation, with automatic fallback if correction degrades quality
- **Dual-control validation**: Uses intra-experiment QC samples to validate corrections without overfitting
- **Sample outlier detection**: Automatic detection of samples with abnormally low signal (one-sided, on LINEAR scale). Can report or exclude outliers.
- **Two-arm pipeline**: Pipeline splits at peptide level - batch correction is applied at the reporting level (peptide or protein)
- **Optional RT correction**: RT-dependent correction is implemented but DISABLED by default (search engine RT calibration may not generalize between samples)

### Scale Conventions

| Stage | Scale | Notes |
|-------|-------|-------|

| **Input** | LINEAR | Raw peak areas from Skyline |
| **Internal** | LOG2 | All rollup/normalization operates on log2 scale |
| **Output** | LINEAR | Final peptide/protein output matrices (parquet/CSV) are always written in LINEAR scale (values are 2^x, not log2(x)) |

**Implementation Note:**
All sample columns in the peptide and protein output matrices are explicitly converted from log2 to linear ($2^x$) immediately before writing the output files. This ensures all downstream quantitative analyses use true abundance values, not log2-transformed values. This is enforced in `chunked_processing.py` for both peptide and protein outputs.

**Do not write log2 values to output files.**

The pipeline automatically handles transforms:
- Input linear values are log2-transformed for processing
- Output values are back-transformed to linear (2^x) before writing

When using functions directly via Python API, check docstrings for scale requirements.

### CV Calculation (CRITICAL)

**CVs must ALWAYS be calculated on LINEAR scale data, NEVER on log-transformed data.**

Correct calculation:
```python
linear_data = 2 ** log2_data  # Convert from log2 to linear
cv = (linear_data.std() / linear_data.mean()) * 100  # CV as percentage
```

Rationale: On log scale, variance is artificially compressed. A CV of 5% on log2 data would be meaningless - true biological CVs for proteomics control samples typically range from 10-30%.

### Processing Pipeline

The current implementation follows this stage structure:

```text
Stage 1: Merge CSVs (streaming, memory-efficient)
    ↓
Stage 2: Transition → Peptide rollup (Tukey median polish)
    ↓
Stage 2b: Peptide Global Normalization (median or VSN)
    ↓ [Optional: RT correction - disabled by default]
Stage 2c: Peptide ComBat Batch Correction
    ↓
    ├──────────────────────────────┐
    ↓                              ↓
Stage 3: Protein Parsimony    PEPTIDE OUTPUT
    ↓                         (corrected_peptides.parquet)
Stage 4: Peptide → Protein Rollup (median polish)
    ↓
Stage 4b: Protein Global Normalization (median)
    ↓
Stage 4c: Protein ComBat Batch Correction
    ↓
Stage 5: Output Generation
    ↓
    PROTEIN OUTPUT
    (corrected_proteins.parquet)
    ↓
Stage 5b: QC Report Generation (HTML + plots)
```

**Key implementation details:**

- **Streaming processing**: Stage 1 uses DuckDB-based streaming to handle ~47GB datasets
- **Batch correction applied twice**: Once at peptide level (Stage 2c), once at protein level (Stage 4c)
- **Independent outputs**: Both peptide and protein files are batch-corrected independently
- **Log files**: Automatically saved to output directory with timestamp (`prism_run_YYYYMMDD_HHMMSS.log`)
- **Metadata columns**: Uses `sample`, `sample_type`, `batch` (with automatic normalization from Skyline formats)

## Project Structure

```
skyline-prism/
├── skyline_prism/           # Main Python package
│   ├── __init__.py          # Package exports
│   ├── cli.py               # Command-line interface (entry point: `prism`)
│   ├── data_io.py           # Skyline report loading and merging
│   ├── normalization.py     # RT-aware correction pipeline
│   ├── batch_correction.py  # ComBat implementation (empirical Bayes)
│   ├── parsimony.py         # Protein grouping and shared peptide handling
│   ├── rollup.py            # Peptide → Protein rollup (median polish, etc.)
│   ├── transition_rollup.py # Transition → Peptide rollup (median polish, quality-weighted, variance learning)
│   ├── validation.py        # QC metrics and reporting (generates HTML QC reports with embedded plots)
│   └── visualization.py     # Plotting functions for QC assessment and normalization evaluation
├── tests/                   # Unit tests (pytest)
│   ├── test_data_io.py
│   ├── test_parsimony.py
│   ├── test_rollup.py
│   └── test_transition_rollup.py
├── SPECIFICATION.md         # Detailed technical specification
├── README.md                # User-facing documentation
├── config_template.yaml     # Configuration file template
├── pyproject.toml           # Package configuration and dependencies
└── .venv/                   # Virtual environment (not in git)
```

## Key Algorithms

### Tukey Median Polish (Default for Rollups)

Used for both transition→peptide and peptide→protein rollups. Decomposes a matrix into:
- Row effects (see table below)
- Column effects (sample abundance - **this is the output**)
- Residuals (noise/outliers - **preserved for biological analysis**)

| Rollup Stage | Row Effects Represent | Column Effects |
|--------------|----------------------|----------------|
| Transition → Peptide | Transition interference (co-eluting analytes) | Peptide abundance |
| Peptide → Protein | Peptide ionization efficiency | Protein abundance |

The median operation automatically downweights outliers without explicit filtering.

**Important**: Following Plubell et al. 2022 (doi:10.1021/acs.jproteome.1c00894), residuals are **preserved, not discarded**. Peptides/transitions with large residuals may indicate biologically interesting proteoform variation, PTMs, or protein processing.

**Implementation**: 
- `skyline_prism/rollup.py` → `tukey_median_polish()` returns `MedianPolishResult` with residuals
- `skyline_prism/rollup.py` → `extract_peptide_residuals()` for output to parquet
- `skyline_prism/rollup.py` → `extract_transition_residuals()` for transition-level residuals

### RT Correction (Spline-based)

Learns RT-dependent technical variation from reference samples only:
1. Calculate residuals: observed - reference mean
2. Fit smoothing spline to residuals vs RT
3. Apply correction to all samples

**Implementation**: `skyline_prism/normalization.py` → `rt_correction_from_reference()`

### ComBat Batch Correction

Full empirical Bayes implementation (Johnson et al. 2007):
- Estimates additive (location) and multiplicative (scale) batch effects
- Uses empirical Bayes shrinkage for robust estimation
- Supports reference batch, parametric/non-parametric priors, mean-only correction

**Implementation**: `skyline_prism/batch_correction.py` → `combat()`, `combat_from_long()`

### Adaptive Rollup (Learned Transition Weighting)

For transition→peptide aggregation, the adaptive method learns optimal weighting parameters:

**Weight Formula** (`AdaptiveRollupParams`):
```
w_t = exp(beta_intensity * (log2(I) - center) + beta_mz * mz_norm + beta_shape * shape_corr)
```

Where:
- `log2(I) - center`: Log2 intensity centered by the peptide's mean log2 intensity
- `mz_norm`: Product m/z normalized to [0, 1] range
- `shape_corr`: Shape correlation from Skyline (median across samples)

Key insight: When all betas = 0, weights = 1 for all transitions (equivalent to simple sum). This provides a principled baseline.

**Constraint**: `beta_log_intensity >= 0` (higher intensity should not decrease weight)

**Learning process**:
1. Parameters optimized on reference samples by minimizing median CV (L-BFGS-B optimizer)
2. Validated on QC samples to prevent overfitting
3. Automatic fallback to simple sum if adaptive doesn't improve CV by `min_improvement_pct`

**Implementation**:
- `skyline_prism/transition_rollup.py` → `learn_adaptive_weights()` - learns parameters
- `skyline_prism/transition_rollup.py` → `rollup_peptide_adaptive()` - applies weights
- `skyline_prism/transition_rollup.py` → `compute_adaptive_weights()` - computes weights from params

**Configuration:**
```yaml
transition_rollup:
  method: "adaptive"
  learn_variance_model: true  # Learn from reference samples
  adaptive_rollup:
    beta_log_intensity: 0.5  # Weight for log2 intensity (must be >= 0)
    beta_mz: 0.0             # Weight for normalized m/z
    beta_shape_corr: 1.0     # Weight for shape correlation
    min_improvement_pct: 5.0 # Required improvement over sum
```

## Development Guidelines

### Style Guidelines

- **No emojis**: Do not use emojis in code, documentation, comments, or output messages. Use plain text instead (e.g., "PASSED" instead of "✓", "WARNING" instead of "⚠️").
- **This is a strict requirement**: All status indicators, section headers, and documentation must use plain ASCII text. Use prefixes like "[WORKING]", "[ISSUE]", "[TODO]" instead of emoji symbols.
- Unicode arrows (→) for flow diagrams are acceptable.

### Virtual Environment

The project uses a Python virtual environment in `.venv/`:

```bash
cd /home/maccoss/GitHub-Repo/maccoss/skyline-prism
source .venv/bin/activate
```

### Running Tests

**Always run tests after making changes:**

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=skyline_prism --cov-report=term-missing

# Run a specific test file
pytest tests/test_parsimony.py -v

# Run a specific test
pytest tests/test_rollup.py::TestTukeyMedianPolish::test_simple_matrix -v
```

**Test expectations:**
- All tests must pass before committing
- New features should include corresponding tests
- Tests are in `tests/` directory using pytest
- Coverage is tracked via pytest-cov

### Code Style

The project uses:
- **black** for code formatting
- **ruff** for linting (with auto-fix)
- **mypy** for type checking

```bash
# Format code
black skyline_prism/

# Lint and auto-fix issues
ruff check skyline_prism/ --fix

# For more aggressive fixes (type annotation modernization, etc.)
ruff check skyline_prism/ --fix --unsafe-fixes

# Type check
mypy skyline_prism/
```

**Always run ruff with `--fix`** to automatically correct linting issues before committing.

### Documentation Updates

**Keep README.md updated:**
- When adding new features, update the README.md to document them
- When changing CLI commands, update the usage examples
- When adding new configuration options, document them in both README.md and config_template.yaml

**SPECIFICATION.md** contains the detailed technical specification. Reference it for algorithm details but avoid modifying it unless the fundamental approach changes.

## Key Files to Understand

### SPECIFICATION.md
The authoritative technical specification. Contains:
- Input/output formats (Skyline report columns)
- Algorithm details (RT correction, median polish, parsimony)
- Processing pipeline stages (two-arm design)
- Configuration parameters

### config_template.yaml
Comprehensive configuration file with all options documented. Can be generated via:
- `prism config-template -o config.yaml` (full template)
- `prism config-template --minimal -o config.yaml` (common options only)

Key sections:
- `transition_rollup`: Transition→peptide rollup (method: sum, median_polish, adaptive)
- `sample_outlier_detection`: Detect low-signal samples (method: iqr or fold_median, action: report or exclude)
- `rt_correction`: RT-aware normalization (method: spline) - DISABLED by default
- `batch_correction`: ComBat settings (method: combat)
- `protein_rollup`: Peptide→protein rollup (method: sum, median_polish, topn, maxlfq, ibaq)
- `parsimony`: Shared peptide handling (all_groups, unique_only, razor)
- `qc_report`: QC report generation (enabled, save_plots, embed_plots, plot selection)

### batch_correction.py
Full ComBat implementation with:
- `combat()`: Main function for wide-format data
- `combat_from_long()`: Wrapper for long-format data (PRISM pipeline format)
- `combat_with_reference_samples()`: Automatic evaluation using reference/QC CVs
- `evaluate_batch_correction()`: Compare before/after metrics

### visualization.py
QC visualization functions for normalization assessment:
- `plot_intensity_distribution()`: Box plots of sample intensity distributions
- `plot_pca()`, `plot_comparative_pca()`: PCA analysis for batch effects
- `plot_control_correlation_heatmap()`: Correlation heatmaps for control samples
- `plot_cv_distribution()`, `plot_comparative_cv()`: CV distributions for precision assessment
- `plot_rt_correction_comparison()`: Before/after comparison of RT correction showing reference (fitted) vs QC (held-out validation)
- `plot_rt_correction_per_sample()`: Per-sample RT correction quality assessment

### pyproject.toml
Package metadata and dependencies. Contains:
- Package name: `skyline-prism`
- CLI entry point: `prism` → `skyline_prism.cli:main`
- Dependencies (core, dev, viz)

## CLI Commands

The package provides a `prism` CLI. The primary command is `prism run`:

```bash
# Run the full PRISM pipeline (recommended)
prism run -i skyline_report.csv -o output_dir/ -c config.yaml -m metadata.tsv
```

This produces:
- `corrected_peptides.parquet` - Peptide-level batch-corrected quantities
- `corrected_proteins.parquet` - Protein-level batch-corrected quantities
- `protein_groups.tsv` - Protein group definitions
- `peptide_residuals.parquet` - Residuals for outlier analysis (if enabled)
- `metadata.json` - Complete processing parameters for reproducibility
- `qc_report.html` - HTML QC report with embedded diagnostic plots
- `qc_plots/` - Directory containing PNG plot files (if `save_plots: true`)

### Reproducibility with --from-provenance

The `metadata.json` output contains all processing parameters, enabling exact re-runs:

```bash
# Re-run with exact same parameters on new data
prism run -i new_data.csv -o output2/ --from-provenance output1/metadata.json

# Override specific settings while keeping others from provenance
prism run -i new_data.csv -o output2/ --from-provenance output1/metadata.json -c overrides.yaml
```

**Implementation**: `skyline_prism/cli.py` -> `load_config_from_provenance()`

Additional utility commands:

```bash
# Merge multiple Skyline reports into unified parquet
prism merge report1.csv report2.csv -o data.parquet -m metadata.tsv

# Regenerate QC report from existing output (without reprocessing)
prism qc -d output_dir/

# Generate annotated configuration template
prism config-template -o config.yaml

# Minimal config template (common options only)
prism config-template --minimal -o config.yaml
```

## Common Tasks

### Adding a New Feature

1. Read SPECIFICATION.md to understand the design
2. Implement in the appropriate module
3. Add tests in `tests/`
4. Run `pytest tests/ -v` to verify
5. Update README.md if user-facing
6. Update config_template.yaml if configurable
7. Commit with descriptive message

### Fixing a Bug

1. Write a failing test that reproduces the bug
2. Fix the bug
3. Verify the test passes
4. Run full test suite
5. Commit with reference to the issue if applicable

### Modifying Imports

The package exports are defined in `skyline_prism/__init__.py`. Key exports include:
- Data I/O: `load_skyline_report`, `merge_skyline_reports`, `load_sample_metadata`
- Rollup: `tukey_median_polish`, `rollup_to_proteins`, `rollup_transitions_to_peptides`
- Normalization: `normalize_pipeline`, `rt_correction_from_reference`
- Batch correction: `combat`, `combat_from_long`, `combat_with_reference_samples`
- Parsimony: `compute_protein_groups`, `ProteinGroup`
- Validation: `validate_correction`, `generate_qc_report`
- Visualization: `plot_intensity_distribution`, `plot_pca`, `plot_cv_distribution`, `plot_rt_correction_comparison`

## Important Notes

- **Skyline** is an external tool (https://skyline.ms) - we process its exports, we don't modify Skyline itself
- **Sample types**: `experimental`, `qc`, `reference` - these have specific meanings in the normalization workflow
- **Column naming**: Internal column names differ from Skyline export names - see `SKYLINE_COLUMN_MAP` in data_io.py
- **Log scale**: Most operations work on log2-transformed abundances
- **Median polish is default**: For both transition→peptide and peptide→protein rollups
- **Two-arm pipeline**: Batch correction happens at the reporting level (peptide or protein), not before rollup

## FASTA-Based Protein Parsimony

The `fasta.py` module provides FASTA parsing for proper protein parsimony:

**Key functions:**
- `parse_fasta()`: Parse UniProt/NCBI format FASTA files
- `strip_modifications()`: Remove modifications from peptide sequences for matching
- `normalize_for_matching()`: Handle I/L ambiguity (MS cannot distinguish)
- `build_peptide_protein_map_from_fasta()`: Build complete peptide-protein mapping via substring search

**Usage in parsimony:**
```python
from skyline_prism.parsimony import build_peptide_protein_map_from_fasta

pep_to_prot, prot_to_pep, prot_names = build_peptide_protein_map_from_fasta(
    df,
    fasta_path="/path/to/search.fasta",
)
```

**Note:** The module also contains in-silico digestion functions (`digest_protein()`, `digest_fasta()`)
which are used for iBAQ (to count theoretical peptides per protein). Peptide-protein mapping for
parsimony uses direct substring search - no enzyme parameters needed.

## iBAQ Support

iBAQ (Intensity-Based Absolute Quantification) is now integrated. It normalizes protein abundances
by the number of theoretical peptides, enabling cross-protein abundance comparison.

**Key function:**
- `get_theoretical_peptide_counts()`: Count theoretical peptides per protein for iBAQ

**Usage:**
```python
from skyline_prism.fasta import get_theoretical_peptide_counts

counts = get_theoretical_peptide_counts(
    "/path/to/database.fasta",
    enzyme="trypsin",
    missed_cleavages=0,  # Strict for iBAQ
)
```

**Configuration:**
```yaml
protein_rollup:
  method: "ibaq"
  ibaq:
    fasta_path: "/path/to/database.fasta"
    enzyme: "trypsin"
    missed_cleavages: 0
```

---

## Current Implementation Status

This section tracks what's currently working, what needs attention, and what's not yet implemented.

### [WORKING] Fully Implemented and Tested (December 2024)

**Core Pipeline:**

- Streaming CSV merge (handles ~47GB datasets)
- Transition → Peptide rollup (Tukey median polish)
- Peptide global normalization (median-based)
- Peptide batch correction (ComBat, empirical Bayes)
- Protein parsimony (FASTA-based grouping)
- Peptide → Protein rollup (Tukey median polish)
- Protein global normalization (median-based)
- Protein batch correction (ComBat, empirical Bayes)
- Log file generation (timestamped in output directory)
- Parquet output with metadata
- Provenance tracking (metadata.json)

**Data Handling:**

- Automatic column detection (handles different Skyline export formats)
- Metadata normalization (`sample`/`sample_type`/`batch` from Skyline formats)
- Sample type pattern matching (reference/QC/experimental detection)
- Batch estimation from source files or timestamps
- Duplicate sample validation (allows same sample across batches)

**Testing:**

- 182 tests passing
- Core algorithms well-tested (median polish, ComBat, parsimony)
- Real-world validation on 238 samples, 3 batches, ~47GB data

### [ISSUE] Known Issues / Needs Attention

**QC Reporting:**

- **CV calculation bug**: Reference/QC median CV shows NaN - sample type matching not working correctly
- **Protein NaN values**: Global median/max shift occasionally NaN for proteins - investigate data quality checks
- **QC report warning**: "list index out of range" during generation in some edge cases

**ComBat Evaluation:**

- **Automatic fallback not implemented**: QC-based decision to revert correction if quality degrades
- **Reference-anchored evaluation**: Method exists but automatic QC evaluation not active
- **Current behavior**: Always applies ComBat when enabled; need to add quality checks

### [DISABLED] Implemented but Disabled by Default

**RT Correction:**

- Fully implemented but **disabled by default**
- Reason: Search engine RT calibration (DIA-NN) may not generalize between samples
- Can enable via `rt_correction.enabled: true` in config
- Uses spline-based correction fitted to reference samples

### [TODO] Not Yet Implemented

**Advanced Features:**

- VSN normalization (placeholder in config)
- Per-batch RT models with cross-validation
- Quality-weighted protein rollup
- iBAQ support (code exists but not integrated into pipeline)

### [PRIORITY] Development Priorities

Based on current usage and known issues:

1. **Fix CV calculation bug** - Critical for QC validation
2. **Investigate protein NaN values** - May indicate data quality issues
3. **Implement ComBat quality checks** - Enable automatic fallback
4. **Improve QC report robustness** - Fix edge cases causing warnings

### [COVERAGE] Test Coverage Details

**High coverage (>85%):**

- `fasta.py`: 95% - Protein parsimony and FASTA parsing
- `transition_rollup.py`: 93% - Transition → Peptide aggregation
- `batch_correction.py`: 89% - ComBat implementation
- `parsimony.py`: 78% - Protein grouping

**Low coverage (<30%):**

- `cli.py`: 13% - Command-line interface (mainly integration code)
- `normalization.py`: 12% - RT correction (disabled by default)
- `data_io.py`: 28% - File I/O (tested via integration)
- `validation.py`: 10% - QC reporting (needs more unit tests)

**Overall**: 47% coverage, 182 tests passing

### [CHANGELOG] Recent Changes Log

**December 2024:**

- Implemented log file generation with timestamps
- Fixed metadata column handling (`sample` vs `replicate_name`)
- Added support for duplicate samples across batches (Reference/QC in multiple plates)
- Improved protein sample column detection using dtype checks
- Updated stage naming (1, 2, 2b, 2c, 3, 4, 4b, 4c, 5, 5b)
- Validated on 238 samples across 3 batches (~47GB total data)
- Added input data summary logging (transitions, peptides, samples)

## Not Yet Implemented

### directLFQ

directLFQ is a protein quantification algorithm that offers linear O(n) runtime scaling, making it suitable for very large cohorts (100s-1000s of samples). It is fundamentally different from maxLFQ - not just an optimization.

**Why it's different from maxLFQ:**
- maxLFQ uses pairwise median log-ratios between samples (O(n²) complexity)
- directLFQ uses an "intensity trace" approach with anchor alignment (O(n) complexity)

**Citation:** Ammar C, Schessner JP, Willems S, Michaelis AC, Mann M. "Accurate label-free quantification by directLFQ to compare unlimited numbers of proteomes." Molecular & Cellular Proteomics. 2023;22(7):100581. doi:10.1016/j.mcpro.2023.100581

**GitHub:** https://github.com/MannLabs/directlfq

**Status:** Not implemented in PRISM. For very large cohorts, users should use the directLFQ package directly. May be added in a future version.

## Design Decisions to Preserve

1. **RT correction from reference only**: Never learn RT effects from experimental samples
2. **Batch correction at reporting level**: Not before protein rollup
3. **Median polish as default**: Quality-weighted is an alternative, not the primary method
4. **All charge states as transitions**: Don't separate precursor→peptide rollup; treat all transitions equally

## Repository Information

- **GitHub**: https://github.com/maccoss/skyline-prism
- **Owner**: maccoss (MacCoss Lab, University of Washington)
- **License**: MIT
