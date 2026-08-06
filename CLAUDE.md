# CLAUDE.md - AI Agent Guidelines for Skyline-PRISM

This document provides context and guidelines for AI agents working on the Skyline-PRISM project.
It is the single source of truth for agent guidance (it supersedes the former `AGENTS.md`).

> **Building or extending the Skyline external tool?** The canonical field guide (RPC, reports, `.blib`,
> chromatograms, packaging), a scaffolding skill, and a `dotnet new` template now live in
> **[uw-maccosslab/skyline-external-tools-ai](https://github.com/uw-maccosslab/skyline-external-tools-ai)**
> (`docs/skyline-external-tools.md`). PRISM is one of the example tools it draws from.
>
> **Read it from the local clone before touching anything under `dotnet/src/SkylinePrism.{Skyline,App}/`:**
> `D:\GitHub-Repo\uw-maccosslab\skyline-external-tools-ai`
> (on Linux/macOS, a sibling clone of the same repo). Start with `CRITICAL-RULES.md` (the hard-won
> gotchas: transform `args[0]`, connect-per-call, `PipeTransmissionMode.Message`, invariant culture,
> `.blib` `Pooling=False`, the launch-verify ship gate), then `TOC.md` -> `docs/skyline-external-tools.md`
> for the section you need. `git pull` it if it looks stale; **update the guide there, not here** -
> `docs/skyline-external-tools.md` in this repo is only a pointer stub.

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

> [!CAUTION]
> **PARQUET FILES MUST ALWAYS CONTAIN LINEAR VALUES (NOT LOG2)**
>
> This is a critical requirement that has been incorrectly regressed multiple times.
> Always verify that output parquet files contain linear abundance values, not log2-transformed values.

| Stage | Scale | Notes |
|-------|-------|-------|

| **Input** | LINEAR | Raw peak areas from Skyline |
| **Internal** | LOG2 | All rollup/normalization operates on log2 scale |
| **Output** | LINEAR | Final peptide/protein output matrices (parquet/CSV) are always written in LINEAR scale (values are 2^x, not log2(x)) |

**Implementation Location:**
The log2→linear conversion is enforced in `skyline_prism/cli.py` at the final output stage:
- Peptide output: Line ~1610 (`for col in pep_sample_cols: peptide_output_df[col] = np.power(2, ...)`)
- Protein output: Line ~1835 (`for col in prot_sample_cols: protein_output_df[col] = np.power(2, ...)`)

**Important:** Intermediate files (`peptides_rollup.parquet`, `proteins_raw.parquet`) remain in LOG2 scale for normalization and batch correction. Only the **final** output files (`corrected_peptides.parquet`, `corrected_proteins.parquet`) are converted to linear.

**Display conventions:**
- **Box plots**: Display LINEAR values (from parquet)
- **PCA plots**: Use LOG2 internally (convert from linear parquet for variance stabilization)
- **CV calculations**: Always on LINEAR values

**Do not write log2 values to final output files.**

The pipeline automatically handles transforms:
- Input linear values are log2-transformed for processing
- Output values are back-transformed to linear (2^x) before writing

When using functions directly via Python API, check docstrings for scale requirements.

### Data Density (CRITICAL - PRISM input is normally COMPLETE)

> [!IMPORTANT]
> **Skyline exports have no missing values.** Skyline integrates *imputed peak boundaries* for every
> replicate, so every transition has an area in every run - even where there is no real signal. The
> peptide x sample matrix that reaches Stage 2b/2c is therefore **dense**, and NaN is the rare
> exception, not the norm.

Do not design or justify an algorithm on the assumption that missing values are common. When
choosing between two behaviours, **the dense case is the one that matters**; the sparse case must be
*supported*, not optimized for. (This was got backwards once: a ComBat change was justified as
"matches the reference implementation on data with missing values - which is all real proteomics
data". The opposite is true.)

Supporting evidence in the repo: the `mini` golden fixtures contain 0 nulls and 0 NaNs; `#N/A`
tokens in a real export are imputed at the **transition** level (`tests/.../MissingValueTests.cs`);
`TransitionRollup` emits NaN for a peptide x sample cell only when that peptide had *no* transition
with data at all in that run.

Missing values do still occur, and must keep working:

- **Merging documents with different target lists.** PRISM takes a LIST of Skyline documents (one
  per plate/batch). A peptide present in one document but absent from another is NaN for every
  replicate of the second - which is also the "feature absent from a whole batch" case that
  `sva::ComBat` cannot express at all (it dies with a singular design).
- Peptides filtered out of one run but not another, and genuinely unintegrated data.

### CV Calculation (CRITICAL)

**CVs must ALWAYS be calculated on LINEAR scale data, NEVER on log-transformed data.**

Correct calculation:
```python
linear_data = 2 ** log2_data  # Convert from log2 to linear
cv = (linear_data.std() / linear_data.mean()) * 100  # CV as percentage
```

Rationale: On log scale, variance is artificially compressed. A CV of 5% on log2 data would be meaningless - true biological CVs for proteomics control samples typically range from 10-30%.

### Peptide Modification Format (CRITICAL)

> [!CAUTION]
> **Skyline exports and BLIB spectral libraries use DIFFERENT modification formats.**
>
> This has caused repeated issues. Always convert between formats when matching peptides to library spectra.

| Source | Format | Example |
|--------|--------|---------|
| **Skyline export** | Unimod IDs in parentheses | `C(unimod:4)`, `M(unimod:35)` |
| **BLIB library** | Mass deltas in brackets | `C[+57.02146]`, `M[+15.99491]` |

**Common modifications:**

| Modification | Unimod Format | BLIB Format |
|--------------|---------------|-------------|
| Carbamidomethyl (Cys) | `C(unimod:4)` | `C[+57.02146]` |
| Oxidation (Met) | `M(unimod:35)` | `M[+15.99491]` |
| Phosphorylation (Ser) | `S(unimod:21)` | `S[+79.96633]` |
| Phosphorylation (Thr) | `T(unimod:21)` | `T[+79.96633]` |
| Phosphorylation (Tyr) | `Y(unimod:21)` | `Y[+79.96633]` |

**Conversion function:**
```python
import re

def convert_unimod_to_blib_format(seq):
    """Convert Unimod format C(unimod:4) to BLIB format C[+57.02146]."""
    conversions = {
        r'C\(unimod:4\)': 'C[+57.02146]',     # Carbamidomethyl
        r'M\(unimod:35\)': 'M[+15.99491]',    # Oxidation
        r'S\(unimod:21\)': 'S[+79.96633]',    # Phospho
        r'T\(unimod:21\)': 'T[+79.96633]',    # Phospho
        r'Y\(unimod:21\)': 'Y[+79.96633]',    # Phospho
    }
    result = seq
    for pattern, replacement in conversions.items():
        result = re.sub(pattern, replacement, result)
    return result
```

**When matching peptides to library:**
1. First try the original key format from the data
2. If not found, convert to BLIB format and retry
3. Also try I/L normalization (mass spec cannot distinguish isoleucine/leucine)

**PRISM implementation:** The `SpectralLibraryLoader.normalize_sequence_for_matching()` handles I/L normalization, but modification format conversion must be done separately when working with raw Skyline exports.

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
│   ├── chunked_processing.py # Memory-efficient chunked/streaming processing
│   ├── data_io.py           # Skyline report loading and merging (includes merge_and_sort_streaming)
│   ├── normalization.py     # RT-aware correction pipeline
│   ├── batch_correction.py  # ComBat implementation (empirical Bayes)
│   ├── parsimony.py         # Protein grouping and shared peptide handling
│   ├── rollup.py            # Peptide → Protein rollup (median polish, etc.)
│   ├── spectral_library.py  # Library-assisted rollup with least squares fitting
│   ├── transition_rollup.py # Transition → Peptide rollup (median polish, quality-weighted, variance learning)
│   ├── validation.py        # QC metrics and reporting (generates HTML QC reports with embedded plots)
│   └── visualization.py     # Plotting functions for QC assessment and normalization evaluation
├── tests/                   # Unit tests (pytest)
│   ├── test_data_io.py
│   ├── test_parsimony.py
│   ├── test_rollup.py
│   ├── test_spectral_library.py  # Library-assisted rollup tests
│   └── test_transition_rollup.py
├── SPECIFICATION.md         # Detailed technical specification
├── README.md                # User-facing documentation
├── config_template.yaml     # Static reference (NOT the source - see cli.py)
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

For transition→peptide aggregation, the adaptive method learns optimal weighting parameters to minimize CV:

**Weight Formula** (`AdaptiveRollupParams`):
```
w_t = exp(beta_mz * normalized_mz + beta_shape_outlier * outlier_frac)
```

Where:
- `normalized_mz`: Product m/z normalized to [0, 1] range
- `outlier_frac`: Fraction of samples where shape correlation < threshold (indicates interference)

**Key insight:** When all betas = 0, weights = 1 for all transitions (equivalent to simple sum). The optimizer only uses learned weights if they improve CV.

**What gets optimized:**
- `beta_mz`: Higher m/z fragments may have better signal (positive = favor high m/z)
- `beta_shape_corr_outlier`: Transitions with frequent interference should be down-weighted (negative = penalize)

**The peptide abundance calculation:**
```
Peptide_abundance = log2(Σ weight_t × intensity_t)
```

The transition intensities are the VALUES being summed. The learned weights adjust how much each transition contributes based on its quality metrics (m/z and interference level).

**Learning process**:
1. Parameters optimized on reference samples by minimizing median CV (L-BFGS-B optimizer)
2. Validated on QC samples (held-out) to prevent overfitting
3. Automatic fallback to simple sum if adaptive doesn't improve CV by `min_improvement_pct`

**Implementation**:
- `skyline_prism/transition_rollup.py` → `learn_adaptive_weights()` - learns parameters
- `skyline_prism/transition_rollup.py` → `rollup_peptide_adaptive()` - applies weights
- `skyline_prism/transition_rollup.py` → `compute_adaptive_weights()` - computes weights from params

**Configuration:**
```yaml
transition_rollup:
  method: "adaptive"
  learn_adaptive_weights: true
  adaptive_rollup:
    beta_mz: 0.0                  # Starting value (optimized automatically)
    beta_shape_corr_outlier: 0.0  # Starting value (optimized automatically)
    shape_corr_low_threshold: 0.7 # Threshold for "low" shape correlation
    min_improvement_pct: 0.1      # Required improvement over sum
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
- When adding new configuration options, update the template functions in `cli.py` (see below)

**CRITICAL: Configuration is a contract - keep the template, schema, and docs in sync**

A config-driven feature has multiple surfaces that MUST stay consistent. Whenever you **add, rename, or
remove** a configuration key, update ALL of the relevant surfaces in the SAME change:

Python engine (`skyline_prism/`):
1. `get_full_config_template()` in `skyline_prism/cli.py` - the annotated full template
2. `get_minimal_config_template()` in `skyline_prism/cli.py` - only if it's a common option
3. `KNOWN_CONFIG_KEYS` in `skyline_prism/cli.py` - the key-validation schema

C# port (`dotnet/`):
1. The config class property in `dotnet/src/SkylinePrism.Core/Config/PrismConfig.cs`
2. `ConfigTemplate.Default()` (and `.Minimal()` if common) in `ConfigTemplate.cs` - the emitted YAML
3. The `KnownKeys` schema (`BuildSchema()`) in `PrismConfig.cs` - so `FindUnknownKeys` accepts it while
   still warning on typos/unknown keys

Both engines:
4. `docs/parameters.md` - the cross-engine parameter reference (key, default, availability).

Confirmation is MANDATORY, not optional:
- **Adding a feature:** run `prism config-template` and confirm the new key appears in the generated YAML
  (active, or a commented example for method-specific keys), and add a row to `docs/parameters.md`. The
  C# test `ConfigValidationTests.ConfigTemplate_HasNoUnknownKeysAndValidates` fails if the emitted
  template contains a key the schema does not know - fix the schema/template, do not silence it.
- **Removing a feature:** delete the key from the template AND the schema AND `docs/parameters.md`. A
  removed key must never linger in the generated YAML.
- **Deliberately NOT porting a feature to C# (or vice-versa):** this is a specific, recorded decision.
  Note it in `docs/parameters.md` (availability "Python only" / "C# only", with the reason) and in
  `dotnet/PORTING_STATUS.md` "Config surface & parity". The C# port must never silently ignore a key -
  make it warn (`FindUnknownKeys`) or abort (`PrismConfig.Validate`).

These functions/files - NOT the static `config_template.yaml` in the repo root - are what the software
emits. `config_template.yaml` is a reference copy only. Users generate their configs via:
```bash
prism config-template -o config.yaml           # Full template
prism config-template --minimal -o config.yaml  # Minimal template
```

**SPECIFICATION.md** contains the detailed technical specification. Reference it for algorithm details but avoid modifying it unless the fundamental approach changes.

## Key Files to Understand

### SPECIFICATION.md
The authoritative technical specification. Contains:
- Input/output formats (Skyline report columns)
- Algorithm details (RT correction, median polish, parsimony)
- Processing pipeline stages (two-arm design)
- Configuration parameters

### Configuration Templates (IMPORTANT)

**The configuration templates are generated from functions in `skyline_prism/cli.py`, NOT from `config_template.yaml`.**

To update configuration options (see "CRITICAL: Configuration is a contract" above for the full
checklist across both engines + `docs/parameters.md`):
1. Edit `get_full_config_template()` in `cli.py` for full template
2. Edit `get_minimal_config_template()` in `cli.py` for minimal template
3. The C# port has its own template in `dotnet/src/SkylinePrism.Core/Config/ConfigTemplate.cs`
   (`Default()` / `Minimal()`) plus the `KnownKeys` schema in `PrismConfig.cs` - keep them in sync
4. The static `config_template.yaml` is just a reference and is NOT read by the software

Users generate templates via:
- `prism config-template -o config.yaml` (full template)
- `prism config-template --minimal -o config.yaml` (common options only)

Key sections:
- `transition_rollup`: Transition→peptide rollup (method: sum, median_polish, adaptive, library_assist)
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
prism run -i skyline_report.csv -o output_dir/ -c config.yaml -m metadata.csv
```

This produces:
- `corrected_peptides.parquet` - Peptide-level normalized/batch-corrected quantities
- `corrected_proteins.parquet` - Protein-level normalized/batch-corrected quantities
- `peptides_rollup.parquet` - Raw peptide abundances from transition rollup (before normalization)
- `proteins_raw.parquet` - Raw protein abundances from peptide rollup (before normalization)
- `protein_groups.csv` - Protein group definitions
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
6. If configurable: update every config surface and confirm the generated YAML has the key - see
   "CRITICAL: Configuration is a contract" above (templates + schema in `cli.py` AND, for the C# port,
   `ConfigTemplate.cs` + `PrismConfig.cs`; plus `docs/parameters.md`). Run `prism config-template` and
   verify the key appears; remove it everywhere when removing a feature.
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
- `build_peptide_protein_map_from_fasta()`: Build peptide-protein mapping via **enzyme-aware** substring search
- `cleavage_boundary_set()` / `terminus_is_enzymatic()`: enzyme terminus check helpers

**Usage in parsimony:**
```python
from skyline_prism.parsimony import build_peptide_protein_map_from_fasta

pep_to_prot, prot_to_pep, prot_names = build_peptide_protein_map_from_fasta(
    df,
    fasta_path="/path/to/search.fasta",
    enzyme="trypsin",           # parsimony.enzyme (see docs/parameters.md)
    enzyme_specificity="full",  # full | semi | none
)
```

**Enzyme-aware membership (important):** substring containment is *necessary but not sufficient* for a
peptide to originate from a protein. The map attaches a peptide to a protein only when it occurs there
with termini consistent with the digestion enzyme (`parsimony.enzyme` / `parsimony.enzyme_specificity`,
default `trypsin` / `full`, with initiator-methionine excision handled). This removes "phantom"
assignments to homologs that share the subsequence but not the flanking cleavage site — e.g.
`AKEGVVAAAEK` is a substring of beta-synuclein (SNCB) but is preceded there by `M`, not `K/R`, so
trypsin cannot liberate it; it is proteotypic to alpha-synuclein (SNCA). Set `enzyme_specificity: none`
to restore the legacy pure-substring behavior. The check only applies on the FASTA path; the Skyline
Protein Accession column is already enzyme-aware. C# mirrors this exactly in
`FastaParser.CleavageBoundaries` / `BuildMap`, so the two engines produce identical maps.

**Note:** The module also contains in-silico digestion functions (`digest_protein()`, `digest_fasta()`)
which are used for iBAQ (to count theoretical peptides per protein). iBAQ digestion and the parsimony
terminus check share the same enzyme rules but are separate code paths.

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

## Interacting with Skyline (C# external tool)

The C# Skyline external tool (`dotnet/src/SkylinePrism.App`) exports the reports the pipeline consumes
and reads document settings — from a **running** Skyline over JSON-RPC, or from a **closed** `.sky` via
`SkylineCmd`. The authoritative how-to is the
field guide in **[uw-maccosslab/skyline-external-tools-ai](https://github.com/uw-maccosslab/skyline-external-tools-ai)**
(`docs/skyline-external-tools.md`), cloned locally at
`D:\GitHub-Repo\uw-maccosslab\skyline-external-tools-ai` — read it before extending the
tool's Skyline integration. Key points (mirrored in the code under `dotnet/src/SkylinePrism.Skyline/`):

- **Transport:** JSON-RPC 2.0 over a **named pipe**. Skyline passes the pipe name as `args[0]`
  (`$(SkylineConnection)`); transform it with `JsonToolConstants.GetJsonPipeName`. **Connect per call**
  — open a fresh `NamedPipeClientStream` (message mode) for each request and close it; never hold the
  pipe open (idle reuse throws deserialization errors). `SkylineSession` implements this; the driver
  talks to it through the small `ISkylineClient` interface so tests can use a fake (no live Skyline).
- **Report export:** prefer exporting whole reports to **parquet** (typed, ~20× faster than paginating),
  falling back to invariant-culture CSV. `SkylineReportDriver.Export` does this and also reads the
  built-in "Replicates" document grid for metadata.
- **Reading document settings** (e.g. the digestion **enzyme**): use the settings-list RPCs —
  `GetSettingsListSelectedItems("Enzymes")` for the active enzyme name, then
  `GetSettingsListItem("Enzymes", name)` for its XML. An enzyme element looks like
  `<enzyme name="Trypsin" cut="KR" no_cut="P" sense="C" />` (`cut` = cleavage residues, `no_cut` =
  blocking residues, `sense` = `C` cleave-after / `N` cleave-before). `SkylineDigestion` parses that XML
  and maps it to a PRISM `parsimony.enzyme` name; `SkylineReportDriver.GetDigestionEnzyme()` wires it up.
  The same `GetSettingsListSelectedItems`/`GetSettingsListItem` pattern reads the active spectral library.
- **General rule (per Nick Shulman):** anything `SkylineCmd` can do is reachable via
  `RunCommand([...flags...])` against the live document — but SkylineCmd aborts the whole batch on the
  first bad flag, so send flags one per `RunCommand` (except fields Skyline mutually validates).

> [!CAUTION]
> **Annotation columns in a generated `.skyr` MUST be quoted:** `<column name="&quot;annotation_Plate&quot;" />`,
> never the bare `annotation_Plate`. Skyline parses `column/@name` as a databinding PropertyPath, whose
> bare-identifier syntax rejects `_` — and the `annotation_` prefix contains one. Unquoted, the export
> aborts with `Error parsing annotation_Plate at location 10: Invalid character _` and **no report is
> written**, so the annotation silently never reaches the metadata. Quoted, the exported column is headed
> with the plain annotation name (`Plate`). Both engines build the view through
> `ReplicatesReportBuilder`, which applies the quoting — go through it rather than hand-rolling XML.

### Inputs: multiple documents, open or closed (`PrismInput`)

The tool takes a LIST of inputs — one per batch/plate — merged into a single cohort. `PrismInput.Prepare`
resolves each to a report + metadata file:

- **Running Skyline** (`SkylineSession`): the launching instance, or any other found by
  `SkylineSession.DiscoverRunning` (which reads `~/.skyline-mcp/connection-*.json` — Skyline only appears
  there while its JSON-RPC server runs, so treat that list as a convenience, not an inventory).
- **Closed `.sky`** (`HeadlessSkylineExporter` over `ISkylineCommandRunner`): the document is loaded once
  per report (`--report-name` takes one), always with `--in` and **never** `--save`. Two runners:
  - `SkylineAppRunner` (**preferred**) drives the installed Skyline headlessly — the SkylineRunner
    protocol, reimplemented so it finds Skyline *or* Skyline-daily. It runs the real `Skyline.exe`, so
    `Skyline.exe.config` applies and **parquet export works**. ⚠️ No exit code: failure is only visible as
    an `Error:` prefix in the piped output (`SkylineAppRunner.IsErrorLine`).
  - `SkylineCmdRunner` (fallback) uses `SkylineCmd.exe` via `SkylineCmdLocator` (ClickOnce **application**
    folder — the one with `Skyline*.exe` beside it; the sibling `…exe_…` copies fail with "Unable to find
    Skyline.exe" — newest first, overridable with `PRISM_SKYLINECMD`). It reports
    `SupportsParquet = false`: `SkylineCmd.exe.config` lacks the Parquet.Net assembly bindings, so a
    `.parquet` request dies with "Could not load file or assembly 'Parquet'".

  Format follows the output **extension** (`--report-format` only accepts `csv|tsv`), and the result is
  verified with `ParquetMagic` rather than trusted.
- **Pre-exported report**: used in place; no Skyline at all. This is what makes the window usable as a
  **standalone PRISM GUI** — `MainWindow` must never require a non-null `_session` to run.
- **Closed-document metadata** comes from `SkyDocumentInfo`, which stream-parses the `.sky` header
  (stopping at `</settings_summary>`, so a 2 GB document reads in ~1 s) for the replicate-targeted
  `<annotation targets="…replicate…">` names, the `<enzyme>` element, and the replicate list. Read-only.

**Batch labels are file stems.** Each input's label is the exported report's file name, because
`DuckDbMerge` derives `Batch` / `Source Document` from the stem and builds sample IDs as
`<replicate>__@__<batch>`. Labels must therefore be unique and file-name safe
(`PrismInput.EnsureUniqueLabels`). Metadata files are passed to the pipeline **positionally**, 1:1 with
the inputs, so `ReplicateMetadata` can key rows by source document — see the caution below.

> [!CAUTION]
> **Replicate names collide across documents.** Reference and QC injections are normally named
> identically in every plate's document. `ReplicateMetadata` therefore stores each file's rows under the
> document-qualified key `<replicate>__@__<document>` in addition to the bare name, and
> `TypeFor`/`BatchFor`/`HasBatchFor` prefer the qualified entry. Keyed by bare name alone the last file
> silently wins — collapsing two batches into one label (so ComBat is skipped without a word) and giving
> every plate the last document's sample types. Do not "simplify" these lookups back to the bare map.

**Why PRISM reads the enzyme from the document:** PRISM's FASTA-based parsimony is enzyme-aware (see
"FASTA-Based Protein Parsimony"). The **CLI** takes the enzyme from `parsimony.enzyme` (default
`trypsin`), but the **external tool** overrides it from the document's digestion settings so the
membership check matches the search that produced the data. If the document enzyme has no PRISM
equivalent (or can't be read), the tool keeps the config default.

---

## Current Implementation Status

This section tracks what's currently working, what needs attention, and what's not yet implemented.

### [WORKING] Fully Implemented and Tested (December 2024)

**Core Pipeline:**

- Streaming CSV merge (handles ~47GB datasets)
- Transition → Peptide rollup (Tukey median polish, adaptive weighted)
- Adaptive rollup weight learning from reference samples
- Peptide global normalization (median-based)
- Peptide batch correction (ComBat, empirical Bayes)
- Protein parsimony (FASTA-based grouping)
- Peptide → Protein rollup (Tukey median polish)
- Protein global normalization (median-based)
- Protein batch correction (ComBat, empirical Bayes)
- Log file generation (timestamped in output directory)
- Parquet output with metadata
- Provenance tracking (metadata.json)
- Config key validation (warns about unknown/typo config keys)

**Data Handling:**

- Automatic column detection (handles different Skyline export formats)
- Metadata normalization (`sample`/`sample_type`/`batch` from Skyline formats)
- Sample type pattern matching (reference/QC/experimental detection)
- Batch estimation from source files or timestamps
- Duplicate sample validation (allows same sample across batches)
- Scale handling: All parquet files output in LINEAR scale

**Testing:**

- 196 tests passing
- Core algorithms well-tested (median polish, ComBat, parsimony)
- Scale handling tests (log2/linear conversions, CV calculation)
- Config validation tests
- Real-world validation on 238 samples, 3 batches, ~47GB data

### [ISSUE] Known Issues / Needs Attention

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

**Overall**: 60% coverage, 291 tests passing

### [CHANGELOG] Recent Changes Log

**December 2024:**

- Implemented log file generation with timestamps
- Fixed metadata column handling (`sample` vs `replicate_name`)
- Added support for duplicate samples across batches (Reference/QC in multiple plates)
- Improved protein sample column detection using dtype checks
- Updated stage naming (1, 2, 2b, 2c, 3, 4, 4b, 4c, 5, 5b)
- Validated on 238 samples across 3 batches (~47GB total data)
- Added input data summary logging (transitions, peptides, samples)

**December 30, 2024:**

- Fixed log2/linear scale handling throughout pipeline (overflow prevention)
- Added config key validation (detects typos like `learn_weights` vs `learn_adaptive_weights`)
- Fixed `learn_adaptive_weights` default to be True when `method: adaptive`
- Fixed `shape_corr_low_threshold` not being passed from config to learning function
- Renamed output files: `peptides_rollup.2.parquet` -> `peptides_rollup.parquet`
- Removed redundant `peptides_normalized.3.parquet` (same as `corrected_peptides.parquet`)
- Added comprehensive scale handling tests (`tests/test_scale_handling.py`)
- Test count increased from 182 to 196

**December 31, 2024:**

- **Released v0.1.2**
- Added automatic batch estimation from acquisition times (`batch_estimation` config)
- Added support for multiple metadata files (`-m file1.csv file2.csv`)
- Added `Replicate` as accepted column name in metadata
- Fixed Sample ID vs Replicate Name mismatch throughout pipeline (helper functions)
- Fixed QC report sample type detection (Reference/QC now appear correctly in plots)
- Fixed duplicate progress logging in streaming peptide rollup

**January 2025:**

- **Library-assisted rollup (v10)**: Spectral library-based interference detection
  - Uses library as PRIOR for row effects (transition ionization efficiency)
  - Two fitting methods available via `fitting_method` config:
    - `median_polish` (DEFAULT, RECOMMENDED): Estimates sample scale via MEDIAN
      - Robust to 1-2 outliers automatically (median ignores up to 50% outliers)
      - Model: `log(obs) = log(lib) + beta_s + epsilon`
      - Final abundance = `exp(beta_s) * sum(ALL library intensities)`
    - `least_squares`: Classic OLS: `scale = (lib . obs) / (lib . lib)`
      - More sensitive to outliers
      - May be better for very clean data
  - Both methods: only HIGH positive residuals indicate interference (obs > 2x predicted)
  - Supports BLIB (Skyline) and Carafe TSV (DIA-NN) library formats
  - Dramatically improves CV for peptides with real interference
- **Vectorized implementation**: All samples processed in parallel using numpy
  - ~10x speedup for library-assisted rollup on large datasets
  - Implementation: `spectral_library.py` -> `library_median_polish_rollup_vectorized()`
  - Legacy: `spectral_library.py` -> `least_squares_rollup_vectorized()`
- **Merge-and-sort streaming**: CSV merge and sort in single DuckDB operation
  - Eliminates redundant sorting pass, faster for large multi-file datasets
  - Implementation: `data_io.py` -> `merge_and_sort_streaming()`
- **Pre-sorted optimization**: Rollup skips sorting when data is already sorted
  - Implementation: `chunked_processing.py` -> `rollup_transitions_sorted(pre_sorted=True)`
- Test count increased from 196 to 291 (comprehensive spectral library tests)

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
5. **Cross-platform CLI, Windows-only GUI** (decided, not an interim state): the `prism` CLI ships for
   Windows/Linux/macOS and `SkylinePrism.Core` stays platform-neutral, while the GUI stays **WPF on
   Windows**. A cross-platform GUI (Avalonia et al.) was considered and declined - Skyline itself is
   Windows-only, so the attached mode could never be portable, and the standalone case is served by the
   CLI. Consequences to respect:
   - Do **not** port `SkylinePrism.App` to another UI framework, and do not add a `net8.0` target to it.
   - GUI-only helpers may sit in `SkylinePrism.App` and depend on the Windows-only
     `SkylinePrism.Skyline` (e.g. `PrismInput`, `StandaloneShortcut`); they do **not** need to move to
     Core "for portability".
   - Anything a headless/Linux user needs must be reachable from the CLI. The GUI's "Show Command Line"
     exists to keep that honest - it emits the exact `prism run` invocation for the current settings.
   - The one thing still worth extracting from `MainWindow` is a view-model for **testability**
     (it is ~1000 lines of code-behind at 0% coverage), not for portability.

## Release Process

Skyline-PRISM currently ships **two implementations side by side**, released on **independent tracks**:

- **Python** package (`skyline-prism` on PyPI) - the original reference implementation.
- **C# (.NET 8)** tools - the `prism` CLI + the Windows Skyline external tool, published to GitHub Releases.

> [!NOTE]
> The side-by-side period is **temporary**. The Python package is planned for retirement once the C#
> port is the sole supported implementation; at that point only the C# (`dotnet-v*`) track remains.
> Until then keep both releasable, but treat Python as legacy - new development targets the C# port.

### Versioning scheme (both tracks)

Both tracks use CalVer **`YY.feature.patch`** (e.g. `26.5.0` = year 2026, feature release 5, patch 0):
**YY** = two-digit year, **feature** bumps for new features, **patch** for bug-fix-only releases. The
version is bumped **only at release time**, not during development. The two tracks share the *scheme*
but keep **distinct tag namespaces and independent counters**, so their numbers need not match and
never collide:

- Python: tag `v{version}` (e.g. `v26.4.2`)
- C#: tag `dotnet-v{version}` (e.g. `dotnet-v26.5.0`)

### Release notes (both tracks)

All release notes live in `release-notes/`; **`release-notes/README.md` is the canonical convention**.
One file per release, with a rolling draft renamed at release time:

- Python: draft `RELEASE_NOTES_next.md` -> released `RELEASE_NOTES_v{version}.md`
- C#: draft `RELEASE_NOTES_dotnet-next.md` -> released `RELEASE_NOTES_dotnet-v{version}.md`

The `dotnet-` prefix is what keeps the C# notes from colliding with the Python `RELEASE_NOTES_v*.md`
when both tracks reach the same CalVer number. Content structure:
`## New Features / ## Bug Fixes / ## Performance / ## Breaking Changes` (omit empty sections); past
tense, lead with user impact, include concrete numbers, reference config keys by name.

> [!IMPORTANT]
> **The notes file IS the GitHub Release description - write it for that audience.** On the C# track
> `dotnet-release.yml` publishes `release-notes/RELEASE_NOTES_${tag}.md` verbatim as the Release body, so
> whatever lands in that file is what users read on the Releases page. Two consequences:
>
> - **Rename the draft before tagging.** The workflow resolves the path from the tag and **fails** if the
>   file is missing - after the artifacts have already built.
> - It renders as GitHub-flavoured Markdown, so keep headings/links/code fences valid; the leading `#`
>   title is redundant with the release's own heading but harmless.
>
> On the Python track you create the Release by hand (that is what triggers the PyPI upload), so paste
> the notes in yourself. Backfill an old Release with
> `gh release edit <tag> --notes-file release-notes/RELEASE_NOTES_<tag>.md`.

### C# (.NET) release - the primary track

Two version sources MUST stay in lockstep; `dotnet-release.yml` fails the release if either differs
from the tag:

- `dotnet/Directory.Build.props` `<Version>` - drives every C# assembly version, the CLI's
  `prism --version` (prints the 4-part `X.Y.Z.0`), and provenance `pipeline_version`.
- `dotnet/src/SkylinePrism.App/tool-inf/info.properties` `Version =` - the Skyline tool manifest.

Steps:

1. Finalize `release-notes/RELEASE_NOTES_dotnet-next.md`; `git mv` it to
   `RELEASE_NOTES_dotnet-v{version}.md`, update its heading, **delete every section heading with no
   entries under it** (the draft is seeded with all four, and this file is published verbatim as the
   GitHub Release body), and create a fresh empty `RELEASE_NOTES_dotnet-next.md`.
2. Bump the version to `{version}` in **both** `Directory.Build.props` and `info.properties` (they
   must match each other and the tag).
3. **Run the ship gate** locally:
   `pwsh -File dotnet/build/package-and-verify.ps1 -Configuration Release` (tests -> packages
   `SkylinePrism.zip` -> extracts and launch-verifies the exe). Confirm `prism --version` prints
   `{version}.0`.
4. Commit, open a PR to `main`, let CI go green (`dotnet-ci.yml` + Python `ci.yml`), run
   `/pw-self-review`, then merge with a **merge commit** (`--no-ff`).
5. Tag the merge commit and push the tag:
   ```bash
   git tag dotnet-v{version} origin/main
   git push origin dotnet-v{version}
   ```
   **Pushing the tag both builds the artifacts AND creates the GitHub Release**
   (`.github/workflows/dotnet-release.yml`). Do NOT hand-create the Release; there is no PyPI upload
   for the C# track.

   The Release **body is the notes file**: the workflow reads
   `release-notes/RELEASE_NOTES_${tag}.md` (so tag `dotnet-v26.7.2` -> `RELEASE_NOTES_dotnet-v26.7.2.md`)
   and fails with an explicit message if it is missing. That is why step 1's rename must happen *before*
   tagging - a tag pushed without its notes file aborts the release job after the artifacts have built.
   (Releases up to and including v26.7.1 were published with an empty body, before the workflow checked
   out the repo; they have since been backfilled with `gh release edit --notes-file`.)

Artifacts (7 assets, all **framework-dependent** - the .NET 8 runtime is NOT bundled; users install
it once):

- `SkylinePrism.zip` - the Windows Skyline external tool (needs the .NET 8 **Desktop** Runtime).
- `prism-{win-x64,win-arm64,linux-x64,linux-arm64,osx-x64,osx-arm64}.{zip,tar.gz}` - the CLI, one
  native build per platform (needs the base .NET 8 Runtime).

Framework-dependent is deliberate (small downloads, one shared runtime). Archives are still ~20-45 MB
because they bundle the app's native deps (DuckDB, SkiaSharp), not the runtime. To publish a bare
`prism` CLI for a new OS/arch, add a row to the `cli` matrix in `dotnet-release.yml`.

### Python release - legacy track

Two version sources, **both bumped together** (they have drifted before):

- `pyproject.toml` `version` - the PyPI package version (source of truth for `importlib.metadata`).
- `skyline_prism/__init__.py` `__version__` - what Python `prism --version` prints and what provenance
  records.

Steps:

1. Finalize `RELEASE_NOTES_next.md`; `git mv` to `RELEASE_NOTES_v{version}.md`, update heading,
   **delete every section heading with no entries under it** (the draft is seeded with all four),
   create a fresh `RELEASE_NOTES_next.md`.
2. Bump `pyproject.toml` `version` AND `skyline_prism/__init__.py` `__version__` to `{version}`.
3. Run `pytest tests/ -v` (all pass).
4. Merge to `main`, then `git tag v{version}` and `git push origin main --tags`.
5. **Publish a GitHub Release** for the `v{version}` tag - that is what triggers the PyPI upload via
   `.github/workflows/release.yml` (trusted publishing / OIDC). Note the asymmetry with the C# track:
   here you create the Release to trigger publish; there the workflow creates the Release for you.

### Which track(s) to release

The tracks are independent - release whichever the change affects. A Python-only fix -> Python release
only; a C#-only change -> C# release only; a cross-cutting algorithm change -> release both, each with
its own notes and version bump.

Canonical references: `release-notes/README.md` (notes + step-by-step), `dotnet/README.md` (C#
build/package/CI), and the workflows under `.github/workflows/` (`dotnet-release.yml`, `release.yml`,
`dotnet-ci.yml`, `ci.yml`).

## Repository Information

- **GitHub**: https://github.com/maccoss/skyline-prism
- **Owner**: maccoss (MacCoss Lab, University of Washington)
- **License**: MIT
