# AGENTS.md - AI Agent Guidelines for Skyline-PRISM

This document provides context and guidelines for AI agents working on the Skyline-PRISM project.

## Project Overview

**Skyline-PRISM** (Proteomics Reference-Integrated Signal Modeling) is a Python package for RT-aware normalization of LC-MS proteomics data exported from [Skyline](https://skyline.ms), with robust protein quantification using Tukey median polish.

### Key Concepts

- **Reference-anchored correction**: Uses inter-experiment reference samples (e.g., commercial plasma/CSF pool) to learn and correct RT-dependent technical variation via spline-based modeling
- **Dual-control validation**: Uses intra-experiment pool samples to validate corrections without overfitting
- **Tukey median polish as default**: Both transition→peptide and peptide→protein rollups use median polish by default for robust outlier handling
- **Two-arm pipeline**: After RT normalization, the pipeline splits - batch correction is applied at the reporting level (peptide or protein)
- **ComBat batch correction**: Full empirical Bayes implementation for removing batch effects

### Processing Pipeline

```
Transitions → [Median Polish] → Peptides → [RT Correction] → 
    ├─→ [Batch Correction] → Normalized Peptides (peptide output)
    └─→ [Median Polish] → Proteins → [Batch Correction] → Normalized Proteins (protein output)
```

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
│   ├── transition_rollup.py # Transition → Peptide rollup (median polish, quality-weighted)
│   └── validation.py        # QC metrics and reporting
├── tests/                   # Unit tests (pytest)
│   ├── test_data_io.py
│   ├── test_parsimony.py
│   └── test_rollup.py
├── SPECIFICATION.md         # Detailed technical specification
├── README.md                # User-facing documentation
├── config_template.yaml     # Configuration file template
├── pyproject.toml           # Package configuration and dependencies
└── .venv/                   # Virtual environment (not in git)
```

## Key Algorithms

### Tukey Median Polish (Default for Rollups)

Used for both transition→peptide and peptide→protein rollups. Decomposes a matrix into:
- Row effects (transition/peptide ionization efficiency)
- Column effects (sample abundance - **this is the output**)
- Residuals (noise/outliers - **preserved for biological analysis**)

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

## Development Guidelines

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
- **ruff** for linting
- **mypy** for type checking

```bash
# Format code
black skyline_prism/

# Lint
ruff check skyline_prism/

# Type check
mypy skyline_prism/
```

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
Comprehensive configuration file with all options documented. Key sections:
- `transition_rollup`: Transition→peptide rollup (method: median_polish, quality_weighted, sum)
- `rt_correction`: RT-aware normalization (method: spline)
- `batch_correction`: ComBat settings (method: combat)
- `protein_rollup`: Peptide→protein rollup (method: median_polish)
- `parsimony`: Shared peptide handling (all_groups, unique_only, razor)

### batch_correction.py
Full ComBat implementation with:
- `combat()`: Main function for wide-format data
- `combat_from_long()`: Wrapper for long-format data (PRISM pipeline format)
- `combat_with_reference_samples()`: Automatic evaluation using reference/pool CVs
- `evaluate_batch_correction()`: Compare before/after metrics

### pyproject.toml
Package metadata and dependencies. Contains:
- Package name: `skyline-prism`
- CLI entry point: `prism` → `skyline_prism.cli:main`
- Dependencies (core, dev, viz)

## CLI Commands

The package provides a `prism` CLI with these subcommands:

```bash
# Merge Skyline reports into unified parquet
prism merge report1.csv report2.csv -o data.parquet -m metadata.tsv

# Run normalization pipeline
prism normalize -i data.parquet -o normalized.parquet -c config.yaml

# Roll up peptides to proteins
prism rollup -i normalized.parquet -o proteins.parquet -g groups.tsv

# Validate normalization quality
prism validate --before data.parquet --after normalized.parquet --report qc.html
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

## Important Notes

- **Skyline** is an external tool (https://skyline.ms) - we process its exports, we don't modify Skyline itself
- **Sample types**: `experimental`, `pool`, `reference` - these have specific meanings in the normalization workflow
- **Column naming**: Internal column names differ from Skyline export names - see `SKYLINE_COLUMN_MAP` in data_io.py
- **Log scale**: Most operations work on log2-transformed abundances
- **Median polish is default**: For both transition→peptide and peptide→protein rollups
- **Two-arm pipeline**: Batch correction happens at the reporting level (peptide or protein), not before rollup

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
