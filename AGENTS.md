# AGENTS.md - AI Agent Guidelines for Skyline-PRISM

This document provides context and guidelines for AI agents working on the Skyline-PRISM project.

## Project Overview

**Skyline-PRISM** (Proteomics Reference-Integrated Signal Modeling) is a Python package for RT-aware normalization of LC-MS proteomics data exported from [Skyline](https://skyline.ms), with robust protein quantification using Tukey median polish.

### Key Concepts

- **Reference-anchored correction**: Uses inter-experiment reference samples (e.g., commercial plasma/CSF pool) to learn and correct RT-dependent technical variation
- **Dual-control validation**: Uses intra-experiment pool samples to validate corrections without overfitting
- **Peptide-first normalization**: Normalizes at peptide level before rolling up to proteins
- **Robust protein quantification**: Tukey median polish handles outlier peptides without pre-identification

## Project Structure

```
skyline-prism/
├── skyline_prism/           # Main Python package
│   ├── __init__.py          # Package exports
│   ├── cli.py               # Command-line interface (entry point: `prism`)
│   ├── data_io.py           # Skyline report loading and merging
│   ├── normalization.py     # RT-aware correction and global normalization
│   ├── parsimony.py         # Protein grouping and shared peptide handling
│   ├── rollup.py            # Peptide → Protein rollup (median polish, etc.)
│   ├── transition_rollup.py # Transition → Peptide rollup (quality-weighted)
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
- Processing pipeline stages
- Configuration parameters

### config_template.yaml
Comprehensive configuration file with all options documented. When adding new configuration options, add them here with comments.

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

The package exports are defined in `skyline_prism/__init__.py`. If you add new public functions, export them there.

## Important Notes

- **Skyline** is an external tool (https://skyline.ms) - we process its exports, we don't modify Skyline itself
- **Sample types**: `experimental`, `pool`, `reference` - these have specific meanings in the normalization workflow
- **Column naming**: Internal column names differ from Skyline export names - see `SKYLINE_COLUMN_MAP` in data_io.py
- **Log scale**: Most operations work on log2-transformed abundances

## Repository Information

- **GitHub**: https://github.com/maccoss/skyline-prism
- **Owner**: maccoss (MacCoss Lab, University of Washington)
- **License**: MIT
