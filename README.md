# Skyline-PRISM

[![dotnet CI](https://github.com/maccoss/skyline-prism/actions/workflows/dotnet-ci.yml/badge.svg)](https://github.com/maccoss/skyline-prism/actions/workflows/dotnet-ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![.NET 8](https://img.shields.io/badge/.NET-8.0-512BD4.svg)](https://dotnet.microsoft.com/)

**PRISM** (Proteomics Reference-Integrated Signal Modeling) normalizes transition-level LC-MS
proteomics data exported from [Skyline](https://skyline.ms) and produces robust peptide- and
protein-level quantities. It uses Tukey median polish for outlier-tolerant rollups, retention-time-aware
normalization, and ComBat (optionally reference-anchored) batch correction, and reports reference/QC
sample CVs before and after each correction.

PRISM ships two ways to run the same pipeline:

- **`prism` CLI** — a cross-platform command-line tool (Windows, Linux, macOS).
- **Skyline external tool** — a Windows GUI that drives a running Skyline over JSON-RPC to export the
  report and run the pipeline, then shows interactive QC plots.

> This repository also contains the original **Python** implementation, which remains the reference.
> See **[README-python.md](README-python.md)** for the `skyline-prism` PyPI package. The C# tools
> reproduce the Python pipeline's numeric results (exact parity on the deterministic stages).

## What it does

- **Robust rollups** — Tukey median polish (default) for transition→peptide and peptide→protein
  aggregation; automatically down-weights outliers without pre-filtering. Also sum, top-N, consensus,
  library-assisted (transition) and sum / top-N / maxLFQ / iBAQ (protein).
- **RT-aware normalization** — retention-time LOWESS normalization (default), plus median, quantile,
  VSN, or none.
- **Batch correction** — full empirical-Bayes ComBat, applied at the peptide and protein level; an
  opt-in reference-anchored variant estimates each batch's effect from inter-experiment reference
  samples across batches.
- **Protein parsimony** — indistinguishable-protein grouping, subset elimination, and shared-peptide
  handling (`all_groups`, `unique_only`, `razor`); mapping from Skyline's protein column or a FASTA.
- **Dual-control QC** — a self-contained HTML report with before/after intensity, PCA, and CV plots,
  a control-correlation heatmap, RT diagnostics, and a pass/fail validation banner.

Input peak areas are **linear**; the final `corrected_peptides` / `corrected_proteins` matrices are
written back in **linear** scale (all internal processing is on log2).

## Install

### CLI

The `prism` CLI ships as a **self-contained** archive — no .NET install required. Download the one
for your platform from the [Releases](https://github.com/maccoss/skyline-prism/releases) page and put
its folder on your `PATH` (the extracted folder holds `prism` plus its runtime, so keep it together).

**Linux / WSL2**
```bash
curl -L https://github.com/maccoss/skyline-prism/releases/latest/download/prism-linux-x64.tar.gz | tar xz -C "$HOME/.local"
echo 'export PATH="$HOME/.local/prism-linux-x64:$PATH"' >> ~/.bashrc && source ~/.bashrc
prism --version
```
QC plots need the system font libraries: `sudo apt install libfontconfig1 libfreetype6`.

**Windows (PowerShell)** — download `prism-win-x64.zip` from Releases, then:
```powershell
Expand-Archive prism-win-x64.zip -DestinationPath $env:LOCALAPPDATA\Programs
# add  %LOCALAPPDATA%\Programs\prism-win-x64  to your PATH (System > Environment Variables)
prism --version
```

**macOS** — `prism-osx-arm64.tar.gz` (Apple Silicon) or `prism-osx-x64.tar.gz` (Intel):
```bash
curl -L https://github.com/maccoss/skyline-prism/releases/latest/download/prism-osx-arm64.tar.gz | tar xz -C "$HOME/.local"
echo 'export PATH="$HOME/.local/prism-osx-arm64:$PATH"' >> ~/.zshrc && source ~/.zshrc
xattr -dr com.apple.quarantine "$HOME/.local/prism-osx-arm64"   # if Gatekeeper blocks the unsigned binary
prism --version
```

**Build from source** instead (requires the [.NET 8 SDK](https://dotnet.microsoft.com/download)):
```bash
git clone https://github.com/maccoss/skyline-prism
cd skyline-prism/dotnet
dotnet build SkylinePrism.CrossPlatform.slnf -c Release
# the CLI is src/SkylinePrism.Cli/bin/Release/net8.0/prism(.exe)
```

### Skyline external tool (Windows)

Download `SkylinePrism.zip` from [Releases](https://github.com/maccoss/skyline-prism/releases) (or
build it — see below) and install it in Skyline via **Tools → Tool Store → Install from file**. The
tool appears under the Tools menu and connects to the running document.

## Quick start (CLI)

```bash
# 1. Generate a configuration template and edit it to taste
prism config-template -o config.yaml

# 2. Run the full pipeline on one or more Skyline transition reports
prism run -i report.csv -o output/ -c config.yaml

# ...or merge several plates, with a Replicates metadata report
prism run -i plate1.csv plate2.csv -o output/ -c config.yaml -m replicates.csv
```

`prism run` writes the corrected matrices, intermediate parquet, protein groups, provenance, and a QC
report to the output directory (see [Output](#output)). Run `prism --help` or `prism <command> --help`
for details.

### Commands

| Command | Purpose |
|---|---|
| `prism run` | Run the full pipeline (rollup → normalize → batch-correct → QC) |
| `prism merge` | Merge Skyline transition reports into one sorted parquet |
| `prism qc` | (Re)generate `qc_report.html` from an existing output directory |
| `prism compare` | Compare control-sample CVs between two runs |
| `prism config-template` | Emit an annotated configuration template (`--minimal` for common options) |

## Pipeline

```
Merge reports  →  Transition→Peptide rollup  →  Peptide normalization  →  Peptide ComBat
                                                                                │
                                            ┌───────────────────────────────────┤
                                            ▼                                   ▼
                              Protein parsimony + rollup                 corrected_peptides
                                            │
                              Protein normalization  →  Protein ComBat  →  corrected_proteins  →  QC report
```

Batch correction is applied at the reporting level (peptide and protein), not before rollup. RT-LOWESS
normalization and reference sample handling are learned from control samples only.

## Configuration

`prism config-template` emits a commented YAML file with every option and its default. Key sections:

```yaml
transition_rollup:    # sum | median_polish (default) | topn | consensus | library_assist
global_normalization: # rt_lowess (default) | median | quantile | vsn | none
batch_correction:     # ComBat; peptide/protein levels; reference_anchored: true|false
parsimony:            # fasta_path; shared_peptide_handling: all_groups | unique_only | razor
protein_rollup:       # median_polish (default) | sum | topn | maxlfq | ibaq
qc_report:            # HTML report + plots
```

See **[docs/parameters.md](docs/parameters.md)** for the full parameter reference — every key, its
default, and whether it is available in the Python package, the C# port, or both.

A run records its full configuration to `parameters.json`; `prism run --from-provenance
out/parameters.json` reproduces an earlier run's settings.

## Output

| File | Scale | Contents |
|---|---|---|
| `corrected_peptides.parquet` | linear | Normalized, batch-corrected peptide quantities |
| `corrected_proteins.parquet` | linear | Normalized, batch-corrected protein quantities |
| `peptides_rollup.parquet` | log2 | Raw peptide abundances from the transition rollup |
| `proteins_raw.parquet` | log2 | Raw protein abundances from the peptide rollup |
| `protein_groups.csv` | — | Protein group / parsimony assignments |
| `sample_metadata.csv` | — | Resolved sample types and batches |
| `parameters.json` | — | Full run configuration (provenance) |
| `qc_report.html` + `qc_plots/` | — | QC report and diagnostic plots |

## Build the Skyline external tool

```bash
dotnet msbuild dotnet/build/package.proj /p:Configuration=Release
# -> dotnet/publish/SkylinePrism.zip  (SkylinePrism.exe + tool-inf/ + runtime DLLs)
```

## Documentation

- **[dotnet/README.md](dotnet/README.md)** — building, testing, project layout, and CI for the C# code.
- **[SPECIFICATION.md](SPECIFICATION.md)** — the authoritative algorithm/format specification.
- **[docs/](docs/)** — [parameters](docs/parameters.md), [methods](docs/methods.md), [output files](docs/output_files.md), [protein parsimony](docs/parsimony.md), and [building a Skyline external tool](docs/skyline-external-tools.md).
- **[README-python.md](README-python.md)** — the original Python `skyline-prism` package.

## License

MIT — see [LICENSE](LICENSE). Skyline is an external tool (<https://skyline.ms>); PRISM processes its
exports and does not modify Skyline.
