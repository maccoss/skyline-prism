# PRISM configuration parameters

Reference for every PRISM configuration key, covering **both** the Python package (`skyline_prism`)
and the C# port (`dotnet/`). The two engines aim for identical numeric results; this table records
where a key exists, its default, and any behavioral difference.

Generate a starting config from either engine:

```bash
# C# port
prism config-template -o config.yaml            # full annotated template
prism config-template --minimal -o config.yaml  # common knobs only

# Python
prism config-template -o config.yaml
prism config-template --minimal -o config.yaml
```

Run with `prism run -i <report.csv> -o <out/> -c config.yaml`.

## Availability legend

| Mark | Meaning |
|------|---------|
| **Both** | Same key in Python and C#, same behavior |
| **C# only** | Exposed in C#; Python hardcodes it or has no such key |
| **Python only** | Python config key **not** implemented in C# |

> **The C# port never silently ignores a config key.** Unrecognized keys print a `WARNING` on stderr
> (`PrismConfig.FindUnknownKeys`), and unsupported *method choices* abort the run (`PrismConfig.Validate`).
> So a Python-only key set in a C# run is always surfaced — see the notes below and
> [`dotnet/PORTING_STATUS.md`](../dotnet/PORTING_STATUS.md) → "Config surface & parity".

---

## `transition_rollup` — transition → peptide (Stage 2)

| Key | Default | Description | Availability |
|-----|---------|-------------|--------------|
| `method` | `median_polish`¹ | `sum`, `median_polish`, `topn`, `consensus`, `library_assist` | **Both** |
| `min_transitions` | `3` | Drop peptides with fewer observed MS2 transitions | **Both** |
| `use_ms1` | `false` | Include MS1 precursor transitions in the rollup | **Both** |
| `topn_count` | `3` | Transitions kept for `method: topn` | **Both** |
| `topn_selection` | `correlation` | `correlation` (needs Shape Correlation) or `intensity` | **Both** |
| `topn_weighting` | `sqrt` | `sqrt` or `sum` | **Both** |
| `consensus_regularization` | `0.1` | Inverse-variance shrinkage for `method: consensus` | **C# only** (Python hardcodes) |
| `library_path` | `null` | Spectral library for `method: library_assist`: **`.blib`** (Skyline) or **`.tsv`** (Carafe/DIA-NN), auto-detected by extension | **Both**² |
| `library_min_fragments` | `3` | Min matched library fragments for a fit | **Both**² |
| `library_mz_tolerance` | `0.02` | m/z tolerance (Da) matching transitions to library | **Both**² |
| `library_outlier_threshold` | `1.0` | Normalized-residual threshold for interference | **Both**² |
| `library_remove_outliers` | `true` | Iteratively drop interference fragments before scaling | **Both**² |
| `library_assist:` *(nested)* | — | `library_path`, `min_matched_fragments`, `mz_tolerance`, `outlier_threshold`, `remove_outliers`, `fitting_method`. C# accepts this and folds it onto the flat `library_*` keys. | **Both** |
| `library_assist.fitting_method` | `median_polish` | `median_polish` only. `least_squares` **aborts** in C#. | Python only (C# errors) |
| `enabled` | `true` | Toggle the transition stage | **Python only** (C# always runs it) |
| `method: adaptive` + `adaptive_rollup:` + `learn_adaptive_weights` | — | Learned transition weights (L-BFGS-B) / precursor to QuantUMS | **Python only** — C# **aborts** on `method: adaptive`; the keys warn as unrecognized |
| `spectral_library_path` / `spectral_library_min_fragments` / `spectral_library_mz_tolerance` | — | Legacy flat aliases | **Python only** (use the C# `library_*` names or the nested block) |

¹ Runtime default with no config block: **Python `median_polish`, C# `sum`** — always set `method` explicitly. Both templates emit `median_polish`.
² Python's canonical form is `spectral_library_*` or the nested `library_assist:` block; C# adds flat `library_*` aliases. The rollup algorithm (median-polish with a library prior) is identical — see `dotnet/PORTING_STATUS.md`.

---

## `global_normalization` — peptide normalization (Stage 2b)

| Key | Default | Description | Availability |
|-----|---------|-------------|--------------|
| `method` | `rt_lowess`³ | `rt_lowess`, `median`, `quantile`, `vsn`, `none` | **Both** |
| `rt_lowess.frac` | `0.3` | LOWESS local-regression window fraction | **Both** |
| `rt_lowess.n_grid_points` | `100` | RT grid points evaluated before interpolation | **Both** |
| `vsn_params.optimize_params` | `false` | Per-sample VSN parameter optimization (Nelder-Mead) | **Python only** — C# runs the unoptimized arcsinh fit (Python's default) |

³ Runtime default with no config block: **Python `median`, C# `rt_lowess`**. Both templates emit `rt_lowess`.

---

## `batch_correction` — ComBat (Stage 2c / 4c)

| Key | Default | Description | Availability |
|-----|---------|-------------|--------------|
| `enabled` | `true` | Master switch (skipped automatically with < 2 batches) | **Both** |
| `method` | `combat` | Only `combat` is implemented. Other values **abort** in C#. | **Both** (C# validates) |
| `reference_anchored` | `false` | Estimate batch effects from reference samples across batches | **Both** |
| `reference_type` | `reference` | Sample type used as the inter-batch reference | **Both** |
| `auto_revert` | `false` | Safety net: if ComBat worsens the control (QC, else reference) median CV by >10%, keep the uncorrected data | **C# only** — Python has the revert only in its legacy `normalize_pipeline`, not its production CLI |
| `peptide_level` | `true` | Apply ComBat at the peptide level | **C# only** (Python always both levels) |
| `protein_level` | `true` | Apply ComBat at the protein level | **C# only** |

---

## `protein_rollup` — peptide → protein (Stage 4)

| Key | Default | Description | Availability |
|-----|---------|-------------|--------------|
| `method` | `median_polish` | `median_polish`, `sum`, `topn`, `maxlfq`, `ibaq` | **Both** |
| `min_peptides` | `3`⁴ | Below this, protein = sum of linear peptide intensity | **Both** |
| `topn.n` | `3` | Peptides averaged for `method: topn` | **Both** |
| `topn.selection` | `median_abundance` | `median_abundance` or `frequency` | **Both** |
| `ibaq.fasta_path` | `null` | FASTA for theoretical peptide counts (falls back to `parsimony.fasta_path`) | **Both** |
| `ibaq.enzyme` | `trypsin` | `trypsin` or `trypsin/p` | **Both** |
| `ibaq.missed_cleavages` | `0` | Missed cleavages for the in-silico digest | **Both** |
| `ibaq.min_peptide_length` | `6` | Min tryptic peptide length counted | **Both** |
| `ibaq.max_peptide_length` | `30` | Max tryptic peptide length counted | **Both** |
| `median_polish.max_iterations` / `median_polish.convergence_tolerance` | `20` / `1e-4` | Median-polish convergence tuning | **Python only** — C# hardcodes these exact Python defaults (`20`, `1e-4`); no divergence unless Python overrides them |
| `shared_peptide_handling` | — | Duplicate of `parsimony.shared_peptide_handling` | **Python only** (set it under `parsimony`) |

⁴ Both engines use `3` at runtime. Python's *template text* shows `2`; the runtime default is `3`.

---

## `protein_normalization` — Stage 4b

| Key | Default | Description | Availability |
|-----|---------|-------------|--------------|
| `method` | `median` | `median` or `none` | **Both** |

---

## `parsimony` — protein grouping (Stage 3)

| Key | Default | Description | Availability |
|-----|---------|-------------|--------------|
| `enabled` | `true` | When false, each accession is its own group | **Both** |
| `fasta_path` | `null` | FASTA for substring-based peptide→protein mapping; null uses the Skyline Protein Accession column | **Both** |
| `shared_peptide_handling` | `all_groups` | `all_groups`, `unique_only`, `razor` | **Both** |

---

## `sample_annotations` — sample-type detection

| Key | Default | Description | Availability |
|-----|---------|-------------|--------------|
| `reference_pattern` | `["-Pool-", "-Pool_", "_Pool_", "CommercialPool", "Ref", "Reference"]` | **Fallback** substrings for reference samples | **Both**⁵ |
| `qc_pattern` | `["-QC-", "-QC_", "_QC_", "QC", "Control", "StudyPool", "Quality Control"]` | **Fallback** substrings for QC samples | **Both**⁵ |
| `experimental_pattern` | — | Explicit experimental-sample patterns | **Python only** — C# treats anything not reference/qc as experimental |

⁵ Sample type is taken from the Replicates **"Sample Type" column first** (`Standard` → reference, `Quality Control` → qc, `Unknown` → experimental). The patterns are a **fallback**, matched (case-sensitive) against the replicate/sample name only for replicates with no Sample Type annotation. C# applies these defaults as an always-on fallback; Python applies patterns only when the `sample_annotations` block is present (the generated template includes it, so both agree).

---

## `sample_outlier_detection` — low-signal samples (Stage 2a)

| Key | Default | Description | Availability |
|-----|---------|-------------|--------------|
| `enabled` | `true` | Detect abnormally low-signal samples (one-sided, linear scale) | **Both** |
| `action` | `report` | `report` or `exclude` | **Both** |
| `method` | `iqr` | `iqr` or `fold_median` | **Both** |
| `iqr_multiplier` | `1.5` | `method: iqr` — flag samples below Q1 − k·IQR | **Both** |
| `fold_threshold` | `0.1` | `method: fold_median` — flag samples below k·median | **Both** |

---

## `metadata` / `data` — column names

C# reads the batch and sample-type column names under **`metadata:`**; Python reads them (plus other
column-name overrides) under **`data:`**. This is a path difference — a Python `data:` block is not read
by C# (and warns).

| C# key | Python key | Default | Description | Availability |
|--------|-----------|---------|-------------|--------------|
| `metadata.batch_column` | `data.batch_column` | `null` (auto-detect) | Column carrying the batch/plate label | Both (different path) |
| `metadata.sample_type_column` | `data.sample_type_column` | `null` (auto-detect) | Sample Type column | Both (different path) |
| — | `data.abundance_column`, `rt_column`, `peptide_column`, `protein_column`, `protein_name_column`, `sample_column`, `transition_column`, `precursor_column`, `fragment_column` | — | Override each input column name | **Python only** — C# auto-detects columns (`SkylineColumns`) and cannot override them |

---

## `processing` — parallelism / memory

| Key | Default | Description | Availability |
|-----|---------|-------------|--------------|
| `n_workers` | `0` | `0` = all cores, `1` = serial, `N` = cap at N | **Both** |
| `peptide_batch_size` | C# `2000`, Python `1000` | Peptides buffered per streamed row group (performance only) | **Both** (default differs; no numeric effect) |

---

## `batch_estimation` — fallback batch assignment

Used only when neither metadata nor the Source Document distinguishes batches.

| Key | Default | Description | Availability |
|-----|---------|-------------|--------------|
| `method` | `auto` | `auto`, `gap`, `fixed`, `source`, `none` | **Both** |
| `gap_iqr_multiplier` | `1.5` | `auto`/`gap` — split when an acquisition-time gap exceeds k·IQR | **Both** |
| `n_batches` | `null` | `method: fixed` — split into exactly N batches | **Both** |
| `min_samples_per_batch` / `max_samples_per_batch` | — | Bounds on estimated batch size | **Python only** (also absent from Python's template) |

---

## `output` — Stage 5

| Key | Default | Description | Availability |
|-----|---------|-------------|--------------|
| `format` | `parquet` | `parquet`, `csv`, `tsv` | **Both** |
| `include_residuals` | `true` | Write `peptide_residuals.parquet` for outlier/proteoform analysis | **Both** |
| `compress` | — | Compress output files | **Python only** (parquet is already compressed) |

---

## `qc_report` — Stage 5b

| Key | Default | Description | Availability |
|-----|---------|-------------|--------------|
| `enabled` | `true` | Generate `qc_report.html` | **Both** |
| `save_plots` | `true` | Also write PNGs to `qc_plots/` | **Both** |
| `filename` | — | Report filename | **Python only** (C# hardcodes `qc_report.html`) |
| `embed_plots` | — | Embed (base64) vs link external PNGs | **Python only** — C# always base64-embeds (self-contained HTML) |
| `plots.intensity_distribution` / `pca_comparison` / `control_correlation` / `cv_distribution` / `boxplot_comparison` | — | Per-plot on/off toggles | **Python only** — C# emits a fixed plot set |

---

## Default-value differences (with no config block)

If you run with **no** `-c config.yaml`, the two engines fall back to slightly different defaults. The
generated templates avoid this by writing explicit values; if you write your own config, set these
explicitly to stay identical across engines.

| Key | Python default | C# default | Effect |
|-----|----------------|-----------|--------|
| `transition_rollup.method` | `median_polish` | `sum` | **Numeric** — set it explicitly |
| `global_normalization.method` | `median` | `rt_lowess` | **Numeric** — set it explicitly |
| `processing.peptide_batch_size` | `1000` | `2000` | Performance only |
| `protein_rollup.min_peptides` | `3` (runtime) | `3` | None (Python *template text* says 2) |

---

*This document is maintained alongside [`dotnet/PORTING_STATUS.md`](../dotnet/PORTING_STATUS.md). When a
parameter is added, ported, or deliberately not ported, update both.*
