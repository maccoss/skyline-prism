# PRISM configuration parameters

Reference for every PRISM configuration key: what it does and what it defaults to.

Generate a starting config:

```bash
prism config-template -o config.yaml            # full annotated template
prism config-template --minimal -o config.yaml  # common knobs only
```

Run with `prism run -i <report.csv> -o <out/> -c config.yaml`.

> **PRISM never silently ignores a config key.** Unrecognized keys print a `WARNING` on stderr
> (`PrismConfig.FindUnknownKeys`), and unsupported *method choices* abort the run
> (`PrismConfig.Validate`) — so a typo, or a key left over from the retired Python engine, is
> always surfaced rather than quietly doing nothing.

---

## `transition_rollup` — transition → peptide (Stage 2)

| Key | Default | Description |
|-----|---------|-------------|
| `method` | `median_polish`¹ | `sum`, `median_polish`, `topn`, `consensus`, `library_assist` |
| `min_transitions` | `3` | Drop peptides with fewer observed MS2 transitions |
| `use_ms1` | `false` | Include MS1 precursor transitions in the rollup |
| `topn_count` | `3` | Transitions kept for `method: topn` |
| `topn_selection` | `correlation` | `correlation` (needs Shape Correlation) or `intensity` |
| `topn_weighting` | `sqrt` | `sqrt` or `sum` |
| `consensus_regularization` | `0.1` | Inverse-variance shrinkage for `method: consensus` |
| `library_path` | `null` | Spectral library for `method: library_assist`: **`.blib`** (Skyline) or **`.tsv`** (Carafe/DIA-NN), auto-detected by extension |
| `library_min_fragments` | `3` | Min matched library fragments for a fit |
| `library_mz_tolerance` | `0.02` | m/z tolerance (Da) matching transitions to library |
| `library_outlier_threshold` | `1.0` | Normalized-residual threshold for interference |
| `library_remove_outliers` | `true` | Iteratively drop interference fragments before scaling |
| `library_fitting_method` | `median_polish` | `median_polish` or `least_squares`. `least_squares` **aborts** in C#. |
| `library_assist:` *(nested)* | — | `library_path`, `min_matched_fragments`, `mz_tolerance`, `outlier_threshold`, `remove_outliers`, `fitting_method`. Folded onto the flat `library_*` keys (nested wins).² |

¹ The built-in default with no config block is **`sum`**, while the generated templates emit `median_polish` — always set `method` explicitly rather than relying on either.
² Precedence when a setting is given more than once: nested `library_assist:` > flat `library_*`. An empty `library_assist:` block is treated as absent.

---

## `global_normalization` — peptide normalization (Stage 2b)

| Key | Default | Description |
|-----|---------|-------------|
| `method` | `rt_lowess`³ | `rt_lowess`, `median`, `quantile`, `vsn`, `none` |
| `rt_lowess.frac` | `0.3` | LOWESS local-regression window fraction |
| `rt_lowess.n_grid_points` | `100` | RT grid points evaluated before interpolation |

³ The built-in default with no config block is **`rt_lowess`**, which is also what the generated templates emit.

---

## `batch_correction` — ComBat (Stage 2c / 4c)

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Master switch (skipped automatically with < 2 batches) |
| `method` | `combat` | Only `combat` is implemented. Other values **abort** in C#. |
| `reference_anchored` | `false` | Estimate batch effects from reference samples across batches. **Measured worse than standard ComBat on a real 4-batch cohort with one reference per batch** (held-out QC CV 20.3% -> 25.7%). Prefer standard ComBat until that is understood, and set `auto_revert: true`, which caught and reverted it |
| `reference_type` | `reference` | Sample type used as the inter-batch reference |
| `auto_revert` | `false` | Safety net: if ComBat worsens the control (QC, else reference) median CV by >10%, keep the uncorrected data |
| `peptide_level` | `true` | Apply ComBat at the peptide level. Corrects the peptide output only - the protein arm branches before this, so it is never applied to proteins |
| `protein_level` | `true` | Apply ComBat at the protein level. **This is the only batch correction the protein output receives**: since dotnet-v26.15.0 the protein arm consumes the normalized, pre-ComBat peptide matrix, so `protein_level: false` means `corrected_proteins` is **not batch-corrected at all** - previously it inherited the peptide correction through its inputs. Changed silently and by design; see CLAUDE.md "Batch correction at reporting level" |

---

## `protein_rollup` — peptide → protein (Stage 4)

| Key | Default | Description |
|-----|---------|-------------|
| `method` | `median_polish` | `median_polish`, `sum`, `topn`, `maxlfq`, `ibaq` |
| `min_peptides` | `3`⁴ | Below this, protein = sum of linear peptide intensity |
| `topn.n` | `3` | Peptides averaged for `method: topn` |
| `topn.selection` | `median_abundance` | `median_abundance` or `frequency` |
| `ibaq.fasta_path` | `null` | FASTA for theoretical peptide counts (falls back to `parsimony.fasta_path`) |
| `ibaq.enzyme` | `trypsin` | `trypsin` or `trypsin/p` |
| `ibaq.missed_cleavages` | `0` | Missed cleavages for the in-silico digest |
| `ibaq.min_peptide_length` | `6` | Min tryptic peptide length counted |
| `ibaq.max_peptide_length` | `30` | Max tryptic peptide length counted |

⁴ The runtime default is `3`.

---

## `marker_normalization` — normalize to a set of proteins (Stage 5a)

Estimates one per-sample score from how a set of marker proteins move together, then removes from
every peptide and protein the part that tracks it. It answers **"what changed per unit of the marked
material"** rather than "what changed in whatever was captured" — the question a capture-based
experiment (EV enrichment, for instance) otherwise cannot separate, because loading normalization
makes total signal equal by construction.

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Off unless asked for — it changes every reported abundance |
| `protein_list` | — | Name of the marker list: one of your saved lists, or a shipped panel (`EV markers`, `Glomerulus`). `Tubular contamination` also ships, but it is a readout for the Dynamic Range plot — normalizing to it would remove the contamination it exists to reveal |
| `protein_list_file` | — | A file of members instead (one identifier per line, or a CSV's first column). Wins over `protein_list`, and is the reproducible form — a name depends on the machine's saved lists, a path does not |
| `method` | `pc1` | `pc1` (first principal component of the z-scored markers) or `mean` |

**Where it runs, and why there.** After the ordinary normalization, never instead of it, and the score
is computed at the **protein** level and applied to **both** outputs:

- The score must come from data whose per-sample loading is already removed. On raw abundances PC1
  loads on injection volume, and residualizing then re-does the loading step using a handful of
  proteins' worth of noise.
- How much marked material a sample contributed is a property of *the sample*, not of the table being
  analyzed, so re-estimating it from peptides would mostly re-measure the same quantity with more noise.

**Why PC1 rather than the mean.** Markers need not move as one block. On the cohort the shipped EV
panel comes from, PC1 explains 70.4% of marker variance and four of the eighteen (`CD81`, `SDCBP`,
`ANXA2`, `ANXA6`) load with the *opposite* sign — a mean partially cancels and blunts the estimate
(the two scores correlate at r = +0.95, with PC1 the more conservative). PC1 weights each marker by
its contribution and keeps the sign structure. The score's sign is oriented so higher always means
more marked material.

See [`methods.md`](methods.md#marker-protein-normalization) for the algorithm, the diagnostics, and
how this relates to published methods (RUV, eigengenes, SVA/EigenMS).

**What you get.** Both corrected outputs are rewritten with the score axis removed, each feature
keeping its own abundance level (only the score-dependent part is taken out, so the values stay on the
scale everything else expects). `marker_normalization.csv` records the per-sample score and the marker
loadings. The markers themselves stay in the outputs, flagged `normalization_marker` — their residual
is near zero by construction, so **exclude them from any result you read off these files**; a test
among them is circular.

Fewer than 3 quantified markers is an error, not a silent fallback. A PC1 explaining under 40% of
marker variance is a warning: the markers are not moving together and the score is a weak summary.

---

## `protein_normalization` — Stage 4b

| Key | Default | Description |
|-----|---------|-------------|
| `method` | `median` | `median` or `none` |

---

## `parsimony` — protein grouping (Stage 3)

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | When false, each accession is its own group |
| `fasta_path` | `null` | FASTA for enzyme-aware peptide→protein mapping; null uses the Skyline Protein Accession column |
| `shared_peptide_handling` | `all_groups` | `all_groups`, `unique_only`, `razor` |
| `enzyme` | `trypsin` | Digestion enzyme for the FASTA-mapping terminus check (ignored when `fasta_path` is null): `trypsin` (not before P), `trypsin/p` (before P too, e.g. DIA-NN), `lysc`, `lysn`, `argc`, `aspn`, `gluc`, `chymotrypsin`, `nonspecific`. The Skyline external tool overrides this from the document's digestion settings |
| `enzyme_specificity` | `full` | Terminus requirement for FASTA membership: `full` (both termini cleavage-consistent — removes phantom homolog assignments), `semi` (either), `none` (legacy pure substring) |

---

## `sample_annotations` — sample-type detection

| Key | Default | Description |
|-----|---------|-------------|
| `reference_pattern` | `["-Pool-", "-Pool_", "_Pool_", "CommercialPool", "Ref", "Reference"]` | **Fallback** substrings for reference samples |
| `qc_pattern` | `["-QC-", "-QC_", "_QC_", "QC", "Control", "StudyPool", "Quality Control"]` | **Fallback** substrings for QC samples |

⁵ Sample type is taken from the Replicates **"Sample Type" column first** (`Standard` → reference, `Quality Control` → qc, `Unknown` → experimental). The patterns are a **fallback**, matched (case-sensitive) against the replicate/sample name only for replicates with no Sample Type annotation. The defaults are an always-on fallback, whether or not a `sample_annotations` block is present.

---

## `sample_outlier_detection` — low-signal samples (Stage 2a)

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Detect abnormally low-signal samples (one-sided, linear scale) |
| `action` | `report` | `report` or `exclude` |
| `method` | `iqr` | `iqr` or `fold_median` |
| `iqr_multiplier` | `1.5` | `method: iqr` — flag samples below Q1 − k·IQR |
| `fold_threshold` | `0.1` | `method: fold_median` — flag samples below k·median |

---

## `data` / `metadata` — column names

Any value set in the **`data:`** section wins over auto-detection. Column matching is robust to
**case, spaces, and underscores**, so one config works for both the invariant/parquet export
(`PeptideModifiedSequenceUnimodIds`) and the English/CSV export (`Peptide Modified Sequence`).
`batch_column` / `sample_type_column` are accepted under **both** `data:` and `metadata:`, with
`data:` taking precedence.

| Key | Default | Description |
|-----|---------|-------------|
| `data.peptide_column` | auto (prefers `PeptideModifiedSequenceUnimodIds`, the Skyline-PRISM.skyr column) | Peptide modified-sequence column |
| `data.protein_column` / `data.protein_name_column` | auto | Protein accession / name columns |
| `data.abundance_column` | auto (`Area`) | Peak-area column |
| `data.rt_column` | auto (`Retention Time`) | Retention-time column |
| `data.sample_column` | auto (`Replicate Name`) | Replicate/sample column |
| `data.transition_column` | auto (`Fragment Ion`) | Transition/fragment column |
| `data.batch_column` / `data.sample_type_column` | auto | Also accepted under `metadata:` (data wins) |
| `data.precursor_column` / `data.fragment_column` | — | Parsed for compatibility; C# auto-detects precursor/product charge |
| `metadata.batch_column` / `metadata.sample_type_column` | auto | C# location for the above two |

---

## `processing` — parallelism / memory

| Key | Default | Description |
|-----|---------|-------------|
| `n_workers` | `0` | `0` = all cores, `1` = serial, `N` = cap at N |
| `peptide_batch_size` | `2000` | Peptides buffered per streamed row group (performance only) |
| `merge_memory_mb` | `0` | Ceiling on DuckDB's buffer pool, in MB. `0` = engine default. Beyond the ceiling work spills to scratch, so a smaller value is slower, never wrong (performance only). Lower it to leave room for Skyline alongside |

---

## `batch_estimation` — fallback batch assignment

Used only when neither metadata nor the Source Document distinguishes batches.

| Key | Default | Description |
|-----|---------|-------------|
| `method` | `none` | `none`, `auto`, `gap`, `fixed`, `source`. **Off by default**: guessing batches from acquisition-time gaps cannot tell a real plate boundary from an ordinary pause, and a wrong guess makes ComBat correct between batches that do not exist |
| `gap_iqr_multiplier` | `1.5` | `auto`/`gap` — split when an acquisition-time gap exceeds k·IQR |
| `n_batches` | `null` | `method: fixed` — split into exactly N batches |

---

## `output` — Stage 5

| Key | Default | Description |
|-----|---------|-------------|
| `format` | `parquet` | `parquet`, `csv`, `tsv` |
| `include_residuals` | `true` | Write median-polish residuals for outlier/proteoform analysis: `peptides_rollup_residuals.parquet` (per transition) and `proteins_raw_residuals.parquet` (per peptide), each written only when its stage uses `median_polish`. C# additionally writes `corrected_peptides_residuals.parquet` / `corrected_proteins_residuals.parquet` - the same residuals rescaled by the ComBat that applied to the output they accompany, so they are comparable across batches |

---

## `qc_report` — Stage 5b

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Generate `qc_report.html` |
| `save_plots` | `true` | Also write PNGs to `qc_plots/` |
| `ms2_signal.enabled` | `false` | Compute MS2 signal accounting: per replicate, how much integrated MS2 signal the run assigns to a peptide. Off by default because it costs one extra streaming pass over the merged table plus an ORDER BY the sample id, comparable to Stage 2. Results are cached as `ms2_signal_accounting.parquet`, so `prism qc -d` replots without recomputing |
| `ms2_signal.measure` | `"signal"` | `signal` or `ions`. **`ions` needs columns PRISM does not export by default, because they are ruinously expensive - see the measurement below.** **signal** sums integrated peak areas against the acquired total ion current - available from any export, but the two sides are not the same quantity and Skyline's `Area` is background-subtracted where a TIC is not, so both gaps need correcting and neither is visible in the answer. **ions** sums Skyline's own ion counts (intensity x injection time per spectrum, across the peak): both sides are counts of ions, neither is background subtracted, and no unit or background correction is needed at all. Ions need an export carrying the LC Peak ion-count columns |
| `ms2_signal.extraction_tolerance` | `"10 ppm"` | The +/- m/z range Skyline extracted each product ion over, which decides when two co-isolated peptides' fragments are the same detector counts. Write it as `"10 ppm"` or `"0.4 m/z"`. The value in force is printed in the log and named on the plot; the Skyline external tool overrides it from the document's own `transition_full_scan` settings |
| `ms2_signal.isolation_scheme` | *(auto)* | Which scheme from `isolation_schemes.xml` to account against, by name. Blank picks it when the file defines exactly one; with several, the accounting is skipped until one is named. Never guessed - fragments in different isolation windows never share signal |
| `ms2_signal.protein_lists` | *(none)* | Saved or shipped protein-list names to account for, one line each on the plot. Lists nest inside the assigned total and may overlap each other |
| `ms2_signal.protein_list_files` | *(none)* | Member files, for lists not saved on this machine - the reproducible form |

**What the accounting measures.** Summing transition areas over-counts: a DIA isolation window
co-isolates tens of peptides, and two fragments whose extraction windows overlap are the same detector
counts credited twice. Every total is therefore a **union** over regions of MS2 signal space -
(isolation window, extraction window, integration bounds) - counted once. The naive sum is kept
alongside so the report can say how much double counting was removed.

**What it does not measure.** The *acquired* MS2 total needs the instrument data files and cannot come
from any Skyline export - `TicArea` is one value per replicate and is MS1 by construction. The plot's
bars are the signal Skyline integrated for the document's targets, and are labelled as such.

**Gross signal, not net.** Skyline's `Area` is background-SUBTRACTED (its own test asserts that
integrating without background gives `Area + BackgroundArea`), whereas an acquired total ion current
includes background. The accounting therefore sums `Area + Background`, so numerator and denominator
are both gross. Quantification is unaffected and keeps the net area - adding background back would put
detector baseline into every abundance. An export without the `Background` column still works, on net
area, and the log says so, because the fraction is then an under-estimate.

**Ions are not exported by default, and the reason is cost.** Measured on the 6.5 GB FLARE document
(46M transition rows, 93 replicates) with SkylineCmd: a three-column report exported in **9.5 minutes**,
while the same report plus five `LC Peak` ion-count columns was 13% done after **63 minutes**, writing
at 4.5 MB/min against the baseline's 122 - **27x slower per byte** - projecting to **9-13 hours**,
single-threaded and holding 15 GB resident. Every PRISM run would pay that for a QC section that is off
by default, so `Skyline-PRISM.skyr` carries `Background` and not the ion columns. To use `measure: ions`
today, add the `LC Peak Transition Ion Count` column to the export yourself and expect the export to
take hours on a document that size. It is worth testing whether the per-transition column alone is
cheaper than all five together - the four precursor-level ones may be what costs, and `measure: ions`
only needs the per-transition one.

**Why ions are nonetheless the better measure.** Skyline's ion counting (Supplementary Table 3 of
doi:10.1021/acs.jproteome.5c00593) gives `LC Peak Analyte Ion Count` - the sum over the spectra inside
the peak boundaries of transition intensity times injection time - and `LC Peak Total Ion Count`, the
same sum over the total ion current. Both sides are then ions, so the unit mismatch and the background
asymmetry described below both disappear. Use the **LC Peak** columns, not the **Apex** ones: an apex
value is a single spectrum, not the peak.

It also cannot be reconstructed after the fact. On AGC-controlled data injection time varies by two
orders of magnitude within one run (0.06 to 10.6 ms measured on an Astral file) and anti-correlates
with intensity, because bright scans reach the target charge quickly. Applying a single injection time
to a summed intensity overstates ions by ~2.9x on that data, so ions have to come from a per-spectrum
calculation - Skyline's, or a re-extraction from the raw file.

**Two things `signal` under-counts, and by roughly how much.** The assigned signal is the signal in the
transitions the *document* carries, not everything a detected peptide produced: on a real cohort the
document held six fragments per precursor, covering a median 77.6% of each peptide's library spectrum
and 1.31x area-weighted. And a transition's `Area` is an integral in intensity-seconds while a summed
per-scan TIC is intensity with no time factor, so the acquired side must be multiplied by the cycle
duration before the ratio means anything - about 1.45 s on that cohort. The two corrections push
opposite ways.

---

## Built-in defaults vs. the generated template

Running with **no** `-c config.yaml` is not the same as running with a generated one: the transition
rollup falls back to a different method than the template writes. Set it explicitly in any config you
write by hand.

| Key | Built-in default | `config-template` writes | Effect |
|-----|------------------|--------------------------|--------|
| `transition_rollup.method` | `sum` | `median_polish` | **Numeric** — set it explicitly |

---

*When a parameter is added or removed, update this document together with the config template and the
key schema — see "Configuration is a contract" in [`CLAUDE.md`](../CLAUDE.md).*
