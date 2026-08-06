# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

- **New "Dynamic Range" tab in the Skyline tool**: log10 abundance against abundance rank - the shape of
  Skyline's Relative Abundance plot - over the **corrected** PRISM matrices. Switch between **Protein** and
  **Peptide** (`corrected_proteins.parquet` / `corrected_peptides.parquet`), and average over **All**
  replicates or any subset you tick; abundances are averaged on the linear scale and only then
  log-transformed, and a protein never measured in the selection is dropped rather than plotted at zero.
  Changing the replicate selection re-ranks, since the ordering itself depends on what is averaged.

  **Clicking a point selects that protein or peptide in the Skyline document tree**, using the element
  locators read from Skyline itself rather than strings built by hand. At peptide level it selects the
  **peptide** node, so its chromatograms come up. The status line reports every protein the peptide is
  present in - PRISM runs its own parsimony, which can group proteins differently from the document, so a
  peptide shared across groups lists them all; selection goes to the first of those groups, falling back
  to the first occurrence in the document tree (and saying so) when PRISM's grouping matches no protein
  node.

- **`corrected_peptides.parquet` now carries protein grouping**: `protein_group`, `leading_protein`,
  `leading_name` and `leading_gene_name`, so the peptide and protein matrices can be joined directly. A
  peptide shared between groups lists all of them, `;`-separated and index-aligned across the four
  columns - membership, not the quantification assignment that `parsimony.shared_peptide_handling`
  controls. (C# engine only; the intermediate `peptides_log2_internal.parquet` is unchanged.)

  **User-defined protein lists** highlight sets of interest - plasma contaminants, EV markers, endothelial
  markers - each with its own colour and an on/off toggle. Define them in the tab's *Protein lists…*
  editor, or import from a text/CSV file; they are saved per user, so the same lists are available in
  every project and output directory. Members may be written as UniProt accessions, gene names or full
  protein names (with or without `sp|` prefixes, species suffixes or isoform numbers) - all forms are
  matched. Point labels (gene name, falling back to accession) can be turned on for the Skyline selection
  or for whole lists from the plot's **right-click menu**, alongside ScottPlot's own Save/Copy items.

- **New "Spectrum density" tab in the Skyline tool**: a map of how many peptide precursors were detected
  in each DIA spectrum of a run - retention time across, precursor m/z up, colour = precursors per cell.
  Pick the run from a drop-down (every acquisition in the merged report is listed), set the **RT bin**
  (default 0.1 min) and the **max q-value** that counts as a detection (default 0.01). Hovering reads out
  the isolation window and exact count under the cursor. Seven colour schemes are available: Viridis (the
  default, as in Skyline-Cadenza), Turbo, Magma, Inferno, Plasma, Thermal and Grayscale.

  **Rows are the acquisition's real isolation windows**, read from the data, not an assumed bin width - so
  a cell really is one DIA spectrum, including for variable-width and staggered/overlapping schemes (a
  precursor in an overlap is counted in both windows it was fragmented in). PRISM gets them two ways:
  from the document's Full-Scan settings when it defines a scheme, and otherwise - the normal
  `Results only` case, where Skyline keeps the windows only inside the raw files - by having **Skyline
  import them from one of the run's data files**, the command-line form of Transition Settings >
  Full-Scan > Isolation scheme > Add > Import from a data file. That import runs against a throwaway
  document, so your own document is never opened or modified, and takes about 10 seconds even for a
  multi-gigabyte Thermo `.raw` on a network share.

  This is not a cosmetic difference: a real forbidden-zone scheme has 3.0014 Th windows starting at
  400.4319, so a uniform 3 Th grid starting at 400 sits ~14% of a window off and cuts through the very
  precursor clusters the scheme was designed to keep intact.

  The windows are resolved when you click **Run PRISM** - when the raw data is most likely still where the
  document says it is - and recorded (with every isolation scheme saved in the attached Skyline, as a
  manual fallback) to `isolation_schemes.xml` in the output directory, so the map still bins on real
  windows when the tab is reopened later with no Skyline running and no data files reachable.

- **Scheduled acquisitions - PRM, MTM and dynamic DIA - are handled as scheduled**, not as
  DIA-without-a-scheme. An isolation
  window is now an m/z range crossed with the retention-time interval it fires in - the same model as
  Skyline-Cadenza's scheduling slots. A precursor counts toward a window only if its peak eluted while that
  window was firing (so two targets sharing an m/z at different times go to their own slots), the RT axis
  spans the whole schedule so slots that fired and detected nothing still appear, and time outside a slot's
  interval draws as *not acquired* rather than as a zero - a zero on the map always means "acquired,
  nothing detected". Multiplexed MTM slots correctly count several co-eluting precursors and are drawn at
  their true width, so they stand out from solo PRM targets.

  **Dynamic DIA** works the same way, with a whole cycle of windows per segment rather than one slot: the
  8 x 8 m/z windows shift along m/z as the gradient runs, so the same m/z is covered by different windows
  at different times. Each precursor is credited to the cycle that was running when it eluted, a peak
  straddling a segment boundary counts in both cycles, and m/z the cycle has marched past renders as
  never-acquired instead of as an empty spectrum.

  Because Skyline cannot import a schedule from the data (its importer needs a repeating cycle), the
  windows come from the **Thermo inclusion list** that was loaded onto the instrument - for a scheduled run
  that file *is* the scheme. Use **Load inclusion list (PRM/MTM)…** in the Spectrum density tab's scheme
  drop-down; the parsed windows are saved into the run's `isolation_schemes.xml` and offered again next
  time. Precursors falling outside every window are counted
  and warned about rather than clamped into the nearest row, so a mismatched scheme is obvious instead of
  silently skewing the map. A clearly labelled uniform-bin fallback remains for output directories with no
  scheme information at all.

  The tab reads `merged_data.parquet` from the output directory, so it works both for a run that has just
  finished and for any previous run's output directory - and needs no Skyline connection.

- **Figure-ready plot styling everywhere**: axis titles, tick labels and legends across all QC plots (PCA,
  CV, intensity, RT and the new density map) are substantially larger and bold, with heavier axis lines and
  larger PCA markers, so a plot copied out of the tool stays readable in a journal column or on a
  projector. The static QC-report PNGs were enlarged to match (1100x780).
- **Consistent fonts across machines**: plots now pin an explicit font family (the first of Segoe UI,
  Helvetica, DejaVu Sans, Liberation Sans, Arial that is installed) instead of taking whatever the
  rendering backend happened to resolve first, which is why the same plot rendered with different-looking
  text on different computers.
- **Consistent fonts across plots, and with the report around them.** The control-correlation heatmap
  set its own font sizes and never went through the shared styling, so it rendered with a small
  title and its own typeface next to plots using neither; its colour bar and per-cell numbers were
  unstyled entirely, as were the Dynamic Range point labels and the PCA hover label. All of them now
  share one family and the same title treatment, and the heatmap's colour bar is legible instead of
  shrinking with the grid. Its per-sample tick labels stay small - there is one per sample, and a
  large cohort has no room for more. The HTML reports now lead their CSS font stack with the family
  the plots actually resolved to, so the page text matches the axis labels in the images it embeds.

## Bug Fixes

- **The Dynamic Range tab no longer comes up blank at protein level.** Its list of protein metadata
  columns had drifted from what the pipeline writes - it was missing `leading_description`,
  `n_unique_peptides` and `low_confidence`, and listed a `confidence` and a `quant_method` that are
  never written. Those three columns were therefore treated as replicates, and parsing a protein
  description as an abundance threw, leaving an empty replicate drop-down and no plot. The column
  list now comes from the writer (`ProteinRollup.MetadataColumns`) instead of being repeated, so the
  QC report and the tab cannot disagree about it again.

- **ComBat no longer destroys the whole matrix when a single value is missing.** Standard ComBat
  used NaN-propagating means and variances where R's `sva::ComBat` (the implementation PRISM is
  based on) uses NaN-aware ones. Because the empirical-Bayes priors are means taken *across*
  features, one missing cell reached them and turned every corrected value in the cohort into NaN.
  A peptide absent from some samples is the normal case in real data, so any cohort with missing
  values and `batch_correction.reference_anchored: false` (the default) was affected. Fixed in both
  engines: one missing cell in, one missing cell out.

- **ComBat no longer invents batch effects it cannot estimate.** A (batch, feature) with fewer than
  2 observations, or no spread among them, has no estimable scale; the previous code substituted a
  placeholder `1.0` and fed it into that batch's `aPrior`/`bPrior`, letting one such feature perturb
  the shrinkage of every other feature in the batch. It now keeps its location correction, skips the
  scale correction, and is excluded from the prior. Features with no variance at all, or absent from
  a whole batch, are held out and returned unchanged. Both are now reported in the log rather than
  happening silently.

  This changes results where such features exist - `proteins_raw` in particular, where a protein
  constant across a whole plate is common.

- **ComBat no longer gives different answers in the two engines for the same data.** Deciding
  "is there any spread in this batch?" with an exact `variance == 0` test is knife-edge: the same
  82 replicates produced a within-batch variance of exactly `0.0` in Python and `7.99e-31` in C#,
  which flipped the feature between "no scale to estimate" and "a scale of 8e-31" - and the two
  engines' protein abundances then differed by 3%. Both now treat a spread below `1e-12` of the
  values' own magnitude as rounding, which is far above accumulated floating-point error and far
  below any real measurement.

- **ComBat is now checked against the implementation it is based on.** New fixtures generated by
  R's `sva::ComBat` 3.58.0 (`dotnet/tests/fixtures/sva/`) hold both engines to an external
  reference, rather than only to each other - which is how the NaN bug survived for so long, with
  the two engines confidently agreeing on the same wrong answer. **PRISM now reproduces
  `sva::ComBat` to floating-point noise on dense input** - the normal case, since Skyline integrates
  imputed peak boundaries for every replicate.

  Three documented differences remain, all deliberate:

  - **`var_pooled`'s denominator.** sva uses `sum(residual^2)/n` when the input is dense but
    `rowVars(na.rm = TRUE)` when it is not. PRISM uses the former throughout, so it matches sva on
    dense input and differs by ~0.3% on input with missing values. Following sva's switch would mean
    one peptide missing from one document shifts every corrected value in the cohort by ~0.3%.
  - **Features constant within a batch** keep the location correction the data supports; sva drops
    them entirely. Because sva's priors are then computed over a smaller feature set, this moves
    every feature slightly, not just the constant ones.
  - **Two inputs sva errors on outright**, PRISM handles: a feature observed once in a batch (sva's
    uncapped `while (change > conv)` loop dies on the resulting `NA`), and a feature absent from a
    batch (`Beta.NA` hits a singular design).

## Performance

- **The merge no longer spills its sort onto a network drive.** DuckDB's scratch directory was always
  placed beside the output, and PRISM output routinely lives on a mapped drive - the Skyline tool
  defaults to a folder beside the document. A large sort then wrote gigabytes of spill over SMB,
  which is slow enough to look like a hang and fails outright on some servers. The scratch directory
  now stays beside the output only when that is a local disk, and falls back to the machine's own
  temp directory for a network or UNC path. Stage 1 logs which directory it chose, and
  `PRISM_TEMP_DIR` overrides it when the automatic choice picks badly (a small system drive, say).
  The directory is also cleaned up after a failed merge now, rather than only after a successful one.

- **Peptide normalization and ComBat (Stage 2b/2c) no longer hold the peptide x sample matrix in
  memory.** This stage was the pipeline's memory wall - Stage 1 (merge) and Stage 2 (transition
  rollup) already streamed, so a cohort of ~100 Skyline documents ran out of memory here and nowhere
  else. It now computes per-sample normalization factors from a column-at-a-time pass, drives
  ComBat's empirical-Bayes step from two summary numbers per (batch, peptide) instead of the
  standardized matrix, and writes both outputs one row group at a time - which also removes the two
  full matrix transposes the writer used to build.

  Measured on a synthetic 20,000-peptide x 600-sample cohort: **peak managed memory fell from 798 MB
  to 102 MB**. The saving grows with sample count, because what remains is bounded by the input's
  row-group size (2,000 peptides) rather than by the number of samples.

  Results are unchanged - a new same-process regression suite compares the streaming and in-memory
  implementations cell by cell at 1e-12 relative across every normalization method, ComBat on and
  off, auto-revert triggering and not, dropped all-NaN rows, and multi-row-group inputs. Three cases
  still use the in-memory implementation and say so in the log: `quantile` normalization (a cell's
  value depends on its whole column at apply time), a CSV/TSV `output.format`, and reference-anchored
  ComBat.

## Breaking Changes
