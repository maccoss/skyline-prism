# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

- **MS2 signal accounting in the QC report: how much of the integrated MS2 signal an analysis actually
  labels.** A new section plots, per replicate, the MS2 signal the run assigns to a peptide, with a
  profile line per selected protein list showing how much of that belongs to each panel. Enable it with
  `qc_report.ms2_signal.enabled: true`, and name panels with `qc_report.ms2_signal.protein_lists` /
  `protein_list_files`.

  The quantity is a **union, not a sum**. Adding up transition areas double-counts: a DIA isolation
  window co-isolates tens of peptides, and two fragments whose extraction windows overlap are the same
  detector counts credited twice - which can push "assigned" past what the instrument acquired. Each
  distinct region of MS2 signal space - (isolation window, extraction window, integration bounds) -
  counts once. On a real cohort replicate that removes 17.7% of the naive sum: 23,245 rows where
  Skyline exported one shared peptide per protein assignment, and 5,623 genuine shares between
  co-isolated peptides, which are reported separately because they mean different things.

  Totals **nest rather than partition**: a region claimed by a list peptide counts for that list and
  for the assigned total, and two lists may both claim it - the question is what portion of the signal
  a panel accounts for, not which panel owns a peptide.

  All values are **raw**: linear Skyline peak areas from `merged_data/`, untouched by normalization,
  ComBat or marker adjustment. The pipeline outputs supply peptide identity only, never magnitude.

  Off by default because it costs one extra streaming pass over the merged table. Results are cached as
  `ms2_signal_accounting.parquet` (and `ms2_signal_lists.parquet`), so `prism qc -d` replots without
  recomputing and keeps working on a directory whose `merged_data/` was cleaned up. The cache is keyed
  on what was asked for - measure, extraction tolerance, isolation scheme, protein lists - so a re-run
  that changes any of them recomputes and says so, rather than replotting the previous run's numbers
  under the new run's caption. With the section turned off, a cache left by an earlier run into the
  same directory is left alone rather than rendered.

  The largest bar is **not** acquired MS2 signal - that needs the instrument files, and no Skyline
  export can supply it (`TicArea` is one value per replicate and MS1 by construction). It is the signal
  Skyline integrated for the document's targets, and it is labelled that way, because reading it as the
  acquired total would turn unknown coverage into apparently complete coverage.

- **MS2 signal accounting can total ions instead of signal, and the Skyline tool can export what that
  needs.** `qc_report.ms2_signal.measure` chooses: `signal` sums each transition's gross peak area
  (`Area + Background`), available from any export; `ions` sums Skyline's `LC Peak Transition Ion
  Count` - intensity times injection time per spectrum, summed across the peak - so the total is a count
  of ions, neither side is background subtracted, and no unit correction applies. Ions are the better
  measure and cannot be recovered from an area after the fact: on AGC-controlled data the injection
  time varies by two orders of magnitude within one run (0.06 to 10.6 ms measured on an Astral file)
  and anti-correlates with intensity, so a constant approximation is wrong by 2.9x.

  The column is expensive for Skyline to compute, so it is not in the standard `PRISM` report. Measured
  on the 6.5 GB FLARE document (46M transition rows, 93 replicates): the three-column baseline exported
  in 9.5 minutes; with five LC Peak ion-count columns it was 13% done after 63 minutes - 27x slower per
  byte, projecting to 9-13 hours. The per-transition column alone was measured too: 7.9M rows in 42
  minutes, 187k rows/min against the baseline's 5.4M, so **29x slower per row - about 4 hours instead of
  9.5 minutes**. Twice as fast as all five, which places the cost in reading each transition's
  chromatogram points rather than in the columns derived from them; there is no cheaper column to ask
  for. A second definition, **`PRISM-Ions`** (`Skyline-PRISM-Ions.skyr`), is
  the `PRISM` report plus that one column and nothing else, so a document exported either way merges
  identically; a test holds the two files to that relation.

  On the Settings tab, **9. MS2 signal accounting** turns the QC section on, picks the measure, and has
  an **Export ion counts** option that exports each Skyline document with `PRISM-Ions` instead. The two
  are deliberately separate: once ion counts are exported, both measures are available; `ions` is grayed
  out - with the reason as its tooltip - until every input will carry the column, and a pre-exported
  report is checked for it from its header (off the UI thread, and re-checked when the file changes, so
  re-exporting one makes `ions` available without re-adding the input). The export is refused outright
  when an input cannot carry the column, because a cohort whose inputs have different columns cannot be
  merged - the run would otherwise die in Stage 1 after hours of export. Each report variant gets its
  own export directory, so switching the measure back does not overwrite the expensive one.

  The tool also reads the document's own product-ion extraction tolerance (`transition_full_scan`) into
  `extraction_tolerance`, the way it already reads the digestion enzyme - asking every input and warning
  when the documents disagree, since one tolerance is applied to the whole cohort. Windows the setting
  cannot express (a resolving-power analyzer, or a QIT window with selective extraction) are left to the
  configured value and named in the log. The QC report waits - briefly, and only at Stage 5b - for the
  isolation-window read, so the section is there on the first run without the window read delaying the
  stages that do not need it.

- **The Skyline tool can now measure ACQUIRED MS2 signal, by reading the instrument files.** This is
  the denominator the MS2 signal accounting needs and no Skyline export carries: `TicArea` is MS1 by
  construction (its TIC filter sets `Ms1ProductFilters` and leaves `Ms2ProductFilters` empty), so it
  cannot stand in - on a real cohort the assigned MS2 signal came to 0.51x `TicArea`, a number that
  looks like a coverage fraction and is not one.

  `SkylinePrism.zip` now bundles the reader, built from
  [pwiz-sharp](https://github.com/ProteoWizard/pwiz/pull/4619), and it adds about 4 MB. Spectra are
  read at metadata detail level so the peak arrays are never decoded - the total ion current is a
  header field - and the instrument's own reported value is used rather than a sum of intensities, so
  a vendor file and its converted mzML agree. Isolation windows are captured on the same pass, which
  removes a Skyline round-trip: a DIA document normally stores none. Measured on a 4.9 GB Astral run:
  963 MS1 and 120,327 MS2 spectra, MS2 TIC 3.07e11, 125 windows spanning 400.4-900.7 m/z at 4.0 Th,
  in 239 s.

  Measured on a real 93-replicate plasma cohort (FLARE-Extended, 3.3 GB per file), the analysis
  assigns **9.8-9.9%** of the acquired MS2 ion current to its 45,680 peptides:

  | replicate | acquired MS2 TIC | assigned | fraction |
  |---|---|---|---|
  | FLARE-001-1-B1-013 | 3.768e11 | 3.729e10 | 9.90% |
  | FLARE-001-2-B4-016 | 4.499e11 | 4.470e10 | 9.94% |
  | FLARE-002-1-G4-076 | 3.850e11 | 3.761e10 | 9.77% |

  The spread is 0.17 percentage points across replicates whose acquired total varies by 19%, so the
  fraction is a property of the method rather than of the injection. It also shows the union correction
  matters once there is a denominator: a naive sum over transition areas would have claimed 12.2-12.9%,
  overstating coverage by about a quarter.

  Thermo formats for now, plus the open ones (mzML, mzXML, mz5). A format nothing can open reports the
  acquired total as unknown, which leaves every other number in the report unaffected. The bundling is
  temporary: Skyline is being ported to C# and will install pwiz itself, at which point the tool should
  use Skyline's own copies.

- **The extraction window Skyline used is read from the document.** `SkyDocumentInfo` now reports the
  `transition_full_scan` product-ion settings and evaluates them to the m/z range Skyline actually
  extracted over, transcribed from Skyline's own `GetDenominator` / `GetFilterWindow` and applied the
  way it applies them. The stated `product_res` is the **+/- tolerance**: `centroided res="10"` extracts
  +/-10 ppm, `qit res="0.7"` extracts +/-0.7 m/z. Documents that do not say enough - an Orbitrap setting
  with no calibration m/z - report the tolerance as unknown rather than a plausible default.

## Bug Fixes

- **The QC report says "Raw" and "Corrected" instead of "Before" and "After", and the CV plots no
  longer look like they are about median normalization.** "Before / After" left the reader to work out
  before and after *what*, and the answer depends on the configuration - with RT-LOWESS, ComBat and
  marker normalization all enabled, "After" is all three. The two panels have always been the raw
  rollup and the arm's final written output, so they are now named after the files they come from:
  `peptides_rollup` / `proteins_raw` is **Raw**, `corrected_peptides` / `corrected_proteins` is
  **Corrected**. Renamed in the section headings, the image captions, the plot titles and the CV
  tables. Separately, the CV histogram legend read "Before median 34.4%", which looked like a claim
  about median normalization rather than the median of the CV distribution; it now reads
  "Raw: median CV 34.4%".
- **The PCA legend no longer sits on top of samples.** It was drawn in a fixed corner, so on a real
  cohort it covered replicates - which matters on the one plot people use to find the odd sample out.
  It now goes in whichever corner holds the fewest points.

- **The "not even one export fits" warning no longer cries wolf.** dotnet-v26.23.1 made free memory a
  bound on export concurrency, and because free memory is always at or below installed, that bound is
  the one that decides - so a 64 GB machine with 33 GB free budgeted 19.8 GB against an export needing
  21.2 GB, reported that nothing fitted, and told the user to close Skyline windows for an export that
  fitted in free memory with 12 GB to spare. The warning also quoted the installed budget while doing
  it, which made its arithmetic read as false. Falling short of the budget and not fitting the machine
  are now separate questions: the budget still decides whether exports may overlap and stays
  conservative, while the warning fires only when one export genuinely does not fit what is free, with
  4 GB held back for the OS and Skyline's own baseline.

## Performance

## Breaking Changes
