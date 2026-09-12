# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

- **PRISM now counts ions, and reports what fraction of them a peptide sequence explains.** Two
  numbers per replicate at each MS level: how many ions reached the detector, and how many of those
  fall inside a region some peptide of your analysis claims. Both come from the instrument files, so
  both are the same quantity measured the same way and the ratio is a genuine fraction.

  ```
  prism ion-accounting -d <output-dir> -r <raw-dir> --product-tolerance "10 ppm" \
      [--precursor-tolerance "10 ppm"] [--max 3] [--lanes 2]
  ```

  On a 39-replicate Astral cohort this reports, per replicate, figures like 40.5% of acquired MS1
  ions assigned against 3.4% of acquired MS2 ions - a gap that is the point rather than an anomaly.
  MS1 is dominated by the precursors that were identified; each 3 Th DIA window fragments everything
  co-isolated in it, and the identified peptides' fragments account for a twentieth of the result.
  The two levels are never drawn on one axis.

  **The unit is what makes this work.** A scan's intensity is a RATE - ions per second - so it is
  multiplied by that scan's ion injection time in seconds to become a count of ions, the same
  quantity Skyline reports as an ion count. An earlier version of this feature instead divided a
  summed peak area, which is an intensity-time integral, by a summed total ion current, which is an
  intensity: a ratio carrying units of time, which is not a fraction at all, and which looked
  entirely plausible as a coverage percentage. PRISM now refuses to draw any fraction above 100%
  rather than clamping it, because a fraction above 100% is impossible and therefore a defect worth
  seeing.

  **Shared signal is counted once.** Two peptides whose fragments fall within the extraction
  tolerance of each other in the same isolation window extract the *same* detector counts. Summing
  their peak areas credits both and can push assigned past acquired; PRISM instead merges every
  peptide's claimed region into disjoint m/z ranges before masking each spectrum, so there is only
  one reading to count. Nothing has to detect or subtract the overlap. The geometry is the
  document's own - the same isotopes, fragments and extraction tolerances Skyline used, and the same
  peak boundaries it integrated - read from `merged_data/` in one pass covering both MS levels.

  **It replaces the `PRISM-Ions` report, which is no longer needed for this.** Skyline computes
  `LC Peak Transition Ion Count` about 29x slower per row - roughly four hours instead of ten
  minutes on a 46M-row document - and even then the per-transition totals cannot be summed
  correctly, because they count shared signal once per transition. PRISM computes the same thing
  from the spectra in single-digit minutes per file.

  Deliberately its own command rather than part of `prism run`: a cohort is often a terabyte of
  instrument files over a network share, against a pipeline that otherwise reads one exported
  report. `--max` reads N replicates for a spot check, `--lanes` (default 4) sets how many files are
  read at once, and progress is written after *every* replicate, so an interrupted run keeps what it
  measured.

  **A partial result is topped up, not repeated and not trusted.** Re-running measures only the
  replicates the cache does not already cover — so a `--max 6` spot check followed by a full run
  reads the other 33 rather than all 39, an interrupted run resumes where it stopped, and adding a
  plate to a finished cohort measures the new plate instead of re-reading every file that was
  already done. On the cohort this was built for that is the difference between minutes and most of
  a day. The isolation scheme is imported from the
  first data file when the document does not carry one, which is the normal case - a DIA analysis
  document stores `<isolation_scheme name="Results only" />` and Skyline keeps the windows in the
  data files.

- **An Ion accounting pane in the tool's Visualization tab, with three interactive views.** Ions per
  replicate for the whole cohort; ions per acquisition cycle across the gradient for one replicate;
  and the assigned *share* across the gradient. The third earns its place because the two absolute
  traces both rise and fall with the elution envelope, so a stretch the analysis cannot explain is
  invisible in them and obvious in the ratio. That axis always starts at zero and fits the data
  above it: a non-zero origin is what makes a chart lie, while a fitted top is what makes a 3%
  trace readable at all - pinning it to 0-100% was tried first and left the line flat on the
  baseline.

  Everything on the pane is a read of two cached parquet files, so switching replicate, MS level,
  view or bin width is instant. The pane opens on the *median* replicate by assigned share rather
  than the first one alphabetically, with the best and worst a click away.

  The nav entry is hidden entirely until the output directory carries measured ion accounting. Every
  plot on it needs a denominator, and a fraction computed against a guessed one reads as coverage
  without being coverage - so there is nothing to offer rather than a pane that cannot draw.


- **The tool window is split into Analysis and Visualization.** Inputs, Settings and Log are about
  producing results and now sit under **Analysis**; QC Plots, Spectrum density and Dynamic Range are
  about reading them and sit under **Visualization**, chosen from a list down the left rather than
  from a tab strip. The output directory, **Run PRISM** and **Stop** stay above both. The plots are
  expected to keep arriving and a tab strip stops being readable at around eight of them; each pane
  also keeps its own state - zoom, ticked replicates, matrices already read - while you are on
  another one.

- **The Marker score plot reads out replicate names on hover**, the way the PCA plot does. Its points
  are jittered within their column so overlapping scores stay separable, which means the horizontal
  position carries no information - hovering is the only way to tell which injection an outlying score
  belongs to. The hover readout on both plots is also larger: it was the smallest text on a plot whose
  axis labels are set at nearly twice the size, while being the one piece of text a user leans in to
  read.

## Bug Fixes

- **A plot panel with no data no longer draws axes.** The three panels had drifted into three
  different empty states: QC Plots rendered nothing at all before the first run, leaving ScottPlot's
  raw default with an unstyled numbered grid to no scale; after a run it reset the chrome but never
  the axis limits, so a missing view inherited the previous plot's scale; and Spectrum density and
  Dynamic Range reset the whole control. All three now show one sentence saying why the panel is
  empty, on a panel with no axes at all - so an empty result cannot be misread as a flat measurement,
  and the numbers on it cannot be read as data that was never there.

- **Re-running an analysis onto a network share could fail Stage 1, and a stopped run could destroy
  the previous merge.** Both came from the same thing: the merge deleted `merged_data/` and then had
  DuckDB rewrite the same path. `Directory.Delete` only *starts* a removal - on Windows a directory
  survives until its last handle closes, and over SMB neither the server-side removal nor the
  client's directory cache is synchronous with the return - so the rebuild raced the teardown and
  failed with `Cannot open file "...\merged_data\_pep_bucket=1\data_0.parquet": The system cannot
  find the path specified`, after `_pep_bucket=0` had already been written. The merge is now built in
  a staging directory beside the target and renamed into place only once it has succeeded.

  The second half is the one to know about even off a share: any failure after the delete - pressing
  **Stop** during Stage 1, a full disk, the share going away - used to leave the output directory
  with no merged data, or worse, with *some* of the partitions. A partial one reads as a whole one,
  so the Spectrum density pane would plot a fraction of the cohort with nothing to say it was
  incomplete. A failed or cancelled merge now leaves the previous one untouched.

- **No published `prism` CLI could read an instrument file.** Nothing in the CLI referenced the
  reader assembly and nothing registered it, so any command that needed one answered "this build has
  no instrument-file reader" regardless of how PRISM was built. The reflection-based bootstrap the
  Windows tool already used now lives in one place both entry points share, every published CLI is
  built with the reader, and the release workflow fails if a published archive does not contain it -
  the failure was silent before, and a CLI that cannot read a file looks exactly like a cohort with
  nothing to measure.

- **One instrument file could serve several replicates.** Reference and QC injections are normally
  named identically in every plate's document, so `QC_1__@__plateA` and `QC_1__@__plateB` both
  matched a single `QC_1.raw` - giving two plates the same denominator, with which plate's numbers
  were real decided by dictionary order. Replicates are now paired to files one-to-one; a name that
  matches a file another replicate also matches is reported as ambiguous and neither is assigned it.

- **Instrument files were found on Windows and not on Linux.** The raw directory was searched with
  one glob per extension, and a glob pattern is matched case-insensitively on Windows but
  case-sensitively on Linux - so a `.RAW` file was invisible there. The directory is now listed once
  and filtered case-insensitively, and the extension list was widened to match what the readers
  actually accept (it was narrower, so some formats were never even offered).

## Performance

- **Ion accounting reads each instrument file once, and the masking is effectively free.** Measured
  on a 4.44 GB Thermo file of 168,920 spectra: 206.7 s in total, of which 204.5 s is decoding
  spectra and **1.0 s** is masking 465,307 claimed regions against every one of them. So the union
  arithmetic that makes shared signal count once costs half a percent of the work, and the cost of
  the feature is the file read - which is why it is a separate, cached step rather than part of
  every run.


## Breaking Changes

- **MS2 signal accounting is removed, and with it `qc_report.ms2_signal`, `prism ms2-signal` and the
  `PRISM-Ions` report.** It shipped in dotnet-v26.24.0 as a first attempt at the question ion
  accounting now answers, and it could not answer it: its numerator was a sum of Skyline peak areas
  (an intensity-time integral, background-subtracted) and its denominator a summed total ion current
  (an intensity, not background-subtracted), so the ratio was never a fraction of anything - and it
  needed an export that took about four hours on a 46M-row document to offer the better of its two
  measures. Ion accounting measures both sides from the instrument files in the same units, in
  single-digit minutes per file.

  What this means in practice:

  - A config carrying `qc_report.ms2_signal` still runs; the key is reported as unrecognized rather
    than silently ignored, and every other setting is unaffected.
  - `prism ms2-signal` is gone. Use `prism ion-accounting -d <output-dir> -r <raw-dir>`, which needs
    the same raw directory and writes `ion_accounting.parquet` / `ion_cycles.parquet`.
  - `Skyline-PRISM-Ions.skyr` is no longer in the tool zip, and the tool's **Export ion counts**
    option is gone. Exports are the standard `PRISM` report again - the fast one.
  - `ms2_signal_accounting.parquet`, `ms2_signal_lists.parquet` and `ms2_signal.parquet` left in an
    output directory by an earlier release are ignored, not read and not deleted. They can be removed.
