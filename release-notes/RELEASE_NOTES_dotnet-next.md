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

  **At MS2 the question has two answers, and the report gives both.** The *quantified* total is the
  signal in the transitions your document carries - the handful of fragments Skyline integrates, and
  the right numerator for "what is my quantification standing on". The *explained* total is every
  theoretical b and y ion at 1+ and 2+ (capped at the precursor charge) plus the surviving precursor
  and its first two isotopes - the right numerator for "how much of the acquisition can this peptide
  account for at all". The unfragmented precursor and the low-m/z fragments are poor quantifiers,
  which is why Skyline does not pick them, and they are still part of the mass balance. Both appear
  per replicate and across the gradient: a third bar on each plot, a third trace on the absolute
  profile, and a second line on the fraction profile.

  The explained set is the theoretical ions UNIONED with the quantified claims, so it can never fall
  below the quantified total even where Skyline integrates an ion the enumeration does not produce.
  Every sequence is reconciled against Skyline's own `Precursor Mz` before its claims are used, and a
  precursor whose mass PRISM cannot reproduce is excluded and COUNTED - that count is reported per
  replicate, because its only other symptom would be a quietly smaller explained total. Validated
  against every distinct precursor of the committed cohort fixture: 385 of them, worst deviation
  0.0022 ppm.

  Two limits worth knowing. **Heavy isotope labels are not yet handled** - the exported sequence is
  the peptide-level one, which carries structural modifications only, so on a document with heavy
  internal standards those precursors fail to reconcile and are excluded rather than mis-claimed.
  And **an export without a `Precursor Charge` column measures no explained total at all**, which the
  plots show as absent rather than as zero.

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

- **An Ion accounting pane in the tool's Visualization tab.** Three pickers, each answering one
  question. **View** is the shape: *Per replicate*, one bar per run so the cohort can be compared
  and an outlier found, or *Across the gradient*, the same totals per acquisition cycle for one
  replicate. **Quantity** is what is measured: *Ions* or *Signal*. **Show** is how: *Totals*, or
  *Fraction of acquired*.

  Fraction earns its place because the absolute traces all rise and fall with the elution envelope,
  so a stretch the analysis cannot explain is invisible in them and obvious in the ratio - and
  because the absolute height is dominated by how much was loaded, which makes two replicates
  incomparable until they are divided by their own acquired totals. That axis always starts at zero
  and fits the data above it: a non-zero origin is what makes a chart lie, while a fitted top is
  what makes a 3% trace readable at all - pinning it to 0-100% was tried first and left the line
  flat on the baseline.

  Everything on the pane is a read of two cached parquet files, so switching replicate, MS level,
  view, quantity or bin width is instant. The pane opens on the *median* replicate by assigned
  fraction rather than the first one alphabetically, with the best and worst a click away.

  It opens on **Signal (TIC)** - what the instrument reports and what a mass spectrometrist reads a
  run in - falling back to Ions on a cache measured before the summed TIC was recorded, so an older
  directory still gets a plot rather than an error. The gradient bin defaults to **0.01 min**: at
  0.6 s that is shorter than one acquisition cycle on these instruments, so it bins essentially
  nothing and the trace is drawn at the rate the run was acquired at. The QC report uses the same
  bin.

  The nav entry is hidden entirely until the output directory carries measured ion accounting. Every
  plot on it needs a denominator, and a fraction computed against a guessed one reads as coverage
  without being coverage - so there is nothing to offer rather than a pane that cannot draw.


- **The Ion Accounting pane plots ions or signal, names the bar under the cursor, and lets you
  choose the order of them.** Three things that were missing from a plot with one bar per replicate
  and no room to label any of them.

  **Quantity: Ions or Signal (TIC).** These are two different quantities, not two units for one. A
  scan's intensity is a RATE, in ions per second: the ion count multiplies it by that scan's ion
  injection time and the TIC does not, leaving a sum of rates - which is what the instrument reports
  and what a mass spectrometrist reads it in. Both are measured in the same pass over the same
  scans. Their assigned fractions are different numbers too, and neither is wrong: the ion fraction
  weights each scan by its injection time and the signal fraction does not, so they agree only where
  the assigned fraction happens to be constant across injection times. Where they diverge, the AGC was
  working. The axis, legend, title, median and hover readout all name the quantity actually drawn.

  **Hover** reads out the replicate: name, sample type, acquired and quantified totals, the fraction
  (or the reason there is none), the explained fraction, when it was acquired and which file it
  came from.

  **Order**: run order, file name, or grouped by sample type. Run order comes from each data file's
  own acquisition start timestamp, now recorded in `ion_accounting.parquet` - the only honest source
  for it, since neither the file name nor the order files were read in is the order they were
  acquired in. A cache written before that column falls back to file name and the status line says
  so, rather than presenting an arbitrary order as an acquisition one. File-name order compares
  digits as numbers, so a 48-well plate reads A1, A2, A10 rather than A1, A10, A11, A12, A2.

  The ion accounting cache key moves to `ions-v4` for the signal columns, so the next measurement of
  an existing output directory re-reads its files once and comes back with ions, signal and run
  order together.

- **The Ion accounting pane's per-replicate view has the QC Plots' Group-by controls.** Pick any
  column of the Replicates report - plate, condition, subject, or Sample Type as before - and the
  bars take one color per value of it, with the legend naming the groups; tick values to show only
  those replicates, or none to show them all. The plot title says which replicates are shown and the
  hover readout names a replicate's group beside it. Grouping also fixed a latent coloring defect:
  bars of a sample type the palette did not know were each given a different cycled color, under a
  legend swatch that matched none of them; every bar of a group now shares the group's hue.

- **The tool can be told the extraction tolerance when no document can state it.** Ion accounting
  refused to run against a guessed tolerance - rightly, since the extraction window decides how much
  fragment sharing is found between co-isolated peptides and every figure would move with nothing
  saying it had - but a PRE-EXPORTED REPORT carries no Full-Scan settings, so there was no way to
  supply one. That included the report PRISM itself writes into `skyline-reports/`, so re-running
  against a previous run's export could never measure ions.

  Settings now has **Product tolerance** and **Precursor tolerance** boxes beside the data
  directory. What you type wins, a document fills in when they are blank, and with neither there is
  still no tolerance rather than a plausible default. The run log names which source was used.

- **The tool window is split into Analysis and Visualization.** Inputs, Settings and Log are about
  producing results and now sit under **Analysis**; QC Plots, Spectrum density and Dynamic Range are
  about reading them and sit under **Visualization**, chosen from a list down the left rather than
  from a tab strip. The output directory, **Run PRISM** and **Stop** stay above both. The plots are
  expected to keep arriving and a tab strip stops being readable at around eight of them; each pane
  also keeps its own state - zoom, ticked replicates, matrices already read - while you are on
  another one.

- **The Spectrum density map now bins on the windows the data was actually acquired with, and PRISM
  writes them down so they outlive the data files.** A DIA analysis document stores
  `<isolation_scheme name="Results only" />` and no windows at all - Skyline reads them from the
  instrument files at import and does not record them - so the map used to open on a built-in layout
  that looks exactly as plausible as the right one. On a real Astral acquisition the true scheme is
  167 windows of 3.0014 Th starting at 400.4319, deliberately placed in the peptide forbidden zones;
  a uniform 3 Th grid starting at 400 sits ~14% of a window off and cuts through the very precursor
  clusters the scheme exists to keep intact.

  The tab now reads the real windows itself, through PRISM's own ProteoWizard reader, and defaults
  the picker to them - labeled **(from the data files)**, so which entry is the acquisition's own
  answer and which are guesses is visible in the list rather than inferred. The read costs one file
  open (~4 s on a 3.3 GB Thermo file over an SMB share) because the windows are scan headers in the
  first two acquisition cycles, and it starts *after* the map is already drawn, so a slow share
  delays an improvement rather than the plot. It needs no Skyline: checked against the same
  acquisition Skyline imported, all 167 windows agree edge for edge.

  A scheme you picked yourself is never overruled by it.

  **And it is recorded twice, on purpose.** The windows previously existed only in
  `isolation_schemes.xml` beside the outputs and in the instrument files - and the instrument files
  are the first thing to be moved off a share when an analysis is finished. A run's `parameters.json`
  now carries an `isolation_schemes` block with the window edges, which file they were read from and
  when, so a result archived on its own can still say what it was acquired with; the QC report's
  Analysis Information names the scheme for the same reason. Nothing about `--from-provenance`
  changes - this is a record of the acquisition, not a processing parameter.

  New:

  ```
  prism isolation-scheme -d <output-dir> [-r <raw-dir>] [--force]
  ```

  the headless equivalent - resolve the windows, record them, print what they are. With no `-r` it
  reports what the directory already knows, which works with no data files present at all.

- **One default per setting, whichever way a run is started.** `transition_rollup.method` defaulted
  to `sum` with no config file and `median_polish` in the config `prism config-template` generates,
  so a run started from the GUI summed transitions while a run started from a generated config
  median-polished them - different numbers, both plausible, with nothing anywhere saying the two
  entry points disagreed. The templates now emit the real defaults: **`sum`** for
  transition -> peptide, **`median_polish`** for peptide -> protein, **`rt_lowess`** for peptide
  normalization. A test compares every one of them against the code rather than trusting prose.

  **If you generated a config before this release and kept it, nothing changes** - your file says
  `median_polish` and will keep median-polishing. A config generated from now on says `sum`, which
  is what a run with no config has always done.

- **A result can say what its numbers were extracted with, after the document has moved on.**
  `parameters.json` gains an `extraction` block beside `isolation_schemes`, recording the product and
  precursor windows the ion accounting used. The extraction window decides how much fragment sharing
  is found between co-isolated peptides, so every assigned figure moves with it - and the tolerances
  live in the Skyline document, which a result outlives as surely as it outlives the instrument
  files.

  All four values are recorded - analyzer, resolution, the m/z a resolving power is calibrated at,
  and whether selective extraction halves the window - rather than the single-number setting string.
  A centroided or QIT window is one number; TOF, Orbitrap and FT-ICR are not, and selective
  extraction changes the window for every analyzer except centroided. An earlier attempt at this
  recorded the string form and therefore recorded nothing at all on those instruments, while reading
  as though it worked because the common case was fine.

- **The run log says which build wrote it.** Its first line, the first line of an ion accounting
  measurement, and the tooltip on the version in the window now carry the build time and the folder
  PRISM loaded from:

  ```
  PRISM 26.24.3, built 2026-09-13 12:19 from C:\Users\...\Tools\SkylinePrism
  ```

  The version alone cannot answer "am I running the fix?" - PRISM versions at RELEASE time, so every
  build between two releases reports the same number. Two multi-hour measurements were run against a
  build that predated the fix they were testing, and nothing on screen could have said so. The
  folder is half the answer: the Skyline tool runs the copy installed under Skyline's own Tools
  directory, which a freshly built zip sitting on disk does not touch.

- **When a file cannot be written, PRISM names the process holding it.** On Windows, through the
  Restart Manager - no elevation needed - so "locked by another process" becomes
  `open on this machine by: MsMpEng (pid 1234)`. When nothing on this machine has it, it says that
  too, which for a file on a share is the useful half of the answer: the holder is another client or
  the file server itself.

- **The Marker score plot reads out replicate names on hover**, the way the PCA plot does. Its points
  are jittered within their column so overlapping scores stay separable, which means the horizontal
  position carries no information - hovering is the only way to tell which injection an outlying score
  belongs to. The hover readout on both plots is also larger: it was the smallest text on a plot whose
  axis labels are set at nearly twice the size, while being the one piece of text a user leans in to
  read.

- **PRISM says what is already in the output directory before it writes there.** Pointing a run at a
  folder that holds a previous cohort's results said nothing at all - it recomputed what the settings
  changed and replaced the rest, and the person who got there by pasting the wrong path found out
  from the file timestamps. The CLI now logs a warning naming the version, the date and the files it
  would replace, and carries on; the tool asks first and does not start unless you say so.

  **The tool asks whenever a finished analysis is there**, not only when the numbers would come out
  different. A re-run with identical settings still deletes and rewrites every file in the folder, and
  if those files are somebody's finished analysis they are just as gone; that the new numbers would
  have matched is no comfort to whoever owned the old ones. The default output folder is the
  document's own `PRISM-Output`, so landing on a previous analysis takes no mistake at all - it is
  what happens unless the path is changed. The question names the machine that produced what is there
  when it was not this one, which is the difference between overwriting your own re-run and
  overwriting a colleague's cohort across a share.

  The CLI stays silent when the previous run used the same version and the same settings, which is how
  a QC report gets regenerated and a partial ion accounting gets topped up: it is a log line on a
  scripted run, not a destructive action someone is about to take by hand.

  The question is asked **per stage**, against the same declarations the stage cache keys on, so a
  setting no stage reads - thread count, whether plots are saved - says nothing, and a setting that
  moves only the protein rollup names the protein rollup and what follows it rather than implying the
  peptide results are going too. The files listed are the ones those stages actually recorded.

  The CLI, which knows its input files, answers the two most expensive stages **exactly** rather than
  predicting them: a report re-exported from Skyline with not one setting changed is caught, because
  the merge is checked against the same stamp of path, size and write time the pipeline itself uses.
  A config comparison cannot see that case at all.

- **Re-measuring ion accounting with different settings says what it is about to discard.** The log
  noted that the cache "is being recomputed"; it now names how many measured replicates are in the
  directory, what they were measured with and what this run uses, and says plainly that re-measuring
  reads every instrument file again. It is the most expensive output PRISM writes - hours of
  instrument reads - and it was the one output with no warning in front of it.

## Bug Fixes

- **The Load histogram labels only whole-number loads.** A spectrum carries 0, 1, 2 ... precursors,
  but on a short axis ScottPlot labeled the half-steps too, so the Spectrum density tab's histogram
  read "0.5 precursors" under a bar that cannot exist. The axis now carries integer ticks only.

- **The QC Plots picker offers Marker score and Marker loadings only while marker normalization is
  switched on in Settings.** Both plots read `marker_normalization.csv`, which only a run with it on
  writes, so with it off they led to "This run did not record marker loadings" - a dead end reached
  from a drop-down. The two entries are hidden rather than grayed out, and a marker plot that was
  showing falls back to PCA when the box is unticked.

- **The RT-binned plots say which minutes each bin covers.** The bars sit at positions 0 to 7, which
  ScottPlot treated as an ordinary numeric axis - so it labelled the midpoints too and eight bins came
  out as `-0.5, 0, 0.5, 1 ... 7.5`, which is neither eight of anything nor a retention time. The axis
  now carries one tick per bin, labelled with the retention times that bin actually covers
  (`2-10`, `10-18`, ... in whole minutes), and is titled "Retention time (min)".

  The index was the deeper problem. The reason to plot CV against RT is to find *when* in the
  gradient a run is noisy, and "bin 6" cannot be held against a chromatogram, an acquisition method,
  or the same question asked of another cohort - all of which are in minutes. The bin edges were
  already being computed and simply never shown. Applies to both **RT-binned CV** and the **RT-bin
  boxplot**, in the QC report and in the tool. A gradient short enough for sub-minute bins keeps the
  decimals it needs rather than printing `12-12` three times over.

- **A Spectrum density cell is what one spectrum saw, not what a stretch of time contained.** Each
  precursor used to add a count to every RT bin its peak spanned, so a cell held the union of
  everything that eluted during that bin: two peptides that were never in the same spectrum, one
  finishing before the other began, counted as two co-detected, and the number grew with the bin. A
  cell is now the greatest number of precursors co-eluting at any one instant inside its column, found
  by sweeping the peak boundaries, so the busiest cell no longer moves with a display setting. On a
  90-minute Lumos DIA run with 8 m/z windows the old 0.1-minute bin overstated the five busiest windows
  by 10-18%.

  The RT bin box in the tool now opens at the documented 0.01 min - about one acquisition cycle -
  rather than the 0.1 it had kept from before the default changed. A wider column reports the worst of
  the spectra inside it; the status line and the histogram's axis say so when the grid had to widen
  the bin. The map, histogram, load curve and hover readout now share one rule for which cells were
  acquired, so a count can no longer appear in a cell drawn as a gap, and the hover readout no longer
  names column 0 for a cursor just left of the axis.

- **Dynamic Range points are the size Skyline draws them.** Skyline's Relative Abundance plot draws
  its points at 12 points, and ZedGraph then scales every symbol by the pane's size - the shorter of
  the width and 1.5x the height, over a 576 px base - so on a full-width pane its points are about
  19 px across and grow with the window. PRISM drew a fixed 9 px, chosen against ZedGraph's unscaled
  default, so the same cohort read as a thin gray line beside Skyline's cloud of proteins. The tool
  now applies Skyline's own rule to the pane it is drawing on, in rendered pixels, and the QC report's
  PNG applies it to its 1400 x 900 canvas. List members are drawn at the same size as the rest, as
  Skyline draws them - the color and being on top are what set them apart - and the selection ring
  sizes itself around the point. The y-axis label is also shorter: "Log10 mean abundance (median
  polish)" rather than "(median_polish rollup)", which a normal-height pane clipped mid-word.

- **A plot panel with no data no longer draws axes.** The three panels had drifted into three
  different empty states: QC Plots rendered nothing at all before the first run, leaving ScottPlot's
  raw default with an unstyled numbered grid to no scale; after a run it reset the chrome but never
  the axis limits, so a missing view inherited the previous plot's scale; and Spectrum density and
  Dynamic Range reset the whole control. All three now show one sentence saying why the panel is
  empty, on a panel with no axes at all - so an empty result cannot be misread as a flat measurement,
  and the numbers on it cannot be read as data that was never there.

- **The Ion Accounting and Spectrum Density panes did file system I/O on the UI thread.** The
  output directory is normally a network share, and against one that is slow, disconnected or
  holding a stale credential every one of these stopped the whole window for the SMB timeout with
  nothing on screen to say why:

  - the Ion Accounting nav probe ran a `File.Exists` from the output box's `TextChanged` - one
    blocking round trip **per keystroke**, plus one on every pane change;
  - the replicate list read the whole sample column of `ion_cycles.parquet` on the UI thread, after
    the load had already awaited its background read, so the window froze on a second read showing
    "Reading the ion accounting...";
  - Spectrum Density located `merged_data` on the UI thread before the background read it belongs
    in, and its isolation-window read probed paths a DOCUMENT recorded - which on a machine that is
    not the one the data was imported on are exactly the paths that no longer resolve, and a dead
    UNC path does not fail fast.

  All of it now runs in one background pass per pane. On a healthy share none of it was visible: the
  same reads measure 135 ms, 50 ms and 80 ms on a real 48-replicate cohort over SMB.

- **PRISM locked its own files, and lost a measurement to it.** A 48-replicate ion accounting run
  read every instrument file over several hours and saved none of the cycles: the write was refused
  because `ion_cycles.parquet` was "locked by another process". Nothing else was running. PRISM was
  the other process.

  Every parquet reader opened the file sharing Read and nothing else, so for as long as a plot was
  reading a file, nothing could write, replace or delete it - including the run that owned it. Every
  writer asked for exclusive access, which is refused while any other handle exists at all. And the
  cycles file was replaced after *every* replicate, so a 48-replicate run rewrote it 48 times and
  had 48 chances to collide with itself.

  Readers no longer take a file hostage, writers no longer demand exclusivity (a second writer is
  still kept out), and the real cycles file is now written once, at the end of a run, with progress
  going to a staging file beside it. An interrupted run loses nothing: whatever it measured is
  picked up on the next read, and the longer of the two is the one that survives.

  This reaches beyond ion accounting. Every parquet PRISM writes went through the same exclusive
  open, so viewing any output file while a run rewrote it could fail the write.

- **A small cohort labelled its worst replicate "Best".** The across-the-gradient panels are picked
  as best, median and worst by assigned fraction and captioned by position, but with three or fewer
  measured replicates the picker returned them worst-first - so the labels were exactly inverted.
  Four or more replicates were always right, which is why it survived: every test that checked the
  order used four.

- **A measurement no longer fails over a file name it cannot claim.** When the cycles file is held
  by something PRISM cannot argue with, the staged measurement is read where it lies - taking the
  newer of the two, so a stale file cannot shadow a current one - and the run reports success,
  because it succeeded. "Ion accounting failed" after forty-eight instrument files and several hours,
  for a rename, was the worst sentence in the product.

- **The error named a file that was not on disk.** `File.Move` reports the DESTINATION path whatever
  went wrong, so a locked staging file was reported as `ion_cycles.parquet is being used by another
  process` - about a file that did not exist. Two rounds of investigation went to the wrong file.
  The message now probes both files and names whichever is actually unavailable.

- **A process killed mid-save no longer loses the whole ion accounting measurement.** Parquet keeps
  its metadata at the end of the file, so adding a replicate overwrites the existing footer and
  writes a new one after it - and a process killed in between leaves a file that reads as **nothing**,
  not as everything up to that point. Measured: a four-replicate file truncated at its footer offset,
  which is exactly what the first write of an append does, gives 0 rows instead of 3,000. This
  repository documents ion accounting dying that way twice, to a native fault no catch block sees.

  The bytes each save is about to overwrite are now kept beside the file, and PRISM puts them back
  by itself the next time anything reads the directory - recovering every replicate that had been
  saved, and losing only the one that was in flight, which never finished. The footer is about 1,577
  bytes per replicate, so the protection costs roughly 188 MB over a 500-replicate run against 376 MB
  of data.

- **Reading the ion accounting cache while it is being written no longer reports an empty one.** For
  the width of one append the file has no footer - momentarily headless rather than damaged - and a
  reader landing there got an exception that every caller turned into "no cycles". Measured with a
  writer appending back to back, which is far harsher than a real measurement: 4.6% of opens failed,
  and a single 50 ms retry recovered every one. Reads now wait an append out.

  More importantly, **an unreadable cache is no longer reported as an empty one.** The two returned
  the same empty list, so a file that could not be opened - for a moment, or permanently - made the
  run conclude that no replicate had traces and measure the whole cohort again, reading every
  instrument file a second time without a word. That decision is still the safe one, but it now says
  why it is being made.

- **Starting a new measurement can no longer mix itself into the previous one.** Replacing the cycles
  file was a delete followed by an open. A delete refused by a scanner or an SMB holder was logged
  nowhere, and if that holder let go during the open's own retry window the new replicates were
  appended to the old measurement instead - two settings keys in one file, with nothing said. The
  replacement is now part of the open, so it either happens or does not.

- **A run that stops keeps the replicates it measured.** `ion_cycles.parquet` was written whole at
  the end of a run, with each progress save going to a staging file beside it, so a run killed at
  replicate 499 of 500 left its measurement under a name nothing was looking for. That happened: a
  48-replicate run died after its last progress save with a complete measurement on disk under the
  staged name. The file is now created at the start and one row group is appended per replicate, so
  it is valid, current and readable under its own name from the first replicate onward. A staging
  file left by the previous build is still picked up.

  Rewriting the whole accumulated table every time also cost more than it looked: 883 MB written to
  persist 36 MB across 48 replicates, and about 92 GB projected at 500. Appending writes each
  replicate once. It also shrank the file - the ~650-byte settings key repeated on all 49,010 rows of
  a single row group was 31.5 MB of that 36 MB, and one distinct value per row group encodes away, so
  the same 48 replicates now come to 4.8 MB.

- **`ion_cycles.parquet` carries the settings key its summary does.** The two files are written
  separately with the summary first, so a failure between them left a new summary beside an older
  set of traces - and replicate names are identical from run to run, so the name alone could not
  tell a stale trace from the current one. Reuse now checks the key and re-measures rather than
  mixing two measurements.

- **Every number beside a plot is the quantity the plot is drawing.** Ions and signal have different
  fractions - the ion count weights each scan by its ion injection time and the summed TIC does not
  - so with Quantity set to Signal the plot drew TIC percentages while the status line, the hover
  readout and the per-replicate line all reported ion-weighted ones. Three surfaces each picked a
  fraction independently; they now share one selector.

- **A fraction over 100% is never drawn.** It is impossible, so it means a defect - a units
  mismatch, an isolation scheme that does not match the acquisition, or claims merged too loosely -
  and clamping it would turn a visible bug into a plausible reading. The per-replicate fraction view
  now withholds those replicates entirely and says how many in the title; the gradient profile names
  impossible bins rather than dropping them, since dropping a point makes the line sail over exactly
  the stretch that is wrong.

- **The QC report's ion sections are readable.** The captions were paragraphs with the same settings
  sentence repeated under every panel; they are short bullets now, with the settings stated once per
  section. The across-the-gradient section drew three fraction panels and then one of absolute MS2
  ions, so two quantities in two units sat under one heading - it is now best, median and worst by
  fraction of acquired MS2 signal, and nothing else. A cache measured before the summed TIC was
  recorded still gets the section, in ions; either way the whole section is one quantity and the
  heading names it.

- **The acquired TIC is visible against the trace drawn over it.** It rendered at roughly 88% white
  on a white page, so the envelope the assigned trace is a fraction *of* was faint on screen and
  gone in print.

- **The tool zip is checked for what it must contain.** The Skyline Tool Store logo reached the zip
  through a single line in `SkylinePrism.App.csproj` and nothing verified it: move the image and
  every release would ship a tool that lists with a blank logo, with CI still green. The manifest,
  the logo and the instrument-file reader are now asserted by one script that the local ship gate,
  CI and the release workflow all run - in the release, before the artifact is uploaded.

- **The ion accounting plots stacked every redraw instead of replacing it.** ScottPlot's `Add`
  methods append, and legend entries ride on the plottables, so each change of view, level,
  replicate or bin width drew on top of the last - a legend that grew another copy of every series
  each time, with the earlier renders' data still underneath. The clear now lives in the three draw
  functions rather than at the call site, where three of four callers remembered it and one did not.

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
