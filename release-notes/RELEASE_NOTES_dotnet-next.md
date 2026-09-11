# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

- **The MS2 signal accounting can now report a fraction, not just a total.** The plot has always
  answered "how much MS2 signal does this analysis assign to a peptide"; it can now answer "what
  fraction of what the instrument actually acquired". The denominator is the one quantity no Skyline
  export carries - `TicArea` is MS1 by construction and must never be substituted - so it is read
  from the instrument data files by a new command:

  ```
  prism ms2-signal -d <output-dir> -r <raw-dir> [--max 3]
  ```

  That writes `ms2_signal.parquet`; re-run `prism qc -d` and the accounting plot gains a background
  bar for acquired MS2, with the assigned bars sitting inside it, and both the title and the caption
  report the median assigned/acquired.

  Deliberately its own command rather than part of `prism run`: a cohort is hundreds of gigabytes of
  raw data, usually over a network share, against a pipeline that otherwise reads one exported
  report. `--max` reads the first N files for a spot check. Replicates are matched to files on the
  file stem, which handles the prefixes acquisition software adds.

  Without it nothing changes and no number is wrong - the plot shows assigned signal, and the
  caption now says explicitly that it is not a fraction of what was acquired, so the bars cannot be
  misread as full coverage. A replicate whose file could not be read keeps no denominator rather
  than being counted as zero, and the plot says how many replicates the fraction covers.

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

## Performance

## Breaking Changes
