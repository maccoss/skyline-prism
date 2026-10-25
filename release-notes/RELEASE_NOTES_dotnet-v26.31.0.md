# Skyline-PRISM (C#) dotnet-v26.31.0 Release Notes

**Selection now works both ways between PRISM and Skyline, modified peptides included.** Picking a
protein or peptide in Skyline rings it on the Volcano as it already did on the Dynamic Range plot, and
clicking a carbamidomethylated (or any modified) peptide in PRISM now selects it in Skyline - it never
did before, because the two name modifications differently. The Differential pane keeps its comparison
when you switch between protein and peptide level, and shows a progress bar for anything slow. No
quantity PRISM computes changed. One output key was renamed (see Breaking Changes).

## New Features

- **The Volcano now follows Skyline's selection.** Selecting a protein or peptide in Skyline's Targets
  tree or a document grid rings it on the Volcano and selects its row in the hit table; the per-feature
  boxplot follows it too if it is open. Until now only the Dynamic Range plot followed Skyline, and the
  Volcano only drove Skyline the other way.
- **Clicking a Dynamic Range point opens the per-feature window**, as a Volcano click does: that
  protein or peptide's log2 abundance in each replicate the plot averages, one box per sample type, with
  every replicate named on hover. It still selects the point in Skyline as well.
- **The Differential pane shows a progress bar** beside its status line whenever a load, a contrast or
  a quant report runs longer than a second. Nothing showed before, so a slow peptide-level contrast
  looked like a click that had done nothing.
- **The Detection view explains itself.** Its tab has a tooltip, and `docs/skyline-tool.md` now leads
  with the question it answers - is a peptide *seen* more often in one group than the other - before
  the choice of test.
- **"Clinical CSV" is now "Additional metadata"**, in the tool and on the command line, since the file
  can carry any per-sample annotation the Skyline document does not. The tool's input is relabeled and
  `prism differential --clinical` is now `--additional-metadata`; it is the same join either way.
  `--clinical` still works, with a deprecation warning, so the command recorded in every quant report
  written before this release still regenerates it.

## Bug Fixes

- **Clicking a modified peptide on the Volcano or the Dynamic Range plot now selects it in Skyline.**
  PRISM names peptides in the report's Unimod form (`C(unimod:4)`) and Skyline's tree names them by mass
  (`C[+57.021464]`), so every peptide carrying a modification - carbamidomethyl cysteine included -
  matched nothing and the click selected nothing. Peptides are now also matched by where their
  modifications sit, after the exact name. The same fix makes following Skyline's selection work for
  modified peptides.
- **Switching the Differential pane between protein and peptide level no longer forgets the
  comparison.** The Group by column, both arms and the Adjust for covariates (with any type you
  switched) carry over, and the contrast reruns at the new level. Before, the pane reset to the first two
  values of the default column and waited.
- **Dynamic Range points are the same size on every computer.** They were sized from the pane's size in
  rendered pixels while all the text on the plot is a fixed pixel size, so on a high-DPI display or a
  large monitor the points came out two to three times larger relative to everything else (37 px at 200%
  scaling against 19 px at 100% for the same window). They are now sized from the pane's
  device-independent size and held between 8 and 18 px.
- **The Markers pane names the right level when no panel member matched.** Its "None of ... matched a
  protein feature" message read a level that was stale whenever the matrix came from the Differential
  pane, so it could say "protein" over a peptide-level evaluation.

## Breaking Changes

- **`quant_parameters.yaml` records a joined table under `additional_metadata_csv`**, not
  `clinical_csv` (and `quant_parameters.json` under `additional_metadata_csvs`, not `clinical_csvs`), and
  the command a quant report records uses `--additional-metadata`. A script that reads either key needs
  the new name; reports already written keep the old one.
