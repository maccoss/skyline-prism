# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

- Every statistical option in the Differential pane now explains itself when hovered: each design
  (Unpaired, Paired, Blocked by subject), level, test (moderated t, Welch, Student, Mann-Whitney,
  paired t, Wilcoxon), variance prior, multiple-testing correction and choice of p-value says what it
  does, what it assumes and when to use it. Before, each picker had one paragraph on the control as
  a whole and the choices had none, so on the segmented controls the help was hard to reach at all.

## Bug Fixes

- A protein with no known gene was labeled "NA" in the volcano, the hit table, boxplot titles, the
  Dynamic Range plot and the `label` column of `differential.csv`, so on a non-model organism most
  hits carried the same useless name. It is now labeled by its UniProt accession, and the
  protein-group id where it has neither. The `NA` placeholder also no longer reaches the `gene`
  column, the hover readout, or the gene list sent to g:Profiler for enrichment, which had also
  split "N/A" into two "symbols". The output files still write `NA`, as before. A shared peptide
  whose first protein group has no known gene or name keeps an empty slot for it, so each name stays
  beside its own group in the Dynamic Range readout and in `differential.csv`; a protein with no
  gene, accession or UniProt id is labeled by its protein-group id in `differential.csv` and the
  hover readout too, not left blank; and a placeholder after a comma, bar or space ("ALB,N/A") no
  longer reaches g:Profiler as the symbols N and A.
- The quant report's volcano was unreadable once pasted onto a slide: 5-6 px points and 11 pt labels
  beside 24-30 pt axis text, labels piled on top of each other where hits sat close together, a label
  at the edge cut off, and a long contrast title running off both sides. Points and labels are now
  sized to the axis text, each label goes on a free side of its point where it cannot be read as a
  neighbour's (a hit with no free side is left unnamed rather than overprinted), up to 10 labels,
  and a long title shrinks to fit. The detection volcano got the same point sizes and title fix.
- The volcano drew every hit red, while the Enrichment view beside it uses red for higher in B and
  blue for lower - so a red point and a red bar meant different things. Volcano hits are now red
  when higher in B and blue when lower, in the Differential pane and the quant report alike, from
  the same color definition the enrichment bars use. The detection volcanoes, which drew every hit
  green, follow the same rule (red for detected more often in B). The threshold guide lines, which
  had taken ScottPlot's cycle colors (a red line beside the blue hits in the pane), are now neutral
  grey on every volcano, and the Volcano's selection ring is orange, since a red ring around a red
  hit was hard to find.
- The Dynamic Range plot's labels criss-crossed once a list put more than a handful on the steep
  top of the curve: leader lines crossed each other and ran through other labels, and labels sat on
  the curve over the points they named. Labels are now laid out together, using the open space on
  both sides of the curve, by simulated annealing over a few hundred spots per label, and measured
  as ScottPlot draws them rather than estimated. On a 26-protein plasma list at a typical pane size
  no leaders cross, no labels overlap and nothing sits on the curve; on a small pane a crowded list
  runs more leaders across the curve, and a leader can pass through another label. Placing 26
  labels takes about half a second; a long list gets a bounded amount of work.
- Starting any copy of PRISM re-pointed the Start Menu shortcut at itself - including a development
  build, which an ordinary `dotnet build` makes without the instrument-file reader. Once one had
  been started, every Start Menu launch ran it, and ion accounting reported "this build has no
  instrument-file reader" although the installed Skyline tool had one. Only a copy Skyline
  installed (in the `Tools` folder beside Skyline itself) re-points the shortcut now, and never to
  an older version than it points at - PRISM in a release Skyline no longer takes it from
  Skyline-daily's newer copy. Any other copy leaves it alone and says why in the diagnostic log.
  Creating the shortcut from the menu still works from any copy.
## Performance

## Breaking Changes
