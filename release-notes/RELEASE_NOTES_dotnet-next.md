# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

## Bug Fixes

- A protein with no known gene was labeled "NA" in the volcano, the hit table, boxplot titles, the
  Dynamic Range plot and the `label` column of `differential.csv`, so on a non-model organism most
  hits carried the same useless name. It is now labeled by its UniProt accession, and the
  protein-group id where it has neither. The `NA` placeholder also no longer reaches the `gene`
  column, the hover readout, or the gene list sent to g:Profiler for enrichment, which had also
  split "N/A" into two "symbols". The output files still write `NA`, as before.
- The quant report's volcano was unreadable once pasted onto a slide: 5-6 px points and 11 pt labels
  beside 24-30 pt axis text, labels piled on top of each other where hits sat close together, a label
  at the edge cut off, and a long contrast title running off both sides. Points and labels are now
  sized to the axis text, each label goes on a free side of its point where it cannot be read as a
  neighbour's (a hit with no free side is left unnamed rather than overprinted), up to 10 labels,
  and a long title shrinks to fit. The detection volcano got the same point sizes and title fix.
- The volcano drew every hit red, while the Enrichment view beside it uses red for higher in B and
  blue for lower - so a red point and a red bar meant different things. Volcano hits are now red
  when higher in B and blue when lower, in the Differential pane and the quant report alike, from
  the same color definition the enrichment bars use. The threshold guide lines, which had taken
  ScottPlot's cycle colors (a red line beside the blue hits in the pane), are now neutral grey on
  every volcano.

## Performance

## Breaking Changes
