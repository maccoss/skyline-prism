# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

- **Enrichment separates up from down.** The Differential pane's Enrichment view, the quant report and
  `prism differential --report` now send the hits higher in B and those lower in B to g:Profiler as
  two lists by default, so each term says which way its genes moved. Before, they were pooled, so a
  pathway that went up was diluted by an unrelated one that went down. **Up and down separately |
  Pooled** sits beside the view tabs; on the command line it is
  `--enrichment-direction separate|pooled`. In the pane the bars are red (higher in B) and blue
  (lower), the table has a direction column, and clicking a term lists only the proteins that moved
  its way. `enrichment_terms.csv` gains a leading `direction` column.

## Bug Fixes

- **Enrichment queried every run as human.** g:Profiler was always asked for `hsapiens`, so a mouse run
  was annotated with human GO and Reactome terms, and mouse-only genes dropped out of both the query
  and the background without a word. The organism is now inferred from the species code most of the
  run's UniProt entry names carry (`ICAM1_MOUSE` gives `mmusculus`; contaminants such as human
  keratins do not decide it), named in the status line and the report, and can be set with
  `--organism`. On a KO vs KO_control mouse plasma contrast, the hits higher in KO now come back
  enriched for lysosome and Reactome "Neutrophil degranulation" (Ctsg, Elane, Mpo, Prtn3, Prg2).

## Performance

## Breaking Changes

- **Enrichment results changed shape.** Enrichment now defaults to separate up and down queries, so the
  same contrast can return different terms than before; pass `--enrichment-direction pooled` (or pick
  **Pooled** in the pane) for the earlier single list. `enrichment_terms.csv` gains a leading
  `direction` column, which shifts every other column one place to the right: a script reading it by
  position must read by name or add one.
