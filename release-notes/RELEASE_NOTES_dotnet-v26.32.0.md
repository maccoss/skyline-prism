# Skyline-PRISM (C#) dotnet-v26.32.0 Release Notes

**Enrichment now separates proteins that went up from those that went down, and asks about the right
organism.** A mouse run used to be annotated as human. The Differential pane remembers your contrast,
errors can be copied, and a file left open in Excel no longer fails a run or a load. The QC report no
longer reads "Validation: FAILED" because normalization removed a loading difference between the
reference and QC injections, and a FAILED banner now says why. No peptide or protein quantity
changed. Enrichment results changed shape (see Breaking Changes).

## New Features

- **The Differential pane keeps your contrast.** Group by, the two arms ticked under each column, the
  Subject column and the Adjust-for ticks (with any switched type) are remembered by name. Switching
  between Protein and Peptide, re-running, retrying a failed load, or moving to another Group by and
  back now restores them wherever the data still has those columns and values. Before, changing Group by
  reseeded the arms and dropped every Adjust-for tick, and a failed load came back on the defaults.
- **Errors can be copied.** A failed run, or any unexpected error, now opens a window with the message,
  the full details in a selectable box, and **Copy to clipboard** and **Open log** buttons, in place of a
  message box whose text could only be copied with an undocumented Ctrl+C.
- **"Close the file, then Retry" instead of a failed run.** Before a run starts, PRISM checks whether
  another program (usually Excel) has a file open that the run will rewrite, such as
  `sample_metadata.csv` or `protein_groups.csv`. If one is open, PRISM names it and offers **Retry** and
  **Cancel**. Previously the run went all the way to Stage 5 and then failed with a stack trace. If a
  file is opened after the run has started, the failure offers the same Retry.
- **Enrichment separates up from down.** The Differential pane's Enrichment view, the quant report and
  `prism differential --report` now send the hits higher in B and those lower in B to g:Profiler as
  two lists by default, so each term says which way its genes moved. Before, they were pooled, so a
  pathway that went up was diluted by an unrelated one that went down. **Up and down separately |
  Pooled** sits beside the view tabs; on the command line it is
  `--enrichment-direction separate|pooled`. In the pane the bars are red (higher in B) and blue
  (lower), the table has a direction column, and clicking a term lists only the proteins that moved
  its way. `enrichment_terms.csv` gains a leading `direction` column.

## Bug Fixes

- **The QC report's PCA check failed runs that normalization had fixed.** The PCA QC-reference
  distance check now measures what batch correction did, not what normalization did. A 264-sample run read "Validation: FAILED" above two green CV improvements
  (reference +30%, QC +6%) because the ratio was 0.43, below the 0.5 cutoff. The ratio was measured
  from the raw rollup, so any loading difference between the reference and QC injections counted
  toward it, even though normalization removes that difference by design. It is now measured from
  the normalized, pre-ComBat peptides (`peptides_log2_internal.parquet`) to the corrected output,
  and still fails below 0.5, because batch correction pulling two different materials together is
  real damage. An output directory without that file gets the raw ratio as a note only.
- **A "Validation: FAILED" banner now names the reason** inside the banner. Before, the reason was
  only in a WARNINGS box below the control-CV table. That box now holds only warnings that did not
  decide the verdict, such as a reference CV that rose.
- **A file open in Excel no longer blocks PRISM from reading it.** With `sample_metadata.csv` open in
  Excel, switching the Differential pane between Protein and Peptide
  failed with "being used by another process" and emptied every picker, so nothing could be grouped.
  PRISM now reads its CSV and TSV inputs (sample metadata, additional metadata CSVs, `protein_groups.csv`,
  protein lists, Skyline reports, Carafe libraries) without demanding exclusive access, so a file open
  in Excel no longer blocks them. Readers that used to swallow the error, such as the QC pane's grouping
  columns and the ion accounting sample types, no longer come back empty.
- **Enrichment queried every run as human.** g:Profiler was always asked for `hsapiens`, so a mouse run
  was annotated with human GO and Reactome terms, and mouse-only genes dropped out of both the query
  and the background without a word. The organism is now inferred from the species code most of the
  run's UniProt entry names carry (`ICAM1_MOUSE` gives `mmusculus`; contaminants such as human
  keratins do not decide it), named in the status line and the report, and can be set with
  `--organism`. On a KO vs KO_control mouse plasma contrast, the hits higher in KO now come back
  enriched for lysosome and Reactome "Neutrophil degranulation" (Ctsg, Elane, Mpo, Prtn3, Prg2).

## Breaking Changes

- **Enrichment results changed shape.** Enrichment now defaults to separate up and down queries, so the
  same contrast can return different terms than before; pass `--enrichment-direction pooled` (or pick
  **Pooled** in the pane) for the earlier single list. `enrichment_terms.csv` gains a leading
  `direction` column, which shifts every other column one place to the right: a script reading it by
  position must read by name or add one.
