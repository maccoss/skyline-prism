# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

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

## Bug Fixes

- With `sample_metadata.csv` open in Excel, switching the Differential pane between Protein and Peptide
  failed with "being used by another process" and emptied every picker, so nothing could be grouped.
  PRISM now reads its CSV and TSV inputs (sample metadata, additional metadata CSVs, `protein_groups.csv`,
  protein lists, Skyline reports, Carafe libraries) without demanding exclusive access, so a file open
  in Excel no longer blocks them. Readers that used to swallow the error, such as the QC pane's grouping
  columns and the ion accounting sample types, no longer come back empty.

## Performance

## Breaking Changes
