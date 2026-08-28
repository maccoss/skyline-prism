# Skyline-PRISM (C#) dotnet-v26.17.0 Release Notes

The replicate annotations - the study design PRISM was already reading out of Skyline - now reach
`sample_metadata.csv` instead of being discarded on the way.

## New Features

- **`sample_metadata.csv` now carries the whole Replicates report, not just the three fields PRISM
  interprets.** It held `sample_id, sample, sample_type, batch` and dropped every replicate annotation -
  so a cohort's design (Subject, Timepoint, responder status, days between draws) never reached the
  output directory in a form downstream analysis could use, even though the tool had exported it and the
  QC tab could group plots by it. Every other column of the report now follows those four, verbatim,
  under its own name.

  The four original columns keep their names and positions, so existing readers are unaffected. The
  report's own `SampleType` is carried alongside PRISM's `sample_type` rather than treated as a
  duplicate of it: the mapping is lossy - `Solvent`, `Blank` and `Double Blank` all become `blank`, and
  Skyline's sample-type list is editable, so a type someone adds becomes `experimental` - and the raw
  value is the only one that survives a custom vocabulary. For a cohort of several documents - each with
  its own Replicates grid, since annotations are per-document settings - the header is the union of
  their columns in first-seen order, and a replicate from a document that does not define a column gets
  an empty cell rather than the other document's value. An annotation whose name clashes exactly with a
  reserved column is written as `<name> (report)` rather than dropped.

## Bug Fixes

- **Every reader of `sample_metadata.csv` parsed it with `split(',')`.** Annotation values contain commas, and
  so could a batch label or a sample name (the writer has always quoted those), which silently shifted
  the fields and dropped the sample from the QC report's control groups. All three readers are now
  quote-aware (`CsvLine`).
