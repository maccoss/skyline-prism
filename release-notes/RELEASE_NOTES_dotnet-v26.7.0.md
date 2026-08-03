# Skyline-PRISM (C#) dotnet-v26.7.0 Release Notes

Feature release: the Skyline external tool becomes a general PRISM front end. It now runs **standalone**
without Skyline, and it can combine **several Skyline documents** - open or closed - into one multi-batch
run, so a study split one document per plate no longer needs each report exported by hand. Also fixes two
bugs that silently corrupted multi-document metadata: replicate annotation columns never reached the
metadata report, and identically named reference/QC replicates in different documents overwrote each
other, which could collapse two batches into one and skip ComBat without a word.

## New Features

- **The Skyline external tool now runs standalone.** `SkylinePrism.exe` no longer requires a Skyline
  connection: started without one it opens in standalone mode, where the new **Inputs** tab takes
  already-exported PRISM reports (`.parquet` / `.csv` / `.tsv`) and runs the full pipeline plus the
  interactive QC plots with no Skyline involved. Launched from Skyline's Tools menu it behaves as
  before, seeding the Inputs list with the open document.
- **Several Skyline documents in one run.** The Inputs tab combines any number of documents - one per
  batch/plate - so multi-batch cohorts split across documents no longer need each report exported by
  hand. Each input gets an editable **Batch label** (defaulting to the document name, de-duplicated)
  that becomes its Source Document / batch label, and the whole set is merged and batch-corrected as a
  single cohort. Inputs may be mixed freely:
  - documents open in the launching Skyline, or in **any other running Skyline instance**
    (`Add open document...`), exported over JSON-RPC as parquet;
  - **closed `.sky` documents** (`Add Skyline document (.sky)...`), exported headlessly by driving the
    installed Skyline with no UI (the SkylineRunner mechanism, reimplemented so it finds either Skyline
    or Skyline-daily). This yields **parquet** - roughly 15x smaller than the CSV and much faster to
    merge. If no installed Skyline shortcut is found it falls back to `SkylineCmd.exe` (discovered in
    the ClickOnce application folder, overridable with `PRISM_SKYLINECMD`), which can only produce CSV
    because `SkylineCmd.exe.config` omits the Parquet.Net assembly bindings that `Skyline.exe.config`
    carries. The merge reads CSV and parquet inputs together, so the two kinds mix freely in one run.
  - already-exported report files, which are read in place.
- **Replicate metadata for closed documents.** PRISM reads the `.sky` header (never modifying it) for
  the document's replicate-targeted Document Annotations, generates a matching PRISM-Replicates report,
  and exports it with `SkylineCmd` - so a closed document yields the same sample types and annotation
  columns as an open one. The digestion enzyme for FASTA parsimony is read from the same header.
- **Show Command Line** now emits the full multi-input `prism run -i ... -m ...` command, with the
  positional input/metadata pairing spelled out.
- **Start Menu shortcut for the standalone GUI.** "Add Start Menu shortcut" creates a shortcut that opens
  PRISM without Skyline. Because Skyline installs its tools inside its own version-stamped program folder,
  the shortcut is automatically re-pointed at the current build each time PRISM is started from Skyline's
  Tools menu, so a Skyline update does not strand it.

## Bug Fixes

- **Replicate annotation columns were silently dropped from the metadata report.** The generated
  PRISM-Replicates report addressed annotations as `annotation_<Name>`, which Skyline rejects when it
  parses the view's column path (`Error parsing annotation_Plate at location 10: Invalid character _`) -
  the `annotation_` prefix's underscore is not legal in a bare property path. The column name must be
  quoted (`"annotation_Plate"`). Previously the batch-annotation column requested in the GUI never
  reached the report.
- **Identically named replicates in different documents overwrote each other.** Replicate metadata was
  merged across files keyed by replicate name alone, so the last file won. Because reference and QC
  injections are normally named the same in every plate's document, combining documents could collapse
  two batches into one label (silently skipping ComBat) and give every plate the last document's sample
  types. Metadata rows are now also keyed by source document (`<replicate>__@__<document>`, matching the
  merged sample ID) and those entries take precedence.

## Performance

- **Closed-document export is parquet, not CSV.** Driving the installed Skyline headlessly (rather than
  `SkylineCmd.exe`) means the report writer can emit parquet, which is typed and roughly **15x smaller** -
  61 KB versus 946 KB on a test document - and correspondingly faster for the merge to read.

## Breaking Changes

- Reports exported by the tool are now named after their input's batch label
  (`<label>.parquet` / `<label>.csv`, with metadata as `<label>.metadata.csv`) instead of the fixed
  `PRISM.parquet` / `Metadata.csv`, so several documents can coexist in `skyline-reports/`. The stem is
  what the merge derives each input's Source Document label from. Single-document runs launched from
  Skyline use the label `PRISM`, so their transition report keeps its old name; only the metadata file
  is renamed (`Metadata.csv` -> `PRISM.metadata.csv`). Reading QC annotations still falls back to a
  legacy `Metadata.csv` if present.
