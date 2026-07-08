# Skyline-PRISM (C#) dotnet-v26.6.0 Release Notes

Skyline external tool usability improvements: a more informative interactive PCA plot and cleaner,
grid-accurate replicate metadata.

## New Features

### Skyline external tool
- **Hover a PCA point to see its replicate.** The interactive PCA QC plot now shows the replicate name
  under the cursor, with a highlight ring on the nearest point. When every replicate shares the same
  `__@__<batch>` suffix (a single merged source document), the redundant suffix is dropped from the
  label; it is kept when batches differ so replicates from different batches stay distinguishable.
- **Browse dialogs open in the output directory.** The output-folder, spectral-library, and
  open-provenance dialogs now start in the nearest existing directory at or above the "Output directory"
  box, instead of the process default.

## Bug Fixes

### Skyline external tool
- **The default "Replicates" metadata now matches the actual Replicates grid.** The default metadata
  report (relabeled `Replicates`, formerly `PRISM-Replicates`) reconstructs Skyline's built-in Replicates
  document grid from its real columns — Sample Type, Analyte Concentration, and every replicate
  annotation — and no longer force-adds `BatchName`, `SampleDilutionFactor`, or `SampleId`, which the
  built-in Replicates view does not show. This removes those phantom fields from the QC "Group by" list so
  it matches what you see in Skyline. Selecting any other named report still exports it normally.
- **The PRISM-Replicates report is no longer installed into the document eagerly.** It cluttered the
  document's saved-report list; it is now added on demand only as a fallback when the live grid read
  fails.
