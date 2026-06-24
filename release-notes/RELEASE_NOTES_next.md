# Skyline-PRISM vNEXT Release Notes

Working draft. Rename to `RELEASE_NOTES_v{version}.md` at release time.

## New Features

### "Batch source" line in `qc_report.html` and `metadata.json`

Made it explicit which of the three batch-label sources actually drove ComBat for a given run, since the same words ("batch") can come from different places and the choice is automatic. `qc_report.html` now shows a one-line `Batch source:` field under Dataset Summary, naming the resolved path (`source documents`, `metadata`, or `acquisition time estimation`) and the human-readable description of where the labels came from. `metadata.json` records the same value under `processing_parameters.batch_correction.batch_source` for provenance.

### Documentation: how PRISM resolves batch labels

`docs/methods.md` has a new "Where do batch labels come from?" subsection under Batch Correction that lays out the priority chain (Source Document column → metadata `batch` column → acquisition-time estimation), the accepted metadata column aliases (`batch`, `Batch`, `Batch Name`), and the log lines users can grep for to confirm which path was taken. Both `prism config-template` and `--minimal` now embed the same short summary as a comment under the `batch_correction:` block so it shows up in any config a user generates.

- **Files modified**: `skyline_prism/cli.py`, `skyline_prism/validation.py`, `docs/methods.md`

## Bug Fixes

<!-- none yet -->

## Performance

<!-- none yet -->

## Breaking Changes

<!-- none yet -->
