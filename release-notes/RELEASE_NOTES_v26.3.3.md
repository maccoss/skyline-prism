# Skyline-PRISM v26.3.3 Release Notes

## Overview

Minor release changing output file formats from TSV to CSV for better Excel compatibility.

## Changes

### Output Files Changed from TSV to CSV

Changed the two TSV output files to CSV format for better compatibility with Excel and other tools:

- `protein_groups.tsv` → `protein_groups.csv`
- `sample_metadata.tsv` → `sample_metadata.csv`

**Rationale:**
- Excel auto-opens CSV files correctly with double-click
- TSV files often require manual import steps in Excel
- All PRISM input files are already CSV, so this provides consistency
- No data format changes - only the delimiter changed from tab to comma

**Updated output directory structure:**

```text
output_dir/
├── corrected_peptides.parquet
├── corrected_proteins.parquet
├── peptides_rollup.parquet
├── proteins_raw.parquet
├── merged_data.parquet
├── protein_groups.csv              # Changed from .tsv
├── sample_metadata.csv             # Changed from .tsv
├── metadata.json
├── qc_report.html
└── prism_run_YYYYMMDD_HHMMSS.log
```

**Code changes:**
- `parsimony.py`: `export_protein_groups()` now writes CSV
- `cli.py`: Sample metadata output uses CSV format
- `chunked_processing.py`: Protein groups output uses CSV format
- `gui/viewer.py`: Updated to read CSV format files

### Documentation Updates

Updated all documentation to reflect the CSV file format change:
- `docs/output_files.md`
- `README.md`
- `AGENTS.md`
- `SPECIFICATION.md`

## Testing

- All 307 tests passing
- Verified CSV output opens correctly in Excel

## Upgrade Notes

- **Breaking change for scripts**: If you have scripts that read `protein_groups.tsv` or `sample_metadata.tsv`, update them to read the `.csv` versions instead
- The `--from-provenance` feature will still work, but will produce CSV outputs even when re-running from older provenance files

