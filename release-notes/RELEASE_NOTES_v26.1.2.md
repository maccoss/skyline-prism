# Skyline-PRISM v26.1.2 Release Notes

## Overview

Significant enhancement to protein output format with UniProt identifiers, gene names, and protein descriptions. Replaces meaningless PG#### identifiers with informative protein metadata sourced directly from Skyline CSV exports.

## Major Changes

### Breaking Change: Enhanced Protein Output Format

The `corrected_proteins.parquet` file now includes three new metadata columns populated from Skyline CSV data:

- **`leading_uniprot_id`**: UniProt accession (e.g., `sp|P62754|RS6_MOUSE` or `tr|A0A075B5L8|A0A075B5L8_MOUSE`)
  - Source: `Protein Accession` column from Skyline report
  - Provides stable, cross-database protein identifiers

- **`leading_gene_name`**: Gene symbol (e.g., `RPS6`, `Igkv4-79`)
  - Source: `Protein Gene` column from Skyline report
  - Enables rapid protein identification

- **`leading_description`**: Full protein name/description (e.g., `Ribosomal protein S6` or `Immunoglobulin kappa variable region`)
  - Source: `Protein` column from Skyline report
  - Provides human-readable protein information

**Old format** (v26.1.1 and earlier):

```csv
protein_group  leading_protein  leading_name    sample1  sample2  ...
PG0001         P62754          Ribosomal...    12450    11200    ...
```

**New format** (v26.1.2):

```csv
protein_group  leading_uniprot_id  leading_gene_name  leading_description              sample1  sample2  ...
PG0001         sp|P62754|RS6_...   RPS6              Ribosomal protein S6 (RSP6)      12450    11200    ...
```

### Implementation Details

- Metadata extracted during parsimony stage from `Protein Accession`, `Protein Gene`, and `Protein` columns in Skyline CSV
- Propagated through peptide→protein rollup and batch correction stages
- Both peptide and protein parquet files remain in LINEAR scale (as per specification)
- Protein groups are still identified by `protein_group` column (PG####) for traceability

### Affected Files

- `corrected_proteins.parquet`: Added 3 new columns
- `protein_groups.tsv`: Added columns `LeadingUniProtID`, `LeadingGeneName`, `LeadingDescription` to export
- Config parameter additions: Optional `protein_gene_column` and `protein_description_column` in config YAML (default to standard Skyline column names)

## Backward Compatibility

This is a **BREAKING CHANGE** for downstream analysis that reads `corrected_proteins.parquet`:

- Old scripts expecting only `[protein_group, leading_protein, leading_name, ...]` will still work (extra columns are ignored)
- PivotTables and Excel imports will include additional columns
- Column order has changed: metadata columns now include UniProt/gene/description before sample columns

## Testing

- **294 tests passing** (updated to cover new metadata flow)
- **61% overall coverage**
- Validated on 238 samples across 3 batches with ~19.8M transition measurements
- Verified UniProt IDs, gene names, and descriptions appear in output parquet files

## Notes

All Skyline reports must include the standard columns:

- `Protein Accession` (or configured via `protein_column`)
- `Protein Gene` (or configured via `protein_gene_column`)
- `Protein` (or configured via `protein_description_column`)

If your Skyline export is missing these columns, the pipeline will handle gracefully with empty strings for missing metadata. No data loss occurs.
