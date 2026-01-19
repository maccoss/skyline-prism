# Skyline-PRISM v26.1.3 Release Notes

## Overview

Patch release fixing protein description extraction. When a FASTA file is provided, protein descriptions are now properly extracted from FASTA headers. When no FASTA is available, descriptions display "NA" instead of showing raw UniProt identifiers.

## Bug Fixes

### Fixed: Protein Description Parsing from FASTA

**Problem**: The `leading_description` column in `corrected_proteins.parquet` was showing full FASTA header text including `OS=`, `GN=`, `PE=`, `SV=` tags (e.g., `"Inactive tyrosine-protein kinase PRAG1 OS=Homo sapiens OX=9606 GN=PRAG1 PE=1 SV=4"`) instead of just the clean protein description.

**Solution**: Updated FASTA header parsing to extract only the description text before UniProt metadata tags.

**Before** (v26.1.2):
```
leading_description: "Inactive tyrosine-protein kinase PRAG1 OS=Homo sapiens OX=9606 GN=PRAG1 PE=1 SV=4"
```

**After** (v26.1.3):
```
leading_description: "Inactive tyrosine-protein kinase PRAG1"
```

### Fixed: Gene Name and Description Fall Back to "NA" When Unavailable

**Problem**: When using Skyline CSV-based parsimony (no FASTA configured), or when FASTA entries lack `GN=` tags, the `leading_gene_name` and `leading_description` columns displayed raw UniProt identifiers or empty values.

**Solution**: 
- `leading_name`: Now shows clean entry name (e.g., `A0A075B5J9_MOUSE`)
- `leading_gene_name`: Now shows `"NA"` when no gene name is available (no `GN=` tag in FASTA or no Gene column in Skyline)
- `leading_description`: Now shows `"NA"` when no proper description is available

## New Feature

### FASTA-Based Parsimony Support

When `fasta_path` is configured in the parsimony section, PRISM now uses the FASTA file to extract:
- **Proper protein descriptions** from FASTA headers
- **Gene names** from `GN=` tags
- **Protein accessions** and **entry names**

To enable, add to your config YAML:

```yaml
parsimony:
  fasta_path: "/path/to/your/search_database.fasta"
  shared_peptide_handling: "all_groups"
```

If the FASTA file is not found, PRISM falls back to Skyline CSV-based parsimony with a warning.

## Technical Details

### Files Modified

- `skyline_prism/fasta.py`: Updated `_parse_header()` to extract clean descriptions
- `skyline_prism/parsimony.py`: 
  - Added `_extract_protein_entry_name()` helper function
  - Updated `build_peptide_protein_map()` to set gene names and descriptions to "NA" when unavailable
  - Updated `build_peptide_protein_map_from_fasta()` to return gene names and descriptions
- `skyline_prism/cli.py`: Added FASTA-based parsimony path when `fasta_path` is configured

### Column Values

| Scenario | `leading_name` | `leading_gene_name` | `leading_description` |
|----------|----------------|---------------------|----------------------|
| FASTA with GN= tag | Gene name or entry name | Gene name from GN= | Clean description |
| FASTA without GN= tag | Entry name | `"NA"` | Clean description |
| No FASTA, UniProt identifier | Entry name (e.g., `S12A2_MOUSE`) | `"NA"` | `"NA"` |
| No FASTA, other identifier | Original value | `"NA"` | `"NA"` |

## Testing

- All 43 parsimony and FASTA tests passing
- Verified description extraction with UniProt human and mouse FASTA files
- Backward compatible with existing Skyline CSV workflows

## Upgrade Notes

- No breaking changes from v26.1.2
- Existing workflows will continue to work
- To get proper protein descriptions, configure `fasta_path` in your config YAML
