# Skyline-PRISM (C#) dotnet-v26.6.1 Release Notes

Bug-fix release: FASTA-based protein parsimony is now enzyme-aware, so shared peptides are no longer
folded into homologous proteins that cannot enzymatically produce them; the Skyline external tool reads
the digestion enzyme from the open document so the map matches the search that produced the data.

## New Features

### Skyline external tool reads the digestion enzyme from the document

When the tool runs FASTA-based parsimony, it now derives the digestion enzyme from the open Skyline
document's Peptide Settings (via the `Enzymes` settings list over JSON-RPC) instead of a hardcoded
default, so the enzyme-aware peptide-to-protein map matches the search that produced the data. If the
document's enzyme has no PRISM equivalent (or can't be read), the tool keeps the config default.

## Bug Fixes

### Enzyme-aware FASTA parsimony: shared peptides are no longer attached to homologs that can't produce them

FASTA-based peptide-to-protein mapping previously used pure subsequence containment, which attached a
peptide to any protein whose sequence contained it — even when the flanking cleavage site required to
liberate that peptide is absent in that protein. This silently contaminated protein rollups and
per-peptide residuals for homologous families (synucleins, immunoglobulin variable regions, paralogs).
The map is now enzyme-aware: a peptide is attached to a protein only when it occurs there with termini
consistent with the digestion enzyme (`parsimony.enzyme` / `parsimony.enzyme_specificity`, defaults
`trypsin` / `full`), with initiator-methionine excision handled. Example: `AKEGVVAAAEK` is a substring
of beta-synuclein but is preceded there by `M` (not `K/R`), so it is now correctly kept proteotypic to
alpha-synuclein. Set `enzyme_specificity: none` to restore the legacy pure-substring behavior. Only the
FASTA path is affected; the Skyline Protein Accession column was already enzyme-aware.
