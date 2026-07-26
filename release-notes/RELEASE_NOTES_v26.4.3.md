# Skyline-PRISM v26.4.3 Release Notes

Bug-fix release: FASTA-based protein parsimony is now enzyme-aware, so shared peptides are no longer
folded into homologous proteins that cannot enzymatically produce them. Also surfaces which batch-label
source drove ComBat in the QC report and provenance.

## New Features

### "Batch source" line in `qc_report.html` and `metadata.json`

Made it explicit which of the three batch-label sources actually drove ComBat for a given run, since the same words ("batch") can come from different places and the choice is automatic. `qc_report.html` now shows a one-line `Batch source:` field under Dataset Summary, naming the resolved path (`source documents`, `metadata`, or `acquisition time estimation`) and the human-readable description of where the labels came from. `metadata.json` records the same value under `processing_parameters.batch_correction.batch_source` for provenance.

### Documentation: how PRISM resolves batch labels

`docs/methods.md` has a new "Where do batch labels come from?" subsection under Batch Correction that lays out the priority chain (Source Document column → metadata `batch` column → acquisition-time estimation), the accepted metadata column aliases (`batch`, `Batch`, `Batch Name`), and the log lines users can grep for to confirm which path was taken. Both `prism config-template` and `--minimal` now embed the same short summary as a comment under the `batch_correction:` block so it shows up in any config a user generates.

- **Files modified**: `skyline_prism/cli.py`, `skyline_prism/validation.py`, `docs/methods.md`

## Bug Fixes

### Enzyme-aware FASTA parsimony: shared peptides no longer attached to homologs that can't produce them

FASTA-based peptide-to-protein mapping (`parsimony.fasta_path`) used pure subsequence containment, which
attached a peptide to any protein whose sequence contained it — even when the flanking cleavage site
required to liberate that peptide is absent in that protein. This silently contaminated protein rollups
and per-peptide residuals for homologous families (synucleins, immunoglobulin variable regions,
paralogs) and double-counted single proteoforms downstream. The map is now enzyme-aware: a peptide is
attached to a protein only when it occurs there with termini consistent with the digestion enzyme, via
two new config keys under `parsimony`:

- `enzyme` (default `trypsin`): `trypsin`, `trypsin/p` (K/R even before P, e.g. DIA-NN), `lysc`, `lysn`,
  `argc`, `aspn`, `gluc`, `chymotrypsin`, or `nonspecific`.
- `enzyme_specificity` (default `full`): `full` (both termini cleavage-consistent), `semi` (either
  terminus), or `none` (legacy pure-substring behavior).

Initiator-methionine excision is handled so real N-terminal peptides are not dropped. Example:
`AKEGVVAAAEK` is a substring of beta-synuclein but is preceded there by `M` (not `K/R`), so it is now
correctly kept proteotypic to alpha-synuclein. Only the FASTA path is affected; the Skyline Protein
Accession column was already enzyme-aware.

- **Files modified**: `skyline_prism/fasta.py`, `skyline_prism/parsimony.py`, `skyline_prism/cli.py`, `docs/parameters.md`
