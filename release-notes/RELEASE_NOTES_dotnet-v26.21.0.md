# Skyline-PRISM (C#) dotnet-v26.21.0 Release Notes

Twenty-eight curated protein panels now ship with PRISM, separated from your own so a shipped panel means
the same thing on every machine. The tool can finally point at a search FASTA - which turns on
enzyme-aware parsimony and makes iBAQ real - and keeps a copy of it beside the results, so a run stays
repeatable after the database moves.

## New Features

- **28 protein panels now ship with PRISM**, up from three, and the Protein lists editor separates them
  from your own. A **Predefined** tab holds the shipped panels, read-only, with **Duplicate to my lists**
  to customize one; **My lists** holds yours. A predefined panel therefore means the same thing on every
  machine - which is what makes one citable - and ticking one stores only whether it is shown, not a copy
  of its members, so it still picks up later corrections.

  The set covers plasma and blood, vascular and epithelial identity, and QC readouts. Three additions are
  worth calling out:

  - **Histones (proteomic ruler)** - summed histone signal tracks DNA and therefore cell number
    (Wisniewski et al., Mol Cell Proteomics 2014, doi:10.1074/mcp.M113.037309), which makes it the right
    denominator when what varies is how much tissue was captured rather than how much protein was loaded.
    Comprehensive by design, and carries both the current HGNC and legacy HIST1H* nomenclatures.
  - **Hemolysis**, **Fibrinogen**, **Keratin contamination** - readouts that say a sample is compromised.
    Fibrinogen was absent from every panel this set was built from, and is the plasma-versus-serum
    discriminator.
  - **Common contaminants (cRAP)** - porcine trypsin, BSA, caseins, the yeast enolase spike-in. Listed by
    ACCESSION, because "ALB" for bovine serum albumin would match human albumin and "ENO1" would match the
    spike-in rather than the housekeeper.

  A list of yours with the same name as a shipped panel still wins that name; the shipped one is now
  reachable as `<name> (PRISM)` rather than being dropped. Panels carry mouse symbols where the ortholog
  differs (Hbb-bs, Ighg2a, Lyz1); matching is case-insensitive, so conserved symbols need no help.

- **The Protein lists editor is reachable from the Settings tab**, beside the marker-normalization
  picker, as well as from the Dynamic Range tab. A marker panel usually has to be curated before it can
  be selected, and having to go to a plotting tab to do it was backwards. Same editor, same one set of
  lists, and a list created there is immediately selectable in the picker without restarting.

- **A FASTA picker on the Settings tab**, and the protein rollup picker now offers `topn`, `maxlfq` and
  `ibaq` alongside `median_polish` and `sum`.

  Nothing in the tool could set a FASTA before, so every GUI run recorded `parsimony.fasta_path` and
  `protein_rollup.ibaq.fasta_path` as null - with two silent consequences. Protein groups always came
  from the Skyline Protein Accession column rather than enzyme-aware FASTA parsimony, which is what
  keeps a peptide from being attributed to a homolog that merely shares the subsequence. And the Dynamic
  Range tab's iBAQ view could never be a real iBAQ: it reads those keys back from parameters.json, found
  nothing, and fell back to the observed peptide count.

  The box says what the current setting means, and calls out in red the case that matters - iBAQ chosen
  with no database - rather than leaving it to a line in the log, which is not where someone picking a
  method is looking.

- **A run now keeps a copy of the FASTA it used**, in `fasta/` beside its outputs, with the original path
  and the copy both recorded in `parameters.json`. An absolute path describes a run; it does not let
  anyone repeat it once the database has been reorganized or the output directory handed to someone
  else - `--from-provenance` would quietly fall back to the Skyline accession column for parsimony, or
  to observed counts for iBAQ, and the same numbers would never come back.

  `--from-provenance` prefers the ORIGINAL whenever it still exists, so a re-run is not invalidated by
  the copy: the stage cache stamps the path, size and write time, and preferring the copy would rebuild
  every downstream stage for nothing. It falls back to the copy only when the original has gone, and
  says which key it redirected rather than substituting a database in silence.

## Bug Fixes

- **A protein group naming several members matched none of them.** An indistinguishable group carries its
  members slash-joined (`H2AC11 / H2AC18 / H2AJ`), and the matcher split only on `|`, so a panel naming
  any one member missed the group entirely. Found on real data: a 158-member histone panel matched four
  proteins in a cohort that plainly had more; it now matches 31. Every panel with proteins that land in
  indistinguishable groups was affected.
- **A panel written against human data now matches mouse.** `H4_HUMAN` and `H4_MOUSE` both reduce to
  `H4`, and the gene column is tokenized like every other identifier column rather than compared whole.
- **The marker-normalization list picker was empty on a fresh install.** It was filled only when a
  provenance file was opened, so on a machine with no saved protein lists it showed nothing and the
  shipped panels looked as though they had not been installed. It is now filled when the window opens,
  and refreshed whenever the lists are edited.
