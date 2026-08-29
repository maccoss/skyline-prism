# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

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

- **The Protein lists editor opens from the Settings tab**, beside the marker-normalization picker, as
  well as from Dynamic Range. A panel usually has to be curated before it can be selected.

- **The Protein lists editor is reachable from the Settings tab**, beside the marker-normalization
  picker, as well as from the Dynamic Range tab. A marker panel usually has to be curated before it can
  be selected, and having to go to a plotting tab to do it was backwards. Same editor, same one set of
  lists, and a list created there is immediately selectable in the picker without restarting.

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

- **The marker-normalization list picker was empty on a fresh install.** It was filled only when a
  provenance file was opened, so on a machine with no saved protein lists it showed nothing and the
  panels PRISM ships (`EV markers`, `Glomerulus`, `Tubular contamination`) looked as though they had not
  been installed - the feature appeared broken while being perfectly configured. It is now filled when
  the window opens, and refreshed whenever the lists are edited.

## Performance

## Breaking Changes
