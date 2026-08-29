# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

- **Normalize to a set of marker proteins** (`marker_normalization`). Estimates one per-sample score
  from how the markers move together - PC1 of the z-scored marker block, computed at the protein level
  - and removes from every peptide and protein the part that tracks it. It answers "what changed per
  unit of the marked material" rather than "what changed in whatever was captured", which is the
  question a capture-based experiment cannot otherwise separate, because loading normalization makes
  total signal equal by construction.

  It runs **after** the ordinary normalization, never instead of it, and one protein-level score is
  applied to **both** outputs: how much marked material a sample contributed is a property of the
  sample, not of the table being analysed.

  The markers are defined by a **protein list** - the same lists the Dynamic Range tab highlights,
  selectable in the GUI or named in the config (`protein_list`), or given as a file
  (`protein_list_file`, the reproducible form). PRISM ships an **`EV markers`** panel: 18 canonical
  extracellular-vesicle proteins, available with no saved lists at all.

  PC1 rather than a mean because markers need not share a sign - in the shipped panel four of eighteen
  load opposite to the rest, so a mean partially cancels. `method: mean` is available for comparison.

  Both corrected outputs are rewritten with the axis removed, each feature keeping its own abundance
  level; `marker_normalization.csv` records the per-sample score and the loadings. The markers stay in
  the outputs flagged `normalization_marker` - their residual is near zero by construction, so exclude
  them from results read off these files. Fewer than 3 quantified markers is an error, and a PC1 under
  40% of marker variance is warned about.

## Bug Fixes

## Performance

## Breaking Changes
