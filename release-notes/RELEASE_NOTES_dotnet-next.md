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

- **Two kidney panels ship alongside the EV one**, and shipped panels now reach the Dynamic Range plot
  without being copied into your own list file first. All three appear in **Protein lists...** and can be
  edited into cohort-specific variants; a list of yours with the same name replaces the shipped one. They
  arrive **unticked**, so nothing is colored until asked for.

  - **`Glomerulus`** - 18 structural markers of glomerular tissue (GBM collagen IV alpha-3/4/5 and
    laminin, basement-membrane proteoglycans, glomerular endothelium, mesangium, podocytes), for
    normalizing single-glomerulus work by how much glomerulus a dissection actually captured. Weighted
    toward structure on purpose: `NPHS1`/`NPHS2` are left out because podocyte loss *is* the phenotype in
    most glomerular disease and a score built on them would regress out the finding along with the
    capture, and `COL4A1`/`COL4A2` because they are ubiquitous basement membrane.
  - **`Tubular contamination`** - 15 proximal-tubule, thick-ascending-limb and distal/collecting markers.
    A **readout**, not a normalizer: dissected glomeruli carry tubular fragments, and these are abundant
    enough that carry-over is visible on the plot. Spread across nephron segments so the plot says which
    segment came along.

- **View the Dynamic Range plot under any protein rollup**, regardless of what the run used. A new
  **Rollup** drop-down offers **sum**, **median polish**, **top N**, **MaxLFQ** and **iBAQ**; **As run**
  (the default) keeps reading `corrected_proteins.parquet`.

  The method decides what the y axis *is*, which the plot could previously only state in its label. A
  summing method carries the peptide count into the answer - on a real cohort C4A (121 peptides) leads
  Skyline's summed view and sits below ITIH2 (44 peptides) under median polish - while iBAQ divides by
  the theoretical peptide count and is the only one of them meant for comparing one protein against
  another.

  Any choice but **As run** re-rolls `corrected_peptides.parquet` with that method, using the run's own
  `min_peptides` / `topn` so the method is the only thing that changes. It does not re-run parsimony or
  the protein-level normalization and batch correction, and the status line says
  `[recomputed from corrected_peptides]` so the two are never confused. iBAQ takes its FASTA from the
  run's `protein_rollup.ibaq.fasta_path` or `parsimony.fasta_path`; without one it divides by the
  observed peptide count and says so. The digest and the rollup run off the UI thread behind a progress
  bar, and the theoretical counts are cached.

## Bug Fixes

## Performance

## Breaking Changes
