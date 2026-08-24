# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

- **Residuals now come in a batch-corrected form too**, so they can be compared across batches.
  `corrected_peptides_residuals.parquet` and `corrected_proteins_residuals.parquet` sit beside the raw
  `peptides_rollup_residuals.parquet` / `proteins_raw_residuals.parquet` and carry the same rows and
  columns, rescaled by the ComBat that applied to the output each one accompanies.

  Only ComBat's **scale** is applied, and that is exact rather than an approximation: a residual is a
  deviation from a fitted profile, so ComBat's location terms cancel out of it and
  `e* = e / sqrt(delta[batch, feature])` is the whole transform. Normalization needs no handling at all -
  peptide and protein normalization are absorbed into the median polish's column effect, so residuals
  are already invariant to them.

  The files are written even when ComBat is disabled or reverted, as faithful copies, so a script can
  read the corrected file unconditionally instead of branching on whether correction ran. Features
  ComBat held out, samples in no corrected batch, and (batch, feature) pairs whose scale was not
  estimable - common when a batch carries a single reference injection - are scaled by exactly 1.0,
  because ComBat did nothing to them either.

  Both new files are covered by the bit-parity gate.

## Bug Fixes

## Performance

## Breaking Changes

- **Protein quantities are no longer batch-corrected twice, and they change as a result.** The protein
  arm now branches from the normalized, **pre-ComBat** peptide matrix, so `corrected_proteins` is
  ComBat-corrected exactly once - at the protein level, where it is reported - instead of inheriting the
  peptide correction and then being corrected again at Stage 4c.

  This is a measured improvement, not a tidy-up. On a 4-batch AD cohort (62 experimental, 5 reference,
  5 **held-out** QC injections), the second correction was making things worse: protein-level held-out
  QC CV went **12.7% -> 13.0%** and the run warned of overfitting (the reference improved 3x more than
  the QC, the signature of fitting the anchor rather than the batch). Correcting once gives
  **16.3% -> 12.4%** with no warning.

  What moves: `corrected_proteins` by a median of **2.7%** (p99 1.28x, long tail), and `proteins_raw` by
  a median of 5.6%. Two independent cohorts agree on the size (2.7% and 2.9%). What does **not** move:
  `peptides_rollup` and `corrected_peptides` are **bit-identical** - the peptide arm is untouched.

  `peptides_log2_internal.parquet` now holds post-normalization, pre-ComBat values, which is what
  `docs/output_files.md` always described it as.

- **`batch_correction.protein_level: false` now means the protein output gets no batch correction at
  all.** Previously it still inherited the peptide-level correction through its inputs. There is
  deliberately no warning for this. Defaults are unchanged (both levels on), so a default run is
  unaffected. See CLAUDE.md "Batch correction at reporting level" for the full flag matrix.
