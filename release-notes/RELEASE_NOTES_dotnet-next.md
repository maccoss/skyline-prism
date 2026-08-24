# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

## Bug Fixes

- **The same inputs now always produce the same outputs.** PRISM was not reproducible run to run at
  `n_workers > 1`: two runs of the same binary, on the same input, with the same config differed in
  the last bits, and on a 2-plate cohort (192 samples) only **17% of `corrected_proteins` cells came
  back bit-identical**. Two independent causes, both fixed:
  - The transition rollup wrote peptides in *completion* order, so `peptides_rollup.parquet`'s row
    order varied. Values keyed by peptide were stable, but ComBat's cross-feature reductions sum over
    rows in file order and floating-point addition is not associative, so the row order leaked into
    the corrected quantities - amplified by the second ComBat pass. Results are now emitted in the
    reader's order at any worker count.
  - `mean_rt` is a mean over each peptide's rows, and DuckDB promises no row order *within* a
    peptide, so it varied too. It is now summed over a sorted copy, which depends on the values and
    not on how they arrive.

  Verified on a real 2-plate cohort: repeated runs at 8 workers are now byte-for-byte identical, and
  8 workers gives byte-identical files to 1 worker - `n_workers` is a performance knob only, with no
  measurable wall-clock cost. **Reported abundances were never affected by the second cause** (the
  median polish is built on medians, which are order-insensitive) and the first moved values by at
  most 105 ulp (~1e-14), so no scientific conclusion changes - but reproducibility did not hold, and
  provenance depends on it. Affects all releases through dotnet-v26.14.0.

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

- **`mean_rt` values shift in the last bits.** Making it order-invariant changes how the sum is
  evaluated, so the column moves by ~1e-14 (e.g. `10.23372064276885` -> `10.233720642768837`). It is
  a diagnostic column, no abundance is affected, and the new value is the reproducible one - but it
  is a value change, so it is called out here rather than buried.
