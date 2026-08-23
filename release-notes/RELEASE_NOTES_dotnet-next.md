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

- **`mean_rt` values shift in the last bits.** Making it order-invariant changes how the sum is
  evaluated, so the column moves by ~1e-14 (e.g. `10.23372064276885` -> `10.233720642768837`). It is
  a diagnostic column, no abundance is affected, and the new value is the reproducible one - but it
  is a value change, so it is called out here rather than buried.
