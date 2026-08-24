# Skyline-PRISM (C#) dotnet-v26.14.1

A reproducibility fix. PRISM was not producing identical output from identical input when
`n_workers > 1` - on a 2-plate cohort only 17% of `corrected_proteins` values came back bit-identical
between two runs of the same binary. The differences were tiny (~1e-14) and no scientific conclusion
changes, but reproducibility is a precondition for the provenance PRISM records, so this is worth
taking promptly. Affects all earlier releases.

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

## Breaking Changes

- **`mean_rt` values shift in the last bits.** Making it order-invariant changes how the sum is
  evaluated, so the column moves by ~1e-14 (e.g. `10.23372064276885` -> `10.233720642768837`). It is
  a diagnostic column, no abundance is affected, and the new value is the reproducible one - but it
  is a value change, so it is called out here rather than buried.
