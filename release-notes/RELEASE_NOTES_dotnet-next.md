# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

- **The peptide->protein median polish now writes its residuals.** `proteins_raw_residuals.parquet`
  gives each peptide's deviation from its protein group's fitted profile - one row per (protein group
  x peptide), one column per sample, LOG2 - which is the evidence for proteoform variation, PTMs and
  protein processing in the sense of Plubell et al. 2022. The Python engine has written this file
  since before the port; the C# engine silently did not, so `output.include_residuals: true` produced
  only half of what Python produced and an analysis built on it could not be reproduced on the C#
  engine. Verified bit-for-bit against a real Python run (SEA-AD MTG pilot, 82 samples): **7,803,038
  residual values, zero differences.**

- **CI now fails if any reported quantity changes, to the last bit.** Every peptide and protein
  value the pipeline writes is fingerprinted per column from its exact IEEE-754 bits and compared
  against a committed reference across the six end-to-end fixtures, so a refactor that shifts a
  number by one ulp is caught and the changed column named. The existing cross-language tests
  compare against the Python goldens with a tolerance (necessary for two independent
  implementations, but it let sub-tolerance drift through unnoticed); this closes that gap. The
  gate runs on Windows, because `Math.Log`/`Exp`/`Pow` are not bit-identical across platform C
  runtimes - measured here, the same commit differs on Linux and macOS in 5-6 columns without
  ComBat and 389 with it. Linux and macOS keep the 1e-9 parity coverage, which catches anything
  large enough to change a reported result.

- **Skyline's parquet export is now the tested path.** Skyline's CSV PRISM report was large and
  slow, so the report moved to parquet and that is what the tool exports by default - but every
  fixture still fed CSV, leaving the parquet branch of the merge with no coverage at all. Two
  ~24 KB parquet fixtures in Skyline's real export convention now drive the bit-parity gate, and a
  new test asserts that a cohort exported as parquet yields **bit-identical** quantities to the
  same cohort exported as CSV. It does - worth confirming, since the two exports differ in column
  naming and in physical types (parquet writes `Area` as double and the charges as int32, where CSV
  type inference can yield integers).

## Bug Fixes

## Performance

## Breaking Changes

- **`peptide_residuals.parquet` is now `peptides_rollup_residuals.parquet`.** The old name described
  the stage's output while its rows are transitions, and it collided with the peptide-row residuals the
  protein rollup now writes. Residual files are now named for the value file they explain and sit
  beside - `peptides_rollup.parquet` -> `peptides_rollup_residuals.parquet`, `proteins_raw.parquet` ->
  `proteins_raw_residuals.parquet` - which also matches the Python engine's names. **Update scripts that
  read the old filename**; nothing in the file's contents or schema changed.
