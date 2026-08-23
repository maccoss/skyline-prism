# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

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
