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
  implementations, but it let sub-tolerance drift through unnoticed); this closes that gap.

- **Skyline's parquet export is now covered by tests.** The Skyline tool prefers parquet when
  exporting reports, but every fixture fed CSV, so the parquet branch of the merge had no coverage
  at all. Two ~24 KB parquet fixtures in Skyline's real export convention now drive a test
  asserting that a cohort exported as parquet yields **bit-identical** quantities to the same
  cohort exported as CSV. It does - which is worth knowing, since the two exports differ in column
  naming and in physical types (parquet writes `Area` as double and charges as int32, where CSV
  type inference can yield integers).

## Bug Fixes

## Performance

## Breaking Changes
