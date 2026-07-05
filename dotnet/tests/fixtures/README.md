# Cross-language golden fixtures

These are committed reference outputs produced by the **Python** PRISM pipeline. The C#
port's parity tests read them and assert the C# pipeline reproduces the same values
(exact tolerance for the deterministic core; functional tolerance for optimizer-driven
methods). They are copied next to the test assembly (see `SkylinePrism.Tests.csproj`)
and resolved via `AppContext.BaseDirectory/fixtures`.

## `mini/merge/`
Tiny byte-faithful slices of two real plate exports (`mini_plate1.csv`, `mini_plate2.csv`
= header + first 2000 data rows of the corresponding
`example-files/subset/2025-IRType-Plasma-PRISM-Plate{1,2}_subset.csv`) plus the golden
`merged_data.parquet` from `data_io.merge_and_sort_streaming`. Drives `MergeParityTests`.

## `mini/e2e-sum/`
`config.yaml` is a fully deterministic pipeline config (sum transition rollup, median
global normalization, standard ComBat, median-polish protein rollup, median protein
normalization; no optimizer-driven stages). `output/` holds every golden the Layer 3-9
parity gates compare against.

The dataset: 6 peptides across 3 proteins, 2 batches, 13 reference (`-Pool_`) + 14 QC
(`-Carl_`) + experimental samples.

### Parity rules baked into this fixture (must be reproduced in C#)
- **Precursor-only peptides are dropped.** With `use_ms1=false`, `DSTVAVVVYDITNVNSFQQTTK`
  (all `precursor` fragment ions) has 0 MS2 transitions after exclusion and is absent
  from `peptides_rollup.parquet` -> 5 rolled-up peptides, not 6.
- **Parsimony vs rollup peptide counts differ.** Parsimony reads the raw merged data, so
  RAB6A reports `n_unique_peptides=4`, but the protein rollup only sees the 3 surviving
  rolled-up peptides -> `n_peptides=3`.
- **Sample columns are keyed by `Sample ID`** (`<Replicate Name>__@__<batch>`), and the
  wide output columns are ordered by sorted Sample ID. Parity tests compare by column
  name, so order is not itself asserted.
- Intermediate parquet (`peptides_rollup`, `peptides_log2_internal`, `proteins_raw`) are
  LOG2; final `corrected_peptides`/`corrected_proteins` are LINEAR.

## Regenerating

Requires a Python env with the PRISM numerical deps (numpy pandas scipy scikit-learn
statsmodels pyarrow pyyaml duckdb matplotlib seaborn) and the repo on `sys.path`.
DuckDB engine version should match the C# `DuckDB.NET.Data.Full` package (currently
Python duckdb 1.5.x <-> DuckDB.NET 1.5.3).

```
# merge fixture: header + first 2000 rows of each plate, then merge
python -m skyline_prism run \
  -i dotnet/tests/fixtures/mini/merge/mini_plate1.csv \
     dotnet/tests/fixtures/mini/merge/mini_plate2.csv \
  -o dotnet/tests/fixtures/mini/e2e-sum/output \
  -c dotnet/tests/fixtures/mini/e2e-sum/config.yaml
# then delete non-golden artifacts: *.log, .duckdb_temp/, *.unsorted.parquet,
# merged_data.fingerprints.json (absolute-path cache, non-portable)
```
