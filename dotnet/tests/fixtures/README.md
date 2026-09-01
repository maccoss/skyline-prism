# Cross-language golden fixtures

These are committed reference outputs produced by the original **Python** PRISM pipeline. The parity
tests read them and assert PRISM still reproduces the same values (exact tolerance for the
deterministic core; functional tolerance for optimizer-driven methods). They are copied next to the
test assembly (see `SkylinePrism.Tests.csproj`) and resolved via `AppContext.BaseDirectory/fixtures`.

> [!IMPORTANT]
> **These goldens are frozen.** The Python engine was retired and removed after `v26.4.4`, so they can
> no longer be regenerated from this repository (the recipe at the bottom needs a Python environment
> with that package). That costs nothing they were doing: they are a fixed independent-implementation
> baseline, and the `quantities.sha256` digests — which compare PRISM against **its own** committed
> reference — are what catches a change in today's numbers. A parity failure still means what it always
> meant: a reported quantity moved away from the reference implementation. Investigate it; do not
> delete the fixture. To regenerate them anyway, check out the `v26.4.4` tag, which still has the
> Python package.

## `mini/merge/`
Tiny byte-faithful slices of two real plate exports (`mini_plate1.csv`, `mini_plate2.csv`
= header + first 2000 data rows of the corresponding
`example-files/subset/2025-IRType-Plasma-PRISM-Plate{1,2}_subset.csv`) plus the golden
`merged_data.parquet` from `data_io.merge_and_sort_streaming`. Drives `MergeParityTests`.

`mini_plate1.parquet` / `mini_plate2.parquet` are **the same 2000 rows in Skyline's parquet
export convention** (~24 KB each - a fifth the size of the merged golden, and 1/20th of the
CSVs). They drive `ExportFormatParityTests`.

### Skyline's two export conventions (why the parquet fixtures exist)

Skyline's CSV and parquet exports of the same report are **not** the same table, and the tool
prefers parquet in production - so before these fixtures, the branch of `DuckDbMerge` that
production actually takes (`read_parquet`, and `DESCRIBE` for the header) had no coverage at all.

| | CSV export | parquet export |
|---|---|---|
| names | spaced English (`Protein Accession`) | PascalCase, no spaces (`ProteinAccession`) |
| TIC column | `Total Ion Current Area` | `TicArea` |
| charges | DuckDB infers (Int64 here) | Int32 |
| `Area` | DuckDB infers - **Int64** for these slices, whose areas are all whole numbers | **Double** |

`SkylineColumns.FindColumn` normalizes case, spaces and underscores, which covers every one of
these except `TicArea` - and nothing in Core reads TIC or `Coeluting`. The type differences are
real though: the rollup sees integer columns on the CSV path and floating-point ones on the
parquet path, which is exactly what the parity test pins down.

One difference is expected and asserted rather than fixed: the **key column's name is inherited
from the input schema**, so `peptides_rollup.parquet` is keyed by `Peptide Modified Sequence
Unimod Ids` from a CSV cohort and `PeptideModifiedSequenceUnimodIds` from a parquet one. Every
value is bit-identical; only that one column name differs.

Regenerate with `Convert.cs`'s mapping (DuckDB, casting each CSV column to the parquet export's
type and renaming it):

```sql
COPY (SELECT CAST("Protein Accession" AS VARCHAR) AS "ProteinAccession",
             CAST("Precursor Charge"  AS INTEGER) AS "PrecursorCharge",
             CAST("Area"              AS DOUBLE ) AS "Area",
             CAST("Total Ion Current Area" AS DOUBLE) AS "TicArea", ... -- all 24 columns
      FROM read_csv('mini_plate1.csv', header=true, delim=',', all_varchar=true))
TO 'mini_plate1.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
```

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

## `cohort/` - the real-cohort regression gate

1,129,728 rows of a real two-plate cohort, entered at the **merge boundary**: the input is the merged
parquet table, not Skyline exports. That is what lets one fixture drive every downstream method, and it
keeps the fixture to 14 MB instead of the 2.7 GB of raw exports it came from.

| | |
|---|---|
| peptides / proteins | 327 peptides, 52 selected proteins (123 distinct protein assignments once shared peptides are followed) |
| samples / batches | 192 samples, 2 batches, 16 reference / 16 QC / 160 experimental |
| shared peptides | 47 peptides belong to two selected proteins, so parsimony has real work |
| protein sizes | proteins with 1 to 30+ peptides, so protein rollup is not uniform |

Drives `CohortRegressionTests`, one case per directory (`config.yaml` + `quantities.sha256`), digested
exactly like `mini/*/quantities.sha256`.

### Why it starts at the merge

Stage 1 is deterministic and covered elsewhere (`MergeParityTests`, `ExportFormatParityTests`), it is the
slowest stage, and it needs inputs that cannot be committed. Everything downstream needs is carried in
the merged table: `ShapeCorrelation` for top-N by correlation, `ProductMz` for library-assisted rollup,
`Batch` for ComBat, `AcquiredTime` for batch estimation.

The test seeds `merged_data/` plus a matching `merged_data.cache.json` into the output directory, so
Stage 1 is skipped. This works because `SourceFingerprint.Compute` hashes each input's path, length and
last-write time - **not its contents** - so empty placeholder files reproduce the fingerprint. The test
asserts the run really did see 192 samples and both batches, because a Stage 1 that re-ran over empty
placeholders would otherwise digest nothing and look like a pass.

### The fixture is anonymized

Replicates are `S001..S189`, batches `Batch1`/`Batch2`, filenames derived from the replicate, and
acquisition timestamps shifted to a fixed 2020 epoch - **gaps preserved**, because batch estimation reads
them, but absolute times are not this repository's to publish. Peptide sequences and UniProt accessions
are kept as they are. Regenerating from a new cohort means redoing that mapping; do not commit raw
sample identifiers.

### Two structural properties are deliberate

- **189 replicate names produce 192 sample IDs**, because three QC injections carry the same name in both
  batches. That is the collision CLAUDE.md warns about, and keeping it means the fixture exercises the
  document-qualified metadata lookup.
- **The two metadata files use different header names** - `ReplicateName` from the RPC export path,
  `Replicate` from the headless one - which is what a mixed cohort really presents.

### `cohort.blib` - the spectral library, subset the same way

2.5 MB, subset from the cohort's own 572 MB BiblioSpec library to the 385 reference spectra that match
the 327 fixture peptides, with `RefSpectraPeaks`, `Modifications` and `RetentionTimes` cut to those
spectra and the file `VACUUM`ed. It drives the `library-assist` case.

Anonymized like the parquet: `SpectrumSourceFiles.fileName` carried the same `.raw` replicate
identifiers and is rewritten to `S###.raw`, and `LibInfo.libLSID` named the study so it is replaced with
a neutral one. `LibInfo.numSpecs` is corrected to the subset count, so the library is internally
consistent rather than claiming 88,242 spectra it no longer has.

`library_path` is written RELATIVE in that case's config, because the fixture's absolute location
depends on where the test assembly was built; the test makes it absolute.

### Verifying a case actually did something

Two checks worth repeating whenever a case is added, because both failure modes leave a digest that
still looks valid:

- **Compare against the case that differs only in the axis under test.** `library-assist` versus
  `norm-median` differ solely in the transition rollup, and 1344 of 1379 columns differ - so the method
  really ran. Comparing against the production default instead would have proved nothing, since that
  case also changes the normalizer.
- **Read the `#` header line.** Every digest records `peptides=`/`proteins=`/`samples=`/`batches=`, so a
  method that silently dropped most peptides shows up in the diff. All eleven cases report 327 peptides
  and 51 proteins.

## `mini/*/quantities.sha256` - the bit-exact regression gate

One line per output column - `file<TAB>column<TAB>sha256` - hashing the **exact IEEE-754 bits** of
every value in that column, rows ordered by the key column. Drives `QuantityRegressionTests`.

These exist because the parity tests above cannot catch what they are for. Those compare C#
against the **Python** goldens with a tolerance (1e-9 for the deterministic core, 3e-2 relative
for ComBat), which is the right bar for two independent implementations - but it means a C#
change that moves a quantity by less than the tolerance passes them in silence. The digests
compare C# against **its own** committed reference, so any change to any value fails and names
the column that moved.

Bits, not formatted text, so signed zero, NaN payloads and last-ulp drift are all caught. Rows
are sorted by key because output row order is explicitly not a parity contract (see CLAUDE.md,
"C# Stage 1 partitions, and does NOT sort").

> [!CAUTION]
> **A failure here is the test working.** Do not regenerate to make it pass. Establish which
> quantity moved and why. If the change is intended and correct, regenerate deliberately and say
> so in the release notes - it means users' numbers change too.

### Inputs are parquet

Skyline's CSV PRISM report was large and slow to export, so the report moved to **parquet** and the
tool now exports that by default - new cohorts arrive as parquet, so that is the path this gate
watches. The digests are therefore keyed on the parquet schema's column names
(`PeptideModifiedSequenceUnimodIds`). CSV is still covered: `ExportFormatParityTests` proves the two
exports give bit-identical quantities, and the cross-language parity tests still drive CSV against
the Python goldens.

### Windows only, and why

Measured on this repo's CI: a commit that is bit-exact on Windows differs on **ubuntu and macOS** -
in 5-6 columns for the ComBat-disabled fixtures, and in **389** columns for `e2e-sum`, the one
fixture with ComBat enabled, whose empirical-Bayes estimation is dense in `exp`/`log`.

That is not a defect. IEEE-754 pins `+ - * / sqrt`, so those agree everywhere, but `Math.Log`,
`Math.Exp` and `Math.Pow` delegate to the platform's libm and are not required to return identical
bits. So the gate is pinned to Windows - the platform the Skyline external tool runs on, and the one
PRISM's numbers are most often produced on. Linux and macOS keep the 1e-9 cross-language parity
tests, which catch any change big enough to alter a reported result; what the digest adds on top is
sub-tolerance drift, which is inherently platform-specific and will show up on Windows anyway.

### Regenerating

```bash
PRISM_UPDATE_DIGESTS=1 dotnet test dotnet/tests/SkylinePrism.Tests/SkylinePrism.Tests.csproj   --filter "FullyQualifiedName~QuantityRegressionTests"
dotnet build dotnet/tests/SkylinePrism.Tests/SkylinePrism.Tests.csproj   # <- required
```

The rebuild is not optional. Regeneration writes to the **source** tree, but the test reads the copy
made next to the test assembly at build time - skip it and the very next run compares fresh output
against the stale digests and fails everywhere.

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
