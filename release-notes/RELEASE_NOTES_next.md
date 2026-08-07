# Skyline-PRISM vNEXT Release Notes

Working draft. Rename to `RELEASE_NOTES_v{version}.md` at release time.

## New Features

- **`processing.merge_memory_mb`** caps DuckDB's buffer pool during the Stage 1 merge/sort. `0` (the
  default) keeps the engine's existing 8 GB limit. Beyond the cap the sort spills to the scratch
  directory, so a smaller value is slower, never wrong.

## Bug Fixes

- **Six tests failed on Windows** (`test_chunked_processing.py`). The peptide rollup left its parquet
  reader open across the delete of the file it was reading - which POSIX allows and Windows does not -
  so the temp file could not be removed, the delete raised, and the caller's temporary directory could
  not be cleaned up either.

- **Reference-anchored ComBat now uses the same estimator as the standard path.** It previously had its
  own implementation, which substituted a placeholder scale of `1.0` where the data supported none and
  fed that into the empirical-Bayes prior, letting one such feature perturb the shrinkage of every
  other feature in its batch. It now also holds out a feature that some batch's references never
  observed, rather than treating the unknown offset as zero.

  **Dense data is unaffected**, which is the normal case. On a 400 x 40 cohort with 3 references per
  batch: no cells changed on dense input; ~25% changed where features are constant within a batch
  (worst 0.3%); with 10% missing values the median change was 0.1%.

  This method now has cross-engine fixtures (`dotnet/tests/fixtures/refanchored/`) holding the C# port
  to the Python engine's output, which it reproduces to 1e-10. It previously had none.

## Performance

<!-- none yet -->

## Breaking Changes

- **Batch estimation from acquisition-time gaps is now OFF by default** (`batch_estimation.method` is
  `none`, was `auto`). It cannot tell a real plate boundary from an ordinary pause in a continuously
  acquired run, and a wrong guess makes ComBat "correct" between batches that do not exist - silently
  changing every abundance. Set `batch_estimation.method: auto` to opt back in.
