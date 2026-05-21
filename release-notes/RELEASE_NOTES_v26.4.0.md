# Skyline-PRISM v26.4.0 Release Notes

Feature release: adds reference-anchored ComBat batch correction, which performs per-analyte single-point external reference calibration (Pino et al. 2020) stabilized with ComBat's empirical-Bayes shrinkage. Opt-in via `batch_correction.reference_anchored: true`; the default behavior (standard grand-mean ComBat) is unchanged.

## New Features

### Reference-anchored ComBat (single-point reference calibration with empirical-Bayes shrinkage)

Added `combat_reference_anchored()` and wired it into the pipeline as an opt-in batch correction (`batch_correction.reference_anchored: true`, default `false`). This implements the long-intended reference-anchored design: per-analyte single-point external reference calibration (Pino et al., 2020) stabilized with ComBat's empirical-Bayes shrinkage (Johnson et al., 2007).

**Why this was needed.** The shipped pipeline only ever applied *standard grand-mean ComBat*, which aligns every batch to the across-batch grand mean and assumes each batch has the same biological composition. The previously existing `combat_with_reference_samples()` is misleadingly named — it runs standard ComBat and only *evaluates* with reference/QC CVs afterward; it never anchored the estimation on reference samples. This was discovered while drafting grant methods text.

**What it does.** When reference samples (identical material run in every batch) are present, PRISM now estimates each batch's technical effect from the **reference samples only**:
- Additive offset `gamma[i,g]` = mean of reference samples in batch i minus the pooled reference level `alpha[g]`. Because references are identical material, this offset is purely technical, so no biology is removed.
- Multiplicative scale `delta[i,g]` = reference-replicate dispersion, estimated for any batch with ≥2 reference replicates (location-only otherwise).
- Both terms are empirical-Bayes shrunk across features within a batch, so an analyte poorly measured in the reference borrows strength from the batch-wide consensus rather than being dictated by one noisy reference measurement. The additive shrinkage weight scales with the number of reference replicates: heavy shrinkage with one reference, converging to raw single-point calibration as replicates grow.

**Properties.**
- Output is calibrated **absolute** log2 abundance on the input scale (back-transformed to linear at output), **not** a ratio to the reference.
- Within-protein sample-to-sample comparisons and CVs behave as expected; cross-protein absolute comparisons benefit from the reference anchoring.
- Batches with no reference samples fall back to standard grand-mean ComBat with a warning (`no_reference_batch="fallback"`; also supports `"error"` and `"skip"`).
- The scale term is learned only from identical reference material, which makes scale correction safer than standard ComBat scale (the latter mixes biological variance into `delta`).

**Config.** Both `prism config-template` and `--minimal` now document `batch_correction.reference_anchored` (default `false`) and `reference_type` (default `"reference"`). `metadata.json` provenance records both. The default behavior is unchanged (standard grand-mean ComBat) so existing configs reproduce.

Eight regression tests in `tests/test_batch_correction.py::TestReferenceAnchoredComBat` cover offset removal + reference alignment, biological-signal preservation, absolute-scale (not ratio) output, EB shrinkage of noisy features, EB dispersion reduction, scale harmonization with ≥2 references, and the no-reference-batch fallback/error paths.

- **Files modified**: `skyline_prism/batch_correction.py`, `skyline_prism/__init__.py`, `skyline_prism/cli.py`, `tests/test_batch_correction.py`, `docs/methods.md`

## Bug Fixes

<!-- none yet -->

## Performance

<!-- none yet -->

## Breaking Changes

<!-- none yet -->
