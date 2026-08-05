# Skyline-PRISM vNEXT Release Notes

Working draft. Rename to `RELEASE_NOTES_v{version}.md` at release time.

## New Features

<!-- none yet -->

## Bug Fixes

- `transition_rollup` now accepts the flat `library_*` keys used by the C# engine and the Skyline
  external tool (`library_path`, `library_min_fragments`, `library_mz_tolerance`,
  `library_outlier_threshold`, `library_remove_outliers`, `library_fitting_method`), so one config
  file runs on both engines. Previously these warned as unrecognized and the library path was not
  found. Precedence when a setting appears more than once: nested `library_assist:` block > flat
  `library_*` > legacy `spectral_library_*`.
- An empty `library_assist:` block no longer crashes `prism run` and `prism compare` with an
  `AttributeError`; it is treated as absent.
- Library-assist settings explicitly set to `0` are no longer replaced by their defaults.

## Performance

<!-- none yet -->

## Breaking Changes

<!-- none yet -->
