# Skyline-PRISM v26.4.4 Release Notes

Bug-fix release: a config written for the C# engine or the Skyline external tool now runs unchanged
on the Python engine. The flat `transition_rollup.library_*` keys are recognized, and an empty
`library_assist:` block no longer aborts the run.

## Bug Fixes

- **The flat `library_*` keys are now accepted.** The C# engine and the Skyline external tool spell
  the library-assisted rollup settings as `transition_rollup.library_path`, `library_min_fragments`,
  `library_mz_tolerance`, `library_outlier_threshold`, `library_remove_outliers`, and
  `library_fitting_method`. Python previously knew only the nested `library_assist:` block and the
  legacy `spectral_library_*` aliases, so a config authored on one engine warned that every one of
  those keys was unrecognized and then failed with "library-assisted method requires library_path in
  config". One config file now runs on both engines. When a setting appears more than once, the
  precedence is nested `library_assist:` > flat `library_*` > legacy `spectral_library_*`, matching
  how the C# port folds the nested block onto its flat fields.
- **An empty `library_assist:` block no longer crashes the run.** A bare `library_assist:` line
  (valid YAML, parsed as `None`) raised `AttributeError: 'NoneType' object has no attribute 'get'`
  in `prism run`, and the same way in `prism compare` when reading a run's `metadata.json`. It is now
  treated as absent.
- **Library-assist settings explicitly set to `0` are honored.** The old resolution chain used
  `config.get(key) or default`, which treats `0` and `0.0` as unset - so `library_outlier_threshold: 0`
  silently became `1.0`. Values are now tested for presence, not truthiness.
