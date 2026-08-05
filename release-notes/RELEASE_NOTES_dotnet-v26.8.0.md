# Skyline-PRISM (C#) dotnet-v26.8.0 Release Notes

Feature release for the QC tab and the config the tool hands back to you: group-by filtering now
takes several values at once, the control-correlation plot defaults to the controls it is named for,
and "Show command line" produces a config listing only the settings actually in play. Also fixes a
QC verdict that could report overfitting when the reference had in fact degraded.

## New Features

- **The QC "Group by" value dropdown is now multi-select.** Each value has a tick box, so groups can be
  combined instead of choosing one at a time - the case that motivated it is comparing Quality Control
  against Standard in the control-correlation heatmap, where the useful reading is that each control type
  correlates with itself but not with the other, and the unknowns bury it. Ticking nothing (or everything)
  still shows every sample.
- **Control correlation defaults to the controls.** Selecting that plot with no filter set now ticks the
  control sample types automatically, matching what the HTML QC report has always done - it computes that
  heatmap over reference + QC only. Previously the tool included every sample, which is not a control
  correlation. An explicit selection is never overridden, and columns with no control values (a Condition
  annotation, say) are left alone rather than filtered to nothing.
- **The config shown by "Show command line" lists only the settings in play.** Both the displayed text
  and the "Copy config (YAML)" button used to dump every property of every section - about 95 keys for
  a run that uses 31 of them - so parameters belonging to other methods read as though they were in
  effect: `topn_*` under `library_assist`, the `ibaq` block under `median_polish`, `rt_lowess` tuning
  under `method: median`. The config now carries the method choices, the tuning keys for the selected
  method, and anything that differs from the built-in default; everything omitted falls back to that
  same default, so the copied config still reproduces the run exactly. It also no longer emits the
  empty `library_assist:` key, which the Python engine could not read.

## Bug Fixes

- **A degraded reference no longer reports "possible overfitting to the reference".** The relative
  variance reduction (RVR = QC improvement / reference improvement) was forced to `+infinity` whenever the
  reference did not improve, including when it got *worse*. That tripped the "RVR > 2" branch and warned
  about overfitting **to** the reference - the opposite of what had happened - and failed the whole
  validation verdict on a degenerate number. RVR is now undefined in that case: the ratio checks are
  skipped, the report says so plainly ("Reference CV did not improve, so the QC-vs-reference ratio could
  not be evaluated - the overfitting check was skipped, not passed"), and the separate, correct
  "Reference CV increased after normalization" warning still stands. Genuine overfitting and poor
  generalization are still flagged as before.
- **The QC report explained why its CV table and its intensity-distribution reduction disagree.** The
  intensity heading reports the spread of per-sample medians across **all** samples, while each CV row is
  computed **within** one sample type. A cohort-wide alignment mostly moves the experimental samples onto
  the controls, so a 98% between-sample reduction can sit next to a ~1% CV change without either being
  wrong. Both are now labelled with the samples they cover, and the CV section explains the distinction.

## Performance

## Breaking Changes
