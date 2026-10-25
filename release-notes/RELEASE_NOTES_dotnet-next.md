# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

## Bug Fixes

- **Open QC Report** now works for any output directory that holds a QC report, not only after a run
  in the same window. Pointing the Output directory at an earlier run's folder already loaded every
  plot, but the button stayed disabled, so the report had to be found and opened by hand. It now
  follows the Output directory box as **Publish to Panorama...** does. It stays disabled while a run
  is in progress, and after a run that was stopped or failed - that directory's report is then the
  previous run's - until a run there finishes. Publish to Panorama now follows the same rule.
- Changing the Output directory now reloads QC Plots when they are on screen. Before, they kept
  showing the previous directory's results under the new path until you switched to another pane and
  back. A directory that does not exist now clears the QC plots instead of leaving the old ones up.
- The Output directory can no longer be changed while a run is in progress. The run writes into it and
  every pane reads from it; the box is now read-only (the path can still be copied) and **Browse...**
  is disabled until the run ends.
- **Stop** during a document export no longer reports "PRISM run failed" with a stack trace. The
  cancellation reached the window wrapped in the per-input error that names the document, and was
  not recognized as a stop; the Log now says the run was stopped. A run that genuinely fails still
  shows the dialog, which now says that Ctrl+C copies its text.
- Rewrote the Skyline Tool Store description. It had led with reference-anchored ComBat, which is opt-in
  and not the default, and it listed none of the tool's panes beyond QC plots. It now describes standard
  ComBat, the Differential, Markers, Dynamic Range, Ion accounting and Spectrum density panes, `.sky.zip`
  input, and the `prism` CLI. The name is now just Skyline-PRISM, here and in `prism --help`, without the
  old acronym expansion.

## Performance

## Breaking Changes
