# Skyline-PRISM (C#) dotnet-v26.27.0 Release Notes

## New Features

- **`prism differential --report`: the quant report, headless.** The same report the Differential pane's
  **Quant report...** button writes (shipped in dotnet-v26.26.0), from the command's own flags, so it
  needs neither Skyline nor Windows:
  `prism differential -d output/ -g condition -a Control -b Disease --report --markers "EV markers (core)"`.
  `--markers` takes panels from the same set the Markers pane offers (saved lists plus the shipped
  panels), `--markers-group-by` picks their grouping column, and `--no-enrichment` skips g:Profiler on a
  machine with no internet access. Every quant report - clicked or typed - now shows the complete
  command that regenerates it, and records it as `command:` in `quant_parameters.yaml`, built from
  what the analysis actually ran. The button and the command run one implementation, and the test
  suite holds them to it: a report built the way the button builds it and the report its recorded
  command writes through the CLI are byte-identical, file for file (the HTML apart from its
  timestamp), for adjusted, paired, stale-covariate and two-clinical-CSV reports with marker panels.
  That comparison runs with enrichment off, since g:Profiler's answer depends on its current database;
  and a rerun elsewhere needs the clinical CSV and any self-saved marker list on that machine
  (`PRISM_PROTEIN_LISTS` names a lists file to use). Where a value cannot be written on a command
  line, such as a level containing a comma, the report says so instead of recording a command that
  would select something else. A mistyped panel or column is refused
  before anything is written, and `--markers`, `--markers-group-by` or `--no-enrichment` without
  `--report` is refused rather than ignored.
- **`prism differential --clinical <csv>`: contrasts on clinical columns, headless.** The pane could
  already join an external clinical table (diagnosis, age, sex) and group or adjust by its columns;
  the CLI could not, so a disease-vs-control contrast defined in such a table had no command-line
  equivalent. `--clinical` runs the same join - key column detected by value - before any column
  name is checked, so `--group-by`, `--adjust-for`, `--subject` and the rest can name a clinical
  column. A table matching fewer than half the samples is refused. A quant report now records which
  clinical file it used (`clinical_csv` in `quant_parameters.yaml`), from the pane or the CLI.

## Bug Fixes

- **Detection no longer fails on a report row with no replicate.** A Skyline report can carry a
  precursor row with no replicate name; the detection matrix read its null sample id and failed the
  whole load, so the Detection view and the quant report's detection section were lost on such a
  document. Those rows are now dropped - they detect nothing in any sample. Found when the same row
  turned up in the Differential Explorer webapp's reader (prism-diff-explorer#1).
- **A paired detection that fell back to unpaired now gives the right reason.** When subjects matched
  across the arms but no matched pair had both samples in merged_data, the note said no subject could
  be matched, which pointed at the wrong column. It now says the pairs were matched and lost to
  merged_data.
- **`--min-per-group` refuses a non-integer.** `2.7` used to run as `2` with nothing said.
