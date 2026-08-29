# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

- **A FASTA picker on the Settings tab**, and the protein rollup picker now offers `topn`, `maxlfq` and
  `ibaq` alongside `median_polish` and `sum`.

  Nothing in the tool could set a FASTA before, so every GUI run recorded `parsimony.fasta_path` and
  `protein_rollup.ibaq.fasta_path` as null - with two silent consequences. Protein groups always came
  from the Skyline Protein Accession column rather than enzyme-aware FASTA parsimony, which is what
  keeps a peptide from being attributed to a homolog that merely shares the subsequence. And the Dynamic
  Range tab's iBAQ view could never be a real iBAQ: it reads those keys back from parameters.json, found
  nothing, and fell back to the observed peptide count.

  The box says what the current setting means, and calls out in red the case that matters - iBAQ chosen
  with no database - rather than leaving it to a line in the log, which is not where someone picking a
  method is looking.

## Bug Fixes

## Performance

## Breaking Changes
