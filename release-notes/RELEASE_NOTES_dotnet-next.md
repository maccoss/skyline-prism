# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

- **Twelve pathway panels ship**, bringing the predefined set to 40: oxidative phosphorylation,
  glycolysis, TCA cycle, proteasome, lysosome, spliceosome/hnRNP, extracellular matrix, actin
  cytoskeleton, antigen presentation, acute phase response, chaperones/proteostasis, and
  redox/antioxidant.

  Hand-written and sized for a plot - 15 to 35 members - rather than imported from an ontology. That
  keeps them clear of anyone's redistribution terms (KEGG cannot be shipped at all) and stops one panel
  coloring a tenth of the points, which a several-hundred-member Reactome set would. For a pathway that
  is not here, import a gene list into a list of your own.

- **A display-only panel is now refused by `marker_normalization`, with an error that says why.**
  Readouts (`Hemolysis`, `Tubular contamination`, `Common contaminants`, ...) and every pathway are
  flagged: their abundance IS the signal, so dividing by it removes the thing being looked for - a
  readout hides the problem, a pathway hides the finding. This was documentation before, which made it a
  silent failure: the run succeeded, the numbers looked plausible, and what had been regressed out was
  the result.

- **`Mitochondrial mass` is renamed `Mitochondrial content`.** "Mass" is the standard term in
  mitochondrial biology, but in a mass-spectrometry tool it reads as m/z. The panel's own description
  already said content.

## Bug Fixes

## Performance

## Breaking Changes
