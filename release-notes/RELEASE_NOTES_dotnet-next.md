# Skyline-PRISM (C#) dotnet-vNEXT Release Notes

Working draft for the next C# (.NET) release. Append entries as they land on the development branch;
rename to `RELEASE_NOTES_dotnet-v{version}.md` at release time - the release workflow publishes this file
as the GitHub Release description and fails if it is missing.

## New Features

- **Twenty-nine panels ship**, bringing the predefined set to 60. Eighteen processes and pathways: oxidative phosphorylation,
  glycolysis, TCA cycle, proteasome, lysosome, spliceosome/hnRNP, extracellular matrix, actin
  cytoskeleton, antigen presentation, acute phase response, chaperones/proteostasis, and
  redox/antioxidant, DNA damage repair, autophagy, unfolded protein response, innate immune signaling,
  apoptosis and fatty acid oxidation.

  Eleven more for brain work: the four major cell types (neuronal, astrocyte, microglia,
  oligodendrocyte/myelin), White matter and Grey matter for dissection composition, and disease panels
  for **Alzheimer's, Parkinson's, ALS/FTD and Huntington's** - the PD one split further into the
  LRRK2-RAB substrates and the lysosomal arm, since those ask different questions.

  The brain panels follow the definitions in the SEA-AD pilot notebooks where the lab already has them,
  so a panel here reproduces those figures rather than approximating them: White matter and Grey matter
  are the exact WM_GENES/GM_GENES sets, and the PD panels are PD_MARKERS, PD_RAB and PD_LYSOSOMAL.

  Grey matter carries a measured caution. On the SEA-AD MTG pilot (73 donors with both GM area and
  pathology) it tracks measured GM fraction at r = +0.37, but also declines with pathology - r = -0.14
  with Braak, -0.22 with CERAD - because all four members are synaptic and synapse loss is the AD
  phenotype. Good for reporting composition; suspect as a denominator in a neurodegeneration cohort.

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

- **A failed headless export no longer passes off the previous export as its own output.** The parquet
  result was accepted on its PAR1 marker alone, which says the bytes at that path are parquet - not that
  this run wrote them. When an export failed before Skyline received its arguments, the file left by an
  earlier run was still sitting there, so the log said `Exported ... 1,377,315,301 bytes, parquet` and
  handed it to the merge. Observed on a 2-plate cohort, which silently analysed a report exported 25 days
  earlier, from a version of the `.sky` that had since been re-integrated. The transition report, the CSV
  fallback and the replicate metadata are each now cleared before re-export and must be replaced by the
  run that claims them; otherwise the export fails instead of reporting stale numbers. Reusing an
  unchanged document's previous export is unaffected - that path checks the document and tool version
  first.

- **A Skyline that never starts is reported as a startup timeout instead of `Pipe is broken.`**
  Abandoning the wait for Skyline's callback meant connecting a dummy client to PRISM's own pipe to
  release the blocked thread, and the waiter counted that as Skyline connecting - so PRISM wrote the
  command-line arguments to an already-closed client and blamed the resulting broken pipe on Skyline. It
  was nondeterministic, so two calls failing for the same reason reported it two different ways.

- **The headless export waits 3 minutes for Skyline to start, not 90 seconds, and keeps waiting while
  Skyline is actually running** (to 10 minutes). PRISM exports several documents at once, so a headless
  start competes with another Skyline already streaming a multi-GB report - and 90 s was not enough for a
  cold ClickOnce start under that load. A Skyline that never appears still fails at 3 minutes.

- **A give-up no longer kills a headless Skyline that another concurrent export is still using.** The
  "not running before we launched" test cannot tell the two apart when two exports launch at once.

## Performance

## Breaking Changes
