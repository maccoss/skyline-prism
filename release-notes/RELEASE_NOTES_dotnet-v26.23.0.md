# Skyline-PRISM (C#) dotnet-v26.23.0 Release Notes

Requires the .NET 10 runtime, as dotnet-v26.22.0 did - see that release if you have not upgraded yet.

## New Features

- **The Predefined tab groups its panels under collapsible category headings** - Normalizers, Plasma and
  blood, Endothelial, Epithelial, Readouts and contamination, Pathways and processes, and Brain and
  neurodegeneration. At 65 panels a flat list was no longer something anyone read top to bottom, and the
  headings carry the distinction that matters most about a panel: whether its abundance is the scale you
  are dividing by or the result you are looking for. **My lists** stays flat - a handful of your own needs
  no navigation, and categorizing them would mean being asked for a category every time you make one.

- **Contaminants are shown by protein name rather than bare accession.** A member may now be written
  `<accession> = <name>` - `P00761 = Trypsin (porcine)`, `P02769 = Serum albumin (bovine, BSA)` - where
  **everything left of the `=` is matched and everything right of it is only displayed**. The whole
  `Common contaminants (cRAP)` panel is labeled this way. The accessions were never optional: these are
  non-human proteins, and `ALB` for bovine serum albumin matches *human* albumin. Neither is the UniProt
  entry name a way out - `ALBU_BOVIN` reduces to the token `ALBU`, and so does human `ALBU_HUMAN`, because
  species suffixes are stripped so panels work across human and mouse. The syntax works in your own lists
  too.

- **Five more pathways**, bringing the predefined set to 65: `Cell cycle and proliferation`,
  `Epithelial-mesenchymal transition`, `Hypoxia response`, `Glucose and lipid metabolism` and
  `Insulin signaling`. All display-only.

  Two are read differently from a normal panel, and say so in the tool. **Insulin signaling is regulated
  by phosphorylation, not abundance** - its members are present whether or not the cascade is active, so a
  flat panel is the expected result and says nothing about signaling; it earns its place on a
  phospho-enriched run. For abundance work, `Glucose and lipid metabolism` measures what that pathway
  *does*. **Epithelial-mesenchymal transition is read as a balance**, not a total: its epithelial half
  (CDH1, claudins, keratins) and mesenchymal half (VIM, CDH2, FN1) move in opposite directions.

- **Twenty-nine more panels ship.** Eighteen processes and pathways: oxidative phosphorylation,
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

- **The contaminants panel no longer claims human alpha-enolase.** `Common contaminants (cRAP)` carried
  the UniProt entry names `ENO1_YEAST` and `TRYP_PIG` alongside its accessions, and panel matching strips
  species suffixes so a panel works across human and mouse - so those reduced to the tokens `ENO1` and
  `TRYP`, the exact collisions the panel is written in accessions to avoid. On any human run, abundant
  alpha-enolase was colored as a contaminant and labeled "yeast, spike-in"; and because the panel is
  declared before `Glycolysis` and the first list to claim a protein wins, ENO1 was taken from the panel
  it belongs to. The accessions `P00924` and `P00761` already cover both spike-ins. Every member of the
  panel is now checked against the UniProt accession pattern, so no entry name can return.

- **A comma inside a display label no longer splits the member in two.** The list editor and the members
  file importer both treat commas as separators, so a spreadsheet row can be pasted in - which tore
  `P02769 = Serum albumin (bovine, BSA)` into a member that still matched and a `BSA)` that never could.
  A line carrying a label is now taken whole; a line without one still splits on commas, semicolons and
  tabs.

- **Stopping a run now shuts down every headless Skyline it started.** The guard that stops one export
  from killing another's Skyline was also applied on cancellation, where it is wrong: the exports share
  the run's cancellation token, so all of them are stopping and none is left for an unrecognized Skyline
  to belong to. Cancelling a two-document export left both Skylines running, holding their documents
  open, with nothing remaining to reap them.

- **A caller's timeout now bounds the wait for Skyline to start.** The startup wait answered only to its
  own limits, so a caller that passed an explicit bound - the isolation-scheme import passes 5 minutes so
  it can never hold a run up, and `PRISM_ISOLATION_TIMEOUT_SEC` shortens it further - could still be held
  for the full base-plus-extended startup before its deadline was consulted at all.

- **The marker report no longer mis-reports which members were found.** It compared each member against
  the raw identifier columns, so any member the matcher reached by tokenization - a member of an
  indistinguishable group like `H2AC11 / H2AC18 / H2AJ`, an entry name, an isoform-suffixed accession -
  was listed as "not quantified" on a run where it had matched. It now asks the matcher itself, counts in
  members so the two halves of the sentence share a basis (a protein with an empty gene column previously
  counted toward neither, so a panel that matched everything could report "0 of 14 quantified"), and does
  not name the same protein twice.

- **A marker list written with display labels normalizes, and its log no longer claims every member was
  missing.** The members reach the matcher stripped of their labels, but the stage's "not quantified"
  list compared the *whole* member string against the identifiers in the data - so a labeled member never
  matched anything and every one of them was reported absent on a run where they had all been found. The
  same comparison also mis-reported members written as accessions, which are checked off by gene symbol
  above; both now compare on the match token, and report the label, which is the half you can act on.

- **PRISM no longer starts more concurrent exports than the machine can hold.** How many documents are
  exported at once was budgeted from a flat 2.5 GB per export - a figure measured on a 5 MB `.sky` with a
  116 MB `.skyd`, where it is almost entirely Skyline's fixed startup cost. On large documents it is not:
  two 11.3 GB plates measured 22.4 GB and 17.2 GB resident, so the estimate was low by about 8x and the
  memory check never actually limited anything. On a 64 GB machine PRISM ran both at once, drove it down
  to 4.5 GB free, and one Skyline stopped making progress - and did **not** recover when the other
  finished and returned 22 GB. Starving a Skyline is permanent, so an over-optimistic budget costs the
  export, not just time.

  The budget now scales with the largest document being exported (roughly twice its `.sky` size, with the
  old 2.5 GB as a floor), and is sized off the largest rather than the average, so one big document among
  many small ones still gets its own headroom. Cohorts of many small documents behave exactly as before.
  A machine where not even one export is expected to fit now says so, with the document size, the budget
  and the free memory - instead of silently proceeding and stalling inside Skyline.

- **A headless export that stops responding is now abandoned instead of waited on forever.** The
  command-line protocol has no exit code and reports only through its output stream, so a stalled Skyline
  is indistinguishable from a slow one - and nothing gave up. Two runs of the same cohort waited on a
  Skyline that was never coming back: one for 7 hours 35 minutes, the next for an hour.

  The bound is on **silence**, not on total duration, because total duration cannot be set without
  knowing how long the document should take: a legitimate export of an 11 GB document runs 20-45 minutes.
  Skyline narrates continuously while it works, and the longest gap between lines measured on a healthy
  export was under a minute, so 20 minutes of complete silence is the signal - and any output resets it,
  so an export that keeps reporting may run as long as it needs. Half way to the limit the log says so
  once, and the failure names the likely cause: fewer documents at a time, or close other Skyline
  windows.

- **Exported reports no longer arrive hidden.** On a network share whose server applies the Unix
  dot-file convention, the temporary name a report was written under carried the DOS *hidden* attribute
  across to the finished file - so a 1.4 GB parquet and its metadata landed invisible in Explorer while
  the pipeline read them perfectly well. Anyone opening the export folder saw nothing and concluded the
  export had failed.

- **A transient metadata failure is no longer cached permanently.** An export whose replicate metadata
  failed was still recorded as reusable, and the reuse check only re-exported when a *recorded* metadata
  file had gone missing - so a record saying there was never any metadata read as "correctly has none"
  and was honoured on every later run of an unchanged document. Sample types were then inferred from
  replicate names indefinitely, which is a different reference/QC split and therefore different batch
  correction, with only "Reusing the previous export" in the log. An export without metadata is no longer
  recorded for reuse at all.

- **A failed headless export no longer passes off the previous export as its own output.** The parquet
  result was accepted on its PAR1 marker alone, which says the bytes at that path are parquet - not that
  this run wrote them. When an export failed before Skyline received its arguments, the file left by an
  earlier run was still sitting there, so the log said `Exported ... 1,377,315,301 bytes, parquet` and
  handed it to the merge. Observed on a 2-plate cohort, which silently analyzed a report exported 25 days
  earlier, from a version of the `.sky` that had since been re-integrated.

  Skyline now writes each report - the transition report, the CSV fallback and the replicate metadata -
  to a sidecar file this run owns, which is moved into place only once it is known good. That answers
  "did this run write this file" directly, instead of inferring it, and has two consequences worth
  knowing. **A failed re-export no longer destroys a usable cached export**: the first fix deleted the
  destination up front, so a transient Skyline failure threw away the most expensive step of a re-run and
  cost a full re-export on the next attempt too. And a correct export is no longer rejected: freshness had
  been decided by file size and modification time, so re-exporting an unchanged document to a share that
  refused the delete, with second-granular timestamps, produced a byte-identical file that read as "not
  replaced" and failed. Reusing an unchanged document's previous export is unaffected - that path checks
  the document and tool version first.

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
