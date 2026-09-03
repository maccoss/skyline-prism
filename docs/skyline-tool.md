# The Skyline external tool

PRISM ships as a Skyline external tool: a Windows window that runs the pipeline on one or more Skyline
documents and then shows interactive plots of the result. Installation is in the
[README](../README.md#skyline-external-tool-windows); this describes what the window does.

It also runs **standalone** — double-click the installed `SkylinePrism.exe` and you get the same window
with no Skyline attached. Everything below works except the parts that talk to a document (exporting from
an open document, and the two-way selection on the Dynamic Range plot).

> Anything the tool can do, the `prism` CLI can do too. **Show command line** on the Settings tab prints
> the exact `prism run` invocation for the current settings, which is the supported way to move a
> configuration to a headless or Linux machine.

## Tabs

| Tab | What it is for |
|-----|----------------|
| **Inputs** | The documents/reports to process, one per batch, each with a batch label |
| **Settings** | Pipeline options; **Run PRISM**, **Stop**, and **Show command line** |
| **QC Plots** | Normalization and batch-correction diagnostics (CV, PCA, intensity, RT, correlation) |
| **Spectrum density** | How many precursors were detected in each DIA spectrum of a run |
| **Dynamic Range** | Log10 abundance against abundance rank, over the corrected matrices |
| **Log** | The full run log — the first place to look when something is slower or emptier than expected |

---

## How PRISM reaches Skyline

Three different mechanisms, chosen by what each input is. Worth knowing, because they have different
capabilities and only one of them is currently affected by a Skyline bug.

| Input on the Inputs tab | Mechanism | Report format |
|---|---|---|
| **Open in Skyline** | Live JSON-RPC to the running instance | **parquet** |
| **Skyline document (.sky)**, closed | `SkylineRunner` — launches the installed Skyline headlessly and opens the document with `--in` | **parquet** |
| **Exported report** | Nothing; the file is used in place | as supplied |

A run may mix all three freely: the merge reads parquet and CSV in the same pass, so ten closed
documents, an open one and a report you exported last week combine into one cohort.

`SkylineCmd.exe` is a fourth way to reach Skyline, used for one job only: reading a DIA acquisition's
isolation windows out of a data file for the Spectrum density tab. That runs against a **throwaway**
document (`--new`) so your own is never touched, and `--new` currently **hangs** through SkylineRunner —
observed on Skyline-daily 26.1.1.209, reproducible with Skyline's own runner, and raised with the Skyline
developers. `SkylineCmd` does the same work in ~9 s.

> The reverse trade also exists: `SkylineCmd` cannot write parquet at all (its `.exe.config` lacks the
> Parquet.Net binding), which is why report export uses SkylineRunner. Report export opens an existing
> document, so it is unaffected by the `--new` stall. `PRISM_SKYLINECMD` forces `SkylineCmd` if you ever
> need to — at the cost of CSV instead of parquet.

---

## Dynamic Range

Log10 abundance against abundance rank — the shape of Skyline's Relative Abundance plot — computed from
the **corrected** PRISM matrices (`corrected_proteins.parquet` / `corrected_peptides.parquet`).

Switch between **Protein** and **Peptide**, and average over **All** replicates or any subset you tick.
Abundances are averaged on the **linear** scale and only then log-transformed, and anything never measured
in the selection is dropped rather than plotted at zero. Changing the replicate selection re-ranks
everything, because the ordering depends on what is being averaged.

### Viewing another rollup

The **Rollup** drop-down re-rolls the proteins under any method — **sum**, **median polish**, **top N**,
**MaxLFQ**, **iBAQ** — whatever the run itself used. This is not cosmetic: the method decides what the y
axis *is*. A summing method carries the peptide count into the answer, so a protein with 121 peptides
outranks one with 44 partly for having more of them; median polish estimates the level of a typical
peptide and does not; **iBAQ divides by the theoretical peptide count and is the only one of them meant
for comparing one protein against another**. Sum is what Skyline's Relative Abundance plot shows.

- **As run** (the default) reads `corrected_proteins.parquet` — the run's own numbers.
- Any other choice re-rolls `corrected_peptides.parquet` with that method, using the run's own
  `min_peptides` / `topn` settings so the method is the only thing that changes. It does **not** re-run
  parsimony or the protein-level normalization and batch correction, so it shows the shape and ordering
  that method gives rather than the numbers a re-run would produce. The status line says
  `[recomputed from corrected_peptides]` whenever that is what you are looking at.
- Shared peptides count toward every group they map to (the pipeline's default `all_groups`). A run
  configured for `unique_only` or `razor` is not reproduced here — which group parsimony gave a shared
  peptide to is not recorded per peptide in the peptide matrix.
- **iBAQ** needs a FASTA, taken from the run's `protein_rollup.ibaq.fasta_path` or `parsimony.fasta_path`.
  Without one it divides by the *observed* peptide count instead, and the status line says so — that is a
  materially different quantity, closer to a per-peptide mean than to an absolute-abundance estimate.

The digest and the rollup run off the UI thread with a progress bar, so a large cohort does not look like
a hang; the theoretical counts are cached, so flipping back to iBAQ is instant. The drop-down is grayed
out at peptide level — there is no protein rollup below a peptide.

The rollup needs `corrected_peptides.parquet`, so an output directory without one — an older run, or
one whose peptide arm never finished — can only show **As run**.

### Selection works in both directions

- **Click a point** → that protein or peptide is selected in Skyline's document tree. At peptide level it
  selects the *peptide* node, so its chromatograms come up.
- **Select in Skyline** → the matching point is ringed on the plot. This follows the Targets tree and the
  document grids, and it follows a selection made at *any* depth: with a transition selected, the protein
  plot rings the protein containing it.

The status line under the plot names whatever was matched. Because PRISM runs its own parsimony, which can
group proteins differently from the document, it reports **every** protein group a peptide belongs to
rather than implying a single assignment.

> Following Skyline's selection polls the document (Skyline cannot notify us), so it runs only while this
> tab is on screen. It stops when you switch tabs or close the window.

### Checking a marker normalization

When a run used `marker_normalization`, two extra entries appear in the **Plot** picker on the QC Plots
tab. They are the two halves of the panel's PC1, read from `marker_normalization.csv` — the numbers the
run actually subtracted, not a recomputation — so **Level** and **View** are grayed out for them: there
is one score per replicate for the whole run, with no before/after and no peptide/protein version.

- **Marker score** — the per-sample score, one column per **Group-by** value. Set Group-by to the
  study's condition. The score is supposed to measure how much of the captured material a sample
  contributed, so it should track section size or input amount. **If it separates the study groups, the
  panel is tracking the phenotype**, and normalizing on it removes the finding along with the capture.
  That judgement cannot be automated — the same separation is what you would see if the biology really
  does change the marked material — which is why it is a plot and not a warning.
- **Marker loadings** — each marker's PC1 contribution, largest first, with opposing markers in red.
  Opposing signs are normal and are why the score is PC1 rather than a mean. The title gives the largest
  marker's share of the axis and says so when it is over half: a panel carried by one protein is a
  single-protein normalization wearing a panel's clothes.

The replicate picker applies to **Marker score**, so a suspicious group can be isolated. **Group-by** is
grayed out for **Marker loadings**, which draws one bar per protein in the panel — grouping or filtering
replicates cannot change it.

Marker score refuses to draw above 12 Group-by values and names a better column instead: one column per
subject cannot show whether the *study's* groups separate, which is the only question it asks. Past six
groups the legend gives way to per-group counts on the tick labels, so a group of 2 and a group of 40
stay distinguishable.

### Protein lists

Named sets of proteins — plasma contaminants, EV markers, endothelial markers — each with its own color
and an on/off tick. Define them in **Protein lists…**, or import from a text/CSV file. They are saved per
user, so the same lists are available in every project and output directory.

The **Protein lists…** button is on the Settings tab as well, beside the marker-normalization picker,
because a panel usually has to be curated before it can be selected there. It is the same editor and the
same one set of lists; a list created from either place is immediately selectable in the picker.

### Predefined and your own

The editor has two tabs. **Predefined** holds the 65 panels PRISM ships, under collapsible category
headings; **My lists** holds yours, as a flat list. The
split is not cosmetic: a predefined panel is **read-only**, so it means the same thing on every machine,
which is what makes one citable in a methods section. **Duplicate to my lists** makes an editable copy.
Ticking a predefined panel stores only whether it is shown and labeled — not a copy of its members — so a
panel you have turned on still picks up any later correction to it.

A list of yours with the same name as a shipped one **wins that name**; the shipped panel stays reachable
as `<name> (PRISM)`. Worth knowing, because a saved list curated for *highlighting* will otherwise stand
in for a shipped panel built for *normalizing* — same name, different purpose, different answer.

Shipped panels arrive **unticked**, so nothing is colored until you ask for it. All of them can also name
a marker normalization (`marker_normalization.protein_list` — see [parameters.md](parameters.md) for the
keys and [methods.md](methods.md#marker-protein-normalization) for what the normalization does).

The headings are not cosmetic - they separate panels by **what they are for**, which matters more than
their membership:

| Category | Panels | What it is for |
|---|---|---|
| **Normalizers** | `EV markers (core)`, `Glomerulus`, `Histones (proteomic ruler)`, `Ribosomal proteins`, `Mitochondrial content`, `White matter`, `Grey matter` | Proportional to how much material was captured, and *not* to the phenotype. Each answers a different "per unit of": marked material, glomerulus, **cell** (histones), biosynthetic capacity, mitochondrion, dissected tissue |
| **Plasma and blood** | `Classic plasma proteins`, `Free soluble acidic plasma proteins`, `Immunoglobulin and complement`, `Lipoproteins (LDL/VLDL/HDL)`, `Platelet microparticles`, `EV markers (extended)` | What is in the sample matrix. Good for highlighting; usable as a denominator only where the biology under study does not move them |
| **Endothelial** | Arterial, venous, capillary, pan-endothelial, brain/BBB, liver sinusoidal, kidney glomerular | Vascular bed identity — which endothelium a signal comes from |
| **Epithelial** | Pan-epithelial, kidney tubule, intestine, lung | Epithelial identity, kept separate from endothelial: different tissues that happen to alliterate |
| **Readouts and contamination** | `Hemolysis`, `Fibrinogen`, `Keratin contamination`, `Tubular contamination`, `Common contaminants (cRAP)`, `Housekeeping proteins` | Their abundance **is** the problem being looked for, so dividing by it removes the evidence |
| **Pathways and processes** | Oxidative phosphorylation, glycolysis, TCA cycle, proteasome, lysosome, spliceosome/hnRNP, extracellular matrix, actin cytoskeleton, antigen presentation, acute phase response, chaperones, redox, DNA damage repair, autophagy, unfolded protein response, innate immune signaling, apoptosis, fatty acid oxidation, cell cycle, epithelial-mesenchymal transition, hypoxia response, glucose and lipid metabolism, insulin signaling | For seeing where a process sits on a plot. Normalizing to one would remove the biology under study |
| **Brain and neurodegeneration** | `Neuronal markers`, `Astrocyte markers`, `Microglia markers`, `Oligodendrocyte and myelin` (identity); `Alzheimer's disease`, `Parkinson's disease` (+ RAB substrates, lysosomal), `ALS and FTD`, `Huntington's disease`, `Synaptic proteins`, `Brain fluid-like proteins` (display) | Cell types can be denominators; the disease panels cannot — in a study of them, their abundance *is* the result. `White matter`/`Grey matter` sit under Normalizers, with the caution below |

**Readouts and pathways are refused by `marker_normalization`** rather than merely discouraged — naming
one gives an error explaining why. Both fail for the same reason: their abundance is the signal, not the
scale, so dividing by it removes exactly what you were looking for. That failure is otherwise silent —
the run succeeds and the numbers look plausible.

The pathways are hand-written and sized for a plot (15–35 members) rather than imported from an ontology,
which keeps them free of anyone's redistribution terms (KEGG in particular cannot be shipped) and stops a
single panel coloring a tenth of the points. For a pathway that is not here, import a gene list from
Reactome or MSigDB into a list of your own.

One pathway is read differently from the rest. **`Insulin signaling` is regulated by phosphorylation, not
by abundance**: its members are present whether or not the cascade is active, so a flat panel is the
expected result and says nothing. It earns its place on a phospho-enriched run, where the sites are the
measurement. For abundance work, `Glucose and lipid metabolism` measures what the pathway *does* rather
than what it is made of. `Epithelial-mesenchymal transition` is also read as a **balance** — its
epithelial and mesenchymal halves move in opposite directions — rather than as a total.

Two details worth knowing:

- **`Histones (proteomic ruler)`** follows Wisniewski et al., *Mol Cell Proteomics* 2014
  ([doi:10.1074/mcp.M113.037309](https://doi.org/10.1074/mcp.M113.037309)): summed histone signal tracks
  DNA, and therefore cell number. It is deliberately comprehensive and carries both the current HGNC and
  the legacy `HIST1H*` nomenclatures. PRISM gives a *relative* per-cell adjustment, not the paper's
  absolute copy-number scaling.
- **`Common contaminants (cRAP)` is listed by accession, not gene symbol** — deliberately. These are
  non-human proteins: `ALB` for bovine serum albumin would match *human* albumin, and `ENO1` would match a
  yeast enolase spike-in rather than the housekeeper. Nor is the UniProt entry name safe: `ALBU_BOVIN`
  reduces to `ALBU`, and so does human `ALBU_HUMAN`, because species suffixes are stripped so panels work
  across human and mouse. The panel carried `ENO1_YEAST` and `TRYP_PIG` until that was noticed, which
  colored human alpha-enolase as a contaminant on every human run; a test now holds every member of the
  panel to the UniProt accession pattern.

  So that the panel is still readable, a member may be written `<accession> = <name>` —
  `P00761 = Trypsin (porcine)`. **Everything left of the `=` is matched; everything right of it is only
  displayed.** The syntax works in your own lists too, and is worth using anywhere you list accessions.
  It is a starting set; import your search's cRAP FASTA to extend it.

Panels carry mouse symbols wherever the ortholog is named differently (`Hbb-bs`, `Ighg2a`, `Lyz1`).
Matching is case-insensitive, so a conserved symbol needs no help.

Two things about the kidney panels are deliberate and worth knowing before trusting them on a new cohort:

- The glomerular panel is weighted toward **structure**. `NPHS1`/`NPHS2` are left out because podocyte
  loss *is* the phenotype in most glomerular disease, and a score dominated by them would regress out the
  finding along with the capture; `COL4A1`/`COL4A2` are left out because they are ubiquitous basement
  membrane and would make the score track any basement membrane rather than the GBM. Check how many of
  the panel are quantified, and whether PC1 separates large sections from small ones (capture — what you
  want) rather than diseased from control (pathology — what you do not).
- **Tubular contamination is not a normalizer.** Its abundance *is* the contamination; normalizing to it
  would remove the thing being measured. Tick it on the plot, do not name it in `marker_normalization`.

Members may be written as UniProt accessions, gene names, or full protein names, with or without `sp|`
prefixes, species suffixes or isoform numbers — all forms are matched.

Tick **"Label this list's members on the plot"** to label a list. The plot's **right-click menu** adds
"Label the Skyline selection", and bulk "Label all protein lists" / "No labels".

---

## MS2 signal accounting

Settings row **9** adds a QC-report section answering "how much of the MS2 signal does this analysis
actually put a name to?" - per replicate, the signal the run assigns to a peptide, with a line per
protein list ticked visible in **Protein lists...** (the same set the Dynamic Range tab highlights).
Each region of MS2 signal space - isolation window, extraction window, integration bounds - is counted
once, so two co-isolated peptides sharing a fragment mass are not both credited with it.

**measure** picks what to total:

| | What it sums | Needs |
|---|---|---|
| `signal` | Each transition's gross peak area (`Area + Background`) | Any export |
| `ions` | Skyline's `LC Peak Transition Ion Count` - intensity x injection time per spectrum, summed across the peak | An export carrying that column |

`ions` is the better measure: both it and an acquired total are then counts of ions, so no unit or
background correction applies, and it cannot be recovered from an area afterwards (on AGC-controlled
data the injection time varies by two orders of magnitude within a run and anti-correlates with
intensity). It is grayed out, with the reason as its tooltip, until every input can supply the column.

**Export ion counts** is what makes that possible: each Skyline document is exported with the
`PRISM-Ions` report - the standard report plus that one column - instead of `PRISM`. Expect roughly a
**30x slower export**: measured at about 4 hours instead of 9.5 minutes on a 6.5 GB, 46M-row document,
because Skyline reads every transition's chromatogram points to compute it. Three things follow:

- **The option and the measure are separate.** Once ion counts are exported, both measures are
  available, so a later re-run can plot either without exporting again.
- **A closed document is exported once per variant** (into `skyline-reports/with-ion-counts/`), so
  switching the measure back and forth does not repeat the four hours. A document open in Skyline has
  no such cache - a live document can hold unsaved edits - so it re-exports on every run.
- **An export Skyline has started cannot be recalled.** Stop ends PRISM's run, not Skyline's export;
  the only way to end that early is to close Skyline, which loses unsaved changes to the document.

The option is ignored for an input that is already an exported report, and PRISM refuses it outright
when one of those lacks the column - inputs whose columns differ cannot be merged into one cohort, so
paying for the slow export there would only fail in Stage 1.

The tool also reads the document's own product-ion extraction tolerance
(`Transition Settings > Full-Scan`) rather than using the config default, since that is what decides
when two fragments are the same detector counts. Every input is asked, and disagreement is a warning.

## Spectrum density

A map of how many peptide precursors were detected in each DIA spectrum of a run: retention time across,
precursor m/z up, color = precursors per cell. Pick the run from the drop-down, set the **RT bin**
(default 0.1 min) and the **max q-value** that counts as a detection (default 0.01). Hovering reads out
the isolation window and exact count under the cursor.

**The rows are the acquisition's real isolation windows**, not an assumed bin width, so a cell really is
one DIA spectrum — including for variable-width and staggered schemes. This matters more than it sounds:
a real forbidden-zone scheme has 3.0014 Th windows starting at 400.4319, so a uniform 3 Th grid starting
at 400 sits ~14% of a window off and cuts through the very precursor clusters the scheme exists to keep
intact.

PRISM gets those windows from the document's Full-Scan settings when it defines a scheme, and otherwise —
the usual `Results only` case, where Skyline keeps the windows only inside the raw files — by having
Skyline read them from one of the run's data files. That read happens when you click **Run PRISM**, while
the raw data is most likely still where the document says it is, and the result is saved to
`isolation_schemes.xml` in the output directory so the map still bins correctly when the tab is reopened
later with no Skyline running.

It runs **alongside** the pipeline, not before it, so a slow or unreachable data file delays nothing. If
it does not finish, the map falls back to clearly-labeled uniform bins and the log says so.

**The tab is for DIA.** Skyline's importer can only read a repeating isolation cycle out of a data file,
so targeted methods (PRM, MTM) have no route to their real windows — getting them means walking the file's
scan headers, which belongs in Skyline or ProteoWizard rather than in an external tool. When a document's
acquisition method is not DIA, the map is still drawn — on whatever grid is available — but the status line
**warns** that the rows are not the windows the data was acquired with and that a cell is therefore not a
co-fragmentation load, and it calls a cell a "row" rather than a "spectrum".

The tab reads the merged data from the output directory, so it works for a run that has just
finished *and* for any previous run's output directory, with no Skyline connection. Either layout
opens: the partitioned `merged_data/` directory, or the single `merged_data.parquet` written before
dotnet-v26.12.0.

---

## Choosing a FASTA

**Protein FASTA** on the Settings tab is the search database, and it is optional. It writes
`parsimony.fasta_path`, which two things read:

- **Enzyme-aware parsimony.** With a FASTA, a peptide is attached to a protein only where the digestion
  enzyme could actually have produced it — the enzyme itself is read from the Skyline document, so the
  check matches the search that produced the data. Without one, protein groups come from the Skyline
  **Protein Accession** column instead. The difference is not cosmetic: `AKEGVVAAAEK` is a substring of
  beta-synuclein but is preceded there by `M`, not `K`/`R`, so trypsin cannot liberate it — it is
  proteotypic to alpha-synuclein. See [parsimony.md](parsimony.md).
- **iBAQ's denominator.** `protein_rollup: ibaq` divides by the theoretical peptide count from an
  in-silico digest of this database. Without a FASTA it divides by the *observed* count instead, which
  is closer to a per-peptide mean than to an absolute-abundance estimate — so the Settings tab says so
  in red rather than leaving it to the log.

The same setting feeds the Dynamic Range tab's **iBAQ** rollup view, which reads it back from the run's
`parameters.json`.

**The run keeps a copy.** Whatever FASTA a run used is copied into `fasta/` in the output directory, and
`parameters.json` records where it came from and where the copy sits. An absolute path describes a run;
it does not let anyone repeat it once the database has been reorganized, renamed, or the output directory
handed to a collaborator. `--from-provenance` prefers the original whenever it is still there — so a
re-run is not invalidated by the copy existing — and falls back to the copy only when the original has
gone, saying so rather than substituting a database silently.

The rollup picker offers `topn`, `maxlfq` and `ibaq` alongside `median_polish` and `sum`; iBAQ was
previously absent because nothing in the tool could give it a database.

---

## Stopping a run

**Stop** cancels the pipeline, and closing the window while a run is in progress asks first and then stops
it. Cancellation is checked *inside* the long stages — the merge query is interrupted through DuckDB, and
the rollup and normalization check per peptide block and per row group — so it takes effect in seconds
rather than at the end of whatever stage was running.

Files already written are left in place. They are intermediates of an incomplete run, and the next run
rebuilds them.

> **One limit:** a report export already running inside a live Skyline cannot be recalled. That work
> belongs to Skyline's process and there is no way to stop it from here, so Skyline finishes writing that
> file — PRISM just stops waiting for it and does not use it.

---

## Environment variables

Escape hatches for when an automatic choice picks badly. None are needed normally.

| Variable | Effect |
|----------|--------|
| `PRISM_TEMP_DIR` | Where DuckDB spills the Stage 1 sort. By default this sits beside the output, unless the output is on a network drive, in which case it falls back to the machine's temp directory. Set this when the automatic choice lands on a small or quota'd disk. Stage 1 logs the directory it chose. |
| `PRISM_ISOLATION_TIMEOUT_SEC` | How long to let Skyline read isolation windows out of a data file before giving up (default 300). Reading them normally takes ~10 s; raise this only if your data really is that slow to reach. |
| `PRISM_SKYLINECMD` | Full path to `SkylineCmd.exe`, when the automatic discovery finds the wrong installation. |

Memory for the Stage 1 merge is sized from the machine's **free** memory and is set with the
`processing.merge_memory_mb` config key rather than an environment variable — see
[parameters.md](parameters.md#processing--parallelism--memory).
