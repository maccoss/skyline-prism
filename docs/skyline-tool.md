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

### Protein lists

Named sets of proteins — plasma contaminants, EV markers, endothelial markers — each with its own color
and an on/off tick. Define them in **Protein lists…**, or import from a text/CSV file. They are saved per
user, so the same lists are available in every project and output directory.

Members may be written as UniProt accessions, gene names, or full protein names, with or without `sp|`
prefixes, species suffixes or isoform numbers — all forms are matched.

Tick **"Label this list's members on the plot"** to label a list. The plot's **right-click menu** adds
"Label the Skyline selection", and bulk "Label all protein lists" / "No labels".

---

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
