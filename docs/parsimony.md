# Protein parsimony: the bipartite graph, near-linear grouping, and multithreading

This document explains how PRISM turns a set of detected peptides into a minimal set of protein groups
(the "protein inference" or "parsimony" problem), why the original grouping step was `O(proteins²)` and
single-threaded, and how it was made **near-linear** and **multi-threaded** without changing a single
grouping result.

Implementation: `dotnet/src/SkylinePrism.Core/Parsimony/FastaParser.cs` (the bipartite graph) and
`ParsimonyEngine.cs` (`ComputeProteinGroups`). The Python side (`skyline_prism/parsimony.py`) and the Rust
reference (maccoss/osprey) implement the same grouping; determinism is asserted in both languages.

---

## 1. The problem

Mass spec identifies **peptides**, but we want to report **proteins**. A peptide can be a substring of many
proteins (shared/homologous sequences), so the mapping is many-to-many. Parsimony finds the smallest set of
proteins ("protein groups") that explains all the observed peptides, and decides which protein "owns" each
shared peptide (the *razor* rule). Getting this wrong either over-reports proteins (every homolog as its own
entry) or misassigns quantities.

---

## 2. The bipartite graph (peptide ↔ protein)

The foundation is a **bipartite graph**: peptides on one side, proteins on the other, with an edge
`peptide — protein` whenever the peptide sequence is a substring of the protein sequence.

`FastaParser.BuildMap` constructs it (`FastaParser.cs:171`):

- Parse the FASTA into `accession → (sequence, name, gene, description)`.
- **Normalize** every protein sequence: uppercase, and (by default) replace `I → L`, because mass spec cannot
  distinguish isoleucine from leucine (`NormalizeForMatching`). Peptides are normalized the same way (mods
  stripped, uppercased, `I→L`).
- For each detected peptide, substring-search it against **every** protein sequence; the proteins that contain
  it are its edges.

```csharp
// FastaParser.cs:186 — the edge-building loop is already parallel over peptides.
Parallel.For(0, peptides.Length, p =>
{
    var norm = NormalizeForMatching(peptides[p], handleIlAmbiguity);
    var hits = new List<string>();
    for (var i = 0; i < normSeq.Length; i++)
        if (normSeq[i].Contains(norm, StringComparison.Ordinal))   // peptide ⊆ protein?
            hits.Add(accs[i]);
    matchesPerPeptide[p] = hits;
});
```

The result is a `PeptideProteinMap` holding the graph as **two adjacency indices** (`ParsimonyEngine.cs:12`):

- `PeptideToProteins[peptide] = { proteins containing it }`
- `ProteinToPeptides[protein] = { its detected peptides }`

These two indices are what make the *grouping* near-linear (Section 5). Note that `BuildMap` itself is
`O(peptides × proteins × substring)`, parallelized over peptides — it was never the single-threaded bottleneck;
the grouping step was.

> When the peptide→protein edges come from Skyline's "Protein Accession" column instead of a FASTA, the same
> `PeptideProteinMap` is built directly from the report; everything below is identical.

---

## 3. The grouping algorithm (`ComputeProteinGroups`)

Given the bipartite graph, grouping runs five deterministic steps. The **only** thing that changed for
performance is Step 1's *implementation*; the results of all five steps are byte-for-byte identical to before.

### Step 1 — Subsumable proteins (strict subsets)
A protein `a` is **subsumable** if its peptide set is a *proper subset* of some other protein `b`'s peptide
set — `a` explains nothing `b` doesn't, so `a` is folded into `b`. Among all valid supersets, we keep the
**lexicographically smallest** so the choice is deterministic. Subsumed proteins are recorded but still listed
as members of their superset's group.

### Step 2 — Indistinguishable proteins (identical sets)
Active proteins with the **exact same** peptide set are indistinguishable (e.g. two isoforms detected by the
same peptides). They are grouped by a signature = their sorted peptide list joined into a string, then hashed
(`bySet`, `ParsimonyEngine.cs:224`). This is `O(active proteins × peptides)` — linear, already fast. The
lexicographically smallest accession is the group's **canonical** protein (`Canonical`, `:240`).

### Step 3 — Unique vs shared peptides
For each peptide, count how many *canonical* proteins it maps to. Exactly one ⇒ **unique** (owned outright);
more than one ⇒ **shared** (contested) (`:245`).

### Step 4 — Iterative greedy razor
Shared peptides are assigned one protein at a time by a greedy set-cover (`:283`): repeatedly pick the
canonical protein with the **most unique peptides** (ties → **largest** total peptide set, then **lowest**
accession — aligned to osprey's group-ID ordering), assign it all remaining shared peptides it can claim, and
remove those from the pool. This is inherently **sequential** (each pick depends on what's left), but it runs
only over the *shared* peptides — usually a small minority — so it is cheap.

### Step 5 — Build groups
Assemble each `ProteinGroup`: the canonical protein + indistinguishable members + subsumed proteins, with its
unique + razor peptides (`:311`).

---

## 4. Why the old Step 1 was slow: `O(proteins²)`

The original subsumable detection compared **every ordered pair** of proteins:

```csharp
// OLD — all-pairs, single-threaded.
foreach (var a in proteins)
    foreach (var b in proteins)
        if (a != b && protToPep[a].IsProperSubsetOf(protToPep[b])) { subsumedBy[a] = b; break; }
```

With `N` candidate proteins this is `N²` pairs, and each pair does a set-subset test costing `O(|peptides of a|)`.
On a real 190k-peptide run the FASTA map yielded tens of thousands of candidate proteins (a peptide from a large
UniProt search maps to many homologs), so `N²` was in the **billions** of set comparisons — and it ran on a
**single core**. It took ~2 hours and hadn't finished. Small datasets (the mini fixtures, ~thousands of proteins)
never exposed it; only real data did.

---

## 5. The near-linear fix: use the graph's adjacency

Key observation: **a proper superset `b` of `a` must contain *every* peptide of `a`.** Therefore `b` is adjacent
(in the bipartite graph) to every peptide of `a` — in particular to `a`'s **rarest** peptide (the one with the
fewest proteins). So instead of scanning all `N` proteins for candidate supersets, we only scan the proteins on
the *other end of one edge* — `PeptideToProteins[rarest peptide of a]`:

```csharp
// NEW — candidate supersets come from the peptide→protein index, not from all proteins.
var pepsA = protToPep[a];
string pivot = a's peptide with the fewest proteins;        // pepToProt[pep].Count minimized
string? smallest = null;
foreach (var b in pepToProt[pivot])                          // only proteins sharing a's rarest peptide
{
    var pepsB = protToPep[b];
    if (pepsB.Count > pepsA.Count && pepsA.IsProperSubsetOf(pepsB)
        && (smallest is null || string.CompareOrdinal(b, smallest) < 0))
        smallest = b;                                        // keep the lexicographically smallest superset
}
if (smallest is not null) subsumedBy[a] = smallest;
```

Correctness is exact, not approximate: the candidate set `pepToProt[pivot]` is guaranteed to contain **all**
proper supersets of `a` (any superset contains the pivot peptide), and we still pick the lexicographically
smallest — identical to the old ordered scan that broke on the first match.

**Complexity.** The work per protein is bounded by the number of proteins sharing its rarest peptide, which is
tiny for the pivot (that's why we choose the rarest peptide). Summed over all proteins this is roughly
`Σ_peptide (proteins-per-peptide)` bounded work — effectively **linear in the number of graph edges** rather than
quadratic in the number of proteins. In practice the ~2-hour step drops to well under a second of algorithmic
work.

This is the sense in which "the bipartite graph gives near-linear performance": the graph's **edges are the
candidate list**. Homology makes `proteins²` explode, but the same homology is captured compactly by the
peptide→protein adjacency, and we walk edges instead of the full protein × protein matrix.

---

## 6. Multithreading

Each protein's smallest-superset computation reads only the **immutable** maps (`protToPep`, `pepToProt`) and
writes exactly **one** output key (its own). That makes Step 1 embarrassingly parallel:

```csharp
// ParsimonyEngine.cs:~196
var subsumedByConc = new ConcurrentDictionary<string, string>(StringComparer.Ordinal);
Parallel.ForEach(proteins, a =>
{
    // ... rarest-peptide pivot + candidate scan (Section 5) ...
    if (smallest is not null) subsumedByConc[a] = smallest;   // each task writes its own distinct key
});
var subsumedBy = new Dictionary<string, string>(subsumedByConc, StringComparer.Ordinal);
```

Thread-safety comes for free because:
- The graph maps are only **read** during this phase (concurrent reads of a `Dictionary`/`HashSet` are safe).
- `HashSet.IsProperSubsetOf` is a read-only operation.
- Every task writes a **different** key, so `ConcurrentDictionary` sees no key contention.

`Parallel.ForEach` uses the whole machine (one partition per core), so the step that pegged a single core now
scales across all of them.

### Keeping it deterministic despite parallelism
A `ConcurrentDictionary` filled by many threads has a non-deterministic *iteration order*. If anything
downstream depended on that order, parallelizing could change the output. It doesn't, because we **rebuild the
superset→subsumed lists by iterating the proteins in sorted order**, not by iterating the concurrent dictionary:

```csharp
// ParsimonyEngine.cs:211 — deterministic list order regardless of parallel fill order.
foreach (var a in proteins)               // 'proteins' is sorted (StringComparer.Ordinal)
    if (subsumedBy.TryGetValue(a, out var sup))
        (subsumingToSubsumed[sup] ??= new()).Add(a);
```

Every other step (indistinguishable signatures, canonical selection, razor ordering, group assembly) already
sorts with `StringComparer.Ordinal` before making a choice, so the entire pipeline is order-independent.

`FastaParser.BuildMap` is likewise parallel (`Parallel.For` over peptides, Section 2) and writes each peptide's
matches into its own array slot, so it is deterministic by construction.

---

## 7. What is *not* parallel, and why that's fine

The **razor** (Step 4) is a greedy set-cover: each iteration's choice depends on which shared peptides remain,
so it cannot be parallelized without changing semantics. But it iterates only over **shared** peptides, which are
a minority in a well-designed assay, and each iteration is cheap. It was never the bottleneck — the `O(N²)`
subsumable scan was.

---

## 8. Complexity summary

| Stage | Old | New |
|-------|-----|-----|
| Build bipartite graph (`BuildMap`) | `O(peptides × proteins)`, **parallel** | unchanged |
| Step 1 subsumable | `O(proteins²)` × subset-cost, **single-threaded** | ~`O(edges)`, **parallel** |
| Step 2 indistinguishable | `O(proteins × peptides)` | unchanged |
| Step 3 unique/shared | `O(edges)` | unchanged |
| Step 4 razor (greedy) | `O(shared peptides × candidates)`, sequential | unchanged (sequential by nature) |
| Step 5 build groups | `O(proteins + peptides)` | unchanged |

---

## 9. Determinism & parity guarantees

- The grouping depends only on the peptide/protein **sets**, never on input row order or thread scheduling.
- Proven by `ParsimonyOspreyTests.Grouping_IsOrderIndependent` (C#) and `tests/test_parsimony_osprey.py`
  (Python), which run the same identical-set / subset / all-mode / razor / determinism cases and keep
  Python ↔ C# ↔ osprey in lockstep.
- The 23 parsimony + pipeline-parity tests all pass with the near-linear + parallel implementation, confirming
  the grouping is **identical** to the original all-pairs version.
