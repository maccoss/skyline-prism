"""Parsimony parity with Osprey (maccoss/osprey, crates/osprey-fdr/src/protein.rs).

Mirrors the C# ``ParsimonyOspreyTests`` so the Python and C# implementations stay locked to the
same Osprey behaviour: identical-set merging, subset elimination, all-mode shared-to-all, the razor
iterative greedy set cover (including the largest-peptide-set tiebreak), and order-independence.
"""

from __future__ import annotations

from skyline_prism.parsimony import compute_protein_groups


def build(prot_to_peps: dict[str, list[str]]):
    """Build groups from a protein -> peptides map (one edge per (protein, peptide))."""
    protein_to_peptides = {p: set(peps) for p, peps in prot_to_peps.items()}
    peptide_to_proteins: dict[str, set[str]] = {}
    for prot, peps in prot_to_peps.items():
        for pep in peps:
            peptide_to_proteins.setdefault(pep, set()).add(prot)
    protein_to_name = {p: p for p in prot_to_peps}
    return compute_protein_groups(protein_to_peptides, peptide_to_proteins, protein_to_name)


def by_leading(groups, leading):
    return next(g for g in groups if g.leading_protein == leading)


def razor(groups):
    return {g.leading_protein: sorted(g.razor_peptides) for g in groups}


def test_identical_sets_merged():
    # Osprey test_basic_parsimony_grouping: P1,P2 identical {A,B,C}; P3 {D,E}.
    groups = build({"P1": ["A", "B", "C"], "P2": ["A", "B", "C"], "P3": ["D", "E"]})
    assert len(groups) == 2
    merged = by_leading(groups, "P1")
    assert sorted(merged.member_proteins) == ["P1", "P2"]
    assert sorted(merged.unique_peptides) == ["A", "B", "C"]


def test_strict_subset_eliminated():
    # Osprey test_subset_elimination: P2 {A,B} is a strict subset of P1 {A,B,C}.
    groups = build({"P1": ["A", "B", "C"], "P2": ["A", "B"]})
    assert len(groups) == 1
    assert groups[0].leading_protein == "P1"
    assert "P2" in groups[0].subsumed_proteins


def test_all_mode_shared_maps_to_all():
    # Osprey test_shared_peptides_all_mode: SHARED is in both groups' mapped peptides.
    groups = build({"P1": ["A", "B", "SHARED"], "P2": ["C", "D", "SHARED"]})
    assert len(groups) == 2
    assert "SHARED" in by_leading(groups, "P1").all_mapped_peptides
    assert "SHARED" in by_leading(groups, "P2").all_mapped_peptides


def test_razor_assigns_to_most_unique():
    # Osprey Example 1: P1 has 3 unique, P2 has 1 -> SHARED -> P1.
    r = razor(build({"P1": ["A", "B", "C", "SHARED"], "P2": ["D", "SHARED"]}))
    assert r["P1"] == ["SHARED"]
    assert r.get("P2", []) == []


def test_razor_cascading():
    # Osprey Example 2: P1 claims X,Y (3 unique); then P2 claims Z.
    r = razor(
        build({"P1": ["A", "B", "C", "X", "Y"], "P2": ["D", "X", "Z"], "P3": ["E", "Y", "Z"]})
    )
    assert r["P1"] == ["X", "Y"]
    assert r["P2"] == ["Z"]
    assert r.get("P3", []) == []


def test_razor_unique_count_tie_prefers_larger_set():
    # Tie on unique count (all 1). Osprey breaks the tie by lowest group ID = largest peptide set,
    # so P2 (3 peptides) claims BOTH shared peptides. Accession-only tiebreak would give X -> P1.
    r = razor(build({"P1": ["A", "X"], "P2": ["B", "X", "Y"], "P3": ["C", "Y"]}))
    assert r["P2"] == ["X", "Y"]
    assert r.get("P1", []) == []
    assert r.get("P3", []) == []


def test_grouping_order_independent():
    # Subsumption + an indistinguishable pair + shared peptides; groups must not depend on order.
    scenario = {
        "P1": ["A", "B", "C", "X", "Y"],
        "P2": ["D", "X", "Z"],
        "P3": ["E", "Y", "Z"],
        "P4": ["A", "B", "C", "X", "Y"],  # indistinguishable from P1
        "P5": ["D", "X"],                  # strict subset of P2
    }

    def sig(groups):
        return sorted(
            (
                g.leading_protein,
                tuple(sorted(g.member_proteins)),
                tuple(sorted(g.unique_peptides)),
                tuple(sorted(g.razor_peptides)),
                tuple(sorted(g.all_mapped_peptides)),
            )
            for g in groups
        )

    baseline = sig(build(scenario))
    for reordered in (
        dict(reversed(list(scenario.items()))),
        dict(sorted(scenario.items(), reverse=True)),
        dict(sorted(scenario.items(), key=lambda kv: kv[1])),
    ):
        assert sig(build(reordered)) == baseline
