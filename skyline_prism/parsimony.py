"""Protein parsimony and grouping module.

Implements greedy set cover algorithm to create protein groups where:
1. Each peptide maps to exactly one protein group
2. Protein groups represent the minimal set that explains all peptides
3. Shared peptides go to the group with the most unique peptides
"""

import logging
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ProteinGroup:
    """Represents a protein group after parsimony analysis."""

    group_id: str
    leading_protein: str              # Representative accession
    leading_protein_name: str         # Gene name or description
    member_proteins: list[str]        # All indistinguishable proteins
    subsumed_proteins: list[str]      # Proteins whose peptides are subset

    peptides: set[str]                # Peptides assigned by parsimony (unique + razor)
    unique_peptides: set[str]         # Peptides only in this group (before parsimony)
    razor_peptides: set[str]          # Shared peptides assigned here by parsimony
    all_mapped_peptides: set[str]     # ALL peptides that map to this protein (shared or not)

    @property
    def n_peptides(self) -> int:
        return len(self.peptides)

    @property
    def n_unique_peptides(self) -> int:
        return len(self.unique_peptides)

    @property
    def n_razor_peptides(self) -> int:
        return len(self.razor_peptides)

    @property
    def n_all_mapped_peptides(self) -> int:
        return len(self.all_mapped_peptides)

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame export."""
        return {
            'GroupID': self.group_id,
            'LeadingProtein': self.leading_protein,
            'LeadingName': self.leading_protein_name,
            'MemberProteins': ';'.join(self.member_proteins),
            'SubsumedProteins': ';'.join(self.subsumed_proteins),
            'NPeptides': self.n_peptides,
            'NUniquePeptides': self.n_unique_peptides,
            'NRazorPeptides': self.n_razor_peptides,
            'NAllMappedPeptides': self.n_all_mapped_peptides,
            'UniquePeptides': ';'.join(sorted(self.unique_peptides)),
            'RazorPeptides': ';'.join(sorted(self.razor_peptides)),
            'AllPeptides': ';'.join(sorted(self.peptides)),
        }


def build_peptide_protein_map(
    df: pd.DataFrame,
    peptide_col: str = 'peptide_sequence',
    protein_col: str = 'protein_ids',
    protein_name_col: str = 'protein_names',
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, str]]:
    """Build bidirectional mapping between peptides and proteins.

    Handles semicolon-separated protein lists in the protein column.

    Args:
        df: DataFrame with peptide and protein columns
        peptide_col: Column containing peptide sequences
        protein_col: Column containing protein IDs (may be semicolon-separated)
        protein_name_col: Column containing protein names

    Returns:
        Tuple of:
        - peptide_to_proteins: dict[peptide] -> set of protein IDs
        - protein_to_peptides: dict[protein] -> set of peptides
        - protein_to_name: dict[protein] -> protein name

    """
    peptide_to_proteins: dict[str, set[str]] = defaultdict(set)
    protein_to_peptides: dict[str, set[str]] = defaultdict(set)
    protein_to_name: dict[str, str] = {}

    for _, row in df[[peptide_col, protein_col, protein_name_col]].drop_duplicates().iterrows():
        peptide = row[peptide_col]

        # Handle semicolon-separated protein lists
        proteins_str = str(row[protein_col]) if pd.notna(row[protein_col]) else ''
        names_str = str(row[protein_name_col]) if pd.notna(row[protein_name_col]) else ''

        proteins = [p.strip() for p in proteins_str.split(';') if p.strip()]
        names = [n.strip() for n in names_str.split(';') if n.strip()]

        # Pad names if fewer than proteins
        while len(names) < len(proteins):
            names.append(proteins[len(names)] if len(names) < len(proteins) else '')

        for protein, name in zip(proteins, names):
            peptide_to_proteins[peptide].add(protein)
            protein_to_peptides[protein].add(peptide)
            if protein not in protein_to_name:
                protein_to_name[protein] = name

    return dict(peptide_to_proteins), dict(protein_to_peptides), protein_to_name


def build_peptide_protein_map_from_fasta(
    df: pd.DataFrame,
    fasta_path: str,
    peptide_col: str = 'peptide_sequence',
    handle_il_ambiguity: bool = True,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, str]]:
    """Build peptide-protein mapping from FASTA database.

    This is the RECOMMENDED method for proper parsimony analysis.
    Uses direct substring matching - each detected peptide is searched
    for in all protein sequences.

    Args:
        df: DataFrame with detected peptides
        fasta_path: Path to FASTA database used for search
        peptide_col: Column containing peptide sequences (may have modifications)
        handle_il_ambiguity: Replace I with L for matching (MS cannot distinguish)

    Returns:
        Tuple of:
        - peptide_to_proteins: dict[peptide] -> set of protein IDs
        - protein_to_peptides: dict[protein] -> set of peptides
        - protein_to_name: dict[protein] -> protein name

    """
    from .fasta import (
        build_peptide_protein_map_from_fasta as fasta_build_map,
    )
    from .fasta import (
        build_protein_name_map,
        get_detected_peptides_from_data,
    )

    # Extract detected peptides from data
    detected_peptides = get_detected_peptides_from_data(df, peptide_col)

    logger.info(
        f"Building peptide-protein map from FASTA: {fasta_path}\n"
        f"  Detected peptides: {len(detected_peptides)}"
    )

    # Build mapping from FASTA
    peptide_to_proteins, protein_to_peptides, protein_entries = fasta_build_map(
        fasta_path=fasta_path,
        detected_peptides=detected_peptides,
        handle_il_ambiguity=handle_il_ambiguity,
    )

    # Build protein name map
    protein_to_name = build_protein_name_map(protein_entries)

    # Log statistics
    n_mapped = len(peptide_to_proteins)
    n_total = len(detected_peptides)
    n_unique = sum(1 for prots in peptide_to_proteins.values() if len(prots) == 1)
    n_shared = n_mapped - n_unique

    logger.info(
        f"FASTA mapping complete:\n"
        f"  Peptides matched: {n_mapped}/{n_total} ({100*n_mapped/n_total:.1f}%)\n"
        f"  Proteins identified: {len(protein_to_peptides)}\n"
        f"  Unique peptides: {n_unique}, Shared peptides: {n_shared}"
    )

    return peptide_to_proteins, protein_to_peptides, protein_to_name


def _find_subsumable_proteins(
    protein_to_peptides: dict[str, set[str]]
) -> dict[str, str]:
    """Find proteins whose peptides are a subset of another protein's peptides.

    Returns:
        Dict mapping subsumed protein -> subsuming protein

    """
    subsumed = {}
    proteins = list(protein_to_peptides.keys())

    for i, prot_a in enumerate(proteins):
        peps_a = protein_to_peptides[prot_a]

        for prot_b in proteins[i+1:]:
            peps_b = protein_to_peptides[prot_b]

            if peps_a < peps_b:  # A is proper subset of B
                subsumed[prot_a] = prot_b
                break
            elif peps_b < peps_a:  # B is proper subset of A
                subsumed[prot_b] = prot_a

    return subsumed


def _find_indistinguishable_proteins(
    protein_to_peptides: dict[str, set[str]]
) -> list[set[str]]:
    """Find sets of proteins with identical peptide sets.

    Returns:
        List of sets, where each set contains indistinguishable proteins

    """
    # Group by frozen peptide set
    peptide_set_to_proteins: dict[frozenset, set[str]] = defaultdict(set)

    for protein, peptides in protein_to_peptides.items():
        key = frozenset(peptides)
        peptide_set_to_proteins[key].add(protein)

    # Return groups with more than one protein
    return [prots for prots in peptide_set_to_proteins.values() if len(prots) > 1]


def compute_protein_groups(
    protein_to_peptides: dict[str, set[str]],
    peptide_to_proteins: dict[str, set[str]],
    protein_to_name: dict[str, str],
) -> list[ProteinGroup]:
    """Apply parsimony algorithm to create protein groups.

    Algorithm:
    1. Remove subset (subsumable) proteins
    2. Group indistinguishable proteins
    3. Greedily assign shared peptides to protein with most unique peptides

    Args:
        protein_to_peptides: Mapping from protein to peptides
        peptide_to_proteins: Mapping from peptide to proteins
        protein_to_name: Mapping from protein ID to name

    Returns:
        List of ProteinGroup objects

    """
    logger.info(f"Starting parsimony with {len(protein_to_peptides)} proteins, "
                f"{len(peptide_to_proteins)} peptides")

    # Step 1: Find and remove subsumable proteins
    subsumed_by = _find_subsumable_proteins(protein_to_peptides)
    logger.info(f"Found {len(subsumed_by)} subsumable proteins")

    # Build reverse mapping: subsuming protein -> list of subsumed
    subsuming_to_subsumed: dict[str, list[str]] = defaultdict(list)
    for subsumed, subsuming in subsumed_by.items():
        subsuming_to_subsumed[subsuming].append(subsumed)

    # Remove subsumed proteins from consideration
    active_proteins = {p for p in protein_to_peptides if p not in subsumed_by}

    # Step 2: Find indistinguishable proteins
    # Work with active proteins only
    active_prot_to_peps = {p: protein_to_peptides[p] for p in active_proteins}
    indistinguishable_groups = _find_indistinguishable_proteins(active_prot_to_peps)

    # Map each protein to its indistinguishable group (if any)
    protein_to_indist_group: dict[str, set[str]] = {}
    for group in indistinguishable_groups:
        for prot in group:
            protein_to_indist_group[prot] = group

    logger.info(f"Found {len(indistinguishable_groups)} groups of indistinguishable proteins")

    # Step 3: Determine unique vs shared peptides
    # A peptide is "unique" if it maps to only one protein (or indistinguishable group)
    # after removing subsumed proteins

    peptide_to_active_proteins: dict[str, set[str]] = defaultdict(set)
    for pep, prots in peptide_to_proteins.items():
        active_prots = prots & active_proteins
        peptide_to_active_proteins[pep] = active_prots

    # Collapse indistinguishable groups for uniqueness determination
    def canonical_protein(prot: str) -> str:
        """Return canonical representative for a protein."""
        if prot in protein_to_indist_group:
            return min(protein_to_indist_group[prot])  # Use alphabetically first
        return prot

    peptide_to_canonical: dict[str, set[str]] = {}
    for pep, prots in peptide_to_active_proteins.items():
        canonical_prots = {canonical_protein(p) for p in prots}
        peptide_to_canonical[pep] = canonical_prots

    # Identify unique peptides (map to exactly one canonical protein)
    unique_peptides: set[str] = {
        pep for pep, prots in peptide_to_canonical.items() if len(prots) == 1
    }
    shared_peptides: set[str] = {
        pep for pep, prots in peptide_to_canonical.items() if len(prots) > 1
    }

    logger.info(f"Unique peptides: {len(unique_peptides)}, Shared peptides: {len(shared_peptides)}")

    # Step 4: Build protein groups
    # First, handle proteins/groups with unique peptides
    # Then, assign shared peptides greedily

    # Get canonical proteins
    canonical_proteins = {canonical_protein(p) for p in active_proteins}

    # Map canonical -> all members (including indistinguishable)
    canonical_to_members: dict[str, list[str]] = {}
    for can in canonical_proteins:
        if can in protein_to_indist_group:
            canonical_to_members[can] = sorted(protein_to_indist_group[can])
        else:
            canonical_to_members[can] = [can]

    # Count unique peptides per canonical protein
    canonical_to_unique_peps: dict[str, set[str]] = defaultdict(set)
    for pep in unique_peptides:
        can = list(peptide_to_canonical[pep])[0]  # Only one by definition
        canonical_to_unique_peps[can].add(pep)

    # Greedy assignment of shared peptides
    remaining_shared = shared_peptides.copy()
    canonical_to_razor_peps: dict[str, set[str]] = defaultdict(set)

    while remaining_shared:
        # Find protein(s) that would gain the most remaining peptides
        # Prioritize by: 1) unique peptide count, 2) alphabetical

        best_can = None
        best_score = (-1, '')

        for can in canonical_proteins:
            # How many remaining shared peptides would this protein get?
            can_peps = protein_to_peptides[canonical_to_members[can][0]]  # Use first member
            remaining_for_can = can_peps & remaining_shared

            n_unique = len(canonical_to_unique_peps[can])
            n_remaining = len(remaining_for_can)

            if n_remaining > 0:
                score = (n_unique, -ord(can[0]) if can else 0, n_remaining)
                if score > best_score:
                    best_score = score
                    best_can = can

        if best_can is None:
            break  # No more assignable peptides

        # Assign remaining shared peptides to this protein
        can_peps = protein_to_peptides[canonical_to_members[best_can][0]]
        to_assign = can_peps & remaining_shared
        canonical_to_razor_peps[best_can].update(to_assign)
        remaining_shared -= to_assign

    # Step 5: Create ProteinGroup objects
    protein_groups = []

    for i, can in enumerate(sorted(canonical_proteins)):
        members = canonical_to_members[can]

        # Get subsumed proteins for any member
        subsumed_list = []
        for member in members:
            subsumed_list.extend(subsuming_to_subsumed.get(member, []))

        # Peptides assigned by parsimony
        unique_peps = canonical_to_unique_peps.get(can, set())
        razor_peps = canonical_to_razor_peps.get(can, set())
        all_peps = unique_peps | razor_peps

        # ALL peptides that originally mapped to any member protein
        # (before razor assignment, including shared peptides)
        all_mapped = set()
        for member in members:
            all_mapped.update(protein_to_peptides.get(member, set()))
        # Also include peptides from subsumed proteins
        for subsumed in subsumed_list:
            all_mapped.update(protein_to_peptides.get(subsumed, set()))

        # Only create group if it has peptides
        if not all_peps:
            continue

        # Leading protein is the canonical (or first member if indistinguishable)
        leading = members[0]

        group = ProteinGroup(
            group_id=f"PG{i+1:04d}",
            leading_protein=leading,
            leading_protein_name=protein_to_name.get(leading, leading),
            member_proteins=members,
            subsumed_proteins=subsumed_list,
            peptides=all_peps,
            unique_peptides=unique_peps,
            razor_peptides=razor_peps,
            all_mapped_peptides=all_mapped,
        )
        protein_groups.append(group)

    logger.info(f"Created {len(protein_groups)} protein groups")

    return protein_groups


def export_protein_groups(
    groups: list[ProteinGroup],
    output_path: str,
) -> None:
    """Export protein groups to TSV file.

    Args:
        groups: List of ProteinGroup objects
        output_path: Path to output TSV file

    """
    rows = [g.to_dict() for g in groups]
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep='\t', index=False)
    logger.info(f"Exported {len(groups)} protein groups to {output_path}")


def annotate_peptides_with_groups(
    df: pd.DataFrame,
    groups: list[ProteinGroup],
    peptide_col: str = 'peptide_sequence',
) -> pd.DataFrame:
    """Add protein group assignment to peptide data.

    Args:
        df: DataFrame with peptide data
        groups: List of ProteinGroup objects from parsimony
        peptide_col: Column containing peptide sequences

    Returns:
        DataFrame with added columns:
        - protein_group_id: assigned group ID
        - peptide_type: 'unique' or 'razor'

    """
    # Build peptide -> group mapping
    peptide_to_group: dict[str, str] = {}
    peptide_to_type: dict[str, str] = {}

    for group in groups:
        for pep in group.unique_peptides:
            peptide_to_group[pep] = group.group_id
            peptide_to_type[pep] = 'unique'
        for pep in group.razor_peptides:
            peptide_to_group[pep] = group.group_id
            peptide_to_type[pep] = 'razor'

    # Add columns
    df = df.copy()
    df['protein_group_id'] = df[peptide_col].map(peptide_to_group)
    df['peptide_type'] = df[peptide_col].map(peptide_to_type)

    # Log unmapped peptides
    unmapped = df['protein_group_id'].isna().sum()
    if unmapped > 0:
        logger.warning(f"{unmapped} rows have unmapped peptides")

    return df


def compute_peptide_weights(
    groups: list[ProteinGroup],
    method: str = 'unique_count'
) -> dict[str, dict[str, float]]:
    """Compute weights for distributed peptide assignment.

    For shared peptides that map to multiple groups, compute weight
    based on evidence for each group.

    Args:
        groups: List of ProteinGroup objects
        method: Weighting method
            - 'uniform': Equal weight to all groups
            - 'unique_count': Weight by number of unique peptides

    Returns:
        Dict[peptide] -> Dict[group_id] -> weight

    """
    # Find shared peptides (appear in multiple groups)
    peptide_groups: dict[str, list[ProteinGroup]] = defaultdict(list)
    for group in groups:
        for pep in group.peptides:
            peptide_groups[pep].append(group)

    shared = {pep: gps for pep, gps in peptide_groups.items() if len(gps) > 1}

    # Compute weights
    weights: dict[str, dict[str, float]] = {}

    for pep, gps in shared.items():
        if method == 'uniform':
            w = 1.0 / len(gps)
            weights[pep] = {g.group_id: w for g in gps}

        elif method == 'unique_count':
            total_unique = sum(g.n_unique_peptides for g in gps)
            if total_unique == 0:
                # Fall back to uniform
                w = 1.0 / len(gps)
                weights[pep] = {g.group_id: w for g in gps}
            else:
                weights[pep] = {
                    g.group_id: g.n_unique_peptides / total_unique for g in gps
                }

    # Non-shared peptides get weight 1.0
    for group in groups:
        for pep in group.peptides:
            if pep not in weights:
                weights[pep] = {group.group_id: 1.0}

    return weights
