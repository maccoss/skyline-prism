"""FASTA parsing and in-silico digestion for protein parsimony.

This module provides:
1. FASTA file parsing (UniProt/NCBI formats)
2. In-silico protein digestion with configurable enzymes
3. Peptide-to-protein mapping from FASTA
4. Modified sequence handling (stripping modifications)
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Enzyme Definitions
# =============================================================================

# Enzyme cleavage rules: (regex_pattern, cleave_after)
# cleave_after=True means cleave C-terminal to match; False means N-terminal
ENZYME_RULES: dict[str, tuple[str, bool]] = {
    # Trypsin: cleave after K or R, unless followed by P
    "trypsin": (r"[KR](?!P)", True),
    # Trypsin/P: cleave after K or R, even if followed by P
    "trypsin/p": (r"[KR]", True),
    # Lys-C: cleave after K
    "lysc": (r"K", True),
    # Lys-N: cleave before K
    "lysn": (r"K", False),
    # Arg-C: cleave after R
    "argc": (r"R", True),
    # Asp-N: cleave before D
    "aspn": (r"D", False),
    # Glu-C (V8): cleave after E (and sometimes D)
    "gluc": (r"E", True),
    # Chymotrypsin: cleave after F, Y, W, L (not followed by P)
    "chymotrypsin": (r"[FYWL](?!P)", True),
    # No enzyme (non-specific) - returns all possible peptides
    # Handled separately
    "nonspecific": (r".", True),
}


@dataclass
class ProteinEntry:
    """A protein entry from a FASTA file."""

    accession: str  # Primary accession (e.g., P04406)
    name: str  # Entry name (e.g., G3P_HUMAN)
    gene_name: str | None  # Gene name if available
    description: str  # Full description line
    sequence: str  # Amino acid sequence
    db_type: str | None = None  # Database type (e.g., 'sp', 'tr' for UniProt)

    @property
    def length(self) -> int:
        """Return the length of the protein sequence."""
        return len(self.sequence)

    @property
    def uniprot_id(self) -> str:
        """Return full UniProt identifier (e.g., sp|P04406|G3P_HUMAN) if available."""
        if self.db_type and self.db_type in ["sp", "tr"]:
            return f"{self.db_type}|{self.accession}|{self.name}"
        # Fallback to accession for non-UniProt entries
        return self.accession


# =============================================================================
# FASTA Parsing
# =============================================================================


def parse_fasta(fasta_path: str | Path) -> dict[str, ProteinEntry]:
    """Parse a FASTA file and return protein entries.

    Supports UniProt and NCBI formats. Automatically detects format.

    UniProt format:
        >sp|P04406|G3P_HUMAN Glyceraldehyde-3-phosphate dehydrogenase OS=Homo sapiens
        >tr|A0A123|A0A123_HUMAN Some protein OS=Homo sapiens

    NCBI format:
        >NP_001256799.1 glyceraldehyde-3-phosphate dehydrogenase [Homo sapiens]
        >gi|123456|ref|NP_001256799.1| protein name

    Args:
        fasta_path: Path to FASTA file

    Returns:
        Dict mapping accession -> ProteinEntry

    """
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    proteins: dict[str, ProteinEntry] = {}

    current_header = None
    current_sequence: list[str] = []

    # Detect if gzipped
    open_func = open
    if fasta_path.suffix == ".gz":
        import gzip

        open_func = gzip.open

    with open_func(fasta_path, "rt") as f:
        for line in f:
            line = line.strip()

            if line.startswith(">"):
                # Save previous entry
                if current_header is not None:
                    entry = _parse_header(current_header, "".join(current_sequence))
                    if entry.accession not in proteins:
                        proteins[entry.accession] = entry
                    else:
                        logger.debug(f"Duplicate accession: {entry.accession}")

                current_header = line[1:]  # Remove '>'
                current_sequence = []
            elif line and current_header is not None:
                # Sequence line - remove any whitespace
                current_sequence.append(line.replace(" ", ""))

        # Don't forget the last entry
        if current_header is not None:
            entry = _parse_header(current_header, "".join(current_sequence))
            if entry.accession not in proteins:
                proteins[entry.accession] = entry

    logger.info(f"Parsed {len(proteins)} proteins from {fasta_path.name}")
    return proteins


def _parse_header(header: str, sequence: str) -> ProteinEntry:
    """Parse a FASTA header line to extract accession and metadata.

    Handles UniProt and NCBI formats.

    For UniProt entries like::

        >sp|Q86YV5|PRAG1_HUMAN Inactive tyrosine-protein kinase PRAG1 OS=...

    Extracts:
        - accession: Q86YV5
        - name: PRAG1_HUMAN
        - gene_name: PRAG1
        - description: Inactive tyrosine-protein kinase PRAG1 (clean)
    """
    accession = ""
    name = ""
    gene_name = None
    description = header

    # Try UniProt format: >sp|P04406|G3P_HUMAN Description OS=... GN=...
    uniprot_match = re.match(r"^([sptr]{2})\|([^|]+)\|([^\s]+)\s*(.*)", header)
    if uniprot_match:
        db_type = uniprot_match.group(1)  # sp or tr
        accession = uniprot_match.group(2)
        name = uniprot_match.group(3)
        full_description = uniprot_match.group(4) or name

        # Extract clean description (text before OS=, OX=, GN=, PE=, SV= tags)
        # These tags mark the start of UniProt metadata fields
        desc_match = re.match(r"^(.*?)\s*(?:OS=|OX=|GN=|PE=|SV=|$)", full_description)
        if desc_match and desc_match.group(1).strip():
            description = desc_match.group(1).strip()
        else:
            description = name  # Fallback to entry name if no description text

        # Extract gene name from GN= field
        gn_match = re.search(r"GN=(\S+)", header)
        if gn_match:
            gene_name = gn_match.group(1)

        return ProteinEntry(
            accession=accession,
            name=name,
            gene_name=gene_name,
            description=description,
            sequence=sequence,
            db_type=db_type,
        )

    # Try NCBI format: >NP_001256799.1 description [species]
    # or >gi|123|ref|NP_001256799.1| description
    ncbi_gi_match = re.match(r"^gi\|(\d+)\|[^|]+\|([^|]+)\|?\s*(.*)", header)
    if ncbi_gi_match:
        accession = ncbi_gi_match.group(2).strip("|")
        description = ncbi_gi_match.group(3) or accession
        return ProteinEntry(
            accession=accession,
            name=accession,
            gene_name=None,
            description=description,
            sequence=sequence,
        )

    ncbi_match = re.match(r"^(\S+)\s*(.*)", header)
    if ncbi_match:
        accession = ncbi_match.group(1)
        description = ncbi_match.group(2) or accession
        return ProteinEntry(
            accession=accession,
            name=accession,
            gene_name=None,
            description=description,
            sequence=sequence,
        )

    # Fallback: use first word as accession
    parts = header.split()
    accession = parts[0] if parts else "UNKNOWN"

    return ProteinEntry(
        accession=accession,
        name=accession,
        gene_name=None,
        description=header,
        sequence=sequence,
    )


# =============================================================================
# In-Silico Digestion
# =============================================================================


def digest_protein(
    sequence: str,
    enzyme: str = "trypsin",
    missed_cleavages: int = 2,
    min_length: int = 6,
    max_length: int = 30,
) -> set[str]:
    """Digest a protein sequence into peptides.

    Args:
        sequence: Amino acid sequence
        enzyme: Enzyme name (see ENZYME_RULES)
        missed_cleavages: Maximum number of missed cleavages allowed
        min_length: Minimum peptide length
        max_length: Maximum peptide length

    Returns:
        Set of peptide sequences

    """
    enzyme = enzyme.lower()

    if enzyme == "nonspecific":
        return _digest_nonspecific(sequence, min_length, max_length)

    if enzyme not in ENZYME_RULES:
        raise ValueError(f"Unknown enzyme: {enzyme}. Available: {', '.join(ENZYME_RULES.keys())}")

    pattern, cleave_after = ENZYME_RULES[enzyme]

    # Find all cleavage sites
    cleavage_sites = []
    for match in re.finditer(pattern, sequence):
        if cleave_after:
            cleavage_sites.append(match.end())
        else:
            cleavage_sites.append(match.start())

    # Add start and end positions
    sites = [0] + sorted(set(cleavage_sites)) + [len(sequence)]

    # Generate peptides with up to `missed_cleavages` missed cleavages
    peptides: set[str] = set()

    for i in range(len(sites) - 1):
        for j in range(i + 1, min(i + 2 + missed_cleavages, len(sites))):
            peptide = sequence[sites[i] : sites[j]]

            if min_length <= len(peptide) <= max_length:
                peptides.add(peptide)

    return peptides


def _digest_nonspecific(
    sequence: str,
    min_length: int,
    max_length: int,
) -> set[str]:
    """Generate all possible peptides (non-specific digestion).

    Warning: This can generate a very large number of peptides!
    """
    peptides: set[str] = set()
    seq_len = len(sequence)

    for start in range(seq_len):
        for length in range(min_length, min(max_length + 1, seq_len - start + 1)):
            peptides.add(sequence[start : start + length])

    return peptides


def digest_fasta(
    proteins: dict[str, ProteinEntry],
    enzyme: str = "trypsin",
    missed_cleavages: int = 2,
    min_length: int = 6,
    max_length: int = 30,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Digest all proteins in a FASTA and build peptide-protein mappings.

    Args:
        proteins: Dict of accession -> ProteinEntry from parse_fasta()
        enzyme: Enzyme name
        missed_cleavages: Maximum missed cleavages
        min_length: Minimum peptide length
        max_length: Maximum peptide length

    Returns:
        Tuple of:
        - protein_to_peptides: dict[accession] -> set of peptides
        - peptide_to_proteins: dict[peptide] -> set of accessions

    """
    protein_to_peptides: dict[str, set[str]] = {}
    peptide_to_proteins: dict[str, set[str]] = {}

    for accession, entry in proteins.items():
        peptides = digest_protein(
            entry.sequence,
            enzyme=enzyme,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            max_length=max_length,
        )

        protein_to_peptides[accession] = peptides

        for peptide in peptides:
            if peptide not in peptide_to_proteins:
                peptide_to_proteins[peptide] = set()
            peptide_to_proteins[peptide].add(accession)

    n_unique = sum(1 for p in peptide_to_proteins if len(peptide_to_proteins[p]) == 1)
    n_shared = len(peptide_to_proteins) - n_unique

    logger.info(
        f"Digested {len(proteins)} proteins -> {len(peptide_to_proteins)} peptides "
        f"({n_unique} unique, {n_shared} shared)"
    )

    return protein_to_peptides, peptide_to_proteins


def get_theoretical_peptide_counts(
    fasta_path: str | Path,
    protein_accessions: set[str] | None = None,
    enzyme: str = "trypsin",
    missed_cleavages: int = 0,
    min_length: int = 6,
    max_length: int = 30,
) -> dict[str, int]:
    """Get theoretical peptide counts per protein for iBAQ calculation.

    For iBAQ, we count the number of theoretical peptides that could be
    detected for each protein, typically using strict digestion (0 missed
    cleavages) to avoid counting overlapping peptides multiple times.

    Args:
        fasta_path: Path to FASTA file
        protein_accessions: If provided, only count for these proteins.
            If None, counts all proteins in the FASTA.
        enzyme: Digestion enzyme (default: trypsin)
        missed_cleavages: Max missed cleavages (default: 0 for iBAQ)
        min_length: Minimum peptide length
        max_length: Maximum peptide length

    Returns:
        Dict of protein_accession -> theoretical peptide count

    Example:
        >>> counts = get_theoretical_peptide_counts(
        ...     "/path/to/database.fasta",
        ...     enzyme="trypsin",
        ...     missed_cleavages=0,  # Strict for iBAQ
        ... )
        >>> print(counts['P04406'])  # GAPDH has ~20 theoretical tryptic peptides

    """
    proteins = parse_fasta(fasta_path)

    counts: dict[str, int] = {}

    for accession, entry in proteins.items():
        # Skip if we have a filter and this protein isn't in it
        if protein_accessions is not None and accession not in protein_accessions:
            continue

        peptides = digest_protein(
            entry.sequence,
            enzyme=enzyme,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            max_length=max_length,
        )
        counts[accession] = len(peptides)

    logger.info(
        f"Calculated theoretical peptide counts for {len(counts)} proteins "
        f"(enzyme={enzyme}, missed_cleavages={missed_cleavages})"
    )

    return counts


# =============================================================================
# Enzyme-Aware Terminus Check (for FASTA peptide-protein mapping)
# =============================================================================

# Valid values for the terminus specificity of a FASTA membership match.
ENZYME_SPECIFICITIES = ("full", "semi", "none")


def normalize_enzyme(enzyme: str) -> str:
    r"""Canonicalize an enzyme name for the terminus check.

    Trims and lowercases, and accepts the ``trypsinp`` and ``trypsin\p``
    spellings as aliases for ``trypsin/p``. Mirrors the C#
    ``FastaParser.NormalizeEnzyme`` so both engines accept the same config values.
    """
    e = (enzyme or "").strip().lower()
    if e in ("trypsin\\p", "trypsinp"):
        return "trypsin/p"
    return e


def cleavage_boundary_set(sequence: str, enzyme: str) -> set[int]:
    """Positions in ``sequence`` where ``enzyme`` creates a peptide boundary.

    A boundary index ``b`` marks a bond the enzyme cleaves: a peptide may START
    at ``b`` (the residue before ``b`` is a cleavage site) or END at ``b`` (the
    residue at ``b - 1`` is a cleavage site). The protein termini (0 and
    ``len(sequence)``) are always boundaries, and index 1 is added when the
    protein starts with ``M`` to allow for initiator-methionine excision (a
    common ragged N-terminus that would otherwise drop legitimate N-terminal
    peptides).

    Boundaries are computed on the ORIGINAL (not I/L-normalized) sequence so
    enzymes whose specificity distinguishes I from L (e.g. chymotrypsin cleaves
    after L but not I) stay correct. Index positions are unchanged by I/L
    normalization, so the returned set still aligns with the normalized sequence
    used for substring search.

    Args:
        sequence: Amino acid sequence (uppercase, not I/L-normalized)
        enzyme: Enzyme name (see ENZYME_RULES) or "nonspecific"

    Returns:
        Set of boundary indices in [0, len(sequence)]

    """
    n = len(sequence)
    boundaries = {0, n}
    if n and sequence[0] == "M":
        boundaries.add(1)  # initiator-methionine excision

    enzyme = normalize_enzyme(enzyme)
    if enzyme == "nonspecific":
        boundaries.update(range(n + 1))
        return boundaries

    if enzyme not in ENZYME_RULES:
        raise ValueError(f"Unknown enzyme: {enzyme}. Available: {', '.join(ENZYME_RULES)}")

    pattern, cleave_after = ENZYME_RULES[enzyme]
    for match in re.finditer(pattern, sequence):
        boundaries.add(match.end() if cleave_after else match.start())

    return boundaries


def terminus_is_enzymatic(
    start: int,
    end: int,
    boundaries: set[int],
    specificity: str,
) -> bool:
    """Whether a peptide occupying ``sequence[start:end]`` is an enzymatic product.

    Args:
        start: Match start index (inclusive)
        end: Match end index (exclusive)
        boundaries: Cleavage boundaries from :func:`cleavage_boundary_set`
        specificity: "full" (both termini must be cleavage-consistent),
            "semi" (at least one terminus), or "none" (no check - always True)

    Returns:
        True if the match is consistent with the enzyme at the given specificity

    """
    if specificity == "none":
        return True
    n_term_ok = start in boundaries
    if specificity == "semi":
        return n_term_ok or (end in boundaries)
    # "full" (default): both termini must be cleavage-consistent.
    return n_term_ok and (end in boundaries)


# =============================================================================
# Modified Sequence Handling
# =============================================================================

# Common modification patterns in peptide sequences
# These cover Skyline, MaxQuant, Spectronaut, DIA-NN, etc.
MODIFICATION_PATTERNS = [
    # Skyline/Panorama: C[+57.021464] or M[+15.994915]
    r"\[[\+\-]?\d+\.?\d*\]",
    # UniMod format: (UniMod:4) or [UniMod:35]
    r"\(UniMod:\d+\)",
    r"\[UniMod:\d+\]",
    # MaxQuant: (ox), (ac), (ph), etc.
    r"\([a-zA-Z]{2,4}\)",
    # ProForma: [Oxidation], [Carbamidomethyl]
    r"\[[A-Za-z]+\]",
    # Numeric in parentheses: M(16) C(57)
    r"\(\d+\.?\d*\)",
    # Plus/minus mass in parentheses: (+15.99) (-18.01)
    r"\([\+\-]\d+\.?\d*\)",
]

# Compiled regex for stripping modifications
_MOD_PATTERN = re.compile("|".join(MODIFICATION_PATTERNS))

# Pattern for terminal modifications like n[+42.011] or c[-17.03]
_TERMINAL_MOD_PATTERN = re.compile(r"^[nc]\[[\+\-]?\d+\.?\d*\]|[nc]\[[\+\-]?\d+\.?\d*\]$")


def strip_modifications(modified_sequence: str) -> str:
    """Remove modifications from a peptide sequence.

    Handles various modification formats from different software:
    - Skyline: C[+57.021] -> C
    - MaxQuant: M(ox) -> M
    - UniMod: M[UniMod:35] -> M
    - ProForma: C[Carbamidomethyl] -> C

    Also handles:
    - Terminal modifications: n[+42]PEPTIDEK -> PEPTIDEK
    - I/L ambiguity is NOT resolved here (handled in matching)

    Args:
        modified_sequence: Peptide sequence with modifications

    Returns:
        Unmodified amino acid sequence

    """
    # Remove terminal modifications first
    sequence = _TERMINAL_MOD_PATTERN.sub("", modified_sequence)

    # Remove all other modifications
    sequence = _MOD_PATTERN.sub("", sequence)

    # Remove any remaining non-amino-acid characters
    # Keep only uppercase letters (standard amino acids)
    sequence = "".join(c for c in sequence if c.isupper())

    return sequence


def normalize_for_matching(
    sequence: str,
    handle_il_ambiguity: bool = True,
) -> str:
    """Normalize a peptide sequence for FASTA matching.

    1. Strip modifications
    2. Convert to uppercase
    3. Optionally handle I/L ambiguity (replace I with L)

    Args:
        sequence: Peptide sequence (may have modifications)
        handle_il_ambiguity: If True, replace I with L (MS cannot distinguish)

    Returns:
        Normalized sequence for matching

    """
    normalized = strip_modifications(sequence).upper()

    if handle_il_ambiguity:
        normalized = normalized.replace("I", "L")

    return normalized


# =============================================================================
# Building Peptide-Protein Map from FASTA for Detected Peptides
# =============================================================================


def build_peptide_protein_map_from_fasta(
    fasta_path: str | Path,
    detected_peptides: set[str],
    handle_il_ambiguity: bool = True,
    enzyme: str = "trypsin",
    enzyme_specificity: str = "full",
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, ProteinEntry]]:
    """Build peptide-protein mapping by searching for peptides in protein sequences.

    This is the main entry point for parsimony analysis. For each detected
    peptide it finds every protein that can enzymatically produce it: the
    peptide must occur as a substring (after stripping modifications and
    handling I/L ambiguity) AND, unless the enzyme check is disabled, at least
    one occurrence must have termini consistent with the digestion enzyme.

    The enzyme check removes "phantom" assignments where a peptide is a
    substring of a protein it cannot come from because the required flanking
    cleavage site is absent (e.g. ``AKEGVVAAAEK`` is a substring of beta-
    synuclein but is preceded there by ``M``, not ``K/R``, so trypsin cannot
    liberate it; it is proteotypic to alpha-synuclein). Substring containment
    is necessary but not sufficient for a peptide to originate from a protein.

    Args:
        fasta_path: Path to FASTA file
        detected_peptides: Set of detected peptide sequences (may have modifications)
        handle_il_ambiguity: Replace I with L for matching (MS cannot distinguish)
        enzyme: Digestion enzyme for the terminus check (see ENZYME_RULES).
            Default "trypsin" (cleave after K/R, but NOT before P). Use
            "trypsin/p" for searches that cleave K/R even before P (e.g. the
            common DIA-NN default). The Skyline external tool overrides this
            from the document's digestion settings.
        enzyme_specificity: Terminus requirement (see ENZYME_SPECIFICITIES):
            - "full" (default): both peptide termini must be cleavage-consistent
              (or a protein terminus). Removes phantom paralog assignments.
            - "semi": at least one terminus must be cleavage-consistent.
            - "none": no enzyme check (legacy pure-substring behavior).

    Returns:
        Tuple of:
        - peptide_to_proteins: dict[detected_peptide] -> set of protein accessions
        - protein_to_peptides: dict[accession] -> set of detected peptides
        - proteins: dict[accession] -> ProteinEntry (for name lookup)

    """
    specificity = enzyme_specificity.lower()
    if specificity not in ENZYME_SPECIFICITIES:
        raise ValueError(
            f"Unknown enzyme_specificity: {enzyme_specificity}. "
            f"Available: {', '.join(ENZYME_SPECIFICITIES)}"
        )

    logger.info(f"Building peptide-protein map from {fasta_path}")
    logger.info(f"Detected peptides: {len(detected_peptides)}")
    if specificity == "none":
        logger.info("  Enzyme terminus check disabled (enzyme_specificity=none)")
    else:
        logger.info(f"  Enzyme terminus check: enzyme={enzyme}, specificity={specificity}")

    # Parse FASTA
    proteins = parse_fasta(fasta_path)

    # Normalize protein sequences for substring matching (handle I/L ambiguity) and, unless
    # the enzyme check is disabled, precompute each protein's cleavage boundaries from the
    # ORIGINAL sequence (indices align with the normalized sequence).
    protein_sequences: dict[str, str] = {}
    protein_boundaries: dict[str, set[int]] = {}
    for accession, entry in proteins.items():
        original = entry.sequence.upper()
        normalized_seq = original.replace("I", "L") if handle_il_ambiguity else original
        protein_sequences[accession] = normalized_seq
        if specificity != "none":
            protein_boundaries[accession] = cleavage_boundary_set(original, enzyme)

    # Map detected peptides to proteins by (enzyme-aware) substring search
    peptide_to_proteins: dict[str, set[str]] = {}
    protein_to_detected: dict[str, set[str]] = {}
    unmatched: list[str] = []

    for detected_pep in detected_peptides:
        # Normalize detected peptide for matching
        normalized = normalize_for_matching(detected_pep, handle_il_ambiguity)
        if not normalized:
            unmatched.append(detected_pep)
            continue

        pep_len = len(normalized)

        # Search for this peptide in all protein sequences
        matched_proteins: set[str] = set()
        for accession, seq in protein_sequences.items():
            pos = seq.find(normalized)
            if pos == -1:
                continue
            if specificity == "none":
                matched_proteins.add(accession)
                continue
            # Accept the protein if ANY occurrence has enzyme-consistent termini.
            boundaries = protein_boundaries[accession]
            while pos != -1:
                if terminus_is_enzymatic(pos, pos + pep_len, boundaries, specificity):
                    matched_proteins.add(accession)
                    break
                pos = seq.find(normalized, pos + 1)

        if matched_proteins:
            peptide_to_proteins[detected_pep] = matched_proteins

            # Update protein_to_detected
            for prot in matched_proteins:
                if prot not in protein_to_detected:
                    protein_to_detected[prot] = set()
                protein_to_detected[prot].add(detected_pep)
        else:
            unmatched.append(detected_pep)

    if unmatched:
        logger.warning(
            f"{len(unmatched)} detected peptides could not be matched to FASTA "
            f"(first 5: {unmatched[:5]})"
        )

    n_matched = len(detected_peptides) - len(unmatched)
    n_unique = sum(1 for p in peptide_to_proteins if len(peptide_to_proteins[p]) == 1)
    n_shared = n_matched - n_unique

    logger.info(
        f"Matched {n_matched}/{len(detected_peptides)} peptides to "
        f"{len(protein_to_detected)} proteins "
        f"({n_unique} unique, {n_shared} shared)"
    )

    return peptide_to_proteins, protein_to_detected, proteins


def get_detected_peptides_from_data(
    data: pd.DataFrame,
    peptide_col: str = "peptide_modified",
) -> set[str]:
    """Extract unique peptide sequences from a data frame.

    Args:
        data: DataFrame with peptide data
        peptide_col: Column containing peptide sequences

    Returns:
        Set of unique peptide sequences

    """
    if peptide_col not in data.columns:
        raise ValueError(f"Column '{peptide_col}' not found in data")

    return set(data[peptide_col].dropna().unique())


# =============================================================================
# Protein Name Lookup
# =============================================================================


def build_protein_name_map(
    proteins: dict[str, ProteinEntry],
) -> dict[str, str]:
    """Build a mapping from protein accession to display name.

    Prefers gene name if available, otherwise uses entry name.

    Args:
        proteins: Dict of accession -> ProteinEntry

    Returns:
        Dict of accession -> display name

    """
    name_map: dict[str, str] = {}

    for accession, entry in proteins.items():
        if entry.gene_name:
            name_map[accession] = entry.gene_name
        elif entry.name and entry.name != accession:
            name_map[accession] = entry.name
        else:
            # Extract from description if possible
            # Try to get first word that looks like a gene name
            desc_parts = entry.description.split()
            if desc_parts:
                name_map[accession] = desc_parts[0]
            else:
                name_map[accession] = accession

    return name_map
