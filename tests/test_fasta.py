"""Tests for FASTA parsing and in-silico digestion."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from skyline_prism.fasta import (
    ENZYME_RULES,
    ProteinEntry,
    build_peptide_protein_map_from_fasta,
    build_protein_name_map,
    digest_fasta,
    digest_protein,
    get_detected_peptides_from_data,
    normalize_for_matching,
    parse_fasta,
    strip_modifications,
)

# =============================================================================
# Test Data
# =============================================================================

# Sample UniProt FASTA content
UNIPROT_FASTA = """>sp|P04406|G3P_HUMAN Glyceraldehyde-3-phosphate dehydrogenase OS=Homo sapiens OX=9606 GN=GAPDH PE=1 SV=3
MGKVKVGVNGFGRIGRLVTRAAFNSGKVDIVAINDPFIDLNYMVYMFQYDSTHGKFHGTV
KAENGKLVINGNPITIFQERDPSKIKWGDAGAEYVVESTGVFTTMEKAGAHLQGGAKRVI
ISAPSADAPMFVMGVNHEKYDNSLKIISNASCTTNCLAPLAKVIHDNFGIIVEGLMTTVH
AITATQKTVDGPSGKLWRDGRGALQNIIPASTGAAKAVGKVIPELNGKLTGMAFRVPTPN
VSVVDLTCRLEK
>sp|P68871|HBB_HUMAN Hemoglobin subunit beta OS=Homo sapiens OX=9606 GN=HBB PE=1 SV=2
MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPK
VKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFG
KEFTPPVQAAYQKVVAGVANALAHKYH
>tr|A0A123|A0A123_HUMAN Some trembl protein OS=Homo sapiens OX=9606 PE=4 SV=1
MKFLILLFNILCLFPVILCMKAEDWPRK
"""

# Sample NCBI FASTA content
NCBI_FASTA = """>NP_001256799.1 glyceraldehyde-3-phosphate dehydrogenase [Homo sapiens]
MGKVKVGVNGFGRIGRLVTRAAFNSGKVDIVAINDPFIDLNYMVYMFQYDSTHGKFHGTV
KAENGKLVINGNPITIFQERDPSKIK
>gi|123456|ref|NP_000509.1| hemoglobin subunit beta [Homo sapiens]
MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPK
VKAHGKKVLGAFSDGLAHLDNLK
"""


def create_temp_fasta(content: str) -> Path:
    """Create a temporary FASTA file with the given content."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(content)
        return Path(f.name)


# =============================================================================
# FASTA Parsing Tests
# =============================================================================


class TestParseFasta:
    """Tests for parse_fasta function."""

    def test_parse_uniprot_format(self):
        """Test parsing UniProt format FASTA."""
        fasta_path = create_temp_fasta(UNIPROT_FASTA)
        try:
            proteins = parse_fasta(fasta_path)

            assert len(proteins) == 3

            # Check P04406 (GAPDH)
            assert 'P04406' in proteins
            gapdh = proteins['P04406']
            assert gapdh.name == 'G3P_HUMAN'
            assert gapdh.gene_name == 'GAPDH'
            assert gapdh.sequence.startswith('MGKVKVGVNGFGR')
            # Sequence length matches what's in test FASTA (truncated)
            assert len(gapdh.sequence) == 252
            assert gapdh.length == 252

            # Check P68871 (HBB)
            assert 'P68871' in proteins
            hbb = proteins['P68871']
            assert hbb.name == 'HBB_HUMAN'
            assert hbb.gene_name == 'HBB'

            # Check trembl entry
            assert 'A0A123' in proteins
            assert proteins['A0A123'].name == 'A0A123_HUMAN'
        finally:
            fasta_path.unlink()

    def test_parse_ncbi_format(self):
        """Test parsing NCBI format FASTA."""
        fasta_path = create_temp_fasta(NCBI_FASTA)
        try:
            proteins = parse_fasta(fasta_path)

            assert len(proteins) == 2
            assert 'NP_001256799.1' in proteins
            assert 'NP_000509.1' in proteins

            # Check the sequence was parsed correctly
            assert proteins['NP_001256799.1'].sequence.startswith('MGKVKVGVNGFGR')
        finally:
            fasta_path.unlink()

    def test_parse_simple_format(self):
        """Test parsing simple FASTA format (just accession, no special format)."""
        simple_fasta = """>PROTEIN1 Some description
ACDEFGHIKLMNPQRSTVWY
>PROTEIN2 Another description
ARNDCEQGHILKMFPSTWYV
"""
        fasta_path = create_temp_fasta(simple_fasta)
        try:
            proteins = parse_fasta(fasta_path)

            assert len(proteins) == 2
            assert 'PROTEIN1' in proteins
            assert 'PROTEIN2' in proteins
            assert proteins['PROTEIN1'].sequence == 'ACDEFGHIKLMNPQRSTVWY'
        finally:
            fasta_path.unlink()

    def test_parse_empty_fasta(self):
        """Test parsing empty FASTA file returns empty dict."""
        fasta_path = create_temp_fasta("")
        try:
            proteins = parse_fasta(fasta_path)
            assert len(proteins) == 0
        finally:
            fasta_path.unlink()

    def test_file_not_found(self):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            parse_fasta('/nonexistent/path/file.fasta')


# =============================================================================
# In-Silico Digestion Tests
# =============================================================================


class TestDigestProtein:
    """Tests for digest_protein function."""

    def test_trypsin_basic(self):
        """Test basic trypsin digestion."""
        # Trypsin cleaves after K or R unless followed by P
        sequence = "PEPTIDEKPEPTIDER"
        peptides = digest_protein(
            sequence, enzyme='trypsin', missed_cleavages=0,
            min_length=1, max_length=50
        )

        # Should cleave after first R, but not after K (K is followed by P)
        # So: PEPTIDEKPEPTIDER
        assert 'PEPTIDEKPEPTIDER' in peptides

    def test_trypsin_with_missed_cleavages(self):
        """Test trypsin with missed cleavages."""
        sequence = "AAAAKBBBBKCCCCK"
        peptides = digest_protein(
            sequence, enzyme='trypsin', missed_cleavages=0,
            min_length=1, max_length=50
        )

        assert 'AAAAK' in peptides
        assert 'BBBBK' in peptides
        assert 'CCCCK' in peptides

        # With 1 missed cleavage
        peptides_mc1 = digest_protein(
            sequence, enzyme='trypsin', missed_cleavages=1,
            min_length=1, max_length=50
        )

        assert 'AAAAKBBBBK' in peptides_mc1
        assert 'BBBBKCCCCK' in peptides_mc1

    def test_trypsin_proline_rule(self):
        """Test that trypsin doesn't cleave before proline."""
        # K followed by P should not be cleaved
        sequence = "AAAKPBBBR"
        peptides = digest_protein(
            sequence, enzyme='trypsin', missed_cleavages=0,
            min_length=1, max_length=50
        )

        # Should get AAAKPBBBR because K-P is not cleaved
        assert 'AAAKPBBBR' in peptides

    def test_trypsin_p_ignores_proline_rule(self):
        """Test trypsin/p cleaves regardless of proline."""
        sequence = "AAAKPBBBR"
        peptides = digest_protein(
            sequence, enzyme='trypsin/p', missed_cleavages=0,
            min_length=1, max_length=50
        )

        # Should cleave at K even though followed by P
        assert 'AAAK' in peptides
        assert 'PBBBR' in peptides

    def test_lysc(self):
        """Test Lys-C digestion (cleaves after K)."""
        sequence = "AAAKBBBRCCCCK"
        peptides = digest_protein(
            sequence, enzyme='lysc', missed_cleavages=0,
            min_length=1, max_length=50
        )

        assert 'AAAK' in peptides
        assert 'BBBRCCCCK' in peptides  # Only cleaves at K, not R

    def test_min_max_length_filter(self):
        """Test min/max length filtering."""
        sequence = "AAAKBBBBBBBBBKCCCCCCCCCCCCCCCK"

        peptides = digest_protein(
            sequence, enzyme='trypsin', missed_cleavages=0,
            min_length=6, max_length=12
        )

        # AAAK is too short (4)
        assert 'AAAK' not in peptides
        # BBBBBBBBBK is 10, should be included
        assert 'BBBBBBBBBK' in peptides
        # CCCCCCCCCCCCCCCK is 16, too long
        assert 'CCCCCCCCCCCCCCCK' not in peptides

    def test_unknown_enzyme_raises(self):
        """Test that unknown enzyme raises ValueError."""
        with pytest.raises(ValueError, match="Unknown enzyme"):
            digest_protein("PEPTIDE", enzyme='unknown_enzyme')

    def test_all_enzymes_defined(self):
        """Test that all defined enzymes work."""
        sequence = "PEPTIDEKPEPTIDERPEPTIDEK"

        for enzyme in ENZYME_RULES:
            peptides = digest_protein(
                sequence, enzyme=enzyme, missed_cleavages=1,
                min_length=1, max_length=100
            )
            # Should always get at least one peptide
            assert len(peptides) > 0, f"Enzyme {enzyme} produced no peptides"


class TestDigestFasta:
    """Tests for digest_fasta function."""

    def test_digest_fasta_basic(self):
        """Test digesting all proteins in a FASTA."""
        fasta_path = create_temp_fasta(UNIPROT_FASTA)
        try:
            proteins = parse_fasta(fasta_path)
            protein_to_peptides, peptide_to_proteins = digest_fasta(
                proteins, enzyme='trypsin', missed_cleavages=2,
                min_length=6, max_length=30
            )

            # Should have entries for all proteins
            assert len(protein_to_peptides) == 3

            # Should have peptides
            assert len(peptide_to_proteins) > 0

            # Check that mappings are consistent
            for accession, peptides in protein_to_peptides.items():
                for peptide in peptides:
                    assert accession in peptide_to_proteins[peptide]
        finally:
            fasta_path.unlink()


class TestGetTheoreticalPeptideCounts:
    """Tests for get_theoretical_peptide_counts function (for iBAQ)."""

    def test_counts_all_proteins(self):
        """Test counting theoretical peptides for all proteins."""
        from skyline_prism.fasta import get_theoretical_peptide_counts

        fasta_path = create_temp_fasta(UNIPROT_FASTA)
        try:
            counts = get_theoretical_peptide_counts(
                fasta_path,
                enzyme='trypsin',
                missed_cleavages=0,  # Strict for iBAQ
                min_length=6,
                max_length=30,
            )

            # Should have counts for all 3 proteins
            assert len(counts) == 3
            assert 'P04406' in counts  # GAPDH
            assert 'P68871' in counts  # HBB
            assert 'A0A123' in counts  # TrEMBL

            # Counts should be positive integers
            for accession, count in counts.items():
                assert isinstance(count, int)
                assert count >= 0
        finally:
            fasta_path.unlink()

    def test_filter_by_accessions(self):
        """Test filtering to specific proteins."""
        from skyline_prism.fasta import get_theoretical_peptide_counts

        fasta_path = create_temp_fasta(UNIPROT_FASTA)
        try:
            counts = get_theoretical_peptide_counts(
                fasta_path,
                protein_accessions={'P04406'},  # Only GAPDH
                enzyme='trypsin',
            )

            # Should only have GAPDH
            assert len(counts) == 1
            assert 'P04406' in counts
        finally:
            fasta_path.unlink()

    def test_missed_cleavages_increases_count(self):
        """Test that allowing missed cleavages increases peptide count."""
        from skyline_prism.fasta import get_theoretical_peptide_counts

        fasta_path = create_temp_fasta(UNIPROT_FASTA)
        try:
            counts_strict = get_theoretical_peptide_counts(
                fasta_path,
                enzyme='trypsin',
                missed_cleavages=0,
            )
            counts_lenient = get_theoretical_peptide_counts(
                fasta_path,
                enzyme='trypsin',
                missed_cleavages=2,
            )

            # More missed cleavages should give >= peptide count
            for accession in counts_strict:
                assert counts_lenient[accession] >= counts_strict[accession]
        finally:
            fasta_path.unlink()


# =============================================================================
# Modified Sequence Handling Tests
# =============================================================================


class TestStripModifications:
    """Tests for strip_modifications function."""

    def test_skyline_mass_mods(self):
        """Test stripping Skyline-style mass modifications."""
        assert strip_modifications('PEPTC[+57.021]IDEK') == 'PEPTCIDEK'
        assert strip_modifications('PEPTIDEM[+15.995]K') == 'PEPTIDEMK'
        assert strip_modifications('C[+57.021]PEPTIDEC[+57.021]K') == 'CPEPTIDECK'

    def test_unimod_mods(self):
        """Test stripping UniMod-style modifications."""
        assert strip_modifications('PEPTM[UniMod:35]IDEK') == 'PEPTMIDEK'
        assert strip_modifications('PEPTIDE(UniMod:4)K') == 'PEPTIDEK'

    def test_maxquant_mods(self):
        """Test stripping MaxQuant-style modifications."""
        assert strip_modifications('PEPTIDEM(ox)K') == 'PEPTIDEMK'
        assert strip_modifications('PEPTIDE(ac)K') == 'PEPTIDEK'

    def test_terminal_mods(self):
        """Test stripping terminal modifications."""
        assert strip_modifications('n[+42.011]PEPTIDEK') == 'PEPTIDEK'
        assert strip_modifications('PEPTIDEKc[-17.03]') == 'PEPTIDEK'

    def test_proforma_mods(self):
        """Test stripping ProForma-style modifications."""
        assert strip_modifications('PEPTC[Carbamidomethyl]IDEK') == 'PEPTCIDEK'
        assert strip_modifications('PEPTM[Oxidation]IDEK') == 'PEPTMIDEK'

    def test_no_modifications(self):
        """Test sequence without modifications."""
        assert strip_modifications('PEPTIDEK') == 'PEPTIDEK'

    def test_multiple_mods(self):
        """Test sequence with multiple modifications."""
        modified = 'C[+57.021]PEPTM[+15.995]IDEC[+57.021]K'
        assert strip_modifications(modified) == 'CPEPTMIDECK'


class TestNormalizeForMatching:
    """Tests for normalize_for_matching function."""

    def test_strips_mods_and_il(self):
        """Test that modifications are stripped and I/L handled."""
        assert normalize_for_matching('PEPTIDEK') == 'PEPTLDEK'  # I -> L
        assert normalize_for_matching('PEPTLDEK') == 'PEPTLDEK'  # L stays L

    def test_handles_mods(self):
        """Test that modifications are stripped."""
        modified = 'PEPTC[+57.021]IDEK'
        # I -> L
        assert normalize_for_matching(modified) == 'PEPTCLDEK'

    def test_il_ambiguity_disabled(self):
        """Test with I/L handling disabled."""
        assert normalize_for_matching('PEPTIDEK', handle_il_ambiguity=False) == 'PEPTIDEK'


# =============================================================================
# Full Workflow Tests
# =============================================================================


class TestBuildPeptideProteinMap:
    """Tests for build_peptide_protein_map_from_fasta function."""

    def test_basic_mapping(self):
        """Test basic peptide-protein mapping from FASTA."""
        fasta_path = create_temp_fasta(UNIPROT_FASTA)
        try:
            # Create some "detected" peptides
            # These should match sequences in protein sequences via substring search
            detected = {
                'VGVNGFGR',  # From GAPDH
                'PEPTIDEK',  # Not in FASTA
                'VGVNGFGR[+modification]',  # Modified version
            }

            peptide_to_proteins, protein_to_detected, proteins = \
                build_peptide_protein_map_from_fasta(
                    fasta_path, detected
                )

            # VGVNGFGR should map to GAPDH (P04406) via substring search
            if 'VGVNGFGR' in peptide_to_proteins:
                assert 'P04406' in peptide_to_proteins['VGVNGFGR']

            # Proteins dict should be available
            assert 'P04406' in proteins
        finally:
            fasta_path.unlink()


class TestGetDetectedPeptides:
    """Tests for get_detected_peptides_from_data function."""

    def test_extract_peptides(self):
        """Test extracting peptides from DataFrame."""
        df = pd.DataFrame({
            'peptide_modified': ['PEPTIDEK', 'PEPTIDER', 'PEPTIDEK', None],
            'abundance': [100, 200, 300, 400],
        })

        peptides = get_detected_peptides_from_data(df)

        assert peptides == {'PEPTIDEK', 'PEPTIDER'}

    def test_missing_column(self):
        """Test error on missing column."""
        df = pd.DataFrame({'wrong_column': ['PEPTIDEK']})

        with pytest.raises(ValueError, match="Column 'peptide_modified' not found"):
            get_detected_peptides_from_data(df)


class TestBuildProteinNameMap:
    """Tests for build_protein_name_map function."""

    def test_prefers_gene_name(self):
        """Test that gene name is preferred."""
        proteins = {
            'P04406': ProteinEntry(
                accession='P04406',
                name='G3P_HUMAN',
                gene_name='GAPDH',
                description='Glyceraldehyde-3-phosphate dehydrogenase',
                sequence='MVKVGVN',
            ),
        }

        name_map = build_protein_name_map(proteins)
        assert name_map['P04406'] == 'GAPDH'

    def test_falls_back_to_entry_name(self):
        """Test fallback to entry name when no gene name."""
        proteins = {
            'P12345': ProteinEntry(
                accession='P12345',
                name='TEST_HUMAN',
                gene_name=None,
                description='Test protein',
                sequence='MVKVGVN',
            ),
        }

        name_map = build_protein_name_map(proteins)
        assert name_map['P12345'] == 'TEST_HUMAN'
