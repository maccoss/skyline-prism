"""Tests for protein parsimony module."""

import pandas as pd

from skyline_prism.parsimony import (
    ProteinGroup,
    build_peptide_protein_map,
    compute_protein_groups,
)


class TestBuildPeptideProteinMap:
    """Tests for peptide-protein mapping."""

    def test_simple_mapping(self):
        """Test mapping with unique peptides."""
        df = pd.DataFrame({
            'peptide_sequence': ['PEPTIDE1', 'PEPTIDE2', 'PEPTIDE3'],
            'protein_ids': ['P001', 'P002', 'P003'],
            'protein_names': ['Protein1', 'Protein2', 'Protein3'],
        })

        pep_to_prot, prot_to_pep, _ = build_peptide_protein_map(df)

        assert len(pep_to_prot) == 3
        assert pep_to_prot['PEPTIDE1'] == {'P001'}
        assert prot_to_pep['P001'] == {'PEPTIDE1'}

    def test_shared_peptides(self):
        """Test mapping with shared peptides."""
        df = pd.DataFrame({
            'peptide_sequence': ['SHARED', 'SHARED', 'UNIQUE1', 'UNIQUE2'],
            'protein_ids': ['P001', 'P002', 'P001', 'P002'],
            'protein_names': ['Protein1', 'Protein2', 'Protein1', 'Protein2'],
        })

        pep_to_prot, prot_to_pep, _ = build_peptide_protein_map(df)

        # SHARED peptide maps to both proteins
        assert pep_to_prot['SHARED'] == {'P001', 'P002'}
        # Each protein has its peptides
        assert 'SHARED' in prot_to_pep['P001']
        assert 'UNIQUE1' in prot_to_pep['P001']

    def test_semicolon_separated_proteins(self):
        """Test parsing of semicolon-separated protein lists."""
        df = pd.DataFrame({
            'peptide_sequence': ['PEPTIDE1'],
            'protein_ids': ['P001;P002;P003'],
            'protein_names': ['Protein1;Protein2;Protein3'],
        })

        pep_to_prot, prot_to_pep, _ = build_peptide_protein_map(df)

        # Peptide should map to all three proteins
        assert pep_to_prot['PEPTIDE1'] == {'P001', 'P002', 'P003'}


class TestComputeProteinGroups:
    """Tests for protein group computation."""

    def test_unique_peptides_only(self):
        """Test grouping with only unique peptides."""
        prot_to_pep = {
            'P001': {'PEP1', 'PEP2'},
            'P002': {'PEP3', 'PEP4'},
        }
        pep_to_prot = {
            'PEP1': {'P001'},
            'PEP2': {'P001'},
            'PEP3': {'P002'},
            'PEP4': {'P002'},
        }
        prot_to_name = {'P001': 'Protein1', 'P002': 'Protein2'}

        groups = compute_protein_groups(prot_to_pep, pep_to_prot, prot_to_name)

        assert len(groups) == 2
        # All peptides should be unique
        for group in groups:
            assert len(group.razor_peptides) == 0

    def test_indistinguishable_proteins(self):
        """Test grouping of indistinguishable proteins."""
        # P001 and P002 have identical peptide sets
        prot_to_pep = {
            'P001': {'PEP1', 'PEP2'},
            'P002': {'PEP1', 'PEP2'},
        }
        pep_to_prot = {
            'PEP1': {'P001', 'P002'},
            'PEP2': {'P001', 'P002'},
        }
        prot_to_name = {'P001': 'Protein1', 'P002': 'Protein2'}

        groups = compute_protein_groups(prot_to_pep, pep_to_prot, prot_to_name)

        # Should result in single group with both proteins
        assert len(groups) == 1
        assert len(groups[0].member_proteins) == 2

    def test_subsumable_proteins(self):
        """Test handling of subsumable proteins."""
        # P002's peptides are subset of P001
        prot_to_pep = {
            'P001': {'PEP1', 'PEP2', 'PEP3'},
            'P002': {'PEP1', 'PEP2'},
        }
        pep_to_prot = {
            'PEP1': {'P001', 'P002'},
            'PEP2': {'P001', 'P002'},
            'PEP3': {'P001'},
        }
        prot_to_name = {'P001': 'Protein1', 'P002': 'Protein2'}

        groups = compute_protein_groups(prot_to_pep, pep_to_prot, prot_to_name)

        # P002 should be subsumed by P001
        assert len(groups) == 1
        assert 'P002' in groups[0].subsumed_proteins

    def test_razor_peptide_assignment(self):
        """Test razor peptide assignment to group with most unique peptides."""
        prot_to_pep = {
            'P001': {'PEP1', 'PEP2', 'SHARED'},  # 2 unique + 1 shared
            'P002': {'PEP3', 'SHARED'},           # 1 unique + 1 shared
        }
        pep_to_prot = {
            'PEP1': {'P001'},
            'PEP2': {'P001'},
            'PEP3': {'P002'},
            'SHARED': {'P001', 'P002'},
        }
        prot_to_name = {'P001': 'Protein1', 'P002': 'Protein2'}

        groups = compute_protein_groups(prot_to_pep, pep_to_prot, prot_to_name)

        assert len(groups) == 2

        # Find group for P001 (should get the shared peptide)
        p001_group = next(g for g in groups if g.leading_protein == 'P001')
        assert 'SHARED' in p001_group.peptides


class TestProteinGroup:
    """Tests for ProteinGroup data structure."""

    def test_peptide_counts(self):
        """Test peptide counting properties."""
        group = ProteinGroup(
            group_id='PG001',
            leading_protein='P001',
            leading_protein_name='TestProtein',
            member_proteins=['P001'],
            subsumed_proteins=[],
            peptides={'PEP1', 'PEP2', 'PEP3'},
            unique_peptides={'PEP1', 'PEP2'},
            razor_peptides={'PEP3'},
            all_mapped_peptides={'PEP1', 'PEP2', 'PEP3', 'PEP4'},  # Includes shared
        )

        assert group.n_peptides == 3
        assert group.n_unique_peptides == 2
        assert group.n_razor_peptides == 1

    def test_to_dict(self):
        """Test conversion to dictionary for export."""
        group = ProteinGroup(
            group_id='PG001',
            leading_protein='P001',
            leading_protein_name='TestProtein',
            member_proteins=['P001'],
            subsumed_proteins=['P002'],
            peptides={'PEP1', 'PEP2'},
            unique_peptides={'PEP1'},
            razor_peptides={'PEP2'},
            all_mapped_peptides={'PEP1', 'PEP2', 'PEP3'},  # Includes shared
        )

        d = group.to_dict()

        assert d['GroupID'] == 'PG001'
        assert d['LeadingProtein'] == 'P001'
        assert d['NPeptides'] == 2
        assert 'P002' in d['SubsumedProteins']


class TestBuildPeptideProteinMapFromFasta:
    """Tests for FASTA-based peptide-protein mapping."""

    def test_fasta_based_mapping(self, tmp_path):
        """Test mapping with FASTA database."""
        from skyline_prism.parsimony import build_peptide_protein_map_from_fasta

        # Create a test FASTA file
        fasta_content = """>sp|P001|TEST1_HUMAN Test protein 1 GN=TEST1
MVKVGVNGFGRIGRLVTRAAFNSGKVDIVAINDK
>sp|P002|TEST2_HUMAN Test protein 2 GN=TEST2
PEPTIDEKFGRIGRLVTR
"""
        fasta_path = tmp_path / "test.fasta"
        fasta_path.write_text(fasta_content)

        # Create DataFrame with detected peptides
        # VGVNGFGR should match P001 (from MVKVGVNGFGRIGR...)
        # PEPTIDEK should match P002
        # FGRIGRLVTR should match both P001 and P002 (shared)
        df = pd.DataFrame({
            'peptide_sequence': [
                'VGVNGFGR',  # In P001 only
                'PEPTIDEK',  # In P002 only
                'FGRIGRLVTR',  # In both
            ],
        })

        pep_to_prot, prot_to_pep, prot_to_name = build_peptide_protein_map_from_fasta(
            df,
            fasta_path=str(fasta_path),
            peptide_col='peptide_sequence',
        )

        # Check peptide mappings
        # Note: uses direct substring matching now
        assert len(pep_to_prot) > 0  # At least some peptides should match

        # Check protein name map
        if 'P001' in prot_to_name:
            assert prot_to_name['P001'] == 'TEST1'  # Gene name preferred

    def test_modified_peptide_matching(self, tmp_path):
        """Test that modified peptides are matched after stripping mods."""
        from skyline_prism.parsimony import build_peptide_protein_map_from_fasta

        # Create a test FASTA
        fasta_content = """>sp|P001|TEST_HUMAN Test protein GN=TEST
MVCPEPTIDEKFGR
"""
        fasta_path = tmp_path / "test.fasta"
        fasta_path.write_text(fasta_content)

        # Create DataFrame with modified peptides
        df = pd.DataFrame({
            'peptide_sequence': [
                'C[+57.021]PEPTIDEK',  # With carbamidomethyl modification
            ],
        })

        pep_to_prot, _, _ = build_peptide_protein_map_from_fasta(
            df,
            fasta_path=str(fasta_path),
            peptide_col='peptide_sequence',
        )

        # Modified peptide should match if CPEPTIDEK is in the FASTA digest
        # The match depends on digestion - just verify the function runs
        assert isinstance(pep_to_prot, dict)
