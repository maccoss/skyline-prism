"""
Tests for data I/O module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from skyline_prism.data_io import (
    validate_skyline_report,
    load_skyline_report,
    load_sample_metadata,
    _standardize_columns,
    SKYLINE_COLUMN_MAP,
    REQUIRED_COLUMNS,
)


class TestColumnStandardization:
    """Tests for column name standardization."""
    
    def test_standard_skyline_columns(self):
        """Test standardization of standard Skyline column names."""
        df = pd.DataFrame({
            'Protein Accession': ['P12345'],
            'Peptide Modified Sequence': ['PEPTIDE'],
            'Precursor Charge': [2],
            'Best Retention Time': [15.5],
            'Total Area Fragment': [1000.0],
            'Replicate Name': ['Sample1'],
        })
        
        result = _standardize_columns(df)
        
        assert 'protein_ids' in result.columns
        assert 'peptide_modified' in result.columns
        assert 'precursor_charge' in result.columns
        assert 'retention_time' in result.columns
        assert 'abundance_fragment' in result.columns
        assert 'replicate_name' in result.columns
    
    def test_alternative_column_names(self):
        """Test standardization of alternative column naming conventions."""
        df = pd.DataFrame({
            'ProteinAccession': ['P12345'],
            'ModifiedSequence': ['PEPTIDE'],
            'PrecursorCharge': [2],
            'RetentionTime': [15.5],
            'ReplicateName': ['Sample1'],
        })
        
        result = _standardize_columns(df)
        
        assert 'protein_ids' in result.columns
        assert 'peptide_modified' in result.columns
        assert 'precursor_charge' in result.columns
        assert 'retention_time' in result.columns
        assert 'replicate_name' in result.columns
    
    def test_unknown_columns_preserved(self):
        """Test that unknown columns are preserved unchanged."""
        df = pd.DataFrame({
            'Protein Accession': ['P12345'],
            'CustomColumn': ['value'],
            'AnotherCustom': [123],
        })
        
        result = _standardize_columns(df)
        
        assert 'CustomColumn' in result.columns
        assert 'AnotherCustom' in result.columns


class TestValidateSkylineReport:
    """Tests for Skyline report validation."""
    
    def test_valid_report(self, tmp_path):
        """Test validation of a valid Skyline report."""
        report_file = tmp_path / "valid_report.csv"
        df = pd.DataFrame({
            'Protein Accession': ['P12345', 'P67890'],
            'Peptide Modified Sequence': ['PEPTIDEK', 'ANOTHERPEPTIDER'],
            'Precursor Charge': [2, 3],
            'Best Retention Time': [15.5, 22.3],
            'Total Area Fragment': [1000.0, 2000.0],
            'Replicate Name': ['Sample1', 'Sample1'],
        })
        df.to_csv(report_file, index=False)
        
        result = validate_skyline_report(report_file)
        
        assert result.is_valid
        assert result.n_rows == 2
        assert len(result.missing_required) == 0
    
    def test_missing_required_columns(self, tmp_path):
        """Test validation catches missing required columns."""
        report_file = tmp_path / "missing_cols.csv"
        df = pd.DataFrame({
            'Protein Accession': ['P12345'],
            'Peptide Modified Sequence': ['PEPTIDEK'],
            # Missing: Precursor Charge, Retention Time, Replicate Name
        })
        df.to_csv(report_file, index=False)
        
        result = validate_skyline_report(report_file)
        
        assert not result.is_valid
        assert len(result.missing_required) > 0
    
    def test_tsv_format(self, tmp_path):
        """Test validation of TSV format reports."""
        report_file = tmp_path / "report.tsv"
        df = pd.DataFrame({
            'Protein Accession': ['P12345'],
            'Peptide Modified Sequence': ['PEPTIDEK'],
            'Precursor Charge': [2],
            'Best Retention Time': [15.5],
            'Total Area Fragment': [1000.0],
            'Replicate Name': ['Sample1'],
        })
        df.to_csv(report_file, index=False, sep='\t')
        
        result = validate_skyline_report(report_file)
        
        assert result.is_valid


class TestLoadSampleMetadata:
    """Tests for sample metadata loading."""
    
    def test_valid_metadata(self, tmp_path):
        """Test loading valid sample metadata."""
        metadata_file = tmp_path / "metadata.tsv"
        df = pd.DataFrame({
            'ReplicateName': ['Sample1', 'Sample2', 'Pool1', 'Ref1'],
            'SampleType': ['experimental', 'experimental', 'pool', 'reference'],
            'Batch': ['batch1', 'batch1', 'batch1', 'batch1'],
            'RunOrder': [1, 2, 3, 4],
        })
        df.to_csv(metadata_file, index=False, sep='\t')
        
        result = load_sample_metadata(metadata_file)
        
        assert len(result) == 4
        assert 'ReplicateName' in result.columns
        assert 'SampleType' in result.columns
    
    def test_invalid_sample_type(self, tmp_path):
        """Test that invalid sample types raise an error."""
        metadata_file = tmp_path / "bad_metadata.tsv"
        df = pd.DataFrame({
            'ReplicateName': ['Sample1'],
            'SampleType': ['invalid_type'],  # Not in valid types
            'Batch': ['batch1'],
            'RunOrder': [1],
        })
        df.to_csv(metadata_file, index=False, sep='\t')
        
        with pytest.raises(ValueError):
            load_sample_metadata(metadata_file)


class TestCreateTestData:
    """Helper functions for creating test data."""
    
    @staticmethod
    def create_skyline_report(
        n_proteins: int = 10,
        n_peptides_per_protein: int = 5,
        n_replicates: int = 6,
        include_reference: bool = True,
        include_pool: bool = True,
    ) -> pd.DataFrame:
        """Create a synthetic Skyline report for testing."""
        rows = []
        
        for prot_idx in range(n_proteins):
            protein_id = f"P{prot_idx:05d}"
            protein_name = f"Protein{prot_idx}"
            
            for pep_idx in range(n_peptides_per_protein):
                peptide = f"PEPTIDE{prot_idx}_{pep_idx}K"
                rt = 10 + pep_idx * 5 + np.random.normal(0, 0.5)
                
                for rep_idx in range(n_replicates):
                    # Determine sample type
                    if include_reference and rep_idx < 2:
                        replicate = f"Reference_{rep_idx + 1}"
                        sample_type = "reference"
                    elif include_pool and rep_idx < 4:
                        replicate = f"Pool_{rep_idx - 1}"
                        sample_type = "pool"
                    else:
                        replicate = f"Sample_{rep_idx + 1}"
                        sample_type = "experimental"
                    
                    # Generate abundance with some variation
                    base_abundance = 1000 * (prot_idx + 1) * (pep_idx + 1)
                    abundance = base_abundance * np.random.lognormal(0, 0.2)
                    
                    rows.append({
                        'Protein Accession': protein_id,
                        'Protein Name': protein_name,
                        'Peptide Modified Sequence': peptide,
                        'Precursor Charge': 2,
                        'Best Retention Time': rt,
                        'Total Area Fragment': abundance,
                        'Replicate Name': replicate,
                    })
        
        return pd.DataFrame(rows)
