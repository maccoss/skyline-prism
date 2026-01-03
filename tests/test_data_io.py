"""Tests for data I/O module."""

import numpy as np
import pandas as pd
import pytest

from skyline_prism.data_io import (
    _standardize_columns,
    load_sample_metadata,
    merge_skyline_reports_streaming,
    validate_skyline_report,
)


class TestColumnStandardization:
    """Tests for column name standardization.

    Note: _standardize_columns now returns the DataFrame unchanged (no renaming).
    The pipeline uses original Skyline column names throughout.
    """

    def test_standard_skyline_columns(self):
        """Test that standard Skyline column names are preserved unchanged."""
        df = pd.DataFrame(
            {
                "Protein Accession": ["P12345"],
                "Peptide Modified Sequence": ["PEPTIDE"],
                "Precursor Charge": [2],
                "Best Retention Time": [15.5],
                "Total Area Fragment": [1000.0],
                "Replicate Name": ["Sample1"],
            }
        )

        result = _standardize_columns(df)

        # Column names are preserved unchanged
        assert "Protein Accession" in result.columns
        assert "Peptide Modified Sequence" in result.columns
        assert "Precursor Charge" in result.columns
        assert "Best Retention Time" in result.columns
        assert "Total Area Fragment" in result.columns
        assert "Replicate Name" in result.columns

    def test_alternative_column_names(self):
        """Test that alternative column naming conventions are preserved unchanged."""
        df = pd.DataFrame(
            {
                "ProteinAccession": ["P12345"],
                "ModifiedSequence": ["PEPTIDE"],
                "PrecursorCharge": [2],
                "RetentionTime": [15.5],
                "ReplicateName": ["Sample1"],
            }
        )

        result = _standardize_columns(df)

        # Column names are preserved unchanged
        assert "ProteinAccession" in result.columns
        assert "ModifiedSequence" in result.columns
        assert "PrecursorCharge" in result.columns
        assert "RetentionTime" in result.columns
        assert "ReplicateName" in result.columns

    def test_unknown_columns_preserved(self):
        """Test that unknown columns are preserved unchanged."""
        df = pd.DataFrame(
            {
                "Protein Accession": ["P12345"],
                "CustomColumn": ["value"],
                "AnotherCustom": [123],
            }
        )

        result = _standardize_columns(df)

        assert "CustomColumn" in result.columns
        assert "AnotherCustom" in result.columns


class TestValidateSkylineReport:
    """Tests for Skyline report validation."""

    def test_valid_report(self, tmp_path):
        """Test validation of a valid Skyline report."""
        report_file = tmp_path / "valid_report.csv"
        df = pd.DataFrame(
            {
                "Protein Accession": ["P12345", "P67890"],
                "Peptide Modified Sequence": ["PEPTIDEK", "ANOTHERPEPTIDER"],
                "Precursor Charge": [2, 3],
                "Best Retention Time": [15.5, 22.3],
                "Total Area Fragment": [1000.0, 2000.0],
                "Replicate Name": ["Sample1", "Sample1"],
            }
        )
        df.to_csv(report_file, index=False)

        result = validate_skyline_report(report_file)

        assert result.is_valid
        assert result.n_rows == 2
        assert len(result.missing_required) == 0

    def test_missing_required_columns(self, tmp_path):
        """Test validation catches missing required columns."""
        report_file = tmp_path / "missing_cols.csv"
        df = pd.DataFrame(
            {
                "Protein Accession": ["P12345"],
                "Peptide Modified Sequence": ["PEPTIDEK"],
                # Missing: Precursor Charge, Retention Time, Replicate Name
            }
        )
        df.to_csv(report_file, index=False)

        result = validate_skyline_report(report_file)

        assert not result.is_valid
        assert len(result.missing_required) > 0

    def test_tsv_format(self, tmp_path):
        """Test validation of TSV format reports."""
        report_file = tmp_path / "report.tsv"
        df = pd.DataFrame(
            {
                "Protein Accession": ["P12345"],
                "Peptide Modified Sequence": ["PEPTIDEK"],
                "Precursor Charge": [2],
                "Best Retention Time": [15.5],
                "Total Area Fragment": [1000.0],
                "Replicate Name": ["Sample1"],
            }
        )
        df.to_csv(report_file, index=False, sep="\t")

        result = validate_skyline_report(report_file)

        assert result.is_valid


class TestLoadSampleMetadata:
    """Tests for sample metadata loading."""

    def test_valid_metadata(self, tmp_path):
        """Test loading valid sample metadata."""
        metadata_file = tmp_path / "metadata.tsv"
        df = pd.DataFrame(
            {
                "ReplicateName": ["Sample1", "Sample2", "Pool1", "Ref1"],
                "SampleType": ["experimental", "experimental", "qc", "reference"],
                "Batch": ["batch1", "batch1", "batch1", "batch1"],
                "RunOrder": [1, 2, 3, 4],
            }
        )
        df.to_csv(metadata_file, index=False, sep="\t")

        result = load_sample_metadata(metadata_file)

        assert len(result) == 4
        # Function normalizes column names to lowercase
        assert "sample" in result.columns
        assert "sample_type" in result.columns

    def test_metadata_without_run_order(self, tmp_path):
        """Test that metadata without RunOrder is valid (it's optional)."""
        metadata_file = tmp_path / "metadata_no_runorder.tsv"
        df = pd.DataFrame(
            {
                "ReplicateName": ["Sample1", "Sample2", "Pool1", "Ref1"],
                "SampleType": ["experimental", "experimental", "qc", "reference"],
                "Batch": ["batch1", "batch1", "batch1", "batch1"],
                # No RunOrder column - it's optional
            }
        )
        df.to_csv(metadata_file, index=False, sep="\t")

        result = load_sample_metadata(metadata_file)

        assert len(result) == 4
        assert "RunOrder" not in result.columns  # Not added by load_sample_metadata

    def test_metadata_without_batch(self, tmp_path):
        """Test that metadata without Batch is valid (it's optional)."""
        metadata_file = tmp_path / "metadata_no_batch.tsv"
        df = pd.DataFrame(
            {
                "ReplicateName": ["Sample1", "Sample2", "Pool1", "Ref1"],
                "SampleType": ["experimental", "experimental", "qc", "reference"],
                # No Batch column - it's optional
            }
        )
        df.to_csv(metadata_file, index=False, sep="\t")

        result = load_sample_metadata(metadata_file)

        assert len(result) == 4
        assert "Batch" not in result.columns  # Not added by load_sample_metadata

    def test_invalid_sample_type(self, tmp_path):
        """Test that invalid sample types raise an error."""
        metadata_file = tmp_path / "bad_metadata.tsv"
        df = pd.DataFrame(
            {
                "ReplicateName": ["Sample1"],
                "SampleType": ["invalid_type"],  # Not in valid types
                "Batch": ["batch1"],
                "RunOrder": [1],
            }
        )
        df.to_csv(metadata_file, index=False, sep="\t")

        with pytest.raises(ValueError):
            load_sample_metadata(metadata_file)

    def test_skyline_column_names(self, tmp_path):
        """Test that Skyline column names are properly normalized."""
        metadata_file = tmp_path / "skyline_metadata.csv"
        df = pd.DataFrame(
            {
                "Replicate Name": ["Sample1", "Sample2", "Pool1", "Ref1"],
                "Sample Type": ["Unknown", "Unknown", "Quality Control", "Standard"],
                "Batch Name": ["batch1", "batch1", "batch1", "batch1"],
            }
        )
        df.to_csv(metadata_file, index=False)

        result = load_sample_metadata(metadata_file)

        # Check column names were normalized to lowercase
        assert "sample" in result.columns
        assert "sample_type" in result.columns
        assert "batch" in result.columns

        # Check Skyline sample types were mapped
        assert set(result["sample_type"].unique()) == {"experimental", "qc", "reference"}

    def test_file_name_fallback(self, tmp_path):
        """Test that File Name is accepted when Replicate Name is not present."""
        metadata_file = tmp_path / "file_name_metadata.csv"
        df = pd.DataFrame(
            {
                "File Name": ["Sample1.raw", "Sample2.raw", "Pool1.raw", "Ref1.raw"],
                "Sample Type": ["Unknown", "Unknown", "Quality Control", "Standard"],
            }
        )
        df.to_csv(metadata_file, index=False)

        result = load_sample_metadata(metadata_file)

        # Function normalizes column names to lowercase
        assert "sample" in result.columns
        assert len(result) == 4


class TestCreateTestData:
    """Helper functions for creating test data."""

    @staticmethod
    def create_skyline_report(
        n_proteins: int = 10,
        n_peptides_per_protein: int = 5,
        n_replicates: int = 6,
        include_reference: bool = True,
        include_qc: bool = True,
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
                    elif include_qc and rep_idx < 4:
                        replicate = f"Pool_{rep_idx - 1}"
                    else:
                        replicate = f"Sample_{rep_idx + 1}"

                    # Generate abundance with some variation
                    base_abundance = 1000 * (prot_idx + 1) * (pep_idx + 1)
                    abundance = base_abundance * np.random.lognormal(0, 0.2)

                    rows.append(
                        {
                            "Protein Accession": protein_id,
                            "Protein Name": protein_name,
                            "Peptide Modified Sequence": peptide,
                            "Precursor Charge": 2,
                            "Best Retention Time": rt,
                            "Total Area Fragment": abundance,
                            "Replicate Name": replicate,
                        }
                    )

        return pd.DataFrame(rows)


class TestMergeStreamingSchemaNormalization:
    """Tests for schema normalization in streaming merge.

    When merging multiple CSV files, PyArrow may infer different types for
    the same column (e.g., 'Acquired Time' as timestamp in one file, string
    in another). The merge function must normalize these to a consistent type.
    """

    def test_merge_with_mixed_timestamp_types(self, tmp_path):
        """Test that files with different date column types can be merged.

        This reproduces the bug where Plate1 had 'Acquired Time' as string
        but Plate2 had it as timestamp, causing a schema mismatch error.
        """
        # Create first CSV with date as a string format that PyArrow parses as string
        csv1 = tmp_path / "plate1.csv"
        df1 = pd.DataFrame(
            {
                "Protein": ["P1", "P1"],
                "Peptide Modified Sequence Unimod Ids": ["PEPTIDEK", "PEPTIDEK"],
                "Precursor Charge": [2, 2],
                "Fragment Ion": ["y5", "y6"],
                "Product Charge": [1, 1],
                "Product Mz": [500.0, 600.0],
                "Area": [1000, 2000],
                "Replicate Name": ["Sample1", "Sample1"],
                # Date format that PyArrow typically parses as string
                "Acquired Time": ["2025-01-15 10:30:00", "2025-01-15 10:30:00"],
            }
        )
        df1.to_csv(csv1, index=False)

        # Create second CSV with date format that PyArrow parses as timestamp
        csv2 = tmp_path / "plate2.csv"
        df2 = pd.DataFrame(
            {
                "Protein": ["P2", "P2"],
                "Peptide Modified Sequence Unimod Ids": ["ANOTHERK", "ANOTHERK"],
                "Precursor Charge": [2, 2],
                "Fragment Ion": ["y5", "y6"],
                "Product Charge": [1, 1],
                "Product Mz": [500.0, 600.0],
                "Area": [3000, 4000],
                "Replicate Name": ["Sample2", "Sample2"],
                # ISO format that PyArrow typically parses as timestamp
                "Acquired Time": ["2025-01-16T11:45:00", "2025-01-16T11:45:00"],
            }
        )
        df2.to_csv(csv2, index=False)

        # Merge should succeed without schema mismatch error
        output_path = tmp_path / "merged.parquet"
        result_path, samples_by_batch, total_rows = merge_skyline_reports_streaming(
            report_paths=[csv1, csv2],
            output_path=output_path,
            batch_names=["Plate1", "Plate2"],
        )

        # Verify merge succeeded
        assert result_path.exists()
        assert total_rows == 4

        # Verify data can be read back
        merged_df = pd.read_parquet(result_path)
        assert len(merged_df) == 4
        assert "Acquired Time" in merged_df.columns

        # Acquired Time should be string type (normalized)
        assert merged_df["Acquired Time"].dtype == object  # pandas string dtype

    def test_merge_with_consistent_types(self, tmp_path):
        """Test that merge works normally when types are already consistent."""
        csv1 = tmp_path / "plate1.csv"
        df1 = pd.DataFrame(
            {
                "Protein": ["P1"],
                "Peptide Modified Sequence Unimod Ids": ["PEPTIDEK"],
                "Precursor Charge": [2],
                "Fragment Ion": ["y5"],
                "Product Charge": [1],
                "Product Mz": [500.0],
                "Area": [1000],
                "Replicate Name": ["Sample1"],
            }
        )
        df1.to_csv(csv1, index=False)

        csv2 = tmp_path / "plate2.csv"
        df2 = pd.DataFrame(
            {
                "Protein": ["P2"],
                "Peptide Modified Sequence Unimod Ids": ["ANOTHERK"],
                "Precursor Charge": [2],
                "Fragment Ion": ["y5"],
                "Product Charge": [1],
                "Product Mz": [500.0],
                "Area": [3000],
                "Replicate Name": ["Sample2"],
            }
        )
        df2.to_csv(csv2, index=False)

        output_path = tmp_path / "merged.parquet"
        result_path, samples_by_batch, total_rows = merge_skyline_reports_streaming(
            report_paths=[csv1, csv2],
            output_path=output_path,
            batch_names=["Plate1", "Plate2"],
        )

        assert result_path.exists()
        assert total_rows == 2

        merged_df = pd.read_parquet(result_path)
        assert len(merged_df) == 2
        assert set(merged_df["Protein"].unique()) == {"P1", "P2"}

    def test_merge_preserves_sample_ids(self, tmp_path):
        """Test that Sample ID column is correctly created during merge."""
        csv1 = tmp_path / "plate1.csv"
        df1 = pd.DataFrame(
            {
                "Protein": ["P1", "P1"],
                "Peptide Modified Sequence Unimod Ids": ["PEPTIDEK", "PEPTIDEK"],
                "Precursor Charge": [2, 2],
                "Fragment Ion": ["y5", "y6"],
                "Product Charge": [1, 1],
                "Product Mz": [500.0, 600.0],
                "Area": [1000, 2000],
                "Replicate Name": ["Pool_001", "Pool_001"],
            }
        )
        df1.to_csv(csv1, index=False)

        csv2 = tmp_path / "plate2.csv"
        df2 = pd.DataFrame(
            {
                "Protein": ["P1", "P1"],
                "Peptide Modified Sequence Unimod Ids": ["PEPTIDEK", "PEPTIDEK"],
                "Precursor Charge": [2, 2],
                "Fragment Ion": ["y5", "y6"],
                "Product Charge": [1, 1],
                "Product Mz": [500.0, 600.0],
                "Area": [3000, 4000],
                # Same replicate name in different batch
                "Replicate Name": ["Pool_001", "Pool_001"],
            }
        )
        df2.to_csv(csv2, index=False)

        output_path = tmp_path / "merged.parquet"
        result_path, samples_by_batch, total_rows = merge_skyline_reports_streaming(
            report_paths=[csv1, csv2],
            output_path=output_path,
            batch_names=["Plate1", "Plate2"],
        )

        merged_df = pd.read_parquet(result_path)

        # Sample ID should distinguish same replicate name across batches
        assert "Sample ID" in merged_df.columns
        sample_ids = merged_df["Sample ID"].unique()
        assert len(sample_ids) == 2  # Pool_001 in Plate1 and Pool_001 in Plate2
        assert any("Plate1" in sid for sid in sample_ids)
        assert any("Plate2" in sid for sid in sample_ids)


class TestLoadSampleMetadataReplicateColumn:
    """Tests for 'Replicate' column support (without 'Name' suffix)."""

    def test_replicate_column_accepted(self, tmp_path):
        """Test that 'Replicate' column is accepted as sample identifier."""
        metadata_file = tmp_path / "metadata.csv"
        df = pd.DataFrame(
            {
                "Replicate": ["Sample1", "Sample2", "Pool1", "Ref1"],
                "Sample Type": ["Unknown", "Unknown", "Quality Control", "Standard"],
            }
        )
        df.to_csv(metadata_file, index=False)

        result = load_sample_metadata(metadata_file)

        # Check column was normalized to 'sample'
        assert "sample" in result.columns
        assert len(result) == 4
        assert set(result["sample"].unique()) == {"Sample1", "Sample2", "Pool1", "Ref1"}

    def test_replicate_column_with_batch(self, tmp_path):
        """Test 'Replicate' column with batch information."""
        metadata_file = tmp_path / "metadata.csv"
        df = pd.DataFrame(
            {
                "Replicate": ["Sample1", "Sample2", "QC1", "Ref1"],
                "Sample Type": ["Unknown", "Unknown", "Quality Control", "Standard"],
                "Batch Name": ["batch1", "batch1", "batch1", "batch1"],
            }
        )
        df.to_csv(metadata_file, index=False)

        result = load_sample_metadata(metadata_file)

        assert "sample" in result.columns
        assert "sample_type" in result.columns
        assert "batch" in result.columns
        # Check Skyline types were mapped
        assert set(result["sample_type"].unique()) == {"experimental", "qc", "reference"}


class TestLoadSampleMetadataFiles:
    """Tests for loading and merging multiple metadata files."""

    def test_single_file(self, tmp_path):
        """Test that single file is loaded correctly."""
        from skyline_prism.data_io import load_sample_metadata_files

        metadata_file = tmp_path / "metadata.csv"
        df = pd.DataFrame(
            {
                "Replicate": ["Sample1", "Sample2"],
                "Sample Type": ["Unknown", "Unknown"],
            }
        )
        df.to_csv(metadata_file, index=False)

        result = load_sample_metadata_files([metadata_file])

        assert len(result) == 2
        assert "sample" in result.columns

    def test_merge_two_files(self, tmp_path):
        """Test merging two metadata files."""
        from skyline_prism.data_io import load_sample_metadata_files

        # Create first file
        meta1 = tmp_path / "batch1_metadata.csv"
        df1 = pd.DataFrame(
            {
                "Replicate": ["Sample_A1", "Sample_A2", "QC_1"],
                "Sample Type": ["Unknown", "Unknown", "Quality Control"],
            }
        )
        df1.to_csv(meta1, index=False)

        # Create second file
        meta2 = tmp_path / "batch2_metadata.csv"
        df2 = pd.DataFrame(
            {
                "Replicate": ["Sample_B1", "Sample_B2", "Ref_1"],
                "Sample Type": ["Unknown", "Unknown", "Standard"],
            }
        )
        df2.to_csv(meta2, index=False)

        result = load_sample_metadata_files([meta1, meta2])

        assert len(result) == 6
        assert set(result["sample"].unique()) == {
            "Sample_A1",
            "Sample_A2",
            "QC_1",
            "Sample_B1",
            "Sample_B2",
            "Ref_1",
        }
        assert set(result["sample_type"].unique()) == {"experimental", "qc", "reference"}

    def test_merge_resolves_duplicates_via_batch_inference(self, tmp_path):
        """Test that duplicate samples across files are resolved by inferring batch names.

        When metadata files don't have a 'Batch' column, the loader infers the batch
        from the filename (e.g., 'batch1_metadata'). This allows 'Sample_A1' in file1
        and 'Sample_A1' in file2 to coexist as separate samples.
        """
        from skyline_prism.data_io import load_sample_metadata_files

        # Create first file
        meta1 = tmp_path / "batch1_metadata.csv"
        df1 = pd.DataFrame(
            {
                "Replicate": ["Sample_A1", "Sample_A2"],
                "Sample Type": ["Unknown", "Unknown"],
            }
        )
        df1.to_csv(meta1, index=False)

        # Create second file with duplicate sample name
        meta2 = tmp_path / "batch2_metadata.csv"
        df2 = pd.DataFrame(
            {
                "Replicate": ["Sample_A1", "Sample_B2"],  # Sample_A1 is duplicate
                "Sample Type": ["Unknown", "Unknown"],
            }
        )
        df2.to_csv(meta2, index=False)

        # Should NOT raise ValueError anymore
        result = load_sample_metadata_files([meta1, meta2])

        # We expect 4 rows total (A1, A2 from batch1; A1, B2 from batch2)
        assert len(result) == 4
        # Batches should be inferred
        assert "batch" in result.columns
        assert set(result["batch"].unique()) == {"batch1_metadata", "batch2_metadata"}
        # Sample IDs should be unique
        # (Though implicit sample_id creation happens later in generate_sample_metadata,
        # load_sample_metadata just loads the DF. The merge correctness is validated here by row count.)

    def test_merge_three_files(self, tmp_path):
        """Test merging three metadata files."""
        from skyline_prism.data_io import load_sample_metadata_files

        files = []
        for i in range(3):
            meta_file = tmp_path / f"batch{i + 1}_metadata.csv"
            df = pd.DataFrame(
                {
                    "Replicate": [f"Sample_{i + 1}_a", f"Sample_{i + 1}_b"],
                    "Sample Type": ["Unknown", "Unknown"],
                }
            )
            df.to_csv(meta_file, index=False)
            files.append(meta_file)

        result = load_sample_metadata_files(files)

        assert len(result) == 6
        assert len(result["sample"].unique()) == 6

    def test_merge_preserves_extra_columns(self, tmp_path):
        """Test that extra columns are preserved when merging."""
        from skyline_prism.data_io import load_sample_metadata_files

        # Create files with extra columns
        meta1 = tmp_path / "batch1_metadata.csv"
        df1 = pd.DataFrame(
            {
                "Replicate": ["Sample_A1"],
                "Sample Type": ["Unknown"],
                "Condition": ["Treatment"],
                "Dose": [100],
            }
        )
        df1.to_csv(meta1, index=False)

        meta2 = tmp_path / "batch2_metadata.csv"
        df2 = pd.DataFrame(
            {
                "Replicate": ["Sample_B1"],
                "Sample Type": ["Unknown"],
                "Condition": ["Control"],
                "Dose": [0],
            }
        )
        df2.to_csv(meta2, index=False)

        result = load_sample_metadata_files([meta1, meta2])

        assert len(result) == 2
        assert "Condition" in result.columns
        assert "Dose" in result.columns

    def test_empty_list_raises_error(self, tmp_path):
        """Test that empty file list raises error."""
        from skyline_prism.data_io import load_sample_metadata_files

        with pytest.raises(ValueError, match="No metadata files"):
            load_sample_metadata_files([])


class TestSampleIdHelpers:
    """Tests for Sample ID to Replicate Name conversion helpers."""

    def test_sample_id_to_replicate_name(self):
        """Test extracting Replicate Name from Sample ID."""
        from skyline_prism.cli import sample_id_to_replicate_name

        # With separator
        assert sample_id_to_replicate_name("Sample_A1__@__Batch1") == "Sample_A1"
        assert (
            sample_id_to_replicate_name("DOE-Col1-EVs-451-004__@__2025-12-DOE")
            == "DOE-Col1-EVs-451-004"
        )

        # Without separator (already a Replicate Name)
        assert sample_id_to_replicate_name("Sample_A1") == "Sample_A1"
        assert sample_id_to_replicate_name("DOE-Col1-EVs-451-004") == "DOE-Col1-EVs-451-004"

    def test_build_sample_type_map_direct_match(self):
        """Test build_sample_type_map with direct matching sample names."""
        from skyline_prism.cli import build_sample_type_map

        metadata_df = pd.DataFrame(
            {
                "sample": ["Sample_A1", "Sample_A2", "QC_1", "Ref_1"],
                "sample_type": ["experimental", "experimental", "qc", "reference"],
            }
        )

        sample_cols = ["Sample_A1", "Sample_A2", "QC_1", "Ref_1"]
        result = build_sample_type_map(sample_cols, metadata_df)

        assert result["Sample_A1"] == "experimental"
        assert result["QC_1"] == "qc"
        assert result["Ref_1"] == "reference"

    def test_build_sample_type_map_with_sample_id(self):
        """Test build_sample_type_map with Sample ID format columns."""
        from skyline_prism.cli import build_sample_type_map

        # Metadata uses Replicate Names
        metadata_df = pd.DataFrame(
            {
                "sample": ["Sample_A1", "Sample_A2", "QC_1", "Ref_1"],
                "sample_type": ["experimental", "experimental", "qc", "reference"],
            }
        )

        # Data columns are Sample IDs (with batch suffix)
        sample_cols = [
            "Sample_A1__@__Batch1",
            "Sample_A2__@__Batch1",
            "QC_1__@__Batch1",
            "Ref_1__@__Batch1",
        ]
        result = build_sample_type_map(sample_cols, metadata_df)

        assert result["Sample_A1__@__Batch1"] == "experimental"
        assert result["QC_1__@__Batch1"] == "qc"
        assert result["Ref_1__@__Batch1"] == "reference"

    def test_build_sample_type_map_empty_metadata(self):
        """Test build_sample_type_map with None or empty metadata."""
        from skyline_prism.cli import build_sample_type_map

        sample_cols = ["Sample_A1", "Sample_A2"]

        # None metadata
        assert build_sample_type_map(sample_cols, None) == {}

        # Missing sample_type column
        metadata_df = pd.DataFrame({"sample": ["Sample_A1", "Sample_A2"]})
        assert build_sample_type_map(sample_cols, metadata_df) == {}
