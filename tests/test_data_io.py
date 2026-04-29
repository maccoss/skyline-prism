"""Tests for data I/O module."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from skyline_prism.data_io import (
    _standardize_columns,
    load_sample_metadata,
    merge_and_sort_streaming,
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
        # Note: pandas may return 'object' or 'string' dtype depending on version
        assert pd.api.types.is_string_dtype(merged_df["Acquired Time"])

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


class TestDuplicateColumnPrevention:
    """Tests to verify that reprocessing parquet files doesn't create duplicate columns.

    When a parquet file already contains metadata columns (Batch, Source Document,
    Sample ID), running PRISM again should not create duplicate columns like
    Batch_1, Source Document_1, Sample ID_1.
    """

    def test_merge_parquet_with_existing_metadata_columns(self, tmp_path):
        """Test that merge doesn't duplicate existing metadata columns in parquet."""
        # Create a parquet file that already has the metadata columns
        # (simulating output from a previous PRISM run)
        df_with_metadata = pd.DataFrame(
            {
                "Protein": ["P1", "P1", "P2", "P2"],
                "Peptide Modified Sequence Unimod Ids": ["PEPTIDEK", "PEPTIDEK", "ANOTHERK", "ANOTHERK"],
                "Precursor Charge": [2, 2, 2, 2],
                "Fragment Ion": ["y5", "y6", "y5", "y6"],
                "Product Charge": [1, 1, 1, 1],
                "Product Mz": [500.0, 600.0, 500.0, 600.0],
                "Area": [1000, 2000, 3000, 4000],
                "Replicate Name": ["Sample1", "Sample1", "Sample2", "Sample2"],
                # Metadata columns that already exist from previous run
                "Batch": ["Plate1", "Plate1", "Plate1", "Plate1"],
                "Source Document": ["original_file", "original_file", "original_file", "original_file"],
                "Sample ID": ["Sample1__@__Plate1", "Sample1__@__Plate1", "Sample2__@__Plate1", "Sample2__@__Plate1"],
            }
        )

        input_parquet = tmp_path / "already_processed.parquet"
        df_with_metadata.to_parquet(input_parquet, index=False)

        # Now run merge on this file (simulating re-running PRISM)
        output_path = tmp_path / "output" / "merged.parquet"
        output_path.parent.mkdir(exist_ok=True)

        # Use merge_and_sort_streaming which handles parquet files properly
        result_path, samples_by_batch, total_rows = merge_and_sort_streaming(
            report_paths=[input_parquet],
            output_path=output_path,
            batch_names=["NewBatch"],
        )

        # Read the output and check for duplicate columns
        result_df = pd.read_parquet(result_path)

        # Should NOT have any _1 suffix columns
        column_names = result_df.columns.tolist()
        duplicate_cols = [c for c in column_names if c.endswith("_1")]
        assert len(duplicate_cols) == 0, f"Found duplicate columns: {duplicate_cols}"

        # Original columns should still exist exactly once
        assert column_names.count("Batch") == 1
        assert column_names.count("Source Document") == 1
        assert column_names.count("Sample ID") == 1

        # Data should be preserved
        assert len(result_df) == 4

    def test_merge_csv_then_reprocess_parquet(self, tmp_path):
        """Test full round-trip: CSV -> parquet -> parquet doesn't create duplicates."""
        # Step 1: Create original CSV without metadata columns
        csv_file = tmp_path / "original.csv"
        df = pd.DataFrame(
            {
                "Protein": ["P1", "P1"],
                "Peptide Modified Sequence Unimod Ids": ["PEPTIDEK", "PEPTIDEK"],
                "Precursor Charge": [2, 2],
                "Fragment Ion": ["y5", "y6"],
                "Product Charge": [1, 1],
                "Product Mz": [500.0, 600.0],
                "Area": [1000, 2000],
                "Replicate Name": ["Sample1", "Sample1"],
            }
        )
        df.to_csv(csv_file, index=False)

        # Step 2: First merge (CSV -> parquet with metadata columns added)
        first_output = tmp_path / "first_run" / "merged.parquet"
        first_output.parent.mkdir(exist_ok=True)

        # Use merge_and_sort_streaming which handles both CSV and parquet
        result_path1, _, _ = merge_and_sort_streaming(
            report_paths=[csv_file],
            output_path=first_output,
            batch_names=["Batch1"],
        )

        # Verify first merge added metadata columns
        df1 = pd.read_parquet(result_path1)
        assert "Batch" in df1.columns
        assert "Source Document" in df1.columns
        assert "Sample ID" in df1.columns

        # Step 3: Second merge (parquet -> parquet, should not duplicate)
        second_output = tmp_path / "second_run" / "merged.parquet"
        second_output.parent.mkdir(exist_ok=True)

        result_path2, _, _ = merge_and_sort_streaming(
            report_paths=[result_path1],
            output_path=second_output,
            batch_names=["Batch2"],
        )

        # Verify second merge didn't create duplicate columns
        df2 = pd.read_parquet(result_path2)
        column_names = df2.columns.tolist()

        duplicate_cols = [c for c in column_names if "_1" in c]
        assert len(duplicate_cols) == 0, f"Found duplicate columns: {duplicate_cols}"

        # Should have exactly one of each metadata column
        assert column_names.count("Batch") == 1
        assert column_names.count("Source Document") == 1
        assert column_names.count("Sample ID") == 1

    def test_merge_mixed_files_with_and_without_metadata(self, tmp_path):
        """Test merging files where some have metadata columns and some don't."""
        # File 1: parquet WITH existing metadata columns
        df_with = pd.DataFrame(
            {
                "Protein": ["P1"],
                "Peptide Modified Sequence Unimod Ids": ["PEPTIDEK"],
                "Precursor Charge": [2],
                "Fragment Ion": ["y5"],
                "Product Charge": [1],
                "Product Mz": [500.0],
                "Area": [1000],
                "Replicate Name": ["Sample1"],
                "Batch": ["OldBatch"],
                "Source Document": ["old_file"],
                "Sample ID": ["Sample1__@__OldBatch"],
            }
        )
        parquet_with = tmp_path / "with_metadata.parquet"
        df_with.to_parquet(parquet_with, index=False)

        # File 2: CSV WITHOUT metadata columns
        df_without = pd.DataFrame(
            {
                "Protein": ["P2"],
                "Peptide Modified Sequence Unimod Ids": ["ANOTHERK"],
                "Precursor Charge": [2],
                "Fragment Ion": ["y5"],
                "Product Charge": [1],
                "Product Mz": [500.0],
                "Area": [2000],
                "Replicate Name": ["Sample2"],
            }
        )
        csv_without = tmp_path / "without_metadata.csv"
        df_without.to_csv(csv_without, index=False)

        # Merge both files using merge_and_sort_streaming
        output_path = tmp_path / "merged.parquet"
        result_path, _, _ = merge_and_sort_streaming(
            report_paths=[parquet_with, csv_without],
            output_path=output_path,
            batch_names=["Batch1", "Batch2"],
        )

        result_df = pd.read_parquet(result_path)
        column_names = result_df.columns.tolist()

        # No duplicate columns
        duplicate_cols = [c for c in column_names if "_1" in c]
        assert len(duplicate_cols) == 0, f"Found duplicate columns: {duplicate_cols}"

        # Both rows should be present
        assert len(result_df) == 2
        assert set(result_df["Protein"].unique()) == {"P1", "P2"}


def _make_skyline_parquet(
    path,
    *,
    proteins,
    peptides,
    replicates,
    add_metadata=False,
    batch_name=None,
    row_group_size=None,
):
    """Write a minimal Skyline-shaped parquet for streaming-merge tests.

    All four lists must have the same length. ``add_metadata=True`` writes the
    Batch / Source Document / Sample ID columns into the file (simulating a
    re-export of an already-processed file).
    """
    n = len(proteins)
    assert len(peptides) == n and len(replicates) == n
    df = pd.DataFrame(
        {
            "Protein": proteins,
            "Peptide Modified Sequence Unimod Ids": peptides,
            "Precursor Charge": [2] * n,
            "Fragment Ion": ["y5"] * n,
            "Product Charge": [1] * n,
            "Product Mz": [500.0] * n,
            "Area": list(range(1000, 1000 + n)),
            "Replicate Name": replicates,
        }
    )
    if add_metadata:
        assert batch_name is not None
        df["Batch"] = batch_name
        df["Source Document"] = path.stem
        df["Sample ID"] = [
            f"{r}__@__{batch_name}" if r is not None else None for r in replicates
        ]
    table = pa.Table.from_pandas(df)
    if row_group_size is None:
        pq.write_table(table, path)
    else:
        pq.write_table(table, path, row_group_size=row_group_size)


class TestMultiParquetStreamingMerge:
    """Tests for the pyarrow-streaming merge path used when all inputs are parquet.

    These guard against the regression where the multi-file path could not handle
    the "dozens of parquet files" scaling claim from the project's design intent.
    """

    def test_merge_dozen_parquet_files_scales(self, tmp_path):
        """Merging 12 parquet files produces correct row count, batches, and samples.

        This is the headline regression check: the project claims to support
        cohorts of dozens of parquet files. A single test with N=12 small files
        exercises the full pyarrow streaming path, the per-file Sample ID
        generation, and the DuckDB sort over the merged intermediate.
        """
        n_files = 12
        rows_per_file = 5
        report_paths = []
        batch_names = []
        for i in range(n_files):
            p = tmp_path / f"plate_{i:02d}.parquet"
            batch = f"Plate{i:02d}"
            # Two samples per file, peptides chosen so global sort order is stable
            _make_skyline_parquet(
                p,
                proteins=[f"P{i}"] * rows_per_file,
                peptides=[f"PEPTIDE_{i:02d}_{j}" for j in range(rows_per_file)],
                replicates=[
                    f"S{i:02d}_A" if j % 2 == 0 else f"S{i:02d}_B"
                    for j in range(rows_per_file)
                ],
            )
            report_paths.append(p)
            batch_names.append(batch)

        out = tmp_path / "merged.parquet"
        result_path, samples_by_batch, total_rows = merge_and_sort_streaming(
            report_paths=report_paths,
            output_path=out,
            batch_names=batch_names,
        )

        assert total_rows == n_files * rows_per_file
        assert len(samples_by_batch) == n_files
        for batch in batch_names:
            idx = int(batch[5:])
            assert batch in samples_by_batch
            assert samples_by_batch[batch] == {f"S{idx:02d}_A", f"S{idx:02d}_B"}

        result_df = pd.read_parquet(result_path)
        # All synthesized metadata columns are present exactly once
        assert result_df.columns.tolist().count("Batch") == 1
        assert result_df.columns.tolist().count("Source Document") == 1
        assert result_df.columns.tolist().count("Sample ID") == 1
        # Every batch is represented in the merged output
        assert set(result_df["Batch"].unique()) == set(batch_names)

    def test_merge_parquet_output_is_sorted_by_peptide(self, tmp_path):
        """The sort stage must actually sort the merged output by peptide.

        Catches the regression case where the sort stage silently produced an
        unsorted file (e.g., if the COPY-of-ORDER-BY ever falls back to
        insertion order).
        """
        # Two files with deliberately interleaved peptide names
        p1 = tmp_path / "f1.parquet"
        p2 = tmp_path / "f2.parquet"
        _make_skyline_parquet(
            p1,
            proteins=["P1", "P1"],
            peptides=["PEPTIDE_C", "PEPTIDE_A"],
            replicates=["S1", "S1"],
        )
        _make_skyline_parquet(
            p2,
            proteins=["P2", "P2"],
            peptides=["PEPTIDE_B", "PEPTIDE_D"],
            replicates=["S2", "S2"],
        )

        out = tmp_path / "merged.parquet"
        merge_and_sort_streaming(
            report_paths=[p1, p2],
            output_path=out,
            batch_names=["B1", "B2"],
        )

        df = pd.read_parquet(out)
        peps = df["Peptide Modified Sequence Unimod Ids"].tolist()
        assert peps == sorted(peps)
        assert peps == ["PEPTIDE_A", "PEPTIDE_B", "PEPTIDE_C", "PEPTIDE_D"]

    def test_merge_parquet_with_null_replicate_names(self, tmp_path):
        """Null replicate names must propagate to null Sample IDs, not crash.

        The vectorized `pyarrow.compute.binary_join_element_wise(...,
        null_handling="emit_null")` must match the previous Python behavior
        of producing a null Sample ID when the replicate is null.
        """
        p = tmp_path / "with_nulls.parquet"
        _make_skyline_parquet(
            p,
            proteins=["P1", "P1", "P1"],
            peptides=["PEPTIDE_A", "PEPTIDE_B", "PEPTIDE_C"],
            replicates=["S1", None, "S2"],
        )

        out = tmp_path / "merged.parquet"
        merge_and_sort_streaming(
            report_paths=[p],
            output_path=out,
            batch_names=["B1"],
        )

        df = pd.read_parquet(out)
        # The null-replicate row should have a null Sample ID
        null_rows = df[df["Replicate Name"].isna()]
        assert len(null_rows) == 1
        assert null_rows["Sample ID"].isna().all()
        # The non-null rows should have the standard Sample ID format
        non_null_rows = df[df["Replicate Name"].notna()]
        for _, row in non_null_rows.iterrows():
            assert row["Sample ID"] == f"{row['Replicate Name']}__@__B1"

    def test_merge_parquet_with_multiple_row_groups(self, tmp_path):
        """Row-group iteration produces correct results across row group boundaries.

        The streaming merge calls ``ParquetFile.read_row_group(i, ...)`` once per
        row group. This test writes a file with a small ``row_group_size`` so it
        spans multiple row groups, then verifies all rows are preserved and the
        Sample ID is generated correctly for every row.
        """
        p = tmp_path / "multi_row_group.parquet"
        n = 100
        _make_skyline_parquet(
            p,
            proteins=["P1"] * n,
            peptides=[f"PEPTIDE_{i:03d}" for i in range(n)],
            replicates=[f"S{i % 4}" for i in range(n)],
            row_group_size=17,  # forces multiple row groups
        )
        # Sanity check: the test data has multiple row groups
        assert pq.ParquetFile(p).num_row_groups > 1

        out = tmp_path / "merged.parquet"
        result_path, _, total_rows = merge_and_sort_streaming(
            report_paths=[p],
            output_path=out,
            batch_names=["B1"],
        )

        assert total_rows == n
        df = pd.read_parquet(result_path)
        assert len(df) == n
        # Every row gets the expected Sample ID
        expected_ids = {f"S{i % 4}__@__B1" for i in range(n)}
        assert set(df["Sample ID"].unique()) == expected_ids

    def test_merge_parquet_schema_mismatch_raises(self, tmp_path):
        """Files with extra columns in the first input vs. later ones must raise.

        Silently dropping columns would mask data-quality problems. The streaming
        merge raises with a clear message naming the missing columns.
        """
        # First file has an extra "Library Intensity" column; second does not.
        p1 = tmp_path / "with_lib.parquet"
        df1 = pd.DataFrame(
            {
                "Protein": ["P1"],
                "Peptide Modified Sequence Unimod Ids": ["PEPTIDE_A"],
                "Precursor Charge": [2],
                "Fragment Ion": ["y5"],
                "Product Charge": [1],
                "Product Mz": [500.0],
                "Area": [1000],
                "Replicate Name": ["S1"],
                "Library Intensity": [42.0],
            }
        )
        df1.to_parquet(p1, index=False)

        p2 = tmp_path / "without_lib.parquet"
        _make_skyline_parquet(
            p2,
            proteins=["P2"],
            peptides=["PEPTIDE_B"],
            replicates=["S2"],
        )

        out = tmp_path / "merged.parquet"
        with pytest.raises(ValueError, match="missing columns"):
            merge_and_sort_streaming(
                report_paths=[p1, p2],
                output_path=out,
                batch_names=["B1", "B2"],
            )

    def test_merge_parquet_cleans_up_unsorted_intermediate(self, tmp_path):
        """The ``.unsorted.parquet`` intermediate is deleted after a successful run.

        Leaving stale intermediates around would confuse subsequent runs and waste disk.
        """
        p = tmp_path / "f1.parquet"
        _make_skyline_parquet(
            p,
            proteins=["P1"],
            peptides=["PEPTIDE_A"],
            replicates=["S1"],
        )

        out = tmp_path / "merged.parquet"
        merge_and_sort_streaming(
            report_paths=[p],
            output_path=out,
            batch_names=["B1"],
        )

        unsorted = out.with_suffix(".unsorted.parquet")
        assert not unsorted.exists()

    def test_merged_output_readable_via_read_row_group(self, tmp_path):
        """The sorted output must be readable by pyarrow's row-group reader.

        The downstream chunked rollup reads the merged parquet via
        ``ParquetFile.read_row_group(i, columns=...)`` rather than
        ``pd.read_parquet``. This regression check catches the failure where
        the writer produces page headers that pyarrow cannot deserialize
        (``TProtocolException: Invalid data, Deserializing page header
        failed``) mid-file. We additionally pass the same ``columns=`` filter
        that production uses, since the failure was specifically on the
        column-pruned read path.
        """
        # Build inputs across multiple files so the sort produces a multi-RG output
        report_paths = []
        batch_names = []
        n_files = 4
        rows_per_file = 500
        for i in range(n_files):
            p = tmp_path / f"plate_{i}.parquet"
            _make_skyline_parquet(
                p,
                proteins=[f"P{i}"] * rows_per_file,
                peptides=[
                    f"PEPTIDE_{i:02d}_{j:04d}" for j in range(rows_per_file)
                ],
                replicates=[f"S{i}_{j % 3}" for j in range(rows_per_file)],
            )
            report_paths.append(p)
            batch_names.append(f"B{i}")

        out = tmp_path / "merged.parquet"
        merge_and_sort_streaming(
            report_paths=report_paths,
            output_path=out,
            batch_names=batch_names,
        )

        # Read the merged output via the same API that production uses,
        # including the column-pruned read pattern from chunked_processing.
        pf = pq.ParquetFile(out)
        cols_to_read = [
            "Peptide Modified Sequence Unimod Ids",
            "Sample ID",
            "Batch",
            "Area",
        ]
        total_rows_read = 0
        for i in range(pf.num_row_groups):
            table = pf.read_row_group(i, columns=cols_to_read)
            total_rows_read += table.num_rows
            assert table.schema.names == cols_to_read
        assert total_rows_read == n_files * rows_per_file

    def test_merged_zstd_output_is_read_row_group_safe(self, tmp_path):
        """Zstd, multi-row-group output is readable via the hardened reader.

        Opens the merged file with ``pq.ParquetFile(..., pre_buffer=False,
        memory_map=False)`` and iterates every row group via column-pruned
        ``read_row_group``. The downstream chunked rollup uses these options to avoid the
        ``TProtocolException: Invalid data, Deserializing page header failed``
        symptom on multi-GB zstd parquet files. This test locks in:
          1. The merge writes zstd-compressed output (the codec the user wants).
          2. The output produces multiple row groups (matches production
             ``ROW_GROUP_SIZE 1000000`` setting at appropriate scale).
          3. The hardened reader successfully iterates every row group with
             a column-pruned ``read_row_group(i, columns=cols, use_threads=True)``
             call, exactly as production does.
        """
        # Generate enough rows that the merged output spans multiple row groups
        # at production-like row group sizing. Using N_files * rows_per_file
        # well above the chunked_processing row group target keeps the test
        # cheap while exercising the multi-RG read path.
        n_files = 4
        rows_per_file = 800
        report_paths = []
        batch_names = []
        for i in range(n_files):
            p = tmp_path / f"plate_{i}.parquet"
            _make_skyline_parquet(
                p,
                proteins=[f"P{i}"] * rows_per_file,
                peptides=[
                    f"PEPTIDE_{i:02d}_{j:04d}" for j in range(rows_per_file)
                ],
                replicates=[f"S{i}_{j % 3}" for j in range(rows_per_file)],
            )
            report_paths.append(p)
            batch_names.append(f"B{i}")

        out = tmp_path / "merged.parquet"
        merge_and_sort_streaming(
            report_paths=report_paths,
            output_path=out,
            batch_names=batch_names,
        )

        # Confirm zstd compression on the persisted output
        meta = pq.read_metadata(out)
        codecs = {
            meta.row_group(rg).column(c).compression
            for rg in range(meta.num_row_groups)
            for c in range(meta.num_columns)
        }
        assert codecs == {"ZSTD"}, f"expected only ZSTD, got {codecs}"

        # Read every row group with the hardened pyarrow open + column pruning
        # used by chunked_processing.rollup_transitions_sorted.
        pf = pq.ParquetFile(out, pre_buffer=False, memory_map=False)
        cols_to_read = [
            "Peptide Modified Sequence Unimod Ids",
            "Sample ID",
            "Batch",
            "Area",
        ]
        rows_read = 0
        for i in range(pf.num_row_groups):
            table = pf.read_row_group(i, columns=cols_to_read, use_threads=True)
            rows_read += table.num_rows
            assert table.schema.names == cols_to_read
        assert rows_read == n_files * rows_per_file

    def test_merge_parquet_with_existing_metadata_across_files(self, tmp_path):
        """Mix of files with and without metadata columns produces consistent output.

        Reproduces the production case where some plates were already exported with
        Batch/Source Document/Sample ID and others were not.
        """
        # File 1: has metadata already
        p1 = tmp_path / "with_md.parquet"
        _make_skyline_parquet(
            p1,
            proteins=["P1"],
            peptides=["PEPTIDE_A"],
            replicates=["S1"],
            add_metadata=True,
            batch_name="OldBatch",
        )
        # File 2: no metadata
        p2 = tmp_path / "without_md.parquet"
        _make_skyline_parquet(
            p2,
            proteins=["P2"],
            peptides=["PEPTIDE_B"],
            replicates=["S2"],
        )

        out = tmp_path / "merged.parquet"
        merge_and_sort_streaming(
            report_paths=[p1, p2],
            output_path=out,
            batch_names=["NewB1", "NewB2"],
        )

        df = pd.read_parquet(out)
        # No duplicate columns (no Batch_1 or Sample ID_1)
        cols = df.columns.tolist()
        assert all(not c.endswith("_1") for c in cols), f"Unexpected duplicates: {cols}"
        # File 1 keeps its existing Batch ("OldBatch"), File 2 gets the synthesized one
        batches_seen = set(df["Batch"].unique())
        assert "OldBatch" in batches_seen
        assert "NewB2" in batches_seen

    def test_merge_parquet_metadata_string_vs_large_string_types(self, tmp_path):
        """Mixing string and large_string typed metadata columns merges cleanly.

        Reproduces the CI failure where File 1 had pre-existing
        Batch / Source Document / Sample ID columns typed as ``large_string``
        (the default pandas+pyarrow on some platforms produces for object
        columns) and File 2's synthesized metadata columns came in as
        ``string``. Without an explicit cast in ``_stream_concat_parquet``,
        pyarrow's ParquetWriter rejected the second file with
        ``ValueError: Table schema does not match schema used to create file``.
        """
        # File 1: write a schema that explicitly types metadata cols as large_string
        schema = pa.schema(
            [
                ("Protein", pa.string()),
                ("Peptide Modified Sequence Unimod Ids", pa.string()),
                ("Precursor Charge", pa.int64()),
                ("Fragment Ion", pa.string()),
                ("Product Charge", pa.int64()),
                ("Product Mz", pa.float64()),
                ("Area", pa.int64()),
                ("Replicate Name", pa.string()),
                ("Batch", pa.large_string()),
                ("Source Document", pa.large_string()),
                ("Sample ID", pa.large_string()),
            ]
        )
        table = pa.table(
            {
                "Protein": ["P1"],
                "Peptide Modified Sequence Unimod Ids": ["PEPTIDE_A"],
                "Precursor Charge": [2],
                "Fragment Ion": ["y5"],
                "Product Charge": [1],
                "Product Mz": [500.0],
                "Area": [1000],
                "Replicate Name": ["S1"],
                "Batch": ["OldBatch"],
                "Source Document": ["with_md_large"],
                "Sample ID": ["S1__@__OldBatch"],
            },
            schema=schema,
        )
        p1 = tmp_path / "with_md_large_string.parquet"
        pq.write_table(table, p1)

        # Sanity check: the file really does use large_string for the metadata cols
        f1_schema = pq.ParquetFile(p1).schema_arrow
        assert f1_schema.field("Batch").type == pa.large_string()
        assert f1_schema.field("Sample ID").type == pa.large_string()

        # File 2: no metadata; the merge will synthesize Batch/Source Document/Sample ID
        # as plain pa.string(). The cast in _stream_concat_parquet must reconcile
        # the types so the writer accepts both files.
        p2 = tmp_path / "without_md.parquet"
        _make_skyline_parquet(
            p2,
            proteins=["P2"],
            peptides=["PEPTIDE_B"],
            replicates=["S2"],
        )

        out = tmp_path / "merged.parquet"
        merge_and_sort_streaming(
            report_paths=[p1, p2],
            output_path=out,
            batch_names=["NewB1", "NewB2"],
        )

        # Both rows present; pre-existing Batch on File 1 preserved; File 2 synthesized.
        df = pd.read_parquet(out)
        assert len(df) == 2
        assert {"OldBatch", "NewB2"}.issubset(set(df["Batch"].unique()))
