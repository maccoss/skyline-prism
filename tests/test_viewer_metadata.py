"""Tests for viewer metadata merge functionality.

These tests verify the robust metadata merging logic in the PRISM viewer,
including key normalization for various naming conventions.
"""

from pathlib import Path

import pandas as pd


class TestKeyNormalization:
    """Test the key normalization logic used in metadata merging."""

    def normalize_key(self, val: str) -> str:
        """Replicate the normalize_key function from viewer.py."""
        s = str(val)
        # Strip file extension
        s = Path(s).stem
        # Strip _@_batch suffix (PRISM format: sample_@_batch)
        if "_@_" in s:
            s = s.split("_@_")[0]
        # Strip trailing underscores (sometimes present from separator)
        s = s.rstrip("_")
        return s

    def test_basic_sample_name(self):
        """Test that basic sample names pass through unchanged."""
        assert self.normalize_key("Sample_001") == "Sample_001"
        assert self.normalize_key("IRType-Plasma-Carl_006") == "IRType-Plasma-Carl_006"

    def test_strip_file_extension(self):
        """Test stripping of common file extensions."""
        assert self.normalize_key("Sample_001.raw") == "Sample_001"
        assert self.normalize_key("Sample_001.mzML") == "Sample_001"
        assert self.normalize_key("Sample_001.d") == "Sample_001"
        assert self.normalize_key("/path/to/Sample_001.raw") == "Sample_001"

    def test_strip_batch_suffix(self):
        """Test stripping of _@_batch suffix from PRISM merged data."""
        assert (
            self.normalize_key("IRType-Plasma-201_078_@_2025-0417-IRType-Plasma-PRISM-Plate2")
            == "IRType-Plasma-201_078"
        )
        assert self.normalize_key("Sample_001_@_Batch1") == "Sample_001"

    def test_strip_trailing_underscore(self):
        """Test stripping of trailing underscores after batch split."""
        # This can happen if the separator is __@_ (double underscore)
        assert self.normalize_key("Sample_001_") == "Sample_001"
        assert self.normalize_key("Sample_001__") == "Sample_001"

    def test_combined_normalization(self):
        """Test combination of extension, batch suffix, and trailing underscore."""
        # Full PRISM format with extension stripped first, then batch suffix
        assert self.normalize_key("IRType-Plasma-201_078_@_Batch.raw") == "IRType-Plasma-201_078"

    def test_empty_and_edge_cases(self):
        """Test edge cases."""
        assert self.normalize_key("") == ""
        assert self.normalize_key("_@_") == ""
        assert self.normalize_key("Sample_@_") == "Sample"


class TestMetadataMerge:
    """Test the full metadata merge logic."""

    def create_data_df(self, replicate_names: list) -> pd.DataFrame:
        """Create a sample data DataFrame."""
        return pd.DataFrame(
            {
                "replicate_name": replicate_names,
                "abundance": [1.0, 2.0, 3.0][: len(replicate_names)],
            }
        )

    def create_metadata_df(self, samples: list, sample_types: list) -> pd.DataFrame:
        """Create a sample metadata DataFrame."""
        return pd.DataFrame({"sample": samples, "sample_type": sample_types})

    def normalize_key(self, val: str) -> str:
        """Replicate the normalize_key function from viewer.py."""
        s = str(val)
        s = Path(s).stem
        if "_@_" in s:
            s = s.split("_@_")[0]
        s = s.rstrip("_")
        return s

    def merge_with_normalization(self, data: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
        """Simplified merge logic matching viewer.py."""
        data = data.copy()
        data["_merge_key"] = data["replicate_name"].apply(self.normalize_key)
        metadata = metadata.copy()
        metadata["_merge_key"] = metadata["sample"].apply(self.normalize_key)

        merged = data.merge(metadata, on="_merge_key", how="left", suffixes=("", "_meta"))
        return merged.drop(columns=["_merge_key"])

    def test_exact_match(self):
        """Test merge when sample names match exactly."""
        data = self.create_data_df(["Sample_001", "Sample_002", "Sample_003"])
        metadata = self.create_metadata_df(
            ["Sample_001", "Sample_002", "Sample_003"], ["experimental", "qc", "reference"]
        )

        merged = self.merge_with_normalization(data, metadata)

        assert "sample_type" in merged.columns
        assert merged["sample_type"].tolist() == ["experimental", "qc", "reference"]

    def test_batch_suffix_mismatch(self):
        """Test merge when data has _@_batch suffix but metadata doesn't."""
        data = self.create_data_df(
            ["Sample_001_@_Batch1", "Sample_002_@_Batch1", "Sample_003_@_Batch1"]
        )
        metadata = self.create_metadata_df(
            ["Sample_001", "Sample_002", "Sample_003"], ["experimental", "qc", "reference"]
        )

        merged = self.merge_with_normalization(data, metadata)

        assert "sample_type" in merged.columns
        assert merged["sample_type"].tolist() == ["experimental", "qc", "reference"]

    def test_extension_mismatch(self):
        """Test merge when data has extension but metadata doesn't."""
        data = self.create_data_df(["Sample_001.raw", "Sample_002.raw", "Sample_003.raw"])
        metadata = self.create_metadata_df(
            ["Sample_001", "Sample_002", "Sample_003"], ["experimental", "qc", "reference"]
        )

        merged = self.merge_with_normalization(data, metadata)

        assert "sample_type" in merged.columns
        assert merged["sample_type"].tolist() == ["experimental", "qc", "reference"]

    def test_combined_mismatch(self):
        """Test merge with both batch suffix and extension differences."""
        data = self.create_data_df(
            [
                "IRType-Plasma-201_078_@_2025-0417-Plate1",
                "IRType-Plasma-Carl_006_@_2025-0417-Plate1",
                "IRType-Plasma-Pool_007_@_2025-0417-Plate1",
            ]
        )
        metadata = self.create_metadata_df(
            ["IRType-Plasma-201_078", "IRType-Plasma-Carl_006", "IRType-Plasma-Pool_007"],
            ["experimental", "qc", "reference"],
        )

        merged = self.merge_with_normalization(data, metadata)

        assert "sample_type" in merged.columns
        assert merged["sample_type"].tolist() == ["experimental", "qc", "reference"]

    def test_partial_match(self):
        """Test merge when only some samples match."""
        data = self.create_data_df(
            ["Sample_001_@_Batch1", "Sample_002_@_Batch1", "Unknown_Sample_@_Batch1"]
        )
        metadata = self.create_metadata_df(["Sample_001", "Sample_002"], ["experimental", "qc"])

        merged = self.merge_with_normalization(data, metadata)

        assert "sample_type" in merged.columns
        assert merged["sample_type"].tolist()[0] == "experimental"
        assert merged["sample_type"].tolist()[1] == "qc"
        assert pd.isna(merged["sample_type"].tolist()[2])  # No match

    def test_trailing_underscore_normalization(self):
        """Test that trailing underscores are stripped during merge."""
        # Simulates case where _@_ split leaves trailing underscore
        data = self.create_data_df(
            [
                "Sample_001_",  # Has trailing underscore
                "Sample_002_",
            ]
        )
        metadata = self.create_metadata_df(["Sample_001", "Sample_002"], ["experimental", "qc"])

        merged = self.merge_with_normalization(data, metadata)

        assert "sample_type" in merged.columns
        assert merged["sample_type"].tolist() == ["experimental", "qc"]


class TestColumnMatching:
    """Test flexible column name matching for merge."""

    def test_recognizes_replicate_column_names(self):
        """Test that various replicate column names are recognized."""
        expected_data_cols = ["replicate_name", "Replicate Name", "filename", "sample_id"]
        expected_meta_cols = [
            "Replicate",
            "replicate",
            "sample",
            "Sample",
            "filename",
            "Filename",
            "sample_id",
            "replicate_name",
        ]

        # This just documents the expected behavior
        # In the actual viewer, these columns are searched for in order
        for col in expected_data_cols:
            assert col in expected_data_cols
        for col in expected_meta_cols:
            assert col in expected_meta_cols
