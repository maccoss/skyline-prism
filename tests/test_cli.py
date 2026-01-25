"""Tests for CLI module."""

import json
import tempfile
from pathlib import Path

import pytest

from skyline_prism.cli import (
    _deep_merge,
    find_column,
    load_config,
    load_config_from_provenance,
)


class TestFindColumn:
    """Tests for find_column helper function."""

    def test_exact_match(self):
        """Test that exact matches are returned."""
        available = {"Area", "Fragment_Ion", "Peptide"}
        assert find_column(available, "Area") == "Area"
        assert find_column(available, "Fragment_Ion") == "Fragment_Ion"

    def test_space_to_underscore(self):
        """Test that space variant finds underscore column."""
        available = {"Fragment_Ion", "Area", "Protein_Accession"}
        assert find_column(available, "Fragment Ion") == "Fragment_Ion"
        assert find_column(available, "Protein Accession") == "Protein_Accession"

    def test_underscore_to_space(self):
        """Test that underscore variant finds space column."""
        available = {"Fragment Ion", "Area", "Protein Accession"}
        assert find_column(available, "Fragment_Ion") == "Fragment Ion"
        assert find_column(available, "Protein_Accession") == "Protein Accession"

    def test_multiple_candidates(self):
        """Test that first matching candidate is returned."""
        available = {"Fragment_Ion", "Area"}
        # First candidate doesn't exist, second does
        assert find_column(available, "NonExistent", "Fragment Ion") == "Fragment_Ion"
        # First candidate exists
        assert find_column(available, "Area", "Fragment Ion") == "Area"

    def test_no_match_returns_none(self):
        """Test that None is returned when no match found."""
        available = {"Area", "Peptide"}
        assert find_column(available, "Fragment Ion") is None
        assert find_column(available, "NonExistent") is None

    def test_mixed_format_columns(self):
        """Test with mixed space/underscore columns."""
        available = {"Fragment_Ion", "Sample ID", "Protein Accession"}
        assert find_column(available, "Fragment Ion") == "Fragment_Ion"
        assert find_column(available, "Sample ID") == "Sample ID"
        assert find_column(available, "Protein_Accession") == "Protein Accession"


class TestDeepMerge:
    """Tests for deep merge utility."""

    def test_simple_merge(self):
        """Test merging flat dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        """Test merging nested dictionaries."""
        base = {
            "section1": {"a": 1, "b": 2},
            "section2": {"c": 3},
        }
        override = {
            "section1": {"b": 20, "d": 4},
            "section3": {"e": 5},
        }
        result = _deep_merge(base, override)
        assert result["section1"] == {"a": 1, "b": 20, "d": 4}
        assert result["section2"] == {"c": 3}
        assert result["section3"] == {"e": 5}

    def test_override_non_dict_with_dict(self):
        """Test that dict values replace non-dict values."""
        base = {"a": 1}
        override = {"a": {"nested": True}}
        result = _deep_merge(base, override)
        assert result["a"] == {"nested": True}


class TestLoadConfig:
    """Tests for configuration loading."""

    def test_default_config(self):
        """Test loading default configuration."""
        config = load_config(None)

        # Check key defaults
        assert config["global_normalization"]["method"] == "median"
        assert config["batch_correction"]["enabled"] is True
        assert config["protein_rollup"]["method"] == "median_polish"
        assert config["parsimony"]["shared_peptide_handling"] == "all_groups"

    def test_yaml_override(self):
        """Test loading config from YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
global_normalization:
  method: rt_lowess
  rt_lowess:
    frac: 0.3
protein_rollup:
  method: topn
""")
            f.flush()
            config_path = Path(f.name)

        try:
            config = load_config(config_path)
            assert config["global_normalization"]["method"] == "rt_lowess"
            assert config["global_normalization"]["rt_lowess"]["frac"] == 0.3
            assert config["protein_rollup"]["method"] == "topn"
            # Defaults should be preserved
            assert config["batch_correction"]["enabled"] is True
        finally:
            config_path.unlink()


class TestLoadConfigFromProvenance:
    """Tests for loading configuration from provenance JSON."""

    def test_load_from_provenance(self):
        """Test loading configuration from metadata.json provenance file."""
        provenance = {
            "pipeline_version": "0.1.0",
            "processing_date": "2024-01-15T10:30:00Z",
            "processing_parameters": {
                "data": {
                    "abundance_column": "TotalAreaMs1",
                    "peptide_column": "ModifiedSequence",
                },
                "global_normalization": {
                    "method": "rt_lowess",
                    "rt_lowess": {"frac": 0.25},
                },
                "protein_rollup": {
                    "method": "topn",
                    "topn": {"n": 5},
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(provenance, f)
            f.flush()
            provenance_path = Path(f.name)

        try:
            config, provenance_data = load_config_from_provenance(provenance_path)

            # Check that provenance values are loaded
            assert config["data"]["abundance_column"] == "TotalAreaMs1"
            assert config["data"]["peptide_column"] == "ModifiedSequence"
            assert config["global_normalization"]["method"] == "rt_lowess"
            assert config["global_normalization"]["rt_lowess"]["frac"] == 0.25
            assert config["protein_rollup"]["method"] == "topn"
            assert config["protein_rollup"]["topn"]["n"] == 5

            # Check that defaults are preserved for unspecified settings
            assert config["batch_correction"]["enabled"] is True
            assert config["parsimony"]["shared_peptide_handling"] == "all_groups"
        finally:
            provenance_path.unlink()

    def test_missing_processing_parameters_raises(self):
        """Test that missing processing_parameters raises ValueError."""
        provenance = {
            "pipeline_version": "0.1.0",
            "processing_date": "2024-01-15T10:30:00Z",
            # Missing 'processing_parameters'
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(provenance, f)
            f.flush()
            provenance_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="processing_parameters"):
                load_config_from_provenance(provenance_path)
        finally:
            provenance_path.unlink()

    def test_provenance_preserves_output_settings(self):
        """Test that output settings are preserved from provenance."""
        provenance = {
            "pipeline_version": "0.1.0",
            "processing_date": "2024-01-15T10:30:00Z",
            "processing_parameters": {
                "output": {
                    "format": "csv",
                    "include_residuals": False,
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(provenance, f)
            f.flush()
            provenance_path = Path(f.name)

        try:
            config, provenance_data = load_config_from_provenance(provenance_path)
            assert config["output"]["format"] == "csv"
            assert config["output"]["include_residuals"] is False
        finally:
            provenance_path.unlink()
