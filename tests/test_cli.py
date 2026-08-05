"""Tests for CLI module."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from skyline_prism.cli import (
    _deep_merge,
    find_column,
    load_config,
    load_config_from_provenance,
    resolve_library_assist_config,
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


class TestResolveLibraryAssistConfig:
    """Tests for library-assisted rollup config resolution across spellings."""

    def test_defaults(self):
        """Test defaults when nothing is configured."""
        resolved = resolve_library_assist_config({"method": "library_assist"})

        assert resolved["library_path"] is None
        assert resolved["min_fragments"] == 3
        assert resolved["mz_tolerance"] == 0.02
        assert resolved["outlier_threshold"] == 1.0
        assert resolved["remove_outliers"] is True
        assert resolved["fitting_method"] == "median_polish"

    def test_flat_csharp_keys(self):
        """Test that the C# engine's flat library_* keys are honored."""
        resolved = resolve_library_assist_config(
            {
                "method": "library_assist",
                "library_path": "lib.blib",
                "library_min_fragments": 4,
                "library_mz_tolerance": 0.05,
                "library_outlier_threshold": 2.0,
                "library_remove_outliers": False,
                "library_fitting_method": "least_squares",
            }
        )

        assert resolved["library_path"] == "lib.blib"
        assert resolved["min_fragments"] == 4
        assert resolved["mz_tolerance"] == 0.05
        assert resolved["outlier_threshold"] == 2.0
        assert resolved["remove_outliers"] is False
        assert resolved["fitting_method"] == "least_squares"

    def test_empty_nested_block_with_flat_keys(self):
        """Test that a bare 'library_assist:' (YAML None) does not shadow flat keys."""
        transition_config = yaml.safe_load("""
method: library_assist
library_path: lib.blib
library_outlier_threshold: 1
library_remove_outliers: true
library_assist:
""")
        assert transition_config["library_assist"] is None

        resolved = resolve_library_assist_config(transition_config)

        assert resolved["library_path"] == "lib.blib"
        assert resolved["outlier_threshold"] == 1
        assert resolved["remove_outliers"] is True

    def test_nested_block_wins(self):
        """Test that the nested block overrides flat keys (matches the C# port)."""
        resolved = resolve_library_assist_config(
            {
                "library_path": "flat.blib",
                "library_min_fragments": 4,
                "library_assist": {
                    "library_path": "nested.blib",
                    "min_matched_fragments": 6,
                },
            }
        )

        assert resolved["library_path"] == "nested.blib"
        assert resolved["min_fragments"] == 6

    def test_legacy_aliases(self):
        """Test that the legacy spectral_library_* aliases still resolve."""
        resolved = resolve_library_assist_config(
            {
                "spectral_library_path": "legacy.blib",
                "spectral_library_min_fragments": 5,
                "spectral_library_mz_tolerance": 0.01,
                "spectral_library_outlier_threshold": 3.0,
            }
        )

        assert resolved["library_path"] == "legacy.blib"
        assert resolved["min_fragments"] == 5
        assert resolved["mz_tolerance"] == 0.01
        assert resolved["outlier_threshold"] == 3.0

    def test_explicit_zero_is_not_replaced_by_the_default(self):
        """Test that a value of 0 is honored rather than treated as unset.

        The old resolution chain used `config.get(key) or default`, so a deliberate
        `library_outlier_threshold: 0` (flag every positive residual as interference) silently
        became 1.0. Zero is a meaningful setting here, not an absent one.
        """
        resolved = resolve_library_assist_config(
            {
                "library_path": "lib.blib",
                "library_outlier_threshold": 0,
                "library_mz_tolerance": 0.0,
                "library_min_fragments": 0,
            }
        )

        assert resolved["outlier_threshold"] == 0
        assert resolved["mz_tolerance"] == 0.0
        assert resolved["min_fragments"] == 0

    def test_explicit_false_is_not_replaced_by_the_default(self):
        """Test that remove_outliers: false survives (same truthiness trap as zero)."""
        assert (
            resolve_library_assist_config(
                {"library_path": "lib.blib", "library_remove_outliers": False}
            )["remove_outliers"]
            is False
        )

    def test_flat_wins_over_legacy(self):
        """Test precedence: nested > flat library_* > legacy spectral_library_*."""
        resolved = resolve_library_assist_config(
            {
                "spectral_library_path": "legacy.blib",
                "library_path": "flat.blib",
            }
        )

        assert resolved["library_path"] == "flat.blib"

    def test_config_emitted_by_the_csharp_tool_loads_here(self):
        """Test that a config produced by the Skyline tool runs on the Python engine.

        The fixture is the verbatim output of the C# ConfigWriter (pinned on that side by
        ConfigWriterTests.EmittedYaml_MatchesTheCrossEngineFixture). Checking it here is what
        catches a key the C# engine emits but the Python schema does not know - C#'s own
        validation cannot see that, and it is the failure that made a tool-authored config
        unusable with the Python CLI.
        """
        fixture = (
            Path(__file__).resolve().parent.parent
            / "dotnet"
            / "tests"
            / "fixtures"
            / "config"
            / "emitted-library-assist.yaml"
        )
        if not fixture.exists():
            pytest.skip(f"C# fixture not present: {fixture}")

        config = load_config(fixture)

        # The only keys Python may legitimately not recognize are the ones docs/parameters.md
        # records as C#-only. Anything else means a key was added on the C# side without a
        # decision about the Python side - port it, or record it here and in parameters.md.
        csharp_only = {"batch_correction.peptide_level", "batch_correction.protein_level"}
        assert set(config["_unknown_keys"]) <= csharp_only, (
            f"unexpected keys unknown to the Python engine: "
            f"{sorted(set(config['_unknown_keys']) - csharp_only)}"
        )

        resolved = resolve_library_assist_config(config["transition_rollup"])
        assert resolved["library_path"] == "spectra.blib"
        assert resolved["min_fragments"] == 3
        assert resolved["mz_tolerance"] == 0.02
        assert resolved["outlier_threshold"] == 1
        assert resolved["remove_outliers"] is True
        assert resolved["fitting_method"] == "median_polish"

    def test_csharp_style_config_has_no_unknown_keys(self):
        """Test that a C#-flavored transition_rollup block validates cleanly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
transition_rollup:
  method: library_assist
  min_transitions: 3
  use_ms1: false
  library_path: lib.blib
  library_min_fragments: 3
  library_mz_tolerance: 0.02
  library_outlier_threshold: 1
  library_remove_outliers: true
  library_fitting_method: median_polish
""")
            f.flush()
            config_path = Path(f.name)

        try:
            config = load_config(config_path)
            assert config["_unknown_keys"] == []
            assert resolve_library_assist_config(config["transition_rollup"])["library_path"] == (
                "lib.blib"
            )
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
