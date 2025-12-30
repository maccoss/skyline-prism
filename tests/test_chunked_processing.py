"""Tests for chunked/streaming processing module.

Tests the parallel processing functionality and memory-efficient
streaming rollup operations.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from skyline_prism.chunked_processing import (
    ChunkedRollupConfig,
    ProteinRollupConfig,
    _process_peptide_batch,
    _process_single_peptide,
    _worker_process_batch,
    get_unique_values_from_parquet,
    rollup_proteins_streaming,
    rollup_transitions_sorted,
)
from skyline_prism.parsimony import ProteinGroup


class TestChunkedRollupConfig:
    """Test configuration dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ChunkedRollupConfig()
        assert config.n_workers == 1
        assert config.peptide_batch_size == 1000
        assert config.max_memory_gb == 8.0
        assert config.method == "sum"

    def test_custom_workers(self):
        """Test setting custom worker count."""
        config = ChunkedRollupConfig(n_workers=4)
        assert config.n_workers == 4


class TestProcessSinglePeptide:
    """Test single peptide processing."""

    @pytest.fixture
    def sample_peptide_data(self):
        """Create sample transition data for one peptide."""
        return pd.DataFrame({
            "Fragment Ion": ["y3", "y4", "y5"] * 3,
            "Precursor Charge": [2, 2, 2] * 3,  # Same precursor charge
            "Product Charge": [1, 1, 1] * 3,   # Same product charge
            "Replicate Name": ["S1", "S1", "S1", "S2", "S2", "S2", "S3", "S3", "S3"],
            "Area": [1000, 2000, 1500, 1100, 2100, 1600, 900, 1900, 1400],
            "Shape Correlation": [0.9, 0.95, 0.85] * 3,
        })

    def test_sum_method(self, sample_peptide_data):
        """Test sum rollup method."""
        config = ChunkedRollupConfig(
            transition_col="Fragment Ion",
            sample_col="Replicate Name",
            abundance_col="Area",
            method="sum",
        )
        samples = ["S1", "S2", "S3"]
        
        result = _process_single_peptide(
            sample_peptide_data, "PEPTIDEK", samples, config
        )
        
        assert result.peptide == "PEPTIDEK"
        assert result.n_transitions == 3
        assert len(result.abundances) == 3
        # Sum should be log2(1000 + 2000 + 1500) = log2(4500) ~ 12.14
        assert result.abundances["S1"] > 12.0

    def test_median_polish_method(self, sample_peptide_data):
        """Test median polish rollup method."""
        config = ChunkedRollupConfig(
            transition_col="Fragment Ion",
            sample_col="Replicate Name",
            abundance_col="Area",
            method="median_polish",
            min_transitions=3,
        )
        samples = ["S1", "S2", "S3"]
        
        result = _process_single_peptide(
            sample_peptide_data, "PEPTIDEK", samples, config
        )
        
        assert result.peptide == "PEPTIDEK"
        assert result.n_transitions == 3
        assert result.residuals is not None  # Median polish returns residuals


class TestWorkerProcessBatch:
    """Test the parallel worker function."""

    @pytest.fixture
    def sample_batch(self):
        """Create a batch of peptide data."""
        pep1_data = pd.DataFrame({
            "Fragment Ion": ["y3", "y4", "y5"] * 2,
            "Precursor Charge": [2, 2, 2] * 2,
            "Product Charge": [1, 1, 1] * 2,
            "Replicate Name": ["S1", "S1", "S1", "S2", "S2", "S2"],
            "Area": [1000, 2000, 1500, 1100, 2100, 1600],
        })
        pep2_data = pd.DataFrame({
            "Fragment Ion": ["y3", "y4", "y5"] * 2,
            "Precursor Charge": [2, 2, 2] * 2,
            "Product Charge": [1, 1, 1] * 2,
            "Replicate Name": ["S1", "S1", "S1", "S2", "S2", "S2"],
            "Area": [500, 800, 600, 550, 850, 650],
        })
        return {"PEPTIDEK": pep1_data, "ANOTHERK": pep2_data}

    def test_worker_returns_dicts(self, sample_batch):
        """Test that worker returns picklable dictionaries."""
        config_dict = {
            "peptide_col": "Peptide Modified Sequence",
            "transition_col": "Fragment Ion",
            "precursor_charge_col": "Precursor Charge",
            "product_charge_col": "Product Charge",
            "sample_col": "Replicate Name",
            "abundance_col": "Area",
            "shape_corr_col": "Shape Correlation",
            "coeluting_col": "Coeluting",
            "rt_col": "Retention Time",
            "batch_col": "Batch",
            "mz_col": "Product Mz",
            "method": "sum",
            "min_transitions": 3,
            "log_transform": True,
            "adaptive_params": None,
            "exclude_precursor": True,
            "topn_count": 3,
            "topn_selection": "correlation",
            "topn_weighting": "sqrt",
            "n_workers": 1,
            "peptide_batch_size": 1000,
            "progress_interval": 10000,
            "max_memory_gb": 8.0,
        }
        samples = ["S1", "S2"]
        
        results = _worker_process_batch((sample_batch, samples, config_dict))
        
        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)
        assert all("peptide" in r for r in results)
        assert all("abundances" in r for r in results)

    def test_worker_handles_empty_batch(self):
        """Test worker with empty batch."""
        config_dict = {
            "peptide_col": "Peptide Modified Sequence",
            "transition_col": "Fragment Ion",
            "precursor_charge_col": "Precursor Charge",
            "product_charge_col": "Product Charge",
            "sample_col": "Replicate Name",
            "abundance_col": "Area",
            "shape_corr_col": "Shape Correlation",
            "coeluting_col": "Coeluting",
            "rt_col": "Retention Time",
            "batch_col": "Batch",
            "mz_col": "Product Mz",
            "method": "sum",
            "min_transitions": 3,
            "log_transform": True,
            "adaptive_params": None,
            "exclude_precursor": True,
            "topn_count": 3,
            "topn_selection": "correlation",
            "topn_weighting": "sqrt",
            "n_workers": 1,
            "peptide_batch_size": 1000,
            "progress_interval": 10000,
            "max_memory_gb": 8.0,
        }
        samples = ["S1", "S2"]
        
        results = _worker_process_batch(({}, samples, config_dict))
        assert results == []


class TestRollupTransitionsSorted:
    """Test the main sorted rollup function."""

    @pytest.fixture
    def temp_parquet_file(self):
        """Create a temporary parquet file with transition data."""
        # Create sample data with multiple peptides
        data = []
        for pep_idx in range(10):
            peptide = f"PEPTIDE{pep_idx}K"
            for trans in ["y3", "y4", "y5"]:
                for sample in ["Sample1", "Sample2", "Sample3"]:
                    data.append({
                        "Peptide Modified Sequence": peptide,
                        "Fragment Ion": trans,
                        "Precursor Charge": 2,
                        "Product Charge": 1,
                        "Replicate Name": sample,
                        "Area": np.random.uniform(1000, 5000),
                        "Retention Time": np.random.uniform(10, 20),
                    })
        
        df = pd.DataFrame(data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "test_transitions.parquet"
            df.to_parquet(parquet_path, index=False)
            yield parquet_path

    def test_single_worker_rollup(self, temp_parquet_file):
        """Test rollup with single worker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "peptides.parquet"
            config = ChunkedRollupConfig(
                peptide_col="Peptide Modified Sequence",
                transition_col="Fragment Ion",
                sample_col="Replicate Name",
                abundance_col="Area",
                method="sum",
                n_workers=1,
                min_transitions=2,
            )
            
            result = rollup_transitions_sorted(
                temp_parquet_file,
                output_path,
                config,
                save_residuals=False,
            )
            
            assert result.n_peptides == 10
            assert result.n_samples == 3
            assert output_path.exists()
            
            # Verify output
            output_df = pd.read_parquet(output_path)
            assert len(output_df) == 10
            assert "Sample1" in output_df.columns

    def test_parallel_rollup(self, temp_parquet_file):
        """Test rollup with multiple workers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "peptides.parquet"
            config = ChunkedRollupConfig(
                peptide_col="Peptide Modified Sequence",
                transition_col="Fragment Ion",
                sample_col="Replicate Name",
                abundance_col="Area",
                method="sum",
                n_workers=2,  # Use 2 workers
                peptide_batch_size=3,  # Small batch for testing
                min_transitions=2,
            )
            
            result = rollup_transitions_sorted(
                temp_parquet_file,
                output_path,
                config,
                save_residuals=False,
            )
            
            assert result.n_peptides == 10
            assert result.n_samples == 3
            assert output_path.exists()
            
            # Verify output matches single-worker result
            output_df = pd.read_parquet(output_path)
            assert len(output_df) == 10

    def test_parallel_vs_single_worker_consistency(self, temp_parquet_file):
        """Test that parallel and single-worker produce same results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Single worker
            output_single = Path(tmpdir) / "peptides_single.parquet"
            config_single = ChunkedRollupConfig(
                peptide_col="Peptide Modified Sequence",
                transition_col="Fragment Ion",
                sample_col="Replicate Name",
                abundance_col="Area",
                method="sum",
                n_workers=1,
                min_transitions=2,
            )
            
            rollup_transitions_sorted(
                temp_parquet_file,
                output_single,
                config_single,
                save_residuals=False,
            )
            
            # Parallel workers
            output_parallel = Path(tmpdir) / "peptides_parallel.parquet"
            config_parallel = ChunkedRollupConfig(
                peptide_col="Peptide Modified Sequence",
                transition_col="Fragment Ion",
                sample_col="Replicate Name",
                abundance_col="Area",
                method="sum",
                n_workers=2,
                peptide_batch_size=3,
                min_transitions=2,
            )
            
            rollup_transitions_sorted(
                temp_parquet_file,
                output_parallel,
                config_parallel,
                save_residuals=False,
            )
            
            # Compare results
            df_single = pd.read_parquet(output_single).sort_values(
                "Peptide Modified Sequence"
            ).reset_index(drop=True)
            df_parallel = pd.read_parquet(output_parallel).sort_values(
                "Peptide Modified Sequence"
            ).reset_index(drop=True)
            
            assert len(df_single) == len(df_parallel)
            
            # Check peptides match
            assert list(df_single["Peptide Modified Sequence"]) == list(
                df_parallel["Peptide Modified Sequence"]
            )
            
            # Check abundances are close (floating point comparison)
            for sample in ["Sample1", "Sample2", "Sample3"]:
                np.testing.assert_array_almost_equal(
                    df_single[sample].values,
                    df_parallel[sample].values,
                    decimal=10,
                )


class TestOutputLinearScale:
    """Test that parquet outputs are always in LINEAR scale, not log2.
    
    This is a critical invariant for the PRISM pipeline. All internal 
    processing uses log2 scale, but ALL parquet output files must contain
    LINEAR scale values.
    
    Failure to maintain this invariant causes downstream analysis errors
    since values like 10-30 (log2 scale) would be misinterpreted as
    actual abundances when they should be 1000-1000000000.
    """

    @pytest.fixture
    def linear_scale_test_data(self):
        """Create test data with known LINEAR scale input values.
        
        Input values are chosen to be clearly distinguishable between
        linear and log2 scale:
        - Linear values: 10000, 20000, 50000, etc.
        - Log2 of these: ~13.3, ~14.3, ~15.6
        
        If output is in log2, values will be ~13-16
        If output is in linear, values will be ~10000-100000
        """
        # Use values that are clearly different in log2 vs linear
        return pd.DataFrame({
            "Peptide Modified Sequence": ["PEPTIDEK"] * 9,
            "Fragment Ion": ["y3", "y4", "y5"] * 3,
            "Precursor Charge": [2, 2, 2] * 3,
            "Product Charge": [1, 1, 1] * 3,
            "Replicate Name": ["S1", "S1", "S1", "S2", "S2", "S2", "S3", "S3", "S3"],
            # LINEAR scale input - these should result in linear output
            "Area": [10000, 20000, 15000,  # S1: sum = 45000
                     11000, 21000, 16000,  # S2: sum = 48000
                     9000, 19000, 14000],  # S3: sum = 42000
            "Retention Time": [10.5] * 9,
        })

    def test_peptide_rollup_output_is_linear_scale(self, linear_scale_test_data):
        """Verify peptide rollup parquet output is in LINEAR scale.
        
        The output values should be close to the sum of input values
        (for sum method) or similar magnitude, NOT log2 transformed.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.parquet"
            output_path = Path(tmpdir) / "peptides.parquet"
            
            linear_scale_test_data.to_parquet(input_path, index=False)
            
            config = ChunkedRollupConfig(
                method="sum",
                min_transitions=2,
                log_transform=True,  # Internal processing uses log2
            )
            
            result = rollup_transitions_sorted(
                parquet_path=input_path,
                output_path=output_path,
                config=config,
                samples=["S1", "S2", "S3"],
            )
            
            # Read output and check values
            peptide_df = pd.read_parquet(output_path)
            
            # Get sample column values
            sample_values = peptide_df[["S1", "S2", "S3"]].values.flatten()
            
            # LINEAR scale check: values should be in the thousands
            # (sum of ~10000 + 20000 + 15000 = 45000 range)
            min_val = sample_values.min()
            max_val = sample_values.max()
            
            # If values are in log2 scale, they would be ~15 (log2(45000) ≈ 15.5)
            # If values are in linear scale, they would be ~42000-48000
            assert min_val > 1000, (
                f"Output appears to be in log2 scale! min={min_val:.2f}. "
                f"Expected linear scale values > 1000"
            )
            assert max_val > 10000, (
                f"Output appears to be in log2 scale! max={max_val:.2f}. "
                f"Expected linear scale values > 10000"
            )
            
            # More specific check: values should be close to sum of inputs
            # S1 sum = 10000 + 20000 + 15000 = 45000
            expected_s1 = 45000
            actual_s1 = peptide_df["S1"].iloc[0]
            # Allow 10% tolerance for any normalization effects
            assert abs(actual_s1 - expected_s1) / expected_s1 < 0.1, (
                f"S1 value {actual_s1:.0f} is not close to expected {expected_s1}. "
                f"Ratio: {actual_s1/expected_s1:.2f}"
            )

    def test_output_not_log2_scale(self, linear_scale_test_data):
        """Explicitly test that output is NOT in log2 scale.
        
        This is a regression test to catch if someone accidentally
        removes the linear conversion or changes the output format.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.parquet"
            output_path = Path(tmpdir) / "peptides.parquet"
            
            linear_scale_test_data.to_parquet(input_path, index=False)
            
            config = ChunkedRollupConfig(
                method="sum",
                min_transitions=2,
                log_transform=True,
            )
            
            rollup_transitions_sorted(
                parquet_path=input_path,
                output_path=output_path,
                config=config,
                samples=["S1", "S2", "S3"],
            )
            
            peptide_df = pd.read_parquet(output_path)
            sample_values = peptide_df[["S1", "S2", "S3"]].values.flatten()
            
            # Log2 of typical proteomics values would be in range 10-30
            # Linear values should be in range 1000-1000000000
            log2_range_max = 35  # log2(34 billion) ≈ 35
            
            # If ANY value is below log2_range_max, it might be log2 scale
            # (assuming reasonable proteomics data with areas > 2^35)
            # But more importantly, values should be >> 100
            assert all(sample_values > 100), (
                f"Output values appear to be in log2 scale! "
                f"Values: {sample_values}. "
                f"All values should be > 100 for linear scale output."
            )

    def test_median_polish_output_is_linear_scale(self, linear_scale_test_data):
        """Verify median polish method also outputs LINEAR scale."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.parquet"
            output_path = Path(tmpdir) / "peptides.parquet"
            
            linear_scale_test_data.to_parquet(input_path, index=False)
            
            config = ChunkedRollupConfig(
                method="median_polish",
                min_transitions=2,
                log_transform=True,
            )
            
            rollup_transitions_sorted(
                parquet_path=input_path,
                output_path=output_path,
                config=config,
                samples=["S1", "S2", "S3"],
            )
            
            peptide_df = pd.read_parquet(output_path)
            sample_values = peptide_df[["S1", "S2", "S3"]].values.flatten()
            
            # Median polish will give different values than sum, but still linear
            min_val = sample_values.min()
            
            assert min_val > 1000, (
                f"Median polish output appears to be in log2 scale! "
                f"min={min_val:.2f}. Expected linear scale values > 1000"
            )

    def test_protein_rollup_output_is_linear_scale(self):
        """Verify protein rollup parquet output is in LINEAR scale.
        
        This tests the peptide -> protein rollup to ensure the final
        protein quantities are written in linear scale.
        """
        # Create peptide-level data (already in LINEAR scale as per spec)
        peptide_data = pd.DataFrame({
            "Peptide Modified Sequence": ["PEPTIDEK", "ANOTHERK", "THIRDPEPK"],
            "n_transitions": [3, 3, 3],
            # LINEAR scale values - typical peptide abundances
            "S1": [50000.0, 45000.0, 40000.0],
            "S2": [52000.0, 47000.0, 42000.0],
            "S3": [48000.0, 43000.0, 38000.0],
        })
        
        # Create protein groups - all peptides belong to same protein
        protein_groups = [
            ProteinGroup(
                group_id="PG001",
                leading_protein="PROT1",
                leading_protein_name="Protein One",
                member_proteins=["PROT1"],
                subsumed_proteins=[],
                peptides={"PEPTIDEK", "ANOTHERK", "THIRDPEPK"},
                unique_peptides={"PEPTIDEK", "ANOTHERK", "THIRDPEPK"},
                razor_peptides=set(),
                all_mapped_peptides={"PEPTIDEK", "ANOTHERK", "THIRDPEPK"},
            )
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            peptide_path = Path(tmpdir) / "peptides.parquet"
            output_path = Path(tmpdir) / "proteins.parquet"
            
            peptide_data.to_parquet(peptide_path, index=False)
            
            config = ProteinRollupConfig(
                peptide_col="Peptide Modified Sequence",
                method="median_polish",
                min_peptides=2,
            )
            
            result = rollup_proteins_streaming(
                peptide_parquet_path=peptide_path,
                protein_groups=protein_groups,
                output_path=output_path,
                config=config,
                samples=["S1", "S2", "S3"],
            )
            
            # Read output and check values
            protein_df = pd.read_parquet(output_path)
            
            # Get sample column values
            sample_cols = ["S1", "S2", "S3"]
            sample_values = protein_df[sample_cols].values.flatten()
            
            # Remove NaN values (if any)
            sample_values = sample_values[~np.isnan(sample_values)]
            
            # LINEAR scale check: protein values should be in the thousands
            # (similar magnitude to peptide inputs)
            min_val = sample_values.min()
            max_val = sample_values.max()
            
            # If values are in log2 scale, they would be ~15-16
            # If values are in linear scale, they would be ~40000-55000
            assert min_val > 1000, (
                f"Protein output appears to be in log2 scale! min={min_val:.2f}. "
                f"Expected linear scale values > 1000"
            )
            assert max_val > 10000, (
                f"Protein output appears to be in log2 scale! max={max_val:.2f}. "
                f"Expected linear scale values > 10000"
            )


class TestGetUniqueValues:
    """Test unique value extraction from parquet."""

    def test_get_unique_values(self):
        """Test extracting unique values from a column."""
        data = pd.DataFrame({
            "peptide": ["A", "A", "B", "B", "C"],
            "value": [1, 2, 3, 4, 5],
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            data.to_parquet(path, index=False)
            
            unique = get_unique_values_from_parquet(path, "peptide")
            
            assert set(unique) == {"A", "B", "C"}
            assert len(unique) == 3
