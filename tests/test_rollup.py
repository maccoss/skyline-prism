"""Tests for protein rollup module."""

import pytest
import pandas as pd
import numpy as np

from skyline_prism.rollup import (
    tukey_median_polish,
    MedianPolishResult,
)


class TestTukeyMedianPolish:
    """Tests for Tukey median polish algorithm."""

    def test_simple_matrix(self):
        """Test median polish on a simple matrix."""
        # Create a simple peptides x samples matrix
        data = pd.DataFrame({
            'Sample1': [10.0, 12.0, 11.0],
            'Sample2': [11.0, 13.0, 12.0],
            'Sample3': [9.0, 11.0, 10.0],
        }, index=['Pep1', 'Pep2', 'Pep3'])

        result = tukey_median_polish(data)

        assert isinstance(result, MedianPolishResult)
        assert len(result.col_effects) == 3  # 3 samples
        assert len(result.row_effects) == 3  # 3 peptides
        assert result.converged

    def test_convergence(self):
        """Test that median polish converges."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(10, 5) + 20,
            index=[f'Pep{i}' for i in range(10)],
            columns=[f'Sample{i}' for i in range(5)]
        )

        result = tukey_median_polish(data, max_iter=100)

        assert result.converged
        assert result.n_iterations < 100

    def test_outlier_robustness(self):
        """Test that median polish is robust to outliers."""
        # Create matrix with one outlier
        data = pd.DataFrame({
            'Sample1': [10.0, 10.0, 10.0, 100.0],  # Last is outlier
            'Sample2': [11.0, 11.0, 11.0, 11.0],
            'Sample3': [12.0, 12.0, 12.0, 12.0],
        }, index=['Pep1', 'Pep2', 'Pep3', 'PepOutlier'])

        result = tukey_median_polish(data)

        # Sample effects should be approximately 10, 11, 12
        # despite the outlier
        effects = result.col_effects
        assert abs(effects['Sample2'] - effects['Sample1'] - 1.0) < 0.5
        assert abs(effects['Sample3'] - effects['Sample2'] - 1.0) < 0.5

    def test_missing_values(self):
        """Test handling of missing values."""
        data = pd.DataFrame({
            'Sample1': [10.0, np.nan, 10.0],
            'Sample2': [11.0, 11.0, np.nan],
            'Sample3': [12.0, 12.0, 12.0],
        }, index=['Pep1', 'Pep2', 'Pep3'])

        result = tukey_median_polish(data)

        # Should still produce results
        assert len(result.col_effects) == 3
        assert not result.col_effects.isna().any()

    def test_single_peptide(self):
        """Test behavior with single peptide (degenerate case)."""
        data = pd.DataFrame({
            'Sample1': [10.0],
            'Sample2': [11.0],
            'Sample3': [12.0],
        }, index=['Pep1'])

        result = tukey_median_polish(data)

        # With single peptide, sample effects should equal the values
        # (centered around the median)
        assert len(result.col_effects) == 3

    def test_preserves_relative_quantification(self):
        """Test that relative differences between samples are preserved."""
        # Create matrix where Sample2 is consistently 2x Sample1 (log2 diff = 1)
        data = pd.DataFrame({
            'Sample1': [10.0, 12.0, 11.0, 13.0],
            'Sample2': [11.0, 13.0, 12.0, 14.0],  # +1 log2
        }, index=['Pep1', 'Pep2', 'Pep3', 'Pep4'])

        result = tukey_median_polish(data)

        diff = result.col_effects['Sample2'] - result.col_effects['Sample1']
        assert abs(diff - 1.0) < 0.1


class TestRollupMethods:
    """Tests for different rollup method implementations."""

    def test_topn_rollup(self):
        """Test Top-N rollup method."""
        from skyline_prism.rollup import rollup_top_n
        
        # Create test matrix with known values
        matrix = pd.DataFrame({
            'Sample1': [10.0, 8.0, 6.0, 4.0, 2.0],
            'Sample2': [12.0, 10.0, 8.0, 6.0, 4.0],
            'Sample3': [11.0, 9.0, np.nan, 5.0, 3.0],  # Has missing
        }, index=['P1', 'P2', 'P3', 'P4', 'P5'])
        
        # Top 3 should be mean of top 3 values
        result = rollup_top_n(matrix, n=3)
        
        # Sample1: top 3 are 10, 8, 6 -> mean = 8.0
        assert abs(result['Sample1'] - 8.0) < 0.01
        # Sample2: top 3 are 12, 10, 8 -> mean = 10.0
        assert abs(result['Sample2'] - 10.0) < 0.01
        # Sample3: top 3 are 11, 9, 5 (skips nan) -> mean = 8.33
        assert abs(result['Sample3'] - 25.0/3) < 0.01

    def test_topn_handles_fewer_peptides(self):
        """Test Top-N when fewer than N peptides available."""
        from skyline_prism.rollup import rollup_top_n
        
        matrix = pd.DataFrame({
            'Sample1': [10.0, 8.0],
            'Sample2': [12.0, np.nan],  # Only 1 valid
        }, index=['P1', 'P2'])
        
        result = rollup_top_n(matrix, n=5)  # Request 5 but only 2 rows
        
        # Should use all available
        assert abs(result['Sample1'] - 9.0) < 0.01
        assert abs(result['Sample2'] - 12.0) < 0.01

    def test_ibaq_rollup(self):
        """Test iBAQ rollup method."""
        from skyline_prism.rollup import rollup_ibaq
        
        # Create test matrix (log2 values)
        # Linear values: P1=[1024, 2048], P2=[256, 512], P3=[64, 128]
        matrix = pd.DataFrame({
            'Sample1': [10.0, 8.0, 6.0],  # log2(1024)=10, log2(256)=8, log2(64)=6
            'Sample2': [11.0, 9.0, 7.0],
        }, index=['P1', 'P2', 'P3'])
        
        # iBAQ = sum(linear) / n_theoretical
        n_theoretical = 5
        result = rollup_ibaq(matrix, n_theoretical_peptides=n_theoretical)
        
        # Sample1: (1024 + 256 + 64) / 5 = 268.8 -> log2 = ~8.07
        expected_s1 = np.log2((1024 + 256 + 64) / 5)
        assert abs(result['Sample1'] - expected_s1) < 0.01
        
        # Sample2: (2048 + 512 + 128) / 5 = 537.6 -> log2 = ~9.07
        expected_s2 = np.log2((2048 + 512 + 128) / 5)
        assert abs(result['Sample2'] - expected_s2) < 0.01

    def test_maxlfq_rollup(self):
        """Test maxLFQ rollup method."""
        from skyline_prism.rollup import rollup_maxlfq
        
        # Create test matrix where peptide ratios are consistent
        # All peptides have 1.0 log2 difference between samples
        matrix = pd.DataFrame({
            'Sample1': [10.0, 8.0, 6.0],
            'Sample2': [11.0, 9.0, 7.0],  # All +1.0 from Sample1
            'Sample3': [12.0, 10.0, 8.0],  # All +1.0 from Sample2
        }, index=['P1', 'P2', 'P3'])
        
        result = rollup_maxlfq(matrix)
        
        # Relative differences should be preserved
        diff_12 = result['Sample2'] - result['Sample1']
        diff_23 = result['Sample3'] - result['Sample2']
        
        assert abs(diff_12 - 1.0) < 0.1, f"Expected ~1.0, got {diff_12}"
        assert abs(diff_23 - 1.0) < 0.1, f"Expected ~1.0, got {diff_23}"

    def test_maxlfq_with_missing(self):
        """Test maxLFQ handles missing values."""
        from skyline_prism.rollup import rollup_maxlfq
        
        matrix = pd.DataFrame({
            'Sample1': [10.0, 8.0, np.nan],
            'Sample2': [11.0, np.nan, 7.0],
            'Sample3': [np.nan, 10.0, 8.0],
        }, index=['P1', 'P2', 'P3'])
        
        result = rollup_maxlfq(matrix)
        
        # Should return values for all samples
        assert not result.isna().any()


class TestQualityWeightedAggregation:
    """Tests for quality-weighted transition aggregation."""

    def test_weight_calculation(self):
        """Test variance model weight calculation."""
        # Will test VarianceModelParams when transition rollup is implemented
        pass


class TestProteinLevelBatchCorrection:
    """Tests for protein-level batch correction (Step 5b in spec)."""

    @pytest.fixture
    def protein_batch_data(self):
        """Create test data with protein-level batch effects."""
        np.random.seed(42)
        
        n_proteins = 50
        
        # Create sample metadata
        samples = []
        for batch_id in [1, 2, 3]:
            # 3 experimental samples per batch
            for i in range(3):
                samples.append({
                    'replicate_name': f'exp_b{batch_id}_{i}',
                    'sample_type': 'experimental',
                    'batch': batch_id,
                })
            # 1 reference per batch
            samples.append({
                'replicate_name': f'ref_b{batch_id}',
                'sample_type': 'reference',
                'batch': batch_id,
            })
            # 1 pool per batch
            samples.append({
                'replicate_name': f'pool_b{batch_id}',
                'sample_type': 'pool',
                'batch': batch_id,
            })
        
        sample_metadata = pd.DataFrame(samples)
        sample_names = sample_metadata['replicate_name'].tolist()
        
        # Create protein abundance matrix with batch effects
        protein_data = {}
        for i, sample in enumerate(sample_names):
            batch = sample_metadata.loc[
                sample_metadata['replicate_name'] == sample, 'batch'
            ].iloc[0]
            
            # Batch-specific offset
            batch_offset = (batch - 2) * 0.5  # -0.5, 0, +0.5
            
            # Base abundance + batch effect + noise
            protein_data[sample] = 10 + batch_offset + np.random.randn(n_proteins) * 0.1
        
        protein_df = pd.DataFrame(protein_data)
        protein_df.index = [f'PG{i:04d}' for i in range(n_proteins)]
        protein_df.index.name = 'protein_group_id'
        
        # Add metadata columns
        protein_df['leading_protein'] = [f'P{i:05d}' for i in range(n_proteins)]
        protein_df['leading_name'] = [f'PROT{i}' for i in range(n_proteins)]
        protein_df['n_peptides'] = 5
        protein_df['n_unique_peptides'] = 3
        
        return protein_df, sample_metadata

    def test_batch_correct_proteins_reduces_batch_effect(self, protein_batch_data):
        """Test that batch correction reduces batch effects in protein data."""
        from skyline_prism.rollup import batch_correct_proteins
        
        protein_df, sample_metadata = protein_batch_data
        
        result = batch_correct_proteins(
            protein_df,
            sample_metadata,
            sample_col='replicate_name',
            batch_col='batch',
            sample_type_col='sample_type',
        )
        
        assert result.corrected_data is not None
        assert result.corrected_data.shape == protein_df.shape
        
        # Get sample columns (exclude metadata)
        sample_cols = [c for c in protein_df.columns 
                      if c not in ['leading_protein', 'leading_name', 
                                   'n_peptides', 'n_unique_peptides']]
        
        # Calculate batch variance before and after
        def batch_variance(df, cols, metadata):
            batch_means = []
            for batch in metadata['batch'].unique():
                batch_samples = metadata.loc[
                    metadata['batch'] == batch, 'replicate_name'
                ].tolist()
                batch_cols = [c for c in cols if c in batch_samples]
                if batch_cols:
                    batch_means.append(df[batch_cols].mean().mean())
            return np.var(batch_means)
        
        var_before = batch_variance(protein_df, sample_cols, sample_metadata)
        var_after = batch_variance(result.corrected_data, sample_cols, sample_metadata)
        
        # Batch variance should decrease
        assert var_after < var_before, \
            f"Batch variance should decrease: {var_before:.4f} -> {var_after:.4f}"

    def test_batch_correct_proteins_with_evaluation(self, protein_batch_data):
        """Test that evaluation metrics are calculated."""
        from skyline_prism.rollup import batch_correct_proteins
        
        protein_df, sample_metadata = protein_batch_data
        
        result = batch_correct_proteins(
            protein_df,
            sample_metadata,
            evaluate=True,
        )
        
        assert result.evaluation is not None
        assert hasattr(result.evaluation, 'reference_cv_before')
        assert hasattr(result.evaluation, 'pool_cv_before')
        assert hasattr(result.evaluation, 'passed')

    def test_batch_correct_proteins_fallback(self, protein_batch_data):
        """Test fallback behavior when evaluation fails."""
        from skyline_prism.rollup import batch_correct_proteins
        
        protein_df, sample_metadata = protein_batch_data
        
        # This should work - fallback is enabled by default
        result = batch_correct_proteins(
            protein_df,
            sample_metadata,
            fallback_on_failure=True,
        )
        
        assert hasattr(result, 'used_fallback')
        assert result.method_log is not None
        assert len(result.method_log) > 0

    def test_batch_correct_proteins_skips_single_batch(self, protein_batch_data):
        """Test that batch correction is skipped with only one batch."""
        from skyline_prism.rollup import batch_correct_proteins
        
        protein_df, sample_metadata = protein_batch_data
        
        # Modify metadata to have only one batch
        single_batch_metadata = sample_metadata.copy()
        single_batch_metadata['batch'] = 1
        
        result = batch_correct_proteins(
            protein_df,
            single_batch_metadata,
        )
        
        # Should skip and return unchanged data
        assert 'only one batch' in ' '.join(result.method_log).lower()

    def test_protein_output_pipeline(self, protein_batch_data):
        """Test the complete protein output pipeline."""
        from skyline_prism.rollup import protein_output_pipeline
        from skyline_prism.parsimony import ProteinGroup
        
        protein_df, sample_metadata = protein_batch_data
        
        # Create mock peptide data and protein groups
        np.random.seed(123)
        n_proteins = 10
        n_peptides_per_protein = 5
        sample_cols = [c for c in protein_df.columns 
                      if c not in ['leading_protein', 'leading_name', 
                                   'n_peptides', 'n_unique_peptides']]
        
        peptide_rows = []
        protein_groups = []
        
        for prot_idx in range(n_proteins):
            group_id = f'PG{prot_idx:04d}'
            peptides = set()
            
            for pep_idx in range(n_peptides_per_protein):
                pep_id = f'peptide_{prot_idx}_{pep_idx}'
                peptides.add(pep_id)
                
                for sample in sample_cols:
                    batch = sample_metadata.loc[
                        sample_metadata['replicate_name'] == sample, 'batch'
                    ].iloc[0]
                    batch_offset = (batch - 2) * 0.5
                    
                    peptide_rows.append({
                        'peptide_modified': pep_id,
                        'replicate_name': sample,
                        'abundance': 10 + batch_offset + np.random.randn() * 0.1,
                    })
            
            # Create mock ProteinGroup
            group = ProteinGroup(
                group_id=group_id,
                leading_protein=f'P{prot_idx:05d}',
                leading_protein_name=f'PROT{prot_idx}',
                member_proteins=[f'P{prot_idx:05d}'],
                subsumed_proteins=[],
                peptides=peptides,
                unique_peptides=peptides,
                razor_peptides=set(),
            )
            protein_groups.append(group)
        
        peptide_data = pd.DataFrame(peptide_rows)
        
        # Run pipeline
        result_df, polish_results, batch_result = protein_output_pipeline(
            peptide_data,
            protein_groups,
            sample_metadata,
            min_peptides=3,
            batch_correction=True,
        )
        
        assert result_df is not None
        assert len(result_df) > 0
        assert batch_result is not None
        assert hasattr(batch_result, 'evaluation')
