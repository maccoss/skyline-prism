"""Tests for protein rollup module."""

import numpy as np
import pandas as pd
import pytest

from skyline_prism.rollup import (
    MedianPolishResult,
    tukey_median_polish,
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
        # (on original scale)
        assert len(result.col_effects) == 3
        # Single peptide: col_effects should match the input values
        np.testing.assert_allclose(result.col_effects.values, data.iloc[0].values)

    def test_original_scale_output(self):
        """Test that col_effects are on original scale, not centered."""
        # Create matrix with values around 20 (typical log2 intensity)
        data = pd.DataFrame({
            'Sample1': [20.0, 22.0, 21.0],
            'Sample2': [21.0, 23.0, 22.0],
            'Sample3': [19.0, 21.0, 20.0],
        }, index=['Pep1', 'Pep2', 'Pep3'])

        result = tukey_median_polish(data)

        # col_effects should be on original scale (around 20)
        # not centered around 0
        assert result.col_effects.mean() > 15  # Definitely not centered
        # Should match column medians for this clean case
        expected = data.median(axis=0)
        np.testing.assert_allclose(result.col_effects.values, expected.values)

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

    def test_topn_rollup_global_selection(self):
        """Test Top-N rollup selects SAME peptides across ALL samples."""
        from skyline_prism.rollup import TopNResult, rollup_top_n

        # Create test matrix with known values
        # P1 and P2 have highest median (9.0 and 8.5)
        # P3 would be selected per-sample in old buggy implementation
        matrix = pd.DataFrame({
            'Sample1': [10.0, 8.0, 6.0, 4.0, 2.0],
            'Sample2': [8.0, 9.0, 6.0, 6.0, 4.0],  # P2 is highest here!
            'Sample3': [9.0, 8.5, 7.0, 5.0, 3.0],
        }, index=['P1', 'P2', 'P3', 'P4', 'P5'])

        # Top 2 by median_abundance should select P1 and P2 globally
        result = rollup_top_n(matrix, n=2, selection='median_abundance')

        # Verify it returns TopNResult
        assert isinstance(result, TopNResult)

        # Verify the SAME peptides selected globally: P1 (median=9) and P2 (median=8.5)
        assert set(result.selected_peptides) == {'P1', 'P2'}
        assert result.n_available == 5
        assert result.selection_method == 'median_abundance'

        # Sample1: mean(P1=10, P2=8) = 9.0
        assert abs(result.abundances['Sample1'] - 9.0) < 0.01
        # Sample2: mean(P1=8, P2=9) = 8.5
        assert abs(result.abundances['Sample2'] - 8.5) < 0.01
        # Sample3: mean(P1=9, P2=8.5) = 8.75
        assert abs(result.abundances['Sample3'] - 8.75) < 0.01

    def test_topn_frequency_selection(self):
        """Test Top-N with frequency-based selection."""
        from skyline_prism.rollup import rollup_top_n

        # P1 detected in 2/3 samples, P2 detected in 3/3 samples
        # By frequency, P2 should be selected first
        matrix = pd.DataFrame({
            'Sample1': [np.nan, 8.0, 6.0],
            'Sample2': [10.0, 9.0, np.nan],
            'Sample3': [np.nan, 7.0, 5.0],
        }, index=['P1', 'P2', 'P3'])

        result = rollup_top_n(matrix, n=2, selection='frequency')

        # P2 (3/3 samples) should be selected, then P1 or P3 (both 2/3)
        # P1 has higher median (10.0) than P3 (5.5), so P1 as tie-breaker
        assert 'P2' in result.selected_peptides
        assert result.selection_method == 'frequency'

    def test_topn_handles_fewer_peptides(self):
        """Test Top-N when fewer than N peptides available."""
        from skyline_prism.rollup import rollup_top_n

        matrix = pd.DataFrame({
            'Sample1': [10.0, 8.0],
            'Sample2': [12.0, 6.0],
        }, index=['P1', 'P2'])

        result = rollup_top_n(matrix, n=5, selection='median_abundance')

        # Should use all available (both P1 and P2)
        assert len(result.selected_peptides) == 2
        assert result.n_available == 2

        # Sample1: mean(10, 8) = 9.0
        assert abs(result.abundances['Sample1'] - 9.0) < 0.01
        # Sample2: mean(12, 6) = 9.0
        assert abs(result.abundances['Sample2'] - 9.0) < 0.01

    def test_topn_empty_matrix(self):
        """Test Top-N with empty matrix returns NaN."""
        from skyline_prism.rollup import rollup_top_n

        matrix = pd.DataFrame(columns=['Sample1', 'Sample2'])
        result = rollup_top_n(matrix, n=3)

        assert result.n_available == 0
        assert len(result.selected_peptides) == 0
        assert pd.isna(result.abundances['Sample1'])
        assert pd.isna(result.abundances['Sample2'])

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


class TestRollupToProteins:
    """Tests for rollup_to_proteins function with low-peptide handling."""

    @pytest.fixture
    def sample_peptide_data(self):
        """Create sample peptide data with varying peptide counts per protein."""
        rows = []
        samples = ['S1', 'S2', 'S3']

        # Protein with 1 peptide
        for s in samples:
            rows.append({
                'peptide_modified': 'pep_single',
                'replicate_name': s,
                'abundance': 10.0 + samples.index(s),
            })

        # Protein with 2 peptides
        for s in samples:
            rows.append({
                'peptide_modified': 'pep_two_a',
                'replicate_name': s,
                'abundance': 8.0 + samples.index(s),
            })
            rows.append({
                'peptide_modified': 'pep_two_b',
                'replicate_name': s,
                'abundance': 12.0 + samples.index(s),
            })

        # Protein with 3 peptides (uses median polish)
        for s in samples:
            for i, pep in enumerate(['pep_three_a', 'pep_three_b', 'pep_three_c']):
                rows.append({
                    'peptide_modified': pep,
                    'replicate_name': s,
                    'abundance': 9.0 + samples.index(s) + i * 0.5,
                })

        # Protein with 5 peptides (well above threshold)
        for s in samples:
            for i in range(5):
                rows.append({
                    'peptide_modified': f'pep_five_{i}',
                    'replicate_name': s,
                    'abundance': 11.0 + samples.index(s) + i * 0.3,
                })

        return pd.DataFrame(rows)

    @pytest.fixture
    def sample_protein_groups(self):
        """Create protein groups with varying peptide counts."""
        from skyline_prism.parsimony import ProteinGroup

        groups = [
            ProteinGroup(
                group_id='PG_single',
                leading_protein='P001',
                leading_protein_name='SinglePeptideProtein',
                member_proteins=['P001'],
                subsumed_proteins=[],
                peptides={'pep_single'},
                unique_peptides={'pep_single'},
                razor_peptides=set(),
                all_mapped_peptides={'pep_single'},
            ),
            ProteinGroup(
                group_id='PG_two',
                leading_protein='P002',
                leading_protein_name='TwoPeptideProtein',
                member_proteins=['P002'],
                subsumed_proteins=[],
                peptides={'pep_two_a', 'pep_two_b'},
                unique_peptides={'pep_two_a', 'pep_two_b'},
                razor_peptides=set(),
                all_mapped_peptides={'pep_two_a', 'pep_two_b'},
            ),
            ProteinGroup(
                group_id='PG_three',
                leading_protein='P003',
                leading_protein_name='ThreePeptideProtein',
                member_proteins=['P003'],
                subsumed_proteins=[],
                peptides={'pep_three_a', 'pep_three_b', 'pep_three_c'},
                unique_peptides={'pep_three_a', 'pep_three_b', 'pep_three_c'},
                razor_peptides=set(),
                all_mapped_peptides={'pep_three_a', 'pep_three_b', 'pep_three_c'},
            ),
            ProteinGroup(
                group_id='PG_five',
                leading_protein='P005',
                leading_protein_name='FivePeptideProtein',
                member_proteins=['P005'],
                subsumed_proteins=[],
                peptides={f'pep_five_{i}' for i in range(5)},
                unique_peptides={f'pep_five_{i}' for i in range(5)},
                razor_peptides=set(),
                all_mapped_peptides={f'pep_five_{i}' for i in range(5)},
            ),
        ]
        return groups

    def test_all_proteins_quantified(self, sample_peptide_data, sample_protein_groups):
        """Test that ALL proteins are quantified, regardless of peptide count."""
        from skyline_prism.rollup import rollup_to_proteins

        result_df, polish_results, topn_results = rollup_to_proteins(
            sample_peptide_data,
            sample_protein_groups,
            abundance_col='abundance',
            sample_col='replicate_name',
            peptide_col='peptide_modified',
            method='median_polish',
        )

        # All 4 proteins should be quantified
        assert len(result_df) == 4, f"Expected 4 proteins, got {len(result_df)}"
        assert 'PG_single' in result_df.index
        assert 'PG_two' in result_df.index
        assert 'PG_three' in result_df.index
        assert 'PG_five' in result_df.index

    def test_n_peptides_in_output(self, sample_peptide_data, sample_protein_groups):
        """Test that n_peptides column is present for users to filter if desired."""
        from skyline_prism.rollup import rollup_to_proteins

        result_df, _, _ = rollup_to_proteins(
            sample_peptide_data,
            sample_protein_groups,
            abundance_col='abundance',
            sample_col='replicate_name',
            peptide_col='peptide_modified',
            method='median_polish',
        )

        # Check n_peptides column exists for user filtering
        assert 'n_peptides' in result_df.columns

    def test_single_peptide_uses_direct_abundance(self, sample_peptide_data, sample_protein_groups):
        """Test that 1-peptide proteins use the peptide abundance directly."""
        from skyline_prism.rollup import rollup_to_proteins

        result_df, polish_results, _ = rollup_to_proteins(
            sample_peptide_data,
            sample_protein_groups,
            abundance_col='abundance',
            sample_col='replicate_name',
            peptide_col='peptide_modified',
            method='median_polish',
        )

        # For 1-peptide protein, abundances should be: S1=10, S2=11, S3=12
        assert abs(result_df.loc['PG_single', 'S1'] - 10.0) < 0.01
        assert abs(result_df.loc['PG_single', 'S2'] - 11.0) < 0.01
        assert abs(result_df.loc['PG_single', 'S3'] - 12.0) < 0.01

        # Should NOT have a polish result for single peptide
        assert 'PG_single' not in polish_results

    def test_two_peptide_uses_mean(self, sample_peptide_data, sample_protein_groups):
        """Test that 2-peptide proteins use mean of peptides."""
        from skyline_prism.rollup import rollup_to_proteins

        result_df, polish_results, _ = rollup_to_proteins(
            sample_peptide_data,
            sample_protein_groups,
            abundance_col='abundance',
            sample_col='replicate_name',
            peptide_col='peptide_modified',
            method='median_polish',
        )

        # For 2-peptide protein, abundances should be mean of:
        # pep_two_a: S1=8, S2=9, S3=10
        # pep_two_b: S1=12, S2=13, S3=14
        # Mean: S1=10, S2=11, S3=12
        assert abs(result_df.loc['PG_two', 'S1'] - 10.0) < 0.01
        assert abs(result_df.loc['PG_two', 'S2'] - 11.0) < 0.01
        assert abs(result_df.loc['PG_two', 'S3'] - 12.0) < 0.01

        # Should NOT have a polish result for two peptides
        assert 'PG_two' not in polish_results

    def test_three_or_more_uses_median_polish(self, sample_peptide_data, sample_protein_groups):
        """Test that proteins with ≥3 peptides use the requested method."""
        from skyline_prism.rollup import rollup_to_proteins

        result_df, polish_results, _ = rollup_to_proteins(
            sample_peptide_data,
            sample_protein_groups,
            abundance_col='abundance',
            sample_col='replicate_name',
            peptide_col='peptide_modified',
            method='median_polish',
        )

        # Should have polish results for proteins with >= 3 peptides
        assert 'PG_three' in polish_results
        assert 'PG_five' in polish_results

    def test_zero_peptide_protein_skipped(self):
        """Test that proteins with 0 peptides are skipped (not quantified)."""
        from skyline_prism.parsimony import ProteinGroup
        from skyline_prism.rollup import rollup_to_proteins

        # Create minimal peptide data
        peptide_data = pd.DataFrame([
            {'peptide_modified': 'pep1', 'replicate_name': 'S1', 'abundance': 10.0},
        ])

        # Create protein group with no peptides matching data
        groups = [
            ProteinGroup(
                group_id='PG_empty',
                leading_protein='P_EMPTY',
                leading_protein_name='EmptyProtein',
                member_proteins=['P_EMPTY'],
                subsumed_proteins=[],
                peptides=set(),  # No peptides!
                unique_peptides=set(),
                razor_peptides=set(),
                all_mapped_peptides=set(),
            ),
            ProteinGroup(
                group_id='PG_has_data',
                leading_protein='P_DATA',
                leading_protein_name='HasDataProtein',
                member_proteins=['P_DATA'],
                subsumed_proteins=[],
                peptides={'pep1'},
                unique_peptides={'pep1'},
                razor_peptides=set(),
                all_mapped_peptides={'pep1'},
            ),
        ]

        result_df, _, _ = rollup_to_proteins(
            peptide_data,
            groups,
            abundance_col='abundance',
            sample_col='replicate_name',
            peptide_col='peptide_modified',
            method='median_polish',
        )

        # Only the protein with data should be in results
        assert len(result_df) == 1
        assert 'PG_has_data' in result_df.index
        assert 'PG_empty' not in result_df.index


class TestQualityWeightedAggregation:
    """Tests for quality-weighted transition aggregation."""

    def test_weight_calculation(self):
        """Test variance model weight calculation."""
        # Will test VarianceModelParams when transition rollup is implemented
        pass


class TestExtractionFunctions:
    """Tests for extraction functions that prepare data for output."""

    def test_extract_topn_selections(self):
        """Test extracting top-N peptide selections."""
        from skyline_prism.rollup import extract_topn_selections, rollup_top_n

        # Create some TopNResult objects
        matrix1 = pd.DataFrame({
            'S1': [10.0, 8.0, 6.0],
            'S2': [9.0, 8.5, 7.0],
        }, index=['Pep1', 'Pep2', 'Pep3'])
        result1 = rollup_top_n(matrix1, n=2, selection='median_abundance')

        matrix2 = pd.DataFrame({
            'S1': [12.0, 10.0],
            'S2': [11.0, 9.0],
        }, index=['PepA', 'PepB'])
        result2 = rollup_top_n(matrix2, n=2, selection='frequency')

        topn_results = {
            'Protein1': result1,
            'Protein2': result2,
        }

        selections_df = extract_topn_selections(topn_results)

        # Check structure
        assert 'protein_group_id' in selections_df.columns
        assert 'peptide' in selections_df.columns
        assert 'selection_rank' in selections_df.columns
        assert 'selection_method' in selections_df.columns

        # Protein1 should have 2 selected peptides
        p1_rows = selections_df[selections_df['protein_group_id'] == 'Protein1']
        assert len(p1_rows) == 2
        assert set(p1_rows['peptide']) == {'Pep1', 'Pep2'}

        # Ranks should be 1 and 2
        assert set(p1_rows['selection_rank']) == {1, 2}

    def test_extract_topn_selections_empty(self):
        """Test extracting from empty results."""
        from skyline_prism.rollup import extract_topn_selections

        empty_df = extract_topn_selections({})

        assert len(empty_df) == 0
        assert 'protein_group_id' in empty_df.columns


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
            # 1 qc per batch
            samples.append({
                'replicate_name': f'qc_b{batch_id}',
                'sample_type': 'qc',
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
        assert hasattr(result.evaluation, 'qc_cv_before')
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
        from skyline_prism.parsimony import ProteinGroup
        from skyline_prism.rollup import protein_output_pipeline

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
                all_mapped_peptides=peptides,  # Same for this test
            )
            protein_groups.append(group)

        peptide_data = pd.DataFrame(peptide_rows)

        # Run pipeline
        result_df, polish_results, batch_result = protein_output_pipeline(
            peptide_data,
            protein_groups,
            sample_metadata,
            batch_correction=True,
        )

        assert result_df is not None
        assert len(result_df) > 0
        assert batch_result is not None
        assert hasattr(batch_result, 'evaluation')
