"""Skyline-PRISM: Proteomics Reference-Integrated Signal Modeling.

A normalization pipeline for LC-MS proteomics data exported from Skyline,
with ComBat batch correction and robust protein quantification.

See: https://skyline.ms for more information about Skyline.
"""

__version__ = "0.1.0"

__all__ = [
    # batch_correction
    "BatchCorrectionEvaluation",
    "ComBatResult",
    "combat",
    "combat_from_long",
    "combat_with_reference_samples",
    "evaluate_batch_correction",
    # data_io
    "BatchEstimationResult",
    "SourceFingerprint",
    "apply_batch_estimation",
    "classify_sample_by_name",
    "compute_file_fingerprint",
    "compute_source_fingerprints",
    "convert_skyline_csv_to_parquet",
    "estimate_batches",
    "generate_sample_metadata",
    "get_parquet_source_fingerprints",
    "load_sample_metadata",
    "load_skyline_report",
    "merge_skyline_reports",
    "merge_skyline_reports_streaming",
    "validate_skyline_report",
    "verify_source_fingerprints",
    # fasta
    "ENZYME_RULES",
    "ProteinEntry",
    "build_peptide_protein_map_from_fasta",
    "build_protein_name_map",
    "digest_fasta",
    "digest_protein",
    "get_detected_peptides_from_data",
    "get_theoretical_peptide_counts",
    "normalize_for_matching",
    "parse_fasta",
    "strip_modifications",
    # normalization
    "median_normalize",
    "normalize_pipeline",
    "quantile_normalize",
    "rt_correction_from_reference",
    "vsn_normalize",
    # parsimony
    "ProteinGroup",
    "build_peptide_protein_map",
    "compute_protein_groups",
    "parsimony_from_fasta",
    # rollup
    "AggregationResult",
    "MedianPolishResult",
    "ProteinBatchCorrectionResult",
    "TopNResult",
    "batch_correct_proteins",
    "extract_peptide_residuals",
    "extract_quality_weights",
    "extract_topn_selections",
    "extract_transition_residuals",
    "flag_outlier_peptides",
    "protein_output_pipeline",
    "rollup_to_proteins",
    "rollup_top_n",
    "tukey_median_polish",
    # transition_rollup
    "AdaptiveRollupParams",
    "AdaptiveRollupResult",
    "TransitionRollupResult",
    "compute_adaptive_weights",
    "learn_adaptive_weights",
    "rollup_peptide_adaptive",
    "rollup_transitions_to_peptides",
    # validation
    "generate_comprehensive_qc_report",
    "generate_qc_report",
    "validate_correction",
    # visualization
    "plot_comparative_cv",
    "plot_comparative_pca",
    "plot_control_correlation_heatmap",
    "plot_cv_distribution",
    "plot_intensity_distribution",
    "plot_normalization_comparison",
    "plot_pca",
    "plot_rt_correction_comparison",
    "plot_rt_correction_per_sample",
    "plot_rt_residuals",
    "plot_sample_correlation_matrix",
    # visualization - wide format
    "plot_comparative_pca_wide",
    "plot_control_correlation_wide",
    "plot_cv_comparison_wide",
    "plot_intensity_distribution_wide",
    "plot_normalization_comparison_wide",
    "plot_pca_wide",
]

from .batch_correction import (
    BatchCorrectionEvaluation,
    ComBatResult,
    combat,
    combat_from_long,
    combat_with_reference_samples,
    evaluate_batch_correction,
)
from .data_io import (
    BatchEstimationResult,
    SourceFingerprint,
    apply_batch_estimation,
    classify_sample_by_name,
    compute_file_fingerprint,
    compute_source_fingerprints,
    convert_skyline_csv_to_parquet,
    estimate_batches,
    generate_sample_metadata,
    get_parquet_source_fingerprints,
    load_sample_metadata,
    load_skyline_report,
    merge_skyline_reports,
    merge_skyline_reports_streaming,
    validate_skyline_report,
    verify_source_fingerprints,
)
from .fasta import (
    ENZYME_RULES,
    ProteinEntry,
    build_peptide_protein_map_from_fasta,
    build_protein_name_map,
    digest_fasta,
    digest_protein,
    get_detected_peptides_from_data,
    get_theoretical_peptide_counts,
    normalize_for_matching,
    parse_fasta,
    strip_modifications,
)
from .normalization import (
    median_normalize,
    normalize_pipeline,
    quantile_normalize,
    rt_correction_from_reference,
    vsn_normalize,
)
from .parsimony import (
    ProteinGroup,
    build_peptide_protein_map,
    compute_protein_groups,
)
from .parsimony import (
    build_peptide_protein_map_from_fasta as parsimony_from_fasta,
)
from .rollup import (
    AggregationResult,
    MedianPolishResult,
    ProteinBatchCorrectionResult,
    TopNResult,
    batch_correct_proteins,
    extract_peptide_residuals,
    extract_quality_weights,
    extract_topn_selections,
    extract_transition_residuals,
    flag_outlier_peptides,
    protein_output_pipeline,
    rollup_to_proteins,
    rollup_top_n,
    tukey_median_polish,
)
from .transition_rollup import (
    AdaptiveRollupParams,
    AdaptiveRollupResult,
    TransitionRollupResult,
    compute_adaptive_weights,
    learn_adaptive_weights,
    rollup_peptide_adaptive,
    rollup_transitions_to_peptides,
)
from .validation import (
    generate_comprehensive_qc_report,
    generate_qc_report,
    validate_correction,
)
from .visualization import (
    plot_comparative_cv,
    plot_comparative_pca,
    plot_comparative_pca_wide,
    plot_control_correlation_heatmap,
    plot_control_correlation_wide,
    plot_cv_comparison_wide,
    plot_cv_distribution,
    plot_intensity_distribution,
    plot_intensity_distribution_wide,
    plot_normalization_comparison,
    plot_normalization_comparison_wide,
    plot_pca,
    plot_pca_wide,
    plot_sample_correlation_matrix,
)
