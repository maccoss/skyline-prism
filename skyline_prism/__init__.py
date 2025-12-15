"""
Skyline-PRISM: Proteomics Reference-Integrated Signal Modeling

A pipeline for reference-anchored, RT-aware normalization of LC-MS proteomics data
exported from Skyline, with robust protein quantification using Tukey median polish.

See: https://skyline.ms for more information about Skyline.
"""

__version__ = "0.1.0"

from .data_io import (
    load_skyline_report,
    merge_skyline_reports,
    load_sample_metadata,
    validate_skyline_report,
)
from .parsimony import (
    build_peptide_protein_map,
    compute_protein_groups,
    ProteinGroup,
)
from .normalization import (
    normalize_pipeline,
    rt_correction_from_reference,
    median_normalize,
)
from .rollup import (
    tukey_median_polish,
    rollup_to_proteins,
    batch_correct_proteins,
    protein_output_pipeline,
    ProteinBatchCorrectionResult,
    MedianPolishResult,
    extract_peptide_residuals,
    extract_transition_residuals,
    flag_outlier_peptides,
)
from .transition_rollup import (
    rollup_transitions_to_peptides,
    VarianceModelParams,
    learn_variance_model,
    TransitionRollupResult,
)
from .validation import (
    validate_correction,
    generate_qc_report,
)
from .batch_correction import (
    combat,
    combat_from_long,
    ComBatResult,
    evaluate_batch_correction,
    combat_with_reference_samples,
    BatchCorrectionEvaluation,
)
