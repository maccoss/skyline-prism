"""Command-line interface for Skyline-PRISM.

PRISM: Proteomics Reference-Integrated Signal Modeling

Normalization and batch correction for Skyline proteomics data with robust protein quantification.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

if TYPE_CHECKING:
    import pandas as pd

from .data_io import (
    generate_sample_metadata,
    get_parquet_source_fingerprints,
    load_sample_metadata,
    load_unified_data,
    merge_skyline_reports,
    merge_skyline_reports_streaming,
    verify_source_fingerprints,
)
from .normalization import normalize_pipeline
from .parsimony import (
    build_peptide_protein_map,
    compute_protein_groups,
    export_protein_groups,
)
from .rollup import (
    rollup_to_proteins,
)
from .validation import (
    generate_comprehensive_qc_report,
    generate_qc_report,
    validate_correction,
)

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    """Configure logging.

    Args:
        verbose: If True, set log level to DEBUG, otherwise INFO.
        log_file: Optional path to write log output. If provided, logs are
                  written to both console and file.

    """
    level = logging.DEBUG if verbose else logging.INFO

    # Get root logger for skyline_prism
    root_logger = logging.getLogger('skyline_prism')
    root_logger.setLevel(level)
    root_logger.propagate = False  # Prevent duplicate output to root logger

    # Clear any existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if log_file specified)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.info(f"Logging to: {log_file}")


def load_config(config_path: Path | None) -> dict:
    """Load configuration from YAML file or return defaults."""
    defaults = {
        'data': {
            'abundance_column': 'Area',
            'rt_column': 'Retention Time',
            'peptide_column': 'Peptide Modified Sequence',
            'protein_column': 'Protein Accession',
            'protein_name_column': 'Protein',
            'sample_column': 'Replicate Name',
            'batch_column': 'Batch',
            'sample_type_column': 'Sample Type',
            'transition_column': 'Fragment Ion',
        },
        'transition_rollup': {
            'enabled': False,
            'method': 'median_polish',
            'min_transitions': 3,
            'learn_adaptive_weights': False,  # Learn from reference samples
        },
        'rt_correction': {
            'enabled': False,  # Disabled by default per SPECIFICATION
            'method': 'spline',
            'spline_df': 5,
            'per_batch': True,
        },
        'global_normalization': {
            'method': 'median',
        },
        'batch_correction': {
            'enabled': True,
            'method': 'combat',
        },
        'parsimony': {
            'shared_peptide_handling': 'all_groups',
        },
        'protein_rollup': {
            'method': 'median_polish',
            'topn': {'n': 3, 'selection': 'median_abundance'},
            'median_polish': {'max_iterations': 20, 'convergence_tolerance': 0.0001},
        },
        'output': {
            'format': 'parquet',
            'include_residuals': True,
            'compress': True,
        },
    }

    if config_path and config_path.exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f)
        # Deep merge user config over defaults
        defaults = _deep_merge(defaults, user_config)

    return defaults


def load_config_from_provenance(provenance_path: Path) -> dict:
    """Load configuration from a previous pipeline run's provenance JSON.

    This enables reproducibility by allowing users to re-run the pipeline
    with the exact same parameters as a previous run.

    Args:
        provenance_path: Path to metadata.json from a previous PRISM run

    Returns:
        Configuration dictionary compatible with load_config() output

    Raises:
        ValueError: If the provenance file is missing required fields

    """
    with open(provenance_path) as f:
        provenance = json.load(f)

    # Check for required fields
    if 'processing_parameters' not in provenance:
        raise ValueError(
            f"Provenance file {provenance_path} does not contain 'processing_parameters'. "
            "This may be from an older version of PRISM."
        )

    # Start with defaults
    config = load_config(None)

    # Extract processing parameters and merge over defaults
    params = provenance['processing_parameters']

    # Map provenance sections to config sections
    for section in ['data', 'transition_rollup', 'rt_correction', 'global_normalization',
                    'batch_correction', 'protein_rollup', 'parsimony', 'output']:
        if section in params:
            config[section] = _deep_merge(config.get(section, {}), params[section])

    logger.info(f"Loaded configuration from provenance: {provenance_path}")
    logger.info(f"  Original pipeline version: {provenance.get('pipeline_version', 'unknown')}")
    logger.info(f"  Original processing date: {provenance.get('processing_date', 'unknown')}")

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def cmd_merge(args: argparse.Namespace) -> int:
    """Merge multiple Skyline reports."""
    logger = logging.getLogger(__name__)

    report_paths = [Path(p) for p in args.reports]
    output_path = Path(args.output)

    metadata = None
    if args.metadata:
        metadata = load_sample_metadata(Path(args.metadata))

    result = merge_skyline_reports(
        report_paths=report_paths,
        output_path=output_path,
        sample_metadata=metadata,
        partition_by_batch=not args.no_partition,
    )

    logger.info(f"Merged {result.n_reports} reports -> {result.output_path}")
    logger.info(f"  {result.n_rows} rows, {result.n_replicates} replicates, "
                f"{result.n_precursors} precursors")

    if result.warnings:
        for w in result.warnings:
            logger.warning(w)

    return 0


def generate_pipeline_metadata(
    config: dict,
    data: pd.DataFrame,
    protein_groups: list,
    method_log: list[str],
    input_files: list[str],
    sample_type_col: str = 'sample_type',
    sample_col: str = 'replicate_name',
    batch_col: str = 'batch',
    validation_metrics: dict | None = None,
) -> dict:
    """Generate pipeline metadata JSON for reproducibility and provenance.

    Creates a comprehensive metadata dictionary containing:
    - Pipeline version and processing timestamp
    - Input file information
    - Sample metadata summary
    - Protein grouping summary
    - All processing parameters from config
    - Validation metrics if available

    Args:
        config: Pipeline configuration dictionary
        data: Processed data DataFrame
        protein_groups: List of ProteinGroup objects
        method_log: List of processing steps performed
        input_files: List of input file paths
        sample_type_col: Column name for sample types
        sample_col: Column name for sample identifiers
        batch_col: Column name for batch identifiers
        validation_metrics: Optional validation metrics dict

    Returns:
        Dictionary with complete pipeline metadata

    """
    # Get version from package
    try:
        from importlib.metadata import version
        pipeline_version = version('skyline-prism')
    except Exception:
        pipeline_version = 'development'

    # Build sample metadata summary
    samples_df = data[[sample_col, sample_type_col]].drop_duplicates()
    sample_counts = samples_df[sample_type_col].value_counts().to_dict()

    sample_metadata = {
        'n_samples': len(samples_df),
        'n_reference': sample_counts.get('reference', 0),
        'n_qc': sample_counts.get('qc', 0),
        'n_experimental': sample_counts.get('experimental', 0),
        'samples': samples_df[sample_col].tolist(),
    }

    # Add batch info if available
    if batch_col in data.columns:
        batches = data[batch_col].dropna().unique().tolist()
        sample_metadata['batches'] = batches
        sample_metadata['n_batches'] = len(batches)

    # Build protein groups summary
    groups_summary = {
        'n_groups': len(protein_groups),
        'n_proteins': sum(len(g.proteins) for g in protein_groups),
        'shared_peptide_handling': config.get('parsimony', {}).get(
            'shared_peptide_handling', 'all_groups'
        ),
    }

    # Build processing parameters from config (includes all settings for reproducibility)
    processing_parameters = {
        'data': config.get('data', {}),  # Column mappings for reproducibility
        'transition_rollup': config.get('transition_rollup', {}),
        'rt_correction': config.get('rt_correction', {}),
        'global_normalization': config.get('global_normalization', {}),
        'batch_correction': config.get('batch_correction', {}),
        'protein_rollup': config.get('protein_rollup', {}),
        'parsimony': config.get('parsimony', {}),
        'output': config.get('output', {}),
    }

    # Build the metadata dictionary
    metadata = {
        'pipeline_version': pipeline_version,
        'processing_date': datetime.now(timezone.utc).isoformat(),
        'source_files': input_files,
        'sample_metadata': sample_metadata,
        'protein_groups': groups_summary,
        'processing_parameters': processing_parameters,
        'method_log': method_log,
        'validation_metrics': validation_metrics or {},
        'warnings': [],
    }

    return metadata


@dataclass
class PipelineResult:
    """Results from the full PRISM pipeline."""

    peptide_data: pd.DataFrame
    protein_data: pd.DataFrame
    protein_groups: list
    method_log: list[str]
    peptide_residuals: pd.DataFrame | None = None
    transition_residuals: pd.DataFrame | None = None


def cmd_run(args: argparse.Namespace) -> int:
    """Run the full PRISM pipeline using streaming processing.

    This is the memory-efficient version that processes data without loading
    the entire dataset into memory.

    Pipeline stages:
    1. Merge input CSVs to parquet (streaming)
    2. Transition -> Peptide rollup
       2b. Peptide Global normalization
       2c. Peptide ComBat batch correction
    3. Protein Parsimony
    4. Peptide -> Protein rollup
       4b. Protein Global Normalization
       4c. Protein ComBat batch correction
    5. Output generation

    Args:
        args: Command-line arguments

    Returns:
        Exit code (0 = success)

    """
    import pandas as pd
    import pyarrow.parquet as pq

    from .chunked_processing import (
        ChunkedRollupConfig,
        ProteinRollupConfig,
        rollup_proteins_streaming,
        rollup_transitions_sorted,
    )
    from .parsimony import (
        build_peptide_protein_map,
        compute_protein_groups,
        export_protein_groups,
    )
    from .transition_rollup import (
        AdaptiveRollupParams,
    )

    # Load configuration
    if hasattr(args, 'from_provenance') and args.from_provenance:
        config = load_config_from_provenance(Path(args.from_provenance))
        method_log = [f"Configuration loaded from provenance: {args.from_provenance}"]
        if args.config:
            yaml_config = load_config(Path(args.config))
            config = _deep_merge(config, yaml_config)
            method_log.append(f"Configuration overrides from: {args.config}")
    else:
        config = load_config(Path(args.config) if args.config else None)
        method_log = []

    # =========================================================================
    # Stage 0: Prepare input data
    # =========================================================================
    if isinstance(args.input, list):
        input_paths = [Path(p) for p in args.input]
    else:
        input_paths = [Path(args.input)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up file logging to output directory
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = output_dir / f'prism_run_{timestamp}.log'
    verbose = getattr(args, 'verbose', False)
    setup_logging(verbose=verbose, log_file=log_file)

    # Get sample type patterns from args or config
    reference_patterns = None
    qc_patterns = None
    if hasattr(args, 'reference_pattern') and args.reference_pattern:
        reference_patterns = args.reference_pattern
    elif 'sample_annotations' in config:
        ref_pattern = config['sample_annotations'].get('reference_pattern')
        if ref_pattern:
            reference_patterns = (
                [ref_pattern] if isinstance(ref_pattern, str) else ref_pattern
            )

    if hasattr(args, 'qc_pattern') and args.qc_pattern:
        qc_patterns = args.qc_pattern
    elif 'sample_annotations' in config:
        qc_pattern = config['sample_annotations'].get('qc_pattern')
        if qc_pattern:
            qc_patterns = (
                [qc_pattern] if isinstance(qc_pattern, str) else qc_pattern
            )

    # Categorize input files
    csv_inputs = [
        p for p in input_paths if p.suffix.lower() in ['.csv', '.tsv', '.txt']
    ]
    parquet_inputs = [
        p for p in input_paths if p.suffix.lower() == '.parquet'
    ]

    # Check for --force-reprocess flag
    force_reprocess = getattr(args, 'force_reprocess', False)

    # =========================================================================
    # Stage 1: Merge CSVs / Prepare Input Data
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Stage 1: Merge CSVs / Prepare Input Data")
    logger.info("=" * 60)

    metadata_df = None
    if len(csv_inputs) >= 1:
        batch_names = [p.stem for p in csv_inputs]
        merged_parquet_path = output_dir / 'merged_data.parquet'

        # Check if we can reuse an existing merged parquet
        use_cached = False
        if not force_reprocess and merged_parquet_path.exists():
            logger.info(f"Found existing merged parquet: {merged_parquet_path}")
            stored_fingerprints = get_parquet_source_fingerprints(merged_parquet_path)

            if stored_fingerprints:
                is_valid, mismatch_reasons = verify_source_fingerprints(
                    csv_inputs, stored_fingerprints
                )
                if is_valid:
                    logger.info("  Source files match - using cached parquet")
                    use_cached = True
                    transition_parquet = merged_parquet_path
                    method_log.append(
                        "Reused cached merged parquet (source files unchanged)"
                    )
                    # Try to load existing metadata if available
                    existing_metadata = output_dir / 'sample_metadata.tsv'
                    if existing_metadata.exists() and not args.metadata:
                        metadata_df = load_sample_metadata(existing_metadata)
                        logger.info(f"  Loaded existing metadata: {existing_metadata}")
                else:
                    logger.info("  Source files changed - reprocessing required:")
                    for reason in mismatch_reasons:
                        logger.info(f"    - {reason}")
            else:
                logger.info(
                    "  No fingerprints in parquet metadata - reprocessing required"
                )

        if not use_cached:
            logger.info(f"Merging {len(csv_inputs)} Skyline reports (streaming)...")
            merged_path, samples_by_batch, total_rows = merge_skyline_reports_streaming(
                csv_inputs,
                merged_parquet_path,
                batch_names=batch_names,
            )
            method_log.append(f"Merged {len(csv_inputs)} reports ({total_rows:,} rows)")
            transition_parquet = merged_path

            # Generate metadata if not provided
            if not args.metadata:
                logger.info("Generating sample metadata from sample names...")
                metadata_df = generate_sample_metadata(
                    samples_by_batch,
                    reference_patterns=reference_patterns,
                    qc_patterns=qc_patterns,
                )
                metadata_path = output_dir / 'sample_metadata.tsv'
                metadata_df.to_csv(metadata_path, sep='\t', index=False)
                method_log.append(f"Generated sample metadata: {metadata_path}")

    elif len(parquet_inputs) >= 1:
        transition_parquet = parquet_inputs[0]
        logger.info(f"Using existing parquet: {transition_parquet}")
        method_log.append(f"Input parquet: {transition_parquet}")

    else:
        logger.error("No input files provided")
        return 1

    # Load explicit metadata if provided
    if args.metadata:
        metadata_df = load_sample_metadata(Path(args.metadata))
        method_log.append(f"Loaded metadata: {args.metadata}")

    # -------------------------------------------------------------------------
    # Stage 1 (continued): Auto-detect column names from data
    # -------------------------------------------------------------------------
    pf = pq.ParquetFile(transition_parquet)
    available_columns = set(pf.schema_arrow.names)
    logger.info(f"  Available columns: {sorted(available_columns)}")

    # Auto-detect peptide column
    peptide_col = config['data']['peptide_column']
    peptide_col_alternatives = [
        'Peptide Modified Sequence Unimod Ids',
        'Peptide Modified Sequence',
        'Peptide',
    ]
    if peptide_col not in available_columns:
        for alt in peptide_col_alternatives:
            if alt in available_columns:
                logger.info(f"  Peptide column '{peptide_col}' not found, using '{alt}'")
                peptide_col = alt
                break
        else:
            logger.error(f"No peptide column found. Available: {sorted(available_columns)}")
            return 1

    # Get other column names (use config or auto-detect)
    # ALWAYS prefer 'Sample ID' (unique across batches) over 'Replicate Name'
    # This ensures duplicate sample names in different batches are kept separate
    if 'Sample ID' in available_columns:
        sample_col = 'Sample ID'
        logger.info("  Using 'Sample ID' for unique sample identification across batches")
    elif config['data']['sample_column'] in available_columns:
        sample_col = config['data']['sample_column']
    elif 'Replicate Name' in available_columns:
        sample_col = 'Replicate Name'
    else:
        sample_col = config['data']['sample_column']

    abundance_col = config['data']['abundance_column']
    if abundance_col not in available_columns and 'Area' in available_columns:
        abundance_col = 'Area'

    transition_col = config['data'].get('transition_column', 'Fragment Ion')
    if transition_col not in available_columns and 'Fragment Ion' in available_columns:
        transition_col = 'Fragment Ion'

    protein_col = config['data']['protein_column']
    if protein_col not in available_columns and 'Protein Accession' in available_columns:
        protein_col = 'Protein Accession'

    protein_name_col = config['data'].get('protein_name_column', 'Protein')
    if protein_name_col not in available_columns and 'Protein' in available_columns:
        protein_name_col = 'Protein'

    logger.info(
        f"Using columns: peptide={peptide_col}, sample={sample_col}, "
        f"abundance={abundance_col}"
    )

    # Report data dimensions before processing
    logger.info("")
    logger.info("Input data summary:")
    pf = pq.ParquetFile(transition_parquet)
    n_rows = pf.metadata.num_rows
    logger.info(f"  Total rows: {n_rows:,}")

    # Get unique peptide and transition counts efficiently
    try:
        import duckdb
        con = duckdb.connect()
        result = con.execute(f"""
            SELECT
                COUNT(DISTINCT "{peptide_col}") as n_peptides,
                COUNT(DISTINCT "{peptide_col}" || '|' || "{transition_col}") as n_transitions,
                COUNT(DISTINCT "{sample_col}") as n_samples
            FROM read_parquet('{transition_parquet}')
        """).fetchone()
        n_peptides, n_transitions, n_samples = result
        con.close()
    except Exception:
        # Fallback: use pyarrow (slower but no external dependency)
        table = pq.read_table(transition_parquet, columns=[peptide_col, transition_col, sample_col])
        df = table.to_pandas()
        n_peptides = df[peptide_col].nunique()
        n_transitions = df[[peptide_col, transition_col]].drop_duplicates().shape[0]
        n_samples = df[sample_col].nunique()

    logger.info(f"  Unique peptides: {n_peptides:,}")
    logger.info(f"  Unique transitions: {n_transitions:,}")
    logger.info(f"  Samples: {n_samples:,}")
    logger.info(f"  Avg transitions per peptide: {n_transitions / n_peptides:.1f}")

    # =========================================================================
    # Stage 2: Transition -> Peptide rollup
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Stage 2: Transition -> Peptide rollup")
    logger.info("=" * 60)

    # Get rollup configuration
    rollup_method = config['transition_rollup'].get('method', 'sum')
    use_ms1 = config['transition_rollup'].get('use_ms1', False)
    min_transitions = config['transition_rollup'].get('min_transitions', 3)
    # Support both 'learn_weights' and 'learn_adaptive_weights' config keys
    learn_weights = config['transition_rollup'].get(
        'learn_adaptive_weights',
        config['transition_rollup'].get('learn_weights', False)
    )

    logger.info(f"  Rollup method: {rollup_method}")
    logger.info(f"  Include MS1 precursors: {use_ms1}")
    logger.info(f"  Min transitions: {min_transitions}")

    # Initialize adaptive rollup params (None for non-adaptive methods)
    adaptive_params = None

    if rollup_method == "adaptive":
        # Check for explicit parameters in config
        ar_config = config['transition_rollup'].get('adaptive_rollup', {})
        adaptive_params = AdaptiveRollupParams(
            beta_log_intensity=0.0,  # Deprecated
            beta_sqrt_intensity=0.0,  # Deprecated
            beta_relative_intensity=ar_config.get('beta_relative_intensity', 0.0),
            beta_mz=ar_config.get('beta_mz', 0.0),
            beta_shape_corr=ar_config.get('beta_shape_corr', 0.0),
            beta_shape_corr_max=0.0,  # Not optimized
            beta_shape_corr_outlier=ar_config.get('beta_shape_corr_outlier', 0.0),
            mz_min=ar_config.get('mz_min', 0.0),
            mz_max=ar_config.get('mz_max', 2000.0),
            log_intensity_center=ar_config.get('log_intensity_center', 15.0),
            shape_corr_low_threshold=ar_config.get('shape_corr_low_threshold', 0.5),
            min_improvement_pct=ar_config.get('min_improvement_pct', 5.0),
        )

        # Learn weights from reference/QC samples if enabled
        if learn_weights:
            from .transition_rollup import learn_adaptive_weights

            # Extract sample types from metadata
            # Use sample_id (unique across batches) if available, else sample
            meta_id_col = 'sample_id' if 'sample_id' in metadata_df.columns else 'sample'
            sample_types = dict(
                zip(metadata_df[meta_id_col], metadata_df['sample_type'])
            )
            ref_samples = [
                s for s, t in sample_types.items() if t == 'reference'
            ]
            pool_samples = [
                s for s, t in sample_types.items() if t == 'qc'
            ]

            if len(ref_samples) >= 2:
                logger.info("  Learning adaptive rollup parameters...")
                logger.info(f"    Reference samples: {len(ref_samples)}")
                logger.info(f"    QC samples: {len(pool_samples)}")

                # Load transition data for learning (sample of data)
                import duckdb
                learn_con = duckdb.connect()
                sample_list_sql = ','.join(f"'{s}'" for s in ref_samples + pool_samples)
                learn_df = learn_con.execute(f"""
                    SELECT *
                    FROM read_parquet('{transition_parquet}')
                    WHERE "{sample_col}" IN ({sample_list_sql})
                """).fetchdf()

                # Data is already in LINEAR scale from merged parquet
                # (log transform happens in rollup, not in merge)

                learn_result = learn_adaptive_weights(
                    learn_df,
                    reference_samples=ref_samples,
                    qc_samples=pool_samples,
                    peptide_col=peptide_col,
                    transition_col=transition_col,
                    sample_col=sample_col,
                    abundance_col=abundance_col,
                    mz_col='Product Mz',
                    shape_corr_col='Shape Correlation',
                    n_iterations=100,
                    initial_params=adaptive_params,
                )

                if learn_result.use_adaptive_weights:
                    adaptive_params = learn_result.params
                    logger.info("  Using learned parameters")
                    logger.info(
                        f"    Reference CV: {learn_result.reference_cv_sum * 100:.1f}% -> "
                        f"{learn_result.reference_cv_adaptive * 100:.1f}%"
                    )
                    if np.isfinite(learn_result.qc_cv_sum):
                        logger.info(
                            f"    QC CV: {learn_result.qc_cv_sum * 100:.1f}% -> "
                            f"{learn_result.qc_cv_adaptive * 100:.1f}%"
                        )
                else:
                    logger.warning(f"  {learn_result.fallback_reason}")
                    logger.warning("  Using sum method as fallback")
                    # Fallback to sum (all betas = 0)
                    adaptive_params = AdaptiveRollupParams(
                        beta_log_intensity=0.0,
                        beta_sqrt_intensity=0.0,
                        beta_relative_intensity=0.0,
                        beta_mz=0.0,
                        beta_shape_corr=0.0,
                        beta_shape_corr_max=0.0,
                        beta_shape_corr_outlier=0.0,
                    )
            else:
                logger.warning(
                    f"  Not enough reference samples for learning ({len(ref_samples)})"
                )
                logger.info("  Using default parameters")

        logger.info(
            f"  Adaptive params: rel_int={adaptive_params.beta_relative_intensity:.3f}, "
            f"mz={adaptive_params.beta_mz:.3f}, "
            f"shape_med={adaptive_params.beta_shape_corr:.3f}, "
            f"shape_out={adaptive_params.beta_shape_corr_outlier:.3f}"
        )

    # Get topn method parameters
    topn_count = config['transition_rollup'].get('topn_count', 3)
    topn_selection = config['transition_rollup'].get('topn_selection', 'correlation')
    topn_weighting = config['transition_rollup'].get('topn_weighting', 'sqrt')
    
    # Get parallel processing parameters
    n_workers = config.get('processing', {}).get('n_workers', 1)
    peptide_batch_size = config.get('processing', {}).get('peptide_batch_size', 1000)

    if rollup_method == "topn":
        logger.info(f"  Top-N count: {topn_count}")
        logger.info(f"  Selection method: {topn_selection}")
        logger.info(f"  Weighting: {topn_weighting}")
    
    if n_workers != 1:
        logger.info(f"  Parallel workers: {n_workers if n_workers > 0 else 'all CPUs'}")

    transition_config = ChunkedRollupConfig(
        peptide_col=peptide_col,
        transition_col=transition_col,
        sample_col=sample_col,
        abundance_col=abundance_col,
        method=rollup_method,
        min_transitions=min_transitions,
        log_transform=True,
        exclude_precursor=not use_ms1,  # exclude_precursor is inverse of use_ms1
        progress_interval=5000,
        adaptive_params=adaptive_params,
        topn_count=topn_count,
        topn_selection=topn_selection,
        topn_weighting=topn_weighting,
        n_workers=n_workers,
        peptide_batch_size=peptide_batch_size,
    )

    peptide_rollup_path = output_dir / 'peptides_rollup.2.parquet'
    peptide_result = rollup_transitions_sorted(
        parquet_path=transition_parquet,
        output_path=peptide_rollup_path,
        config=transition_config,
        save_residuals=config['output'].get('include_residuals', True) and rollup_method == 'median_polish',
    )
    samples = peptide_result.samples
    method_log.append(
        f"Transition rollup ({rollup_method}): {peptide_result.n_peptides:,} peptides, "
        f"{len(samples)} samples"
    )

    # -------------------------------------------------------------------------
    # Stage 2b: Peptide Global Normalization
    # -------------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("Stage 2b: Peptide Global Normalization")
    logger.info("-" * 60)

    peptide_df = pd.read_parquet(peptide_rollup_path)

    # Data is in wide format: peptide_col, n_transitions, mean_rt, sample1, sample2, ...
    # Get sample columns (all columns after metadata columns)
    meta_cols = [peptide_col, 'n_transitions', 'mean_rt']
    meta_cols = [c for c in meta_cols if c in peptide_df.columns]
    sample_cols = [c for c in peptide_df.columns if c not in meta_cols]

    # Filter out peptides with all NaN (< min_transitions, failed rollup)
    # These cannot be normalized or batch-corrected
    data_matrix = peptide_df[sample_cols].values
    valid_mask = ~np.all(np.isnan(data_matrix), axis=1)
    n_filtered = (~valid_mask).sum()
    if n_filtered > 0:
        logger.info(f"  Filtering {n_filtered} peptides with insufficient transitions (all NaN)")
        peptide_df = peptide_df[valid_mask].reset_index(drop=True)

    # Keep copy for QC comparison (before normalization)
    peptide_pre_norm_df = peptide_df.copy()

    # -------------------------------------------------------------------------
    # Sample Outlier Detection (one-sided, low signal only)
    # -------------------------------------------------------------------------
    outlier_config = config.get('sample_outlier_detection', {})
    outlier_detection_enabled = outlier_config.get('enabled', True)
    outlier_samples = []

    if outlier_detection_enabled:
        outlier_action = outlier_config.get('action', 'report')
        outlier_method = outlier_config.get('method', 'iqr')

        # Calculate sample medians on LINEAR scale (not log2!)
        # Data is in log2, so we need to convert to linear for proper statistics
        linear_medians = {}
        for col in sample_cols:
            # Convert log2 to linear, then get median
            linear_values = 2 ** peptide_df[col].dropna()
            linear_medians[col] = linear_values.median()

        linear_medians_series = pd.Series(linear_medians)
        overall_median = linear_medians_series.median()

        if outlier_method == 'iqr':
            # IQR-based detection on linear scale (one-sided, low only)
            iqr_mult = outlier_config.get('iqr_multiplier', 1.5)
            q1 = linear_medians_series.quantile(0.25)
            q3 = linear_medians_series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - iqr_mult * iqr

            outlier_samples = [
                s for s, m in linear_medians.items() if m < lower_bound
            ]

            if outlier_samples:
                logger.warning("Sample outlier detection (IQR method, linear scale):")
                logger.warning(f"  Q1={q1:.0f}, Q3={q3:.0f}, IQR={iqr:.0f}")
                logger.warning(f"  Lower bound: {lower_bound:.0f} (Q1 - {iqr_mult}*IQR)")
                logger.warning(f"  Overall median: {overall_median:.0f}")
                for s in outlier_samples:
                    fold_below = linear_medians[s] / overall_median
                    logger.warning(
                        f"  OUTLIER: {s} - median={linear_medians[s]:.0f} "
                        f"({fold_below:.1%} of overall median)"
                    )
        else:  # fold_median method
            fold_thresh = outlier_config.get('fold_threshold', 0.1)
            threshold = fold_thresh * overall_median

            outlier_samples = [
                s for s, m in linear_medians.items() if m < threshold
            ]

            if outlier_samples:
                logger.warning("Sample outlier detection (fold-median method, linear scale):")
                logger.warning(f"  Overall median: {overall_median:.0f}")
                logger.warning(f"  Threshold: {threshold:.0f} ({fold_thresh:.0%} of median)")
                for s in outlier_samples:
                    fold_below = linear_medians[s] / overall_median
                    logger.warning(
                        f"  OUTLIER: {s} - median={linear_medians[s]:.0f} "
                        f"({fold_below:.1%} of overall median)"
                    )

        if outlier_samples:
            method_log.append(
                f"Outlier detection: {len(outlier_samples)} samples flagged as low-signal outliers"
            )

            if outlier_action == 'exclude':
                logger.warning(
                    f"  Excluding {len(outlier_samples)} outlier samples from analysis"
                )
                sample_cols = [c for c in sample_cols if c not in outlier_samples]
                # Update DataFrames to remove outlier columns
                # Keep metadata columns plus remaining sample columns
                keep_cols = [c for c in peptide_df.columns if c not in outlier_samples]
                peptide_df = peptide_df[keep_cols]
                peptide_pre_norm_df = peptide_pre_norm_df[keep_cols]
                method_log.append(
                    f"  Excluded samples: {', '.join(outlier_samples)}"
                )
            else:
                logger.warning(
                    "  Action='report' - outliers will be included in analysis"
                )
        else:
            logger.info("  No sample outliers detected")

    # Apply global median normalization (on log2 scale)
    # For each sample column, subtract sample median and add global median
    sample_medians = peptide_df[sample_cols].median()
    global_peptide_median = sample_medians.median()
    norm_factors = sample_medians - global_peptide_median

    # Apply normalization to each sample column
    for col in sample_cols:
        peptide_df[col] = peptide_df[col] - norm_factors[col]

    max_pep_shift = norm_factors.abs().max()
    # Report in linear scale for interpretability
    linear_global_median = 2 ** global_peptide_median
    fold_change_shift = 2 ** max_pep_shift
    logger.info(
        f"  Global median = {linear_global_median:.0f} (linear), "
        f"max shift = {fold_change_shift:.2f}x"
    )
    method_log.append(f"Peptide median normalization: max shift = {fold_change_shift:.2f}x")

    # -------------------------------------------------------------------------
    # Stage 2c: Peptide ComBat Batch Correction
    # -------------------------------------------------------------------------
    batch_correction_enabled = config.get('batch_correction', {}).get('enabled', True)

    if batch_correction_enabled and metadata_df is not None:
        logger.info("-" * 60)
        logger.info("Stage 2c: Peptide ComBat Batch Correction")
        logger.info("-" * 60)

        # Get batch info from metadata
        batch_col = 'batch'
        sample_type_col = 'sample_type'
        # Use sample_id (unique across batches) if available, else sample/replicate_name
        if 'sample_id' in metadata_df.columns:
            meta_sample_col = 'sample_id'
        elif 'sample' in metadata_df.columns:
            meta_sample_col = 'sample'
        else:
            meta_sample_col = 'replicate_name'

        if batch_col in metadata_df.columns:
            # Build sample -> batch mapping
            sample_to_batch = dict(
                zip(metadata_df[meta_sample_col], metadata_df[batch_col])
            )

            # Check if all sample columns have batch info
            batches = [sample_to_batch.get(s) for s in sample_cols]
            n_batches = len(set(b for b in batches if b is not None))

            if n_batches < 2:
                logger.warning(f"Only {n_batches} batch - skipping batch correction")
            else:
                from .batch_correction import combat

                # Get sample types for reference-anchored ComBat
                sample_to_type = {}
                if sample_type_col in metadata_df.columns:
                    sample_to_type = dict(
                        zip(metadata_df[meta_sample_col], metadata_df[sample_type_col])
                    )

                # Prepare data for ComBat (features x samples matrix)
                data_matrix = peptide_df[sample_cols].values
                batch_labels = [sample_to_batch.get(s, 'unknown') for s in sample_cols]

                logger.info(f"  Applying ComBat across {n_batches} batches...")

                # Run ComBat - returns corrected matrix directly
                corrected_matrix = combat(data_matrix, batch_labels)

                # Replace sample columns with corrected values
                for i, col in enumerate(sample_cols):
                    peptide_df[col] = corrected_matrix[:, i]

                method_log.append(f"Peptide ComBat: {n_batches} batches corrected")

                # Evaluate using reference and QC samples if available
                if sample_to_type:
                    ref_cols = [c for c in sample_cols
                                if sample_to_type.get(c) == 'reference']
                    qc_cols = [c for c in sample_cols
                                 if sample_to_type.get(c) == 'qc']

                    if ref_cols and qc_cols:
                        # Calculate CV improvement on LINEAR scale (never on log2!)
                        ref_linear = 2 ** peptide_df[ref_cols]
                        qc_linear = 2 ** peptide_df[qc_cols]
                        ref_cv = (ref_linear.std(axis=1) / ref_linear.mean(axis=1)) * 100
                        qc_cv = (qc_linear.std(axis=1) / qc_linear.mean(axis=1)) * 100
                        logger.info(
                            f"  Reference median CV: {ref_cv.median():.1f}%, "
                            f"QC median CV: {qc_cv.median():.1f}%"
                        )
        else:
            logger.warning("No batch column in metadata - skipping batch correction")
    elif batch_correction_enabled:
        logger.warning("No metadata available - skipping batch correction")
    else:
        logger.info("  Batch correction disabled in config")

    # Save normalized peptides
    peptide_normalized_path = output_dir / 'peptides_normalized.3.parquet'
    peptide_df.to_parquet(peptide_normalized_path, index=False)
    logger.info(f"  Saved normalized peptides: {peptide_normalized_path}")

    # =========================================================================
    # Stage 3: Protein Parsimony
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Stage 3: Protein Parsimony")
    logger.info("=" * 60)

    columns_for_parsimony = [peptide_col, protein_col, protein_name_col]
    columns_for_parsimony = [c for c in columns_for_parsimony if c in available_columns]

    logger.info("  Reading peptide-protein mappings...")
    mapping_table = pf.read(columns=columns_for_parsimony)
    mapping_df = mapping_table.to_pandas().drop_duplicates()
    logger.info(f"  Found {len(mapping_df):,} unique peptide-protein records")

    pep_to_prot, prot_to_pep, prot_to_name = build_peptide_protein_map(
        mapping_df,
        peptide_col=peptide_col,
        protein_col=protein_col,
        protein_name_col=protein_name_col,
    )

    protein_groups = compute_protein_groups(prot_to_pep, pep_to_prot, prot_to_name)
    logger.info(f"  Computed {len(protein_groups)} protein groups")

    groups_output = output_dir / "protein_groups.tsv"
    export_protein_groups(protein_groups, str(groups_output))
    method_log.append(f"Protein groups: {len(protein_groups)}")

    # =========================================================================
    # Stage 4: Peptide -> Protein Rollup
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Stage 4: Peptide -> Protein Rollup")
    logger.info("=" * 60)

    protein_config = ProteinRollupConfig(
        peptide_col=peptide_col,
        sample_col=sample_col,
        method=config['protein_rollup'].get('method', 'median_polish'),
        shared_peptide_handling=config['parsimony'].get(
            'shared_peptide_handling', 'all_groups'
        ),
        min_peptides=config['protein_rollup'].get('min_peptides', 3),
        topn_n=config['protein_rollup'].get('topn', {}).get('n', 3),
        progress_interval=1000,
    )

    protein_path = output_dir / 'proteins_raw.parquet'
    protein_result = rollup_proteins_streaming(
        peptide_parquet_path=peptide_normalized_path,
        protein_groups=protein_groups,
        output_path=protein_path,
        config=protein_config,
        samples=samples,
        save_residuals=config['output'].get('include_residuals', True),
    )
    method_log.append(f"Protein rollup: {protein_result.n_proteins:,} proteins")

    # -------------------------------------------------------------------------
    # Stage 4b: Protein Global Normalization
    # -------------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("Stage 4b: Protein Global Normalization")
    logger.info("-" * 60)

    protein_df = pd.read_parquet(protein_path)

    # Data is in wide format: protein_group, leading_protein, etc., sample1, sample2, ...
    # Get sample columns (numeric columns that aren't metadata)
    prot_meta_cols = [
        'protein_group', 'leading_protein', 'leading_name',
        'n_peptides', 'n_unique_peptides', 'low_confidence'
    ]
    prot_meta_cols = [c for c in prot_meta_cols if c in protein_df.columns]
    prot_sample_cols = [
        c for c in protein_df.columns
        if c not in prot_meta_cols and protein_df[c].dtype in ['float64', 'float32', 'int64', 'int32']
    ]

    # Keep a copy of raw protein data for QC comparison
    protein_raw_df = protein_df.copy()

    # Apply global median normalization (on log2 scale)
    prot_sample_medians = protein_df[prot_sample_cols].median()
    global_protein_median = prot_sample_medians.median()
    prot_norm_factors = prot_sample_medians - global_protein_median

    # Apply normalization to each sample column
    for col in prot_sample_cols:
        protein_df[col] = protein_df[col] - prot_norm_factors[col]

    max_prot_shift = prot_norm_factors.abs().max()
    # Report in linear scale for interpretability
    linear_prot_median = 2 ** global_protein_median
    fold_prot_shift = 2 ** max_prot_shift
    logger.info(
        f"  Global median = {linear_prot_median:.0f} (linear), "
        f"max shift = {fold_prot_shift:.2f}x"
    )
    method_log.append(f"Protein median normalization: max shift = {fold_prot_shift:.2f}x")

    # -------------------------------------------------------------------------
    # Stage 4c: Protein ComBat Batch Correction
    # -------------------------------------------------------------------------
    if batch_correction_enabled and metadata_df is not None:
        logger.info("-" * 60)
        logger.info("Stage 4c: Protein ComBat Batch Correction")
        logger.info("-" * 60)

        batch_col = 'batch'
        sample_type_col = 'sample_type'
        # Use sample_id (unique across batches) if available
        if 'sample_id' in metadata_df.columns:
            meta_sample_col = 'sample_id'
        elif 'sample' in metadata_df.columns:
            meta_sample_col = 'sample'
        else:
            meta_sample_col = 'replicate_name'

        if batch_col in metadata_df.columns:
            sample_to_batch = dict(
                zip(metadata_df[meta_sample_col], metadata_df[batch_col])
            )

            # Check if all sample columns have batch info
            prot_batches = [sample_to_batch.get(s) for s in prot_sample_cols]
            n_batches = len(set(b for b in prot_batches if b is not None))

            if n_batches < 2:
                logger.warning(f"Only {n_batches} batch - skipping batch correction")
            else:
                from .batch_correction import combat

                # Prepare data for ComBat (features x samples matrix)
                prot_data_matrix = protein_df[prot_sample_cols].values
                prot_batch_labels = [
                    sample_to_batch.get(s, 'unknown') for s in prot_sample_cols
                ]

                logger.info(f"  Applying ComBat across {n_batches} batches...")

                # Run ComBat - returns corrected matrix directly
                prot_corrected = combat(prot_data_matrix, prot_batch_labels)

                # Replace sample columns with corrected values
                for i, col in enumerate(prot_sample_cols):
                    protein_df[col] = prot_corrected[:, i]

                method_log.append(f"Protein ComBat: {n_batches} batches corrected")

                # Evaluate using reference and QC samples if available
                if sample_type_col in metadata_df.columns:
                    sample_to_type = dict(
                        zip(metadata_df[meta_sample_col], metadata_df[sample_type_col])
                    )
                    ref_cols = [c for c in prot_sample_cols
                                if sample_to_type.get(c) == 'reference']
                    qc_cols = [c for c in prot_sample_cols
                                 if sample_to_type.get(c) == 'qc']

                    if ref_cols and qc_cols:
                        # Calculate CV on LINEAR scale (never on log2!)
                        ref_linear = 2 ** protein_df[ref_cols]
                        qc_linear = 2 ** protein_df[qc_cols]
                        ref_cv = (ref_linear.std(axis=1) / ref_linear.mean(axis=1)) * 100
                        qc_cv = (qc_linear.std(axis=1) / qc_linear.mean(axis=1)) * 100
                        logger.info(
                            f"  Reference median CV: {ref_cv.median():.1f}%, "
                            f"QC median CV: {qc_cv.median():.1f}%"
                        )
        else:
            logger.warning("No batch column in metadata - skipping batch correction")
    elif batch_correction_enabled:
        logger.warning("No metadata available - skipping batch correction")
    else:
        logger.info("  Batch correction disabled in config")

    # =========================================================================
    # Stage 5: Output Generation
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Stage 5: Output Generation")
    logger.info("=" * 60)

    output_format = config['output'].get('format', 'parquet')

    # Data is in wide format with sample values already normalized/corrected
    # Save as-is (values are on log2 scale)
    peptide_output = output_dir / f"corrected_peptides.{output_format}"
    if output_format == 'parquet':
        peptide_df.to_parquet(peptide_output, index=False)
    else:
        sep = '\t' if output_format == 'tsv' else ','
        peptide_df.to_csv(peptide_output, sep=sep, index=False)
    logger.info(f"  Saved peptides: {peptide_output}")

    # Save proteins
    protein_output = output_dir / f"corrected_proteins.{output_format}"
    if output_format == 'parquet':
        protein_df.to_parquet(protein_output, index=False)
    else:
        sep = '\t' if output_format == 'tsv' else ','
        protein_df.to_csv(protein_output, sep=sep, index=False)
    logger.info(f"  Saved proteins: {protein_output}")

    # Generate pipeline metadata
    metadata = {
        'pipeline_version': '0.3.0',
        'processing_date': datetime.now(timezone.utc).isoformat(),
        'source_files': [str(p) for p in input_paths],
        'processing_parameters': {
            'transition_rollup': {
                'method': transition_config.method,
                'min_transitions': transition_config.min_transitions,
                'use_ms1': not transition_config.exclude_precursor,
            },
            'protein_rollup': {
                'method': protein_config.method,
                'shared_peptide_handling': protein_config.shared_peptide_handling,
                'min_peptides': protein_config.min_peptides,
            },
            'batch_correction': {
                'enabled': batch_correction_enabled,
                'method': 'combat_reference_anchored',
            },
        },
        'method_log': method_log,
        'output_files': {
            'peptides': str(peptide_output),
            'proteins': str(protein_output),
            'protein_groups': str(groups_output),
        },
        'statistics': {
            'n_samples': len(samples),
            'n_peptides': peptide_result.n_peptides,
            'n_proteins': protein_result.n_proteins,
            'n_protein_groups': len(protein_groups),
        },
    }
    metadata_output = output_dir / "metadata.json"
    with open(metadata_output, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"  Saved metadata: {metadata_output}")

    # -------------------------------------------------------------------------
    # Stage 5b: Generate QC Report
    # -------------------------------------------------------------------------
    qc_config = config.get('qc_report', {})
    if qc_config.get('enabled', True):
        logger.info("-" * 60)
        logger.info("Stage 5b: Generating QC Report")
        logger.info("-" * 60)

        # Build sample types mapping
        sample_types_map = {}
        if metadata_df is not None:
            # Use sample_id (unique across batches) if available
            if 'sample_id' in metadata_df.columns:
                meta_sample_col = 'sample_id'
            elif 'sample' in metadata_df.columns:
                meta_sample_col = 'sample'
            else:
                meta_sample_col = 'replicate_name'
            if 'sample_type' in metadata_df.columns:
                sample_types_map = dict(
                    zip(metadata_df[meta_sample_col], metadata_df['sample_type'])
                )
        qc_report_path = output_dir / qc_config.get('filename', 'qc_report.html')

        try:
            generate_comprehensive_qc_report(
                peptide_raw=peptide_pre_norm_df,
                peptide_corrected=peptide_df,
                protein_raw=protein_raw_df,
                protein_corrected=protein_df,
                sample_cols=sample_cols,
                sample_types=sample_types_map,
                output_path=qc_report_path,
                method_log=method_log,
                config=config,
                save_plots=qc_config.get('save_plots', True),
                embed_plots=qc_config.get('embed_plots', True),
            )
            method_log.append(f"QC report: {qc_report_path}")
        except Exception as e:
            logger.warning(f"Failed to generate QC report: {e}")

    # Summary
    logger.info("=" * 60)
    logger.info("PRISM Pipeline Complete")
    logger.info("=" * 60)
    for step in method_log:
        logger.info(f"  {step}")
    logger.info(f"  Output: {output_dir}")

    return 0


def cmd_qc(args: argparse.Namespace) -> int:
    """Regenerate QC report from existing PRISM output directory.

    This command reads the processed parquet files from a previous PRISM run
    and regenerates the QC report with current visualization code. Useful for
    updating reports after visualization improvements without reprocessing data.

    Expected files in the output directory:
    - peptides_rollup.2.parquet OR peptides_normalized.3.parquet (raw peptides)
    - corrected_peptides.parquet (corrected peptides)
    - proteins_raw.parquet (raw proteins)
    - corrected_proteins.parquet (corrected proteins)
    - sample_metadata.tsv (sample types)
    """
    import pandas as pd

    output_dir = Path(args.dir)

    if not output_dir.exists():
        logger.error(f"Output directory not found: {output_dir}")
        return 1

    logger.info("=" * 60)
    logger.info("PRISM QC Report Regeneration")
    logger.info("=" * 60)
    logger.info(f"  Directory: {output_dir}")

    # Find peptide raw data (try multiple names for compatibility)
    peptide_raw_path = None
    for name in ['peptides_rollup.2.parquet', 'peptides_normalized.3.parquet',
                 'peptides_rawsum.1.parquet', 'peptides_medianpolish.2.parquet']:
        path = output_dir / name
        if path.exists():
            peptide_raw_path = path
            break

    if peptide_raw_path is None:
        logger.error("Could not find peptide raw data file")
        return 1

    # Load data files
    required_files = {
        'peptide_raw': peptide_raw_path,
        'peptide_corrected': output_dir / 'corrected_peptides.parquet',
        'protein_raw': output_dir / 'proteins_raw.parquet',
        'protein_corrected': output_dir / 'corrected_proteins.parquet',
    }

    for name, path in required_files.items():
        if not path.exists():
            logger.error(f"Required file not found: {path}")
            return 1

    logger.info("Loading processed data...")
    peptide_raw = pd.read_parquet(required_files['peptide_raw'])
    peptide_corrected = pd.read_parquet(required_files['peptide_corrected'])
    protein_raw = pd.read_parquet(required_files['protein_raw'])
    protein_corrected = pd.read_parquet(required_files['protein_corrected'])

    logger.info(f"  Peptides: {len(peptide_corrected):,}")
    logger.info(f"  Proteins: {len(protein_corrected):,}")

    # Load sample metadata for sample types
    sample_types = {}
    metadata_path = output_dir / 'sample_metadata.tsv'
    if metadata_path.exists():
        metadata_df = pd.read_csv(metadata_path, sep='\t')
        if 'sample' in metadata_df.columns and 'sample_type' in metadata_df.columns:
            sample_types = dict(zip(metadata_df['sample'], metadata_df['sample_type']))
            logger.info(f"  Sample types loaded: {len(sample_types)}")
    else:
        logger.warning("sample_metadata.tsv not found, sample types will not be colored")

    # Determine sample columns (exclude metadata columns)
    meta_cols = ['Peptide Modified Sequence Unimod Ids', 'Peptide Modified Sequence',
                 'Peptide', 'n_transitions', 'mean_rt', 'protein', 'n_peptides']
    sample_cols = [c for c in peptide_corrected.columns if c not in meta_cols]
    logger.info(f"  Samples: {len(sample_cols)}")

    # Generate QC report
    report_path = output_dir / args.output
    logger.info(f"Generating QC report: {report_path}")

    try:
        generate_comprehensive_qc_report(
            peptide_raw=peptide_raw,
            peptide_corrected=peptide_corrected,
            protein_raw=protein_raw,
            protein_corrected=protein_corrected,
            sample_cols=sample_cols,
            sample_types=sample_types,
            output_path=report_path,
            method_log=["QC report regenerated from existing data"],
            config={},
            save_plots=not args.no_save_plots,
            embed_plots=not args.no_embed,
        )
        logger.info("QC report generated successfully")
    except Exception as e:
        logger.error(f"Failed to generate QC report: {e}")
        return 1

    return 0


def cmd_config_template(args: argparse.Namespace) -> int:
    """Output an annotated configuration template file.

    This command generates a comprehensive, fully-annotated YAML configuration
    template that serves as a starting point for new PRISM analyses.

    The template includes:
    - All configuration options with default values
    - Detailed comments explaining each option
    - Guidance on when to change settings from defaults
    """
    if args.minimal:
        template = get_minimal_config_template()
    else:
        template = get_full_config_template()

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(template)
        print(f"Configuration template written to: {output_path}")
    else:
        print(template)

    return 0


def get_minimal_config_template() -> str:
    """Return a minimal config template with commonly-changed options."""
    return '''\
# PRISM Minimal Configuration Template
# =====================================
# This template includes only the most commonly-changed options.
# For all options, run: prism config-template
#
# Usage: prism run -i data.csv -o output/ -c this_config.yaml

# =============================================================================
# Sample Type Detection (IMPORTANT - customize for your naming convention)
# =============================================================================
sample_annotations:
  # Patterns to identify inter-experiment reference samples
  # (used for learning technical variation)
  reference_pattern:
    - "-Pool_"
    - "_Pool_"
    - "CommercialPool"

  # Patterns to identify intra-experiment QC samples
  # (used for validating corrections)
  qc_pattern:
    - "-QC_"
    - "_QC_"
    - "StudyPool"

# =============================================================================
# Protein Parsimony (REQUIRED - path to your search FASTA)
# =============================================================================
parsimony:
  # Path to the FASTA database used for the original search
  fasta_path: "/path/to/your/search_database.fasta"
  
  # Strategy for shared peptides: all_groups, unique_only, razor
  shared_peptide_handling: "all_groups"

# =============================================================================
# Processing Options
# =============================================================================
processing:
  # Number of parallel workers (0 = all CPUs, 1 = single-threaded)
  n_workers: 1

transition_rollup:
  # Method: sum (default), median_polish, adaptive, topn
  method: "sum"
  
  # Minimum transitions required per peptide
  min_transitions: 3

protein_rollup:
  # Method: sum (default), median_polish, topn, ibaq, maxlfq
  method: "sum"

# =============================================================================
# Output Options
# =============================================================================
output:
  # File format: parquet, csv, tsv
  format: "parquet"
  
  # Include residuals for outlier analysis
  include_residuals: true

qc_report:
  enabled: true
  save_plots: true
  embed_plots: true
'''


def get_full_config_template() -> str:
    """Return the full annotated config template."""
    return '''\
# =============================================================================
# PRISM Configuration Template
# =============================================================================
# Skyline-PRISM: Proteomics Reference-Integrated Signal Modeling
#
# Normalization and batch correction for LC-MS proteomics data exported from
# Skyline, with robust protein quantification using Tukey median polish and
# reference-anchored batch correction.
#
# This template includes ALL configuration options with detailed documentation.
# Copy this file and modify for your experiment.
#
# Usage:
#   prism run -i skyline_report.csv -o output_dir/ -c config.yaml
#
# For a minimal template with only common options, run:
#   prism config-template --minimal
#
# See https://github.com/maccoss/skyline-prism for full documentation.
# =============================================================================

# =============================================================================
# Data Column Mapping
# =============================================================================
# Map your Skyline report column names to PRISM internal names.
# These are the defaults for standard Skyline exports.
#
# IMPORTANT: Verify these match your Skyline report column headers.

data:
  # Abundance measurement column (usually from transitions)
  abundance_column: "Area"
  
  # Retention time column
  rt_column: "Retention Time"
  
  # Peptide identification (modified sequence distinguishes peptidoforms)
  peptide_column: "Peptide Modified Sequence"
  
  # Protein identification
  protein_column: "Protein Accession"
  protein_name_column: "Protein Name"
  
  # Sample identification
  sample_column: "Replicate Name"

# =============================================================================
# Sample Type Detection
# =============================================================================
# Patterns to automatically identify sample types from replicate names.
# These are substring matches (case-sensitive) checked in order.
#
# Sample types:
# - reference: Inter-experiment reference (e.g., commercial plasma pool)
#              Used for learning technical variation (RT correction, etc.)
# - qc: Intra-experiment QC (e.g., pooled study samples)
#       Used for validating corrections without overfitting
# - experimental: All other samples

sample_annotations:
  # Inter-experiment reference samples
  # CUSTOMIZE these patterns for your sample naming convention
  reference_pattern:
    - "-Pool_"
    - "_Pool_"
    - "GoldenWest"
    - "CommercialPool"
    - "InterExpRef"
  
  # Intra-experiment QC samples
  qc_pattern:
    - "-QC_"
    - "_QC_"
    - "StudyPool"
    - "IntraPool"

# =============================================================================
# Batch Estimation
# =============================================================================
# When batch information is not provided in metadata, PRISM estimates batches
# from acquisition timestamps or source document names.
#
# TIP: Include "Result File > Acquired Time" in your Skyline report for best
# automatic batch detection.

batch_estimation:
  # Expected samples per batch (for validation)
  min_samples_per_batch: 12
  max_samples_per_batch: 100
  
  # IQR multiplier for detecting batch breaks from time gaps
  # Higher = fewer batch breaks detected
  gap_iqr_multiplier: 1.5
  
  # Force a specific number of batches (null = automatic)
  n_batches: null

# =============================================================================
# Sample Outlier Detection
# =============================================================================
# Detect samples with abnormally low signal (potential failed injections).
# Detection is one-sided (only flags low outliers, not high).

sample_outlier_detection:
  enabled: true
  
  # Action when outliers detected: "report" (log only) or "exclude"
  action: "report"
  
  # Detection method: "iqr" or "fold_median"
  method: "iqr"
  
  # IQR multiplier (for iqr method): lower = more aggressive detection
  iqr_multiplier: 1.5
  
  # Fold threshold (for fold_median method): e.g., 0.1 = 10% of median
  fold_threshold: 0.1

# =============================================================================
# Processing Performance
# =============================================================================
# Parallel processing settings. Adjust based on your machine.

processing:
  # Number of parallel workers for peptide rollup
  #   1 = single-threaded (lowest memory, recommended for most cases)
  #   0 = use all available CPUs
  #   N = use N worker processes
  n_workers: 1
  
  # Peptides per batch (larger = faster but more memory)
  peptide_batch_size: 1000

# =============================================================================
# Transition to Peptide Rollup
# =============================================================================
# Aggregate transition-level data to peptide-level quantities.
# Only used if input is transition-level data from Skyline.

transition_rollup:
  enabled: true
  
  # Rollup method:
  #   sum          - Simple sum of fragment intensities (default, robust)
  #   median_polish - Tukey median polish (robust to interference)
  #   adaptive     - Learned weights based on intensity, m/z, shape correlation
  #   topn         - Select top N transitions by correlation
  method: "sum"
  
  # Include MS1 precursor signal? (false = MS2 only, recommended)
  use_ms1: false
  
  # Minimum transitions required per peptide
  min_transitions: 3
  
  # Top-N method parameters (only used if method: topn)
  topn_count: 3
  topn_selection: "correlation"  # or "intensity"
  topn_weighting: "sqrt"         # or "sum"
  
  # Adaptive method parameters (only used if method: adaptive)
  learn_weights: false
  adaptive_rollup:
    beta_relative_intensity: 0.0
    beta_mz: 0.0
    beta_shape_corr: 0.0
    beta_shape_corr_outlier: 0.0
    shape_corr_low_threshold: 0.7
    min_improvement_pct: 5.0

# =============================================================================
# RT-Aware Normalization (DISABLED BY DEFAULT)
# =============================================================================
# Correct RT-dependent technical variation using spline models fit to
# reference samples.
#
# NOTE: Disabled by default. Search engines (DIA-NN, etc.) often apply
# per-file RT calibration that doesn't generalize between samples.

rt_correction:
  enabled: false
  method: "spline"
  spline_df: 5
  per_batch: true
  min_peptides_per_bin: 10

# =============================================================================
# Global Normalization
# =============================================================================
# Correct for overall sample loading differences.

global_normalization:
  # Method:
  #   median   - Subtract sample median (recommended)
  #   vsn      - Variance Stabilizing Normalization
  #   quantile - Force identical distributions (aggressive)
  #   none     - Skip normalization
  method: "median"
  
  # VSN parameters (only used if method: vsn)
  vsn_params:
    optimize_params: false

# =============================================================================
# Batch Correction
# =============================================================================
# Remove systematic batch effects using ComBat (empirical Bayes).

batch_correction:
  enabled: true
  method: "combat"

# =============================================================================
# Protein Parsimony
# =============================================================================
# Handle shared peptides that map to multiple proteins.
#
# IMPORTANT: Requires the FASTA database used for the original search.

parsimony:
  # Path to search FASTA database (REQUIRED)
  fasta_path: null  # e.g., "/path/to/uniprot_human_reviewed.fasta"
  
  # Strategy for shared peptides:
  #   all_groups  - Apply to all protein groups (recommended)
  #   unique_only - Only use peptides unique to one protein
  #   razor       - Assign to group with most peptides (MaxQuant-style)
  shared_peptide_handling: "all_groups"

# =============================================================================
# Protein Rollup
# =============================================================================
# Combine peptides into protein-level quantities.

protein_rollup:
  # Method:
  #   sum           - Simple sum (default)
  #   median_polish - Tukey median polish (robust to outliers)
  #   topn          - Average of top N most intense peptides
  #   ibaq          - Intensity-Based Absolute Quantification
  #   maxlfq        - Maximum LFQ algorithm (MaxQuant-style)
  method: "sum"
  
  # Minimum peptides required per protein
  min_peptides: 2
  
  # Top-N parameters (if method: topn)
  topn:
    n: 3
    selection: "median_abundance"
  
  # iBAQ parameters (if method: ibaq)
  ibaq:
    fasta_path: null
    enzyme: "trypsin"
    missed_cleavages: 0
    min_peptide_length: 6
    max_peptide_length: 30
  
  # Median polish parameters
  median_polish:
    max_iterations: 20
    convergence_tolerance: 0.0001

# =============================================================================
# Output Options
# =============================================================================

output:
  # File format: parquet (recommended), csv, tsv
  format: "parquet"
  
  # Include median polish residuals for outlier analysis
  include_residuals: true
  
  # Compress output files
  compress: true

# =============================================================================
# QC Report
# =============================================================================
# HTML report with diagnostic plots for quality assessment.

qc_report:
  enabled: true
  filename: "qc_report.html"
  
  # Save individual PNG plots
  save_plots: true
  
  # Embed plots in HTML (makes file self-contained but larger)
  embed_plots: true
  
  # Plots to include
  plots:
    intensity_distribution: true
    pca_comparison: true
    control_correlation: true
    cv_distribution: true
    rt_correction: true
'''


def main() -> int:
    """Provide main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog='prism',
        description='Skyline-PRISM: Proteomics Reference-Integrated Signal Modeling\n\n'
                    'Normalization and batch correction for Skyline proteomics data\n'
                    'with robust protein quantification using Tukey median polish.\n\n'
                    'Primary usage:\n'
                    '  prism run -i data.parquet -o output_dir/ -c config.yaml\n\n'
                    'See https://github.com/maccoss/skyline-prism for documentation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1.0')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Run command (primary) - executes full pipeline
    run_parser = subparsers.add_parser(
        'run',
        help='Run the full PRISM pipeline (recommended)',
        description='Execute the complete PRISM pipeline: normalization, batch correction, '
                    'and protein rollup. Produces both peptide-level and protein-level outputs.'
    )
    run_parser.add_argument('-i', '--input', nargs='+', required=True,
                           help='Input file(s): one or more Skyline report CSV/TSV files, '
                                'or a single merged parquet file. Multiple CSV files are '
                                'treated as separate batches. If a merged_data.parquet '
                                'already exists in the output directory and matches the '
                                'input files, it will be reused (use --force-reprocess to '
                                'override).')
    run_parser.add_argument('-o', '--output-dir', required=True,
                           help='Output directory for results')
    run_parser.add_argument('-c', '--config', help='Configuration YAML file')
    run_parser.add_argument('-m', '--metadata', help='Sample metadata TSV (optional - '
                           'will auto-generate from sample names if not provided)')
    run_parser.add_argument('--reference-pattern', nargs='+',
                           help='Patterns to identify reference samples (e.g., -Pool_). '
                                'Samples matching these are used for RT correction learning.')
    run_parser.add_argument('--qc-pattern', nargs='+',
                           help='Patterns to identify QC samples (e.g., -Carl_). '
                                'Samples matching these are used for validation.')
    run_parser.add_argument(
        '--from-provenance',
        help='Load configuration from a previous run\'s metadata.json file. '
             'Enables reproducibility by using the exact same parameters. '
             'If --config is also provided, it overrides provenance settings.'
    )
    run_parser.add_argument(
        '--force-reprocess',
        action='store_true',
        help='Force reprocessing of input files even if a valid cached parquet '
             'exists. By default, if a merged_data.parquet file exists and its '
             'source file fingerprints match the input files, it will be reused.'
    )

    # Merge command - utility for combining reports
    merge_parser = subparsers.add_parser(
        'merge', help='Merge multiple Skyline reports into one parquet file'
    )
    merge_parser.add_argument('reports', nargs='+', help='Skyline report files')
    merge_parser.add_argument('-o', '--output', required=True, help='Output parquet path')
    merge_parser.add_argument('-m', '--metadata', help='Sample metadata TSV')
    merge_parser.add_argument('--no-partition', action='store_true',
                             help='Do not partition by batch')

    # QC report command - regenerate QC report from existing output
    qc_parser = subparsers.add_parser(
        'qc', help='Regenerate QC report from existing PRISM output'
    )
    qc_parser.add_argument(
        '-d', '--dir', required=True,
        help='PRISM output directory containing processed parquet files'
    )
    qc_parser.add_argument(
        '-o', '--output', default='qc_report.html',
        help='Output HTML report filename (default: qc_report.html)'
    )
    qc_parser.add_argument(
        '--no-save-plots', action='store_true',
        help='Do not save individual plot PNG files'
    )
    qc_parser.add_argument(
        '--no-embed', action='store_true',
        help='Do not embed plots in HTML (link to files instead)'
    )

    # Config template command - output annotated config template
    config_parser = subparsers.add_parser(
        'config-template',
        help='Output an annotated configuration template file',
        description='Generate a fully-annotated configuration template YAML file. '
                    'This is the recommended starting point for configuring a new analysis.'
    )
    config_parser.add_argument(
        '-o', '--output',
        help='Output file path (default: print to stdout)'
    )
    config_parser.add_argument(
        '--minimal',
        action='store_true',
        help='Output minimal config with only commonly-changed options'
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.command == 'run':
        return cmd_run(args)
    elif args.command == 'merge':
        return cmd_merge(args)
    elif args.command == 'qc':
        return cmd_qc(args)
    elif args.command == 'config-template':
        return cmd_config_template(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
