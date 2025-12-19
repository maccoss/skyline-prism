"""Command-line interface for Skyline-PRISM.

PRISM: Proteomics Reference-Integrated Signal Modeling

RT-aware normalization for Skyline proteomics data with robust protein quantification.
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

import yaml

if TYPE_CHECKING:
    import pandas as pd

from .batch_correction import combat_from_long
from .data_io import (
    apply_batch_estimation,
    load_sample_metadata,
    load_skyline_report,
    load_unified_data,
    merge_skyline_reports,
)
from .normalization import normalize_pipeline
from .parsimony import (
    build_peptide_protein_map,
    compute_protein_groups,
    export_protein_groups,
)
from .rollup import (
    extract_peptide_residuals,
    extract_transition_residuals,
    rollup_to_proteins,
)
from .transition_rollup import (
    learn_variance_model,
    rollup_transitions_to_peptides,
)
from .validation import generate_qc_report, validate_correction

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def load_config(config_path: Path | None) -> dict:
    """Load configuration from YAML file or return defaults."""
    defaults = {
        'data': {
            'abundance_column': 'TotalAreaFragment',
            'rt_column': 'BestRetentionTime',
            'peptide_column': 'PeptideModifiedSequence',
            'protein_column': 'ProteinAccession',
            'sample_column': 'ReplicateName',
            'batch_column': 'Batch',
            'sample_type_column': 'SampleType',
        },
        'transition_rollup': {
            'enabled': False,
            'method': 'median_polish',
            'min_transitions': 3,
            'learn_variance_model': False,  # Learn from reference samples
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
            'median_polish': {'max_iterations': 10, 'convergence_tolerance': 0.0001},
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


def cmd_normalize(args: argparse.Namespace) -> int:
    """Run normalization pipeline."""
    logger = logging.getLogger(__name__)

    config = load_config(Path(args.config) if args.config else None)

    # Load data
    input_path = Path(args.input)
    data = load_unified_data(input_path)

    logger.info(f"Loaded {len(data)} rows from {input_path}")

    # Run normalization
    result = normalize_pipeline(
        data,
        sample_type_col=config['data'].get('sample_type_column', 'sample_type'),
        precursor_col=config['data'].get('peptide_column', 'precursor_id'),
        abundance_col=config['data'].get('abundance_column', 'abundance'),
        rt_col=config['data'].get('rt_column', 'retention_time'),
        replicate_col=config['data'].get('sample_column', 'replicate_name'),
        batch_col=config['data'].get('batch_column', 'batch'),
        rt_correction=config['rt_correction'].get('enabled', True),
        global_method=config['global_normalization'].get('method', 'median'),
        spline_df=config['rt_correction'].get('spline_df', 5),
        per_batch=config['rt_correction'].get('per_batch', True),
    )

    # Save normalized peptides
    output_path = Path(args.output)
    result.normalized_data.to_parquet(output_path, index=False)
    logger.info(f"Saved normalized peptides to {output_path}")

    # Log processing steps
    for step in result.method_log:
        logger.info(f"  {step}")

    return 0


def cmd_rollup(args: argparse.Namespace) -> int:
    """Roll up peptides to proteins."""
    logger = logging.getLogger(__name__)

    config = load_config(Path(args.config) if args.config else None)

    # Load peptide data
    data = load_unified_data(Path(args.input))

    # Build peptide-protein mapping
    pep_to_prot, prot_to_pep, prot_to_name = build_peptide_protein_map(
        data,
        peptide_col=config['data'].get('peptide_column', 'peptide_modified'),
        protein_col=config['data'].get('protein_column', 'protein_ids'),
    )

    # Compute protein groups
    protein_groups = compute_protein_groups(prot_to_pep, pep_to_prot, prot_to_name)

    # Export protein groups
    if args.groups_output:
        export_protein_groups(protein_groups, args.groups_output)

    # Roll up to proteins
    protein_df, polish_results, topn_results = rollup_to_proteins(
        data,
        protein_groups,
        abundance_col=config['data'].get('abundance_column', 'abundance'),
        sample_col=config['data'].get('sample_column', 'replicate_name'),
        peptide_col=config['data'].get('peptide_column', 'peptide_modified'),
        method=config['protein_rollup'].get('method', 'median_polish'),
        min_peptides=config['protein_rollup'].get('min_peptides', 3),
        shared_peptide_handling=config['parsimony'].get('shared_peptide_handling', 'all_groups'),
        topn_n=config['protein_rollup'].get('topn', {}).get('n', 3),
        topn_selection=config['protein_rollup'].get('topn', {}).get('selection', 'median_abundance'),
    )

    # Save protein data
    output_path = Path(args.output)
    protein_df.to_parquet(output_path)
    logger.info(f"Saved {len(protein_df)} proteins to {output_path}")

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate normalization quality."""
    logging.getLogger(__name__)

    before_data = load_unified_data(Path(args.before))
    after_data = load_unified_data(Path(args.after))

    metrics = validate_correction(
        before_data,
        after_data,
        sample_type_col=args.sample_type_col,
        abundance_col_before=args.abundance_before,
        abundance_col_after=args.abundance_after,
    )

    if args.report:
        generate_qc_report(
            metrics,
            normalization_log=['Loaded from files'],  # TODO: pass actual log
            output_path=args.report,
        )

    return 0 if metrics.passed else 1


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
        'n_pool': sample_counts.get('pool', 0),
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

    # Build processing parameters from config
    processing_parameters = {
        'transition_rollup': config.get('transition_rollup', {}),
        'rt_correction': config.get('rt_correction', {}),
        'global_normalization': config.get('global_normalization', {}),
        'batch_correction': config.get('batch_correction', {}),
        'protein_rollup': config.get('protein_rollup', {}),
        'parsimony': config.get('parsimony', {}),
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
    """Run the full PRISM pipeline.

    Pipeline stages:
    1. Load and merge input data
    2. Transition → Peptide rollup (if enabled/needed)
    3. RT correction (if enabled)
    4. Global normalization
    5. Two-arm split:
       a. Peptide arm: Batch correction → peptide output
       b. Protein arm: Rollup → Batch correction → protein output
    """
    config = load_config(Path(args.config) if args.config else None)
    method_log = []

    # =========================================================================
    # Stage 0: Load input data
    # =========================================================================
    input_path = Path(args.input)

    if input_path.suffix in ['.csv', '.tsv', '.txt']:
        # Single Skyline report
        data = load_skyline_report(input_path)
        method_log.append(f"Loaded Skyline report: {input_path}")
    else:
        # Parquet file (already merged)
        data = load_unified_data(input_path)
        method_log.append(f"Loaded data from: {input_path}")

    logger.info(f"Loaded {len(data)} rows, {data[config['data']['sample_column']].nunique()} samples")

    # Load metadata if provided
    metadata_df = None
    if args.metadata:
        metadata_df = load_sample_metadata(Path(args.metadata))
        # Merge metadata with data
        data = data.merge(
            metadata_df,
            left_on=config['data']['sample_column'],
            right_on=metadata_df.columns[0],  # Assume first column is sample ID
            how='left'
        )
        method_log.append(f"Merged sample metadata from: {args.metadata}")

    # Get column names from config
    abundance_col = config['data']['abundance_column']
    peptide_col = config['data']['peptide_column']
    protein_col = config['data']['protein_column']
    sample_col = config['data']['sample_column']
    batch_col = config['data']['batch_column']
    sample_type_col = config['data'].get('sample_type_column', 'sample_type')
    rt_col = config['data'].get('rt_column', 'BestRetentionTime')

    # =========================================================================
    # Batch estimation if batch column is missing
    # =========================================================================
    if batch_col not in data.columns or data[batch_col].isna().all():
        logger.info("Batch column not found - estimating batches...")
        data, batch_result = apply_batch_estimation(
            data,
            metadata=metadata_df,
            min_samples_per_batch=config.get('batch_estimation', {}).get(
                'min_samples_per_batch', 12
            ),
            max_samples_per_batch=config.get('batch_estimation', {}).get(
                'max_samples_per_batch', 100
            ),
            gap_iqr_multiplier=config.get('batch_estimation', {}).get(
                'gap_iqr_multiplier', 1.5
            ),
            n_batches_fallback=config.get('batch_estimation', {}).get(
                'n_batches', None
            ),
        )
        method_log.append(
            f"Batch estimation: {batch_result.n_batches} batches via "
            f"'{batch_result.method}'"
        )
        for warning in batch_result.warnings:
            logger.warning(warning)
            method_log.append(f"Warning: {warning}")

    # Preserve raw abundance before any corrections (for output)
    data['abundance_raw'] = data[abundance_col].copy()

    # =========================================================================
    # Stage 1: Transition → Peptide rollup (if enabled)
    # =========================================================================
    transition_residuals = None
    learned_variance_params = None

    if config['transition_rollup'].get('enabled', False):
        logger.info("Running transition → peptide rollup...")

        rollup_method = config['transition_rollup'].get('method', 'median_polish')
        variance_params = None

        # Learn variance model from reference samples if using quality_weighted
        if (
            rollup_method == 'quality_weighted'
            and config['transition_rollup'].get('learn_variance_model', False)
        ):
            # Identify reference samples
            if sample_type_col in data.columns:
                reference_samples = data.loc[
                    data[sample_type_col] == 'reference', sample_col
                ].unique().tolist()
            else:
                # Try to detect from sample names using config patterns
                import re
                ref_pattern = config.get('sample_annotations', {}).get(
                    'reference_pattern', 'Reference|GoldenWest'
                )
                reference_samples = [
                    s for s in data[sample_col].unique()
                    if re.search(ref_pattern, s, re.IGNORECASE)
                ]

            if len(reference_samples) >= 2:
                logger.info(
                    f"Learning variance model from "
                    f"{len(reference_samples)} reference samples..."
                )
                learned_variance_params = learn_variance_model(
                    data,
                    reference_samples=reference_samples,
                    peptide_col=peptide_col,
                    sample_col=sample_col,
                    abundance_col=abundance_col,
                    shape_corr_col=config['transition_rollup'].get(
                        'quality_columns', {}
                    ).get('shape_correlation', 'shape_correlation'),
                    coeluting_col=config['transition_rollup'].get(
                        'quality_columns', {}
                    ).get('coeluting', 'coeluting'),
                )
                variance_params = learned_variance_params
                method_log.append("Learned variance model from reference samples")
            else:
                logger.warning(
                    f"Need ≥2 reference samples to learn variance model, "
                    f"found {len(reference_samples)}. Using defaults."
                )

        rollup_result = rollup_transitions_to_peptides(
            data,
            abundance_col=abundance_col,
            peptide_col=peptide_col,
            sample_col=sample_col,
            method=rollup_method,
            min_transitions=config['transition_rollup'].get('min_transitions', 3),
            params=variance_params,
        )

        data = rollup_result.peptide_data
        abundance_col = 'abundance'  # Rollup creates this column
        method_log.append(f"Transition rollup: {rollup_method}")

        # Extract transition residuals if using median polish
        if (
            config['output'].get('include_residuals', True)
            and rollup_result.median_polish_results
        ):
            transition_residuals = extract_transition_residuals(rollup_result)
            method_log.append("Extracted transition-level residuals")

    # =========================================================================
    # Stage 2: RT correction (if enabled)
    # =========================================================================
    if config['rt_correction'].get('enabled', False):
        logger.info("Running RT-aware correction...")

        norm_result = normalize_pipeline(
            data,
            sample_type_col=sample_type_col,
            precursor_col=peptide_col,
            abundance_col=abundance_col,
            rt_col=rt_col,
            replicate_col=sample_col,
            batch_col=batch_col,
            rt_correction=True,
            global_method=config['global_normalization'].get('method', 'median'),
            spline_df=config['rt_correction'].get('spline_df', 5),
            per_batch=config['rt_correction'].get('per_batch', True),
            batch_correction=False,  # We do batch correction separately per arm
        )

        data = norm_result.normalized_data
        abundance_col = 'abundance_normalized'
        method_log.extend(norm_result.method_log)
    else:
        # Just do global normalization without RT correction
        logger.info("Running global normalization (RT correction disabled)...")

        norm_result = normalize_pipeline(
            data,
            sample_type_col=sample_type_col,
            precursor_col=peptide_col,
            abundance_col=abundance_col,
            rt_col=rt_col,
            replicate_col=sample_col,
            batch_col=batch_col,
            rt_correction=False,
            global_method=config['global_normalization'].get('method', 'median'),
            batch_correction=False,
        )

        data = norm_result.normalized_data
        abundance_col = 'abundance_normalized'
        method_log.extend(norm_result.method_log)

    # =========================================================================
    # Stage 3: Build protein groups (needed for protein rollup)
    # =========================================================================
    logger.info("Computing protein groups...")

    pep_to_prot, prot_to_pep, prot_to_name = build_peptide_protein_map(
        data,
        peptide_col=peptide_col,
        protein_col=protein_col,
    )

    protein_groups = compute_protein_groups(prot_to_pep, pep_to_prot, prot_to_name)
    method_log.append(f"Computed {len(protein_groups)} protein groups")

    # =========================================================================
    # Stage 4a: Peptide arm - Batch correction
    # =========================================================================
    peptide_data = data.copy()

    if config['batch_correction'].get('enabled', True):
        logger.info("Applying batch correction to peptides...")

        # Check if we have batch information
        if batch_col in peptide_data.columns and peptide_data[batch_col].nunique() > 1:
            peptide_data = combat_from_long(
                peptide_data,
                abundance_col=abundance_col,
                feature_col=peptide_col,
                sample_col=sample_col,
                batch_col=batch_col,
            )
            method_log.append("Peptide batch correction: ComBat")
        else:
            logger.info("Skipping peptide batch correction: single batch detected")
            method_log.append("Peptide batch correction: skipped (single batch)")
    else:
        pass

    # =========================================================================
    # Stage 4b: Protein arm - Rollup then batch correction
    # =========================================================================
    logger.info("Rolling up peptides to proteins...")

    protein_df, polish_results, topn_results = rollup_to_proteins(
        data,  # Use pre-batch-corrected data for rollup
        protein_groups,
        abundance_col=abundance_col,
        sample_col=sample_col,
        peptide_col=peptide_col,
        method=config['protein_rollup'].get('method', 'median_polish'),
        shared_peptide_handling=config['parsimony'].get('shared_peptide_handling', 'all_groups'),
        topn_n=config['protein_rollup'].get('topn', {}).get('n', 3),
        topn_selection=config['protein_rollup'].get('topn', {}).get('selection', 'median_abundance'),
    )

    method_log.append(f"Protein rollup: {config['protein_rollup']['method']}")

    # Extract peptide residuals from protein rollup if using median polish
    peptide_residuals = None
    if (config['output'].get('include_residuals', True) and
        config['protein_rollup'].get('method') == 'median_polish' and
        polish_results):
        peptide_residuals = extract_peptide_residuals(polish_results)
        method_log.append("Extracted peptide-level residuals from protein rollup")

    # Batch correct proteins
    if config['batch_correction'].get('enabled', True):
        logger.info("Applying batch correction to proteins...")

        # Protein data needs batch info - merge from sample metadata
        sample_batch = data[[sample_col, batch_col]].drop_duplicates()
        protein_df = protein_df.merge(sample_batch, on=sample_col, how='left')

        if batch_col in protein_df.columns and protein_df[batch_col].nunique() > 1:
            protein_df = combat_from_long(
                protein_df,
                abundance_col='abundance',
                feature_col='protein_group',
                sample_col=sample_col,
                batch_col=batch_col,
            )
            method_log.append("Protein batch correction: ComBat")
        else:
            logger.info("Skipping protein batch correction: single batch detected")
            method_log.append("Protein batch correction: skipped (single batch)")
    else:
        pass

    # =========================================================================
    # Stage 5: Save outputs
    # =========================================================================
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_format = config['output'].get('format', 'parquet')

    # Save peptide data
    peptide_output = output_dir / f"corrected_peptides.{output_format}"
    if output_format == 'parquet':
        peptide_data.to_parquet(peptide_output, index=False)
    elif output_format == 'csv':
        peptide_data.to_csv(peptide_output, index=False)
    elif output_format == 'tsv':
        peptide_data.to_csv(peptide_output, sep='\t', index=False)
    logger.info(f"Saved peptides to {peptide_output}")

    # Save protein data
    protein_output = output_dir / f"corrected_proteins.{output_format}"
    if output_format == 'parquet':
        protein_df.to_parquet(protein_output, index=False)
    elif output_format == 'csv':
        protein_df.to_csv(protein_output, index=False)
    elif output_format == 'tsv':
        protein_df.to_csv(protein_output, sep='\t', index=False)
    logger.info(f"Saved proteins to {protein_output}")

    # Save protein groups
    groups_output = output_dir / "protein_groups.tsv"
    export_protein_groups(protein_groups, str(groups_output))
    logger.info(f"Saved protein groups to {groups_output}")

    # Save residuals if requested
    if config['output'].get('include_residuals', True):
        if peptide_residuals is not None:
            residuals_output = output_dir / f"peptide_residuals.{output_format}"
            if output_format == 'parquet':
                peptide_residuals.to_parquet(residuals_output, index=False)
            else:
                peptide_residuals.to_csv(residuals_output, sep='\t' if output_format == 'tsv' else ',', index=False)
            logger.info(f"Saved peptide residuals to {residuals_output}")

        if transition_residuals is not None:
            residuals_output = output_dir / f"transition_residuals.{output_format}"
            if output_format == 'parquet':
                transition_residuals.to_parquet(residuals_output, index=False)
            else:
                transition_residuals.to_csv(residuals_output, sep='\t' if output_format == 'tsv' else ',', index=False)
            logger.info(f"Saved transition residuals to {residuals_output}")

    # Generate and save metadata JSON
    input_files = [str(args.input_data)]
    if args.metadata:
        input_files.append(str(args.metadata))

    metadata = generate_pipeline_metadata(
        config=config,
        data=data,
        protein_groups=protein_groups,
        method_log=method_log,
        input_files=input_files,
        sample_type_col=sample_type_col,
        sample_col=sample_col,
        batch_col=batch_col,
    )

    metadata_output = output_dir / "metadata.json"
    with open(metadata_output, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Saved pipeline metadata to {metadata_output}")

    # Log summary
    logger.info("=" * 60)
    logger.info("PRISM Pipeline Complete")
    logger.info("=" * 60)
    for step in method_log:
        logger.info(f"  {step}")
    logger.info(f"Output directory: {output_dir}")

    return 0


def main() -> int:
    """Main entry point."""
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
    run_parser.add_argument('-i', '--input', required=True,
                           help='Input file (Skyline report CSV/TSV or merged parquet)')
    run_parser.add_argument('-o', '--output-dir', required=True,
                           help='Output directory for results')
    run_parser.add_argument('-c', '--config', help='Configuration YAML file')
    run_parser.add_argument('-m', '--metadata', help='Sample metadata TSV')

    # Merge command - utility for combining reports
    merge_parser = subparsers.add_parser('merge', help='Merge multiple Skyline reports into one file')
    merge_parser.add_argument('reports', nargs='+', help='Skyline report files')
    merge_parser.add_argument('-o', '--output', required=True, help='Output parquet path')
    merge_parser.add_argument('-m', '--metadata', help='Sample metadata TSV')
    merge_parser.add_argument('--no-partition', action='store_true',
                             help='Do not partition by batch')

    # Legacy commands (kept for backwards compatibility)
    norm_parser = subparsers.add_parser('normalize', help='(Legacy) Run normalization only')
    norm_parser.add_argument('-i', '--input', required=True, help='Input parquet')
    norm_parser.add_argument('-o', '--output', required=True, help='Output parquet')
    norm_parser.add_argument('-c', '--config', help='Configuration YAML')

    rollup_parser = subparsers.add_parser('rollup', help='(Legacy) Roll up peptides to proteins only')
    rollup_parser.add_argument('-i', '--input', required=True, help='Input peptide parquet')
    rollup_parser.add_argument('-o', '--output', required=True, help='Output protein parquet')
    rollup_parser.add_argument('-g', '--groups-output', help='Output protein groups TSV')
    rollup_parser.add_argument('-c', '--config', help='Configuration YAML')

    # Validate command
    val_parser = subparsers.add_parser('validate', help='Validate normalization quality')
    val_parser.add_argument('--before', required=True, help='Data before normalization')
    val_parser.add_argument('--after', required=True, help='Data after normalization')
    val_parser.add_argument('--report', help='Output HTML report path')
    val_parser.add_argument('--sample-type-col', default='sample_type')
    val_parser.add_argument('--abundance-before', default='abundance')
    val_parser.add_argument('--abundance-after', default='abundance_normalized')

    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.command == 'run':
        return cmd_run(args)
    elif args.command == 'merge':
        return cmd_merge(args)
    elif args.command == 'normalize':
        return cmd_normalize(args)
    elif args.command == 'rollup':
        return cmd_rollup(args)
    elif args.command == 'validate':
        return cmd_validate(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
