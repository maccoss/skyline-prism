"""
Command-line interface for RT-aware normalization.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml

from .data_io import (
    load_skyline_report,
    merge_skyline_reports,
    load_sample_metadata,
    load_unified_data,
)
from .parsimony import (
    build_peptide_protein_map,
    compute_protein_groups,
    export_protein_groups,
    annotate_peptides_with_groups,
)
from .normalization import normalize_pipeline
from .rollup import rollup_to_proteins
from .validation import validate_correction, generate_qc_report


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def load_config(config_path: Optional[Path]) -> dict:
    """Load configuration from YAML file or return defaults."""
    defaults = {
        'data': {
            'abundance_column': 'TotalAreaFragment',
            'rt_column': 'BestRetentionTime',
            'peptide_column': 'PeptideModifiedSequence',
            'protein_column': 'ProteinAccession',
            'sample_column': 'ReplicateName',
            'batch_column': 'Batch',
        },
        'rt_correction': {
            'enabled': True,
            'method': 'spline',
            'spline_df': 5,
            'per_batch': True,
        },
        'global_normalization': {
            'method': 'median',
        },
        'parsimony': {
            'shared_peptide_handling': 'all_groups',
        },
        'protein_rollup': {
            'method': 'median_polish',
            'min_peptides': 3,
        },
    }
    
    if config_path and config_path.exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f)
        # Merge user config over defaults
        for section, values in user_config.items():
            if section in defaults and isinstance(values, dict):
                defaults[section].update(values)
            else:
                defaults[section] = values
    
    return defaults


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
    protein_df, polish_results = rollup_to_proteins(
        data,
        protein_groups,
        abundance_col=config['data'].get('abundance_column', 'abundance'),
        sample_col=config['data'].get('sample_column', 'replicate_name'),
        peptide_col=config['data'].get('peptide_column', 'peptide_modified'),
        method=config['protein_rollup'].get('method', 'median_polish'),
        min_peptides=config['protein_rollup'].get('min_peptides', 3),
        shared_peptide_handling=config['parsimony'].get('shared_peptide_handling', 'all_groups'),
    )
    
    # Save protein data
    output_path = Path(args.output)
    protein_df.to_parquet(output_path)
    logger.info(f"Saved {len(protein_df)} proteins to {output_path}")
    
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate normalization quality."""
    logger = logging.getLogger(__name__)
    
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


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='RT-aware normalization for proteomics data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge Skyline reports')
    merge_parser.add_argument('reports', nargs='+', help='Skyline report files')
    merge_parser.add_argument('-o', '--output', required=True, help='Output parquet path')
    merge_parser.add_argument('-m', '--metadata', help='Sample metadata TSV')
    merge_parser.add_argument('--no-partition', action='store_true', 
                             help='Do not partition by batch')
    
    # Normalize command
    norm_parser = subparsers.add_parser('normalize', help='Run normalization pipeline')
    norm_parser.add_argument('-i', '--input', required=True, help='Input parquet')
    norm_parser.add_argument('-o', '--output', required=True, help='Output parquet')
    norm_parser.add_argument('-c', '--config', help='Configuration YAML')
    
    # Rollup command
    rollup_parser = subparsers.add_parser('rollup', help='Roll up peptides to proteins')
    rollup_parser.add_argument('-i', '--input', required=True, help='Input peptide parquet')
    rollup_parser.add_argument('-o', '--output', required=True, help='Output protein parquet')
    rollup_parser.add_argument('-g', '--groups-output', help='Output protein groups TSV')
    rollup_parser.add_argument('-c', '--config', help='Configuration YAML')
    
    # Validate command
    val_parser = subparsers.add_parser('validate', help='Validate normalization')
    val_parser.add_argument('--before', required=True, help='Data before normalization')
    val_parser.add_argument('--after', required=True, help='Data after normalization')
    val_parser.add_argument('--report', help='Output HTML report path')
    val_parser.add_argument('--sample-type-col', default='sample_type')
    val_parser.add_argument('--abundance-before', default='abundance')
    val_parser.add_argument('--abundance-after', default='abundance_normalized')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    if args.command == 'merge':
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
