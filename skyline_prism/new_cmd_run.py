def cmd_run(args: argparse.Namespace) -> int:
    """Run the full PRISM pipeline using streaming processing.

    This is the memory-efficient version that processes data without loading
    the entire dataset into memory.

    Pipeline stages:
    1. Merge input CSVs to parquet (streaming)
    2. Transition -> Peptide rollup (streaming, sorted by peptide)
    3. Protein parsimony (from peptide-protein mappings)
    4. Peptide -> Protein rollup (streaming by protein group)
    5. Output generation

    Args:
        args: Command-line arguments

    Returns:
        Exit code (0 = success)

    """
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

    # Get sample type patterns from args or config
    reference_patterns = None
    pool_patterns = None
    if hasattr(args, 'reference_pattern') and args.reference_pattern:
        reference_patterns = args.reference_pattern
    elif 'sample_annotations' in config:
        ref_pattern = config['sample_annotations'].get('reference_pattern')
        if ref_pattern:
            reference_patterns = (
                [ref_pattern] if isinstance(ref_pattern, str) else ref_pattern
            )

    if hasattr(args, 'pool_pattern') and args.pool_pattern:
        pool_patterns = args.pool_pattern
    elif 'sample_annotations' in config:
        pool_pattern = config['sample_annotations'].get('pool_pattern')
        if pool_pattern:
            pool_patterns = (
                [pool_pattern] if isinstance(pool_pattern, str) else pool_pattern
            )

    # Categorize input files
    csv_inputs = [
        p for p in input_paths if p.suffix.lower() in ['.csv', '.tsv', '.txt']
    ]
    parquet_inputs = [
        p for p in input_paths if p.suffix.lower() == '.parquet'
    ]

    # =========================================================================
    # Stage 1: Merge/prepare transition-level parquet
    # =========================================================================
    metadata_df = None
    if len(csv_inputs) >= 1:
        logger.info(f"Merging {len(csv_inputs)} Skyline reports (streaming)...")

        batch_names = [p.stem for p in csv_inputs]
        merged_parquet_path = output_dir / 'merged_data.parquet'

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
                pool_patterns=pool_patterns,
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

    # =========================================================================
    # Stage 2: Auto-detect column names from data
    # =========================================================================
    pf = pq.ParquetFile(transition_parquet)
    available_columns = set(pf.schema_arrow.names)
    logger.info(f"Available columns: {sorted(available_columns)}")

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
    sample_col = config['data']['sample_column']
    if sample_col not in available_columns and 'Replicate Name' in available_columns:
        sample_col = 'Replicate Name'

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

    logger.info(f"Using columns: peptide={peptide_col}, sample={sample_col}, abundance={abundance_col}")

    # =========================================================================
    # Stage 3: Transition -> Peptide rollup (streaming)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Stage 3: Transition -> Peptide rollup (streaming)")
    logger.info("=" * 60)

    transition_config = ChunkedRollupConfig(
        peptide_col=peptide_col,
        transition_col=transition_col,
        sample_col=sample_col,
        abundance_col=abundance_col,
        method=config['transition_rollup'].get('method', 'median_polish'),
        min_transitions=config['transition_rollup'].get('min_transitions', 3),
        log_transform=True,
        progress_interval=5000,
    )

    peptide_path = output_dir / 'peptides.parquet'
    peptide_result = rollup_transitions_sorted(
        parquet_path=transition_parquet,
        output_path=peptide_path,
        config=transition_config,
        save_residuals=config['output'].get('include_residuals', True),
    )
    samples = peptide_result.samples
    method_log.append(
        f"Transition rollup: {peptide_result.n_peptides:,} peptides, "
        f"{len(samples)} samples"
    )

    # =========================================================================
    # Stage 4: Protein parsimony
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Stage 4: Protein parsimony")
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
    # Stage 5: Peptide -> Protein rollup (streaming)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Stage 5: Peptide -> Protein rollup (streaming)")
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

    protein_path = output_dir / 'proteins.parquet'
    protein_result = rollup_proteins_streaming(
        peptide_parquet_path=peptide_path,
        protein_groups=protein_groups,
        output_path=protein_path,
        config=protein_config,
        samples=samples,
        save_residuals=config['output'].get('include_residuals', True),
    )
    method_log.append(
        f"Protein rollup: {protein_result.n_proteins:,} proteins"
    )

    # =========================================================================
    # Stage 6: Output generation
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Stage 6: Output generation")
    logger.info("=" * 60)

    import pandas as pd

    output_format = config['output'].get('format', 'parquet')

    # Convert peptide output to linear scale and save
    peptide_df = pd.read_parquet(peptide_path)
    if 'abundance' in peptide_df.columns:
        peptide_df['abundance_linear'] = 2 ** peptide_df['abundance']

    peptide_output = output_dir / f"corrected_peptides.{output_format}"
    if output_format == 'parquet':
        peptide_df.to_parquet(peptide_output, index=False)
    else:
        sep = '\t' if output_format == 'tsv' else ','
        peptide_df.to_csv(peptide_output, sep=sep, index=False)
    logger.info(f"  Saved peptides: {peptide_output}")

    # Convert protein output to linear scale and save
    protein_df = pd.read_parquet(protein_path)
    if 'abundance' in protein_df.columns:
        protein_df['abundance_linear'] = 2 ** protein_df['abundance']

    protein_output = output_dir / f"corrected_proteins.{output_format}"
    if output_format == 'parquet':
        protein_df.to_parquet(protein_output, index=False)
    else:
        sep = '\t' if output_format == 'tsv' else ','
        protein_df.to_csv(protein_output, sep=sep, index=False)
    logger.info(f"  Saved proteins: {protein_output}")

    # Generate pipeline metadata
    metadata = generate_pipeline_metadata(
        config=config,
        method_log=method_log,
        input_files=[str(p) for p in input_paths],
        output_dir=str(output_dir),
    )
    metadata_output = output_dir / "metadata.json"
    with open(metadata_output, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"  Saved metadata: {metadata_output}")

    # Summary
    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"  Peptides: {peptide_result.n_peptides:,}")
    logger.info(f"  Proteins: {protein_result.n_proteins:,}")
    logger.info(f"  Samples: {len(samples)}")
    logger.info("=" * 60)

    return 0
