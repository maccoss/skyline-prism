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

from . import __version__

if TYPE_CHECKING:
    import pandas as pd

from .data_io import (
    generate_sample_metadata,
    get_parquet_source_fingerprints,
    load_sample_metadata,
    load_sample_metadata_files,
    merge_and_sort_streaming,
    merge_skyline_reports,
    verify_source_fingerprints,
)
from .parsimony import (
    build_peptide_protein_map,
    compute_protein_groups,
    export_protein_groups,
)
from .validation import (
    generate_comprehensive_qc_report,
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
    root_logger = logging.getLogger("skyline_prism")
    root_logger.setLevel(level)
    root_logger.propagate = False  # Prevent duplicate output to root logger

    # Clear any existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if log_file specified)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.info(f"Logging to: {log_file}")


# Sample ID separator used when merging batches
SAMPLE_ID_SEPARATOR = "__@__"


def sample_id_to_replicate_name(sample_id: str) -> str:
    """Extract Replicate Name from Sample ID.

    Sample ID format: "ReplicateName__@__BatchName"
    Returns the ReplicateName part.

    If the Sample ID doesn't contain the separator, returns it unchanged
    (it's already a Replicate Name).
    """
    if SAMPLE_ID_SEPARATOR in sample_id:
        return sample_id.split(SAMPLE_ID_SEPARATOR)[0]
    return sample_id


def build_sample_type_map(
    sample_cols: list[str],
    metadata_df: pd.DataFrame,
) -> dict[str, str]:
    """Build a mapping from data column names to sample types.

    Handles the case where data columns are Sample IDs (with batch suffix)
    but metadata uses Replicate Names (without batch suffix).

    Args:
        sample_cols: Column names from the data (may be Sample IDs or Replicate Names)
        metadata_df: Metadata DataFrame with 'sample' and 'sample_type' columns

    Returns:
        Dict mapping each sample_col to its sample_type

    """
    if metadata_df is None or "sample_type" not in metadata_df.columns:
        return {}

    # Get sample column name from metadata (normalized to 'sample')
    if "sample" not in metadata_df.columns:
        return {}

    # Build reverse lookup: replicate_name -> sample_type
    meta_lookup = dict(zip(metadata_df["sample"], metadata_df["sample_type"]))

    # Map data columns to sample types
    # Try direct match first, then try extracting replicate name from sample ID
    result = {}
    for col in sample_cols:
        if col in meta_lookup:
            result[col] = meta_lookup[col]
        else:
            # Try extracting replicate name from Sample ID format
            replicate_name = sample_id_to_replicate_name(col)
            if replicate_name in meta_lookup:
                result[col] = meta_lookup[replicate_name]

    return result


def build_sample_batch_map(
    sample_cols: list[str],
    metadata_df: pd.DataFrame,
) -> dict[str, str]:
    """Build a mapping from data column names to batch names.

    Handles the case where data columns are Sample IDs (with batch suffix)
    but metadata uses Replicate Names (without batch suffix).

    Args:
        sample_cols: Column names from the data (may be Sample IDs or Replicate Names)
        metadata_df: Metadata DataFrame with 'sample' and 'batch' columns

    Returns:
        Dict mapping each sample_col to its batch

    """
    if metadata_df is None or "batch" not in metadata_df.columns:
        return {}

    if "sample" not in metadata_df.columns:
        return {}

    # Build reverse lookup: replicate_name -> batch
    meta_lookup = dict(zip(metadata_df["sample"], metadata_df["batch"]))

    # Map data columns to batches
    result = {}
    for col in sample_cols:
        if col in meta_lookup:
            result[col] = meta_lookup[col]
        else:
            # Try extracting replicate name from Sample ID format
            replicate_name = sample_id_to_replicate_name(col)
            if replicate_name in meta_lookup:
                result[col] = meta_lookup[replicate_name]

    return result


def estimate_batches_from_parquet(
    parquet_path: Path,
    sample_cols: list[str],
    method: str = "auto",
    n_batches: int | None = None,
    gap_iqr_multiplier: float = 1.5,
) -> dict[str, str]:
    """Estimate batch assignments from acquisition times in parquet file.

    Args:
        parquet_path: Path to transition parquet file
        sample_cols: Sample column names (may be Sample IDs)
        method: Estimation method:
            - "auto": Try gap detection, fall back to fixed if no gaps
            - "fixed": Divide into n_batches equal batches by acquisition time
            - "gap": Only use gap detection (return empty if no gaps found)
        n_batches: Number of batches for fixed method (required if method="fixed")
        gap_iqr_multiplier: IQR multiplier for gap detection

    Returns:
        Dict mapping sample_col -> batch label

    """
    import duckdb

    con = duckdb.connect()

    # Get unique samples with acquisition times
    # Try both Sample ID and Replicate Name columns
    try:
        result = con.execute(f"""
            SELECT DISTINCT "Sample ID", "Replicate Name", "Acquired Time"
            FROM read_parquet('{parquet_path}')
            WHERE "Acquired Time" IS NOT NULL
        """).fetchdf()
    except Exception:
        # Fall back to just Replicate Name if Sample ID doesn't exist
        try:
            result = con.execute(f"""
                SELECT DISTINCT "Replicate Name", "Acquired Time"
                FROM read_parquet('{parquet_path}')
                WHERE "Acquired Time" IS NOT NULL
            """).fetchdf()
            result["Sample ID"] = result["Replicate Name"]
        except Exception:
            con.close()
            logger.warning("Cannot estimate batches: 'Acquired Time' column not found")
            return {}
    con.close()

    if len(result) < 2:
        logger.warning("Cannot estimate batches: insufficient samples with acquisition times")
        return {}

    # Parse acquisition times and sort
    import pandas as pd

    result["acquired_time"] = pd.to_datetime(result["Acquired Time"], errors="coerce")
    result = result.dropna(subset=["acquired_time"]).sort_values("acquired_time")

    if len(result) < 2:
        return {}

    # Build mapping from sample_col to batch
    sample_to_batch = {}

    if method == "fixed" and n_batches is not None and n_batches > 1:
        # Divide into N equal batches by acquisition order
        n_samples = len(result)
        batch_size = n_samples // n_batches
        remainder = n_samples % n_batches

        batch_assignments = []
        for i in range(n_batches):
            # Distribute remainder across first batches
            size = batch_size + (1 if i < remainder else 0)
            batch_assignments.extend([f"batch_{i + 1}"] * size)

        # Assign batches based on sorted order
        for idx, (_, row) in enumerate(result.iterrows()):
            sample_id = row.get("Sample ID") or row.get("Replicate Name")
            rep_name = row.get("Replicate Name") or sample_id
            batch = batch_assignments[idx]
            # Map both Sample ID and Replicate Name
            sample_to_batch[sample_id] = batch
            if rep_name != sample_id:
                sample_to_batch[rep_name] = batch

        logger.info(f"  Estimated {n_batches} batches by acquisition time (fixed division)")

    elif method in ["auto", "gap"]:
        # Detect gaps in acquisition times
        gaps = result["acquired_time"].diff().dt.total_seconds() / 60  # minutes

        # IQR-based outlier detection for gaps
        q1 = gaps.dropna().quantile(0.25)
        q3 = gaps.dropna().quantile(0.75)
        iqr = q3 - q1
        threshold = q3 + gap_iqr_multiplier * iqr
        # Ensure threshold is reasonable (at least 10% above median)
        median_gap = gaps.dropna().median()
        threshold = max(threshold, median_gap * 1.1)

        # Find batch breaks
        is_break = gaps > threshold
        batch_num = is_break.fillna(False).cumsum() + 1

        n_detected = batch_num.max()

        if n_detected > 1:
            # Assign batches based on gap detection
            for idx, (_, row) in enumerate(result.iterrows()):
                sample_id = row.get("Sample ID") or row.get("Replicate Name")
                rep_name = row.get("Replicate Name") or sample_id
                batch = f"batch_{batch_num.iloc[idx]}"
                sample_to_batch[sample_id] = batch
                if rep_name != sample_id:
                    sample_to_batch[rep_name] = batch

            logger.info(f"  Estimated {n_detected} batches from acquisition time gaps")
            logger.info(f"    Gap threshold: {threshold:.1f} min (median: {median_gap:.1f} min)")
        elif method == "auto" and n_batches is not None and n_batches > 1:
            # Fall back to fixed division
            return estimate_batches_from_parquet(
                parquet_path,
                sample_cols,
                method="fixed",
                n_batches=n_batches,
                gap_iqr_multiplier=gap_iqr_multiplier,
            )
        else:
            logger.info("  No significant acquisition time gaps detected")
            return {}

    # Map sample_cols to their batch assignments
    result_map = {}
    for col in sample_cols:
        if col in sample_to_batch:
            result_map[col] = sample_to_batch[col]
        else:
            # Try extracting replicate name from Sample ID
            rep_name = sample_id_to_replicate_name(col)
            if rep_name in sample_to_batch:
                result_map[col] = sample_to_batch[rep_name]

    return result_map


# Known configuration keys for validation
# This helps detect typos in config files
KNOWN_CONFIG_KEYS = {
    "data": {
        "abundance_column",
        "rt_column",
        "peptide_column",
        "protein_column",
        "protein_name_column",
        "sample_column",
        "batch_column",
        "sample_type_column",
        "transition_column",
        "precursor_column",
        "fragment_column",
    },
    "sample_annotations": {
        "reference_pattern",
        "qc_pattern",
        "experimental_pattern",
    },
    "batch_estimation": {
        "method",  # "auto", "fixed", or "gap"
        "min_samples_per_batch",
        "max_samples_per_batch",
        "gap_iqr_multiplier",
        "n_batches",
    },
    "sample_outlier_detection": {
        "enabled",
        "action",
        "method",
        "iqr_multiplier",
        "fold_threshold",
    },
    "processing": {
        "n_workers",
        "peptide_batch_size",
    },
    "transition_rollup": {
        "enabled",
        "method",
        "use_ms1",
        "min_transitions",
        "topn_count",
        "topn_selection",
        "topn_weighting",
        "learn_adaptive_weights",
        "learn_weights",  # Both supported for compatibility
        "adaptive_rollup",  # Nested section
        "spectral_library_path",  # Path to spectral library for library-assisted method
        "spectral_library_min_fragments",  # Min fragments for library matching
        "spectral_library_mz_tolerance",  # m/z tolerance for library matching
        "library_assist",  # Nested section for library-assisted rollup config
    },
    "transition_rollup.adaptive_rollup": {
        "beta_relative_intensity",
        "beta_mz",
        "beta_shape_corr",
        "beta_shape_corr_outlier",
        "beta_log_intensity",
        "beta_sqrt_intensity",
        "beta_shape_corr_max",  # Deprecated but recognized
        "shape_corr_low_threshold",
        "min_improvement_pct",
        "mz_min",
        "mz_max",
        "log_intensity_center",
    },
    "transition_rollup.library_assist": {
        "library_path",
        "mz_tolerance",
        "min_matched_fragments",
        "outlier_threshold",
        "fitting_method",
        "remove_outliers",
    },
    "global_normalization": {
        "method",
        "vsn_params",
        "rt_lowess",  # rt_lowess is nested section
    },
    "global_normalization.rt_lowess": {
        "frac",
        "n_grid_points",
    },
    "batch_correction": {
        "enabled",
        "method",
    },
    "parsimony": {
        "enabled",
        "fasta_path",
        "shared_peptide_handling",
    },
    "protein_rollup": {
        "method",
        "min_peptides",
        "topn",
        "ibaq",
        "median_polish",
        "shared_peptide_handling",
    },
    "protein_rollup.topn": {
        "n",
        "selection",
    },
    "protein_rollup.ibaq": {
        "fasta_path",
        "enzyme",
        "missed_cleavages",
        "min_peptide_length",
        "max_peptide_length",
    },
    "protein_rollup.median_polish": {
        "max_iterations",
        "convergence_tolerance",
    },
    "output": {
        "format",
        "include_residuals",
        "compress",
    },
    "qc_report": {
        "enabled",
        "filename",
        "save_plots",
        "embed_plots",
        "plots",
    },
    "qc_report.plots": {
        "intensity_distribution",
        "pca_comparison",
        "control_correlation",
        "cv_distribution",
        "boxplot_comparison",
    },
}


def load_config(config_path: Path | None) -> dict:
    """Load configuration from YAML file or return defaults."""
    defaults = {
        "data": {
            "abundance_column": "Area",
            "rt_column": "Retention Time",
            "peptide_column": "Peptide Modified Sequence",
            "protein_column": "Protein Accession",
            "protein_name_column": "Protein",
            "sample_column": "Replicate Name",
            "batch_column": "Batch",
            "sample_type_column": "Sample Type",
            "transition_column": "Fragment Ion",
        },
        "transition_rollup": {
            "enabled": False,
            "method": "median_polish",
            "min_transitions": 3,
            # Note: learn_adaptive_weights defaults handled dynamically below
            # (True if method=adaptive, False otherwise)
        },
        "global_normalization": {
            "method": "median",
        },
        "batch_correction": {
            "enabled": True,
            "method": "combat",
        },
        "parsimony": {
            "shared_peptide_handling": "all_groups",
        },
        "protein_rollup": {
            "method": "median_polish",
            "topn": {"n": 3, "selection": "median_abundance"},
            "median_polish": {"max_iterations": 20, "convergence_tolerance": 0.0001},
        },
        "output": {
            "format": "parquet",
            "include_residuals": True,
            "compress": True,
        },
    }

    unknown_keys = []
    if config_path and config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            user_config = yaml.safe_load(f)

        # Validate config keys before merging (logging happens later)
        unknown_keys = _find_unknown_config_keys(user_config)

        # Deep merge user config over defaults
        defaults = _deep_merge(defaults, user_config)

    # Store unknown keys in config for later warning
    defaults["_unknown_keys"] = unknown_keys

    return defaults


def _find_unknown_config_keys(user_config: dict) -> list[str]:
    """Find unknown keys in user config.

    Returns a list of unknown keys (e.g., ['transition_rollup.learn_weights']).
    """
    unknown_keys = []

    def check_section(config_dict: dict, section_path: str = ""):
        """Recursively check keys in a config section."""
        for key, value in config_dict.items():
            full_key = f"{section_path}.{key}" if section_path else key

            # Determine which known keys to check against
            if section_path:
                known_section = KNOWN_CONFIG_KEYS.get(section_path, set())
            else:
                known_section = set(KNOWN_CONFIG_KEYS.keys())

            if key not in known_section:
                unknown_keys.append(full_key)

            # Recursively check nested dicts (but only for known nested sections)
            if isinstance(value, dict):
                nested_section = f"{section_path}.{key}" if section_path else key
                if nested_section in KNOWN_CONFIG_KEYS:
                    check_section(value, nested_section)

    check_section(user_config)
    return unknown_keys


def load_config_from_provenance(provenance_path: Path) -> tuple[dict, dict]:
    """Load configuration and file paths from a previous pipeline run's provenance JSON.

    This enables reproducibility by allowing users to re-run the pipeline
    with the exact same parameters as a previous run, and optionally
    reuse the same input and metadata files.

    Args:
        provenance_path: Path to metadata.json from a previous PRISM run

    Returns:
        Tuple of (config, provenance_data):
        - config: Configuration dictionary compatible with load_config() output
        - provenance_data: Full provenance dict containing source_files, metadata_files, etc.

    Raises:
        ValueError: If the provenance file is missing required fields

    """
    with open(provenance_path) as f:
        provenance = json.load(f)

    # Check for required fields
    if "processing_parameters" not in provenance:
        raise ValueError(
            f"Provenance file {provenance_path} does not contain 'processing_parameters'. "
            "This may be from an older version of PRISM."
        )

    # Start with defaults
    config = load_config(None)

    # Extract processing parameters and merge over defaults
    params = provenance["processing_parameters"]

    # Map provenance sections to config sections
    for section in [
        "data",
        "transition_rollup",
        "global_normalization",
        "batch_correction",
        "protein_rollup",
        "parsimony",
        "output",
    ]:
        if section in params:
            config[section] = _deep_merge(config.get(section, {}), params[section])

    logger.info(f"Loaded configuration from provenance: {provenance_path}")
    logger.info(f"  Original pipeline version: {provenance.get('pipeline_version', 'unknown')}")
    logger.info(f"  Original processing date: {provenance.get('processing_date', 'unknown')}")

    return config, provenance


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
    logger.info(
        f"  {result.n_rows} rows, {result.n_replicates} replicates, "
        f"{result.n_precursors} precursors"
    )

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
    sample_type_col: str = "sample_type",
    sample_col: str = "replicate_name",
    batch_col: str = "batch",
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

        pipeline_version = version("skyline-prism")
    except Exception:
        pipeline_version = "development"

    # Build sample metadata summary
    samples_df = data[[sample_col, sample_type_col]].drop_duplicates()
    sample_counts = samples_df[sample_type_col].value_counts().to_dict()

    sample_metadata = {
        "n_samples": len(samples_df),
        "n_reference": sample_counts.get("reference", 0),
        "n_qc": sample_counts.get("qc", 0),
        "n_experimental": sample_counts.get("experimental", 0),
        "samples": samples_df[sample_col].tolist(),
    }

    # Add batch info if available
    if batch_col in data.columns:
        batches = data[batch_col].dropna().unique().tolist()
        sample_metadata["batches"] = batches
        sample_metadata["n_batches"] = len(batches)

    # Build protein groups summary
    groups_summary = {
        "n_groups": len(protein_groups),
        "n_proteins": sum(len(g.proteins) for g in protein_groups),
        "shared_peptide_handling": config.get("parsimony", {}).get(
            "shared_peptide_handling", "all_groups"
        ),
    }

    # Build processing parameters from config (includes all settings for reproducibility)
    processing_parameters = {
        "data": config.get("data", {}),  # Column mappings for reproducibility
        "transition_rollup": config.get("transition_rollup", {}),
        "global_normalization": config.get("global_normalization", {}),
        "batch_correction": config.get("batch_correction", {}),
        "protein_rollup": config.get("protein_rollup", {}),
        "parsimony": config.get("parsimony", {}),
        "output": config.get("output", {}),
    }

    # Build the metadata dictionary
    metadata = {
        "pipeline_version": pipeline_version,
        "processing_date": datetime.now(timezone.utc).isoformat(),
        "source_files": input_files,
        "sample_metadata": sample_metadata,
        "protein_groups": groups_summary,
        "processing_parameters": processing_parameters,
        "method_log": method_log,
        "validation_metrics": validation_metrics or {},
        "warnings": [],
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
    from .transition_rollup import (
        AdaptiveRollupParams,
    )

    # Load configuration
    provenance_data = None
    if hasattr(args, "from_provenance") and args.from_provenance:
        provenance_path = Path(args.from_provenance)
        config, provenance_data = load_config_from_provenance(provenance_path)
        provenance_dir = provenance_path.parent
        method_log = [f"Configuration loaded from provenance: {args.from_provenance}"]

        # Auto-load input files if not explicitly provided
        if not args.input and "source_files" in provenance_data:
            source_files = provenance_data["source_files"]
            missing = [f for f in source_files if not Path(f).exists()]
            if missing:
                logger.error("Some source files from provenance not found:")
                for f in missing:
                    logger.error(f"  - {f}")
                logger.error("Please provide input files with -i flag")
                return 1
            args.input = source_files
            logger.info(f"  Using {len(source_files)} input files from provenance")

        # Auto-load metadata files if not explicitly provided
        if not args.metadata and provenance_data.get("metadata_source") == "user_provided":
            meta_files = provenance_data.get("metadata_files", [])
            if meta_files:
                missing = [f for f in meta_files if not Path(f).exists()]
                if missing:
                    # Fall back to saved sample_metadata.tsv
                    saved_meta = provenance_dir / "sample_metadata.tsv"
                    if saved_meta.exists():
                        logger.info(f"  Original metadata files moved; using saved {saved_meta}")
                        args.metadata = [str(saved_meta)]
                    else:
                        logger.warning(f"  Metadata files not found and no saved TSV: {missing}")
                else:
                    args.metadata = meta_files
                    logger.info(f"  Using {len(meta_files)} metadata files from provenance")

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"prism_run_{timestamp}.log"
    verbose = getattr(args, "verbose", False)
    setup_logging(verbose=verbose, log_file=log_file)

    # Warn about any unknown config keys (helps catch typos)
    unknown_keys = config.pop("_unknown_keys", [])
    if unknown_keys:
        logger.warning("=" * 60)
        logger.warning("Unknown configuration keys detected (possible typos):")
        for key in unknown_keys:
            logger.warning(f"  - {key}")
        logger.warning("=" * 60)

    # Get sample type patterns from args or config
    reference_patterns = None
    qc_patterns = None
    if hasattr(args, "reference_pattern") and args.reference_pattern:
        reference_patterns = args.reference_pattern
    elif "sample_annotations" in config:
        ref_pattern = config["sample_annotations"].get("reference_pattern")
        if ref_pattern:
            reference_patterns = [ref_pattern] if isinstance(ref_pattern, str) else ref_pattern

    if hasattr(args, "qc_pattern") and args.qc_pattern:
        qc_patterns = args.qc_pattern
    elif "sample_annotations" in config:
        qc_pattern = config["sample_annotations"].get("qc_pattern")
        if qc_pattern:
            qc_patterns = [qc_pattern] if isinstance(qc_pattern, str) else qc_pattern

    # Categorize input files
    csv_inputs = [p for p in input_paths if p.suffix.lower() in [".csv", ".tsv", ".txt"]]
    parquet_inputs = [p for p in input_paths if p.suffix.lower() == ".parquet"]

    # Check for --force-reprocess flag
    force_reprocess = getattr(args, "force_reprocess", False)

    # =========================================================================
    # Stage 1: Merge CSVs / Prepare Input Data
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Stage 1: Merge CSVs / Prepare Input Data")
    logger.info("=" * 60)

    metadata_df = None
    n_transitions = 0  # Track total transition rows for metadata
    is_pre_sorted = False  # Track if merge+sort was done (skip sorting in rollup)

    if len(csv_inputs) >= 1:
        batch_names = [p.stem for p in csv_inputs]
        merged_parquet_path = output_dir / "merged_data.parquet"

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
                    n_transitions = pq.ParquetFile(transition_parquet).metadata.num_rows
                    method_log.append("Reused cached merged parquet (source files unchanged)")
                    # Try to load existing metadata if available
                    existing_metadata = output_dir / "sample_metadata.tsv"
                    if existing_metadata.exists() and not args.metadata:
                        metadata_df = load_sample_metadata(existing_metadata)
                        logger.info(f"  Loaded existing metadata: {existing_metadata}")
                else:
                    logger.info("  Source files changed - reprocessing required:")
                    for reason in mismatch_reasons:
                        logger.info(f"    - {reason}")
            else:
                logger.info("  No fingerprints in parquet metadata - reprocessing required")

        if not use_cached:
            logger.info(f"Merging and sorting {len(csv_inputs)} Skyline reports (streaming)...")
            merged_path, samples_by_batch, total_rows = merge_and_sort_streaming(
                csv_inputs,
                merged_parquet_path,
                sort_column=None,  # Auto-detect peptide column from CSV
                batch_names=batch_names,
            )
            n_transitions = total_rows
            is_pre_sorted = True  # Flag for rollup to skip sorting
            method_log.append(f"Merged and sorted {len(csv_inputs)} reports ({total_rows:,} rows)")
            transition_parquet = merged_path

            # Generate metadata if not provided
            if not args.metadata:
                logger.info("Generating sample metadata from sample names...")
                metadata_df = generate_sample_metadata(
                    samples_by_batch,
                    reference_patterns=reference_patterns,
                    qc_patterns=qc_patterns,
                )
                metadata_path = output_dir / "sample_metadata.tsv"
                metadata_df.to_csv(metadata_path, sep="\t", index=False)
                method_log.append(f"Generated sample metadata: {metadata_path}")

    elif len(parquet_inputs) >= 1:
        merged_parquet_path = parquet_inputs[0]  # treat input as "merged"
        transition_parquet = parquet_inputs[0]
        n_transitions = pq.ParquetFile(transition_parquet).metadata.num_rows
        logger.info(f"Using existing parquet: {transition_parquet}")
        method_log.append(f"Input parquet: {transition_parquet}")

    else:
        logger.error("No input files provided")
        return 1

    # Load explicit metadata if provided
    # Track metadata source and files for provenance
    metadata_source = "auto_generated"
    metadata_input_files: list[str] = []

    if args.metadata:
        metadata_paths = [Path(p) for p in args.metadata]
        metadata_df = load_sample_metadata_files(metadata_paths)

        # Always save merged metadata to output directory
        merged_metadata_path = output_dir / "sample_metadata.tsv"
        metadata_df.to_csv(merged_metadata_path, sep="\t", index=False)
        logger.info(f"  Saved merged metadata: {merged_metadata_path}")

        # Track for provenance
        metadata_source = "user_provided"
        metadata_input_files = [str(p.resolve()) for p in metadata_paths]

        if len(metadata_paths) == 1:
            method_log.append(f"Loaded metadata: {args.metadata[0]}")
        else:
            method_log.append(f"Loaded and merged metadata from {len(metadata_paths)} files")

    # -------------------------------------------------------------------------
    # Stage 1 (continued): Auto-detect column names from data
    # -------------------------------------------------------------------------
    pf = pq.ParquetFile(transition_parquet)
    available_columns = set(pf.schema_arrow.names)
    logger.info(f"  Available columns: {sorted(available_columns)}")

    # Auto-detect peptide column
    peptide_col = config["data"]["peptide_column"]
    peptide_col_alternatives = [
        "Peptide Modified Sequence Unimod Ids",
        "Peptide Modified Sequence",
        "Peptide",
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
    if "Sample ID" in available_columns:
        sample_col = "Sample ID"
        logger.info("  Using 'Sample ID' for unique sample identification across batches")
    elif config["data"]["sample_column"] in available_columns:
        sample_col = config["data"]["sample_column"]
    elif "Replicate Name" in available_columns:
        sample_col = "Replicate Name"
    else:
        sample_col = config["data"]["sample_column"]

    abundance_col = config["data"]["abundance_column"]
    if abundance_col not in available_columns and "Area" in available_columns:
        abundance_col = "Area"

    transition_col = config["data"].get("transition_column", "Fragment Ion")
    if transition_col not in available_columns and "Fragment Ion" in available_columns:
        transition_col = "Fragment Ion"

    protein_col = config["data"]["protein_column"]
    if protein_col not in available_columns and "Protein Accession" in available_columns:
        protein_col = "Protein Accession"

    protein_name_col = config["data"].get("protein_name_column", "Protein")
    if protein_name_col not in available_columns and "Protein" in available_columns:
        protein_name_col = "Protein"

    # Optional gene and description columns from Skyline CSV
    protein_gene_col = config["data"].get("protein_gene_column", "Protein Gene")
    if protein_gene_col not in available_columns and "Protein Gene" in available_columns:
        protein_gene_col = "Protein Gene"
    elif protein_gene_col not in available_columns:
        protein_gene_col = None

    protein_description_col = config["data"].get("protein_description_column", "Protein")
    # For description, we use the "Protein" column which contains the full protein name/description
    if protein_description_col not in available_columns and "Protein" in available_columns:
        protein_description_col = "Protein"
    elif protein_description_col not in available_columns:
        protein_description_col = None

    logger.info(
        f"Using columns: peptide={peptide_col}, sample={sample_col}, abundance={abundance_col}"
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
    rollup_method = config["transition_rollup"].get("method", "sum")
    # Normalize method name (accept various spellings for library-assisted)
    if rollup_method in ("library_assist", "library-assist"):
        rollup_method = "library-assisted"
    use_ms1 = config["transition_rollup"].get("use_ms1", False)
    min_transitions = config["transition_rollup"].get("min_transitions", 3)
    # Support both 'learn_adaptive_weights' (preferred) and 'learn_weights' (deprecated)
    # Default to True when method=adaptive, False otherwise
    default_learn = rollup_method == "adaptive"
    learn_weights = config["transition_rollup"].get(
        "learn_adaptive_weights", config["transition_rollup"].get("learn_weights", default_learn)
    )

    logger.info(f"  Rollup method: {rollup_method}")
    logger.info(f"  Include MS1 precursors: {use_ms1}")
    logger.info(f"  Min transitions: {min_transitions}")

    # Initialize adaptive rollup params (None for non-adaptive methods)
    adaptive_params = None

    if rollup_method == "adaptive":
        # Check for explicit parameters in config
        ar_config = config["transition_rollup"].get("adaptive_rollup", {})
        adaptive_params = AdaptiveRollupParams(
            beta_log_intensity=0.0,  # Deprecated
            beta_sqrt_intensity=0.0,  # Deprecated
            beta_relative_intensity=ar_config.get("beta_relative_intensity", 0.0),
            beta_mz=ar_config.get("beta_mz", 0.0),
            beta_shape_corr=ar_config.get("beta_shape_corr", 0.0),
            beta_shape_corr_max=0.0,  # Not optimized
            beta_shape_corr_outlier=ar_config.get("beta_shape_corr_outlier", 0.0),
            mz_min=ar_config.get("mz_min", 0.0),
            mz_max=ar_config.get("mz_max", 2000.0),
            log_intensity_center=ar_config.get("log_intensity_center", 15.0),
            shape_corr_low_threshold=ar_config.get("shape_corr_low_threshold", 0.5),
            min_improvement_pct=ar_config.get("min_improvement_pct", 5.0),
        )

        # Learn weights from reference/QC samples if enabled
        if learn_weights and metadata_df is not None:
            from .transition_rollup import learn_adaptive_weights

            # Extract sample types from metadata
            # Use sample_id (unique across batches) if available, else sample
            meta_id_col = "sample_id" if "sample_id" in metadata_df.columns else "sample"
            sample_types = dict(zip(metadata_df[meta_id_col], metadata_df["sample_type"]))
            ref_samples = [s for s, t in sample_types.items() if t == "reference"]
            pool_samples = [s for s, t in sample_types.items() if t == "qc"]

            if len(ref_samples) >= 2:
                logger.info("  Learning adaptive rollup parameters...")
                logger.info(f"    Reference samples: {len(ref_samples)}")
                logger.info(f"    QC samples: {len(pool_samples)}")

                # Load transition data for learning - only needed columns
                import time

                import duckdb

                logger.info("    Loading training data...")
                load_start = time.time()

                learn_con = duckdb.connect()
                sample_list_sql = ",".join(f"'{s}'" for s in ref_samples + pool_samples)

                # Only select columns needed for adaptive learning
                needed_cols = [
                    peptide_col,
                    transition_col,
                    sample_col,
                    abundance_col,
                    "Product Mz",
                    "Shape Correlation",
                    "Product Charge",
                    "Precursor Charge",
                ]
                # Also include Replicate Name if we're using Sample ID
                # (needed for matching metadata sample names)
                if sample_col == "Sample ID":
                    needed_cols.append("Replicate Name")
                # Filter to columns that exist
                col_check = learn_con.execute(f"""
                    SELECT column_name FROM (
                        DESCRIBE SELECT * FROM read_parquet('{transition_parquet}') LIMIT 0
                    )
                """).fetchdf()
                available_cols = set(col_check["column_name"].tolist())
                select_cols = [c for c in needed_cols if c in available_cols]
                select_sql = ", ".join(f'"{c}"' for c in select_cols)

                # Determine which column to use for filtering
                # Metadata sample names may match Replicate Name, not Sample ID
                # (Sample ID often includes batch suffix like "__@__batchname")
                filter_col = sample_col
                if sample_col == "Sample ID" and "Replicate Name" in available_cols:
                    # Check if metadata samples match Replicate Name values
                    # by checking if any ref/qc sample exists in Replicate Name
                    test_sample = (ref_samples + pool_samples)[0] if ref_samples else None
                    if test_sample:
                        test_result = learn_con.execute(f"""
                            SELECT COUNT(*) FROM read_parquet('{transition_parquet}')
                            WHERE "Replicate Name" = '{test_sample}' LIMIT 1
                        """).fetchone()[0]
                        if test_result > 0:
                            filter_col = "Replicate Name"
                            logger.info("    Using 'Replicate Name' for sample filtering")

                learn_df = learn_con.execute(f"""
                    SELECT {select_sql}
                    FROM read_parquet('{transition_parquet}')
                    WHERE "{filter_col}" IN ({sample_list_sql})
                """).fetchdf()
                learn_con.close()

                load_time = time.time() - load_start
                logger.info(f"    Loaded {len(learn_df):,} rows in {load_time:.1f}s")

                # Data is already in LINEAR scale from merged parquet
                # (log transform happens in rollup, not in merge)

                learn_result = learn_adaptive_weights(
                    learn_df,
                    reference_samples=ref_samples,
                    qc_samples=pool_samples,
                    peptide_col=peptide_col,
                    transition_col=transition_col,
                    sample_col=filter_col,  # Use filter_col since samples match this column
                    abundance_col=abundance_col,
                    mz_col="Product Mz",
                    shape_corr_col="Shape Correlation",
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
                    # Switch to sum method (faster than adaptive with all zeros)
                    rollup_method = "sum"
                    adaptive_params = None
            else:
                logger.warning(f"  Not enough reference samples for learning ({len(ref_samples)})")
                logger.info("  Using default parameters")

        # Only log adaptive params if actually using adaptive method
        if rollup_method == "adaptive" and adaptive_params is not None:
            logger.info(
                f"  Adaptive params: mz={adaptive_params.beta_mz:.3f}, "
                f"shape_outlier={adaptive_params.beta_shape_corr_outlier:.3f}"
            )

    # Get topn method parameters
    topn_count = config["transition_rollup"].get("topn_count", 3)
    topn_selection = config["transition_rollup"].get("topn_selection", "correlation")
    topn_weighting = config["transition_rollup"].get("topn_weighting", "sqrt")

    # Library-assisted method: load spectral library
    spectral_library = None
    if rollup_method == "library-assisted":
        from .spectral_library import SpectralLibraryRollup

        # Support both config formats:
        # 1. Nested: transition_rollup.library_assist.library_path
        # 2. Flat: transition_rollup.spectral_library_path
        lib_assist_config = config["transition_rollup"].get("library_assist", {})
        spectral_library_path = lib_assist_config.get("library_path") or config[
            "transition_rollup"
        ].get("spectral_library_path")
        if not spectral_library_path:
            logger.error(
                "library-assisted method requires library_path in config. "
                "Use transition_rollup.library_assist.library_path"
            )
            return 1

        lib_path = Path(spectral_library_path)
        if not lib_path.is_absolute():
            # Relative paths are relative to config file or current directory
            lib_path = Path.cwd() / lib_path

        if not lib_path.exists():
            logger.error(f"Spectral library not found: {lib_path}")
            return 1

        # Get additional parameters from nested or flat format
        lib_min_fragments = lib_assist_config.get("min_matched_fragments") or config[
            "transition_rollup"
        ].get("spectral_library_min_fragments", 3)
        lib_mz_tolerance = lib_assist_config.get("mz_tolerance") or config["transition_rollup"].get(
            "spectral_library_mz_tolerance", 0.02
        )
        lib_outlier_threshold = lib_assist_config.get("outlier_threshold") or config[
            "transition_rollup"
        ].get("spectral_library_outlier_threshold", 1.0)
        # Whether to remove outliers (default True, set False for debugging)
        lib_remove_outliers = lib_assist_config.get("remove_outliers", True)
        # Fitting method: "median_polish" (default, recommended) or "least_squares"
        lib_fitting_method = lib_assist_config.get("fitting_method", "median_polish")
        if lib_fitting_method not in ("median_polish", "least_squares"):
            logger.warning(
                f"Unknown fitting_method '{lib_fitting_method}', using 'median_polish'"
            )
            lib_fitting_method = "median_polish"

        logger.info(f"  Spectral library: {lib_path}")
        logger.info(f"  Fitting method: {lib_fitting_method}")
        logger.info(f"  Min matched fragments: {lib_min_fragments}")
        logger.info(f"  Outlier threshold: {lib_outlier_threshold}")
        logger.info(f"  Remove outliers: {lib_remove_outliers}")
        spectral_library = SpectralLibraryRollup(
            library_path=lib_path,
            min_fragments=lib_min_fragments,
            mz_tolerance=lib_mz_tolerance,
            use_robust=lib_remove_outliers,
            outlier_threshold=lib_outlier_threshold,
            fitting_method=lib_fitting_method,
        )
        spectral_library.load_library()
        logger.info(f"  Library spectra loaded: {len(spectral_library.library):,}")
    else:
        spectral_library = None
        lib_outlier_threshold = 1.0  # Default value when not using library-assisted

    # Get parallel processing parameters
    n_workers = config.get("processing", {}).get("n_workers", 1)
    peptide_batch_size = config.get("processing", {}).get("peptide_batch_size", 1000)

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
        spectral_library=spectral_library,  # For library-assisted method
        spectral_library_outlier_threshold=lib_outlier_threshold,
        # Save consensus diagnostics when using consensus method
        consensus_diagnostics_path=(
            str(output_dir / "consensus_diagnostics.csv") if rollup_method == "consensus" else None
        ),
    )

    peptide_rollup_path = output_dir / "peptides_rollup.parquet"
    peptide_result = rollup_transitions_sorted(
        parquet_path=transition_parquet,
        output_path=peptide_rollup_path,
        config=transition_config,
        save_residuals=config["output"].get("include_residuals", True)
        and rollup_method == "median_polish",
        pre_sorted=is_pre_sorted,  # Skip sorting if merge+sort already done
    )
    samples = peptide_result.samples
    method_log.append(
        f"Transition rollup ({rollup_method}): {peptide_result.n_peptides:,} peptides, "
        f"{len(samples)} samples"
    )

    # Log library-assisted statistics
    if rollup_method == "library-assisted" and spectral_library is not None:
        lib_stats = spectral_library.get_statistics()
        logger.info(f"  Library match rate: {lib_stats['match_rate'] * 100:.1f}%")
        logger.info(
            f"    Matched: {lib_stats['n_matched']:,}, Unmatched: {lib_stats['n_unmatched']:,}"
        )
        method_log.append(
            f"  Library: {lib_stats['n_matched']:,} matched, "
            f"{lib_stats['n_unmatched']:,} unmatched ({lib_stats['match_rate'] * 100:.1f}%)"
        )

    # -------------------------------------------------------------------------
    # Stage 2b: Peptide Normalization
    # -------------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("Stage 2b: Peptide Normalization")
    logger.info("-" * 60)

    peptide_df = pd.read_parquet(peptide_rollup_path)

    # Data is in wide format: peptide_col, n_transitions, mean_rt, sample1, sample2, ...
    # Get sample columns (all columns after metadata columns)
    meta_cols = [peptide_col, "n_transitions", "mean_rt"]
    meta_cols = [c for c in meta_cols if c in peptide_df.columns]
    sample_cols = [c for c in peptide_df.columns if c not in meta_cols]

    # Data is already in LOG2 scale from chunked_processing.py
    # (The linear to log2 conversion is done inside the rollup functions)

    # Filter out peptides with all NaN (< min_transitions, failed rollup)
    # These cannot be normalized or batch-corrected
    data_matrix = peptide_df[sample_cols].values
    valid_mask = ~np.all(np.isnan(data_matrix), axis=1)
    n_filtered = (~valid_mask).sum()
    if n_filtered > 0:
        logger.info(f"  Filtering {n_filtered} peptides with insufficient transitions (all NaN)")
        peptide_df = peptide_df[valid_mask].reset_index(drop=True)

    # Keep copy for QC comparison (before normalization) - now in log2 scale
    peptide_pre_norm_df = peptide_df.copy()

    # -------------------------------------------------------------------------
    # Sample Outlier Detection (one-sided, low signal only)
    # -------------------------------------------------------------------------
    outlier_config = config.get("sample_outlier_detection", {})
    outlier_detection_enabled = outlier_config.get("enabled", True)
    outlier_samples = []

    if outlier_detection_enabled:
        outlier_action = outlier_config.get("action", "report")
        outlier_method = outlier_config.get("method", "iqr")
        logger.info(
            f"  Running sample outlier detection (method={outlier_method}, action={outlier_action})"
        )

        # Calculate sample medians on LINEAR scale (not log2!)
        # Data is in log2, so we need to convert to linear for proper statistics
        linear_medians = {}
        for col in sample_cols:
            # Convert log2 to linear, then get median
            linear_values = 2 ** peptide_df[col].dropna()
            linear_medians[col] = linear_values.median()

        linear_medians_series = pd.Series(linear_medians)
        overall_median = linear_medians_series.median()

        if outlier_method == "iqr":
            # IQR-based detection on linear scale (one-sided, low only)
            iqr_mult = outlier_config.get("iqr_multiplier", 1.5)
            q1 = linear_medians_series.quantile(0.25)
            q3 = linear_medians_series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - iqr_mult * iqr

            outlier_samples = [s for s, m in linear_medians.items() if m < lower_bound]

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
            fold_thresh = outlier_config.get("fold_threshold", 0.1)
            threshold = fold_thresh * overall_median

            outlier_samples = [s for s, m in linear_medians.items() if m < threshold]

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

            if outlier_action == "exclude":
                logger.warning(f"  Excluding {len(outlier_samples)} outlier samples from analysis")
                sample_cols = [c for c in sample_cols if c not in outlier_samples]
                # Update DataFrames to remove outlier columns
                # Keep metadata columns plus remaining sample columns
                keep_cols = [c for c in peptide_df.columns if c not in outlier_samples]
                peptide_df = peptide_df[keep_cols]
                peptide_pre_norm_df = peptide_pre_norm_df[keep_cols]
                method_log.append(f"  Excluded samples: {', '.join(outlier_samples)}")
            else:
                logger.warning("  Action='report' - outliers will be included in analysis")
        else:
            logger.info("  No sample outliers detected")

    # Get normalization method from config
    norm_config = config.get("global_normalization", {})
    norm_method = norm_config.get("method", "median")
    rt_lowess_result = None  # Store for QC plots if RT-lowess is used

    # Helper function to compute CVs for reference and QC samples
    def compute_sample_type_cvs(df, sample_cols, sample_to_type):
        """Compute median CVs for reference and QC samples (on LINEAR scale)."""
        ref_cols = [c for c in sample_cols if sample_to_type.get(c) == "reference"]
        qc_cols = [c for c in sample_cols if sample_to_type.get(c) == "qc"]
        ref_cv, qc_cv = None, None
        if ref_cols:
            ref_linear = 2 ** df[ref_cols]
            ref_cv = ((ref_linear.std(axis=1) / ref_linear.mean(axis=1)) * 100).median()
        if qc_cols:
            qc_linear = 2 ** df[qc_cols]
            qc_cv = ((qc_linear.std(axis=1) / qc_linear.mean(axis=1)) * 100).median()
        return ref_cv, qc_cv

    # Build sample_to_type mapping for CV logging
    # Uses build_sample_type_map to handle Sample ID vs Replicate Name mismatch
    sample_to_type = build_sample_type_map(sample_cols, metadata_df)

    if norm_method == "rt_lowess":
        # RT-lowess normalization: align all samples to global median lowess curve
        from .normalization import apply_rt_lowess_normalization

        rt_lowess_config = norm_config.get("rt_lowess", {})
        frac = rt_lowess_config.get("frac", 0.3)
        n_grid_points = rt_lowess_config.get("n_grid_points", 100)

        # Log CVs BEFORE normalization
        if sample_to_type:
            ref_cv_before, qc_cv_before = compute_sample_type_cvs(
                peptide_df, sample_cols, sample_to_type
            )
            cv_parts = []
            if ref_cv_before is not None:
                cv_parts.append(f"Reference: {ref_cv_before:.1f}%")
            if qc_cv_before is not None:
                cv_parts.append(f"QC: {qc_cv_before:.1f}%")
            if cv_parts:
                logger.info(f"  Before normalization - median CV: {', '.join(cv_parts)}")

        rt_lowess_result = apply_rt_lowess_normalization(
            peptide_df,
            sample_cols,
            rt_col="mean_rt",
            frac=frac,
            n_grid_points=n_grid_points,
        )

        # Replace peptide_df with normalized version
        peptide_df = rt_lowess_result.normalized_df

        # Log CVs AFTER normalization
        if sample_to_type:
            ref_cv_after, qc_cv_after = compute_sample_type_cvs(
                peptide_df, sample_cols, sample_to_type
            )
            cv_parts = []
            if ref_cv_after is not None:
                cv_parts.append(f"Reference: {ref_cv_after:.1f}%")
            if qc_cv_after is not None:
                cv_parts.append(f"QC: {qc_cv_after:.1f}%")
            if cv_parts:
                logger.info(f"  After normalization - median CV: {', '.join(cv_parts)}")

        # Compute summary stats for logging
        max_correction = 0.0
        for curve in rt_lowess_result.sample_curves.values():
            if curve is not None:
                diff = np.abs(curve - rt_lowess_result.global_curve)
                max_correction = max(max_correction, np.nanmax(diff))

        method_log.append(
            f"RT-lowess normalization: frac={frac}, "
            f"max correction={max_correction:.2f} log2 ({2**max_correction:.2f}x)"
        )

    elif norm_method == "median" or norm_method is None:
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
        linear_global_median = 2**global_peptide_median
        fold_change_shift = 2**max_pep_shift
        logger.info(
            f"  Global median = {linear_global_median:.0f} (linear), "
            f"max shift = {fold_change_shift:.2f}x"
        )
        method_log.append(f"Peptide median normalization: max shift = {fold_change_shift:.2f}x")

    elif norm_method == "none":
        logger.info("  Skipping global normalization (method='none')")
        method_log.append("Peptide global normalization: skipped")

    else:
        logger.warning(f"  Unknown normalization method: {norm_method}, using median")
        # Fall back to median
        sample_medians = peptide_df[sample_cols].median()
        global_peptide_median = sample_medians.median()
        norm_factors = sample_medians - global_peptide_median
        for col in sample_cols:
            peptide_df[col] = peptide_df[col] - norm_factors[col]
        method_log.append("Peptide median normalization (fallback)")

    # -------------------------------------------------------------------------
    # Stage 2c: Peptide ComBat Batch Correction
    # -------------------------------------------------------------------------
    batch_correction_enabled = config.get("batch_correction", {}).get("enabled", True)

    if batch_correction_enabled and metadata_df is not None:
        logger.info("-" * 60)
        logger.info("Stage 2c: Peptide ComBat Batch Correction")
        logger.info("-" * 60)

        # Get batch info from metadata using helper function
        # This handles Sample ID vs Replicate Name mismatch
        sample_to_batch = build_sample_batch_map(sample_cols, metadata_df)

        if sample_to_batch:
            # Check if all sample columns have batch info
            batches = [sample_to_batch.get(s) for s in sample_cols]
            n_batches = len(set(b for b in batches if b is not None))

            if n_batches < 2:
                logger.warning(f"Only {n_batches} batch - skipping batch correction")
            else:
                from .batch_correction import combat

                # Get sample types for reference-anchored ComBat
                sample_to_type = build_sample_type_map(sample_cols, metadata_df)

                # Prepare data for ComBat (features x samples matrix)
                data_matrix = peptide_df[sample_cols].values
                batch_labels = [sample_to_batch.get(s, "unknown") for s in sample_cols]

                logger.info(f"  Applying ComBat across {n_batches} batches...")

                # Run ComBat - returns corrected matrix directly
                corrected_matrix = combat(data_matrix, batch_labels)

                # Replace sample columns with corrected values
                for i, col in enumerate(sample_cols):
                    peptide_df[col] = corrected_matrix[:, i]

                method_log.append(f"Peptide ComBat: {n_batches} batches corrected")

                # Evaluate using reference and QC samples if available
                if sample_to_type:
                    ref_cols = [c for c in sample_cols if sample_to_type.get(c) == "reference"]
                    qc_cols = [c for c in sample_cols if sample_to_type.get(c) == "qc"]

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
            # No batch column in metadata - try to estimate batches
            batch_est_config = config.get("batch_estimation", {})
            est_method = batch_est_config.get("method", "auto")
            est_n_batches = batch_est_config.get("n_batches")
            gap_iqr = batch_est_config.get("gap_iqr_multiplier", 1.5)

            logger.info("  No batch column in metadata - estimating from acquisition times...")

            sample_to_batch = estimate_batches_from_parquet(
                transition_parquet,
                sample_cols,
                method=est_method,
                n_batches=est_n_batches,
                gap_iqr_multiplier=gap_iqr,
            )

            if sample_to_batch:
                batches = [sample_to_batch.get(s) for s in sample_cols]
                n_batches = len(set(b for b in batches if b is not None))

                if n_batches >= 2:
                    from .batch_correction import combat

                    sample_to_type = build_sample_type_map(sample_cols, metadata_df)
                    data_matrix = peptide_df[sample_cols].values
                    batch_labels = [sample_to_batch.get(s, "unknown") for s in sample_cols]

                    logger.info(f"  Applying ComBat across {n_batches} estimated batches...")

                    corrected_matrix = combat(data_matrix, batch_labels)

                    for i, col in enumerate(sample_cols):
                        peptide_df[col] = corrected_matrix[:, i]

                    method_log.append(f"Peptide ComBat: {n_batches} estimated batches")

                    if sample_to_type:
                        ref_cols = [c for c in sample_cols if sample_to_type.get(c) == "reference"]
                        qc_cols = [c for c in sample_cols if sample_to_type.get(c) == "qc"]

                        if ref_cols and qc_cols:
                            ref_linear = 2 ** peptide_df[ref_cols]
                            qc_linear = 2 ** peptide_df[qc_cols]
                            ref_cv = (ref_linear.std(axis=1) / ref_linear.mean(axis=1)) * 100
                            qc_cv = (qc_linear.std(axis=1) / qc_linear.mean(axis=1)) * 100
                            logger.info(
                                f"  Reference median CV: {ref_cv.median():.1f}%, "
                                f"QC median CV: {qc_cv.median():.1f}%"
                            )
                else:
                    logger.warning(f"Only {n_batches} batch estimated - skipping correction")
            else:
                logger.warning("Could not estimate batches - skipping batch correction")
    elif batch_correction_enabled:
        logger.warning("No metadata available - skipping batch correction")
    else:
        logger.info("  Batch correction disabled in config")

    # Save log2 peptides for protein rollup (internal use)
    # Protein rollup expects log2 input, so we save this BEFORE converting to linear
    peptide_log2_path = output_dir / "peptides_log2_internal.parquet"
    peptide_df.to_parquet(peptide_log2_path, index=False)
    logger.info(f"  Saved log2 peptides for protein rollup: {peptide_log2_path}")

    # Convert from log2 to LINEAR scale for final output (per PRISM specification)
    output_format = config["output"].get("format", "parquet")
    peptide_output = output_dir / f"corrected_peptides.{output_format}"

    # Identify sample columns (numeric columns that are not metadata)
    pep_metadata_cols = [
        peptide_col,
        "n_transitions",
        "mean_rt",
        "Peptide Modified Sequence Unimod Ids",
    ]
    pep_sample_cols = [
        c
        for c in peptide_df.columns
        if c not in pep_metadata_cols and peptide_df[c].dtype in ["float64", "float32"]
    ]

    # Convert log2 to linear for output
    peptide_output_df = peptide_df.copy()
    for col in pep_sample_cols:
        peptide_output_df[col] = np.power(2, peptide_output_df[col])
    logger.info(f"  Converted {len(pep_sample_cols)} sample columns from log2 to linear scale")

    if output_format == "parquet":
        peptide_output_df.to_parquet(peptide_output, index=False)
    else:
        sep = "\t" if output_format == "tsv" else ","
        peptide_output_df.to_csv(peptide_output, sep=sep, index=False)
    logger.info(f"  Saved corrected peptides: {peptide_output}")
    # Stage 3: Protein Parsimony
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Stage 3: Protein Parsimony")
    logger.info("=" * 60)

    # Check if FASTA path is configured for FASTA-based parsimony
    fasta_path = config.get("parsimony", {}).get("fasta_path")
    use_fasta_parsimony = fasta_path and Path(fasta_path).exists()

    if use_fasta_parsimony:
        logger.info(f"  Using FASTA-based parsimony: {fasta_path}")

        # Read peptide sequences for FASTA matching
        mapping_table = pf.read(columns=[peptide_col])
        mapping_df = mapping_table.to_pandas().drop_duplicates()

        from .parsimony import build_peptide_protein_map_from_fasta

        (
            pep_to_prot,
            prot_to_pep,
            prot_to_name,
            prot_to_gene,
            prot_to_description,
        ) = build_peptide_protein_map_from_fasta(
            mapping_df,
            fasta_path=fasta_path,
            peptide_col=peptide_col,
            handle_il_ambiguity=True,
        )
    else:
        if fasta_path:
            logger.warning(f"  FASTA file not found: {fasta_path}")
            logger.warning("  Falling back to Skyline CSV-based parsimony")
        else:
            logger.info("  Using Skyline CSV-based parsimony (no FASTA configured)")

        columns_for_parsimony = [peptide_col, protein_col, protein_name_col]
        if protein_gene_col:
            columns_for_parsimony.append(protein_gene_col)
        if protein_description_col:
            columns_for_parsimony.append(protein_description_col)
        columns_for_parsimony = [c for c in columns_for_parsimony if c in available_columns]

        logger.info("  Reading peptide-protein mappings...")
        mapping_table = pf.read(columns=columns_for_parsimony)
        mapping_df = mapping_table.to_pandas().drop_duplicates()
        logger.info(f"  Found {len(mapping_df):,} unique peptide-protein records")

        (
            pep_to_prot,
            prot_to_pep,
            prot_to_name,
            prot_to_gene,
            prot_to_description,
        ) = build_peptide_protein_map(
            mapping_df,
            peptide_col=peptide_col,
            protein_col=protein_col,
            protein_name_col=protein_name_col,
            protein_gene_col=protein_gene_col,
            protein_description_col=protein_description_col,
        )

    protein_groups = compute_protein_groups(
        prot_to_pep,
        pep_to_prot,
        prot_to_name,
        prot_to_gene,
        prot_to_description,
    )
    logger.info(f"  Computed {len(protein_groups)} protein groups")

    groups_output = output_dir / "protein_groups.tsv"
    export_protein_groups(protein_groups, str(groups_output))

    # =========================================================================
    # Stage 4: Peptide -> Protein Rollup
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Stage 4: Peptide -> Protein Rollup")
    logger.info("=" * 60)

    protein_config = ProteinRollupConfig(
        peptide_col=peptide_col,
        sample_col=sample_col,
        method=config["protein_rollup"].get("method", "median_polish"),
        shared_peptide_handling=config["parsimony"].get("shared_peptide_handling", "all_groups"),
        min_peptides=config["protein_rollup"].get("min_peptides", 3),
        topn_n=config["protein_rollup"].get("topn", {}).get("n", 3),
        progress_interval=1000,
    )

    protein_path = output_dir / "proteins_raw.parquet"
    protein_result = rollup_proteins_streaming(
        peptide_parquet_path=peptide_log2_path,  # Use log2 version, not linear output
        protein_groups=protein_groups,
        output_path=protein_path,
        config=protein_config,
        samples=pep_sample_cols,  # Use filtered sample list (excludes outliers)
        save_residuals=config["output"].get("include_residuals", True),
    )
    rollup_method = config["protein_rollup"].get("method", "median_polish")
    method_log.append(
        f"Protein rollup ({rollup_method}): {protein_result.n_proteins:,} proteins "
        f"from {len(protein_groups):,} groups"
    )

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
        "protein_group",
        "leading_protein",
        "leading_name",
        "leading_uniprot_id",
        "leading_gene_name",
        "leading_description",
        "n_peptides",
        "n_unique_peptides",
        "low_confidence",
    ]
    prot_meta_cols = [c for c in prot_meta_cols if c in protein_df.columns]
    prot_sample_cols = [
        c
        for c in protein_df.columns
        if c not in prot_meta_cols
        and protein_df[c].dtype in ["float64", "float32", "int64", "int32"]
    ]

    # Data is already in LOG2 scale from chunked_processing.py
    # (The linear to log2 conversion is done inside the rollup functions)

    # Keep a copy of raw protein data (in log2 scale) for QC comparison
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
    linear_prot_median = 2**global_protein_median
    fold_prot_shift = 2**max_prot_shift
    logger.info(
        f"  Global median = {linear_prot_median:.0f} (linear), max shift = {fold_prot_shift:.2f}x"
    )
    method_log.append(f"Protein median normalization: max shift = {fold_prot_shift:.2f}x")

    # -------------------------------------------------------------------------
    # Stage 4c: Protein ComBat Batch Correction
    # -------------------------------------------------------------------------
    if batch_correction_enabled and metadata_df is not None:
        logger.info("-" * 60)
        logger.info("Stage 4c: Protein ComBat Batch Correction")
        logger.info("-" * 60)

        batch_col = "batch"
        sample_type_col = "sample_type"
        # Use sample_id (unique across batches) if available
        if "sample_id" in metadata_df.columns:
            meta_sample_col = "sample_id"
        elif "sample" in metadata_df.columns:
            meta_sample_col = "sample"
        else:
            meta_sample_col = "replicate_name"

        if batch_col in metadata_df.columns:
            sample_to_batch = dict(zip(metadata_df[meta_sample_col], metadata_df[batch_col]))

            # Check if all sample columns have batch info
            prot_batches = [sample_to_batch.get(s) for s in prot_sample_cols]
            n_batches = len(set(b for b in prot_batches if b is not None))

            if n_batches < 2:
                logger.warning(f"Only {n_batches} batch - skipping batch correction")
            else:
                from .batch_correction import combat

                # Prepare data for ComBat (features x samples matrix)
                prot_data_matrix = protein_df[prot_sample_cols].values
                prot_batch_labels = [sample_to_batch.get(s, "unknown") for s in prot_sample_cols]

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
                    ref_cols = [c for c in prot_sample_cols if sample_to_type.get(c) == "reference"]
                    qc_cols = [c for c in prot_sample_cols if sample_to_type.get(c) == "qc"]

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

    # Peptides already saved after Stage 2c as corrected_peptides.parquet
    logger.info(f"  Peptides: {peptide_output} (saved after Stage 2c)")

    # Save proteins
    # Convert from log2 to LINEAR scale for final output (per PRISM specification)
    protein_output = output_dir / f"corrected_proteins.{output_format}"

    # Identify sample columns (numeric columns that are not metadata)
    prot_metadata_cols = [
        "protein_group",
        "leading_protein",
        "leading_name",
        "leading_uniprot_id",
        "leading_gene_name",
        "leading_description",
        "n_peptides",
        "n_unique_peptides",
        "low_confidence",
    ]
    prot_sample_cols = [
        c
        for c in protein_df.columns
        if c not in prot_metadata_cols and protein_df[c].dtype in ["float64", "float32"]
    ]

    # Convert log2 to linear for output
    protein_output_df = protein_df.copy()
    for col in prot_sample_cols:
        protein_output_df[col] = np.power(2, protein_output_df[col])
    logger.info(
        f"  Converted {len(prot_sample_cols)} protein sample columns from log2 to linear scale"
    )

    if output_format == "parquet":
        protein_output_df.to_parquet(protein_output, index=False)
    else:
        sep = "\t" if output_format == "tsv" else ","
        protein_output_df.to_csv(protein_output, sep=sep, index=False)
    logger.info(f"  Saved proteins: {protein_output}")

    # Generate pipeline metadata
    metadata = {
        "pipeline_version": __version__,
        "processing_date": datetime.now(timezone.utc).isoformat(),
        "source_files": [str(p.resolve()) for p in input_paths],
        "metadata_files": metadata_input_files,
        "metadata_source": metadata_source,
        "processing_parameters": {
            "transition_rollup": {
                "method": transition_config.method,
                "min_transitions": transition_config.min_transitions,
                "use_ms1": not transition_config.exclude_precursor,
            },
            "protein_rollup": {
                "method": protein_config.method,
                "shared_peptide_handling": protein_config.shared_peptide_handling,
                "min_peptides": protein_config.min_peptides,
            },
            "batch_correction": {
                "enabled": batch_correction_enabled,
                "method": config.get("batch_correction", {}).get("method", "combat"),
            },
        },
        "method_log": method_log,
        "output_files": {
            "peptides": str(peptide_output),
            "proteins": str(protein_output),
            "protein_groups": str(groups_output),
        },
        "statistics": {
            "n_samples": len(samples),
            "n_peptides": peptide_result.n_peptides,
            "n_transitions": n_transitions,
            "n_proteins": protein_result.n_proteins,
            "n_protein_groups": len(protein_groups),
        },
    }
    metadata_output = output_dir / "metadata.json"
    with open(metadata_output, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"  Saved metadata: {metadata_output}")

    # -------------------------------------------------------------------------
    # Stage 5b: Generate QC Report
    # -------------------------------------------------------------------------
    qc_config = config.get("qc_report", {})
    if qc_config.get("enabled", True):
        logger.info("-" * 60)
        logger.info("Stage 5b: Generating QC Report")
        logger.info("-" * 60)

        # Build sample types mapping (handles Sample ID vs Replicate Name mismatch)
        sample_types_map = build_sample_type_map(sample_cols, metadata_df)
        qc_report_path = output_dir / qc_config.get("filename", "qc_report.html")

        try:
            # Build sample_batches mapping for RT-lowess plots
            sample_batches_map = build_sample_batch_map(sample_cols, metadata_df) or None

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
                save_plots=qc_config.get("save_plots", True),
                embed_plots=qc_config.get("embed_plots", True),
                rt_lowess_result=rt_lowess_result,
                sample_batches=sample_batches_map,
                pipeline_metadata=metadata,
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
    - peptides_rollup.parquet (raw peptides from transition rollup)
    - corrected_peptides.parquet (normalized/batch-corrected peptides)
    - proteins_raw.parquet (raw proteins from peptide rollup)
    - corrected_proteins.parquet (normalized/batch-corrected proteins)
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

    peptide_raw_path = output_dir / "peptides_rollup.parquet"
    if not peptide_raw_path.exists():
        logger.error(f"Could not find peptide raw data file: {peptide_raw_path}")
        return 1

    # Load data files
    required_files = {
        "peptide_raw": peptide_raw_path,
        "peptide_corrected": output_dir / "corrected_peptides.parquet",
        "protein_raw": output_dir / "proteins_raw.parquet",
        "protein_corrected": output_dir / "corrected_proteins.parquet",
    }

    for name, path in required_files.items():
        if not path.exists():
            logger.error(f"Required file not found: {path}")
            return 1

    logger.info("Loading processed data...")
    peptide_raw = pd.read_parquet(required_files["peptide_raw"])
    peptide_corrected = pd.read_parquet(required_files["peptide_corrected"])
    protein_raw = pd.read_parquet(required_files["protein_raw"])
    protein_corrected = pd.read_parquet(required_files["protein_corrected"])

    logger.info(f"  Peptides: {len(peptide_corrected):,}")
    logger.info(f"  Proteins: {len(protein_corrected):,}")

    # Load sample metadata for sample types
    sample_types = {}
    metadata_path = output_dir / "sample_metadata.tsv"
    if metadata_path.exists():
        metadata_df = pd.read_csv(metadata_path, sep="\t")
        if "sample_type" in metadata_df.columns:
            # Use sample_id (full name) if available, else fallback to sample
            if "sample_id" in metadata_df.columns:
                sample_types = dict(zip(metadata_df["sample_id"], metadata_df["sample_type"]))
            elif "sample" in metadata_df.columns:
                sample_types = dict(zip(metadata_df["sample"], metadata_df["sample_type"]))
            logger.info(f"  Sample types loaded: {len(sample_types)}")
    else:
        logger.warning("sample_metadata.tsv not found, sample types will not be colored")

    # Determine sample columns (exclude metadata columns)
    meta_cols = [
        "Peptide Modified Sequence Unimod Ids",
        "Peptide Modified Sequence",
        "Peptide",
        "n_transitions",
        "mean_rt",
        "protein",
        "n_peptides",
    ]
    sample_cols = [c for c in peptide_corrected.columns if c not in meta_cols]
    logger.info(f"  Samples: {len(sample_cols)}")

    # Convert raw data to log2 scale (rollup files are in linear scale)
    # Check if raw data appears to be in linear scale (values > 100 suggest linear)
    peptide_raw_log2 = peptide_raw.copy()
    protein_raw_log2 = protein_raw.copy()

    # Peptide raw: check median of first sample column
    if sample_cols and peptide_raw[sample_cols[0]].median() > 100:
        logger.info("  Converting raw peptide data to log2 scale...")
        for col in sample_cols:
            if col in peptide_raw_log2.columns:
                peptide_raw_log2[col] = np.log2(peptide_raw_log2[col].replace(0, np.nan))

    # Protein raw: same check
    prot_sample_cols = [
        c
        for c in protein_raw.columns
        if c not in meta_cols
        and c
        not in [
            "protein_group",
            "leading_protein",
            "leading_name",
            "n_unique_peptides",
            "low_confidence",
        ]
    ]
    if prot_sample_cols and protein_raw[prot_sample_cols[0]].median() > 100:
        logger.info("  Converting raw protein data to log2 scale...")
        for col in prot_sample_cols:
            if col in protein_raw_log2.columns:
                protein_raw_log2[col] = np.log2(protein_raw_log2[col].replace(0, np.nan))

    # Generate QC report
    report_path = output_dir / args.output
    logger.info(f"Generating QC report: {report_path}")

    try:
        generate_comprehensive_qc_report(
            peptide_raw=peptide_raw_log2,
            peptide_corrected=peptide_corrected,
            protein_raw=protein_raw_log2,
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
    return """\
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
  # Method: sum (default), consensus, median_polish, adaptive, topn, library_assist
  method: "sum"

  # Minimum transitions required per peptide
  min_transitions: 3

  # For library_assist method: path to spectral library (.blib or .tsv)
  # library_assist:
  #   library_path: "/path/to/library.blib"
  #   fitting_method: "median_polish"  # or "least_squares"
  #   outlier_threshold: 1.0  # 1.0=obs>2x pred, 2.0=obs>3x pred (conservative)
  #   remove_outliers: true   # Set false to disable outlier removal (debugging)

global_normalization:
  method: "rt_lowess"
  rt_lowess:
    frac: 0.3
    n_grid_points: 100

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
"""


def get_full_config_template() -> str:
    """Return the full annotated config template."""
    return """\
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
# When batch information is not provided in metadata, PRISM can estimate batches
# from acquisition timestamps.
#
# TIP: Include "Result File > Acquired Time" in your Skyline report for best
# automatic batch detection.

batch_estimation:
  # Estimation method:
  #   "auto"  - Try gap detection first, fall back to fixed if no gaps found
  #   "fixed" - Divide evenly into n_batches by acquisition time order
  #   "gap"   - Only use gap detection (skip batch correction if no gaps)
  method: "auto"

  # Number of batches for "fixed" method (required for fixed, optional fallback for auto)
  # Set to null to disable fixed-count batch estimation
  n_batches: null

  # IQR multiplier for gap detection (lower = more sensitive to time gaps)
  # A gap > Q3 + (gap_iqr_multiplier * IQR) is considered a batch break
  gap_iqr_multiplier: 1.5

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
  #   sum            - Simple sum of fragment intensities (default, robust)
  #   consensus      - Inverse-variance weighted sum (down-weights outlier transitions)
  #   median_polish  - Tukey median polish (robust to interference)
  #   adaptive       - Learned weights based on intensity, m/z, shape correlation
  #   topn           - Select top N transitions by correlation
  #   library_assist - Uses spectral library to detect/exclude interfered transitions
  #                    via iterative least squares fitting (recommended with library)
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
  # The optimizer learns beta_mz and beta_shape_corr_outlier from reference samples
  # by minimizing CV. These weights adjust transition contributions based on:
  #   - beta_mz: Higher m/z transitions may have better signal (positive = favor high m/z)
  #   - beta_shape_corr_outlier: Down-weight transitions with interference (negative = penalize)
  learn_adaptive_weights: true
  adaptive_rollup:
    # Starting values for optimization (usually left at 0)
    beta_mz: 0.0
    beta_shape_corr_outlier: 0.0
    # Threshold for "low" shape correlation (transitions often below this have interference)
    shape_corr_low_threshold: 0.7
    # Minimum improvement required to use learned weights vs simple sum
    # This is specific to adaptive method - library_assist always uses library fitting
    min_improvement_pct: 0.1

  # Library-assist parameters (if method: library_assist)
  # Uses spectral library as a prior for transition ionization efficiency (row effects).
  # Compares observed fragment ratios to library and removes outliers iteratively.
  # Only HIGH residuals are outliers (signal > expected = interference).
  # Low or negative residuals are valid - they indicate low abundance, not interference.
  library_assist:
    library_path: null          # Path to .blib or .tsv spectral library (REQUIRED)
    mz_tolerance: 0.02          # m/z tolerance for matching fragments (Da)
    min_matched_fragments: 3    # Minimum fragments for valid fit (else NaN)
    fitting_method: "median_polish"  # "median_polish" (RECOMMENDED) or "least_squares"
                                # median_polish: Uses MEDIAN to estimate sample scale factor
                                #   - Robust to 1-2 interfered transitions automatically
                                #   - Based on: log(obs) = log(lib) + beta_s + epsilon
                                # least_squares: Classic OLS: scale = (lib . obs) / (lib . lib)
                                #   - More sensitive to outliers
                                #   - May be better for very clean data
    outlier_threshold: 1.0      # Normalized residual threshold: (obs-pred)/pred > threshold
                                # 1.0 = observed > 2x predicted (standard sensitivity)
                                # 2.0 = observed > 3x predicted (conservative)
                                # Removes one outlier at a time, respects min_matched_fragments
    remove_outliers: true       # Set to false to disable outlier removal entirely
                                # Useful for debugging or when median robustness is sufficient

# =============================================================================
# Global Normalization
# =============================================================================
# Correct for overall sample loading differences.

global_normalization:
  # Method:
  #   median    - Subtract sample median (recommended for simple cases)
  #   rt_lowess - RT-dependent Lowess normalization (RECOMMENDED default)
  #               Corrects for RT-dependent systematic effects (ion suppression, gradient issues)
  #   vsn       - Variance Stabilizing Normalization
  #   quantile  - Force identical distributions (aggressive)
  #   none      - Skip normalization
  method: "rt_lowess"

  # RT-lowess parameters (only used when method: rt_lowess)
  rt_lowess:
    # Lowess smoothing fraction (0.0-1.0)
    # Lower = more flexible curve, higher = smoother curve
    # Default 0.3 is a good balance for typical LC-MS gradients
    frac: 0.3
    # Number of RT grid points for interpolation
    n_grid_points: 100

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
"""


def cmd_viewer(args: argparse.Namespace) -> int:
    """Launch the interactive PRISM viewer.

    Args:
        args: Command-line arguments containing output_dir.

    Returns:
        Exit code (0 = success).

    """
    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        logger.error(f"Directory not found: {output_dir}")
        return 1

    try:
        # Override sys.argv for the viewer
        import sys

        from skyline_prism.gui.viewer import main as viewer_main

        original_argv = sys.argv
        sys.argv = ["prism-viewer", str(output_dir)]
        try:
            return viewer_main()
        finally:
            sys.argv = original_argv
    except ImportError as e:
        logger.error(f"PyQt6 not available: {e}\nInstall with: pip install skyline-prism[gui]")
        return 1


def main() -> int:
    """Provide main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="prism",
        description="Skyline-PRISM: Proteomics Reference-Integrated Signal Modeling\n\n"
        "Normalization and batch correction for Skyline proteomics data\n"
        "with robust protein quantification using Tukey median polish.\n\n"
        "Primary usage:\n"
        "  prism run -i data.parquet -o output_dir/ -c config.yaml\n\n"
        "See https://github.com/maccoss/skyline-prism for documentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command (primary) - executes full pipeline
    run_parser = subparsers.add_parser(
        "run",
        help="Run the full PRISM pipeline (recommended)",
        description="Execute the complete PRISM pipeline: normalization, batch correction, "
        "and protein rollup. Produces both peptide-level and protein-level outputs.",
    )
    run_parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        help="Input file(s): one or more Skyline report CSV/TSV files, "
        "or a single merged parquet file. Multiple CSV files are "
        "treated as separate batches. If a merged_data.parquet "
        "already exists in the output directory and matches the "
        "input files, it will be reused (use --force-reprocess to "
        "override).",
    )
    run_parser.add_argument(
        "-o", "--output-dir", required=True, help="Output directory for results"
    )
    run_parser.add_argument("-c", "--config", help="Configuration YAML file")
    run_parser.add_argument(
        "-m",
        "--metadata",
        nargs="+",
        help="Sample metadata file(s) (CSV or TSV). Multiple files will be "
        "merged. If not provided, metadata is auto-generated from sample names.",
    )
    run_parser.add_argument(
        "--reference-pattern",
        nargs="+",
        help="Patterns to identify reference samples (e.g., -Pool_). "
        "Samples matching these are used for RT correction learning.",
    )
    run_parser.add_argument(
        "--qc-pattern",
        nargs="+",
        help="Patterns to identify QC samples (e.g., -Carl_). "
        "Samples matching these are used for validation.",
    )
    run_parser.add_argument(
        "--from-provenance",
        help="Load configuration from a previous run's metadata.json file. "
        "Enables reproducibility by using the exact same parameters. "
        "If --config is also provided, it overrides provenance settings.",
    )
    run_parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force reprocessing of input files even if a valid cached parquet "
        "exists. By default, if a merged_data.parquet file exists and its "
        "source file fingerprints match the input files, it will be reused.",
    )

    # Merge command - utility for combining reports
    merge_parser = subparsers.add_parser(
        "merge", help="Merge multiple Skyline reports into one parquet file"
    )
    merge_parser.add_argument("reports", nargs="+", help="Skyline report files")
    merge_parser.add_argument("-o", "--output", required=True, help="Output parquet path")
    merge_parser.add_argument("-m", "--metadata", help="Sample metadata TSV")
    merge_parser.add_argument(
        "--no-partition", action="store_true", help="Do not partition by batch"
    )

    # QC report command - regenerate QC report from existing output
    qc_parser = subparsers.add_parser("qc", help="Regenerate QC report from existing PRISM output")
    qc_parser.add_argument(
        "-d",
        "--dir",
        required=True,
        help="PRISM output directory containing processed parquet files",
    )
    qc_parser.add_argument(
        "-o",
        "--output",
        default="qc_report.html",
        help="Output HTML report filename (default: qc_report.html)",
    )
    qc_parser.add_argument(
        "--no-save-plots", action="store_true", help="Do not save individual plot PNG files"
    )
    qc_parser.add_argument(
        "--no-embed", action="store_true", help="Do not embed plots in HTML (link to files instead)"
    )

    # Config template command - output annotated config template
    config_parser = subparsers.add_parser(
        "config-template",
        help="Output an annotated configuration template file",
        description="Generate a fully-annotated configuration template YAML file. "
        "This is the recommended starting point for configuring a new analysis.",
    )
    config_parser.add_argument("-o", "--output", help="Output file path (default: print to stdout)")
    config_parser.add_argument(
        "--minimal",
        action="store_true",
        help="Output minimal config with only commonly-changed options",
    )

    # Viewer command - launch interactive PyQt viewer
    viewer_parser = subparsers.add_parser(
        "viewer",
        help="Launch the interactive PRISM results viewer",
        description="Open the PyQt-based viewer to explore PRISM analysis results. "
        "Requires PyQt6 (install with: pip install skyline-prism[gui]).",
    )
    viewer_parser.add_argument(
        "output_dir",
        help="PRISM output directory containing processed parquet files",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "merge":
        return cmd_merge(args)
    elif args.command == "qc":
        return cmd_qc(args)
    elif args.command == "config-template":
        return cmd_config_template(args)
    elif args.command == "viewer":
        return cmd_viewer(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
