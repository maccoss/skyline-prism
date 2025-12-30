"""Chunked/streaming processing for large datasets.

This module handles memory-efficient processing of large proteomics datasets
that cannot fit in memory. It provides chunked versions of the main PRISM
pipeline stages:

1. Transition → Peptide rollup (streaming by peptide groups)
2. Peptide → Protein rollup (streaming by protein groups)

Key design principles:
- Never load entire dataset into memory
- Process peptides/proteins independently (embarrassingly parallel)
- Write output incrementally
- Support multiprocessing for CPU-bound operations
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .parsimony import ProteinGroup
from .rollup import tukey_median_polish
from .transition_rollup import (
    AdaptiveRollupParams,
    rollup_peptide_adaptive,
    rollup_peptide_topn,
)

logger = logging.getLogger(__name__)

# Threshold for large file detection (in bytes) - 500 MB
LARGE_FILE_THRESHOLD_BYTES = 500 * 1024 * 1024


@dataclass
class ChunkedRollupConfig:
    """Configuration for chunked rollup processing."""

    # Column names (using Skyline export names as defaults)
    peptide_col: str = "Peptide Modified Sequence"
    transition_col: str = "Fragment Ion"
    precursor_charge_col: str = "Precursor Charge"  # For unique transition ID
    product_charge_col: str = "Product Charge"  # For unique transition ID
    sample_col: str = "Replicate Name"
    abundance_col: str = "Area"
    shape_corr_col: str = "Shape Correlation"
    coeluting_col: str = "Coeluting"
    rt_col: str = "Retention Time"
    batch_col: str = "Batch"
    mz_col: str = "Product Mz"  # For adaptive rollup

    # Rollup parameters
    method: str = "sum"
    min_transitions: int = 3
    log_transform: bool = True
    adaptive_params: AdaptiveRollupParams | None = None  # For adaptive method
    exclude_precursor: bool = True  # Exclude MS1 precursor ions from rollup

    # Top-N method parameters
    topn_count: int = 3  # Number of transitions to select
    topn_selection: str = "correlation"  # "correlation" or "intensity"
    topn_weighting: str = "sqrt"  # "sum" or "sqrt"

    # Processing parameters
    n_workers: int = 1  # Number of parallel workers (0 = all CPUs)
    peptide_batch_size: int = 1000  # Process this many peptides at once
    progress_interval: int = 10000  # Log progress every N peptides
    max_memory_gb: float = 8.0  # Maximum memory to use for parallel processing (per worker)


@dataclass
class PeptideRollupResult:
    """Result for a single peptide's rollup."""

    peptide: str
    abundances: dict[str, float]  # sample -> abundance (log2)
    uncertainties: dict[str, float]  # sample -> uncertainty
    n_transitions: int
    residuals: dict[str, dict[str, float]] | None = None  # transition -> sample -> residual
    mean_rt: float | None = None  # Mean retention time for this peptide


@dataclass
class StreamingRollupResult:
    """Result of streaming transition → peptide rollup."""

    output_path: Path
    n_peptides: int
    n_samples: int
    samples: list[str]
    method: str
    residuals_path: Path | None = None


def _get_required_columns(config: ChunkedRollupConfig) -> tuple[list[str], list[str]]:
    """Get list of columns needed from parquet file."""
    cols = [
        config.peptide_col,
        config.transition_col,
        config.precursor_charge_col,  # For unique transition ID
        config.product_charge_col,  # For unique transition ID
        config.sample_col,
        config.abundance_col,
    ]
    # Optional columns
    optional = [config.shape_corr_col, config.coeluting_col, config.rt_col, config.batch_col]
    return cols, optional


def _process_single_peptide(
    pep_data: pd.DataFrame,
    peptide: str,
    samples: list[str],
    config: ChunkedRollupConfig,
) -> PeptideRollupResult:
    """Process a single peptide's transitions to get peptide-level abundance.

    This is the core rollup logic, isolated for parallel execution.
    """
    # Filter out MS1 precursor ions - only use MS2 fragment ions for rollup
    if config.exclude_precursor:
        pep_data = pep_data[~pep_data[config.transition_col].str.startswith('precursor')]

    # Create unique transition ID combining Fragment Ion + Precursor Charge + Product Charge
    # This ensures transitions from different precursor charges are kept separate
    pep_data = pep_data.copy()
    pep_data["_transition_id"] = (
        pep_data[config.transition_col].astype(str)
        + "_z"
        + pep_data[config.precursor_charge_col].astype(str)
        + "_"
        + pep_data[config.product_charge_col].astype(str)
    )

    # Pivot to get transition × sample matrix
    intensity_matrix = pep_data.pivot_table(
        index="_transition_id",
        columns=config.sample_col,
        values=config.abundance_col,
        aggfunc="first",
    )

    # Ensure all samples present
    for sample in samples:
        if sample not in intensity_matrix.columns:
            intensity_matrix[sample] = np.nan
    intensity_matrix = intensity_matrix.reindex(columns=samples)

    # Impute missing/invalid values with low abundance quantity
    # Step 1: Convert negative values to 0 (these are invalid measurements)
    intensity_matrix = intensity_matrix.clip(lower=0)

    # Step 2: Calculate imputation value from valid positive measurements
    # Use half the 1st percentile of positive values (robust to outliers)
    positive_values = intensity_matrix.values[intensity_matrix.values > 0]
    if len(positive_values) > 0:
        p1_value = np.percentile(positive_values, 1)
        impute_value = max(p1_value * 0.5, 1.0)  # At least 1.0
    else:
        impute_value = 1.0  # Fallback if no positive values

    # Step 3: Replace zeros AND NaN with imputation value
    # Zero Area means "detected but below LOD", NaN means missing measurement
    # Both get imputed to a low but non-zero value for downstream analysis
    intensity_matrix = intensity_matrix.fillna(impute_value)
    intensity_matrix = intensity_matrix.replace(0, impute_value)

    # Log transform (all values now guaranteed positive)
    if config.log_transform:
        intensity_matrix = np.log2(intensity_matrix)

    # Get mean RT if available
    mean_rt = None
    if config.rt_col in pep_data.columns:
        mean_rt = pep_data[config.rt_col].mean()

    if config.method == "median_polish":
        if len(intensity_matrix) >= config.min_transitions:
            result = tukey_median_polish(intensity_matrix)
            n_used = len(intensity_matrix)
            # Scale col_effects by n_transitions to make comparable to sum
            # In log2 space: add log2(n) is equivalent to multiply by n in linear
            scale_factor = np.log2(n_used) if n_used > 0 else 0
            abundances = {s: v + scale_factor for s, v in result.col_effects.to_dict().items()}
            # Uncertainty from residual variance
            residual_var = result.residuals.var().mean()
            uncertainties = {s: np.sqrt(residual_var) for s in samples}
            # Store residuals for output
            residuals = {
                str(t): result.residuals.loc[t].to_dict() for t in result.residuals.index
            }
        else:
            abundances = {s: np.nan for s in samples}
            uncertainties = {s: np.nan for s in samples}
            n_used = 0
            residuals = None

    elif config.method == "sum":
        linear = 2**intensity_matrix if config.log_transform else intensity_matrix
        summed = linear.sum(axis=0)
        if config.log_transform:
            abundances = np.log2(summed.clip(lower=1)).to_dict()
        else:
            abundances = summed.to_dict()
        uncertainties = {s: np.nan for s in samples}
        n_used = (~intensity_matrix.isna()).sum().min()
        residuals = None

    elif config.method == "topn":
        # Top-N selection by correlation or intensity
        if config.shape_corr_col in pep_data.columns:
            shape_corr_matrix = pep_data.pivot_table(
                index="_transition_id",
                columns=config.sample_col,
                values=config.shape_corr_col,
                aggfunc="first",
            )
            shape_corr_matrix = shape_corr_matrix.reindex(
                index=intensity_matrix.index, columns=samples
            ).fillna(0.0)  # Missing = low correlation
        else:
            shape_corr_matrix = pd.DataFrame(
                1.0, index=intensity_matrix.index, columns=samples
            )

        abund_series, uncert_series, _, n_used = rollup_peptide_topn(
            intensity_matrix,
            shape_corr_matrix,
            n_transitions=config.topn_count,
            selection_method=config.topn_selection,
            weighting=config.topn_weighting,
            min_transitions=config.min_transitions,
        )
        abundances = abund_series.to_dict()
        uncertainties = uncert_series.to_dict()
        residuals = None

    elif config.method == "adaptive":
        params = config.adaptive_params or AdaptiveRollupParams()

        # Get shape correlation matrix
        if config.shape_corr_col in pep_data.columns:
            shape_corr_matrix = pep_data.pivot_table(
                index="_transition_id",
                columns=config.sample_col,
                values=config.shape_corr_col,
                aggfunc="first",
            )
            shape_corr_matrix = shape_corr_matrix.reindex(
                index=intensity_matrix.index, columns=samples
            ).fillna(1.0)
        else:
            shape_corr_matrix = pd.DataFrame(
                1.0, index=intensity_matrix.index, columns=samples
            )

        # Get m/z values per transition
        if config.mz_col in pep_data.columns:
            mz_pivot = pep_data.pivot_table(
                index="_transition_id",
                columns=config.sample_col,
                values=config.mz_col,
                aggfunc="first",
            )
            mz_values = mz_pivot.apply(
                lambda x: x.dropna().iloc[0] if x.notna().any() else 0.0, axis=1
            )
            mz_values = mz_values.reindex(intensity_matrix.index).fillna(0)
        else:
            mz_values = pd.Series(0.0, index=intensity_matrix.index)

        abund_series, uncert_series, _, n_used = rollup_peptide_adaptive(
            intensity_matrix,
            mz_values,
            shape_corr_matrix,
            params,
            config.min_transitions,
        )
        abundances = abund_series.to_dict()
        uncertainties = uncert_series.to_dict()
        residuals = None

    else:
        raise ValueError(f"Unknown rollup method: {config.method}")

    return PeptideRollupResult(
        peptide=peptide,
        abundances=abundances,
        uncertainties=uncertainties,
        n_transitions=n_used,
        residuals=residuals,
        mean_rt=mean_rt,
    )


def _worker_process_batch(
    args: tuple[dict[str, pd.DataFrame], list[str], dict],
) -> list[dict]:
    """Worker function for parallel peptide processing.
    
    Takes a tuple of (peptide_data_dict, samples, config_dict) and returns
    a list of result dictionaries (easily picklable).
    
    This function is designed to be pickle-able for multiprocessing.
    """
    peptide_data_dict, samples, config_dict = args
    
    # Reconstruct config from dict (dataclasses aren't always pickle-friendly)
    config = ChunkedRollupConfig(**config_dict)
    
    results = []
    for peptide, pep_data in peptide_data_dict.items():
        result = _process_single_peptide(pep_data, peptide, samples, config)
        # Convert to dict for pickling
        results.append({
            "peptide": result.peptide,
            "abundances": result.abundances,
            "uncertainties": result.uncertainties,
            "n_transitions": result.n_transitions,
            "residuals": result.residuals,
            "mean_rt": result.mean_rt,
        })
    return results


def _process_peptide_batch(
    peptide_data_dict: dict[str, pd.DataFrame],
    samples: list[str],
    config: ChunkedRollupConfig,
) -> list[PeptideRollupResult]:
    """Process a batch of peptides (for parallel execution)."""
    results = []
    for peptide, pep_data in peptide_data_dict.items():
        result = _process_single_peptide(pep_data, peptide, samples, config)
        results.append(result)
    return results


def get_unique_values_from_parquet(
    parquet_path: Path,
    column: str,
) -> list:
    """Get unique values from a column without loading entire file."""
    pf = pq.ParquetFile(parquet_path)
    # Read just the one column
    table = pf.read(columns=[column])
    unique = table.column(column).unique().to_pylist()
    return unique


def stream_peptide_groups(
    parquet_path: Path,
    config: ChunkedRollupConfig,
    peptide_batch_size: int = 1000,
) -> Iterator[dict[str, pd.DataFrame]]:
    """Stream peptide groups from parquet file.

    Reads the parquet file in row groups, accumulates rows by peptide,
    and yields complete peptide batches.

    This is memory-efficient because:
    - Only holds one row group + incomplete peptides in memory
    - Yields peptides as soon as batch is complete
    """
    # Get columns we need
    required_cols, optional_cols = _get_required_columns(config)

    # Read parquet metadata
    pf = pq.ParquetFile(parquet_path)

    # Determine which optional columns exist
    schema_names = set(pf.schema_arrow.names)
    cols_to_read = required_cols.copy()
    for col in optional_cols:
        if col in schema_names:
            cols_to_read.append(col)

    # We need to accumulate all data for each peptide before processing
    # Strategy: scan once to get peptide -> row indices, then read by peptide
    # Alternative: sort parquet by peptide first

    # For now, use the simpler approach: read all data for each peptide using filtering
    # This is less efficient but works with unsorted data

    logger.info("Scanning parquet for unique peptides...")
    peptides = get_unique_values_from_parquet(parquet_path, config.peptide_col)
    logger.info(f"  Found {len(peptides):,} unique peptides")

    # Process in batches
    batch = {}
    for i, peptide in enumerate(peptides):
        # Read just this peptide's data
        # Note: This does a scan per peptide - not optimal but memory-safe
        # For truly large files, we'd want to sort first
        table = pq.read_table(
            parquet_path,
            columns=cols_to_read,
            filters=[(config.peptide_col, "=", peptide)],
        )
        batch[peptide] = table.to_pandas()

        if len(batch) >= peptide_batch_size:
            yield batch
            batch = {}

        if (i + 1) % config.progress_interval == 0:
            logger.info(f"  Loaded {i + 1:,} / {len(peptides):,} peptides...")

    # Yield remaining
    if batch:
        yield batch


def rollup_transitions_streaming(
    parquet_path: Path,
    output_path: Path,
    config: ChunkedRollupConfig,
    samples: list[str] | None = None,
    save_residuals: bool = True,
) -> StreamingRollupResult:
    """Roll up transitions to peptides using streaming/chunked processing.

    Memory-efficient version of rollup_transitions_to_peptides that never
    loads the entire dataset into memory.

    Args:
        parquet_path: Path to merged transition-level parquet file
        output_path: Path for output peptide-level parquet file
        config: Rollup configuration
        samples: List of sample names (if None, extracted from data)
        save_residuals: Whether to save residuals to separate file

    Returns:
        StreamingRollupResult with output paths and statistics

    """
    logger.info(f"Starting streaming transition rollup: {parquet_path}")
    logger.info(f"  Method: {config.method}")
    logger.info(f"  Workers: {config.n_workers or mp.cpu_count()}")

    # Get samples if not provided
    if samples is None:
        samples = get_unique_values_from_parquet(parquet_path, config.sample_col)
        samples = sorted(samples)
    logger.info(f"  Samples: {len(samples)}")

    # Prepare output
    output_path = Path(output_path)
    peptide_rows = []
    residual_rows = []
    n_peptides = 0
    n_filtered = 0

    # Determine number of workers
    n_workers = config.n_workers if config.n_workers > 0 else mp.cpu_count()

    # Process peptides in streaming fashion
    if n_workers == 1:
        # Single-threaded processing
        for batch in stream_peptide_groups(parquet_path, config, config.peptide_batch_size):
            results = _process_peptide_batch(batch, samples, config)

            for result in results:
                # Skip peptides with insufficient transitions
                if result.n_transitions < config.min_transitions:
                    n_filtered += 1
                    continue

                # Build peptide row
                row = {
                    config.peptide_col: result.peptide,
                    "n_transitions": result.n_transitions,
                }
                if result.mean_rt is not None:
                    row["mean_rt"] = result.mean_rt
                row.update(result.abundances)
                peptide_rows.append(row)

                # Build residual rows if available
                if save_residuals and result.residuals:
                    for transition, sample_residuals in result.residuals.items():
                        res_row = {
                            config.peptide_col: result.peptide,
                            config.transition_col: transition,
                        }
                        res_row.update(sample_residuals)
                        residual_rows.append(res_row)

                n_peptides += 1

            if n_peptides % config.progress_interval == 0:
                logger.info(f"  Processed {n_peptides:,} peptides...")

    else:
        # Multi-process version using ProcessPoolExecutor
        logger.info(f"  Using {n_workers} parallel workers")
        
        # Convert config to dict for pickling
        config_dict = {
            "peptide_col": config.peptide_col,
            "transition_col": config.transition_col,
            "precursor_charge_col": config.precursor_charge_col,
            "product_charge_col": config.product_charge_col,
            "sample_col": config.sample_col,
            "abundance_col": config.abundance_col,
            "shape_corr_col": config.shape_corr_col,
            "coeluting_col": config.coeluting_col,
            "rt_col": config.rt_col,
            "batch_col": config.batch_col,
            "mz_col": config.mz_col,
            "method": config.method,
            "min_transitions": config.min_transitions,
            "log_transform": config.log_transform,
            "adaptive_params": config.adaptive_params,
            "exclude_precursor": config.exclude_precursor,
            "topn_count": config.topn_count,
            "topn_selection": config.topn_selection,
            "topn_weighting": config.topn_weighting,
            "n_workers": 1,  # Workers don't spawn sub-workers
            "peptide_batch_size": config.peptide_batch_size,
            "progress_interval": config.progress_interval,
            "max_memory_gb": config.max_memory_gb,
        }
        
        # Collect batches first, then process in parallel
        # This trades some memory for better parallelization
        batches_to_process = []
        for batch in stream_peptide_groups(parquet_path, config, config.peptide_batch_size):
            batches_to_process.append((batch, samples, config_dict))
            
            # Process when we have enough batches for all workers
            if len(batches_to_process) >= n_workers:
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = [executor.submit(_worker_process_batch, b) for b in batches_to_process]
                    for future in as_completed(futures):
                        try:
                            batch_results = future.result()
                            for result in batch_results:
                                # Skip peptides with insufficient transitions
                                if result["n_transitions"] < config.min_transitions:
                                    n_filtered += 1
                                    continue
                                    
                                # Build peptide row
                                row = {
                                    config.peptide_col: result["peptide"],
                                    "n_transitions": result["n_transitions"],
                                }
                                if result["mean_rt"] is not None:
                                    row["mean_rt"] = result["mean_rt"]
                                row.update(result["abundances"])
                                peptide_rows.append(row)
                                
                                # Build residual rows if available
                                if save_residuals and result["residuals"]:
                                    for transition, sample_residuals in result["residuals"].items():
                                        res_row = {
                                            config.peptide_col: result["peptide"],
                                            config.transition_col: transition,
                                        }
                                        res_row.update(sample_residuals)
                                        residual_rows.append(res_row)
                                        
                                n_peptides += 1
                        except Exception as e:
                            logger.error(f"Worker error: {e}")
                            raise
                
                batches_to_process = []
                if n_peptides % config.progress_interval == 0:
                    logger.info(f"  Processed {n_peptides:,} peptides...")
        
        # Process remaining batches
        if batches_to_process:
            with ProcessPoolExecutor(max_workers=min(n_workers, len(batches_to_process))) as executor:
                futures = [executor.submit(_worker_process_batch, b) for b in batches_to_process]
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        for result in batch_results:
                            if result["n_transitions"] < config.min_transitions:
                                n_filtered += 1
                                continue
                            row = {
                                config.peptide_col: result["peptide"],
                                "n_transitions": result["n_transitions"],
                            }
                            if result["mean_rt"] is not None:
                                row["mean_rt"] = result["mean_rt"]
                            row.update(result["abundances"])
                            peptide_rows.append(row)
                            
                            if save_residuals and result["residuals"]:
                                for transition, sample_residuals in result["residuals"].items():
                                    res_row = {
                                        config.peptide_col: result["peptide"],
                                        config.transition_col: transition,
                                    }
                                    res_row.update(sample_residuals)
                                    residual_rows.append(res_row)
                            n_peptides += 1
                    except Exception as e:
                        logger.error(f"Worker error: {e}")
                        raise

    logger.info(f"  Completed: {n_peptides:,} peptides processed")
    if n_filtered > 0:
        logger.info(f"  Filtered: {n_filtered:,} peptides with < {config.min_transitions} transitions")

    # Write peptide output

    peptide_df = pd.DataFrame(peptide_rows)
    # Reorder columns: peptide, metadata, then samples
    meta_cols = [config.peptide_col, "n_transitions"]
    if "mean_rt" in peptide_df.columns:
        meta_cols.append("mean_rt")
    sample_cols = [s for s in samples if s in peptide_df.columns]
    peptide_df = peptide_df[meta_cols + sample_cols]

    # Convert sample columns from log2 to linear before writing output
    # This is required by the PRISM specification and for downstream analysis
    if len(sample_cols) > 0:
        peptide_df[sample_cols] = peptide_df[sample_cols].apply(lambda x: 2 ** x)
    peptide_df.to_parquet(output_path, compression="zstd", index=False)
    logger.info(f"  Wrote peptide abundances: {output_path} (linear scale)")

    # Write residuals if requested
    residuals_path = None
    if save_residuals and residual_rows:
        residuals_name = output_path.name.replace(".parquet", "_residuals.parquet")
        residuals_path = output_path.parent / residuals_name
        residuals_df = pd.DataFrame(residual_rows)
        residuals_df.to_parquet(residuals_path, compression="zstd", index=False)
        logger.info(f"  Wrote residuals: {residuals_path}")

    return StreamingRollupResult(
        output_path=output_path,
        n_peptides=n_peptides,
        n_samples=len(samples),
        samples=samples,
        method=config.method,
        residuals_path=residuals_path,
    )


def rollup_transitions_sorted(
    parquet_path: Path,
    output_path: Path,
    config: ChunkedRollupConfig,
    samples: list[str] | None = None,
    save_residuals: bool = True,
    sort_buffer_mb: int = 2048,
) -> StreamingRollupResult:
    """Roll up transitions to peptides using sorted streaming.

    This is more efficient than rollup_transitions_streaming for very large
    files because it first sorts the parquet by peptide, then streams through
    in a single pass.

    Steps:
    1. Sort parquet by peptide (writes temp sorted file)
    2. Stream through sorted file, processing each peptide as it completes
    3. Write output incrementally

    Args:
        parquet_path: Path to merged transition-level parquet file
        output_path: Path for output peptide-level parquet file
        config: Rollup configuration
        samples: List of sample names (if None, extracted from data)
        save_residuals: Whether to save residuals to separate file
        sort_buffer_mb: Memory budget for sorting in MB

    Returns:
        StreamingRollupResult with output paths and statistics

    """
    logger.info(f"Starting sorted streaming rollup: {parquet_path}")

    # Get samples
    if samples is None:
        samples = get_unique_values_from_parquet(parquet_path, config.sample_col)
        samples = sorted(samples)
    logger.info(f"  Samples: {len(samples)}")

    # Step 1: Create sorted version
    sorted_path = output_path.parent / f".{output_path.stem}_sorted_temp.parquet"
    logger.info(f"  Sorting by {config.peptide_col}...")

    # Read, sort, write
    # For very large files, this should use an external sort
    # For now, use DuckDB if available, else in-memory
    try:
        import duckdb

        logger.info("  Using DuckDB for efficient sorting...")
        conn = duckdb.connect()
        conn.execute(f"""
            COPY (
                SELECT * FROM read_parquet('{parquet_path}')
                ORDER BY "{config.peptide_col}"
            ) TO '{sorted_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        conn.close()
    except ImportError:
        logger.warning("  DuckDB not available, using PyArrow (may be slower)...")
        table = pq.read_table(parquet_path)
        # Sort using PyArrow
        sorted_table = table.sort_by([(config.peptide_col, "ascending")])
        pq.write_table(sorted_table, sorted_path, compression="zstd")
        del table, sorted_table

    logger.info("  Sorting complete, streaming rollup...")

    # Determine number of workers
    n_workers = config.n_workers if config.n_workers > 0 else mp.cpu_count()
    use_parallel = n_workers > 1
    
    if use_parallel:
        logger.info(f"  Using {n_workers} parallel workers")

    # Step 2: Stream through sorted file
    pf = pq.ParquetFile(sorted_path)
    required_cols, optional_cols = _get_required_columns(config)
    schema_names = set(pf.schema_arrow.names)
    cols_to_read = required_cols.copy()
    for col in optional_cols:
        if col in schema_names:
            cols_to_read.append(col)

    peptide_rows = []
    residual_rows = []
    n_peptides = 0
    n_filtered = 0

    current_peptide = None
    current_data = []
    
    # For parallel processing: collect completed peptides into batches
    pending_peptides: dict[str, pd.DataFrame] = {}
    
    # Config dict for parallel workers
    config_dict = None
    if use_parallel:
        config_dict = {
            "peptide_col": config.peptide_col,
            "transition_col": config.transition_col,
            "precursor_charge_col": config.precursor_charge_col,
            "product_charge_col": config.product_charge_col,
            "sample_col": config.sample_col,
            "abundance_col": config.abundance_col,
            "shape_corr_col": config.shape_corr_col,
            "coeluting_col": config.coeluting_col,
            "rt_col": config.rt_col,
            "batch_col": config.batch_col,
            "mz_col": config.mz_col,
            "method": config.method,
            "min_transitions": config.min_transitions,
            "log_transform": config.log_transform,
            "adaptive_params": config.adaptive_params,
            "exclude_precursor": config.exclude_precursor,
            "topn_count": config.topn_count,
            "topn_selection": config.topn_selection,
            "topn_weighting": config.topn_weighting,
            "n_workers": 1,
            "peptide_batch_size": config.peptide_batch_size,
            "progress_interval": config.progress_interval,
            "max_memory_gb": config.max_memory_gb,
        }

    def process_batch_parallel(batch_dict: dict[str, pd.DataFrame]) -> None:
        """Process a batch of peptides in parallel and collect results."""
        nonlocal n_peptides, n_filtered
        
        # Split batch into chunks for workers
        peptide_items = list(batch_dict.items())
        chunk_size = max(1, len(peptide_items) // n_workers)
        chunks = []
        for i in range(0, len(peptide_items), chunk_size):
            chunk_dict = dict(peptide_items[i:i + chunk_size])
            chunks.append((chunk_dict, samples, config_dict))
        
        with ProcessPoolExecutor(max_workers=min(n_workers, len(chunks))) as executor:
            futures = [executor.submit(_worker_process_batch, chunk) for chunk in chunks]
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    for result in batch_results:
                        if result["n_transitions"] < config.min_transitions:
                            n_filtered += 1
                            continue
                        row = {
                            config.peptide_col: result["peptide"],
                            "n_transitions": result["n_transitions"],
                        }
                        if result["mean_rt"] is not None:
                            row["mean_rt"] = result["mean_rt"]
                        row.update(result["abundances"])
                        peptide_rows.append(row)
                        
                        if save_residuals and result["residuals"]:
                            for transition, sample_residuals in result["residuals"].items():
                                res_row = {
                                    config.peptide_col: result["peptide"],
                                    config.transition_col: transition,
                                }
                                res_row.update(sample_residuals)
                                residual_rows.append(res_row)
                        n_peptides += 1
                except Exception as e:
                    logger.error(f"Worker error: {e}")
                    raise

    def flush_peptide():
        """Process accumulated data for current peptide."""
        nonlocal n_peptides, n_filtered
        if current_data:
            pep_df = pd.concat(current_data, ignore_index=True)
            
            if use_parallel:
                # Add to pending batch for parallel processing
                pending_peptides[current_peptide] = pep_df
            else:
                # Process immediately (single-threaded)
                result = _process_single_peptide(pep_df, current_peptide, samples, config)
                
                if result.n_transitions < config.min_transitions:
                    n_filtered += 1
                    return

                # Build peptide row
                row = {
                    config.peptide_col: result.peptide,
                    "n_transitions": result.n_transitions,
                }
                if result.mean_rt is not None:
                    row["mean_rt"] = result.mean_rt
                row.update(result.abundances)
                peptide_rows.append(row)

                # Build residual rows
                if save_residuals and result.residuals:
                    for transition, sample_residuals in result.residuals.items():
                        res_row = {
                            config.peptide_col: result.peptide,
                            config.transition_col: transition,
                        }
                        res_row.update(sample_residuals)
                        residual_rows.append(res_row)

                n_peptides += 1

    # Stream through row groups
    for i in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(i, columns=cols_to_read)
        df = table.to_pandas()

        for peptide in df[config.peptide_col].unique():
            if peptide != current_peptide:
                # Flush previous peptide
                flush_peptide()
                current_peptide = peptide
                current_data = []

                if n_peptides > 0 and n_peptides % config.progress_interval == 0:
                    logger.info(f"  Processed {n_peptides:,} peptides...")
                
                # In parallel mode, process batch when it reaches target size
                if use_parallel and len(pending_peptides) >= config.peptide_batch_size:
                    process_batch_parallel(pending_peptides)
                    pending_peptides.clear()

            # Accumulate data for this peptide
            current_data.append(df[df[config.peptide_col] == peptide])

    # Flush final peptide
    flush_peptide()
    
    # Process any remaining pending peptides in parallel mode
    if use_parallel and pending_peptides:
        process_batch_parallel(pending_peptides)
        pending_peptides.clear()

    logger.info(f"  Completed: {n_peptides:,} peptides processed")
    if n_filtered > 0:
        logger.info(f"  Filtered: {n_filtered:,} peptides with < {config.min_transitions} transitions")

    # Clean up temp file (use missing_ok=True in case it was already removed)
    sorted_path.unlink(missing_ok=True)

    # Write outputs
    peptide_df = pd.DataFrame(peptide_rows)
    meta_cols = [config.peptide_col, "n_transitions"]
    if "mean_rt" in peptide_df.columns:
        meta_cols.append("mean_rt")
    sample_cols = [s for s in samples if s in peptide_df.columns]

    peptide_df = peptide_df[meta_cols + sample_cols]
    # Convert sample columns from log2 to linear before writing output
    if len(sample_cols) > 0:
        peptide_df[sample_cols] = peptide_df[sample_cols].apply(lambda x: 2 ** x)
    peptide_df.to_parquet(output_path, compression="zstd", index=False)
    logger.info(f"  Wrote peptide abundances: {output_path} (linear scale)")

    residuals_path = None
    if save_residuals and residual_rows:
        residuals_name = output_path.name.replace(".parquet", "_residuals.parquet")
        residuals_path = output_path.parent / residuals_name
        residuals_df = pd.DataFrame(residual_rows)
        residuals_df.to_parquet(residuals_path, compression="zstd", index=False)
        logger.info(f"  Wrote residuals: {residuals_path}")

    return StreamingRollupResult(
        output_path=output_path,
        n_peptides=n_peptides,
        n_samples=len(samples),
        samples=samples,
        method=config.method,
        residuals_path=residuals_path,
    )


# =============================================================================
# Protein Rollup (Peptide → Protein) - Streaming
# =============================================================================


@dataclass
class ProteinRollupConfig:
    """Configuration for protein-level rollup."""

    # Column names in peptide parquet (using Skyline export names)
    peptide_col: str = "Peptide Modified Sequence"
    sample_col: str = "Replicate Name"
    abundance_col: str = "Area"  # Or sample column names for wide format

    # Rollup parameters
    method: str = "median_polish"
    shared_peptide_handling: str = "all_groups"
    min_peptides: int = 3
    topn_n: int = 3
    topn_selection: str = "median_abundance"

    # Processing parameters
    progress_interval: int = 1000


@dataclass
class ProteinRollupResult:
    """Result for a single protein group's rollup."""

    group_id: str
    leading_protein: str
    leading_name: str
    n_peptides: int
    n_unique_peptides: int
    abundances: dict[str, float]  # sample -> abundance (log2)
    residuals: dict[str, dict[str, float]] | None = None  # peptide -> sample -> residual
    low_confidence: bool = False


@dataclass
class StreamingProteinResult:
    """Result of streaming peptide → protein rollup."""

    output_path: Path
    n_proteins: int
    n_samples: int
    samples: list[str]
    method: str
    residuals_path: Path | None = None


def _process_single_protein(
    group: ProteinGroup,
    peptide_matrix: pd.DataFrame,
    samples: list[str],
    config: ProteinRollupConfig,
) -> ProteinRollupResult:
    """Process a single protein group to get protein-level abundance.

    Args:
        group: ProteinGroup from parsimony
        peptide_matrix: DataFrame with peptide abundances (peptide × sample)
        samples: List of sample names
        config: Rollup configuration

    Returns:
        ProteinRollupResult with abundances

    """
    group_id = group.group_id

    # Select peptides based on shared_peptide_handling
    if config.shared_peptide_handling == "unique_only":
        peptides = group.unique_peptides
    elif config.shared_peptide_handling == "razor":
        peptides = group.peptides
    else:  # "all_groups"
        peptides = group.all_mapped_peptides

    # Filter to available peptides
    available = [p for p in peptides if p in peptide_matrix.index]
    n_peptides = len(available)

    if n_peptides == 0:
        return ProteinRollupResult(
            group_id=group_id,
            leading_protein=group.leading_protein,
            leading_name=group.leading_protein_name,
            n_peptides=0,
            n_unique_peptides=len(group.unique_peptides),
            abundances={s: np.nan for s in samples},
            low_confidence=True,
        )

    matrix = peptide_matrix.loc[available, samples]

    # Determine confidence
    low_confidence = n_peptides < config.min_peptides

    residuals = None

    # Helper function to compute sum in linear space
    def sum_linear(mat: pd.DataFrame) -> dict:
        """Sum peptides in linear space, return log2 result."""
        linear = 2**mat
        summed = linear.sum(axis=0)
        return np.log2(summed.clip(lower=1)).to_dict()

    if n_peptides == 1:
        # Single peptide - use directly (same for all methods)
        abundances = matrix.iloc[0].to_dict()
    elif n_peptides == 2:
        # Two peptides - method-dependent handling
        if config.method in ("sum", "topn"):
            # Sum methods: sum in linear space
            abundances = sum_linear(matrix)
        else:
            # median_polish with <3 peptides: fall back to mean (geometric mean in log space)
            abundances = matrix.mean(axis=0).to_dict()
    elif config.method == "median_polish":
        # Median polish for robust rollup (requires 3+ peptides)
        result = tukey_median_polish(matrix)
        abundances = result.col_effects.to_dict()
        # Store residuals
        residuals = {
            str(p): result.residuals.loc[p].to_dict() for p in result.residuals.index
        }
    elif config.method == "topn":
        # Top N by median abundance, then sum in linear space
        median_per_peptide = matrix.median(axis=1)
        top_peptides = median_per_peptide.nlargest(config.topn_n).index.tolist()
        abundances = sum_linear(matrix.loc[top_peptides])
    elif config.method == "sum":
        # Sum all peptides in linear space
        abundances = sum_linear(matrix)
    else:
        # Fallback to mean
        abundances = matrix.mean(axis=0).to_dict()

    return ProteinRollupResult(
        group_id=group_id,
        leading_protein=group.leading_protein,
        leading_name=group.leading_protein_name,
        n_peptides=n_peptides,
        n_unique_peptides=len(group.unique_peptides),
        abundances=abundances,
        residuals=residuals,
        low_confidence=low_confidence,
    )


def rollup_proteins_streaming(
    peptide_parquet_path: Path,
    protein_groups: list[ProteinGroup],
    output_path: Path,
    config: ProteinRollupConfig,
    samples: list[str] | None = None,
    save_residuals: bool = True,
    is_wide_format: bool = True,
) -> StreamingProteinResult:
    """Roll up peptides to proteins using streaming processing.

    This is memory-efficient because:
    - Reads peptide parquet in chunks (row groups)
    - For each protein group, loads only its peptides
    - Writes output incrementally

    Args:
        peptide_parquet_path: Path to peptide-level parquet file
        protein_groups: List of ProteinGroup objects from parsimony
        output_path: Path for output protein-level parquet file
        config: Rollup configuration
        samples: List of sample names (if None, inferred from parquet)
        save_residuals: Whether to save peptide residuals
        is_wide_format: Whether peptide parquet is wide format (samples as columns)

    Returns:
        StreamingProteinResult with output paths and statistics

    """
    logger.info(f"Starting streaming protein rollup: {peptide_parquet_path}")
    logger.info(f"  Method: {config.method}")
    logger.info(f"  Protein groups: {len(protein_groups)}")
    logger.info(f"  Shared peptide handling: {config.shared_peptide_handling}")

    # Read peptide parquet - for protein rollup we CAN load the full peptide matrix
    # because it's much smaller than transitions (83K rows vs 197M rows)
    # But we'll do it efficiently
    pf = pq.ParquetFile(peptide_parquet_path)

    if is_wide_format:
        # Wide format: peptide_col as a column, samples as other columns
        table = pf.read()
        peptide_df = table.to_pandas()

        # Identify sample columns vs metadata columns
        if samples is None:
            # Assume non-metadata columns are samples
            meta_cols = {config.peptide_col, "n_transitions", "mean_rt"}
            samples = [c for c in peptide_df.columns if c not in meta_cols]
        samples = list(samples)

        # Set peptide as index
        peptide_matrix = peptide_df.set_index(config.peptide_col)[samples]

    else:
        # Long format - need to pivot
        table = pf.read()
        peptide_df = table.to_pandas()

        if samples is None:
            samples = sorted(peptide_df[config.sample_col].unique())

        peptide_matrix = peptide_df.pivot_table(
            index=config.peptide_col,
            columns=config.sample_col,
            values=config.abundance_col,
            aggfunc="first",
        )
        peptide_matrix = peptide_matrix.reindex(columns=samples)

    logger.info(f"  Loaded peptide matrix: {peptide_matrix.shape}")
    logger.info(f"  Samples: {len(samples)}")

    # Process each protein group
    protein_rows = []
    residual_rows = []
    n_proteins = 0
    n_skipped = 0

    for i, group in enumerate(protein_groups):
        result = _process_single_protein(group, peptide_matrix, samples, config)

        if result.n_peptides == 0:
            n_skipped += 1
            continue

        # Build protein row
        row = {
            "protein_group": result.group_id,
            "leading_protein": result.leading_protein,
            "leading_name": result.leading_name,
            "n_peptides": result.n_peptides,
            "n_unique_peptides": result.n_unique_peptides,
            "low_confidence": result.low_confidence,
        }
        row.update(result.abundances)
        protein_rows.append(row)

        # Build residual rows
        if save_residuals and result.residuals:
            for peptide, sample_residuals in result.residuals.items():
                res_row = {
                    "protein_group": result.group_id,
                    config.peptide_col: peptide,
                }
                res_row.update(sample_residuals)
                residual_rows.append(res_row)

        n_proteins += 1

        if (i + 1) % config.progress_interval == 0:
            logger.info(f"  Processed {i + 1:,} / {len(protein_groups):,} proteins...")

    logger.info(f"  Completed: {n_proteins:,} proteins, {n_skipped} skipped (0 peptides)")

    # Write protein output
    protein_df = pd.DataFrame(protein_rows)
    meta_cols = [
        "protein_group",
        "leading_protein",
        "leading_name",
        "n_peptides",
        "n_unique_peptides",
        "low_confidence",
    ]
    sample_cols = [s for s in samples if s in protein_df.columns]

    protein_df = protein_df[meta_cols + sample_cols]
    # Convert sample columns from log2 to linear before writing output
    if len(sample_cols) > 0:
        protein_df[sample_cols] = protein_df[sample_cols].apply(lambda x: 2 ** x)
    protein_df.to_parquet(output_path, compression="zstd", index=False)
    logger.info(f"  Wrote protein abundances: {output_path} (linear scale)")

    # Write residuals
    residuals_path = None
    if save_residuals and residual_rows:
        residuals_name = output_path.name.replace(".parquet", "_residuals.parquet")
        residuals_path = output_path.parent / residuals_name
        residuals_df = pd.DataFrame(residual_rows)
        residuals_df.to_parquet(residuals_path, compression="zstd", index=False)
        logger.info(f"  Wrote residuals: {residuals_path}")

    return StreamingProteinResult(
        output_path=output_path,
        n_proteins=n_proteins,
        n_samples=len(samples),
        samples=samples,
        method=config.method,
        residuals_path=residuals_path,
    )


# =============================================================================
# Full Pipeline - Streaming
# =============================================================================


@dataclass
class StreamingPipelineConfig:
    """Configuration for full streaming pipeline."""

    # Transition → Peptide config
    transition_config: ChunkedRollupConfig

    # Protein rollup config
    protein_config: ProteinRollupConfig

    # Column mappings
    protein_col: str = "protein_ids"
    protein_name_col: str = "protein_names"
    peptide_sequence_col: str = "peptide_sequence"

    # FASTA for parsimony (optional)
    fasta_path: Path | None = None


def is_large_file(path: Path) -> bool:
    """Check if a file is large enough to require streaming processing."""
    return path.stat().st_size > LARGE_FILE_THRESHOLD_BYTES


def run_streaming_pipeline(
    transition_parquet: Path,
    output_dir: Path,
    config: StreamingPipelineConfig,
    save_residuals: bool = True,
) -> tuple[Path, Path]:
    """Run the full PRISM pipeline using streaming processing.

    This is the memory-efficient version for large datasets.

    Args:
        transition_parquet: Path to merged transition-level parquet
        output_dir: Output directory
        config: Pipeline configuration
        save_residuals: Whether to save residual files

    Returns:
        Tuple of (peptide_output_path, protein_output_path)

    """
    from .parsimony import build_peptide_protein_map, compute_protein_groups

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Transition → Peptide rollup (streaming)
    logger.info("=" * 60)
    logger.info("Stage 1: Transition -> Peptide rollup (streaming)")
    logger.info("=" * 60)

    peptide_path = output_dir / "peptides.parquet"
    peptide_result = rollup_transitions_sorted(
        parquet_path=transition_parquet,
        output_path=peptide_path,
        config=config.transition_config,
        save_residuals=save_residuals,
    )
    samples = peptide_result.samples

    # Step 2: Build peptide-protein map and run parsimony
    # We need to get peptide→protein mapping from the transition data
    logger.info("=" * 60)
    logger.info("Stage 2: Protein parsimony")
    logger.info("=" * 60)

    # Read just the columns needed for parsimony
    pf = pq.ParquetFile(transition_parquet)
    pep_col = config.transition_config.peptide_col
    prot_col = config.protein_col
    prot_name_col = config.protein_name_col

    # Get unique peptide-protein mappings
    logger.info("  Reading peptide-protein mappings...")
    columns_for_parsimony = [pep_col, prot_col, prot_name_col]
    available_cols = set(pf.schema_arrow.names)
    columns_for_parsimony = [c for c in columns_for_parsimony if c in available_cols]

    mapping_table = pf.read(columns=columns_for_parsimony)
    mapping_df = mapping_table.to_pandas().drop_duplicates()
    logger.info(f"  Found {len(mapping_df):,} unique peptide-protein records")

    # Build maps
    pep_to_prot, prot_to_pep, prot_to_name = build_peptide_protein_map(
        mapping_df,
        peptide_col=pep_col,
        protein_col=prot_col,
        protein_name_col=prot_name_col,
    )

    # Run parsimony
    protein_groups = compute_protein_groups(prot_to_pep, pep_to_prot, prot_to_name)
    logger.info(f"  Computed {len(protein_groups)} protein groups")

    # Save protein groups
    groups_df = pd.DataFrame([g.to_dict() for g in protein_groups])
    groups_path = output_dir / "protein_groups.tsv"
    groups_df.to_csv(groups_path, sep="\t", index=False)
    logger.info(f"  Saved protein groups: {groups_path}")

    # Step 3: Peptide → Protein rollup (streaming)
    logger.info("=" * 60)
    logger.info("Stage 3: Peptide -> Protein rollup (streaming)")
    logger.info("=" * 60)

    protein_path = output_dir / "proteins.parquet"
    protein_result = rollup_proteins_streaming(
        peptide_parquet_path=peptide_path,
        protein_groups=protein_groups,
        output_path=protein_path,
        config=config.protein_config,
        samples=samples,
        save_residuals=save_residuals,
    )

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"  Peptides: {peptide_result.n_peptides:,}")
    logger.info(f"  Proteins: {protein_result.n_proteins:,}")
    logger.info(f"  Samples: {len(samples)}")
    logger.info("=" * 60)

    return peptide_path, protein_path
