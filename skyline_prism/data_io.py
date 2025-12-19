"""Data I/O module for loading and merging Skyline reports."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Standard column name mapping from Skyline exports
SKYLINE_COLUMN_MAP = {
    # Standard Skyline report columns
    'Protein Name': 'protein_names',
    'Protein Accession': 'protein_ids',
    'Peptide Sequence': 'peptide_sequence',
    'Peptide Modified Sequence': 'peptide_modified',
    'Precursor Charge': 'precursor_charge',
    'Precursor Mz': 'precursor_mz',
    'Best Retention Time': 'retention_time',
    'Total Area Fragment': 'abundance_fragment',
    'Total Area MS1': 'abundance_ms1',
    'Replicate Name': 'replicate_name',
    'Isotope Dot Product': 'idotp',
    'Average Mass Error PPM': 'mass_error_ppm',
    'Library Dot Product': 'library_dotp',
    'Detection Q Value': 'detection_qvalue',
    'Truncated': 'truncated',
    'Is Decoy': 'is_decoy',

    # Transition-level columns
    'Fragment Ion': 'fragment_ion',
    'Fragment Ion Type': 'fragment_ion_type',
    'Fragment Ion Ordinal': 'fragment_ion_ordinal',
    'Product Charge': 'product_charge',
    'Product Mz': 'product_mz',
    'Area': 'area',

    # Transition quality metrics (Peptides > Precursors > Transition Results)
    'Shape Correlation': 'shape_correlation',  # Correlation with median transition
    'Coeluting': 'coeluting',                  # Apex within integration boundaries
    'Fwhm': 'fwhm',                            # Full width at half max
    'Start Time': 'start_time',
    'End Time': 'end_time',

    # Result file metadata (Result File)
    'Acquired Time': 'acquired_time',          # Acquisition timestamp for batch estimation

    # EncyclopeDIA via Skyline
    'Normalized Area': 'abundance_fragment',

    # Alternative naming conventions
    'ProteinName': 'protein_names',
    'ProteinAccession': 'protein_ids',
    'ModifiedSequence': 'peptide_modified',
    'PrecursorCharge': 'precursor_charge',
    'RetentionTime': 'retention_time',
    'FragmentArea': 'abundance_fragment',
    'Ms1Area': 'abundance_ms1',
    'ReplicateName': 'replicate_name',
    'FragmentIon': 'fragment_ion',
    'ShapeCorrelation': 'shape_correlation',
}

# Required columns for processing
REQUIRED_COLUMNS = [
    'protein_ids',
    'peptide_modified',
    'precursor_charge',
    'retention_time',
    'replicate_name',
]

# At least one of these abundance columns required
ABUNDANCE_COLUMNS = ['abundance_fragment', 'abundance_ms1']

# Sample metadata required columns (Batch is optional - can be estimated)
METADATA_REQUIRED = ['ReplicateName', 'SampleType', 'RunOrder']
METADATA_BATCH_COLUMNS = ['Batch', 'Batch Name']  # Skyline uses 'Batch Name'
VALID_SAMPLE_TYPES = {'experimental', 'pool', 'reference'}


@dataclass
class ValidationResult:
    """Result of validating a Skyline report."""

    is_valid: bool
    filepath: Path
    missing_required: list[str] = field(default_factory=list)
    missing_abundance: bool = False
    extra_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_rows: int = 0
    n_replicates: int = 0

    def __str__(self) -> str:
        if self.is_valid:
            return f"Valid: {self.filepath.name} ({self.n_rows} rows, {self.n_replicates} replicates)"
        else:
            issues = []
            if self.missing_required:
                issues.append(f"Missing columns: {self.missing_required}")
            if self.missing_abundance:
                issues.append("No abundance column found")
            return f"Invalid: {self.filepath.name} - {'; '.join(issues)}"


@dataclass
class MergeResult:
    """Result of merging multiple Skyline reports."""

    output_path: Path
    n_reports: int
    n_replicates: int
    n_precursors: int
    n_rows: int
    warnings: list[str] = field(default_factory=list)
    replicate_sources: dict[str, str] = field(default_factory=dict)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to standard names using the mapping."""
    rename_map = {}
    for orig, standard in SKYLINE_COLUMN_MAP.items():
        if orig in df.columns:
            rename_map[orig] = standard

    return df.rename(columns=rename_map)


def validate_skyline_report(filepath: Path) -> ValidationResult:
    """Validate that a Skyline report has required columns.

    Args:
        filepath: Path to the Skyline report (CSV or TSV)

    Returns:
        ValidationResult with validation details

    """
    filepath = Path(filepath)
    result = ValidationResult(is_valid=True, filepath=filepath)

    # Detect delimiter
    suffix = filepath.suffix.lower()
    sep = '\t' if suffix in ['.tsv', '.txt'] else ','

    try:
        # Read just the header first
        df_head = pd.read_csv(filepath, sep=sep, nrows=5)
        df_head = _standardize_columns(df_head)

        # Check required columns
        for col in REQUIRED_COLUMNS:
            if col not in df_head.columns:
                result.missing_required.append(col)
                result.is_valid = False

        # Check for at least one abundance column
        has_abundance = any(col in df_head.columns for col in ABUNDANCE_COLUMNS)
        if not has_abundance:
            result.missing_abundance = True
            result.is_valid = False

        # Warnings for missing optional columns
        if 'idotp' not in df_head.columns:
            result.warnings.append("No isotope dot product column - quality filtering limited")
        if 'detection_qvalue' not in df_head.columns:
            result.warnings.append("No detection Q-value column")

        # If valid, get some stats
        if result.is_valid:
            df_full = pd.read_csv(filepath, sep=sep)
            df_full = _standardize_columns(df_full)
            result.n_rows = len(df_full)
            result.n_replicates = df_full['replicate_name'].nunique()

    except Exception as e:
        result.is_valid = False
        result.warnings.append(f"Error reading file: {str(e)}")

    return result


def load_skyline_report(
    filepath: Path,
    source_name: Optional[str] = None,
    validate: bool = True
) -> pd.DataFrame:
    """Load a single Skyline report with standardized column names.

    Args:
        filepath: Path to CSV/TSV report
        source_name: Identifier for this document (defaults to filename stem)
        validate: Whether to validate before loading

    Returns:
        DataFrame with standardized column names

    Raises:
        ValueError: If validation fails and validate=True

    """
    filepath = Path(filepath)

    if validate:
        validation = validate_skyline_report(filepath)
        if not validation.is_valid:
            raise ValueError(f"Invalid Skyline report: {validation}")

    # Detect delimiter
    suffix = filepath.suffix.lower()
    sep = '\t' if suffix in ['.tsv', '.txt'] else ','

    # Load data
    df = pd.read_csv(filepath, sep=sep)
    df = _standardize_columns(df)

    # Add source tracking
    if source_name is None:
        source_name = filepath.stem
    df['source_document'] = source_name

    # Create precursor_id if not present
    if 'precursor_id' not in df.columns:
        df['precursor_id'] = df['peptide_modified'] + '_' + df['precursor_charge'].astype(str)

    # Determine primary abundance column
    if 'abundance_fragment' in df.columns and df['abundance_fragment'].notna().any():
        df['abundance'] = df['abundance_fragment']
        df['abundance_type'] = 'fragment'
    elif 'abundance_ms1' in df.columns and df['abundance_ms1'].notna().any():
        df['abundance'] = df['abundance_ms1']
        df['abundance_type'] = 'ms1'
    else:
        raise ValueError("No valid abundance data found")

    return df


def load_sample_metadata(filepath: Path) -> pd.DataFrame:
    """Load and validate sample metadata file.

    Args:
        filepath: Path to metadata TSV/CSV

    Returns:
        Validated metadata DataFrame

    Raises:
        ValueError: If validation fails

    """
    filepath = Path(filepath)

    # Detect delimiter
    suffix = filepath.suffix.lower()
    sep = '\t' if suffix in ['.tsv', '.txt'] else ','

    meta = pd.read_csv(filepath, sep=sep)

    # Normalize batch column name (Skyline uses 'Batch Name')
    for batch_col in METADATA_BATCH_COLUMNS:
        if batch_col in meta.columns and batch_col != 'Batch':
            meta = meta.rename(columns={batch_col: 'Batch'})
            logger.info(f"Renamed '{batch_col}' column to 'Batch'")
            break

    # Check required columns
    missing = [col for col in METADATA_REQUIRED if col not in meta.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")

    # Note if Batch column is missing (will be estimated later)
    if 'Batch' not in meta.columns:
        logger.info("No Batch column in metadata - batches will be estimated")

    # Validate SampleType values
    invalid_types = set(meta['SampleType'].unique()) - VALID_SAMPLE_TYPES
    if invalid_types:
        raise ValueError(
            f"Invalid SampleType values: {invalid_types}. "
            f"Must be one of: {VALID_SAMPLE_TYPES}"
        )

    # Check for duplicate replicate names
    duplicates = meta[meta['ReplicateName'].duplicated()]['ReplicateName'].tolist()
    if duplicates:
        raise ValueError(f"Duplicate ReplicateName entries: {duplicates}")

    # Ensure RunOrder is numeric
    meta['RunOrder'] = pd.to_numeric(meta['RunOrder'], errors='coerce')
    if meta['RunOrder'].isna().any():
        raise ValueError("RunOrder must be numeric")

    return meta


def merge_skyline_reports(
    report_paths: list[Path],
    output_path: Path,
    sample_metadata: Optional[pd.DataFrame] = None,
    partition_by_batch: bool = True,
) -> MergeResult:
    """Merge multiple Skyline reports into unified parquet.

    Args:
        report_paths: List of paths to Skyline reports
        output_path: Path for output parquet file/directory
        sample_metadata: Optional metadata DataFrame (or will look for metadata.tsv)
        partition_by_batch: Whether to partition parquet by batch

    Returns:
        MergeResult with merge statistics

    """
    output_path = Path(output_path)
    result = MergeResult(
        output_path=output_path,
        n_reports=len(report_paths),
        n_replicates=0,
        n_precursors=0,
        n_rows=0,
    )

    # Validate all reports first
    logger.info(f"Validating {len(report_paths)} reports...")
    validations = [validate_skyline_report(p) for p in report_paths]
    invalid = [v for v in validations if not v.is_valid]
    if invalid:
        raise ValueError(f"Invalid reports: {[str(v) for v in invalid]}")

    # Load and concatenate
    logger.info("Loading and merging reports...")
    dfs = []
    for path in report_paths:
        df = load_skyline_report(path, validate=False)
        dfs.append(df)

        # Track which replicates came from which file
        for rep in df['replicate_name'].unique():
            if rep in result.replicate_sources:
                result.warnings.append(
                    f"Replicate '{rep}' appears in multiple files: "
                    f"{result.replicate_sources[rep]} and {path.name}"
                )
            result.replicate_sources[rep] = path.name

    merged = pd.concat(dfs, ignore_index=True)

    # Join sample metadata
    if sample_metadata is not None:
        # Standardize column name for join
        meta = sample_metadata.rename(columns={'ReplicateName': 'replicate_name'})

        # Check for unmatched replicates
        data_reps = set(merged['replicate_name'].unique())
        meta_reps = set(meta['replicate_name'].unique())

        unmatched_data = data_reps - meta_reps
        if unmatched_data:
            result.warnings.append(
                f"Replicates in data but not metadata: {unmatched_data}"
            )

        unmatched_meta = meta_reps - data_reps
        if unmatched_meta:
            result.warnings.append(
                f"Replicates in metadata but not data: {unmatched_meta}"
            )

        # Merge
        merged = merged.merge(
            meta[['replicate_name', 'SampleType', 'Batch', 'RunOrder']],
            on='replicate_name',
            how='left'
        )
        merged = merged.rename(columns={
            'SampleType': 'sample_type',
            'Batch': 'batch',
            'RunOrder': 'run_order'
        })

    # Compute stats
    result.n_rows = len(merged)
    result.n_replicates = merged['replicate_name'].nunique()
    result.n_precursors = merged['precursor_id'].nunique()

    # Write parquet
    logger.info(f"Writing parquet to {output_path}...")

    if partition_by_batch and 'batch' in merged.columns:
        # Partitioned write
        table = pa.Table.from_pandas(merged)
        pq.write_to_dataset(
            table,
            root_path=str(output_path),
            partition_cols=['batch']
        )
    else:
        # Single file
        merged.to_parquet(output_path, index=False)

    logger.info(f"Merge complete: {result.n_rows} rows, {result.n_replicates} replicates")

    return result


def load_unified_data(path: Path) -> pd.DataFrame:
    """Load unified parquet data (handles both single file and partitioned).

    Args:
        path: Path to parquet file or directory

    Returns:
        DataFrame with all data

    """
    path = Path(path)

    if path.is_dir():
        # Partitioned dataset
        return pd.read_parquet(path)
    else:
        # Single file
        return pd.read_parquet(path)


# Convenience function to identify internal QC peptides
def identify_internal_qcs(df: pd.DataFrame) -> tuple[set[str], set[str]]:
    """Identify PRTC and enolase peptides in the data.

    Returns:
        Tuple of (prtc_precursor_ids, eno_precursor_ids)

    """
    # PRTC identification - look for protein name/id patterns
    prtc_mask = (
        df['protein_names'].str.contains('PRTC', case=False, na=False) |
        df['protein_ids'].str.contains('PRTC', case=False, na=False)
    )
    prtc_ids = set(df.loc[prtc_mask, 'precursor_id'].unique())

    # Enolase identification - yeast enolase 1
    eno_patterns = ['ENO1_YEAST', 'P00924', 'enolase']
    eno_mask = pd.Series(False, index=df.index)
    for pattern in eno_patterns:
        eno_mask |= df['protein_names'].str.contains(pattern, case=False, na=False)
        eno_mask |= df['protein_ids'].str.contains(pattern, case=False, na=False)
    eno_ids = set(df.loc[eno_mask, 'precursor_id'].unique())

    return prtc_ids, eno_ids


@dataclass
class BatchEstimationResult:
    """Result of automatic batch estimation."""

    batch_column: pd.Series
    method: str  # 'metadata', 'source_document', 'acquisition_gap', 'equal_division'
    n_batches: int
    warnings: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"BatchEstimation: {self.n_batches} batches via '{self.method}'"
        )


def estimate_batches(
    df: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
    min_samples_per_batch: int = 12,
    max_samples_per_batch: int = 100,
    gap_iqr_multiplier: float = 1.5,
    n_batches_fallback: Optional[int] = None,
) -> BatchEstimationResult:
    """Estimate batch assignments when not provided in metadata.

    Uses a priority-based approach:
    1. Metadata file with Batch column (if provided)
    2. Source document (different Skyline CSV/TSV files = different batches)
    3. Acquisition time gaps (outlier gaps indicate batch boundaries)
    4. Equal division into n_batches_fallback batches based on acquisition order

    For acquisition time gap detection, we use IQR-based outlier detection:
    A gap is considered a batch break if it exceeds Q3 + (iqr_multiplier * IQR).

    Example: If runs are typically 65 min apart (±2 min), the IQR would be ~4 min.
    With the default multiplier of 1.5, any gap > ~71 min would indicate a batch break.
    A 90 min gap (e.g., overnight break) would clearly be detected.

    Args:
        df: DataFrame with replicate data (must have 'replicate_name' column)
        metadata: Optional metadata DataFrame with 'Batch' or 'Batch Name' column
        min_samples_per_batch: Minimum expected samples per batch (default 12)
        max_samples_per_batch: Maximum expected samples per batch (default 100)
        gap_iqr_multiplier: IQR multiplier for outlier gap detection (default 1.5)
        n_batches_fallback: Number of batches for equal division fallback

    Returns:
        BatchEstimationResult with batch assignments and method used

    """
    # Get unique replicates with their properties
    replicate_info = df.groupby('replicate_name').first().reset_index()
    n_replicates = len(replicate_info)

    result = BatchEstimationResult(
        batch_column=pd.Series(dtype=str),
        method='unknown',
        n_batches=0,
    )

    # Priority 1: Metadata file with Batch column (supports 'Batch' or 'Batch Name')
    if metadata is not None:
        batch_col = None
        for col in ['Batch', 'Batch Name']:
            if col in metadata.columns:
                batch_col = col
                break

        if batch_col is not None:
            meta_batches = metadata.set_index('ReplicateName')[batch_col]
            if replicate_info['replicate_name'].isin(meta_batches.index).all():
                result.batch_column = replicate_info['replicate_name'].map(meta_batches)
                result.method = 'metadata'
                result.n_batches = result.batch_column.nunique()
                result.details['source'] = 'User-provided metadata file'
                result.details['column_name'] = batch_col
                logger.info(f"Batch assignment from metadata: {result.n_batches} batches")
                return result
            else:
                missing = set(replicate_info['replicate_name']) - set(meta_batches.index)
                result.warnings.append(
                    f"Metadata missing batch for replicates: {missing}"
                )

    # Priority 2: Source document (different Skyline files = different batches)
    if 'source_document' in replicate_info.columns:
        source_docs = replicate_info['source_document'].unique()
        if len(source_docs) > 1:
            result.batch_column = replicate_info['source_document'].copy()
            result.method = 'source_document'
            result.n_batches = len(source_docs)
            result.details['source_documents'] = list(source_docs)
            logger.info(
                f"Batch assignment from source documents: {result.n_batches} batches"
            )
            return result

    # Priority 3: Acquisition time gaps (IQR-based outlier detection)
    if 'acquired_time' in replicate_info.columns:
        acq_times = pd.to_datetime(replicate_info['acquired_time'], errors='coerce')
        if acq_times.notna().sum() > 1:
            # Sort by acquisition time
            sorted_idx = acq_times.sort_values().index
            sorted_times = acq_times.loc[sorted_idx]
            sorted_replicates = replicate_info.loc[sorted_idx, 'replicate_name']

            # Calculate gaps between consecutive acquisitions (in minutes)
            gaps = sorted_times.diff()
            gaps_minutes = gaps.dt.total_seconds() / 60

            # Use IQR-based outlier detection
            # Normal LC-MS runs have consistent inter-run times (e.g., 65±2 min)
            # Batch breaks show as outlier gaps (e.g., overnight = 90+ min)
            valid_gaps = gaps_minutes.dropna()

            if len(valid_gaps) >= 3:  # Need enough gaps for IQR calculation
                q1 = valid_gaps.quantile(0.25)
                q3 = valid_gaps.quantile(0.75)
                iqr = q3 - q1

                # Threshold: Q3 + multiplier * IQR (classic outlier detection)
                # With multiplier=1.5, this detects mild outliers
                threshold_minutes = q3 + gap_iqr_multiplier * iqr

                # Ensure threshold is at least 10% above median to avoid
                # false positives when gaps are very consistent (low IQR)
                median_gap = valid_gaps.median()
                min_threshold = median_gap * 1.1
                threshold_minutes = max(threshold_minutes, min_threshold)

                large_gaps = gaps_minutes > threshold_minutes

                if large_gaps.any():
                    # Assign batch numbers based on large gaps
                    batch_numbers = large_gaps.cumsum()
                    batch_numbers = batch_numbers.fillna(0).astype(int)

                    # Create batch labels
                    batch_labels = 'batch_' + (batch_numbers + 1).astype(str)

                    # Map back to replicate names
                    batch_map = dict(zip(sorted_replicates, batch_labels))
                    result.batch_column = replicate_info['replicate_name'].map(
                        batch_map
                    )
                    result.method = 'acquisition_gap'
                    result.n_batches = result.batch_column.nunique()
                    result.details['median_gap_minutes'] = float(median_gap)
                    result.details['q1_minutes'] = float(q1)
                    result.details['q3_minutes'] = float(q3)
                    result.details['iqr_minutes'] = float(iqr)
                    result.details['threshold_minutes'] = float(threshold_minutes)
                    result.details['gap_locations'] = list(
                        sorted_replicates[large_gaps].values
                    )
                    # Include the actual gap sizes at break points
                    result.details['gap_sizes_at_breaks'] = [
                        float(gaps_minutes.loc[idx])
                        for idx in large_gaps[large_gaps].index
                    ]

                    # Validate batch sizes
                    batch_sizes = result.batch_column.value_counts()
                    if (batch_sizes < min_samples_per_batch).any():
                        result.warnings.append(
                            f"Some batches have fewer than {min_samples_per_batch} "
                            f"samples: {batch_sizes[batch_sizes < min_samples_per_batch].to_dict()}"
                        )
                    if (batch_sizes > max_samples_per_batch).any():
                        result.warnings.append(
                            f"Some batches have more than {max_samples_per_batch} "
                            f"samples: {batch_sizes[batch_sizes > max_samples_per_batch].to_dict()}"
                        )

                    logger.info(
                        f"Batch assignment from acquisition gaps: "
                        f"{result.n_batches} batches"
                    )
                    return result

    # Priority 4: Equal division by acquisition time or replicate order
    if n_batches_fallback is not None and n_batches_fallback > 0:
        n_batches = n_batches_fallback
    else:
        # Estimate reasonable number of batches
        n_batches = max(1, n_replicates // ((min_samples_per_batch + max_samples_per_batch) // 2))
        n_batches = min(n_batches, n_replicates // min_samples_per_batch) if n_replicates >= min_samples_per_batch else 1

    if n_batches <= 1:
        # Single batch - all samples together (batch correction will be skipped)
        result.batch_column = pd.Series(
            ['batch_1'] * n_replicates,
            index=replicate_info.index
        )
        result.method = 'single_batch'
        result.n_batches = 1
        result.details['reason'] = (
            'No batch boundaries detected - single Skyline document, '
            'no acquisition time gaps, and no forced batch division'
        )
        logger.info(
            "Single batch detected - batch correction will be skipped"
        )
        return result

    # Sort by acquisition time if available, otherwise by replicate name
    if 'acquired_time' in replicate_info.columns:
        acq_times = pd.to_datetime(replicate_info['acquired_time'], errors='coerce')
        if acq_times.notna().sum() > 0:
            sort_col = acq_times
        else:
            sort_col = replicate_info['replicate_name']
    else:
        sort_col = replicate_info['replicate_name']

    sorted_idx = sort_col.sort_values().index
    sorted_replicates = replicate_info.loc[sorted_idx, 'replicate_name'].reset_index(
        drop=True
    )

    # Divide into equal batches
    batch_assignments = pd.cut(
        range(len(sorted_replicates)),
        bins=n_batches,
        labels=[f'batch_{i+1}' for i in range(n_batches)]
    )
    batch_map = dict(zip(sorted_replicates, batch_assignments))
    result.batch_column = replicate_info['replicate_name'].map(batch_map)
    result.method = 'equal_division'
    result.n_batches = n_batches
    result.details['samples_per_batch'] = n_replicates // n_batches
    result.warnings.append(
        f"No batch information available - divided {n_replicates} samples "
        f"into {n_batches} equal batches by acquisition order"
    )

    logger.info(
        f"Batch assignment by equal division: {result.n_batches} batches"
    )
    return result


def apply_batch_estimation(
    df: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
    **kwargs
) -> tuple[pd.DataFrame, BatchEstimationResult]:
    """Apply batch estimation to a DataFrame if batch column is missing.

    Args:
        df: DataFrame with data (must have 'replicate_name' column)
        metadata: Optional metadata DataFrame
        **kwargs: Additional arguments passed to estimate_batches()

    Returns:
        Tuple of (DataFrame with 'batch' column, BatchEstimationResult)

    """
    # Check if batch already assigned
    if 'batch' in df.columns and df['batch'].notna().all():
        # Already has batch - create a result reflecting this
        result = BatchEstimationResult(
            batch_column=df.groupby('replicate_name')['batch'].first(),
            method='existing',
            n_batches=df['batch'].nunique(),
        )
        return df, result

    # Estimate batches
    result = estimate_batches(df, metadata=metadata, **kwargs)

    # Apply to DataFrame
    batch_map = dict(zip(
        df.groupby('replicate_name').first().reset_index()['replicate_name'],
        result.batch_column
    ))
    df = df.copy()
    df['batch'] = df['replicate_name'].map(batch_map)

    return df, result
