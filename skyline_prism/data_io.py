"""Data I/O module for loading and merging Skyline reports."""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Size threshold for using streaming reader (1 GB)
LARGE_FILE_THRESHOLD_BYTES = 1 * 1024 * 1024 * 1024

# Standard Skyline column names (as exported from Skyline reports)
# These are the exact names as they appear in Skyline report exports.
# Users can configure different column names in their config file if needed.
#
# Key columns and their Skyline sources:
# - 'Protein' or 'Protein Name': Proteins > Protein
# - 'Protein Accession': Proteins > Protein Accession  
# - 'Peptide Modified Sequence': Peptides > Peptide Modified Sequence
# - 'Precursor Charge': Precursors > Precursor Charge
# - 'Replicate Name': Replicates > Replicate Name
# - 'Area': Transition Results > Area
# - 'Retention Time': Transition Results > Retention Time
# - 'Fragment Ion': Transitions > Fragment Ion

# Standard column names expected from Skyline exports
# These match the exact column headers from Skyline report exports
SKYLINE_STANDARD_COLUMNS = {
    # Core columns for PRISM processing
    'protein_name': 'Protein',  # Protein display name
    'protein_accession': 'Protein Accession',  # Protein accession (UniProt, etc.)
    'protein_gene': 'Protein Gene',  # Gene name
    'peptide_sequence': 'Peptide',  # Unmodified sequence
    'peptide_modified': 'Peptide Modified Sequence',  # With modifications
    'precursor_charge': 'Precursor Charge',
    'precursor_mz': 'Precursor Mz',
    'fragment_ion': 'Fragment Ion',  # y7, b5, precursor, etc.
    'product_charge': 'Product Charge',
    'product_mz': 'Product Mz',
    'area': 'Area',  # Transition peak area
    'retention_time': 'Retention Time',
    'replicate_name': 'Replicate Name',
    'batch_name': 'Batch Name',  # Skyline's batch column
    # Quality metrics
    'detection_qvalue': 'Detection Q Value',
    'idotp': 'Isotope Dot Product',
    'fwhm': 'Fwhm',
    'shape_correlation': 'Shape Correlation',
    'coeluting': 'Coeluting',
    'truncated': 'Truncated',
    # File info
    'file_name': 'File Name',
    'tic_area': 'Total Ion Current Area',
    'acquired_time': 'Acquired Time',
}

# Default column names - users should use these Skyline names in config
# or override with their actual column names
DEFAULT_COLUMNS = {
    'abundance': 'Area',
    'rt': 'Retention Time',
    'peptide': 'Peptide Modified Sequence',
    'protein': 'Protein Accession',
    'protein_name': 'Protein',
    'sample': 'Replicate Name',
    'transition': 'Fragment Ion',
    'batch': 'Batch Name',
}

# Required columns for processing (using Skyline names)
REQUIRED_COLUMNS = [
    'Protein Accession',
    'Peptide Modified Sequence',
    'Replicate Name',
]

# At least one of these abundance columns required
ABUNDANCE_COLUMNS = ['Area', 'Total Area Fragment', 'Total Area MS1']

# Sample metadata column name alternatives (Skyline conventions + PRISM conventions)
# Order matters - first match is used
METADATA_REPLICATE_COLUMNS = ['sample', 'Replicate Name', 'ReplicateName', 'File Name']
METADATA_SAMPLE_TYPE_COLUMNS = ['sample_type', 'Sample Type', 'SampleType']
METADATA_BATCH_COLUMNS = ['batch', 'Batch', 'Batch Name']  # Skyline uses 'Batch Name'

# Skyline Sample Type to PRISM sample_type mapping
# Skyline options: Unknown, Standard, Quality Control, Solvent, Blank, Double Blank
SKYLINE_SAMPLE_TYPE_MAP = {
    'Unknown': 'experimental',
    'Standard': 'reference',
    'Quality Control': 'qc',
    # Others are typically excluded from analysis
    'Solvent': 'blank',
    'Blank': 'blank',
    'Double Blank': 'blank',
}

VALID_SAMPLE_TYPES = {'experimental', 'qc', 'reference', 'blank'}

# Default patterns for classifying samples by name
# These can be overridden in config
DEFAULT_SAMPLE_TYPE_PATTERNS = {
    'reference': ['-Pool_', '-Pool', '_Pool_', '_Pool'],  # Reference samples (e.g., commercial plasma pool)
    'qc': ['-Carl_', '-Carl', '_QC_', '_QC', '-QC_', '-QC'],  # Intra-experiment QC
    # Everything else is experimental
}


def classify_sample_by_name(
    sample_name: str,
    reference_patterns: list[str] | None = None,
    qc_patterns: list[str] | None = None,
) -> str:
    """Classify sample type based on naming patterns.

    Args:
        sample_name: The sample/replicate name to classify
        reference_patterns: Patterns indicating inter-experiment reference samples
        qc_patterns: Patterns indicating intra-experiment QC samples

    Returns:
        Sample type: 'reference', 'qc', or 'experimental'

    """
    if reference_patterns is None:
        reference_patterns = DEFAULT_SAMPLE_TYPE_PATTERNS['reference']
    if qc_patterns is None:
        qc_patterns = DEFAULT_SAMPLE_TYPE_PATTERNS['qc']

    for pattern in reference_patterns:
        if pattern in sample_name:
            return 'reference'

    for pattern in qc_patterns:
        if pattern in sample_name:
            return 'qc'

    return 'experimental'


def generate_sample_metadata(
    samples_by_batch: dict[str, set[str]],
    reference_patterns: list[str] | None = None,
    qc_patterns: list[str] | None = None,
) -> pd.DataFrame:
    """Generate sample metadata DataFrame from sample names.

    Automatically classifies samples based on naming patterns and assigns
    batch information. Creates a unique sample_id that combines sample name
    and batch to handle duplicate sample names across batches (e.g., Reference or QC
    samples run in each plate).

    Args:
        samples_by_batch: Dict mapping batch_name -> set of sample names
        reference_patterns: Patterns for reference samples (patterns like -Pool_ match reference samples)
        qc_patterns: Patterns for QC samples (default: -Carl_, -QC_, etc.)

    Returns:
        DataFrame with columns: sample_id, sample, sample_type, batch
        where sample_id is unique across all batches

    """
    rows = []
    for batch_name, sample_names in samples_by_batch.items():
        for sample_name in sorted(sample_names):
            sample_type = classify_sample_by_name(
                sample_name, reference_patterns, qc_patterns
            )
            # Create unique sample_id combining sample name and batch
            # This ensures samples with the same name in different batches
            # are treated as separate replicates
            sample_id = f"{sample_name}__@__{batch_name}"
            rows.append({
                'sample_id': sample_id,
                'sample': sample_name,
                'sample_type': sample_type,
                'batch': batch_name,
            })

    df = pd.DataFrame(rows)

    # Log summary
    logger.info("Generated sample metadata:")
    for batch in df['batch'].unique():
        batch_df = df[df['batch'] == batch]
        counts = batch_df['sample_type'].value_counts().to_dict()
        logger.info(f"  {batch}: {counts}")

    return df


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
    """Keep columns as-is - no renaming.
    
    Previously this renamed Skyline columns to internal names. Now we preserve
    the original Skyline column names throughout the pipeline.
    """
    # No renaming - keep original Skyline column names
    return df


def _is_large_file(filepath: Path) -> bool:
    """Check if file exceeds the large file threshold."""
    return filepath.stat().st_size > LARGE_FILE_THRESHOLD_BYTES


def convert_skyline_csv_to_parquet(
    csv_path: Path,
    output_path: Path,
    batch_name: str | None = None,
    block_size_mb: int = 256,
) -> tuple[Path, set[str], int]:
    """Convert a large Skyline CSV to Parquet using streaming.

    Uses PyArrow's streaming CSV reader for memory-efficient processing
    of large files. Writes Parquet with zstd compression.

    Args:
        csv_path: Path to input CSV file
        output_path: Path for output Parquet file
        batch_name: Batch identifier to add as a column (optional)
        block_size_mb: Block size for streaming reader in MB

    Returns:
        Tuple of (output_path, set of sample names, row count)

    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_size_gb = csv_path.stat().st_size / 1e9
    logger.info(f"Converting {csv_path.name} ({file_size_gb:.1f} GB) to Parquet...")

    # Detect delimiter
    suffix = csv_path.suffix.lower()
    delimiter = '\t' if suffix in ['.tsv', '.txt'] else ','

    # PyArrow CSV read options for streaming
    read_options = pa_csv.ReadOptions(
        use_threads=True,
        block_size=block_size_mb * 1024 * 1024,
    )
    parse_options = pa_csv.ParseOptions(
        delimiter=delimiter,
    )
    convert_options = pa_csv.ConvertOptions(
        strings_can_be_null=True,
    )

    # Build column rename map for PyArrow
    # Open streaming reader
    reader = pa_csv.open_csv(
        csv_path,
        read_options=read_options,
        parse_options=parse_options,
        convert_options=convert_options,
    )

    sample_names: set[str] = set()
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    batch_count = 0

    # Use Skyline's standard column name for replicate
    replicate_col = 'Replicate Name'

    for batch in reader:
        batch_count += 1

        # Keep original Skyline column names - no renaming

        # Extract unique sample names efficiently using PyArrow
        if replicate_col in batch.schema.names:
            rep_col = batch.column(replicate_col)
            unique_vals = rep_col.unique()
            for val in unique_vals.to_pylist():
                if val is not None:
                    sample_names.add(val)

        # Add batch column if specified (using Skyline's naming convention)
        if batch_name:
            batch = batch.append_column(
                'Batch',
                pa.array([batch_name] * len(batch), type=pa.string())
            )

        # Add source_document column
        batch = batch.append_column(
            'Source Document',
            pa.array([csv_path.stem] * len(batch), type=pa.string())
        )

        # Initialize writer with first batch's schema
        if writer is None:
            writer = pq.ParquetWriter(
                output_path,
                batch.schema,
                compression='zstd',
                compression_level=3,
            )

        writer.write_batch(batch)
        total_rows += len(batch)

        if batch_count % 10 == 0:
            logger.debug(f"  Processed {batch_count} batches, {total_rows:,} rows...")

    if writer:
        writer.close()

    logger.info(
        f"  Converted: {total_rows:,} rows, {len(sample_names)} samples -> {output_path.name}"
    )

    return output_path, sample_names, total_rows


# =============================================================================
# Source File Fingerprinting
# =============================================================================


@dataclass
class SourceFingerprint:
    """Fingerprint of a source file for cache validation.

    Stores file metadata that can be used to verify if a cached parquet file
    was generated from the same source files.
    """

    path: str
    filename: str
    size: int
    mtime_iso: str
    md5: str | None = None  # Optional, computed only in strict mode


def compute_file_fingerprint(
    file_path: Path,
    compute_md5: bool = False,
) -> SourceFingerprint:
    """Compute fingerprint for a single source file.

    Args:
        file_path: Path to the file
        compute_md5: If True, compute MD5 checksum (slower but more reliable)

    Returns:
        SourceFingerprint with file metadata

    """
    file_path = Path(file_path).resolve()
    stat = file_path.stat()

    mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

    md5_hash = None
    if compute_md5:
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192 * 1024), b''):  # 8MB chunks
                md5.update(chunk)
        md5_hash = md5.hexdigest()

    return SourceFingerprint(
        path=str(file_path),
        filename=file_path.name,
        size=stat.st_size,
        mtime_iso=mtime.isoformat(),
        md5=md5_hash,
    )


def compute_source_fingerprints(
    file_paths: list[Path],
    compute_md5: bool = False,
) -> list[dict]:
    """Compute fingerprints for multiple source files.

    Args:
        file_paths: List of paths to source files
        compute_md5: If True, compute MD5 checksums

    Returns:
        List of fingerprint dictionaries (JSON-serializable)

    """
    fingerprints = []
    for fp in file_paths:
        fingerprint = compute_file_fingerprint(fp, compute_md5=compute_md5)
        fingerprints.append({
            'path': fingerprint.path,
            'filename': fingerprint.filename,
            'size': fingerprint.size,
            'mtime_iso': fingerprint.mtime_iso,
            'md5': fingerprint.md5,
        })
    return fingerprints


def verify_source_fingerprints(
    current_files: list[Path],
    stored_fingerprints: list[dict],
    strict: bool = False,
) -> tuple[bool, list[str]]:
    """Verify that current source files match stored fingerprints.

    Args:
        current_files: List of current input file paths
        stored_fingerprints: Fingerprints from parquet metadata
        strict: If True, also verify MD5 checksums (requires recomputing)

    Returns:
        Tuple of (is_valid, list of mismatch reasons)

    """
    reasons = []

    # Check file count
    if len(current_files) != len(stored_fingerprints):
        reasons.append(
            f"File count mismatch: {len(current_files)} current vs "
            f"{len(stored_fingerprints)} stored"
        )
        return False, reasons

    # Build lookup by filename for matching
    stored_by_name = {fp['filename']: fp for fp in stored_fingerprints}

    for current_path in current_files:
        current_path = Path(current_path).resolve()
        filename = current_path.name

        if filename not in stored_by_name:
            reasons.append(f"File not in stored fingerprints: {filename}")
            continue

        stored = stored_by_name[filename]

        # Check file exists
        if not current_path.exists():
            reasons.append(f"File no longer exists: {filename}")
            continue

        # Check size
        current_size = current_path.stat().st_size
        if current_size != stored['size']:
            reasons.append(
                f"Size mismatch for {filename}: "
                f"{current_size:,} bytes vs {stored['size']:,} bytes stored"
            )
            continue

        # Check mtime
        current_mtime = datetime.fromtimestamp(
            current_path.stat().st_mtime, tz=timezone.utc
        )
        stored_mtime = datetime.fromisoformat(stored['mtime_iso'])
        if current_mtime != stored_mtime:
            reasons.append(
                f"Modification time mismatch for {filename}: "
                f"{current_mtime.isoformat()} vs {stored['mtime_iso']} stored"
            )
            continue

        # Optional strict MD5 check
        if strict and stored.get('md5'):
            current_fp = compute_file_fingerprint(current_path, compute_md5=True)
            if current_fp.md5 != stored['md5']:
                reasons.append(
                    f"MD5 mismatch for {filename}: "
                    f"{current_fp.md5} vs {stored['md5']} stored"
                )

    return len(reasons) == 0, reasons


def get_parquet_source_fingerprints(parquet_path: Path) -> list[dict] | None:
    """Read source fingerprints from parquet sidecar file or embedded metadata.

    First checks for a sidecar JSON file (memory-efficient for large files),
    then falls back to embedded parquet metadata for backwards compatibility.

    Args:
        parquet_path: Path to parquet file

    Returns:
        List of fingerprint dicts, or None if not found

    """
    # First check for sidecar JSON file (preferred for large files)
    sidecar_path = parquet_path.with_suffix('.fingerprints.json')
    if sidecar_path.exists():
        try:
            with open(sidecar_path) as f:
                data = json.load(f)
                return data.get('source_fingerprints')
        except Exception as e:
            logger.debug(f"Could not read fingerprints from sidecar {sidecar_path}: {e}")

    # Fall back to embedded parquet metadata (backwards compatibility)
    try:
        pf = pq.ParquetFile(parquet_path)
        metadata = pf.schema_arrow.metadata
        if metadata and b'prism_source_fingerprints' in metadata:
            fingerprints_json = metadata[b'prism_source_fingerprints'].decode('utf-8')
            return json.loads(fingerprints_json)
    except Exception as e:
        logger.debug(f"Could not read fingerprints from {parquet_path}: {e}")

    return None


def merge_skyline_reports_streaming(
    report_paths: list[Path],
    output_path: Path,
    batch_names: list[str] | None = None,
    block_size_mb: int = 256,
    compute_fingerprint_md5: bool = False,
) -> tuple[Path, dict[str, set[str]], int]:
    """Merge multiple large Skyline CSVs to a single Parquet file using streaming.

    Memory-efficient alternative to merge_skyline_reports for large files.
    Processes each CSV file in streaming fashion and appends to a single
    Parquet file.

    Source file fingerprints are embedded in the parquet metadata for cache
    validation on subsequent runs.

    Args:
        report_paths: List of paths to Skyline CSV reports
        output_path: Path for output Parquet file
        batch_names: Optional list of batch names (one per report).
                     If None, uses filename stems as batch names.
        block_size_mb: Block size for streaming reader in MB
        compute_fingerprint_md5: If True, compute MD5 checksums for fingerprints

    Returns:
        Tuple of (output_path, dict of batch_name -> sample_names, total_row_count)

    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if batch_names is None:
        batch_names = [p.stem for p in report_paths]

    if len(batch_names) != len(report_paths):
        raise ValueError("batch_names must have same length as report_paths")

    # Compute source file fingerprints for cache validation
    logger.info("Computing source file fingerprints...")
    fingerprints = compute_source_fingerprints(
        report_paths, compute_md5=compute_fingerprint_md5
    )

    total_size_gb = sum(p.stat().st_size for p in report_paths) / 1e9
    logger.info(
        f"Merging {len(report_paths)} reports ({total_size_gb:.1f} GB total) to Parquet..."
    )

    all_samples: dict[str, set[str]] = {}
    total_rows = 0
    writer: pq.ParquetWriter | None = None

    # Use Skyline's standard column name for replicate
    replicate_col = 'Replicate Name'

    for csv_path, batch_name in zip(report_paths, batch_names):
        csv_path = Path(csv_path)
        file_size_gb = csv_path.stat().st_size / 1e9
        logger.info(f"  Processing {csv_path.name} ({file_size_gb:.1f} GB)...")

        # Detect delimiter
        suffix = csv_path.suffix.lower()
        delimiter = '\t' if suffix in ['.tsv', '.txt'] else ','

        read_options = pa_csv.ReadOptions(
            use_threads=True,
            block_size=block_size_mb * 1024 * 1024,
        )
        parse_options = pa_csv.ParseOptions(delimiter=delimiter)
        convert_options = pa_csv.ConvertOptions(strings_can_be_null=True)

        reader = pa_csv.open_csv(
            csv_path,
            read_options=read_options,
            parse_options=parse_options,
            convert_options=convert_options,
        )

        sample_names: set[str] = set()
        file_rows = 0
        batch_count = 0

        for batch in reader:
            batch_count += 1

            # Keep original Skyline column names - no renaming

            # Extract unique sample names efficiently using PyArrow
            if replicate_col in batch.schema.names:
                rep_col = batch.column(replicate_col)
                # Use PyArrow's unique() for efficiency instead of iterating
                unique_vals = rep_col.unique()
                for val in unique_vals.to_pylist():
                    if val is not None:
                        sample_names.add(val)

            # Add batch and source columns (using Skyline naming conventions)
            batch = batch.append_column(
                'Batch', pa.array([batch_name] * len(batch), type=pa.string())
            )
            batch = batch.append_column(
                'Source Document', pa.array([csv_path.stem] * len(batch), type=pa.string())
            )

            # Create unique Sample ID combining Replicate Name and Batch
            # This ensures samples with the same name in different batches are treated
            # as separate replicates (e.g., Reference or QC samples run in each plate)
            if replicate_col in batch.schema.names:
                rep_array = batch.column(replicate_col)
                # Create Sample ID as "ReplicateName__BatchName"
                sample_ids = [
                    f"{rep}__@__{batch_name}" if rep is not None else None
                    for rep in rep_array.to_pylist()
                ]
                batch = batch.append_column(
                    'Sample ID', pa.array(sample_ids, type=pa.string())
                )

            # Normalize schema: cast timestamp columns to string for consistency
            # Different CSV files may parse date/time columns differently
            # (e.g., "Acquired Time" may be timestamp in one file, string in another)
            # Only rebuild batch if there are actually timestamp columns to convert
            timestamp_cols = [
                i for i, fld in enumerate(batch.schema)
                if pa.types.is_timestamp(fld.type)
            ]
            if timestamp_cols:
                # Build new schema with timestamps converted to strings
                new_fields = []
                new_arrays = []
                for i, fld in enumerate(batch.schema):
                    col = batch.column(i)
                    if i in timestamp_cols:
                        col = col.cast(pa.string())
                        fld = pa.field(fld.name, pa.string())
                    new_fields.append(fld)
                    new_arrays.append(col)
                batch = pa.RecordBatch.from_arrays(
                    new_arrays,
                    schema=pa.schema(new_fields)
                )

            # Initialize or append to writer
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    batch.schema,
                    compression='zstd',
                    compression_level=3,
                )

            writer.write_batch(batch)
            file_rows += len(batch)

            # Progress logging every 20 batches (~5GB for 256MB blocks)
            if batch_count % 20 == 0:
                logger.info(f"    Progress: {file_rows:,} rows processed...")

        all_samples[batch_name] = sample_names
        total_rows += file_rows
        logger.info(f"    Completed: {file_rows:,} rows, {len(sample_names)} samples")

    if writer:
        writer.close()

    # Embed source fingerprints in parquet metadata
    logger.info("Embedding source fingerprints in parquet metadata...")
    _add_fingerprints_to_parquet(output_path, fingerprints)

    logger.info(f"Merge complete: {total_rows:,} total rows -> {output_path.name}")

    return output_path, all_samples, total_rows


def _add_fingerprints_to_parquet(parquet_path: Path, fingerprints: list[dict]) -> None:
    """Add source fingerprints as a sidecar JSON file.

    For large files, embedding metadata in the parquet file requires reading
    the entire file into memory which is not feasible. Instead, we write the
    fingerprints to a sidecar JSON file next to the parquet.

    Args:
        parquet_path: Path to existing parquet file
        fingerprints: List of fingerprint dictionaries

    """
    # Write fingerprints to sidecar JSON file
    sidecar_path = parquet_path.with_suffix('.fingerprints.json')
    fingerprint_data = {
        'prism_version': '0.2.0',
        'source_fingerprints': fingerprints,
    }
    with open(sidecar_path, 'w') as f:
        json.dump(fingerprint_data, f, indent=2)
    logger.info(f"  Wrote fingerprints to {sidecar_path.name}")


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
            # Use Skyline column name (standardize_columns no longer renames)
            replicate_col = 'Replicate Name'
            if replicate_col in df_full.columns:
                result.n_replicates = df_full[replicate_col].nunique()
            elif 'replicate_name' in df_full.columns:
                result.n_replicates = df_full['replicate_name'].nunique()
            else:
                result.n_replicates = 0

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

    Supports Skyline metadata exports with automatic column name normalization:
    - ReplicateName: accepts 'Replicate Name', 'File Name'
    - SampleType: accepts 'Sample Type' with Skyline values mapped:
        - Standard → reference
        - Quality Control → qc
        - Unknown → experimental
        - Solvent/Blank/Double Blank → blank
    - Batch: accepts 'Batch Name'

    Args:
        filepath: Path to metadata TSV/CSV

    Returns:
        Validated metadata DataFrame with standardized column names

    Raises:
        ValueError: If validation fails

    """
    filepath = Path(filepath)

    # Detect delimiter
    suffix = filepath.suffix.lower()
    sep = '\t' if suffix in ['.tsv', '.txt'] else ','

    meta = pd.read_csv(filepath, sep=sep)

    # Normalize sample name column to 'sample'
    replicate_col = None
    for col_name in METADATA_REPLICATE_COLUMNS:
        if col_name in meta.columns:
            replicate_col = col_name
            if col_name != 'sample':
                meta = meta.rename(columns={col_name: 'sample'})
                logger.info(f"Renamed '{col_name}' column to 'sample'")
            break
    if replicate_col is None:
        raise ValueError(
            f"Missing replicate name column. Expected one of: {METADATA_REPLICATE_COLUMNS}"
        )

    # Normalize sample type column to 'sample_type'
    sample_type_col = None
    for col_name in METADATA_SAMPLE_TYPE_COLUMNS:
        if col_name in meta.columns:
            sample_type_col = col_name
            if col_name != 'sample_type':
                meta = meta.rename(columns={col_name: 'sample_type'})
                logger.info(f"Renamed '{col_name}' column to 'sample_type'")
            break
    if sample_type_col is None:
        raise ValueError(
            f"Missing sample type column. Expected one of: {METADATA_SAMPLE_TYPE_COLUMNS}"
        )

    # Normalize batch column to 'batch' (Skyline uses 'Batch Name')
    for batch_col in METADATA_BATCH_COLUMNS:
        if batch_col in meta.columns and batch_col != 'batch':
            meta = meta.rename(columns={batch_col: 'batch'})
            logger.info(f"Renamed '{batch_col}' column to 'batch'")
            break

    # Map Skyline sample types to PRISM types
    original_types = meta['sample_type'].unique()
    skyline_types = set(original_types) & set(SKYLINE_SAMPLE_TYPE_MAP.keys())
    if skyline_types:
        logger.info(f"Mapping Skyline sample types: {list(skyline_types)}")
        meta['sample_type'] = meta['sample_type'].map(
            lambda x: SKYLINE_SAMPLE_TYPE_MAP.get(x, x)
        )

    # Note if batch column is missing (will be estimated later)
    if 'batch' not in meta.columns:
        logger.info("No batch column in metadata - batches will be estimated")

    # Validate sample_type values
    invalid_types = set(meta['sample_type'].unique()) - VALID_SAMPLE_TYPES
    if invalid_types:
        raise ValueError(
            f"Invalid sample_type values: {invalid_types}. "
            f"Must be one of: {VALID_SAMPLE_TYPES} "
            f"(or Skyline types: {list(SKYLINE_SAMPLE_TYPE_MAP.keys())})"
        )

    # Check for duplicate sample names within the same batch
    # (duplicates across batches are allowed - e.g., the same reference/pool sample
    # run in multiple batches)
    if 'batch' in meta.columns:
        duplicates = (
            meta.groupby(['sample', 'batch'])
            .size()
            .reset_index(name='count')
        )
        duplicates = duplicates[duplicates['count'] > 1]
        if not duplicates.empty:
            dup_list = duplicates[['sample', 'batch']].values.tolist()
            raise ValueError(f"Duplicate sample entries within batch: {dup_list}")
    else:
        duplicates = meta[meta['sample'].duplicated()]['sample'].tolist()
        if duplicates:
            raise ValueError(f"Duplicate sample entries: {duplicates}")

    # Ensure RunOrder is numeric if present
    if 'RunOrder' in meta.columns:
        meta['RunOrder'] = pd.to_numeric(meta['RunOrder'], errors='coerce')
        if meta['RunOrder'].isna().any():
            raise ValueError("RunOrder must be numeric")
    else:
        logger.info(
            "No RunOrder column in metadata - will be calculated from acquired_time"
        )

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
        merge_cols = ['replicate_name', 'SampleType']
        rename_map = {'SampleType': 'sample_type'}
        
        if 'Batch' in meta.columns:
            merge_cols.append('Batch')
            rename_map['Batch'] = 'batch'
        if 'RunOrder' in meta.columns:
            merge_cols.append('RunOrder')
            rename_map['RunOrder'] = 'run_order'
        
        merged = merged.merge(
            meta[merge_cols],
            on='replicate_name',
            how='left'
        )
        merged = merged.rename(columns=rename_map)
        
        # Calculate run_order from acquired_time if not provided
        if 'run_order' not in merged.columns and 'acquired_time' in merged.columns:
            logger.info("Calculating run_order from acquired_time")
            # Get unique replicate/acquired_time pairs
            rep_times = (
                merged[['replicate_name', 'acquired_time']]
                .drop_duplicates()
                .sort_values('acquired_time')
            )
            rep_times['run_order'] = range(1, len(rep_times) + 1)
            merged = merged.merge(
                rep_times[['replicate_name', 'run_order']],
                on='replicate_name',
                how='left'
            )

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
