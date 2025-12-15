"""
Data I/O module for loading and merging Skyline reports.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import logging

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

# Sample metadata required columns
METADATA_REQUIRED = ['ReplicateName', 'SampleType', 'Batch', 'RunOrder']
VALID_SAMPLE_TYPES = {'experimental', 'pool', 'reference'}


@dataclass
class ValidationResult:
    """Result of validating a Skyline report."""
    is_valid: bool
    filepath: Path
    missing_required: List[str] = field(default_factory=list)
    missing_abundance: bool = False
    extra_columns: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
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
    warnings: List[str] = field(default_factory=list)
    replicate_sources: Dict[str, str] = field(default_factory=dict)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to standard names using the mapping.
    """
    rename_map = {}
    for orig, standard in SKYLINE_COLUMN_MAP.items():
        if orig in df.columns:
            rename_map[orig] = standard
    
    return df.rename(columns=rename_map)


def validate_skyline_report(filepath: Path) -> ValidationResult:
    """
    Validate that a Skyline report has required columns.
    
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
    """
    Load a single Skyline report with standardized column names.
    
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
    """
    Load and validate sample metadata file.
    
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
    
    # Check required columns
    missing = [col for col in METADATA_REQUIRED if col not in meta.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")
    
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
    report_paths: List[Path],
    output_path: Path,
    sample_metadata: Optional[pd.DataFrame] = None,
    partition_by_batch: bool = True,
) -> MergeResult:
    """
    Merge multiple Skyline reports into unified parquet.
    
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
    """
    Load unified parquet data (handles both single file and partitioned).
    
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
def identify_internal_qcs(df: pd.DataFrame) -> Tuple[Set[str], Set[str]]:
    """
    Identify PRTC and enolase peptides in the data.
    
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
