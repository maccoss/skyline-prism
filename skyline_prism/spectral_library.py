"""Spectral library module for loading and matching library spectra.

This module supports loading spectral libraries from:
- Carafe TSV format (DIA-NN compatible)
- BLIB format (Skyline's SQLite-based spectral library)

The primary use case is providing reference fragment intensities for
least-squares rollup quantification, which can handle interference
by fitting observed intensities to expected library patterns.

Key concepts:
- Spectral libraries contain "expected" relative fragment intensities
- Least squares fitting: observed = scale * library + interference
- Robust to 1-2 interfered fragments when 4+ fragments are available
"""

from __future__ import annotations

import logging
import sqlite3
import struct
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class FragmentSpectrum:
    """Represents a single peptide's fragment spectrum from a library.

    All intensities are normalized (relative) with the base peak = 1.0.
    """

    # Peptide identification
    modified_sequence: str  # Modified peptide sequence
    stripped_sequence: str  # Unmodified sequence
    precursor_charge: int
    precursor_mz: float

    # Fragment information
    # Keys: (fragment_type, fragment_number, fragment_charge, loss_type)
    # e.g., ("b", 3, 1, "noloss") or ("y", 7, 2, "H2O")
    # Values: relative intensity (0-1, normalized to base peak = 1.0)
    fragments: dict[tuple[str, int, int, str], float] = field(default_factory=dict)

    # Alternative: m/z-based lookup for matching by mass
    # Keys: product m/z (rounded to match tolerance)
    # Values: relative intensity
    fragments_by_mz: dict[float, float] = field(default_factory=dict)

    # Optional: retention time (for RT matching)
    retention_time: float | None = None

    # Protein accession(s)
    protein_ids: list[str] = field(default_factory=list)


class SpectralLibraryLoader(ABC):
    """Abstract base class for spectral library loaders."""

    @abstractmethod
    def load(self) -> dict[str, FragmentSpectrum]:
        """Load library and return dict of spectra.

        Returns:
            Dict mapping peptide key (modified_sequence + charge) to FragmentSpectrum.
            Key format: "PEPTIDEK_2" for charge 2.

        """
        pass

    @staticmethod
    def make_peptide_key(modified_sequence: str, charge: int) -> str:
        """Create standardized peptide key for library lookup.

        Args:
            modified_sequence: Modified peptide sequence
            charge: Precursor charge state

        Returns:
            Key string in format "SEQUENCE_CHARGE"

        """
        return f"{modified_sequence}_{charge}"

    @staticmethod
    def normalize_sequence_for_matching(sequence: str) -> str:
        """Normalize peptide sequence for library matching.

        Handles I/L ambiguity (mass spectrometry cannot distinguish).

        Args:
            sequence: Peptide sequence (modified or unmodified)

        Returns:
            Normalized sequence with L replaced by I

        """
        return sequence.replace("L", "I")

    @staticmethod
    def strip_modifications(sequence: str) -> str:
        """Strip all modifications from a peptide sequence.

        Handles multiple modification formats:
        - Unimod format: M(unimod:35)
        - Mass delta format: M[+15.99491]
        - Skyline format: M[Oxidation (M)]

        Args:
            sequence: Modified peptide sequence

        Returns:
            Unmodified (stripped) sequence

        """
        import re

        # Remove (unimod:N) format
        result = re.sub(r"\(unimod:\d+\)", "", sequence)
        # Remove [+N.NNNNN] format (mass delta)
        result = re.sub(r"\[[+-]?\d+\.?\d*\]", "", result)
        # Remove [Modification Name] format
        result = re.sub(r"\[[^\]]+\]", "", result)
        # Remove any remaining parenthetical modifications
        result = re.sub(r"\([^)]+\)", "", result)

        return result

    @staticmethod
    def make_stripped_key(modified_sequence: str, charge: int) -> str:
        """Create a stripped (unmodified) peptide key for fallback matching.

        This allows matching between different modification notation systems
        by comparing only the bare sequence and charge.

        Args:
            modified_sequence: Modified peptide sequence
            charge: Precursor charge state

        Returns:
            Key string in format "STRIPPEDSEQUENCE_CHARGE"

        """
        stripped = SpectralLibraryLoader.strip_modifications(modified_sequence)
        return f"{stripped}_{charge}"


class CarafeTSVLoader(SpectralLibraryLoader):
    """Load Carafe/DIA-NN compatible TSV spectral libraries.

    Uses DuckDB for efficient streaming loading of large libraries (7GB+).

    Expected columns:
    - ModifiedPeptide: Modified sequence (e.g., _PEPTIDEK_)
    - StrippedPeptide: Unmodified sequence
    - PrecursorMz: Precursor m/z
    - PrecursorCharge: Precursor charge state
    - Tr_recalibrated: Retention time (optional)
    - ProteinID: Protein accession(s)
    - FragmentMz: Fragment m/z
    - RelativeIntensity: Relative intensity (0-1)
    - FragmentType: Ion type (b, y, etc.)
    - FragmentNumber: Ion number (1, 2, 3, ...)
    - FragmentCharge: Fragment charge state
    - FragmentLossType: Neutral loss (noloss, H2O, NH3, etc.)
    """

    def __init__(self, path: Path | str):
        """Initialize loader with path to TSV file.

        Args:
            path: Path to Carafe TSV spectral library

        """
        self.path = Path(path)

    def load(self) -> dict[str, FragmentSpectrum]:
        """Load Carafe TSV spectral library using DuckDB for efficiency.

        Returns:
            Dict mapping peptide keys to FragmentSpectrum objects

        """
        import duckdb

        logger.info(f"Loading Carafe TSV library from {self.path} (using DuckDB)")

        con = duckdb.connect()

        # First check columns available
        col_query = f"""
            SELECT column_name FROM (
                DESCRIBE SELECT * FROM read_csv_auto('{self.path}', delim='\\t', header=true)
            )
        """
        columns = [row[0] for row in con.execute(col_query).fetchall()]

        # Check required columns
        required_cols = [
            "ModifiedPeptide",
            "PrecursorCharge",
            "FragmentMz",
            "RelativeIntensity",
            "FragmentType",
            "FragmentNumber",
            "FragmentCharge",
        ]
        missing = [c for c in required_cols if c not in columns]
        if missing:
            con.close()
            raise ValueError(f"Missing required columns in Carafe TSV: {missing}")

        # Optional columns
        has_stripped = "StrippedPeptide" in columns
        has_precursor_mz = "PrecursorMz" in columns
        has_rt = "Tr_recalibrated" in columns
        has_protein = "ProteinID" in columns
        has_loss = "FragmentLossType" in columns

        # Build select statement with available columns
        select_cols = [
            "ModifiedPeptide",
            "PrecursorCharge",
            "FragmentMz",
            "RelativeIntensity",
            "FragmentType",
            "FragmentNumber",
            "FragmentCharge",
        ]
        if has_stripped:
            select_cols.append("StrippedPeptide")
        if has_precursor_mz:
            select_cols.append("PrecursorMz")
        if has_rt:
            select_cols.append("Tr_recalibrated")
        if has_protein:
            select_cols.append("ProteinID")
        if has_loss:
            select_cols.append("FragmentLossType")

        # Query and aggregate in DuckDB - much faster than pandas groupby
        query = f"""
            SELECT
                ModifiedPeptide,
                PrecursorCharge,
                {"StrippedPeptide," if has_stripped else ""}
                {"PrecursorMz," if has_precursor_mz else ""}
                {"Tr_recalibrated," if has_rt else ""}
                {"FIRST(ProteinID) as ProteinID," if has_protein else ""}
                LIST(FragmentMz) as frag_mz_list,
                LIST(RelativeIntensity) as frag_intensity_list,
                LIST(FragmentType) as frag_type_list,
                LIST(FragmentNumber) as frag_num_list,
                LIST(FragmentCharge) as frag_charge_list
                {"," + "LIST(FragmentLossType) as frag_loss_list" if has_loss else ""}
            FROM read_csv_auto('{self.path}', delim='\\t', header=true)
            WHERE Decoy = 0 OR Decoy IS NULL
            GROUP BY ModifiedPeptide, PrecursorCharge
                {"," + "StrippedPeptide" if has_stripped else ""}
                {"," + "PrecursorMz" if has_precursor_mz else ""}
                {"," + "Tr_recalibrated" if has_rt else ""}
        """

        logger.info("  Aggregating fragments by peptide (this may take a minute)...")
        result = con.execute(query).fetchall()
        col_names = [desc[0] for desc in con.description]
        con.close()

        logger.info(f"  Processing {len(result)} peptide spectra...")

        # Build spectra dictionary
        spectra: dict[str, FragmentSpectrum] = {}

        for row in result:
            row_dict = dict(zip(col_names, row))

            mod_seq = str(row_dict["ModifiedPeptide"]).strip("_")
            charge = int(row_dict["PrecursorCharge"])
            key = self.make_peptide_key(mod_seq, charge)

            # Extract peptide-level info
            stripped = str(row_dict.get("StrippedPeptide", mod_seq))
            precursor_mz = float(row_dict.get("PrecursorMz", 0.0) or 0.0)
            rt = row_dict.get("Tr_recalibrated")
            rt = float(rt) if rt is not None else None
            protein_str = row_dict.get("ProteinID")
            protein_ids = str(protein_str).split(";") if protein_str else []

            spectrum = FragmentSpectrum(
                modified_sequence=mod_seq,
                stripped_sequence=stripped,
                precursor_charge=charge,
                precursor_mz=precursor_mz,
                retention_time=rt,
                protein_ids=protein_ids,
            )

            # Add fragments from aggregated lists
            mz_list = row_dict["frag_mz_list"]
            intensity_list = row_dict["frag_intensity_list"]
            type_list = row_dict["frag_type_list"]
            num_list = row_dict["frag_num_list"]
            charge_list = row_dict["frag_charge_list"]
            loss_list = row_dict.get("frag_loss_list", ["noloss"] * len(mz_list))

            for i in range(len(mz_list)):
                frag_type = str(type_list[i])
                frag_num = int(num_list[i])
                frag_charge = int(charge_list[i])
                frag_loss = str(loss_list[i]) if loss_list else "noloss"
                frag_mz = float(mz_list[i])
                rel_intensity = float(intensity_list[i])

                # Add to type-based lookup
                frag_key = (frag_type, frag_num, frag_charge, frag_loss)
                spectrum.fragments[frag_key] = rel_intensity

                # Add to m/z-based lookup (rounded to 0.01 Da)
                mz_key = round(frag_mz, 2)
                spectrum.fragments_by_mz[mz_key] = rel_intensity

            spectra[key] = spectrum

        logger.info(f"Loaded {len(spectra)} peptide spectra from Carafe library")
        return spectra


class BLIBLoader(SpectralLibraryLoader):
    """Load Skyline BLIB (SQLite) spectral libraries.

    BLIB format stores spectra in SQLite with:
    - RefSpectra: peptide/precursor info
    - RefSpectraPeaks: m/z and intensity blobs
    - RefSpectraPeakAnnotations: fragment annotations (may be empty)
    """

    def __init__(self, path: Path | str):
        """Initialize loader with path to BLIB file.

        Args:
            path: Path to BLIB spectral library

        """
        self.path = Path(path)

    def _decode_mz_blob(self, blob: bytes) -> np.ndarray:
        """Decode zlib-compressed m/z blob (float64).

        Args:
            blob: Compressed bytes

        Returns:
            Array of m/z values

        """
        try:
            # Check for zlib magic bytes (78 9c)
            if blob[:2] == b"\x78\x9c":
                decompressed = zlib.decompress(blob)
            else:
                # Try decompressing anyway, some versions may differ
                try:
                    decompressed = zlib.decompress(blob)
                except zlib.error:
                    # Not compressed, use raw bytes
                    decompressed = blob

            # Parse as float64 array
            n_values = len(decompressed) // 8
            mz_values = np.array(struct.unpack(f"<{n_values}d", decompressed))
            return mz_values

        except Exception as e:
            logger.warning(f"Failed to decode m/z blob: {e}")
            return np.array([])

    def _decode_intensity_blob(self, blob: bytes) -> np.ndarray:
        """Decode intensity blob (raw float32).

        Args:
            blob: Raw bytes (not compressed)

        Returns:
            Array of intensity values (normalized to max=1.0)

        """
        try:
            # Check if zlib compressed
            if blob[:2] == b"\x78\x9c":
                try:
                    blob = zlib.decompress(blob)
                except zlib.error:
                    pass

            # Parse as float32 array
            n_values = len(blob) // 4
            intensities = np.array(struct.unpack(f"<{n_values}f", blob))

            # Normalize to max = 1.0
            if len(intensities) > 0 and intensities.max() > 0:
                intensities = intensities / intensities.max()

            return intensities

        except Exception as e:
            logger.warning(f"Failed to decode intensity blob: {e}")
            return np.array([])

    def load(self) -> dict[str, FragmentSpectrum]:
        """Load BLIB spectral library.

        Returns:
            Dict mapping peptide keys to FragmentSpectrum objects

        """
        logger.info(f"Loading BLIB library from {self.path}")

        if not self.path.exists():
            raise FileNotFoundError(f"BLIB file not found: {self.path}")

        spectra: dict[str, FragmentSpectrum] = {}

        conn = sqlite3.connect(str(self.path))
        try:
            cursor = conn.cursor()

            # Query RefSpectra for peptide info
            cursor.execute("""
                SELECT id, peptideSeq, precursorMZ, precursorCharge,
                       peptideModSeq, retentionTime, numPeaks
                FROM RefSpectra
            """)

            ref_spectra = cursor.fetchall()

            for spec_id, pep_seq, prec_mz, prec_charge, mod_seq, rt, n_peaks in ref_spectra:
                # Use modified sequence if available, otherwise bare sequence
                seq = mod_seq if mod_seq else pep_seq
                key = self.make_peptide_key(seq, int(prec_charge))

                # Skip duplicates (keep first encountered)
                if key in spectra:
                    continue

                # Get peaks
                cursor.execute(
                    """
                    SELECT peakMZ, peakIntensity
                    FROM RefSpectraPeaks
                    WHERE RefSpectraID = ?
                """,
                    (spec_id,),
                )

                peaks_row = cursor.fetchone()
                if not peaks_row:
                    continue

                mz_blob, intensity_blob = peaks_row

                mz_values = self._decode_mz_blob(mz_blob)
                intensities = self._decode_intensity_blob(intensity_blob)

                if len(mz_values) != len(intensities):
                    logger.warning(
                        f"Mismatch in peak count for {key}: "
                        f"{len(mz_values)} m/z vs {len(intensities)} intensities"
                    )
                    continue

                spectrum = FragmentSpectrum(
                    modified_sequence=seq,
                    stripped_sequence=pep_seq,
                    precursor_charge=int(prec_charge),
                    precursor_mz=float(prec_mz),
                    retention_time=float(rt) if rt else None,
                )

                # BLIB typically doesn't have fragment annotations
                # so we only use m/z-based matching
                for mz, intensity in zip(mz_values, intensities):
                    mz_key = round(float(mz), 2)
                    spectrum.fragments_by_mz[mz_key] = float(intensity)

                spectra[key] = spectrum

        finally:
            conn.close()

        logger.info(f"Loaded {len(spectra)} peptide spectra from BLIB library")
        return spectra


def load_spectral_library(path: Path | str) -> dict[str, FragmentSpectrum]:
    """Load spectral library from file, auto-detecting format.

    Supported formats:
    - .tsv: Carafe/DIA-NN format
    - .blib: Skyline BLIB format

    Args:
        path: Path to spectral library file

    Returns:
        Dict mapping peptide keys to FragmentSpectrum objects

    Raises:
        ValueError: If file format is not recognized

    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Spectral library not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".tsv":
        loader = CarafeTSVLoader(path)
    elif suffix == ".blib":
        loader = BLIBLoader(path)
    else:
        raise ValueError(
            f"Unknown spectral library format: {suffix}. "
            f"Supported formats: .tsv (Carafe), .blib (Skyline)"
        )

    return loader.load()


def match_transition_to_library(
    fragment_ion: str,
    product_mz: float,
    library_spectrum: FragmentSpectrum,
    mz_tolerance: float = 0.02,
) -> float | None:
    """Match a transition to library spectrum and return expected relative intensity.

    Matching is attempted in order:
    1. By fragment annotation (e.g., "y6" -> ("y", 6, 1, "noloss"))
    2. By m/z within tolerance

    Args:
        fragment_ion: Skyline fragment ion string (e.g., "y6", "b3++", "y7 - H2O")
        product_mz: Observed product m/z
        library_spectrum: Reference spectrum from library
        mz_tolerance: m/z matching tolerance in Da (default: 0.02)

    Returns:
        Expected relative intensity (0-1), or None if no match found

    """
    # Try annotation-based matching first (more reliable)
    parsed = _parse_fragment_ion(fragment_ion)
    if parsed:
        frag_type, frag_num, frag_charge, frag_loss = parsed
        frag_key = (frag_type, frag_num, frag_charge, frag_loss)
        if frag_key in library_spectrum.fragments:
            return library_spectrum.fragments[frag_key]

    # Fall back to m/z matching
    mz_rounded = round(product_mz, 2)

    # Search within tolerance
    best_match = None
    best_diff = mz_tolerance + 1

    for lib_mz, rel_intensity in library_spectrum.fragments_by_mz.items():
        diff = abs(lib_mz - mz_rounded)
        if diff < best_diff and diff <= mz_tolerance:
            best_diff = diff
            best_match = rel_intensity

    return best_match


def _parse_fragment_ion(fragment_ion: str) -> tuple[str, int, int, str] | None:
    """Parse Skyline fragment ion string into components.

    Examples:
    - "y6" -> ("y", 6, 1, "noloss")
    - "b3++" -> ("b", 3, 2, "noloss")
    - "y7 - H2O" -> ("y", 7, 1, "H2O")
    - "precursor" -> None (skip precursor)

    Args:
        fragment_ion: Skyline fragment ion string

    Returns:
        Tuple of (type, number, charge, loss) or None if unparseable

    """
    import re

    # Skip precursor ions
    if fragment_ion.lower().startswith("precursor"):
        return None

    # Handle neutral losses
    loss = "noloss"
    if " - " in fragment_ion:
        parts = fragment_ion.split(" - ")
        fragment_ion = parts[0]
        loss = parts[1].strip()
    elif " -" in fragment_ion:
        parts = fragment_ion.split(" -")
        fragment_ion = parts[0]
        loss = parts[1].strip()

    # Count charge (++ or +++ or number after ^)
    charge = 1
    if "^" in fragment_ion:
        # Handle charge notation like y6^2
        match = re.search(r"\^(\d+)", fragment_ion)
        if match:
            charge = int(match.group(1))
        fragment_ion = re.sub(r"\^\d+", "", fragment_ion)
    else:
        # Count + symbols
        charge = fragment_ion.count("+") or 1
        fragment_ion = fragment_ion.replace("+", "")

    # Parse ion type and number
    match = re.match(r"([a-z])(\d+)", fragment_ion.lower())
    if not match:
        return None

    ion_type = match.group(1)
    ion_number = int(match.group(2))

    return (ion_type, ion_number, charge, loss)


def least_squares_rollup_vectorized(
    observed_matrix: np.ndarray,
    library_intensities: np.ndarray,
    min_fragments: int = 2,
    outlier_threshold: float = 3.0,
    max_iterations: int = 5,
    remove_outliers: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized least squares rollup across all samples simultaneously.

    This is a high-performance version that solves for all samples in parallel
    using BLAS matrix operations instead of looping over samples.

    Algorithm:
    1. First pass: Solve scale_s = (L . O_s) / (L . L) for all samples at once
    2. Compute residuals for all samples in parallel
    3. For each iteration, identify and mask outliers per sample
    4. Refit with masked observations
    5. Return abundances = scales * sum(library)

    Key optimization: The library vector is shared across all samples, so
    L.L is computed once per iteration and L.O becomes a matrix-vector product.

    Performance: For 500 samples x 15 transitions, this is ~50x faster than
    calling least_squares_rollup() in a loop because it uses BLAS operations.

    Args:
        observed_matrix: (T, S) matrix of observed intensities, T=transitions, S=samples
        library_intensities: (T,) vector of library intensities
        min_fragments: Minimum valid fragments required
        outlier_threshold: MAD multiplier for outlier detection (high residuals only)
        max_iterations: Maximum outlier removal iterations
        remove_outliers: Whether to detect and exclude outliers

    Returns:
        Tuple of:
        - abundances: (S,) array of peptide abundances
        - r_squared: (S,) array of R-squared values
        - n_used: (S,) array of number of fragments used per sample

    """
    n_transitions, n_samples = observed_matrix.shape
    lib = library_intensities.astype(np.float64)

    # Build valid mask: library > 0 for all
    lib_valid = (lib > 0) & ~np.isnan(lib)
    n_valid = np.sum(lib_valid)

    if n_valid < min_fragments:
        return (
            np.full(n_samples, np.nan),
            np.full(n_samples, np.nan),
            np.zeros(n_samples, dtype=np.int32),
        )

    # Extract valid fragments
    lib_v = lib[lib_valid]  # (n_valid,)
    obs_v = observed_matrix[lib_valid, :]  # (n_valid, S)

    # Replace NaN with 0 in observations (treat as missing signal, zeros are valid)
    obs_v = np.nan_to_num(obs_v, nan=0.0)

    # Initialize inclusion mask (all True) - tracks which fragments are included per sample
    included = np.ones((n_valid, n_samples), dtype=bool)
    lib_sum = np.sum(lib_v)

    # Column vector of library intensities for broadcasting
    lib_col = lib_v[:, None]  # (T, 1)

    for iteration in range(max_iterations):
        # Compute scales for all samples: s = (L . O) / (L . L)
        # Using masked observations - excluded fragments contribute 0
        obs_masked = np.where(included, obs_v, 0.0)

        # Dot products vectorized across all samples at once
        lib_dot_obs = np.sum(lib_col * obs_masked, axis=0)  # (S,)
        lib_dot_lib = np.sum(np.where(included, lib_col**2, 0.0), axis=0)  # (S,)

        # Solve for scale, avoid division by zero
        scales = np.where(lib_dot_lib > 0, lib_dot_obs / lib_dot_lib, 0.0)
        scales = np.maximum(scales, 0.0)  # Ensure non-negative

        if not remove_outliers:
            break

        # Compute residuals: obs - scale * lib for all samples at once
        predicted = scales[None, :] * lib_col  # (T, S) via broadcasting
        residuals = obs_v - predicted

        # Only HIGH residuals are outliers (interference = signal > expected)
        # Low or negative residuals are valid - just low abundance
        positive_residuals = np.maximum(residuals, 0.0)

        # Vectorized MAD computation per sample for outlier detection
        # Set excluded positions to NaN for nanmedian
        pos_res_masked = np.where(included & (positive_residuals > 0), positive_residuals, np.nan)

        # Count positive residuals per sample
        n_positive = np.sum(included & (positive_residuals > 0), axis=0)  # (S,)
        enough_residuals = n_positive > 2

        new_outliers = np.zeros_like(included, dtype=bool)

        if np.any(enough_residuals):
            # Compute MAD (Median Absolute Deviation) per sample
            # Only for samples with enough positive residuals
            mad = np.nanmedian(pos_res_masked[:, enough_residuals], axis=0)  # (n_enough,)
            mad_std = np.where(mad > 0, mad * 1.4826, 1.0)

            # Compute z-scores for samples with enough residuals
            z_scores = positive_residuals[:, enough_residuals] / mad_std[None, :]

            # Identify outliers: high z-score AND currently included
            outliers_subset = (z_scores > outlier_threshold) & included[:, enough_residuals]
            new_outliers[:, enough_residuals] = outliers_subset

        if not np.any(new_outliers):
            # Converged - no new outliers found
            break

        # Exclude new outliers from future iterations
        included = included & ~new_outliers

    # Final fit with clean data
    obs_final = np.where(included, obs_v, 0.0)
    lib_dot_obs_final = np.sum(lib_col * obs_final, axis=0)
    lib_dot_lib_final = np.sum(np.where(included, lib_col**2, 0.0), axis=0)

    scales_final = np.where(lib_dot_lib_final > 0, lib_dot_obs_final / lib_dot_lib_final, 0.0)
    scales_final = np.maximum(scales_final, 0.0)

    # Compute abundances: scale * sum(library)
    abundances = scales_final * lib_sum

    # Compute R-squared per sample
    predicted_final = scales_final[None, :] * lib_col
    residuals_final = np.where(included, obs_v - predicted_final, 0.0)

    ss_res = np.sum(residuals_final**2, axis=0)
    n_included = np.sum(included, axis=0)
    obs_mean = np.sum(obs_final, axis=0) / np.maximum(n_included, 1)
    ss_tot = np.sum(np.where(included, (obs_v - obs_mean[None, :]) ** 2, 0.0), axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        r_squared = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)

    # Count fragments used per sample
    n_used = n_included.astype(np.int32)

    # Mark samples with too few fragments as NaN
    too_few = n_used < min_fragments
    abundances[too_few] = np.nan
    r_squared[too_few] = np.nan

    return abundances, r_squared, n_used


@dataclass
class LeastSquaresResult:
    """Result of least squares fitting for a single peptide/sample.

    The core equation solved is:
        observed = scale * library + residuals

    Where:
        scale = (library . observed) / (library . library)
        residuals = observed - scale * library

    The peptide abundance is: scale * sum(library)
    This represents the total signal attributable to the expected
    fragmentation pattern, robust to interference in individual fragments.

    Quality Assessment:
        - R² near 1.0: Good fit, observed pattern matches library
        - R² near 0 or negative: Poor fit, pattern doesn't match
        - is_reliable: True if R² >= min_r_squared threshold (default 0.5)

    """

    # Fitted scaling factor (least squares solution)
    scale: float

    # Peptide abundance (scale * sum of library intensities), always >= 0
    abundance: float

    # R-squared of the fit (goodness of fit)
    # Poor R² (< 0.5) indicates the observed pattern doesn't match the library
    r_squared: float

    # Number of fragments matched
    n_matched: int

    # Residual standard error
    residual_std: float

    # Indices of potential outlier fragments (high residuals)
    outlier_indices: list[int] = field(default_factory=list)

    # Whether the fit is considered reliable (R² above threshold)
    is_reliable: bool = True

    # Brief description if fit quality is poor
    quality_warning: str | None = None


def least_squares_rollup(
    observed_intensities: np.ndarray,
    library_intensities: np.ndarray,
    min_fragments: int = 2,
    outlier_threshold: float = 3.0,
    max_iterations: int = 5,
    remove_outliers: bool = True,
) -> LeastSquaresResult | None:
    """Compute peptide abundance via iterative least squares with outlier removal.

    Algorithm:
    1. Fit: observed = scale * library (closed-form least squares)
    2. Identify outliers (residuals > threshold × MAD, high only = interference)
    3. Exclude outliers and refit
    4. Repeat until stable or max iterations
    5. Return abundance = scale × sum(library)

    Key design decisions:
    - Zeros are VALID: A low-abundance peptide may only have signal in top 2-3
      fragments. If those match the most intense library peaks, that's good data.
    - Only HIGH residuals are outliers: signal > expected = interference.
      Low/zero signal is not interference, it's just low abundance.
    - Abundance from scaled library: We use scale × sum(library) so the
      quantification is derived from the library pattern, not raw sums.

    The closed-form solution: s* = (L · O) / (L · L)

    Args:
        observed_intensities: Observed fragment intensities (linear scale)
        library_intensities: Expected relative intensities from library (0-1)
        min_fragments: Minimum fragments required (default 2 since zeros are OK)
        outlier_threshold: MAD-scaled threshold for outlier detection (default 3.0)
        max_iterations: Maximum outlier removal iterations (default 5)
        remove_outliers: If True, iteratively remove outliers and refit

    Returns:
        LeastSquaresResult or None if insufficient data

    References:
        - Becker & Bern, 2002: Least squares isotope mass spectra
        - CHIMERYS (Frejno et al., 2025): Linear deconvolution of chimeric spectra

    """
    # Filter to valid fragments
    # - Observed can be >= 0 (zeros are valid - low abundance or no interference)
    # - Library must be > 0 (we need positive reference values)
    valid_mask = ~np.isnan(observed_intensities) & ~np.isnan(library_intensities)
    valid_mask &= observed_intensities >= 0  # Allow zeros
    valid_mask &= library_intensities > 0

    n_valid = np.sum(valid_mask)
    if n_valid < min_fragments:
        return None

    # Get valid data
    valid_indices = np.where(valid_mask)[0]
    obs = observed_intensities[valid_mask].copy()
    lib = library_intensities[valid_mask].copy()

    # Track which fragments are included (not outliers)
    included_mask = np.ones(len(obs), dtype=bool)
    all_outlier_indices = []

    # Iterative outlier removal
    for iteration in range(max_iterations):
        # Get currently included data
        obs_iter = obs[included_mask]
        lib_iter = lib[included_mask]

        n_included = np.sum(included_mask)
        if n_included < min_fragments:
            # Not enough fragments left - use previous iteration or fail
            if iteration == 0:
                return None
            break

        # Closed-form least squares solution: s* = (L · O) / (L · L)
        lib_dot_lib = np.dot(lib_iter, lib_iter)
        if lib_dot_lib <= 0:
            return None

        lib_dot_obs = np.dot(lib_iter, obs_iter)
        scale = lib_dot_obs / lib_dot_lib

        # Ensure non-negative scale
        if scale < 0:
            scale = 0.0

        if not remove_outliers:
            break

        # Compute residuals for ALL fragments (to detect outliers)
        predicted = scale * lib
        residuals = obs - predicted

        # Only HIGH residuals are outliers (interference = signal > expected)
        # Low or negative residuals are fine - just low abundance or noise
        positive_residuals = np.maximum(residuals, 0)

        # Use MAD of positive residuals for robust threshold
        if np.sum(positive_residuals > 0) > 2:
            # MAD of positive residuals only
            pos_res = positive_residuals[positive_residuals > 0]
            mad = np.median(pos_res)
            mad_std = mad * 1.4826 if mad > 0 else np.std(pos_res)

            if mad_std > 0:
                # Outliers are fragments with HIGH positive residuals
                z_scores = positive_residuals / mad_std
                new_outliers = (z_scores > outlier_threshold) & included_mask

                if not np.any(new_outliers):
                    # No new outliers - converged
                    break

                # Record outlier indices (in original array)
                for i in np.where(new_outliers)[0]:
                    orig_idx = valid_indices[i]
                    if orig_idx not in all_outlier_indices:
                        all_outlier_indices.append(int(orig_idx))

                # Exclude new outliers
                included_mask &= ~new_outliers
            else:
                break
        else:
            break

    # Final fit with clean data
    obs_final = obs[included_mask]
    lib_final = lib[included_mask]
    n_final = len(obs_final)

    if n_final < min_fragments:
        # Fall back to using all data if too many outliers removed
        obs_final = obs
        lib_final = lib
        n_final = len(obs_final)
        all_outlier_indices = []

    # Final scale calculation
    lib_dot_lib = np.dot(lib_final, lib_final)
    if lib_dot_lib <= 0:
        return None

    scale = np.dot(lib_final, obs_final) / lib_dot_lib
    if scale < 0:
        scale = 0.0

    # Compute R-squared on INCLUDED fragments
    predicted_final = scale * lib_final
    residuals_final = obs_final - predicted_final
    ss_res = np.sum(residuals_final**2)
    ss_tot = np.sum((obs_final - np.mean(obs_final)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Residual standard error
    residual_std = np.sqrt(ss_res / (n_final - 1)) if n_final > 1 else 0.0

    # Peptide abundance: scale × sum of ALL library intensities
    # This recovers the full signal based on the library pattern
    abundance = scale * np.sum(lib)

    # Assess fit quality
    # Poor R² suggests either: 1) severe interference, or 2) suspect ID
    min_r_squared = 0.5
    is_reliable = r_squared >= min_r_squared
    quality_warning = None
    if not is_reliable:
        quality_warning = "poor_fit"
    elif len(all_outlier_indices) > n_valid // 2:
        quality_warning = "many_outliers"

    return LeastSquaresResult(
        scale=scale,
        abundance=abundance,
        r_squared=r_squared,
        n_matched=int(n_final),
        residual_std=residual_std,
        outlier_indices=all_outlier_indices,
        is_reliable=is_reliable,
        quality_warning=quality_warning,
    )


# =============================================================================
# MS1 Isotope Distribution Functions
# =============================================================================

# Averagine elemental composition per Da (Senko et al., 1995)
# Average amino acid: C4.9384 H7.7583 N1.3577 O1.4773 S0.0417
AVERAGINE_C = 4.9384
AVERAGINE_H = 7.7583
AVERAGINE_N = 1.3577
AVERAGINE_O = 1.4773
AVERAGINE_S = 0.0417
AVERAGINE_MASS = 111.1254  # Average mass per residue

# Isotope abundances specific to biological molecules
# Source: D.E. Matthews (personal communication), as used in IDCalc
# Reference: MacCoss MJ, IDCalc: Isotope Distribution Calculator
#            https://github.com/maccoss/IDCalc
#
# Matthews values are used for 13C, 15N, 17O, 18O as they reflect
# the isotopic composition of biological samples.
# IUPAC natural abundance is used for 2H (deuterium).
#
# All values are relative to the monoisotopic peak = 100.
# The 13C abundance (1.0958793%) is the critical value for peptides.

# Relative abundances: M+n / M+0 * 100
BIOLOGICAL_ISOTOPE_ABUNDANCES = {
    # Carbon: 13C/12C (Matthews)
    "C": {
        "M0": 100.0,
        "M1": 1.0958793,  # 13C
    },
    # Hydrogen: 2H/1H (IUPAC natural abundance)
    "H": {
        "M0": 100.0,
        "M1": 0.0115,  # 2H (deuterium) - IUPAC value
    },
    # Oxygen: 17O, 18O relative to 16O (Matthews)
    "O": {
        "M0": 100.0,
        "M1": 0.03799194,  # 17O
        "M2": 0.20499609,  # 18O
    },
    # Nitrogen: 15N/14N (Matthews)
    "N": {
        "M0": 100.0,
        "M1": 0.368351851,  # 15N
    },
    # Sulfur: 33S, 34S, 36S relative to 32S
    "S": {
        "M0": 100.0,
        "M1": 0.789308,  # 33S
        "M2": 4.430646,  # 34S
        "M4": 0.021048,  # 36S
    },
}

# Convert to probability fractions for calculations
# p = abundance / (100 + abundance) for M+1 isotopes
# These are the probabilities of observing the heavier isotope
P_C13 = 1.0958793 / (100.0 + 1.0958793)  # ~0.01084 (Matthews)
P_H2 = 0.0115 / (100.0 + 0.0115)  # ~0.000115 (IUPAC)
P_N15 = 0.368351851 / (100.0 + 0.368351851)  # ~0.00367 (Matthews)
P_O17 = 0.03799194 / (100.0 + 0.03799194)  # ~0.000380 (Matthews)
P_O18 = 0.20499609 / (100.0 + 0.20499609)  # ~0.00205 (Matthews)
P_S33 = 0.789308 / (100.0 + 0.789308)  # ~0.00783
P_S34 = 4.430646 / (100.0 + 4.430646)  # ~0.04244


def predict_isotope_distribution(
    mass: float,
    n_isotopes: int = 5,
) -> np.ndarray:
    """Predict isotope distribution using averagine model with biological isotope abundances.

    Uses the averagine approximation (Senko et al., 1995) to estimate
    elemental composition from peptide mass, then calculates isotope
    distribution using biological isotope abundances from D.E. Matthews.

    The key difference from IUPAC abundances is 13C/12C = 1.0958793%
    (vs 1.07% natural), which better reflects biological samples.

    Args:
        mass: Monoisotopic mass of the peptide
        n_isotopes: Number of isotope peaks to calculate (M, M+1, M+2, ...)

    Returns:
        Array of relative intensities normalized to max = 1.0

    References:
        - Senko et al., 1995: Averagine model
        - Matthews, D.E.: Biological isotope abundances (personal communication)
        - MacCoss MJ, IDCalc: https://github.com/maccoss/IDCalc

    """
    # Estimate elemental composition from mass using averagine
    n_residues = mass / AVERAGINE_MASS

    n_c = int(round(n_residues * AVERAGINE_C))
    n_h = int(round(n_residues * AVERAGINE_H))
    n_n = int(round(n_residues * AVERAGINE_N))
    n_o = int(round(n_residues * AVERAGINE_O))
    n_s = int(round(n_residues * AVERAGINE_S))

    # Calculate expected number of heavy isotopes (lambda for Poisson approximation)
    # Using biological isotope abundances from Matthews/IDCalc
    #
    # For +1 Da contributions:
    #   13C (dominant), 15N, 17O, 33S, 2H
    # For +2 Da contributions (handled via convolution):
    #   18O, 34S

    # Expected count of +1 Da heavy isotopes
    lambda_1 = n_c * P_C13 + n_n * P_N15 + n_o * P_O17 + n_s * P_S33 + n_h * P_H2

    # For a more accurate calculation, we would convolve the +1 and +2 distributions
    # Here we use Poisson approximation which is accurate for typical peptide sizes

    from math import exp, factorial

    isotopes = np.zeros(n_isotopes)
    for k in range(n_isotopes):
        # Poisson probability for k heavy isotopes
        if lambda_1 > 0:
            isotopes[k] = (lambda_1**k) * exp(-lambda_1) / factorial(min(k, 20))
        else:
            isotopes[k] = 1.0 if k == 0 else 0.0

    # Normalize to max = 1.0
    if isotopes.max() > 0:
        isotopes = isotopes / isotopes.max()

    return isotopes


def correct_ms1_isotopes(
    observed_intensities: np.ndarray,
    expected_distribution: np.ndarray,
    min_intensity_fraction: float = 0.5,
) -> tuple[float, np.ndarray, list[int]]:
    """Correct MS1 isotope intensities for interference using Scribe method.

    For each isotope peak with expected intensity >50% of max, calculate
    the acquired/expected ratio. Use the minimum ratio (cleanest isotope)
    as the true scale, and trim excess intensity from interfered isotopes.

    Reference: Searle et al., Scribe method

    Args:
        observed_intensities: Observed MS1 isotope peak intensities (M, M+1, M+2, ...)
        expected_distribution: Predicted relative isotope intensities (normalized to max=1)
        min_intensity_fraction: Only consider isotopes with expected intensity
                                >= this fraction of max (default 0.5)

    Returns:
        Tuple of:
        - corrected_total: Total intensity after correction (scale * sum(expected))
        - corrected_intensities: Individual corrected peak intensities
        - interfered_indices: Indices of isotopes that had excess intensity removed

    """
    n_peaks = min(len(observed_intensities), len(expected_distribution))
    if n_peaks == 0:
        return 0.0, np.array([]), []

    obs = observed_intensities[:n_peaks]
    exp = expected_distribution[:n_peaks]

    # Find major isotopes (>50% of max expected intensity)
    major_mask = exp >= min_intensity_fraction * exp.max()

    if not np.any(major_mask):
        # No major isotopes, can't correct
        return float(np.sum(obs)), obs.copy(), []

    # Calculate acquired/expected ratio for major isotopes
    # Avoid division by zero
    ratios = np.zeros(n_peaks)
    for i in range(n_peaks):
        if major_mask[i] and exp[i] > 0:
            ratios[i] = obs[i] / exp[i]
        else:
            ratios[i] = np.inf  # Ignore non-major isotopes

    # Use minimum ratio as the "true" scale (least interfered)
    major_ratios = ratios[major_mask & (ratios < np.inf)]
    if len(major_ratios) == 0:
        return float(np.sum(obs)), obs.copy(), []

    true_scale = np.min(major_ratios)

    # Correct intensities: cap each at scale * expected
    corrected = np.zeros(n_peaks)
    interfered = []

    for i in range(n_peaks):
        expected_intensity = true_scale * exp[i]
        if obs[i] > expected_intensity * 1.01:  # Allow 1% tolerance
            corrected[i] = expected_intensity
            if major_mask[i]:
                interfered.append(i)
        else:
            corrected[i] = obs[i]

    # Total corrected intensity
    corrected_total = true_scale * np.sum(exp)

    return corrected_total, corrected, interfered


@dataclass
class MS1CorrectionResult:
    """Result of MS1 isotope correction."""

    # Corrected total abundance
    abundance: float

    # Scale factor (minimum ratio)
    scale: float

    # Number of isotope peaks used
    n_isotopes: int

    # Indices of interfered isotopes
    interfered_indices: list[int]

    # Whether correction was applied (had interference)
    was_corrected: bool


def ms1_isotope_rollup(
    observed_isotopes: np.ndarray,
    precursor_mz: float,
    charge: int,
    n_isotopes: int = 5,
    min_intensity_fraction: float = 0.5,
) -> MS1CorrectionResult | None:
    """Calculate MS1 abundance with isotope interference correction.

    Uses the Scribe method (Searle et al.) to correct for isotope interference:
    1. Predict expected isotope distribution from mass
    2. Calculate acquired/expected ratio for major isotopes
    3. Use minimum ratio as true scale (cleanest isotope)
    4. Return corrected abundance = scale * sum(expected)

    Args:
        observed_isotopes: Observed intensities for M, M+1, M+2, etc.
        precursor_mz: Precursor m/z
        charge: Precursor charge state
        n_isotopes: Number of isotope peaks to use
        min_intensity_fraction: Threshold for major isotopes (default 0.5)

    Returns:
        MS1CorrectionResult or None if insufficient data

    """
    if len(observed_isotopes) < 2:
        return None

    # Calculate monoisotopic mass from m/z and charge
    proton_mass = 1.007276
    mass = (precursor_mz - proton_mass) * charge

    # Predict isotope distribution
    expected = predict_isotope_distribution(mass, n_isotopes=n_isotopes)

    # Correct for interference
    corrected_total, corrected, interfered = correct_ms1_isotopes(
        observed_isotopes,
        expected,
        min_intensity_fraction=min_intensity_fraction,
    )

    # Calculate scale
    exp_sum = np.sum(expected[: len(observed_isotopes)])
    scale = corrected_total / exp_sum if exp_sum > 0 else 0.0

    return MS1CorrectionResult(
        abundance=corrected_total,
        scale=scale,
        n_isotopes=min(len(observed_isotopes), len(expected)),
        interfered_indices=interfered,
        was_corrected=len(interfered) > 0,
    )


def library_assisted_rollup_peptide(
    transitions_df: pd.DataFrame,
    library_spectrum: FragmentSpectrum,
    sample_cols: list[str],
    fragment_col: str = "Fragment Ion",
    product_mz_col: str = "Product Mz",
    min_fragments: int = 2,
    mz_tolerance: float = 0.02,
    use_robust: bool = True,
    use_ms1: bool = False,
    ms1_isotopes_df: pd.DataFrame | None = None,
    precursor_mz: float | None = None,
    precursor_charge: int | None = None,
) -> dict[str, float]:
    """Compute peptide abundances across samples using library-assisted least squares.

    For each sample, this function:
    1. Matches observed transitions to library fragments
    2. Builds intensity vectors (observed and expected)
    3. Fits: observed = scale * expected via least squares
    4. Iteratively removes outliers (high residuals = interference)
    5. Returns peptide abundance = scale * sum(expected)
    6. If use_ms1=True, also fits MS1 isotopes and adds to abundance

    Key design principles:
    - Zeros are valid: Low-abundance peptides may only have 2-3 fragments with signal.
      If those match the top library peaks, that's good data.
    - Only HIGH residuals are outliers: signal > expected = interference.
    - Different fragments can be outliers in different samples.
    - Abundance = scale × sum(library) gives consistent quantification.
    - Poor R² across all samples flags potential suspect IDs.

    Args:
        transitions_df: DataFrame with one row per transition for a single peptide
        library_spectrum: Reference spectrum from spectral library
        sample_cols: List of sample column names to process
        fragment_col: Column with fragment ion annotation (default: "Fragment Ion")
        product_mz_col: Column with product m/z (default: "Product Mz")
        min_fragments: Minimum matched fragments required (default 2 since zeros are OK)
        mz_tolerance: m/z tolerance for library matching (Da)
        use_robust: Use iterative outlier removal for robustness (default True)
        use_ms1: If True, also include MS1 isotope contribution
        ms1_isotopes_df: DataFrame with MS1 isotope intensities (M, M+1, M+2, ...)
        precursor_mz: Precursor m/z (required if use_ms1=True)
        precursor_charge: Precursor charge (required if use_ms1=True)

    Returns:
        Dict mapping sample column name to peptide abundance (or NaN if insufficient data)

    """
    result = {}

    # Match transitions to library
    n_transitions = len(transitions_df)
    library_intensities = np.zeros(n_transitions)
    matched_mask = np.zeros(n_transitions, dtype=bool)

    for i, (_, row) in enumerate(transitions_df.iterrows()):
        fragment_ion = str(row.get(fragment_col, ""))
        product_mz = float(row.get(product_mz_col, 0.0))

        lib_intensity = match_transition_to_library(
            fragment_ion=fragment_ion,
            product_mz=product_mz,
            library_spectrum=library_spectrum,
            mz_tolerance=mz_tolerance,
        )

        if lib_intensity is not None:
            library_intensities[i] = lib_intensity
            matched_mask[i] = True

    n_matched = np.sum(matched_mask)
    if n_matched < min_fragments:
        # Return NaN for all samples if insufficient library matches
        return {col: np.nan for col in sample_cols}

    # Build observation matrix (T, S) for vectorized processing
    # This processes ALL samples in parallel using BLAS operations
    observed_matrix = transitions_df[sample_cols].values.astype(float).T  # (S, T)
    observed_matrix = observed_matrix.T  # (T, S)

    # Mask unmatched transitions (set to NaN)
    lib_matched = library_intensities.copy()
    lib_matched[~matched_mask] = np.nan
    observed_matrix[~matched_mask, :] = np.nan

    # Vectorized least squares fit for ALL samples at once
    abundances, r_squared, n_used = least_squares_rollup_vectorized(
        observed_matrix=observed_matrix,
        library_intensities=lib_matched,
        min_fragments=min_fragments,
        outlier_threshold=3.0,
        remove_outliers=use_robust,
    )

    # Handle MS1 isotope contribution if enabled
    if use_ms1 and ms1_isotopes_df is not None and precursor_mz and precursor_charge:
        for i, sample_col in enumerate(sample_cols):
            if sample_col in ms1_isotopes_df.columns and not np.isnan(abundances[i]):
                observed_isotopes = ms1_isotopes_df[sample_col].values.astype(float)
                ms1_result = ms1_isotope_rollup(
                    observed_isotopes=observed_isotopes,
                    precursor_mz=precursor_mz,
                    charge=precursor_charge,
                )
                if ms1_result is not None:
                    abundances[i] += ms1_result.abundance

    # Build result dict
    result = {col: abundances[i] for i, col in enumerate(sample_cols)}
    return result


class SpectralLibraryRollup:
    """Library-assisted transition-to-peptide rollup using least squares fitting.

    This class manages spectral library loading and provides efficient rollup
    for large datasets by caching library spectra.

    Usage:
        rollup = SpectralLibraryRollup(library_path)
        rollup.load_library()

        # For each peptide group:
        abundances = rollup.rollup_peptide(
            transitions_df,
            modified_sequence,
            charge,
            sample_cols,
        )

    """

    def __init__(
        self,
        library_path: Path | str,
        min_fragments: int = 3,
        mz_tolerance: float = 0.02,
        use_robust: bool = True,
    ):
        """Initialize library-assisted rollup.

        Args:
            library_path: Path to spectral library (.tsv or .blib)
            min_fragments: Minimum matched fragments for fitting
            mz_tolerance: m/z tolerance for library matching (Da)
            use_robust: Use iteratively reweighted least squares

        """
        self.library_path = Path(library_path)
        self.min_fragments = min_fragments
        self.mz_tolerance = mz_tolerance
        self.use_robust = use_robust
        self.library: dict[str, FragmentSpectrum] = {}
        self._stripped_lookup: dict[str, str] = {}  # stripped_key -> original_key
        self._loaded = False

        # Statistics
        self.n_matched = 0
        self.n_unmatched = 0

    def load_library(self) -> None:
        """Load spectral library from file."""
        self.library = load_spectral_library(self.library_path)

        # Build stripped-key lookup for fallback matching when modification
        # notation differs (e.g., unimod:35 vs [+15.99491])
        self._stripped_lookup = {}
        for key, spectrum in self.library.items():
            stripped_key = SpectralLibraryLoader.make_stripped_key(
                spectrum.modified_sequence, spectrum.precursor_charge
            )
            # Also normalize I/L
            stripped_key_normalized = stripped_key.replace("L", "I")
            # Store mapping (first one wins if duplicates)
            if stripped_key not in self._stripped_lookup:
                self._stripped_lookup[stripped_key] = key
            if stripped_key_normalized not in self._stripped_lookup:
                self._stripped_lookup[stripped_key_normalized] = key

        self._loaded = True
        logger.info(f"SpectralLibraryRollup: loaded {len(self.library)} spectra")

    def get_spectrum(
        self,
        modified_sequence: str,
        charge: int,
    ) -> FragmentSpectrum | None:
        """Get library spectrum for a peptide.

        Handles sequence normalization (I/L ambiguity) and modification format
        differences (e.g., unimod:35 vs [+15.99491]).

        Args:
            modified_sequence: Modified peptide sequence
            charge: Precursor charge state

        Returns:
            FragmentSpectrum or None if not in library

        """
        if not self._loaded:
            self.load_library()

        # Try exact match first
        key = SpectralLibraryLoader.make_peptide_key(modified_sequence, charge)
        if key in self.library:
            return self.library[key]

        # Try with I/L normalization
        normalized = SpectralLibraryLoader.normalize_sequence_for_matching(modified_sequence)
        normalized_key = SpectralLibraryLoader.make_peptide_key(normalized, charge)
        if normalized_key in self.library:
            return self.library[normalized_key]

        # Try stripped sequence fallback (for different modification notations)
        # This matches "M(unimod:35)" format to "M[+15.99491]" format
        stripped_key = SpectralLibraryLoader.make_stripped_key(modified_sequence, charge)
        if stripped_key in self._stripped_lookup:
            original_key = self._stripped_lookup[stripped_key]
            return self.library[original_key]

        # Try stripped + I/L normalized
        stripped_normalized = stripped_key.replace("L", "I")
        if stripped_normalized in self._stripped_lookup:
            original_key = self._stripped_lookup[stripped_normalized]
            return self.library[original_key]

        return None

    def rollup_peptide(
        self,
        transitions_df: pd.DataFrame,
        modified_sequence: str,
        charge: int,
        sample_cols: list[str],
        fragment_col: str = "Fragment Ion",
        product_mz_col: str = "Product Mz",
    ) -> dict[str, float]:
        """Compute peptide abundances using library-assisted least squares.

        Args:
            transitions_df: DataFrame with one row per transition
            modified_sequence: Modified peptide sequence
            charge: Precursor charge state
            sample_cols: Sample column names to process
            fragment_col: Column with fragment ion annotation
            product_mz_col: Column with product m/z

        Returns:
            Dict mapping sample column name to peptide abundance

        """
        # Get library spectrum
        spectrum = self.get_spectrum(modified_sequence, charge)

        if spectrum is None:
            self.n_unmatched += 1
            # Fall back to simple sum if no library match
            result = {}
            for col in sample_cols:
                # Simple sum fallback
                values = transitions_df[col].values.astype(float)
                valid = values[~np.isnan(values) & (values > 0)]
                result[col] = np.sum(valid) if len(valid) > 0 else np.nan
            return result

        self.n_matched += 1

        # Use library-assisted rollup
        return library_assisted_rollup_peptide(
            transitions_df=transitions_df,
            library_spectrum=spectrum,
            sample_cols=sample_cols,
            fragment_col=fragment_col,
            product_mz_col=product_mz_col,
            min_fragments=self.min_fragments,
            mz_tolerance=self.mz_tolerance,
            use_robust=self.use_robust,
        )

    def get_statistics(self) -> dict[str, int]:
        """Get rollup statistics.

        Returns:
            Dict with n_matched, n_unmatched, library_size

        """
        return {
            "n_matched": self.n_matched,
            "n_unmatched": self.n_unmatched,
            "library_size": len(self.library),
            "match_rate": (
                self.n_matched / (self.n_matched + self.n_unmatched)
                if (self.n_matched + self.n_unmatched) > 0
                else 0.0
            ),
        }

    def rollup_peptide_with_r2(
        self,
        transitions_df: pd.DataFrame,
        modified_sequence: str,
        charge: int,
        sample_cols: list[str],
        fragment_col: str = "Fragment Ion",
        product_mz_col: str = "Product Mz",
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Compute peptide abundances AND R² values using library-assisted least squares.

        This version returns both the abundance and the fit quality (R²) for each sample,
        which can be used to evaluate which charge state has the best correlation with
        the library spectrum.

        Args:
            transitions_df: DataFrame with one row per transition
            modified_sequence: Modified peptide sequence
            charge: Precursor charge state
            sample_cols: Sample column names to process
            fragment_col: Column with fragment ion annotation
            product_mz_col: Column with product m/z

        Returns:
            Tuple of:
            - Dict mapping sample column name to peptide abundance
            - Dict mapping sample column name to R² value (fit quality)

        """
        # Get library spectrum
        spectrum = self.get_spectrum(modified_sequence, charge)

        abundances: dict[str, float] = {}
        r2_values: dict[str, float] = {}

        if spectrum is None:
            self.n_unmatched += 1
            # Fall back to simple sum if no library match
            for col in sample_cols:
                values = transitions_df[col].values.astype(float)
                valid = values[~np.isnan(values) & (values > 0)]
                abundances[col] = np.sum(valid) if len(valid) > 0 else np.nan
                r2_values[col] = np.nan  # No R² for simple sum
            return abundances, r2_values

        self.n_matched += 1

        # Match transitions to library
        n_transitions = len(transitions_df)
        library_intensities = np.zeros(n_transitions)
        matched_mask = np.zeros(n_transitions, dtype=bool)

        for i, (_, row) in enumerate(transitions_df.iterrows()):
            fragment_ion = str(row.get(fragment_col, ""))
            product_mz = float(row.get(product_mz_col, 0.0))

            lib_intensity = match_transition_to_library(
                fragment_ion=fragment_ion,
                product_mz=product_mz,
                library_spectrum=spectrum,
                mz_tolerance=self.mz_tolerance,
            )

            if lib_intensity is not None:
                library_intensities[i] = lib_intensity
                matched_mask[i] = True

        n_matched = np.sum(matched_mask)
        if n_matched < self.min_fragments:
            # Return NaN for all samples if insufficient library matches
            return {col: np.nan for col in sample_cols}, {col: np.nan for col in sample_cols}

        # Process each sample
        for sample_col in sample_cols:
            observed = transitions_df[sample_col].values.astype(float)

            obs_matched = observed.copy()
            lib_matched = library_intensities.copy()
            obs_matched[~matched_mask] = np.nan
            lib_matched[~matched_mask] = np.nan

            # Fit least squares (with or without outlier removal based on use_robust)
            fit_result = least_squares_rollup(
                observed_intensities=obs_matched,
                library_intensities=lib_matched,
                min_fragments=self.min_fragments,
                remove_outliers=self.use_robust,
            )

            if fit_result is not None:
                abundances[sample_col] = fit_result.abundance
                r2_values[sample_col] = fit_result.r_squared
            else:
                abundances[sample_col] = np.nan
                r2_values[sample_col] = np.nan

        return abundances, r2_values
