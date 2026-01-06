#!/usr/bin/env python3
"""Analyze consensus rollup to identify problematic transitions.

This standalone script processes the merged parquet file in chunks to identify
transitions with high residual variance (most down-weighted by consensus rollup).

Run AFTER a successful PRISM run with consensus method:

    python analyze_consensus_rollup.py \\
        -p prism-output/merged_data.parquet \\
        -o consensus_diagnostics.csv \\
        --top 1000

Column definitions:
- peptide_col: Column with peptide sequence (auto-detected)
- transition_col: Column with fragment ion identifier
- sample_col: Column with sample names
- abundance_col: Column with intensity values

Output CSV columns:
- peptide, transition, sample: identifiers
- residual: log2 deviation from expected
- abs_residual: absolute value (for sorting)
- transition_variance: variance of this transition's residuals
- transition_weight: inverse-variance weight (lower = more down-weighted)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def analyze_peptide_consensus(
    pep_data: pd.DataFrame,
    peptide_name: str,
    transition_col: str,
    sample_col: str,
    abundance_col: str,
    top_n_per_peptide: int = 3,
    regularization: float = 0.1,
) -> list[dict]:
    """Compute consensus residuals for one peptide, return top N problematic entries."""

    # Pivot to transition × sample matrix
    try:
        intensity_matrix = pep_data.pivot_table(
            index=transition_col,
            columns=sample_col,
            values=abundance_col,
            aggfunc="first",
        )
    except Exception:
        return []

    if len(intensity_matrix) < 3:
        return []

    # Log2 transform (handle zeros/negatives)
    matrix = np.log2(intensity_matrix.clip(lower=1).values)

    # Step 1: Transition offsets (row medians)
    row_medians = np.nanmedian(matrix, axis=1, keepdims=True)

    # Step 2: Sample effects (column medians of centered)
    centered = matrix - row_medians
    col_medians = np.nanmedian(centered, axis=0, keepdims=True)

    # Step 3: Residuals
    residuals = matrix - row_medians - col_medians

    # Step 4: Per-transition variance
    with np.errstate(invalid="ignore"):
        trans_variances = np.nanvar(residuals, axis=1)
    trans_variances = np.nan_to_num(trans_variances, nan=np.inf)

    # Step 5: Weights
    weights = 1.0 / (trans_variances + regularization)
    weight_sum = np.nansum(weights)
    if np.isfinite(weight_sum) and weight_sum > 0:
        weights = weights * (len(intensity_matrix) / weight_sum)

    # Collect all entries
    entries = []
    for i, trans in enumerate(intensity_matrix.index):
        for j, sample in enumerate(intensity_matrix.columns):
            resid = residuals[i, j]
            if np.isfinite(resid):
                entries.append(
                    {
                        "peptide": peptide_name,
                        "transition": trans,
                        "sample": sample,
                        "residual": float(resid),
                        "abs_residual": float(abs(resid)),
                        "transition_variance": float(trans_variances[i]),
                        "transition_weight": float(weights[i]),
                    }
                )

    # Return top N by abs_residual
    entries.sort(key=lambda x: x["abs_residual"], reverse=True)
    return entries[:top_n_per_peptide]


def main():
    parser = argparse.ArgumentParser(
        description="Analyze consensus rollup to find problematic transitions"
    )
    parser.add_argument(
        "-p",
        "--parquet",
        required=True,
        help="Path to merged_data.parquet from PRISM output",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="consensus_diagnostics.csv",
        help="Output CSV file (default: consensus_diagnostics.csv)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=1000,
        help="Total top problematic entries to output (default: 1000)",
    )
    parser.add_argument(
        "--top-per-peptide",
        type=int,
        default=3,
        help="Top entries to keep per peptide (default: 3)",
    )
    parser.add_argument(
        "--peptide-col",
        help="Column for peptide (auto-detected if not specified)",
    )
    parser.add_argument(
        "--transition-col",
        default="Fragment Ion",
        help="Column for transition identifier",
    )
    parser.add_argument(
        "--sample-col",
        default="Sample ID",
        help="Column for sample identifier",
    )
    parser.add_argument(
        "--abundance-col",
        default="Area",
        help="Column for intensity values",
    )

    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        logger.error(f"Parquet file not found: {parquet_path}")
        sys.exit(1)

    logger.info(f"Reading parquet: {parquet_path}")
    pf = pq.ParquetFile(parquet_path)

    # Auto-detect peptide column
    schema_names = pf.schema_arrow.names
    peptide_col = args.peptide_col
    if not peptide_col:
        for col in [
            "Peptide Modified Sequence Unimod Ids",
            "Peptide Modified Sequence",
            "peptide_modified",
        ]:
            if col in schema_names:
                peptide_col = col
                break
    if not peptide_col:
        logger.error("Could not auto-detect peptide column. Use --peptide-col")
        sys.exit(1)

    logger.info(f"  Peptide col: {peptide_col}")
    logger.info(f"  Transition col: {args.transition_col}")
    logger.info(f"  Sample col: {args.sample_col}")

    # Get required columns
    required_cols = [peptide_col, args.transition_col, args.sample_col, args.abundance_col]
    cols_to_read = [c for c in required_cols if c in schema_names]

    # Stream through row groups and process by peptide
    all_results = []
    current_peptide = None
    current_data = []
    n_peptides = 0

    logger.info("Streaming analysis...")

    for i in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(i, columns=cols_to_read)
        df = table.to_pandas()

        for _, row in df.iterrows():
            peptide = row[peptide_col]

            if peptide != current_peptide:
                # Process previous peptide
                if current_data:
                    pep_df = pd.DataFrame(current_data)
                    results = analyze_peptide_consensus(
                        pep_df,
                        current_peptide,
                        args.transition_col,
                        args.sample_col,
                        args.abundance_col,
                        args.top_per_peptide,
                    )
                    all_results.extend(results)
                    n_peptides += 1

                    if n_peptides % 5000 == 0:
                        logger.info(
                            f"  Processed {n_peptides} peptides, collected {len(all_results)} entries..."
                        )

                current_peptide = peptide
                current_data = []

            current_data.append(row.to_dict())

    # Process final peptide
    if current_data:
        pep_df = pd.DataFrame(current_data)
        results = analyze_peptide_consensus(
            pep_df,
            current_peptide,
            args.transition_col,
            args.sample_col,
            args.abundance_col,
            args.top_per_peptide,
        )
        all_results.extend(results)
        n_peptides += 1

    logger.info(f"  Completed: {n_peptides} peptides, {len(all_results)} candidate entries")

    if not all_results:
        logger.warning("No results to output")
        sys.exit(0)

    # Sort all results and take top N
    all_results.sort(key=lambda x: x["abs_residual"], reverse=True)
    final_results = all_results[: args.top]

    # Save
    output_df = pd.DataFrame(final_results)
    output_df.to_csv(args.output, index=False)
    logger.info(f"Saved {len(final_results)} top problematic entries to {args.output}")

    # Print preview
    print("\n" + "=" * 70)
    print("TOP PROBLEMATIC TRANSITIONS (largest residuals)")
    print("=" * 70)
    print(output_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
