#!/usr/bin/env python3
"""Comparison test for different transition and protein rollup methods."""
import json
import re
import subprocess
from pathlib import Path

# Paths - use existing parquet from previous run
BASE_DIR = Path("/home/maccoss/GitHub-Repo/maccoss/skyline-prism/example-files")
INPUT_FILE = BASE_DIR / "out_test_sum" / "merged_data.parquet"
METADATA_FILE = BASE_DIR / "out_test_sum" / "sample_metadata.tsv"
FASTA_FILE = BASE_DIR / "uniprot_mouse_may2025_contam_yeastENO1.fasta"
OUTPUT_BASE = BASE_DIR / "out_compare"

# Configurations to test
# (name, transition_method, topn_count, topn_selection, topn_weighting,
#  protein_method, protein_topn)
CONFIGS = [
    # Sum transitions (best so far)
    ("sum_medpol", "sum", None, None, None, "median_polish", None),
    ("sum_protsum", "sum", None, None, None, "sum", None),
    ("sum_prot3", "sum", None, None, None, "topn", 3),
    # Top 3 by correlation
    ("top3_corr_medpol", "topn", 3, "correlation", "sum", "median_polish", None),
    ("top3_corr_protsum", "topn", 3, "correlation", "sum", "sum", None),
    ("top3_corr_prot3", "topn", 3, "correlation", "sum", "topn", 3),
    # Top 3 by intensity
    ("top3_int_medpol", "topn", 3, "intensity", "sum", "median_polish", None),
    ("top3_int_protsum", "topn", 3, "intensity", "sum", "sum", None),
    ("top3_int_prot3", "topn", 3, "intensity", "sum", "topn", 3),
    # Top 4 by correlation
    ("top4_corr_medpol", "topn", 4, "correlation", "sum", "median_polish", None),
    ("top4_corr_protsum", "topn", 4, "correlation", "sum", "sum", None),
    ("top4_corr_prot3", "topn", 4, "correlation", "sum", "topn", 3),
    # Top 4 by intensity
    ("top4_int_medpol", "topn", 4, "intensity", "sum", "median_polish", None),
    ("top4_int_protsum", "topn", 4, "intensity", "sum", "sum", None),
    ("top4_int_prot3", "topn", 4, "intensity", "sum", "topn", 3),
]


def create_config_file(
    output_dir: Path,
    transition_method: str,
    topn_count: int | None,
    topn_selection: str | None,
    topn_weighting: str | None,
    protein_method: str,
    protein_topn: int | None,
) -> Path:
    """Create a YAML config file for this run."""
    config_content = f"""# Auto-generated config for comparison test
transition_rollup:
  method: "{transition_method}"
  topn_count: {topn_count or 3}
  topn_selection: "{topn_selection or 'correlation'}"
  topn_weighting: "{topn_weighting or 'sum'}"
  min_transitions: 3
  use_ms1: false

protein_rollup:
  method: "{protein_method}"
  topn_n: {protein_topn or 3}
  topn_selection: "median_abundance"
  min_peptides: 3  # For confidence marking; median_polish needs 3+, otherwise falls back to sum
  shared_peptide_handling: "all_groups"

global_normalization:
  method: "median"

batch_correction:
  enabled: true
  method: "combat"

parsimony:
  fasta_path: "{FASTA_FILE}"
  shared_peptide_handling: "all_groups"

output:
  include_residuals: false
"""
    config_path = output_dir / "config.yaml"
    config_path.write_text(config_content)
    return config_path


def run_config(
    name: str,
    transition_method: str,
    topn_count: int | None,
    topn_selection: str | None,
    topn_weighting: str | None,
    protein_method: str,
    protein_topn: int | None,
) -> dict:
    """Run a single configuration and extract results."""
    output_dir = OUTPUT_BASE / name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create config file
    config_path = create_config_file(
        output_dir,
        transition_method,
        topn_count,
        topn_selection,
        topn_weighting,
        protein_method,
        protein_topn,
    )

    # Build command
    cmd = [
        "prism", "run",
        "-i", str(INPUT_FILE),
        "-o", str(output_dir),
        "-m", str(METADATA_FILE),
        "-c", str(config_path),
    ]

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Config: {config_path}")
    print(f"{'='*60}")

    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse output for CV values
    output = result.stdout + result.stderr

    # Look for CV values in output
    peptide_cv = None
    protein_cv = None

    # Parse patterns like "Reference CV (median): XX.X%"
    pep_match = re.search(
        r"Peptide.*Reference CV.*?(\d+\.?\d*)%", output, re.IGNORECASE
    )
    if pep_match:
        peptide_cv = float(pep_match.group(1))

    prot_match = re.search(
        r"Protein.*Reference CV.*?(\d+\.?\d*)%", output, re.IGNORECASE
    )
    if prot_match:
        protein_cv = float(prot_match.group(1))

    # Alternative parsing - look for the final summary
    if peptide_cv is None:
        match = re.search(
            r"Reference median CV: (\d+\.?\d*)%.*peptide", output, re.IGNORECASE
        )
        if match:
            peptide_cv = float(match.group(1))

    if protein_cv is None:
        match = re.search(
            r"Reference median CV: (\d+\.?\d*)%.*protein", output, re.IGNORECASE
        )
        if match:
            protein_cv = float(match.group(1))

    # Try to read from metadata.json if not found in output
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
                if "qc_metrics" in metadata:
                    qc = metadata["qc_metrics"]
                    if peptide_cv is None and "peptide_reference_cv" in qc:
                        peptide_cv = qc["peptide_reference_cv"]
                    if protein_cv is None and "protein_reference_cv" in qc:
                        protein_cv = qc["protein_reference_cv"]
        except Exception as e:
            print(f"Warning: Could not read metadata: {e}")

    return {
        "name": name,
        "transition_method": transition_method,
        "topn_count": topn_count,
        "topn_selection": topn_selection,
        "protein_method": protein_method,
        "protein_topn": protein_topn,
        "peptide_cv": peptide_cv,
        "protein_cv": protein_cv,
        "success": result.returncode == 0,
    }


def main():
    """Run all configurations and summarize results."""
    # Verify input paths exist
    print(f"Base directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_BASE}")
    print(f"Input file: {INPUT_FILE}")
    print(f"Metadata file: {METADATA_FILE}")
    print(f"FASTA file: {FASTA_FILE}")

    if not INPUT_FILE.exists():
        print(f"\nERROR: Input file not found: {INPUT_FILE}")
        return

    if not METADATA_FILE.exists():
        print(f"\nERROR: Metadata file not found: {METADATA_FILE}")
        return

    if not FASTA_FILE.exists():
        print(f"\nWARNING: FASTA file not found: {FASTA_FILE}")
        print("Will try to continue without it...")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    results = []

    for config in CONFIGS:
        try:
            result = run_config(*config)
            results.append(result)
            print(f"  Peptide CV: {result['peptide_cv']}%")
            print(f"  Protein CV: {result['protein_cv']}%")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "name": config[0],
                "error": str(e),
                "success": False,
            })

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)
    header = (
        f"{'Configuration':<25} {'Transition':<12} {'Protein':<12} "
        f"{'Pep CV':<10} {'Prot CV':<10}"
    )
    print(header)
    print("-" * 80)

    for r in results:
        if r.get("success"):
            trans_desc = r["transition_method"]
            if r.get("topn_count"):
                trans_desc = f"top{r['topn_count']}_{r['topn_selection'][:3]}"
            prot_desc = r["protein_method"]
            if r.get("protein_topn"):
                prot_desc = f"top{r['protein_topn']}"

            pep_cv = f"{r['peptide_cv']:.1f}%" if r.get("peptide_cv") else "N/A"
            prot_cv = f"{r['protein_cv']:.1f}%" if r.get("protein_cv") else "N/A"

            print(f"{r['name']:<25} {trans_desc:<12} {prot_desc:<12} "
                  f"{pep_cv:<10} {prot_cv:<10}")
        else:
            print(f"{r['name']:<25} FAILED")

    # Save results to JSON
    results_file = OUTPUT_BASE / "comparison_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
