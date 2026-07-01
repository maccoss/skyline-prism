namespace SkylinePrism.Core.Config;

/// <summary>
/// Emits an annotated PRISM configuration template. (The Python `prism config-template`
/// output is the authoritative reference; byte-for-byte matching is a later refinement.)
/// </summary>
public static class ConfigTemplate
{
    public static string Default() => """
# PRISM Configuration Template (C# port)
# =====================================
# Usage: prism run -i data.csv -o output/ -c this_config.yaml

# Sample type detection (regex-free substring patterns against the sample/replicate name)
sample_annotations:
  reference_pattern:
    - "-Pool_"
  qc_pattern:
    - "-QC_"

# Protein parsimony. Set fasta_path for FASTA-based mapping; null uses the Skyline
# Protein Accession column.
parsimony:
  fasta_path: null
  shared_peptide_handling: "all_groups"

# Transition -> peptide rollup: "sum" or "median_polish".
transition_rollup:
  method: "median_polish"
  min_transitions: 3
  use_ms1: false

# Peptide global normalization: "median" or "none".
global_normalization:
  method: "median"

# ComBat batch correction (skipped automatically when < 2 batches).
batch_correction:
  enabled: true
  method: "combat"
  reference_anchored: false
  reference_type: "reference"

# Peptide -> protein rollup: "median_polish" or "sum".
protein_rollup:
  method: "median_polish"
  min_peptides: 2

# Protein global normalization: "median" or "none".
protein_normalization:
  method: "median"

# Output.
output:
  format: "parquet"
  include_residuals: false

# QC report (generation arrives with the C# QC layer).
qc_report:
  enabled: false

""";
}
