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

# Transition -> peptide rollup: "sum", "median_polish", or "library_assist".
transition_rollup:
  method: "median_polish"
  min_transitions: 3
  use_ms1: false
  # For method: library_assist (Skyline .blib):
  # library_path: null
  # library_min_fragments: 3
  # library_mz_tolerance: 0.02
  # library_outlier_threshold: 1.0

# Peptide normalization: "rt_lowess" (default), "median", "quantile", "vsn", or "none".
global_normalization:
  method: "rt_lowess"

# Low-signal sample outlier detection. action: "report" or "exclude".
sample_outlier_detection:
  enabled: true
  action: "report"
  method: "iqr"          # or "fold_median"

# ComBat batch correction (skipped automatically when < 2 batches).
batch_correction:
  enabled: true
  peptide_level: true
  protein_level: true
  method: "combat"

# Peptide -> protein rollup: "median_polish" or "sum".
protein_rollup:
  method: "median_polish"
  min_peptides: 3

# Protein normalization: "median" or "none".
protein_normalization:
  method: "median"

# Parallelism / memory. n_workers: 0 = all cores, 1 = serial, N = cap at N.
processing:
  n_workers: 0
  peptide_batch_size: 2000

# Output.
output:
  format: "parquet"
  include_residuals: false

# QC report (self-contained HTML + before/after plots).
qc_report:
  enabled: true

""";
}
