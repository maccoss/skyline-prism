namespace SkylinePrism.Core.Config;

/// <summary>
/// Emits an annotated PRISM configuration template. (The Python `prism config-template`
/// output is the authoritative reference; byte-for-byte matching is a later refinement.)
/// </summary>
public static class ConfigTemplate
{
    public static string Default() => """
# PRISM Configuration Template
# =====================================
# Usage: prism run -i data.csv -o output/ -c this_config.yaml

# Input column mapping. Auto-detection handles both the invariant/parquet export
# (PeptideModifiedSequenceUnimodIds) and the English/CSV export (Peptide Modified Sequence) - leave
# these commented unless a column needs to be forced. A set value wins over auto-detect.
# data:
#   peptide_column: "Peptide Modified Sequence Unimod Ids"   # Skyline-PRISM.skyr's peptide column
#   protein_column: "Protein Accession"
#   protein_name_column: "Protein"
#   abundance_column: "Area"
#   rt_column: "Retention Time"
#   sample_column: "Replicate Name"
#   transition_column: "Fragment Ion"
#   batch_column: null
#   sample_type_column: null

# Sample type comes from the Skyline Replicates "Sample Type" column first (Standard -> reference,
# Quality Control -> qc). These substring patterns (matched against the replicate/sample name) are a
# FALLBACK, used only for replicates that have no Sample Type annotation.
sample_annotations:
  reference_pattern: ["-Pool-", "-Pool_", "_Pool_", "CommercialPool", "Ref", "Reference"]
  qc_pattern: ["-QC-", "-QC_", "_QC_", "QC", "Control", "StudyPool", "Quality Control"]

# Protein parsimony. Set fasta_path for FASTA-based mapping; null uses the Skyline
# Protein Accession column.
parsimony:
  fasta_path: null                     # set to a .fasta for substring-based parsimony mapping
  shared_peptide_handling: "all_groups"  # all_groups | unique_only | razor
  # Enzyme-aware terminus check for fasta_path mapping (ignored when null). Only attaches a peptide to
  # a protein when it occurs there with enzyme-consistent termini, removing phantom homolog assignments
  # (e.g. AKEGVVAAAEK is a substring of beta-synuclein but preceded there by M, not K/R). The Skyline
  # external tool overrides `enzyme` from the document's digestion settings; the CLI defaults to trypsin.
  enzyme: "trypsin"                    # trypsin (not before P) | trypsin/p (before P too, e.g. DIA-NN) | lysc | lysn | argc | aspn | gluc | chymotrypsin | nonspecific
  enzyme_specificity: "full"           # full (both termini) | semi (either) | none (legacy substring)

# Transition -> peptide rollup: "sum", "median_polish", "topn", "consensus", or "library_assist".
transition_rollup:
  method: "median_polish"
  min_transitions: 3
  topn_count: 3             # for method: topn
  topn_selection: "correlation"  # correlation | intensity (correlation needs Shape Correlation)
  topn_weighting: "sqrt"         # sqrt | sum
  use_ms1: false
  # For method: library_assist (Skyline .blib). A nested `library_assist:` block with the same
  # keys (library_path, min_matched_fragments, mz_tolerance, outlier_threshold, remove_outliers)
  # is also accepted. fitting_method is median_polish only (least_squares is not ported).
  # library_path: null
  # library_min_fragments: 3
  # library_mz_tolerance: 0.02
  # library_outlier_threshold: 1.0
  # library_remove_outliers: true   # iteratively drop interference fragments before scaling

# Peptide normalization: "rt_lowess" (default), "median", "quantile", "vsn", or "none".
global_normalization:
  method: "rt_lowess"
  rt_lowess:              # tuning for method: rt_lowess
    frac: 0.3             # LOWESS local-regression window fraction
    n_grid_points: 100    # RT grid points the curve is evaluated on before interpolation

# Low-signal sample outlier detection (one-sided, on linear scale).
sample_outlier_detection:
  enabled: true
  action: "report"        # report | exclude
  method: "iqr"           # iqr | fold_median
  # iqr_multiplier: 1.5    # method: iqr - flag samples below Q1 - k*IQR
  # fold_threshold: 0.1    # method: fold_median - flag samples below k*median

# ComBat batch correction (skipped automatically when < 2 batches).
batch_correction:
  enabled: true
  peptide_level: true
  protein_level: true
  method: "combat"
  reference_anchored: false   # estimate batch effects from reference samples across batches
  reference_type: "reference" # sample type used as the inter-batch reference
  auto_revert: false          # if ComBat worsens control (QC/reference) CV by >10%, keep uncorrected data

# Peptide -> protein rollup: "median_polish", "sum", "topn", "maxlfq", or "ibaq".
protein_rollup:
  method: "median_polish"
  min_peptides: 3
  topn:                  # for method: topn
    n: 3
    selection: "median_abundance"  # median_abundance | frequency
  # For method: ibaq (theoretical peptide counts from an in-silico digest):
  # ibaq:
  #   fasta_path: null    # falls back to parsimony.fasta_path
  #   enzyme: "trypsin"
  #   missed_cleavages: 0
  #   min_peptide_length: 6
  #   max_peptide_length: 30

# Protein normalization: "median" or "none".
protein_normalization:
  method: "median"

# Parallelism / memory. n_workers: 0 = all cores, 1 = serial, N = cap at N.
# merge_memory_mb caps DuckDB's buffer pool, in MB; 0 sizes it from the machine (a modest
# fraction of free memory, capped at 8 GB - the merge streams, so nothing here scales with
# the size of the cohort). Anything beyond the cap spills to the scratch directory, so a
# smaller value is slower, never wrong. Lower it to leave more room for Skyline alongside.
processing:
  n_workers: 0
  peptide_batch_size: 2000
  merge_memory_mb: 0

# Metadata columns in the Replicates/metadata report. null = auto-detect.
metadata:
  batch_column: null         # column carrying the batch/plate label
  sample_type_column: null   # Sample Type column (reference / qc / experimental)

# Fallback batch assignment, used ONLY when neither the metadata nor the Source Document
# distinguishes batches. Off by default: guessing batches from acquisition-time gaps cannot
# distinguish a real plate boundary from an ordinary pause, and a wrong guess makes ComBat
# "correct" between batches that do not exist. Turn it on deliberately.
batch_estimation:
  method: "none"             # none | auto | gap | fixed | source
  gap_iqr_multiplier: 1.5    # auto/gap: split when an acquisition-time gap exceeds k*IQR
  # n_batches: 3             # method: fixed - split into exactly N equal batches

# Output.
output:
  format: "parquet"
  include_residuals: true   # write median-polish residuals (peptides_rollup_residuals + proteins_raw_residuals)

# QC report (self-contained HTML + before/after plots).
qc_report:
  enabled: true

""";

    /// <summary>A trimmed template with just the common knobs (prism config-template --minimal).</summary>
    public static string Minimal() => """
# PRISM Configuration (minimal)
# Usage: prism run -i data.csv -o output/ -c this_config.yaml

transition_rollup:
  method: "median_polish"   # sum | median_polish | library_assist

global_normalization:
  method: "rt_lowess"       # rt_lowess | median | quantile | vsn | none

batch_correction:
  enabled: true             # skipped when < 2 batches

protein_rollup:
  method: "median_polish"   # median_polish | sum

processing:
  n_workers: 0              # 0 = all cores, 1 = serial

qc_report:
  enabled: true

""";
}
