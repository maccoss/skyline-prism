using System;
using System.Collections.Generic;
using System.IO;
using SkylinePrism.Core.IO;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace SkylinePrism.Core.Config;

/// <summary>
/// PRISM run configuration, deserialized from the YAML config (snake_case keys map to
/// PascalCase properties via the underscored naming convention). Defaults match the
/// Python pipeline defaults.
/// </summary>
public sealed class PrismConfig
{
    public DataSection Data { get; set; } = new();
    public TransitionRollupSection TransitionRollup { get; set; } = new();
    public GlobalNormalizationSection GlobalNormalization { get; set; } = new();
    public BatchCorrectionSection BatchCorrection { get; set; } = new();
    public ProteinRollupSection ProteinRollup { get; set; } = new();
    public ProteinNormalizationSection ProteinNormalization { get; set; } = new();
    public SampleAnnotationsSection SampleAnnotations { get; set; } = new();
    public ParsimonySection Parsimony { get; set; } = new();
    public OutputSection Output { get; set; } = new();
    public QcReportSection QcReport { get; set; } = new();
    public SampleOutlierDetectionSection SampleOutlierDetection { get; set; } = new();
    public MetadataSection Metadata { get; set; } = new();
    public ProcessingSection Processing { get; set; } = new();
    public BatchEstimationSection BatchEstimation { get; set; } = new();

    public static PrismConfig Load(string path) => Parse(File.ReadAllText(path));

    public static PrismConfig Parse(string yaml)
    {
        var deserializer = new DeserializerBuilder()
            .WithNamingConvention(UnderscoredNamingConvention.Instance)
            .IgnoreUnmatchedProperties()
            .Build();
        var config = deserializer.Deserialize<PrismConfig>(yaml) ?? new PrismConfig();
        config.TransitionRollup.ResolveLibraryAssist();
        return config;
    }

    /// <summary>
    /// Load + resolve a config file, report any unrecognized keys via <paramref name="warn"/>
    /// (Python-only or typo keys that C# would otherwise silently ignore), and throw on unsupported
    /// method choices (adaptive rollup, least_squares library fit, non-combat batch correction).
    /// </summary>
    public static PrismConfig LoadValidated(string path, Action<string>? warn = null)
    {
        var yaml = File.ReadAllText(path);
        var config = Parse(yaml);
        if (warn is not null)
            foreach (var key in FindUnknownKeys(yaml))
                warn($"config key '{key}' is not recognized by the C# PRISM port and will be ignored "
                    + "(a typo, or a Python-only setting not ported - see PORTING_STATUS.md).");
        config.Validate();
        return config;
    }

    /// <summary>Throw on config choices whose Python behavior is not implemented in the C# port.</summary>
    public void Validate()
    {
        var trm = TransitionRollup.Method?.ToLowerInvariant();
        if (trm == "adaptive")
            throw new NotSupportedException(
                "transition_rollup.method 'adaptive' is not implemented in the C# port. "
                + "Use median_polish, sum, topn, consensus, or library_assist. "
                + "(Adaptive/QuantUMS rollup is a documented non-port - see PORTING_STATUS.md.)");
        var validTr = new[] { "sum", "median_polish", "topn", "consensus", "library_assist" };
        if (trm is not null && Array.IndexOf(validTr, trm) < 0)
            throw new NotSupportedException(
                $"transition_rollup.method '{TransitionRollup.Method}' is not recognized. "
                + $"Valid: {string.Join(", ", validTr)}.");

        if (TransitionRollup.LibraryFittingMethod?.ToLowerInvariant() == "least_squares")
            throw new NotSupportedException(
                "library_assist.fitting_method 'least_squares' is not implemented in the C# port "
                + "(only median_polish). See PORTING_STATUS.md.");

        var bcm = BatchCorrection.Method?.ToLowerInvariant();
        if (bcm is not null && bcm != "combat")
            throw new NotSupportedException(
                $"batch_correction.method '{BatchCorrection.Method}' is not implemented in the C# port (only combat).");
    }

    // Nested schema of every config key the C# pipeline actually reads. Keys absent here are
    // reported by FindUnknownKeys. (null leaf = scalar key; nested dict = a subsection.)
    private static readonly IReadOnlyDictionary<string, object?> KnownKeys = BuildSchema();

    /// <summary>Dotted paths of keys in <paramref name="yaml"/> that the C# port does not read.</summary>
    public static List<string> FindUnknownKeys(string yaml)
    {
        var warnings = new List<string>();
        object? root;
        try
        {
            root = new DeserializerBuilder().Build().Deserialize<object>(yaml);
        }
        catch
        {
            return warnings; // malformed YAML surfaces elsewhere
        }
        if (root is IDictionary<object, object> map)
            WalkUnknown(map, KnownKeys, "", warnings);
        return warnings;
    }

    private static void WalkUnknown(
        IDictionary<object, object> node, IReadOnlyDictionary<string, object?> schema,
        string prefix, List<string> warnings)
    {
        foreach (var kv in node)
        {
            var key = kv.Key?.ToString() ?? "";
            var path = prefix.Length == 0 ? key : prefix + "." + key;
            if (!schema.TryGetValue(key, out var sub))
            {
                warnings.Add(path);
                continue;
            }
            if (sub is IReadOnlyDictionary<string, object?> subSchema
                && kv.Value is IDictionary<object, object> childMap)
                WalkUnknown(childMap, subSchema, path, warnings);
        }
    }

    private static Dictionary<string, object?> Leaves(params string[] keys)
    {
        var d = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (var k in keys)
            d[k] = null;
        return d;
    }

    private static Dictionary<string, object?> BuildSchema()
    {
        var tr = Leaves("method", "min_transitions", "topn_count", "topn_selection", "topn_weighting",
            "consensus_regularization", "use_ms1", "library_path", "library_min_fragments",
            "library_mz_tolerance", "library_outlier_threshold", "library_remove_outliers",
            "library_fitting_method");
        tr["library_assist"] = Leaves("library_path", "min_matched_fragments", "mz_tolerance",
            "outlier_threshold", "remove_outliers", "fitting_method");

        var gn = Leaves("method");
        gn["rt_lowess"] = Leaves("frac", "n_grid_points");

        var pr = Leaves("method", "min_peptides");
        pr["topn"] = Leaves("n", "selection");
        pr["ibaq"] = Leaves("fasta_path", "enzyme", "missed_cleavages", "min_peptide_length", "max_peptide_length");

        return new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["data"] = Leaves("abundance_column", "rt_column", "peptide_column", "protein_column",
                "protein_name_column", "sample_column", "transition_column", "precursor_column",
                "fragment_column", "batch_column", "sample_type_column"),
            ["transition_rollup"] = tr,
            ["global_normalization"] = gn,
            ["batch_correction"] = Leaves("enabled", "peptide_level", "protein_level", "method",
                "reference_anchored", "reference_type", "auto_revert"),
            ["protein_rollup"] = pr,
            ["protein_normalization"] = Leaves("method"),
            ["sample_annotations"] = Leaves("reference_pattern", "qc_pattern"),
            ["parsimony"] = Leaves(
                "enabled", "fasta_path", "shared_peptide_handling", "enzyme", "enzyme_specificity"),
            ["output"] = Leaves("format", "include_residuals"),
            ["qc_report"] = Leaves("enabled", "save_plots"),
            ["sample_outlier_detection"] = Leaves("enabled", "action", "method", "iqr_multiplier", "fold_threshold"),
            ["metadata"] = Leaves("batch_column", "sample_type_column"),
            ["processing"] = Leaves("n_workers", "peptide_batch_size", "merge_memory_mb"),
            ["batch_estimation"] = Leaves("method", "n_batches", "gap_iqr_multiplier"),
        };
    }

    /// <summary>
    /// Explicit input-column-name overrides (Python's <c>data</c> section). Any set value wins over
    /// auto-detection when it resolves against the input columns (matched ignoring case / spaces /
    /// underscores, so English "Peptide Modified Sequence" and invariant "PeptideModifiedSequence" both
    /// work). Leave null to auto-detect. batch_column / sample_type_column here are also honored (they
    /// fall back to the metadata section).
    /// </summary>
    public sealed class DataSection
    {
        public string? AbundanceColumn { get; set; }
        public string? RtColumn { get; set; }
        public string? PeptideColumn { get; set; }
        public string? ProteinColumn { get; set; }
        public string? ProteinNameColumn { get; set; }
        public string? SampleColumn { get; set; }
        public string? TransitionColumn { get; set; }
        public string? PrecursorColumn { get; set; }
        public string? FragmentColumn { get; set; }
        public string? BatchColumn { get; set; }
        public string? SampleTypeColumn { get; set; }

        public ColumnOverrides ToOverrides() => new(
            Peptide: PeptideColumn,
            Protein: ProteinColumn,
            ProteinName: ProteinNameColumn,
            Abundance: AbundanceColumn,
            RetentionTime: RtColumn,
            Sample: SampleColumn,
            Transition: TransitionColumn,
            Batch: BatchColumn);
    }

    public sealed class TransitionRollupSection
    {
        // method: sum | median_polish | topn | consensus | library_assist
        public string Method { get; set; } = "sum";
        public int MinTransitions { get; set; } = 3;

        // Keys + defaults match the Python config (topn_count / topn_selection / topn_weighting):
        // default selection is correlation, default weighting is sqrt.
        public int TopnCount { get; set; } = 3;
        public string TopnSelection { get; set; } = "correlation"; // correlation | intensity
        public string TopnWeighting { get; set; } = "sqrt";        // sqrt | sum
        public double ConsensusRegularization { get; set; } = 0.1;
        public bool UseMs1 { get; set; }

        // Library-assisted rollup (method: library_assist). Flat keys (C# form) plus the nested
        // `library_assist:` block (Python's canonical form) are both accepted; ResolveLibraryAssist
        // folds the nested block onto these flat fields (nested wins).
        public string? LibraryPath { get; set; }
        public int LibraryMinFragments { get; set; } = 3;
        public double LibraryMzTolerance { get; set; } = 0.02;
        public double LibraryOutlierThreshold { get; set; } = 1.0;

        /// <summary>Iteratively remove high-residual (interference) fragments before the final scale.</summary>
        public bool LibraryRemoveOutliers { get; set; } = true;

        /// <summary>median_polish only in C#; least_squares is validated-out (not ported).</summary>
        public string LibraryFittingMethod { get; set; } = "median_polish";

        /// <summary>Nested Python-style library_assist block; resolved onto the flat fields above.</summary>
        public LibraryAssistSection? LibraryAssist { get; set; }

        /// <summary>Fold the nested <c>library_assist:</c> block onto the flat library_* fields (nested wins).</summary>
        public void ResolveLibraryAssist()
        {
            if (LibraryAssist is null)
                return;
            var la = LibraryAssist;
            if (la.LibraryPath is not null) LibraryPath = la.LibraryPath;
            if (la.MinMatchedFragments is not null) LibraryMinFragments = la.MinMatchedFragments.Value;
            if (la.MzTolerance is not null) LibraryMzTolerance = la.MzTolerance.Value;
            if (la.OutlierThreshold is not null) LibraryOutlierThreshold = la.OutlierThreshold.Value;
            if (la.RemoveOutliers is not null) LibraryRemoveOutliers = la.RemoveOutliers.Value;
            if (la.FittingMethod is not null) LibraryFittingMethod = la.FittingMethod;
        }
    }

    /// <summary>Python's nested <c>transition_rollup.library_assist:</c> block.</summary>
    public sealed class LibraryAssistSection
    {
        public string? LibraryPath { get; set; }
        public int? MinMatchedFragments { get; set; }
        public double? MzTolerance { get; set; }
        public double? OutlierThreshold { get; set; }
        public bool? RemoveOutliers { get; set; }
        public string? FittingMethod { get; set; }
    }

    public sealed class GlobalNormalizationSection
    {
        // RT-lowess is the recommended default: it removes RT-dependent systematic variation
        // (ion suppression, gradient drift) in addition to overall loading differences.
        public string Method { get; set; } = "rt_lowess";

        /// <summary>LOWESS tuning for method: rt_lowess (matches Python global_normalization.rt_lowess).</summary>
        public RtLowessSection RtLowess { get; set; } = new();
    }

    public sealed class RtLowessSection
    {
        /// <summary>Fraction of points in each local regression window (statsmodels frac).</summary>
        public double Frac { get; set; } = 0.3;

        /// <summary>Number of RT grid points the fitted curve is evaluated on before interpolation.</summary>
        public int NGridPoints { get; set; } = 100;
    }

    public sealed class BatchCorrectionSection
    {
        /// <summary>Master switch. When false, neither peptide nor protein ComBat runs.</summary>
        public bool Enabled { get; set; } = true;

        /// <summary>Apply ComBat at the peptide level (Stage 2c).</summary>
        public bool PeptideLevel { get; set; } = true;

        /// <summary>Apply ComBat at the protein level (Stage 4c).</summary>
        public bool ProteinLevel { get; set; } = true;

        public string Method { get; set; } = "combat";
        public bool ReferenceAnchored { get; set; }
        public string ReferenceType { get; set; } = "reference";

        /// <summary>
        /// Safety net (opt-in): after ComBat, if it worsened the control-sample CV (QC preferred, else
        /// reference) by more than 10%, revert to the uncorrected data. Off by default to match Python's
        /// production path (the revert exists only in Python's legacy normalize_pipeline).
        /// </summary>
        public bool AutoRevert { get; set; }
    }

    public sealed class ProteinRollupSection
    {
        // method: median_polish | sum | topn | maxlfq | ibaq
        public string Method { get; set; } = "median_polish";
        public int MinPeptides { get; set; } = 3;

        /// <summary>Nested topN parameters (matches Python protein_rollup.topn.{n,selection}).</summary>
        public ProteinTopnSection Topn { get; set; } = new();

        public IbaqSection Ibaq { get; set; } = new();
    }

    public sealed class ProteinTopnSection
    {
        /// <summary>Peptides to average for method: topn.</summary>
        public int N { get; set; } = 3;
        public string Selection { get; set; } = "median_abundance";
    }

    public sealed class IbaqSection
    {
        /// <summary>FASTA for theoretical peptide counts; falls back to parsimony.fasta_path.</summary>
        public string? FastaPath { get; set; }
        public string Enzyme { get; set; } = "trypsin";
        public int MissedCleavages { get; set; }

        /// <summary>Min/max tryptic peptide length counted toward the iBAQ denominator (fasta.py digest bounds).</summary>
        public int MinPeptideLength { get; set; } = 6;
        public int MaxPeptideLength { get; set; } = 30;
    }

    public sealed class ProteinNormalizationSection
    {
        public string Method { get; set; } = "median";
    }

    public sealed class SampleAnnotationsSection
    {
        // Fallback only: sample type comes from the Replicates-grid "Sample Type" column first (Standard
        // -> reference, Quality Control -> qc). These substring patterns (case-sensitive, matched against
        // the replicate/sample name) are used ONLY when a replicate has no Sample Type annotation.
        public List<string> ReferencePattern { get; set; } = new()
            { "-Pool-", "-Pool_", "_Pool_", "CommercialPool", "Ref", "Reference" };
        public List<string> QcPattern { get; set; } = new()
            { "-QC-", "-QC_", "_QC_", "QC", "Control", "StudyPool", "Quality Control" };
    }

    public sealed class ParsimonySection
    {
        /// <summary>When false, each protein accession is its own group (no grouping/razor).</summary>
        public bool Enabled { get; set; } = true;
        public string? FastaPath { get; set; }
        public string SharedPeptideHandling { get; set; } = "all_groups";

        /// <summary>
        /// Digestion enzyme for the FASTA-mapping terminus check (ignored when FastaPath is null; the
        /// Skyline Protein Accession column is already enzyme-aware). Default "trypsin" (cleave after
        /// K/R, but NOT before P); use "trypsin/p" for K/R-P cleavage (e.g. DIA-NN). The Skyline
        /// external tool overrides this from the document's digestion settings.
        /// </summary>
        public string Enzyme { get; set; } = "trypsin";

        /// <summary>
        /// Terminus requirement for FASTA membership: "full" (both termini cleavage-consistent -
        /// removes phantom paralog assignments), "semi" (either terminus), or "none" (legacy pure
        /// substring). Skyline background-proteome digestion is full-specific, so "full" is the default.
        /// </summary>
        public string EnzymeSpecificity { get; set; } = "full";
    }

    public sealed class OutputSection
    {
        public string Format { get; set; } = "parquet";
        public bool IncludeResiduals { get; set; } = true;
    }

    public sealed class QcReportSection
    {
        public bool Enabled { get; set; } = true;
        public bool SavePlots { get; set; } = true;
        // Note: plots are always base64-embedded (self-contained HTML). Linking to external PNGs
        // (Python's embed_plots: false) is a deliberate non-port - see PORTING_STATUS.md.
    }

    public sealed class MetadataSection
    {
        /// <summary>
        /// Explicit column name in the Replicates report to use as the batch label. The report's
        /// columns are annotation-dependent, so this may need to be set per document. Null =
        /// auto-detect a "Batch"-like column (and fall back to the Source Document otherwise).
        /// </summary>
        public string? BatchColumn { get; set; }

        /// <summary>Explicit Sample Type column name; null = auto-detect (usually "Sample Type").</summary>
        public string? SampleTypeColumn { get; set; }
    }

    public sealed class BatchEstimationSection
    {
        /// <summary>
        /// auto | gap | fixed | source | none. Used only when no explicit batch is available.
        /// <para>
        /// Defaults to <c>none</c>: INVENTING batches is worse than having none. Gap detection
        /// cannot tell a real plate boundary from an ordinary pause in a continuously acquired run,
        /// and when it guesses wrong ComBat "corrects" between batches that do not exist - which
        /// silently alters every abundance. Batches should come from a real annotation; this is an
        /// opt-in convenience for runs where one is genuinely unavailable.
        /// </para>
        /// </summary>
        public string Method { get; set; } = "none";
        public int? NBatches { get; set; }
        public double GapIqrMultiplier { get; set; } = 1.5;
    }

    public sealed class ProcessingSection
    {
        /// <summary>Rollup worker threads: 0 = all logical cores, 1 = serial, N = cap at N.</summary>
        public int NWorkers { get; set; }

        /// <summary>Peptides buffered per streamed parquet row group (flush granularity).</summary>
        public int PeptideBatchSize { get; set; } = 2000;

        /// <summary>
        /// Ceiling on DuckDB's buffer pool during the Stage 1 merge, in MB. 0 = size it from the
        /// machine (see <c>DuckDbMerge.AutoMemoryBudgetMb</c>). Work beyond the ceiling spills to
        /// the sort scratch directory, so this trades speed for footprint and cannot cause a wrong
        /// answer. Raise it when the merge spills on a machine with RAM to spare; lower it to leave
        /// room for something else running alongside.
        /// </summary>
        public int MergeMemoryMb { get; set; }
    }

    public sealed class SampleOutlierDetectionSection
    {
        public bool Enabled { get; set; } = true;
        public string Action { get; set; } = "report";
        public string Method { get; set; } = "iqr";
        public double IqrMultiplier { get; set; } = 1.5;
        public double FoldThreshold { get; set; } = 0.1;
    }
}
