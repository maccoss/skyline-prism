using System.Collections.Generic;
using System.IO;
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

    public static PrismConfig Load(string path)
    {
        var yaml = File.ReadAllText(path);
        return Parse(yaml);
    }

    public static PrismConfig Parse(string yaml)
    {
        var deserializer = new DeserializerBuilder()
            .WithNamingConvention(UnderscoredNamingConvention.Instance)
            .IgnoreUnmatchedProperties()
            .Build();
        return deserializer.Deserialize<PrismConfig>(yaml) ?? new PrismConfig();
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

        // Library-assisted rollup (method: library_assist).
        public string? LibraryPath { get; set; }
        public int LibraryMinFragments { get; set; } = 3;
        public double LibraryMzTolerance { get; set; } = 0.02;
        public double LibraryOutlierThreshold { get; set; } = 1.0;
    }

    public sealed class GlobalNormalizationSection
    {
        // RT-lowess is the recommended default: it removes RT-dependent systematic variation
        // (ion suppression, gradient drift) in addition to overall loading differences.
        public string Method { get; set; } = "rt_lowess";
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
    }

    public sealed class ProteinRollupSection
    {
        // method: median_polish | sum | topn | maxlfq | ibaq
        public string Method { get; set; } = "median_polish";
        public int MinPeptides { get; set; } = 3;

        /// <summary>Peptides to average for method: topn.</summary>
        public int TopN { get; set; } = 3;

        public IbaqSection Ibaq { get; set; } = new();
    }

    public sealed class IbaqSection
    {
        /// <summary>FASTA for theoretical peptide counts; falls back to parsimony.fasta_path.</summary>
        public string? FastaPath { get; set; }
        public string Enzyme { get; set; } = "trypsin";
        public int MissedCleavages { get; set; }
    }

    public sealed class ProteinNormalizationSection
    {
        public string Method { get; set; } = "median";
    }

    public sealed class SampleAnnotationsSection
    {
        public List<string> ReferencePattern { get; set; } = new() { "-Pool_", "_Pool_", "CommercialPool" };
        public List<string> QcPattern { get; set; } = new() { "-QC_", "_QC_", "StudyPool" };
    }

    public sealed class ParsimonySection
    {
        /// <summary>When false, each protein accession is its own group (no grouping/razor).</summary>
        public bool Enabled { get; set; } = true;
        public string? FastaPath { get; set; }
        public string SharedPeptideHandling { get; set; } = "all_groups";
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
        public bool EmbedPlots { get; set; } = true;
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
        /// <summary>auto | gap | fixed | source | none. Used only when no explicit batch is available.</summary>
        public string Method { get; set; } = "auto";
        public int? NBatches { get; set; }
        public double GapIqrMultiplier { get; set; } = 1.5;
    }

    public sealed class ProcessingSection
    {
        /// <summary>Rollup worker threads: 0 = all logical cores, 1 = serial, N = cap at N.</summary>
        public int NWorkers { get; set; }

        /// <summary>Peptides buffered per streamed parquet row group (flush granularity).</summary>
        public int PeptideBatchSize { get; set; } = 2000;
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
