using System;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Config;
using Xunit;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace SkylinePrism.Tests.Config;

/// <summary>
/// Guards the config the Skyline tool shows/copies ("Show command line" -> "Copy config (YAML)"):
/// it must list only the settings that apply to the selected methods, yet still reproduce the run
/// exactly when fed back to the CLI.
/// </summary>
public class ConfigWriterTests
{
    // Full property-by-property dump, used to prove the minimal YAML round-trips without loss.
    // The batch / sample-type columns are folded into data: first, because the writer deliberately
    // emits them there (both engines read data.*, only C# reads metadata.*) and the pipeline
    // resolves Data.BatchColumn ?? Metadata.BatchColumn. A value arriving back under data: instead
    // of metadata: is therefore a representational change, not a lost setting. Folding BOTH sides
    // normalizes that away while still failing if the writer actually dropped the value.
    private static string FullDump(PrismConfig c)
    {
        c.Data.BatchColumn ??= c.Metadata.BatchColumn;
        c.Data.SampleTypeColumn ??= c.Metadata.SampleTypeColumn;
        c.Metadata.BatchColumn = null;
        c.Metadata.SampleTypeColumn = null;
        return new SerializerBuilder()
            .WithNamingConvention(UnderscoredNamingConvention.Instance)
            .Build()
            .Serialize(c);
    }

    private static PrismConfig LibraryAssistConfig()
    {
        var c = new PrismConfig();
        c.TransitionRollup.Method = "library_assist";
        c.TransitionRollup.LibraryPath = @"V:\project\library.blib";
        c.TransitionRollup.MinTransitions = 3;
        c.ProteinRollup.Method = "median_polish";
        c.SampleOutlierDetection.Action = "exclude";
        c.QcReport.SavePlots = false;
        return c;
    }

    [Fact]
    public void LibraryAssist_OmitsKeysTheMethodIgnores()
    {
        var yaml = ConfigWriter.ToYaml(LibraryAssistConfig());

        // Present: the library knobs that actually drive the rollup.
        Assert.Contains(@"library_path: V:\project\library.blib", yaml);
        Assert.Contains("library_min_fragments: 3", yaml);
        Assert.Contains("library_mz_tolerance: 0.02", yaml);
        Assert.Contains("library_outlier_threshold: 1", yaml);
        Assert.Contains("library_remove_outliers: true", yaml);
        // Named even at its default: C# fits median_polish only, but Python also offers
        // least_squares, so a config carried to the CLI has to say which fit set the sample scale.
        Assert.Contains("library_fitting_method: median_polish", yaml);

        // Absent: other methods' parameters, and blocks that do not apply.
        Assert.DoesNotContain("topn_count", yaml);
        Assert.DoesNotContain("topn_selection", yaml);
        Assert.DoesNotContain("topn_weighting", yaml);
        Assert.DoesNotContain("consensus_regularization", yaml);
        Assert.DoesNotContain("ibaq", yaml);
        Assert.DoesNotContain("min_peptide_length", yaml);
    }

    [Fact]
    public void AlgorithmNamingKeys_AreWrittenEvenAtTheirDefault()
    {
        // Which algorithm ran must be answerable from the file alone; only numeric/boolean tuning
        // values are elided at their defaults.
        var yaml = ConfigWriter.ToYaml(LibraryAssistConfig());

        Assert.Contains("method: library_assist", yaml);
        Assert.Contains("library_fitting_method: median_polish", yaml);
        Assert.Contains("method: combat", yaml);          // batch_correction
        Assert.Contains("method: iqr", yaml);             // sample_outlier_detection
        Assert.Contains("method: median_polish", yaml);   // protein_rollup
        Assert.Contains("method: rt_lowess", yaml);       // global_normalization
        Assert.Contains("method: median", yaml);          // protein_normalization

        // ...but their tuning knobs stay out while they sit at the default.
        Assert.DoesNotContain("iqr_multiplier", yaml);
        Assert.DoesNotContain("auto_revert", yaml);
        Assert.DoesNotContain("n_grid_points", yaml);
    }

    [Fact]
    public void TopN_EmitsTopNKeysAndOmitsLibraryKeys()
    {
        var c = new PrismConfig();
        c.TransitionRollup.Method = "topn";
        c.TransitionRollup.TopnCount = 5;
        c.TransitionRollup.TopnSelection = "intensity";

        var yaml = ConfigWriter.ToYaml(c);

        Assert.Contains("topn_count: 5", yaml);
        Assert.Contains("topn_selection: intensity", yaml);
        Assert.Contains("topn_weighting: sqrt", yaml);
        Assert.DoesNotContain("library_", yaml);
        Assert.DoesNotContain("consensus_regularization", yaml);
    }

    [Fact]
    public void Consensus_EmitsOnlyItsOwnTuningKey()
    {
        var c = new PrismConfig();
        c.TransitionRollup.Method = "consensus";

        var yaml = ConfigWriter.ToYaml(c);

        Assert.Contains("consensus_regularization: 0.1", yaml);
        Assert.DoesNotContain("topn_count", yaml);
        Assert.DoesNotContain("library_", yaml);
    }

    [Fact]
    public void DisabledSections_DropTheirTuningKeys()
    {
        var c = new PrismConfig();
        c.BatchCorrection.Enabled = false;
        c.Parsimony.Enabled = false;
        c.SampleOutlierDetection.Enabled = false;

        var yaml = ConfigWriter.ToYaml(c);

        Assert.Contains("enabled: false", yaml);
        Assert.DoesNotContain("peptide_level", yaml);
        Assert.DoesNotContain("protein_level", yaml);
        Assert.DoesNotContain("reference_type", yaml);
        Assert.DoesNotContain("shared_peptide_handling", yaml);
        Assert.DoesNotContain("action:", yaml);
    }

    [Fact]
    public void ParsimonyWithoutFasta_OmitsEnzymeKeys()
    {
        // Without a FASTA the map comes from Skyline's Protein Accession column, which is already
        // enzyme-aware - the enzyme keys would be inert noise. This is the Skyline tool's usual case.
        var yaml = ConfigWriter.ToYaml(new PrismConfig());

        Assert.DoesNotContain("enzyme", yaml);
        Assert.DoesNotContain("fasta_path", yaml);
    }

    [Fact]
    public void ParsimonyWithFasta_EmitsEnzymeKeys()
    {
        var c = new PrismConfig();
        c.Parsimony.FastaPath = "db.fasta";
        c.Parsimony.Enzyme = "trypsin/p";

        var yaml = ConfigWriter.ToYaml(c);

        Assert.Contains("fasta_path: db.fasta", yaml);
        Assert.Contains("enzyme: trypsin/p", yaml);
        Assert.Contains("enzyme_specificity: full", yaml);
    }

    [Fact]
    public void UnsetOptionalKeys_AreOmitted()
    {
        var yaml = ConfigWriter.ToYaml(new PrismConfig());

        // A plain object dump emits every column override as an empty key; those read as "unset by
        // hand" rather than "auto-detected", and an empty value is exactly the dangling-key shape
        // that trips the Python loader.
        Assert.DoesNotContain("abundance_column", yaml);
        Assert.DoesNotContain("rt_column", yaml);
        Assert.DoesNotContain("sample_type_column", yaml);
        Assert.DoesNotContain("library_assist", yaml);
        Assert.DoesNotContain(": \n", yaml);
        Assert.DoesNotContain(": \r\n", yaml);
    }

    [Fact]
    public void ExplicitlySetOptionalKeys_AreKept()
    {
        var c = new PrismConfig();
        c.Metadata.BatchColumn = "Plate";
        c.Data.AbundanceColumn = "Area";
        c.Processing.NWorkers = 8;

        var yaml = ConfigWriter.ToYaml(c);

        Assert.Contains("batch_column: Plate", yaml);
        Assert.Contains("abundance_column: Area", yaml);
        Assert.Contains("n_workers: 8", yaml);
    }

    [Theory]
    [InlineData("library_assist")]
    [InlineData("topn")]
    [InlineData("consensus")]
    [InlineData("median_polish")]
    [InlineData("sum")]
    public void RoundTrip_ReproducesEveryTransitionRollupSetting(string method)
    {
        var c = LibraryAssistConfig();
        c.TransitionRollup.Method = method;
        // Inert values ARE dropped by design (a stale library path under method: sum changes
        // nothing), so only carry the library path for the method that reads it.
        if (method != "library_assist")
            c.TransitionRollup.LibraryPath = null;

        var reparsed = PrismConfig.Parse(ConfigWriter.ToYaml(c));

        // Every property of the reparsed config must match the original: what the writer leaves out
        // has to be recoverable from the built-in defaults, or the shown command line would lie.
        Assert.Equal(FullDump(c), FullDump(reparsed));
    }

    [Fact]
    public void RoundTrip_ReproducesNonDefaultSettingsAcrossAllSections()
    {
        var c = new PrismConfig();
        c.Data.PeptideColumn = "Peptide Modified Sequence";
        c.Metadata.BatchColumn = "Plate";
        c.Metadata.SampleTypeColumn = "Type";
        c.TransitionRollup.Method = "topn";
        c.TransitionRollup.TopnCount = 6;
        c.TransitionRollup.TopnWeighting = "sum";
        c.TransitionRollup.UseMs1 = true;
        c.TransitionRollup.MinTransitions = 2;
        c.GlobalNormalization.Method = "rt_lowess";
        c.GlobalNormalization.RtLowess.Frac = 0.5;
        c.GlobalNormalization.RtLowess.NGridPoints = 50;
        c.BatchCorrection.ReferenceAnchored = true;
        c.BatchCorrection.ReferenceType = "qc";
        c.BatchCorrection.AutoRevert = true;
        c.BatchCorrection.ProteinLevel = false;
        c.Parsimony.FastaPath = "db.fasta";
        c.Parsimony.EnzymeSpecificity = "semi";
        c.Parsimony.SharedPeptideHandling = "unique_only";
        c.ProteinRollup.Method = "ibaq";
        c.ProteinRollup.MinPeptides = 2;
        c.ProteinRollup.Ibaq.FastaPath = "db.fasta";
        c.ProteinRollup.Ibaq.MissedCleavages = 1;
        c.ProteinRollup.Ibaq.MaxPeptideLength = 40;
        c.ProteinNormalization.Method = "none";
        c.SampleOutlierDetection.Method = "fold_median";
        c.SampleOutlierDetection.FoldThreshold = 0.2;
        c.SampleOutlierDetection.Action = "exclude";
        c.Output.IncludeResiduals = false;
        c.QcReport.SavePlots = false;
        c.Processing.NWorkers = 4;
        c.Processing.PeptideBatchSize = 500;
        c.BatchEstimation.Method = "fixed";
        c.BatchEstimation.NBatches = 3;
        c.BatchEstimation.GapIqrMultiplier = 2.0;
        c.SampleAnnotations.ReferencePattern = new() { "-Pool-" };
        c.SampleAnnotations.QcPattern = new() { "-QC-" };

        var reparsed = PrismConfig.Parse(ConfigWriter.ToYaml(c));

        Assert.Equal(FullDump(c), FullDump(reparsed));
    }

    [Fact]
    public void RoundTrip_ReproducesProteinTopnBlock()
    {
        var c = new PrismConfig();
        c.ProteinRollup.Method = "topn";
        c.ProteinRollup.Topn.N = 5;
        c.ProteinRollup.Topn.Selection = "frequency";

        var reparsed = PrismConfig.Parse(ConfigWriter.ToYaml(c));

        Assert.Equal(FullDump(c), FullDump(reparsed));
    }

    [Fact]
    public void EmittedYaml_MatchesTheCrossEngineFixture()
    {
        // Golden file, checked from BOTH sides: this test pins that the fixture still reflects what
        // ConfigWriter emits, and tests/test_cli.py pins that the Python engine loads that same file
        // with no unrecognized keys. Neither side alone is enough - C#'s own FindUnknownKeys cannot
        // notice a key that only the Python schema is missing, which is exactly the failure that
        // made a tool-authored config unusable with the CLI in the first place.
        var path = Path.Combine(AppContext.BaseDirectory, "fixtures", "config", "emitted-library-assist.yaml");
        Assert.True(File.Exists(path), $"fixture missing: {path}");

        var c = new PrismConfig();
        c.TransitionRollup.Method = "library_assist";
        c.TransitionRollup.LibraryPath = "spectra.blib";
        c.SampleOutlierDetection.Action = "exclude";
        c.QcReport.SavePlots = false;
        c.Metadata.BatchColumn = "Plate";

        Assert.Equal(
            File.ReadAllText(path).Replace("\r\n", "\n").TrimEnd(),
            ConfigWriter.ToYaml(c).Replace("\r\n", "\n").TrimEnd());
    }

    [Fact]
    public void EmittedYaml_UsesOnlyKeysTheSchemaKnows()
    {
        // Same guard as ConfigTemplate: a key here that FindUnknownKeys rejects would warn on every
        // CLI re-run of a copied config.
        Assert.Empty(PrismConfig.FindUnknownKeys(ConfigWriter.ToYaml(LibraryAssistConfig())));

        var topn = new PrismConfig();
        topn.TransitionRollup.Method = "topn";
        topn.ProteinRollup.Method = "ibaq";
        Assert.Empty(PrismConfig.FindUnknownKeys(ConfigWriter.ToYaml(topn)));
    }

    [Fact]
    public void EmittedYaml_ValidatesAndStaysShort()
    {
        var yaml = ConfigWriter.ToYaml(LibraryAssistConfig());
        PrismConfig.Parse(yaml).Validate(); // must not throw

        // A plain object dump of PrismConfig is ~95 keys; the point of the writer is that a typical
        // run fits on a screen. Canary against regressing to a full dump, not an exact count.
        var keyLines = yaml.Split('\n').Count(l => l.Contains(':') && !l.TrimStart().StartsWith('#'));
        Assert.InRange(keyLines, 1, 40);
    }
}
