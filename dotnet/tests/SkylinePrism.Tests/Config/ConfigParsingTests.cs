using System;
using System.IO;
using SkylinePrism.Core.Config;
using Xunit;

namespace SkylinePrism.Tests.Config;

/// <summary>Guards that the YAML keys for the newer options bind (UnderscoredNamingConvention).</summary>
public class ConfigParsingTests
{
    // The tool's "Show Command Line" serializes the UI config to YAML; prism run -c must read it back
    // to the same settings. Guards that serialize -> parse round-trips.
    [Fact]
    public void SerializedConfig_RoundTripsThroughParse()
    {
        var config = new PrismConfig();
        config.TransitionRollup.Method = "topn";
        config.TransitionRollup.TopnCount = 5;
        config.GlobalNormalization.Method = "quantile";
        config.ProteinRollup.Method = "maxlfq";
        config.ProteinRollup.Topn.N = 7;
        config.BatchCorrection.PeptideLevel = false;

        var yaml = new YamlDotNet.Serialization.SerializerBuilder()
            .WithNamingConvention(YamlDotNet.Serialization.NamingConventions.UnderscoredNamingConvention.Instance)
            .Build()
            .Serialize(config);
        var back = PrismConfig.Parse(yaml);

        Assert.Equal("topn", back.TransitionRollup.Method);
        Assert.Equal(5, back.TransitionRollup.TopnCount);
        Assert.Equal("quantile", back.GlobalNormalization.Method);
        Assert.Equal("maxlfq", back.ProteinRollup.Method);
        Assert.Equal(7, back.ProteinRollup.Topn.N);
        Assert.False(back.BatchCorrection.PeptideLevel);
    }

    /// <summary>
    /// A section header whose keys are all commented out sets nothing, so the section keeps its
    /// defaults. YamlDotNet reads the empty value as null, and a null section was dereferenced -
    /// at once for transition_rollup, later in the run for any other.
    /// </summary>
    [Theory]
    [InlineData("transition_rollup:\n  # method: sum\n")]
    [InlineData("qc_report:\n")]
    [InlineData("global_normalization:\n  rt_lowess:\n")]
    public void Parse_ASectionHeaderWithNothingUnderIt_KeepsItsDefaults(string yaml)
    {
        var config = PrismConfig.Parse(yaml);

        Assert.Equal(
            System.Text.Json.JsonSerializer.Serialize(new PrismConfig()),
            System.Text.Json.JsonSerializer.Serialize(config));
    }

    /// <summary>
    /// A key written with no value - a list whose entries are all commented out, a flag or a number
    /// left blank - is the same as leaving the key out, so it keeps its default. YamlDotNet reads the
    /// blank as null: a null list was dereferenced by sample-type classification, and a blank flag or
    /// number would otherwise have bound as false or 0 with nothing said.
    /// </summary>
    [Theory]
    [InlineData("sample_annotations:\n  reference_pattern:\n  #  - Pool\n")]
    [InlineData("batch_correction:\n  enabled:\n")]
    [InlineData("transition_rollup:\n  min_transitions:\n")]
    [InlineData("global_normalization:\n  rt_lowess:\n    frac:\n")]
    [InlineData("transition_rollup:\n  method:\n")]
    public void Parse_AKeyWithNoValue_KeepsItsDefault(string yaml)
    {
        var config = PrismConfig.Parse(yaml);

        Assert.Equal(
            System.Text.Json.JsonSerializer.Serialize(new PrismConfig()),
            System.Text.Json.JsonSerializer.Serialize(config));
    }

    /// <summary>
    /// What a blank cannot express is still expressible: <c>[]</c> is an empty list, and a key whose
    /// default is null (a FASTA path) takes an explicit null.
    /// </summary>
    [Fact]
    public void Parse_AnEmptyListAndANullPath_AreValues()
    {
        var config = PrismConfig.Parse("sample_annotations:\n  reference_pattern: []\nparsimony:\n  fasta_path: null\n");

        Assert.Empty(config.SampleAnnotations.ReferencePattern);
        Assert.Null(config.Parsimony.FastaPath);
    }

    [Fact]
    public void NewOptionKeys_Bind()
    {
        var yaml = """
transition_rollup:
  method: "topn"
  topn_count: 7
  topn_selection: "intensity"
protein_rollup:
  method: "maxlfq"
  topn:
    n: 9
processing:
  n_workers: 4
  peptide_batch_size: 500
parsimony:
  fasta_path: "db.fasta"
  shared_peptide_handling: "razor"
""";
        var path = Path.Combine(Path.GetTempPath(), "prism_cfg_" + Guid.NewGuid().ToString("N") + ".yaml");
        File.WriteAllText(path, yaml);
        try
        {
            var cfg = PrismConfig.Load(path);
            Assert.Equal("topn", cfg.TransitionRollup.Method);
            Assert.Equal(7, cfg.TransitionRollup.TopnCount);
            Assert.Equal("intensity", cfg.TransitionRollup.TopnSelection);
            Assert.Equal("maxlfq", cfg.ProteinRollup.Method);
            Assert.Equal(9, cfg.ProteinRollup.Topn.N);
            Assert.Equal(4, cfg.Processing.NWorkers);
            Assert.Equal(500, cfg.Processing.PeptideBatchSize);
            Assert.Equal("db.fasta", cfg.Parsimony.FastaPath);
            Assert.Equal("razor", cfg.Parsimony.SharedPeptideHandling);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void Templates_Parse()
    {
        foreach (var yaml in new[] { ConfigTemplate.Default(), ConfigTemplate.Minimal() })
        {
            var path = Path.Combine(Path.GetTempPath(), "prism_tpl_" + Guid.NewGuid().ToString("N") + ".yaml");
            File.WriteAllText(path, yaml);
            try
            {
                var cfg = PrismConfig.Load(path); // must not throw
                Assert.NotNull(cfg.TransitionRollup.Method);
            }
            finally { File.Delete(path); }
        }
    }
}
