using System.Collections.Generic;
using System.IO;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.Pipeline;
using Xunit;

namespace SkylinePrism.Tests.Pipeline;

/// <summary>parameters.json provenance round-trips the full config so a run can be reproduced.</summary>
public class ProvenanceTests
{
    [Fact]
    public void WriteThenLoad_RestoresConfigSettings()
    {
        var config = new PrismConfig();
        config.TransitionRollup.Method = "library_assist";
        config.TransitionRollup.LibraryPath = @"C:\libs\doc.blib";
        config.TransitionRollup.MinTransitions = 4;
        config.GlobalNormalization.Method = "quantile";
        config.BatchCorrection.Enabled = true;
        config.BatchCorrection.PeptideLevel = false;
        config.BatchCorrection.ProteinLevel = true;
        config.Parsimony.Enabled = false;
        config.ProteinRollup.Method = "sum";
        config.ProteinNormalization.Method = "none";
        config.SampleOutlierDetection.Action = "exclude";

        var path = Path.Combine(Path.GetTempPath(), "prism_prov_" + System.Guid.NewGuid().ToString("N") + ".json");
        try
        {
            Provenance.Write(path, config, new[] { "a.csv", "b.csv" },
                new Provenance.Stats(4, 100, 20, 18), "2026-01-01T00:00:00.0000000Z");
            var loaded = Provenance.LoadConfig(path);

            Assert.Equal("library_assist", loaded.TransitionRollup.Method);
            Assert.Equal(@"C:\libs\doc.blib", loaded.TransitionRollup.LibraryPath);
            Assert.Equal(4, loaded.TransitionRollup.MinTransitions);
            Assert.Equal("quantile", loaded.GlobalNormalization.Method);
            Assert.True(loaded.BatchCorrection.Enabled);
            Assert.False(loaded.BatchCorrection.PeptideLevel);
            Assert.True(loaded.BatchCorrection.ProteinLevel);
            Assert.False(loaded.Parsimony.Enabled);
            Assert.Equal("sum", loaded.ProteinRollup.Method);
            Assert.Equal("none", loaded.ProteinNormalization.Method);
            Assert.Equal("exclude", loaded.SampleOutlierDetection.Action);

            var sources = Provenance.SourceFiles(path);
            Assert.Equal(new[] { "a.csv", "b.csv" }, sources);
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }
}
