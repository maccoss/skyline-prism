using System;
using System.IO;
using SkylinePrism.Core.Config;
using Xunit;

namespace SkylinePrism.Tests.Config;

/// <summary>Guards that the YAML keys for the newer options bind (UnderscoredNamingConvention).</summary>
public class ConfigParsingTests
{
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
