using System;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.Pipeline;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.Pipeline;

/// <summary>
/// Re-running into the same output directory must reuse the stages whose inputs and settings have not
/// changed - and must produce the SAME NUMBERS as a run that recomputed everything. A cache that is
/// merely fast is worthless; the property being tested is that it changes nothing observable.
/// </summary>
public class StageCacheReuseTests
{
    private static string[] Inputs()
    {
        var mergeDir = Fixtures.Path2("mini", "merge");
        return new[]
        {
            Path.Combine(mergeDir, "mini_plate1.csv"),
            Path.Combine(mergeDir, "mini_plate2.csv"),
        };
    }

    private static PrismConfig Config()
    {
        var c = PrismConfig.Load(Path.Combine(Fixtures.Path2("mini", "e2e-medpolish"), "config.yaml"));
        c.QcReport.Enabled = false;
        return c;
    }

    private static byte[] Hash(string path) =>
        System.Security.Cryptography.SHA256.HashData(File.ReadAllBytes(path));

    [Fact]
    public void ASecondRunWithNoChanges_ReusesEveryStage_AndProducesIdenticalOutputs()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_cache_" + Guid.NewGuid().ToString("N"));
        try
        {
            var log1 = new System.Collections.Generic.List<string>();
            PrismPipeline.Run(Inputs(), dir, Config(), null, log1.Add);

            var before = new[] { "corrected_peptides.parquet", "corrected_proteins.parquet" }
                .ToDictionary(f => f, f => Hash(Path.Combine(dir, f)));

            var log2 = new System.Collections.Generic.List<string>();
            PrismPipeline.Run(Inputs(), dir, Config(), null, log2.Add);

            // Every expensive stage announced reuse rather than recomputing.
            var second = string.Join("\n", log2);
            Assert.Contains("Reusing peptides_rollup.parquet", second);
            Assert.Contains("Reusing protein_groups.csv", second);
            Assert.Contains("Reusing peptides_log2_internal", second);
            Assert.Contains("Reusing proteins_raw.parquet", second);
            Assert.Contains("Reusing corrected_proteins", second);

            // ...and the outputs are byte-identical to the run that computed them.
            foreach (var (file, hash) in before)
                Assert.Equal(hash, Hash(Path.Combine(dir, file)));
        }
        finally
        {
            if (Directory.Exists(dir))
                Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void ChangingTheProteinRollup_ReusesThePeptideArm_AndRecomputesTheProteinArm()
    {
        // The case this exists for: one setting changed, and only the stages downstream of it re-run.
        var dir = Path.Combine(Path.GetTempPath(), "prism_cache_" + Guid.NewGuid().ToString("N"));
        try
        {
            PrismPipeline.Run(Inputs(), dir, Config(), null, _ => { });
            var peptidesBefore = Hash(Path.Combine(dir, "corrected_peptides.parquet"));
            var proteinsBefore = Hash(Path.Combine(dir, "corrected_proteins.parquet"));

            var changed = Config();
            changed.ProteinRollup.Method = "sum";
            var log = new System.Collections.Generic.List<string>();
            PrismPipeline.Run(Inputs(), dir, changed, null, log.Add);
            var second = string.Join("\n", log);

            // The peptide arm is untouched by a protein-rollup change: reused, and byte-identical.
            Assert.Contains("Reusing peptides_rollup.parquet", second);
            Assert.Contains("Reusing peptides_log2_internal", second);
            Assert.Contains("Reusing protein_groups.csv", second);
            Assert.Equal(peptidesBefore, Hash(Path.Combine(dir, "corrected_peptides.parquet")));

            // The protein arm re-ran, and the numbers moved because the method did.
            Assert.DoesNotContain("Reusing proteins_raw.parquet", second);
            Assert.DoesNotContain("Reusing corrected_proteins", second);
            Assert.NotEqual(proteinsBefore, Hash(Path.Combine(dir, "corrected_proteins.parquet")));
        }
        finally
        {
            if (Directory.Exists(dir))
                Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void ChangingTheTransitionRollup_RecomputesEverythingDownstream()
    {
        // Invalidation has to CHAIN: a change at the bottom cannot leave a stale peptide matrix or a
        // stale protein matrix built from it.
        var dir = Path.Combine(Path.GetTempPath(), "prism_cache_" + Guid.NewGuid().ToString("N"));
        try
        {
            PrismPipeline.Run(Inputs(), dir, Config(), null, _ => { });
            var proteinsBefore = Hash(Path.Combine(dir, "corrected_proteins.parquet"));

            var changed = Config();
            changed.TransitionRollup.Method = "sum";
            var log = new System.Collections.Generic.List<string>();
            PrismPipeline.Run(Inputs(), dir, changed, null, log.Add);
            var second = string.Join("\n", log);

            Assert.DoesNotContain("Reusing peptides_rollup.parquet", second);
            Assert.DoesNotContain("Reusing peptides_log2_internal", second);
            Assert.DoesNotContain("Reusing proteins_raw.parquet", second);
            Assert.NotEqual(proteinsBefore, Hash(Path.Combine(dir, "corrected_proteins.parquet")));

            // Parsimony reads the merged data, not the peptide matrix, so it correctly survives.
            Assert.Contains("Reusing protein_groups.csv", second);
        }
        finally
        {
            if (Directory.Exists(dir))
                Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void DeletingAnOutputForcesItsStageToRecompute()
    {
        // The cache vouches for files, not just fingerprints: one deleted by hand must come back.
        var dir = Path.Combine(Path.GetTempPath(), "prism_cache_" + Guid.NewGuid().ToString("N"));
        try
        {
            PrismPipeline.Run(Inputs(), dir, Config(), null, _ => { });
            File.Delete(Path.Combine(dir, "peptides_rollup.parquet"));

            var log = new System.Collections.Generic.List<string>();
            PrismPipeline.Run(Inputs(), dir, Config(), null, log.Add);

            Assert.DoesNotContain("Reusing peptides_rollup.parquet", string.Join("\n", log));
            Assert.True(File.Exists(Path.Combine(dir, "peptides_rollup.parquet")));
        }
        finally
        {
            if (Directory.Exists(dir))
                Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void ForceReprocess_ReusesNothing()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_cache_" + Guid.NewGuid().ToString("N"));
        try
        {
            PrismPipeline.Run(Inputs(), dir, Config(), null, _ => { });

            var log = new System.Collections.Generic.List<string>();
            PrismPipeline.Run(Inputs(), dir, Config(), null, log.Add, forceReprocess: true);

            Assert.DoesNotContain("Reusing", string.Join("\n", log));
        }
        finally
        {
            if (Directory.Exists(dir))
                Directory.Delete(dir, recursive: true);
        }
    }
}
