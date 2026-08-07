using System;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Pipeline;
using SkylinePrism.Core.Qc;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// The Dynamic Range tab reads <c>corrected_proteins.parquet</c> / <c>corrected_peptides.parquet</c>
/// and finds its replicates by taking every column that is not a known metadata column. Stage 2b/2c
/// now writes those files a row group at a time through <see cref="StreamingWideWriter"/> rather than
/// in one shot, so this pins that the tab still sees the same schema - an extra or renamed metadata
/// column would leave the replicate list empty and the plot blank, with no error anywhere.
/// </summary>
public class DynamicRangeReadsStreamedOutputTests
{
    [Theory]
    [InlineData(AbundanceLevel.Protein, "corrected_proteins.parquet")]
    [InlineData(AbundanceLevel.Peptide, "corrected_peptides.parquet")]
    public void CorrectedMatrix_ExposesItsReplicatesToTheDynamicRangeTab(AbundanceLevel level, string file)
    {
        var mergeDir = Fixtures.Path2("mini", "merge");
        var dir = Fixtures.Path2("mini", "e2e-sum");
        var config = PrismConfig.Load(Path.Combine(dir, "config.yaml"));
        var outDir = Path.Combine(Path.GetTempPath(), "prism_dr_" + Guid.NewGuid().ToString("N"));

        try
        {
            PrismPipeline.Run(
                new[]
                {
                    Path.Combine(mergeDir, "mini_plate1.csv"),
                    Path.Combine(mergeDir, "mini_plate2.csv"),
                },
                outDir, config);

            var table = ParquetTable.Load(Path.Combine(outDir, file));
            var samples = DynamicRange.SampleColumns(table, level);

            // The fixture has 166 replicates; an empty list here is exactly the blank-plot symptom.
            Assert.Equal(166, samples.Count);
            Assert.All(samples, s => Assert.Contains("__@__", s, StringComparison.Ordinal));

            var entries = DynamicRange.Compute(table, level, samples);
            Assert.NotEmpty(entries);
            Assert.All(entries, e => Assert.True(
                e.MeanAbundance > 0 && !double.IsNaN(e.Log10Abundance),
                $"{e.Label} has no usable abundance - the tab would drop it"));

            // Ranked most abundant first, 0-based and contiguous: the plot's x axis.
            Assert.Equal(entries.OrderByDescending(e => e.Log10Abundance).Select(e => e.Key),
                entries.Select(e => e.Key));
        }
        finally
        {
            if (Directory.Exists(outDir))
                Directory.Delete(outDir, recursive: true);
        }
    }
}
