using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Parsimony;
using SkylinePrism.Core.Pipeline;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.Pipeline;

/// <summary>
/// Cross-language exact parity for the rollup methods that e2e-sum doesn't exercise: median_polish
/// (transition + protein), maxLFQ, topN (both stages), and consensus. Each fixture disables ComBat
/// so the whole pipeline is deterministic and every stage - peptides_rollup, proteins_raw, and the
/// corrected outputs - matches the Python golden to 1e-9.
/// </summary>
public class PipelineMethodParityTests
{
    private const string PepCol = "Peptide Modified Sequence Unimod Ids";
    private static readonly string[] PepMeta = { PepCol, "n_transitions", "mean_rt" };
    private static readonly string[] ProtMeta =
    {
        "protein_group", "leading_protein", "leading_name", "leading_uniprot_id",
        "leading_gene_name", "leading_description", "n_peptides", "n_unique_peptides", "low_confidence",
    };

    [Theory]
    [InlineData("e2e-medpolish")]
    [InlineData("e2e-maxlfq")]
    [InlineData("e2e-topn")]
    [InlineData("e2e-prot-topn")]
    [InlineData("e2e-consensus")]
    public void Method_ReproducesGoldenExactly(string fixture)
    {
        var mergeDir = Fixtures.Path2("mini", "merge");
        var dir = Fixtures.Path2("mini", fixture);
        var golden = Path.Combine(dir, "output");
        var inputs = new[]
        {
            Path.Combine(mergeDir, "mini_plate1.csv"),
            Path.Combine(mergeDir, "mini_plate2.csv"),
        };
        var config = PrismConfig.Load(Path.Combine(dir, "config.yaml"));

        var tempOut = Path.Combine(Path.GetTempPath(), "prism_m_" + Guid.NewGuid().ToString("N"));
        try
        {
            PrismPipeline.Run(inputs, tempOut, config);

            // ComBat is disabled in these fixtures, so every stage is exact.
            CompareWide(golden, tempOut, "peptides_rollup.parquet", PepCol, PepMeta);       // transition method
            CompareWide(golden, tempOut, "proteins_raw.parquet", "protein_group", ProtMeta); // protein method
            CompareWide(golden, tempOut, "corrected_peptides.parquet", PepCol, PepMeta);      // normalized (linear)
            CompareWide(golden, tempOut, "corrected_proteins.parquet", "protein_group", ProtMeta);

            var gGroups = ProteinGroupsCsv.Read(Path.Combine(golden, "protein_groups.csv"))
                .ToDictionary(g => g.GroupId);
            var aGroups = ProteinGroupsCsv.Read(Path.Combine(tempOut, "protein_groups.csv"))
                .ToDictionary(g => g.GroupId);
            Assert.Equal(gGroups.Keys.OrderBy(x => x), aGroups.Keys.OrderBy(x => x));
            foreach (var (id, g) in gGroups)
                Assert.Equal(g.LeadingProtein, aGroups[id].LeadingProtein);
        }
        finally
        {
            if (Directory.Exists(tempOut))
                Directory.Delete(tempOut, recursive: true);
        }
    }

    private static void CompareWide(
        string goldenDir, string actualDir, string file, string keyCol, IReadOnlyList<string> metaCols,
        double absTol = 1e-7, double relTol = 1e-9)
    {
        var golden = ParquetTable.Load(Path.Combine(goldenDir, file));
        var actual = ParquetTable.Load(Path.Combine(actualDir, file));
        Assert.Equal(golden.RowCount, actual.RowCount);

        var sampleCols = golden.ColumnNames.Where(c => !metaCols.Contains(c)).ToList();
        var gKeys = golden.GetString(keyCol);
        var aKeys = actual.GetString(keyCol);
        var aIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var i = 0; i < aKeys.Length; i++)
            aIndex[aKeys[i]!] = i;

        var gCols = sampleCols.ToDictionary(c => c, golden.GetDouble);
        var aCols = sampleCols.ToDictionary(c => c, actual.GetDouble);

        for (var gi = 0; gi < gKeys.Length; gi++)
        {
            var ai = aIndex[gKeys[gi]!];
            foreach (var col in sampleCols)
            {
                var e = gCols[col][gi]!.Value;
                var a = aCols[col][ai]!.Value;
                var diff = Math.Abs(e - a);
                var tol = absTol + relTol * Math.Abs(e);
                Assert.True(diff <= tol, $"{file} {gKeys[gi]}/{col}: {e} vs {a} (|d|={diff}, tol={tol})");
            }
        }
    }
}
