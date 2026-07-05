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
/// Layer 9 end-to-end: PrismPipeline.Run on the mini plate CSVs with the e2e-sum config
/// reproduces the golden outputs. peptides_rollup and protein_groups are exact; the
/// ComBat-corrected outputs (corrected_peptides/proteins) match within a tolerance
/// dominated by the mini fixture's ill-conditioned peptide ComBat (see ComBat notes),
/// which propagates through the protein arm. Real datasets stay far tighter.
/// </summary>
public class PipelineParityTests
{
    private const string PepCol = "Peptide Modified Sequence Unimod Ids";
    private static string MergeDir => Fixtures.Path2("mini", "merge");
    private static string E2eDir => Fixtures.Path2("mini", "e2e-sum");
    private static string GoldenOut => Path.Combine(E2eDir, "output");

    [Fact]
    public void FullRun_ReproducesGoldenOutputs()
    {
        var inputs = new[]
        {
            Path.Combine(MergeDir, "mini_plate1.csv"),
            Path.Combine(MergeDir, "mini_plate2.csv"),
        };
        var config = PrismConfig.Load(Path.Combine(E2eDir, "config.yaml"));

        var tempOut = Path.Combine(Path.GetTempPath(), "prism_e2e_" + Guid.NewGuid().ToString("N"));
        try
        {
            var result = PrismPipeline.Run(inputs, tempOut, config);
            Assert.Equal(5, result.NPeptides);
            Assert.Equal(3, result.NProteins);
            Assert.Equal(166, result.NSamples);
            Assert.Equal(new[] { "mini_plate1", "mini_plate2" }, result.Batches);

            foreach (var f in new[]
                     {
                         "merged_data.parquet", "peptides_rollup.parquet",
                         "peptides_log2_internal.parquet", "corrected_peptides.parquet",
                         "protein_groups.csv", "proteins_raw.parquet", "corrected_proteins.parquet",
                         "sample_metadata.csv",
                     })
                Assert.True(File.Exists(Path.Combine(tempOut, f)), $"missing output: {f}");

            // peptides_rollup: exact (pre-ComBat).
            CompareWide(Path.Combine(GoldenOut, "peptides_rollup.parquet"),
                Path.Combine(tempOut, "peptides_rollup.parquet"), PepCol, absTol: 1e-9, relTol: 1e-9,
                metaCols: new[] { PepCol, "n_transitions", "mean_rt" });

            // protein_groups: exact structural.
            var gGroups = ProteinGroupsCsv.Read(Path.Combine(GoldenOut, "protein_groups.csv"))
                .ToDictionary(g => g.GroupId);
            var aGroups = ProteinGroupsCsv.Read(Path.Combine(tempOut, "protein_groups.csv"))
                .ToDictionary(g => g.GroupId);
            Assert.Equal(gGroups.Keys.OrderBy(x => x), aGroups.Keys.OrderBy(x => x));
            foreach (var (id, g) in gGroups)
                Assert.Equal(g.LeadingProtein, aGroups[id].LeadingProtein);

            // ComBat-corrected outputs: relative tolerance (ill-conditioned mini-fixture ComBat).
            CompareWide(Path.Combine(GoldenOut, "corrected_peptides.parquet"),
                Path.Combine(tempOut, "corrected_peptides.parquet"), PepCol, absTol: 1e-6, relTol: 3e-2,
                metaCols: new[] { PepCol, "n_transitions", "mean_rt" });
            CompareWide(Path.Combine(GoldenOut, "corrected_proteins.parquet"),
                Path.Combine(tempOut, "corrected_proteins.parquet"), "protein_group", absTol: 1e-6, relTol: 3e-2,
                metaCols: ProteinMeta);
        }
        finally
        {
            if (Directory.Exists(tempOut))
                Directory.Delete(tempOut, recursive: true);
        }
    }

    private static readonly string[] ProteinMeta =
    {
        "protein_group", "leading_protein", "leading_name", "leading_uniprot_id",
        "leading_gene_name", "leading_description", "n_peptides", "n_unique_peptides", "low_confidence",
    };

    private static void CompareWide(string goldenPath, string actualPath, string keyCol,
        double absTol, double relTol, IReadOnlyList<string> metaCols)
    {
        var golden = ParquetTable.Load(goldenPath);
        var actual = ParquetTable.Load(actualPath);
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
                Assert.True(diff <= tol, $"mismatch {gKeys[gi]}/{col}: {e} vs {a} (|d|={diff}, tol={tol})");
            }
        }
    }
}
