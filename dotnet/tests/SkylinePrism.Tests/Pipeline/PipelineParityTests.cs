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
                         "peptides_rollup.parquet",
                         "peptides_log2_internal.parquet", "corrected_peptides.parquet",
                         "protein_groups.csv", "proteins_raw.parquet", "corrected_proteins.parquet",
                         "sample_metadata.csv",
                     })
                Assert.True(File.Exists(Path.Combine(tempOut, f)), $"missing output: {f}");

            // The merged data is a partition directory, not a file - checked through MergedDataset so
            // this test says which layout it expects rather than just that "something" is there.
            Assert.True(MergedDataset.Exists(Path.Combine(tempOut, "merged_data")), "missing merged_data");

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

            // ComBat-corrected PEPTIDE output: relative tolerance (ill-conditioned mini-fixture ComBat).
            CompareWide(Path.Combine(GoldenOut, "corrected_peptides.parquet"),
                Path.Combine(tempOut, "corrected_peptides.parquet"), PepCol, absTol: 1e-6, relTol: 3e-2,
                metaCols: new[] { PepCol, "n_transitions", "mean_rt" });

            // The PROTEIN arm is deliberately NOT compared to Python on this fixture, and that is a
            // recorded divergence rather than a gap.
            //
            // This is the only fixture with ComBat enabled, and the two engines now disagree there by
            // design: since dotnet-v26.15.0 the C# protein arm branches from the normalized, pre-ComBat
            // peptide matrix and is corrected once, while Python feeds it the ComBat-corrected peptides
            // and corrects again. On a real 4-batch cohort the difference is a median 2.7% on
            // corrected_proteins - far outside any tolerance worth calling parity, and correcting once
            // is the better of the two (held-out QC CV 16.3% -> 12.4%, against 12.7% -> 13.0%).
            //
            // Protein-arm numeric parity with Python is still covered, exactly, by the ComBat-DISABLED
            // fixtures in PipelineMethodParityTests: five method combinations compare proteins_raw and
            // corrected_proteins at 1e-9. So what is untested here is only the interaction with ComBat,
            // which is the thing the engines intentionally no longer share. Structural agreement
            // (the same groups, the same leading proteins) is asserted above and still holds.
            Assert.True(
                File.Exists(Path.Combine(tempOut, "corrected_proteins.parquet")),
                "corrected_proteins.parquet should still be produced even though it is not compared here.");
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
