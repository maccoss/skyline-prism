using System;
using System.IO;
using System.Linq;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Parsimony;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.Parsimony;

/// <summary>
/// Layer 6 parity: ParsimonyEngine.Run on the merged fixture reproduces the golden
/// protein_groups.csv (GroupIDs, leading metadata, peptide lists, counts). Exercises the
/// Skyline CSV-based peptide->protein map + the greedy razor / grouping algorithm.
/// </summary>
public class ParsimonyParityTests
{
    private static string E2eDir => Fixtures.Path2("mini", "e2e-sum", "output");

    [Fact]
    public void ComputeProteinGroups_MatchesGolden()
    {
        var merged = Path.Combine(E2eDir, "merged_data.parquet");
        var golden = ProteinGroupsCsv.Read(Path.Combine(E2eDir, "protein_groups.csv"));

        var cols = SkylineColumns.Detect(ParquetTable.Load(merged).ColumnNames.ToHashSet());
        var actual = ParsimonyEngine.Run(merged, cols);

        var goldenById = golden.ToDictionary(g => g.GroupId, StringComparer.Ordinal);
        var actualById = actual.ToDictionary(g => g.GroupId, StringComparer.Ordinal);

        Assert.Equal(
            goldenById.Keys.OrderBy(x => x, StringComparer.Ordinal),
            actualById.Keys.OrderBy(x => x, StringComparer.Ordinal));

        foreach (var (id, g) in goldenById)
        {
            var a = actualById[id];
            Assert.Equal(g.LeadingProtein, a.LeadingProtein);
            Assert.Equal(g.LeadingUniProtId, a.LeadingUniProtId);
            Assert.Equal(g.LeadingGeneName, a.LeadingGeneName);
            Assert.Equal(g.LeadingName, a.LeadingName);
            Assert.Equal(g.LeadingDescription, a.LeadingDescription);
            AssertSameSet(g.UniquePeptides, a.UniquePeptides, $"{id} unique");
            AssertSameSet(g.RazorPeptides, a.RazorPeptides, $"{id} razor");
            AssertSameSet(g.Peptides, a.Peptides, $"{id} all");
        }
    }

    [Fact]
    public void Write_RoundTrips_ToGoldenCsvContent()
    {
        var merged = Path.Combine(E2eDir, "merged_data.parquet");
        var cols = SkylineColumns.Detect(ParquetTable.Load(merged).ColumnNames.ToHashSet());
        var groups = ParsimonyEngine.Run(merged, cols);

        var tempCsv = Path.Combine(Path.GetTempPath(), "pg_" + Guid.NewGuid().ToString("N") + ".csv");
        try
        {
            ProteinGroupsCsv.Write(groups, tempCsv);
            var reread = ProteinGroupsCsv.Read(tempCsv);
            Assert.Equal(groups.Count, reread.Count);
            for (var i = 0; i < groups.Count; i++)
                Assert.Equal(groups[i].GroupId, reread[i].GroupId);
        }
        finally
        {
            if (File.Exists(tempCsv))
                File.Delete(tempCsv);
        }
    }

    private static void AssertSameSet(System.Collections.Generic.IEnumerable<string> expected,
        System.Collections.Generic.IEnumerable<string> actual, string what)
    {
        Assert.Equal(
            expected.OrderBy(x => x, StringComparer.Ordinal),
            actual.OrderBy(x => x, StringComparer.Ordinal));
    }
}
