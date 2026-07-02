using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Parsimony;
using Xunit;

namespace SkylinePrism.Tests.Parsimony;

/// <summary>protein_groups.csv write/read round-trip, including CSV escaping of a comma in a field.</summary>
public class ProteinGroupsCsvTests
{
    [Fact]
    public void WriteThenRead_RoundTrips_WithCommaEscaping()
    {
        var group = new ProteinGroup
        {
            GroupId = "PG0001",
            LeadingProtein = "P1",
            LeadingUniProtId = "P1",
            LeadingGeneName = "G1",
            LeadingName = "Protein 1",
            LeadingDescription = "Histone H2A, member X", // embedded comma -> must be quoted
            MemberProteins = new List<string> { "P1", "P2" },
            SubsumedProteins = new List<string> { "P3" },
            Peptides = new List<string> { "PEPA", "PEPB" },
            UniquePeptides = new List<string> { "PEPA" },
            RazorPeptides = new List<string> { "PEPB" },
            AllMappedPeptides = new List<string> { "PEPA", "PEPB" },
        };

        var path = Path.Combine(Path.GetTempPath(), "prism_pg_" + System.Guid.NewGuid().ToString("N") + ".csv");
        try
        {
            ProteinGroupsCsv.Write(new[] { group }, path);
            var back = ProteinGroupsCsv.Read(path);

            Assert.Single(back);
            var g = back[0];
            Assert.Equal("PG0001", g.GroupId);
            Assert.Equal("P1", g.LeadingProtein);
            Assert.Equal("Histone H2A, member X", g.LeadingDescription); // comma survived the round-trip
            Assert.Equal(new[] { "P1", "P2" }, g.MemberProteins.OrderBy(x => x));
            Assert.Equal(new[] { "P3" }, g.SubsumedProteins);
            Assert.Equal(new[] { "PEPA" }, g.UniquePeptides);
            Assert.Equal(new[] { "PEPB" }, g.RazorPeptides);
            Assert.Equal(new[] { "PEPA", "PEPB" }, g.Peptides.OrderBy(x => x));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void Read_EmptyFile_ReturnsNoGroups()
    {
        var path = Path.Combine(Path.GetTempPath(), "prism_pg_empty_" + System.Guid.NewGuid().ToString("N") + ".csv");
        File.WriteAllText(path, "");
        try
        {
            Assert.Empty(ProteinGroupsCsv.Read(path));
        }
        finally { File.Delete(path); }
    }
}
