using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// The Dynamic Range plot's data: rank vs log10 abundance over the CORRECTED matrices, and the
/// user-defined protein lists that highlight sets of interest on it.
/// </summary>
public class DynamicRangeTests
{
    /// <summary>
    /// Write a corrected_proteins-shaped parquet: the real output format, LINEAR values, NaN for a
    /// protein that was never measured.
    /// </summary>
    private static string WriteProteins(
        string dir, string[] genes, string[] accessions, params double[][] rows)
    {
        var path = Path.Combine(dir, "corrected_proteins.parquet");
        var n = genes.Length;
        var sampleNames = Enumerable.Range(1, rows.Length).Select(i => "R" + i).ToList();
        var meta = new List<ParquetWideWriter.MetaColumn>
        {
            ParquetWideWriter.Strings("protein_group",
                Enumerable.Range(1, n).Select(i => $"PG{i:D4}").ToArray()),
            ParquetWideWriter.Strings("leading_protein", accessions),
            ParquetWideWriter.Strings("leading_name",
                accessions.Zip(genes, (a, g) => $"sp|{a}|{g}_HUMAN").ToArray()),
            ParquetWideWriter.Strings("leading_uniprot_id", accessions),
            ParquetWideWriter.Strings("leading_gene_name", genes),
        };
        ParquetWideWriter.Write(path, meta, sampleNames, rows, n);
        return path;
    }

    private static string WriteDefaultProteins(string dir) => WriteProteins(
        dir,
        new[] { "ALB", "HBB", "CD9", "IKBKG" },
        new[] { "P02768", "P68871", "P21926", "Q9Y6K9" },
        new[] { 1000000.0, 100000, 1000, double.NaN },   // R1
        new[] { 1200000.0, 90000, 1200, double.NaN },    // R2
        new[] { 1100000.0, 110000, 800, double.NaN });   // R3

    [Fact]
    public void Compute_RanksByMeanLinearAbundanceDescending()
    {
        var dir = TempDir();
        try
        {
            var table = ParquetTable.Load(WriteDefaultProteins(dir));
            var entries = DynamicRange.Compute(table, AbundanceLevel.Protein);

            // The unmeasured protein is dropped, not plotted at zero.
            Assert.Equal(3, entries.Count);
            Assert.Equal(new[] { 1, 2, 3 }, entries.Select(e => e.Rank));
            Assert.Equal(new[] { "ALB", "HBB", "CD9" }, entries.Select(e => e.Label));

            // Mean is on the LINEAR scale, then log10 - not a mean of logs.
            var alb = entries[0];
            Assert.Equal((1000000 + 1200000 + 1100000) / 3.0, alb.MeanAbundance, 6);
            Assert.Equal(Math.Log10(alb.MeanAbundance), alb.Log10Abundance, 10);
            Assert.Equal(3, alb.SamplesUsed);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Compute_HonoursAReplicateSubsetAndCanReorderTheRanking()
    {
        var dir = TempDir();
        try
        {
            var path = WriteProteins(
                dir, new[] { "AAA", "BBB" }, new[] { "P1", "P2" },
                new[] { 100.0, 1000 },      // R1
                new[] { 10000.0, 1000 });   // R2
            var table = ParquetTable.Load(path);

            // Averaged over both replicates AAA wins; over R1 alone BBB does. The ranking has to follow
            // the selection, which is why changing replicates recomputes rather than just redraws.
            Assert.Equal("AAA", DynamicRange.Compute(table, AbundanceLevel.Protein)[0].Label);
            Assert.Equal("BBB", DynamicRange.Compute(table, AbundanceLevel.Protein, new[] { "R1" })[0].Label);
            Assert.Equal(1, DynamicRange.Compute(table, AbundanceLevel.Protein, new[] { "R1" })[0].SamplesUsed);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Compute_MissingValuesDoNotDragTheMeanDown()
    {
        var dir = TempDir();
        try
        {
            var path = WriteProteins(
                dir, new[] { "AAA" }, new[] { "P1" },
                new[] { 1000.0 }, new[] { double.NaN }, new[] { 1000.0 });
            var entry = DynamicRange.Compute(ParquetTable.Load(path), AbundanceLevel.Protein).Single();

            // Mean over the TWO replicates that measured it, not over three with a zero.
            Assert.Equal(1000, entry.MeanAbundance, 6);
            Assert.Equal(2, entry.SamplesUsed);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void SampleColumns_ExcludeTheMetadataColumns()
    {
        var dir = TempDir();
        try
        {
            var table = ParquetTable.Load(WriteDefaultProteins(dir));
            Assert.Equal(new[] { "R1", "R2", "R3" }, DynamicRange.SampleColumns(table, AbundanceLevel.Protein));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Compute_PeptideLevelKeysOnTheModifiedSequence()
    {
        var dir = TempDir();
        try
        {
            var path = Path.Combine(dir, "corrected_peptides.parquet");
            ParquetWideWriter.Write(
                path,
                new[]
                {
                    ParquetWideWriter.Strings("Peptide Modified Sequence Unimod Ids",
                        new[] { "PEPTIDEC(unimod:4)K", "ANOTHERPEPTIDER" }),
                    ParquetWideWriter.Longs("n_transitions", new long[] { 6, 4 }),
                    ParquetWideWriter.Doubles("mean_rt", new[] { 15.2, 22.8 }),
                },
                new[] { "R1", "R2" },
                new[] { new[] { 5000.0, 900 }, new[] { 5200.0, 1100 } },
                2);
            var entries = DynamicRange.Compute(ParquetTable.Load(path), AbundanceLevel.Peptide);

            Assert.Equal(2, entries.Count);
            Assert.Equal("PEPTIDEC(unimod:4)K", entries[0].Key);
            Assert.Equal("PEPTIDEC(unimod:4)K", entries[0].Label); // no gene at peptide level
            // n_transitions / mean_rt must not be mistaken for replicates.
            Assert.Equal(2, entries[0].SamplesUsed);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Compute_PeptideLevelReadsTheProteinGroupsStampedOnTheOutput()
    {
        // corrected_peptides carries the groups each peptide maps to, so a peptide can be navigated to in
        // Skyline and joined to corrected_proteins. A SHARED peptide lists every group it maps to.
        var dir = TempDir();
        try
        {
            var path = Path.Combine(dir, "corrected_peptides.parquet");
            ParquetWideWriter.Write(
                path,
                new[]
                {
                    ParquetWideWriter.Strings("Peptide Modified Sequence Unimod Ids",
                        new[] { "UNIQUEPEPTIDEK", "SHAREDPEPTIDEK" }),
                    ParquetWideWriter.Longs("n_transitions", new long[] { 6, 5 }),
                    ParquetWideWriter.Doubles("mean_rt", new[] { 15.2, 18.4 }),
                    ParquetWideWriter.Strings("protein_group", new[] { "PG0001", "PG0002;PG0007" }),
                    ParquetWideWriter.Strings("leading_protein", new[] { "P02768", "P68871;Q9Y6K9" }),
                    ParquetWideWriter.Strings("leading_name",
                        new[] { "sp|P02768|ALBU_HUMAN", "sp|P68871|HBB_HUMAN;sp|Q9Y6K9|NEMO_HUMAN" }),
                    ParquetWideWriter.Strings("leading_gene_name", new[] { "ALB", "HBB;IKBKG" }),
                },
                new[] { "R1", "R2" },
                new[] { new[] { 5000.0, 900 }, new[] { 5200.0, 1100 } },
                2);

            var table = ParquetTable.Load(path);
            // The grouping columns must not be mistaken for replicates.
            Assert.Equal(new[] { "R1", "R2" }, DynamicRange.SampleColumns(table, AbundanceLevel.Peptide));

            var entries = DynamicRange.Compute(table, AbundanceLevel.Peptide);
            var unique = entries.Single(e => e.Key == "UNIQUEPEPTIDEK");
            var sharedPeptide = entries.Single(e => e.Key == "SHAREDPEPTIDEK");

            // Every group is carried, so the UI can report all proteins a peptide is present in.
            Assert.Equal(new[] { "PG0002", "PG0007" }, sharedPeptide.ProteinGroups);
            Assert.Equal(new[] { "sp|P68871|HBB_HUMAN", "sp|Q9Y6K9|NEMO_HUMAN" }, sharedPeptide.ProteinNames);
            Assert.True(sharedPeptide.IsShared);
            Assert.Equal(new[] { "PG0001" }, unique.ProteinGroups);
            Assert.False(unique.IsShared);

            Assert.Equal("sp|P02768|ALBU_HUMAN", unique.ProteinName);
            Assert.Equal("P02768", unique.Accession);
            Assert.Equal("ALB", unique.Gene);
            // A shared peptide resolves to its FIRST group for navigation and list matching.
            Assert.Equal("sp|P68871|HBB_HUMAN", sharedPeptide.ProteinName);
            Assert.Equal("P68871", sharedPeptide.Accession);
            Assert.Equal("HBB", sharedPeptide.Gene);
            // Peptides stay labelled by sequence - gene labels would collide across a protein's peptides.
            Assert.Equal("SHAREDPEPTIDEK", sharedPeptide.Label);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Compute_ProteinLevelCarriesItsOwnGroup()
    {
        var dir = TempDir();
        try
        {
            var entries = DynamicRange.Compute(
                ParquetTable.Load(WriteDefaultProteins(dir)), AbundanceLevel.Protein);
            var alb = entries[0];
            // A protein row is one group; the same fields are populated so the UI needs no special case.
            Assert.Equal(new[] { "PG0001" }, alb.ProteinGroups);
            Assert.Equal(new[] { "sp|P02768|ALB_HUMAN" }, alb.ProteinNames); // as WriteProteins builds it
            Assert.False(alb.IsShared);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Theory]
    [InlineData("PG0002;PG0007", "PG0002")]
    [InlineData("PG0002", "PG0002")]
    [InlineData(" PG0002 ; PG0007 ", "PG0002")]
    [InlineData("", null)]
    [InlineData(null, null)]
    public void FirstGroup_TakesTheLeadingEntryOfAMultiGroupValue(string? value, string? expected)
        => Assert.Equal(expected, DynamicRange.FirstGroup(value));

    [Fact]
    public void ProteinList_MatchesASharedPeptidesFirstGroup()
    {
        // A list of EV markers should still light up peptides of an EV protein, including shared ones.
        var entry = new AbundanceEntry(
            "SHAREDPEPTIDEK", "SHAREDPEPTIDEK", "P68871", "HBB", "sp|P68871|HBB_HUMAN", 1e3, 3, 5, 2);
        var set = new ProteinListSet { Lists = { new ProteinList { Name = "EV", Members = { "HBB" } } } };
        Assert.Equal("EV", set.BuildMatcher().Match(entry)?.Name);
    }

    [Theory]
    // A curated list may be keyed on any of these; all must find the same protein.
    [InlineData("P02768")]
    [InlineData("p02768")]              // case-insensitive
    [InlineData("ALB")]                 // gene name
    [InlineData("sp|P02768|ALBU_HUMAN")]// full protein name
    [InlineData("ALBU_HUMAN")]          // name without the sp| prefix
    [InlineData("ALBU")]                // species suffix dropped
    [InlineData("P02768-2")]            // isoform suffix
    public void ProteinList_MatchesWhicheverIdentifierTheListUses(string member)
    {
        var entry = new AbundanceEntry(
            "sp|P02768|ALBU_HUMAN", "ALB", "P02768", "ALB", "sp|P02768|ALBU_HUMAN", 1e6, 6, 1, 3);
        var set = new ProteinListSet
        {
            Lists = { new ProteinList { Name = "Plasma", ColorHex = "#d62728", Members = { member } } },
        };

        Assert.Equal("Plasma", set.BuildMatcher().Match(entry)?.Name);
    }

    [Fact]
    public void ProteinList_DoesNotMatchUnrelatedProteins()
    {
        var entry = new AbundanceEntry(
            "sp|P68871|HBB_HUMAN", "HBB", "P68871", "HBB", "sp|P68871|HBB_HUMAN", 1e5, 5, 2, 3);
        var set = new ProteinListSet
        {
            Lists = { new ProteinList { Name = "Plasma", Members = { "P02768", "ALB" } } },
        };
        Assert.Null(set.BuildMatcher().Match(entry));
    }

    [Fact]
    public void ProteinList_HiddenListsAreNotMatchedAndOrderIsPriority()
    {
        var entry = new AbundanceEntry("sp|P1|A", "AAA", "P1", "AAA", "sp|P1|A", 1, 0, 1, 1);
        var set = new ProteinListSet
        {
            Lists =
            {
                new ProteinList { Name = "Hidden", Visible = false, Members = { "P1" } },
                new ProteinList { Name = "First visible", Members = { "P1" } },
                new ProteinList { Name = "Second visible", Members = { "P1" } },
            },
        };

        // Hidden lists drop out entirely; of the visible ones the earlier wins a shared protein.
        var matcher = set.BuildMatcher();
        Assert.Equal(2, matcher.Lists.Count);
        Assert.Equal("First visible", matcher.Match(entry)?.Name);
    }

    [Fact]
    public void ProteinListSet_RoundTripsThroughJson()
    {
        var dir = TempDir();
        try
        {
            var path = Path.Combine(dir, ProteinListSet.FileName);
            var set = new ProteinListSet
            {
                Lists =
                {
                    new ProteinList
                    {
                        Name = "EV markers", ColorHex = "#1f77b4", Visible = true, ShowLabels = true,
                        Members = { "CD9", "CD63", "CD81" },
                    },
                    new ProteinList { Name = "Contaminants", ColorHex = "#d62728", Visible = false },
                },
            };
            set.Save(path);

            var loaded = ProteinListSet.Load(path);
            Assert.Equal(2, loaded.Lists.Count);
            Assert.Equal("EV markers", loaded.Lists[0].Name);
            Assert.Equal("#1f77b4", loaded.Lists[0].ColorHex);
            Assert.True(loaded.Lists[0].ShowLabels);
            Assert.Equal(new[] { "CD9", "CD63", "CD81" }, loaded.Lists[0].Members);
            Assert.False(loaded.Lists[1].Visible);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void ProteinListSet_LoadSurvivesAMissingOrCorruptFile()
    {
        // A broken file must not stop the tool opening - the user just starts with no lists.
        var dir = TempDir();
        try
        {
            Assert.Empty(ProteinListSet.Load(Path.Combine(dir, "nope.json")).Lists);
            var corrupt = Path.Combine(dir, "corrupt.json");
            File.WriteAllText(corrupt, "{ not json at all ");
            Assert.Empty(ProteinListSet.Load(corrupt).Lists);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void ReadMembersFile_HandlesTheShapesCuratedListsArriveIn()
    {
        var dir = TempDir();
        try
        {
            var path = Path.Combine(dir, "markers.csv");
            File.WriteAllLines(path, new[]
            {
                "Accession,Gene,Note",     // header row, skipped
                "P02768,ALB,plasma",       // first column wins
                "",                        // blank
                "# comment",               // comment
                "\"P68871\",HBB,",         // quoted
                "P02768,ALB,duplicate",    // de-duplicated
            });

            Assert.Equal(new[] { "P02768", "P68871" }, ProteinListSet.ReadMembersFile(path));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    private static string TempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_dynrange_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }
}
