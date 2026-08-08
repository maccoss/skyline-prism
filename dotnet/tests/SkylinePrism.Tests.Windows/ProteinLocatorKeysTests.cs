using System;
using System.Linq;
using SkylinePrism.App;
using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// Which names a protein point is looked up under when a click has to find it in the Skyline
/// document tree. Clicking a protein appeared to do nothing for some groups; the cause was that a
/// group of several proteins reports its accessions and genes as one " / "-joined string, which
/// matches no node.
/// </summary>
public class ProteinLocatorKeysTests
{
    private static AbundanceEntry Protein(string name, string? accession, string? gene) => new(
        Key: name, Label: gene ?? accession ?? name, Accession: accession, Gene: gene,
        ProteinName: name, MeanAbundance: 1e6, Log10Abundance: 6, Rank: 1, SamplesUsed: 3);

    /// <summary>The real shape that failed: PG0001 from an 11,320-group mouse cohort.</summary>
    [Fact]
    public void AMultiProteinGroup_OffersEachMemberSeparately()
    {
        var entry = Protein(
            "A0A075B5J9_MOUSE",
            "A0A075B5J9 / A0A075B5K3",
            "Igkv17-127 / Igkv17-121");

        var keys = ProteinLocatorKeys.For(entry).ToList();

        // The joined strings are still offered (harmless, and correct if a node is named that way)...
        Assert.Contains("A0A075B5J9 / A0A075B5K3", keys);
        // ...but so is each member, which is what actually matches a document node.
        Assert.Contains("A0A075B5J9", keys);
        Assert.Contains("A0A075B5K3", keys);
        Assert.Contains("Igkv17-127", keys);
        Assert.Contains("Igkv17-121", keys);
    }

    /// <summary>The leading name must be tried before the accessions, so the intended node wins.</summary>
    [Fact]
    public void TheLeadingNameIsTriedFirst()
    {
        var keys = ProteinLocatorKeys.For(Protein("ALBU_MOUSE", "P07724", "Alb")).ToList();

        Assert.Equal("ALBU_MOUSE", keys[0]);
        Assert.True(
            keys.IndexOf("P07724") < keys.IndexOf("Alb"),
            "accession should be tried before the gene name");
    }

    /// <summary>A single-protein group must not be disturbed by the splitting.</summary>
    [Fact]
    public void ASingleProteinGroupIsUnchanged()
    {
        var keys = ProteinLocatorKeys.For(Protein("A0A075B5K0_MOUSE", "A0A075B5K0", "Igkv14-126")).ToList();

        Assert.Equal(new[] { "A0A075B5K0_MOUSE", "A0A075B5K0", "Igkv14-126" }, keys.Distinct());
    }

    [Fact]
    public void MissingFieldsAreSkippedRatherThanYieldingBlanks()
    {
        var keys = ProteinLocatorKeys.For(Protein("PG0007", accession: null, gene: "")).ToList();

        Assert.Equal(new[] { "PG0007" }, keys.Distinct());
        Assert.All(keys, k => Assert.False(string.IsNullOrWhiteSpace(k)));
    }

    /// <summary>A stray separator must not produce an empty key that matches nothing usefully.</summary>
    [Fact]
    public void EmptyPartsOfAJoinedListAreDropped()
    {
        var keys = ProteinLocatorKeys.For(Protein("X_MOUSE", "P1 /  / P2", null)).ToList();

        Assert.Contains("P1", keys);
        Assert.Contains("P2", keys);
        Assert.All(keys, k => Assert.False(string.IsNullOrWhiteSpace(k)));
    }
}
