using System;
using System.Collections.Generic;
using System.Linq;
using ScottPlot;
using SkylinePrism.App;
using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// Label placement on the Dynamic Range plot. Every label used to go up-and-right at a fixed
/// offset, which is unreadable once more than a handful are shown on a dense curve: they land on
/// top of each other. These pin that the placer spreads them out, uses the space below the points
/// as well as above, and never silently drops one.
/// </summary>
public class RangeLabelPlacerTests
{
    private const double Width = 900;
    private const double Height = 600;

    private static AbundanceEntry Entry(int rank, double log10, string label) => new(
        Key: label, Label: label, Accession: null, Gene: null, ProteinName: null,
        MeanAbundance: Math.Pow(10, log10), Log10Abundance: log10, Rank: rank, SamplesUsed: 3);

    private static RangeLabelPlacer NewPlacer(int nPoints = 2000, double top = 9, double bottom = 4)
        => new(nPoints, top, bottom, Width, Height);

    /// <summary>
    /// The failure that prompted this: several labels on adjacent points. With a fixed offset they
    /// all occupy the same place; here they must not overlap at all.
    /// </summary>
    [Fact]
    public void NeighbouringPoints_GetNonOverlappingLabels()
    {
        var placer = NewPlacer();
        var boxes = new List<RangeLabelPlacer.Box>();

        // Eight consecutive points down a steep part of the curve - the crowded case.
        for (var i = 0; i < 8; i++)
            boxes.Add(placer.Place(Entry(500 + i * 3, 7.5 - i * 0.02, "GENE" + i)).Box);

        for (var a = 0; a < boxes.Count; a++)
            for (var b = a + 1; b < boxes.Count; b++)
                Assert.False(Overlaps(boxes[a], boxes[b]),
                    $"labels {a} and {b} overlap: {Describe(boxes[a])} vs {Describe(boxes[b])}");
    }

    /// <summary>
    /// The space under the curve must actually be used. Nothing forces a particular label downward,
    /// so this asserts the aggregate: with a crowd, some labels end up below their point.
    /// </summary>
    [Fact]
    public void CrowdedLabels_UseTheSpaceBelowThePointsToo()
    {
        var placer = NewPlacer();
        var below = 0;
        for (var i = 0; i < 10; i++)
        {
            var entry = Entry(400 + i * 4, 7.0 - i * 0.01, "PROT" + i);
            if (placer.Place(entry).Y < entry.Log10Abundance)
                below++;
        }

        Assert.True(below > 0, "every label was placed above its point; the space below is unused");
    }

    /// <summary>A lone label should still sit in the natural up-and-right position.</summary>
    [Fact]
    public void ASingleLabel_GoesUpAndRight()
    {
        var entry = Entry(500, 7.0, "ALB");

        var placement = NewPlacer().Place(entry);

        Assert.True(placement.X > entry.Rank);
        Assert.True(placement.Y > entry.Log10Abundance);
        Assert.Equal(Alignment.LowerLeft, placement.Alignment);
    }

    /// <summary>
    /// The alignment has to match the quadrant, or the text renders on the wrong side of its anchor
    /// and the leader line points into the middle of the label.
    /// </summary>
    [Fact]
    public void AlignmentMatchesTheQuadrantChosen()
    {
        var placer = NewPlacer();
        for (var i = 0; i < 12; i++)
        {
            var entry = Entry(300 + i * 5, 6.5 - i * 0.01, "P" + i);
            var p = placer.Place(entry);

            var expected = (p.X > entry.Rank, p.Y > entry.Log10Abundance) switch
            {
                (true, true) => Alignment.LowerLeft,
                (false, true) => Alignment.LowerRight,
                (true, false) => Alignment.UpperLeft,
                _ => Alignment.UpperRight,
            };
            Assert.Equal(expected, p.Alignment);
        }
    }

    /// <summary>
    /// A point at the right-hand edge must not push its label off the canvas - that is a label the
    /// user asked for and cannot see.
    /// </summary>
    [Fact]
    public void LabelsStayInsideThePlottedRange()
    {
        const int n = 2000;
        var placer = NewPlacer(n);

        foreach (var rank in new[] { 0, 1, n / 2, n - 2, n - 1 })
        {
            var p = placer.Place(Entry(rank, 6.0, "EDGE"));
            Assert.InRange(p.X, 0, n);
        }
    }

    /// <summary>
    /// When there is genuinely no clear spot, a label is still placed - crowding beats dropping a
    /// label silently. 60 labels on 60 adjacent points cannot all be clear.
    /// </summary>
    [Fact]
    public void AnImpossibleCrowdStillPlacesEveryLabel()
    {
        var placer = NewPlacer();
        var placed = 0;
        for (var i = 0; i < 60; i++)
        {
            var p = placer.Place(Entry(1000 + i, 6.0, "LONGPROTEINNAME" + i));
            Assert.False(double.IsNaN(p.X) || double.IsNaN(p.Y));
            placed++;
        }
        Assert.Equal(60, placed);
    }

    /// <summary>
    /// Spacing is computed in pixels and converted through the plot's extents, so the same labels on
    /// the same points must come out the same however wide the abundance range is.
    /// </summary>
    [Fact]
    public void SpacingIsIndependentOfTheAbundanceRange()
    {
        var narrow = new RangeLabelPlacer(2000, 6.2, 6.0, Width, Height);
        var wide = new RangeLabelPlacer(2000, 12.0, 2.0, Width, Height);

        var a = narrow.Place(Entry(500, 6.1, "GENE"));
        var b = wide.Place(Entry(500, 7.0, "GENE"));

        // Same fraction of the y range in both cases.
        Assert.Equal((a.Y - 6.1) / 0.2, (b.Y - 7.0) / 10.0, 6);
        Assert.Equal(a.X, b.X, 6);
    }

    private static bool Overlaps(RangeLabelPlacer.Box a, RangeLabelPlacer.Box b)
        => Math.Min(a.X2, b.X2) > Math.Max(a.X1, b.X1)
           && Math.Min(a.Y2, b.Y2) > Math.Max(a.Y1, b.Y1);

    private static string Describe(RangeLabelPlacer.Box b)
        => $"[{b.X1:0.#}-{b.X2:0.#} x {b.Y1:0.###}-{b.Y2:0.###}]";
}
