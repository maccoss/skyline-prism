using System;
using System.Collections.Generic;
using System.Linq;
using ScottPlot;
using SkylinePrism.App;
using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// Label placement on the Dynamic Range plot. Labels placed one at a time kept off each other but
/// not much else: with twenty-odd list members on the steep top of the curve, their leader lines
/// crossed, ran through other labels, and labels sat on the curve itself. These pin the layout as a
/// whole on a curve shaped like a real one.
/// </summary>
public class RangeLabelPlacerTests
{
    private const double WidthPx = 900;
    private const double HeightPx = 600;
    private const int N = 3000;
    private const double MarkerRadiusPx = 8;

    /// <summary>A ranked abundance curve: steep at the top, flattening, as a log-rank falloff gives.</summary>
    private static double Log10At(int rank) => 9 - 2.0 * Math.Log10(rank);

    private static readonly IReadOnlyList<AbundanceEntry> Curve =
        Enumerable.Range(1, N).Select(r => Entry(r, "P" + r)).ToList();

    private static AbundanceEntry Entry(int rank, string label, double? log10 = null) => new(
        Key: label, Label: label, Accession: null, Gene: null, ProteinName: null,
        MeanAbundance: Math.Pow(10, log10 ?? Log10At(rank)), Log10Abundance: log10 ?? Log10At(rank),
        Rank: rank, SamplesUsed: 3);

    private static RangeLabelPlacer NewPlacer(double yMin = 1.5, double yMax = 9.5) =>
        new(Curve, 0, N + 1, yMin, yMax, WidthPx, HeightPx, MarkerRadiusPx);

    /// <summary>
    /// The failure that prompted the rewrite: the classic plasma list's 26 members, crowded on the
    /// steep top of the curve and around its bend, where a plasma run puts them.
    /// </summary>
    private static readonly string[] PlasmaGenes =
    {
        "F2", "KNG1", "TTR", "RBP4", "HP", "ALB", "TF", "CP", "GPX3", "A2M", "GC", "CNDP1", "GPLD1",
        "HPX", "PLG", "A1BG", "ORM1", "SELENOP", "ORM2", "ADAMTS13", "FGL1", "KLKB1", "AFM", "ATRN",
        "BTD", "BCHE",
    };

    private static List<AbundanceEntry> Crowd() =>
        new[] { 1, 2, 3, 5, 7, 10, 13, 17, 22, 28, 35, 44, 55, 68, 84, 103, 126, 154, 188, 230, 280, 340, 410, 600, 900, 1500 }
            .Select((r, i) => Entry(r, PlasmaGenes[i]))
            .ToList();

    // The reported plot, as the app lays it out: a plasma run's 6,650 proteins - a steep top, a long
    // gentle slope, a drop at the tail - in a 960 x 650 pane, whose data area is 860 x 545 px with
    // 11 px point radii, the 26 classic plasma proteins at about the ranks that run put them.
    private const int PlasmaN = 6650;

    private static double PlasmaLog10At(int r) =>
        r <= 800 ? 7.6 - 1.03 * Math.Log10(r)
        : r <= 6000 ? 4.6 - 0.00025 * (r - 800)
        : 3.3 - 1.6 * Math.Pow((r - 6000) / 650.0, 2);

    private static readonly int[] PlasmaRanks =
        { 5, 25, 45, 55, 85, 140, 170, 185, 200, 215, 230, 380, 410, 430, 450, 470, 500, 540, 580, 620, 680, 780, 1530, 1750, 1850, 3800 };

    // About what ScottPlot measures for the bold 16 px label text (ALB 29.7 x 21.3, ADAMTS13 86.4),
    // fixed here so the test does not depend on the fonts installed.
    private static (double Width, double Height) MeasureLikeTheApp(string text) => (10.5 * text.Length, 21.3);

    private static (RangeLabelPlacer Placer, List<AbundanceEntry> Labels) ThePlasmaPlot()
    {
        var curve = Enumerable.Range(1, PlasmaN).Select(r => Entry(r, "P" + r, PlasmaLog10At(r))).ToList();
        var labels = PlasmaRanks.Select((r, i) => Entry(r, PlasmaGenes[i], PlasmaLog10At(r))).ToList();
        return (new RangeLabelPlacer(curve, -340, PlasmaN + 340, 1.26, 8.04, 860, 545, 11, measureText: MeasureLikeTheApp), labels);
    }

    /// <summary>
    /// The reported plot: its labels' leaders crossed and labels ran over one another. On it there
    /// must now be no crossing, no overlap, no leader through a label, and nothing on the curve.
    /// </summary>
    [Fact]
    public void ThePlasmaPlot_HasNoCrossingLeaders_NoOverlaps_AndNothingOnTheCurve()
        => AssertClean(ThePlasmaPlot().Placer, PlasmaLayout.Value);

    // Laid out once for the tests that only look at the result; annealing is the slow part.
    private static readonly Lazy<IReadOnlyList<RangeLabelPlacer.Placement>> PlasmaLayout =
        new(() => ThePlasmaPlot() is var (placer, labels) ? placer.PlaceAll(labels) : throw new InvalidOperationException());

    /// <summary>
    /// A harder crowd than any run gives - a curve that never flattens, in a smaller pane, with the
    /// generous estimated box sizes. Not every label can have clear space here, but no two leaders
    /// may cross, and every label is placed.
    /// </summary>
    [Fact]
    public void AHarderCrowd_StillHasNoCrossingLeaders()
    {
        var p = NewPlacer().PlaceAll(Crowd());

        Assert.Equal(26, p.Count);
        for (var a = 0; a < p.Count; a++)
            for (var b = a + 1; b < p.Count; b++)
                Assert.False(RangeLabelPlacer.SegmentsCross(
                        p[a].PointXPx, p[a].PointYPx, p[a].AnchorXPx, p[a].AnchorYPx,
                        p[b].PointXPx, p[b].PointYPx, p[b].AnchorXPx, p[b].AnchorYPx),
                    $"the leaders of labels {a} and {b} cross");
    }

    private static void AssertClean(RangeLabelPlacer placer, IReadOnlyList<RangeLabelPlacer.Placement> p)
    {
        for (var a = 0; a < p.Count; a++)
        {
            Assert.False(placer.CoversCurve(p[a].BoxPx), $"label {a} covers the curve");
            for (var b = a + 1; b < p.Count; b++)
            {
                Assert.False(RangeLabelPlacer.Overlaps(p[a].BoxPx, p[b].BoxPx), $"labels {a} and {b} overlap");
                Assert.False(RangeLabelPlacer.SegmentsCross(
                        p[a].PointXPx, p[a].PointYPx, p[a].AnchorXPx, p[a].AnchorYPx,
                        p[b].PointXPx, p[b].PointYPx, p[b].AnchorXPx, p[b].AnchorYPx),
                    $"the leaders of labels {a} and {b} cross");
                Assert.False(RangeLabelPlacer.SegmentHitsBox(
                    p[a].PointXPx, p[a].PointYPx, p[a].AnchorXPx, p[a].AnchorYPx, p[b].BoxPx),
                    $"label {a}'s leader runs through label {b}");
                Assert.False(RangeLabelPlacer.SegmentHitsBox(
                    p[b].PointXPx, p[b].PointYPx, p[b].AnchorXPx, p[b].AnchorYPx, p[a].BoxPx),
                    $"label {b}'s leader runs through label {a}");
            }
        }
    }

    /// <summary>
    /// A falling curve leaves open space on both sides - above and right, below and left - and a
    /// crowd needs both. All-above was what piled labels on top of each other.
    /// </summary>
    [Fact]
    public void ACrowd_UsesTheSpaceOnBothSidesOfTheCurve()
    {
        var p = PlasmaLayout.Value;

        Assert.Contains(p, q => q.AnchorXPx > q.PointXPx && q.AnchorYPx > q.PointYPx);
        Assert.Contains(p, q => q.AnchorXPx < q.PointXPx || q.AnchorYPx < q.PointYPx);
    }

    /// <summary>A lone label sits in the conventional spot: up and to the right, its corner at the leader.</summary>
    [Fact]
    public void ASingleLabel_GoesUpAndRight()
    {
        var entry = Entry(500, "ALB");

        var placement = NewPlacer().PlaceAll(new[] { entry }).Single();

        Assert.True(placement.X > entry.Rank, $"placed at {placement}");
        Assert.True(placement.Y > entry.Log10Abundance, $"placed at {placement}");
        Assert.Equal(Alignment.LowerLeft, placement.Alignment);
    }

    /// <summary>
    /// The text must extend AWAY from its point, or it renders over the point and the leader ends in
    /// the middle of the label: the anchor is on the box's edge and the point is outside the box.
    /// </summary>
    [Fact]
    public void EveryLabel_ExtendsAwayFromItsPoint()
    {
        foreach (var q in NewPlacer().PlaceAll(Crowd()))
        {
            var b = q.BoxPx;
            Assert.False(q.PointXPx > b.X1 && q.PointXPx < b.X2 && q.PointYPx > b.Y1 && q.PointYPx < b.Y2,
                "the label covers its own point");
            var onVerticalEdge = (Near(q.AnchorXPx, b.X1) || Near(q.AnchorXPx, b.X2)) && q.AnchorYPx >= b.Y1 - 1e-6 && q.AnchorYPx <= b.Y2 + 1e-6;
            var onHorizontalEdge = (Near(q.AnchorYPx, b.Y1) || Near(q.AnchorYPx, b.Y2)) && q.AnchorXPx >= b.X1 - 1e-6 && q.AnchorXPx <= b.X2 + 1e-6;
            Assert.True(onVerticalEdge || onHorizontalEdge, "the leader does not end on the label's edge");
        }
    }

    /// <summary>A label off the data area is a label the user asked for and cannot see.</summary>
    [Fact]
    public void LabelsStayInsideTheDataArea()
    {
        var edges = new[] { 1, 2, N / 2, N - 1, N }.Select(r => Entry(r, "EDGEPROTEIN")).ToList();
        foreach (var q in NewPlacer().PlaceAll(edges))
        {
            Assert.InRange(q.BoxPx.X1, 0, WidthPx);
            Assert.InRange(q.BoxPx.X2, 0, WidthPx);
            Assert.InRange(q.BoxPx.Y1, 0, HeightPx);
            Assert.InRange(q.BoxPx.Y2, 0, HeightPx);
        }
    }

    /// <summary>
    /// When there is genuinely no clear spot, every label is still placed - crowding beats silently
    /// dropping one. 60 long labels on 60 adjacent points cannot all be clear.
    /// </summary>
    [Fact]
    public void AnImpossibleCrowdStillPlacesEveryLabel()
    {
        var crowd = Enumerable.Range(0, 60).Select(i => Entry(1000 + i, "LONGPROTEINNAME" + i)).ToList();

        var p = NewPlacer().PlaceAll(crowd);

        Assert.Equal(60, p.Count);
        Assert.All(p, q => Assert.False(double.IsNaN(q.X) || double.IsNaN(q.Y)));
    }

    /// <summary>
    /// A long list - every member labeled - gets a bounded amount of work, because the layout runs
    /// on the UI thread and every step compares a spot with every other label. Without the budget,
    /// 400 labels meant 25 polish passes over 144 spots against 400 labels each.
    /// </summary>
    [Fact]
    public void ALongList_GetsABoundedAmountOfWork()
    {
        var many = Enumerable.Range(0, 400).Select(i => Entry(1 + i * 7, "PROT" + i)).ToList();

        var placer = NewPlacer();
        var p = placer.PlaceAll(many);

        Assert.Equal(400, p.Count);
        // Counted, not timed, so a slow CI machine cannot fail it and a regression cannot hide in a
        // generous time limit: the greedy start, one polish pass and its current-cost check come to
        // about 1.5 budgets.
        Assert.InRange(placer.PairEvaluations, 1, (long)(2 * RangeLabelPlacer.LargeListPairBudget));
    }

    /// <summary>The search is seeded: the same plot always gets the same labels, redraw after redraw.</summary>
    [Fact]
    public void TheLayoutIsTheSameEveryTime()
    {
        var (placer, labels) = ThePlasmaPlot();
        var first = PlasmaLayout.Value;
        var second = placer.PlaceAll(labels);

        Assert.Equal(first, second);
    }

    /// <summary>
    /// Spacing is worked out in pixels, so the same labels on the same points come out at the same
    /// place on screen however the y axis is scaled.
    /// </summary>
    [Fact]
    public void TheLayoutIsTheSameInPixelsWhateverTheAxisRange()
    {
        var narrow = NewPlacer(1.5, 9.5).PlaceAll(Crowd());
        var wide = new RangeLabelPlacer(
            Curve.Select(e => e with { Log10Abundance = 2 * e.Log10Abundance }).ToList(),
            0, N + 1, 3, 19, WidthPx, HeightPx, MarkerRadiusPx)
            .PlaceAll(Crowd().Select(e => e with { Log10Abundance = 2 * e.Log10Abundance }).ToList());

        for (var i = 0; i < narrow.Count; i++)
        {
            Assert.Equal(narrow[i].AnchorXPx, wide[i].AnchorXPx, 6);
            Assert.Equal(narrow[i].AnchorYPx, wide[i].AnchorYPx, 6);
        }
    }

    [Fact]
    public void Geometry_SegmentsCross_OnlyOnAProperCrossing()
    {
        Assert.True(RangeLabelPlacer.SegmentsCross(0, 0, 10, 10, 0, 10, 10, 0));
        Assert.False(RangeLabelPlacer.SegmentsCross(0, 0, 10, 0, 0, 5, 10, 5));     // parallel
        Assert.False(RangeLabelPlacer.SegmentsCross(0, 0, 5, 5, 5, 5, 10, 0));      // share an end
        Assert.False(RangeLabelPlacer.SegmentsCross(0, 0, 4, 4, 6, 6, 10, 10));     // collinear, apart
    }

    [Fact]
    public void Geometry_SegmentHitsBox_OnlyThroughItsInside()
    {
        var box = new RangeLabelPlacer.Box(10, 20, 10, 20);
        Assert.True(RangeLabelPlacer.SegmentHitsBox(0, 15, 30, 15, box));
        Assert.True(RangeLabelPlacer.SegmentHitsBox(0, 0, 15, 15, box));            // ends inside
        Assert.False(RangeLabelPlacer.SegmentHitsBox(0, 0, 30, 0, box));
        Assert.False(RangeLabelPlacer.SegmentHitsBox(0, 25, 9, 25, box));
    }

    [Fact]
    public void Geometry_CoversCurve_SeesTheCurveAndNotTheSpaceBesideIt()
    {
        var placer = NewPlacer();
        // Rank 100 sits about 30 px across and 335 px up, at this scale.
        var x = 100 * WidthPx / (N + 1);
        var y = (Log10At(100) - 1.5) * HeightPx / 8;
        Assert.True(placer.CoversCurve(new RangeLabelPlacer.Box(x - 5, x + 5, y - 5, y + 5)));
        Assert.False(placer.CoversCurve(new RangeLabelPlacer.Box(x + 60, x + 120, y + 60, y + 90)));   // above-right
        Assert.False(placer.CoversCurve(new RangeLabelPlacer.Box(x - 25, x - 15, y - 120, y - 90)));  // below-left
    }

    private static bool Near(double a, double b) => Math.Abs(a - b) < 1e-6;
}
