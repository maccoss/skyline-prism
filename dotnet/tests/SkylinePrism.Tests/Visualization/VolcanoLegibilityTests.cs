using System;
using System.Collections.Generic;
using System.Linq;
using ScottPlot;
using SkylinePrism.Core.DifferentialAnalysis;
using SkylinePrism.Core.DifferentialAnalysis.Enrichment;
using SkylinePrism.Core.Visualization;
using Xunit;

namespace SkylinePrism.Tests.Visualization;

/// <summary>
/// The quant report's volcano is pasted onto slides, so its marks and names must be readable at the
/// size of the axis text around them. They were not: 5 and 6 px points and 11 pt labels beside
/// 24-30 pt axis text, and a contrast title that ran off both edges of the image.
/// </summary>
public class VolcanoLegibilityTests
{
    /// <summary>40 non-hits and 12 hits, the hits strongest first: hit k has p = 1e-(20-k).</summary>
    private static DifferentialResult Result()
    {
        var rows = new List<DifferentialRow>();
        for (var k = 0; k < 12; k++)
        {
            var p = Math.Pow(10, -(20 - k));
            rows.Add(new DifferentialRow($"PG{k:D4}", 2.0 + k * 0.1, 4, 15, 8, p, p, 14, 16));
        }
        for (var k = 0; k < 40; k++)
            rows.Add(new DifferentialRow($"PG{100 + k:D4}", (k - 20) * 0.05, 1, 15, 0.5, 0.5, 0.9, 15, 15));
        return new DifferentialResult(rows, 10, 10, rows.Count, rows.Count, 18, 4, "global",
            Array.Empty<string>(), Array.Empty<string>(), Array.Empty<string>());
    }

    private static Plot Draw(string? title = null)
    {
        var plt = new Plot();
        PlotRenderer.DrawDifferentialVolcano(plt, Result(), SignificanceRule.Default, corrected: true,
            "log2FC", id => "G" + id, title);
        return plt;
    }

    [Fact]
    public void MarksAreSizedToTheAxisText()
    {
        var plt = Draw();
        var tick = plt.Axes.Bottom.TickLabelStyle.FontSize;
        var marks = plt.GetPlottables<ScottPlot.Plottables.Markers>().ToList();

        Assert.Equal(2, marks.Count);
        var background = marks.Single(m => m.Data.GetScatterPoints().Count == 40);
        var hits = marks.Single(m => m.Data.GetScatterPoints().Count == 12);
        // A hit is the point of the plot and is drawn larger than the cloud it stands out from.
        Assert.True(hits.MarkerSize > background.MarkerSize);
        // Both against the tick text they sit beside: at least about a third of its size for the
        // cloud, over half for a hit. The old 5 and 6 px were a fifth and a quarter of 24 pt.
        Assert.True(background.MarkerSize >= tick / 3, $"background {background.MarkerSize} px vs {tick} pt ticks");
        Assert.True(hits.MarkerSize >= tick / 2, $"hits {hits.MarkerSize} px vs {tick} pt ticks");
    }

    [Fact]
    public void TheStrongestHitsAreNamed_Legibly_AndNotAll()
    {
        var plt = Draw();
        var tick = plt.Axes.Bottom.TickLabelStyle.FontSize;
        var labels = plt.GetPlottables<ScottPlot.Plottables.Text>().ToList();

        Assert.InRange(labels.Count, 1, PlotRenderer.VolcanoLabeledHits);
        // Strongest first: hit 0 has the smallest p and is always named; the rest in order of strength.
        Assert.Equal("GPG0000", labels[0].LabelText);
        var named = labels.Select(t => int.Parse(t.LabelText[3..])).ToList();
        Assert.Equal(named.OrderBy(k => k), named);
        Assert.All(labels, t =>
        {
            Assert.True(t.LabelFontSize >= 0.75 * tick, $"label {t.LabelFontSize} pt vs {tick} pt ticks");
            Assert.Equal(PlotRenderer.PlotFontName, t.LabelFontName);
        });
    }

    /// <summary>
    /// Hits are colored by direction with the enrichment bars' colors - red higher in B, blue lower -
    /// so a red point and a red bar mean the same thing. They used to be red on both sides.
    /// </summary>
    [Fact]
    public void HitsAreColoredByDirection_AsTheEnrichmentBarsAre()
    {
        var rows = new List<DifferentialRow>();
        for (var k = 0; k < 3; k++)
            rows.Add(new DifferentialRow($"UP{k}", 2.0 + k, 4, 15, 8, 1e-10, 1e-10, 14, 16));
        for (var k = 0; k < 5; k++)
            rows.Add(new DifferentialRow($"DN{k}", -2.0 - k, 4, 15, 8, 1e-10, 1e-10, 16, 14));
        for (var k = 0; k < 7; k++)
            rows.Add(new DifferentialRow($"NS{k}", (k - 3) * 0.1, 1, 15, 0.5, 0.5, 0.9, 15, 15));
        var res = new DifferentialResult(rows, 10, 10, rows.Count, rows.Count, 18, 4, "global",
            Array.Empty<string>(), Array.Empty<string>(), Array.Empty<string>());

        var plt = new Plot();
        PlotRenderer.DrawDifferentialVolcano(plt, res, SignificanceRule.Default, corrected: true,
            "log2FC", id => id);
        var marks = plt.GetPlottables<ScottPlot.Plottables.Markers>().ToList();

        Assert.Equal(3, marks.Count);
        var up = marks.Single(m => m.Data.GetScatterPoints().Count == 3);
        var down = marks.Single(m => m.Data.GetScatterPoints().Count == 5);
        Assert.All(up.Data.GetScatterPoints(), p => Assert.True(p.X > 0));
        Assert.All(down.Data.GetScatterPoints(), p => Assert.True(p.X < 0));
        Assert.Equal(Color.FromHex(Enrichment.DirectionColor(EnrichmentDirection.Up)), up.Color);
        Assert.Equal(Color.FromHex(Enrichment.DirectionColor(EnrichmentDirection.Down)), down.Color);
        Assert.NotEqual(up.Color, down.Color);

        // The guides are neutral: in ScottPlot's cycle colors one of them was red on the blue side.
        var guides = plt.GetPlottables<ScottPlot.Plottables.AxisLine>().ToList();
        Assert.Equal(3, guides.Count);
        Assert.All(guides, g => Assert.Equal(PlotRenderer.VolcanoGuideColor, g.LineColor));
    }

    /// <summary>
    /// The shared hit drawing every volcano goes through, the app pane's included: red where the
    /// effect is positive, blue where negative - the same split as the enrichment up and down lists -
    /// and a zero effect, which is in neither list, in its own dark color rather than the pooled green.
    /// </summary>
    [Fact]
    public void AddVolcanoHits_SplitsBySign_AndAZeroEffectIsInNeitherSide()
    {
        var plt = new Plot();
        var added = PlotRenderer.AddVolcanoHits(plt,
            new[] { 1.5, -2.0, 0.0, 3.0, -0.5 }, new[] { 5.0, 6.0, 7.0, 8.0, 9.0 }, 14);

        Assert.Equal(3, added.Count);
        var up = added[0].Data.GetScatterPoints();
        var down = added[1].Data.GetScatterPoints();
        var zero = added[2].Data.GetScatterPoints();
        Assert.Equal(new[] { 1.5, 3.0 }, up.Select(p => p.X));
        Assert.Equal(new[] { 5.0, 8.0 }, up.Select(p => p.Y));
        Assert.Equal(new[] { -2.0, -0.5 }, down.Select(p => p.X));
        Assert.Equal(new[] { 0.0 }, zero.Select(p => p.X));
        Assert.Equal(Color.FromHex(Enrichment.DirectionColor(EnrichmentDirection.Up)), added[0].Color);
        Assert.Equal(Color.FromHex(Enrichment.DirectionColor(EnrichmentDirection.Down)), added[1].Color);
        Assert.Equal(PlotRenderer.VolcanoNoDirectionColor, added[2].Color);
        Assert.NotEqual(Color.FromHex(Enrichment.DirectionColor(EnrichmentDirection.Both)), added[2].Color);
        Assert.All(added, m => Assert.Equal(14f, m.MarkerSize));
    }

    /// <summary>An empty side adds no series, so a one-sided contrast has one hit series, not three.</summary>
    [Fact]
    public void AddVolcanoHits_LeavesOutAnEmptySide()
    {
        var plt = new Plot();
        var added = PlotRenderer.AddVolcanoHits(plt, new[] { 1.0, 2.0 }, new[] { 3.0, 4.0 }, 10);

        Assert.Single(added);
        Assert.Single(plt.GetPlottables<ScottPlot.Plottables.Markers>());
        Assert.Empty(PlotRenderer.AddVolcanoHits(new Plot(), Array.Empty<double>(), Array.Empty<double>(), 10));
    }

    /// <summary>
    /// The rule's guides: the p cut and the fold-change cut on both sides; with no fold-change cut,
    /// the p line alone - a line at zero would read as a threshold.
    /// </summary>
    [Theory]
    [InlineData(1.0, 3)]
    [InlineData(0.0, 1)]
    public void AddVolcanoGuides_FromTheRule(double lfcCut, int lines)
    {
        var plt = new Plot();
        PlotRenderer.AddVolcanoGuides(plt, SignificanceRule.Default with { Log2FcThreshold = lfcCut });

        var guides = plt.GetPlottables<ScottPlot.Plottables.AxisLine>().ToList();
        Assert.Equal(lines, guides.Count);
        var vertical = guides.OfType<ScottPlot.Plottables.VerticalLine>().Select(v => v.X).OrderBy(x => x);
        Assert.Equal(lfcCut > 0 ? new[] { -lfcCut, lfcCut } : Array.Empty<double>(), vertical);
        Assert.All(guides, g => Assert.Equal(PlotRenderer.VolcanoGuideColor, g.LineColor));
    }

    /// <summary>
    /// The detection volcano colors its hits by direction too: detected more often in B is red, less
    /// often blue. It drew every hit green.
    /// </summary>
    [Fact]
    public void DetectionVolcano_ColorsHitsByDirection()
    {
        var rows = new List<SkylinePrism.Core.DifferentialAnalysis.Detection.DetectionRow>
        {
            DetectionRow("UP", 0.2, 0.9, 1e-6),
            DetectionRow("DN", 0.9, 0.1, 1e-6),
            DetectionRow("NS", 0.5, 0.5, 0.8),
        };
        var plt = new Plot();
        PlotRenderer.DrawDetectionVolcano(plt, rows, SignificanceRule.Default, corrected: true);

        var marks = plt.GetPlottables<ScottPlot.Plottables.Markers>().ToList();
        Assert.Equal(3, marks.Count);
        Assert.Contains(marks, m => m.Color == Color.FromHex(Enrichment.DirectionColor(EnrichmentDirection.Up))
            && m.Data.GetScatterPoints().Single().X > 0);
        Assert.Contains(marks, m => m.Color == Color.FromHex(Enrichment.DirectionColor(EnrichmentDirection.Down))
            && m.Data.GetScatterPoints().Single().X < 0);
        Assert.DoesNotContain(marks, m => m.Color == Color.FromHex("#2ca02c"));
    }

    private static SkylinePrism.Core.DifferentialAnalysis.Detection.DetectionRow DetectionRow(
        string id, double rateA, double rateB, double p) =>
        new(id, (int)(rateA * 10), 10, (int)(rateB * 10), 10, rateA, rateB, p, p);

    private static readonly PlotRenderer.LabelBox Canvas = new(0, 0, 1000, 600);

    /// <summary>
    /// Two hits side by side at the top: the second's label above would run into the first's, which
    /// is what a crowded volcano showed as two names run together. The first leans away, leaving the
    /// second its place above, and both are named.
    /// </summary>
    [Fact]
    public void ALabelThatWouldCollideMovesToAFreeSide()
    {
        var points = new[] { (500f, 100f), (560f, 100f) };
        var sizes = new[] { (100f, 25f), (100f, 25f) };

        var placed = PlotRenderer.PlaceVolcanoLabels(points, sizes, Canvas, 7, 10);

        Assert.Equal(new[] { (0, PlotRenderer.LabelSide.AboveLeft), (1, PlotRenderer.LabelSide.Above) }, placed);
    }

    /// <summary>
    /// On a crowded top row: no two placed labels overlap, none covers another hit, and each is
    /// clearly nearer its own hit than any other, so none can be read as a neighbour's.
    /// </summary>
    [Fact]
    public void PlacedLabelsNeverOverlapEachOtherOrAHit()
    {
        var points = Enumerable.Range(0, 12).Select(k => (300f + 40 * k, 80f + 6 * (k % 3))).ToList();
        var sizes = points.Select(_ => (110f, 25f)).ToList();

        var placed = PlotRenderer.PlaceVolcanoLabels(points, sizes, Canvas, 7, 12);

        var boxes = placed.Select(p => BoxOf(points[p.Index], sizes[p.Index], p.Side, 7)).ToList();
        for (var a = 0; a < boxes.Count; a++)
        {
            for (var b = a + 1; b < boxes.Count; b++)
                Assert.False(boxes[a].Overlaps(boxes[b]), $"labels {placed[a]} and {placed[b]} overlap");
            for (var k = 0; k < points.Count; k++)
            {
                if (k == placed[a].Index)
                    continue;
                var mark = new PlotRenderer.LabelBox(points[k].Item1 - 7, points[k].Item2 - 7,
                    points[k].Item1 + 7, points[k].Item2 + 7);
                Assert.False(boxes[a].Overlaps(mark), $"label {placed[a]} covers hit {k}");
                var own = boxes[a].CenterDistanceTo(points[placed[a].Index].Item1, points[placed[a].Index].Item2);
                Assert.True(boxes[a].CenterDistanceTo(points[k].Item1, points[k].Item2)
                        >= PlotRenderer.LabelOwnershipMargin * own,
                    $"label {placed[a]} is nearly as close to hit {k} as to its own");
            }
        }
        Assert.NotEmpty(placed);
    }

    /// <summary>The box a placed label occupies - restated here from the layout rule, not read back.</summary>
    private static PlotRenderer.LabelBox BoxOf((float X, float Y) p, (float W, float H) s,
        PlotRenderer.LabelSide side, float r)
    {
        var gap = r + PlotRenderer.LabelGap;
        var lean = r / 2;
        var (x, y) = p;
        var (w, h) = s;
        return side switch
        {
            PlotRenderer.LabelSide.Above => new(x - w / 2, y - gap - h, x + w / 2, y - gap),
            PlotRenderer.LabelSide.AboveRight => new(x + lean, y - gap - h, x + lean + w, y - gap),
            PlotRenderer.LabelSide.AboveLeft => new(x - lean - w, y - gap - h, x - lean, y - gap),
            PlotRenderer.LabelSide.Below => new(x - w / 2, y + gap, x + w / 2, y + gap + h),
            PlotRenderer.LabelSide.Right => new(x + gap, y - h / 2, x + gap + w, y + h / 2),
            _ => new(x - gap - w, y - h / 2, x - gap, y + h / 2),
        };
    }

    /// <summary>
    /// A hit at the right edge is labeled toward its left rather than cut off by the edge - the
    /// truncated "A0A024R16" a crowded volcano printed - and with the top row free, above-left is the first such side.
    /// </summary>
    [Fact]
    public void ALabelAtTheEdgeStaysOnTheCanvas()
    {
        var placed = PlotRenderer.PlaceVolcanoLabels(
            new[] { (990f, 300f) }, new[] { (120f, 25f) }, Canvas, 7, 10);

        Assert.Equal(new[] { (0, PlotRenderer.LabelSide.AboveLeft) }, placed);
        Assert.True(BoxOf((990f, 300f), (120f, 25f), placed[0].Side, 7).Inside(Canvas));
    }

    /// <summary>
    /// Space is reserved only for points that can still be labeled. With one label to place, the
    /// neighbour will never get one, so the strongest hit keeps its place straight above rather than
    /// leaning away to keep room for it.
    /// </summary>
    [Fact]
    public void NoRoomIsKeptForAPointThatCannotBeLabeled()
    {
        var points = new[] { (500f, 100f), (560f, 100f) };
        var sizes = new[] { (100f, 25f), (100f, 25f) };

        var placed = PlotRenderer.PlaceVolcanoLabels(points, sizes, Canvas, 7, maxLabels: 1);

        Assert.Equal(new[] { (0, PlotRenderer.LabelSide.Above) }, placed);
    }

    /// <summary>
    /// Thousands of hits, all crowded where nothing fits, as a peptide-level contrast can have: only
    /// the strongest few are tried, so the layout stays quick instead of scanning every hit against
    /// every other. Everything here collides, so nothing is placed - and it must say so fast.
    /// </summary>
    [Fact]
    public void ThousandsOfCrowdedHitsAreLaidOutQuickly()
    {
        var points = Enumerable.Range(0, 20_000).Select(k => (500f + k % 7, 300f + k % 5)).ToList();
        var sizes = points.Select(_ => (120f, 25f)).ToList();

        var clock = System.Diagnostics.Stopwatch.StartNew();
        var placed = PlotRenderer.PlaceVolcanoLabels(points, sizes, Canvas, 7, 10);
        clock.Stop();

        Assert.Empty(placed);
        // Generous: the bounded layout is ~50 x 12 x 20,000 cheap checks; the unbounded one was
        // ~20,000 x 12 x 60,000 and took minutes.
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(10), $"took {clock.Elapsed}");
    }

    /// <summary>
    /// A hit boxed in on every side is left unnamed - a label on top of other hits would hide them -
    /// and the next hit is named instead, up to the cap.
    /// </summary>
    [Fact]
    public void AHitWithNoRoomIsSkipped_AndTheCapHolds()
    {
        // Point 0 is surrounded by marks 1-4 just beyond its marker on each side.
        var points = new List<(float, float)> { (500, 300), (500, 270), (500, 330), (560, 300), (440, 300) };
        points.AddRange(Enumerable.Range(0, 20).Select(k => ((float)(50 + 45 * k), 550f)));
        var sizes = points.Select(_ => (100f, 25f)).ToList();

        var placed = PlotRenderer.PlaceVolcanoLabels(points, sizes, Canvas, 7, 5);

        Assert.DoesNotContain(placed, p => p.Index == 0);
        Assert.Equal(5, placed.Count);
    }

    [Fact]
    public void ALongContrastTitleShrinksInsteadOfRunningOffTheImage()
    {
        var shortTitle = Draw("Disease vs Control").Axes.Title.Label.FontSize;
        var longTitle = Draw("Treatment_Responder_Week12 vs Placebo_NonResponder_Week12 by Arm_Response")
            .Axes.Title.Label.FontSize;

        Assert.True(longTitle < shortTitle, $"long {longTitle} pt, short {shortTitle} pt");
    }

    /// <summary>
    /// The size follows the length: a title of several ticked levels per arm, long enough to run off
    /// at any fixed size step, is set small enough that its estimated width fits the image - and a
    /// short one keeps the size every other QC plot's title has.
    /// </summary>
    [Theory]
    [InlineData("Disease vs Control")]
    [InlineData("Treatment_Responder_Week12 vs Placebo_NonResponder_Week12 by Arm_Response")]
    [InlineData("Disease_A + Disease_B + Disease_C vs Control_X + Control_Y by Arm_Visit_Cohort")]
    public void ATitleFitsTheImageWidth(string title)
    {
        var size = Draw(title).Axes.Title.Label.FontSize;

        Assert.InRange(size, PlotRenderer.MinTitleFontSize, PlotRenderer.MaxTitleFontSize);
        var estimatedWidth = PlotRenderer.TitleEmPerChar * size * title.Length;
        Assert.True(estimatedWidth <= 1100 || size == PlotRenderer.MinTitleFontSize,
            $"'{title}' at {size} pt is about {estimatedWidth:0} px on a 1100 px image");
        if (title.Length < 30)
            Assert.Equal(PlotRenderer.MaxTitleFontSize, size, 3);
    }
}
