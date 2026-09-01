using System;
using System.Collections.Generic;
using System.Linq;
using ScottPlot;
using SkylinePrism.Core.Visualization;
using Xunit;

namespace SkylinePrism.Tests.Visualization;

/// <summary>
/// Legend placement on the PCA scatter.
///
/// <para>ScottPlot draws the legend inside the axes and leaves it where it is put, so on a real cohort
/// it lands on top of samples. That matters more than ordinary clutter here: the points it hides are
/// individual replicates, and this plot exists so someone can find the odd one.</para>
///
/// <para>The heuristic is approximate by construction - the legend's real size depends on how many
/// series it lists and how long their names are - so these tests assert the property that matters
/// (it does not choose a crowded corner when an empty one exists) rather than pinning exact corners
/// for arrangements where several answers are equally good.</para>
/// </summary>
public class LegendPlacementTests
{
    /// <summary>Points in a box around (cx, cy), spread by r, laid out deterministically.</summary>
    private static (List<double> X, List<double> Y) Cluster(double cx, double cy, double r, int n)
    {
        var xs = new List<double>();
        var ys = new List<double>();
        for (var i = 0; i < n; i++)
        {
            var a = 2 * Math.PI * i / n;
            xs.Add(cx + r * Math.Cos(a));
            ys.Add(cy + r * Math.Sin(a));
        }
        return (xs, ys);
    }

    /// <summary>
    /// The point of the whole thing: when samples occupy the corner the legend would default to, it has
    /// to move.
    ///
    /// <para>Written this way deliberately. A first version modelled the reported plot - bulk of the
    /// samples left, controls low and right - and asserted only "not lower right". That passed, but it
    /// would have passed just as well if the method always returned upper right, because upper right
    /// happened to be empty in that arrangement. Putting the data in the DEFAULT corner is what makes
    /// the assertion mean something.</para>
    /// </summary>
    [Fact]
    public void MovesTheLegendOutOfTheCornerTheDataOccupies()
    {
        var (xs, ys) = Cluster(90, 90, 12, 40);       // a dense cluster in the upper right
        var (lx, ly) = Cluster(-60, -60, 40, 60);     // and the rest away from it
        xs.AddRange(lx); ys.AddRange(ly);

        var corner = PlotRenderer.ChooseLegendCorner(xs, ys);

        Assert.NotEqual(Alignment.UpperRight, corner);
        // And it must not simply swap to another occupied corner.
        Assert.NotEqual(Alignment.LowerLeft, corner);
    }

    /// <summary>
    /// The arrangement actually reported: samples spread across the left, controls clustered right, and
    /// the legend drawn over them at lower right. Asserts the chosen corner really is the emptiest,
    /// counted independently here rather than trusting the method's own arithmetic.
    /// </summary>
    [Fact]
    public void OnTheReportedArrangementItChoosesAnEmptyCorner()
    {
        var (xs, ys) = Cluster(-60, 0, 60, 120);
        var (cx, cy) = Cluster(150, -10, 12, 8);
        xs.AddRange(cx); ys.AddRange(cy);

        var corner = PlotRenderer.ChooseLegendCorner(xs, ys);

        double minX = xs.Min(), maxX = xs.Max(), minY = ys.Min(), maxY = ys.Max();
        double w = (maxX - minX) * 0.34, h = (maxY - minY) * 0.34;
        var right = corner is Alignment.UpperRight or Alignment.LowerRight;
        var upper = corner is Alignment.UpperRight or Alignment.UpperLeft;
        var inCorner = xs.Zip(ys).Count(p =>
            (right ? p.First >= maxX - w : p.First <= minX + w) &&
            (upper ? p.Second >= maxY - h : p.Second <= minY + h));

        Assert.Equal(0, inCorner);
    }

    /// <summary>
    /// With three corners occupied and one free, the free one is the only right answer - so this pins
    /// the corner exactly, unlike the looser assertions above.
    /// </summary>
    [Theory]
    [InlineData(Alignment.UpperRight)]
    [InlineData(Alignment.UpperLeft)]
    [InlineData(Alignment.LowerRight)]
    [InlineData(Alignment.LowerLeft)]
    public void PicksTheOneEmptyCorner(Alignment expected)
    {
        var corners = new Dictionary<Alignment, (double X, double Y)>
        {
            [Alignment.UpperRight] = (100, 100),
            [Alignment.UpperLeft] = (-100, 100),
            [Alignment.LowerRight] = (100, -100),
            [Alignment.LowerLeft] = (-100, -100),
        };
        var xs = new List<double>();
        var ys = new List<double>();
        foreach (var (corner, p) in corners)
        {
            if (corner == expected)
                continue;
            var (cxs, cys) = Cluster(p.X, p.Y, 5, 20);
            xs.AddRange(cxs); ys.AddRange(cys);
        }
        // The extremes have to exist in the data or the empty corner is not inside the axis limits at
        // all - the corner boxes are measured from the data's own range.
        xs.Add(corners[expected].X); ys.Add(corners[expected].Y);

        Assert.Equal(expected, PlotRenderer.ChooseLegendCorner(xs, ys));
    }

    /// <summary>
    /// Degenerate inputs must not throw or pick arbitrarily; they fall back to where every other legend
    /// in the report sits. A single point and a perfectly horizontal line both give corner boxes of zero
    /// area, so no corner is genuinely emptier than another.
    /// </summary>
    [Fact]
    public void DegenerateInputsFallBackToUpperRight()
    {
        Assert.Equal(Alignment.UpperRight,
            PlotRenderer.ChooseLegendCorner(Array.Empty<double>(), Array.Empty<double>()));
        Assert.Equal(Alignment.UpperRight,
            PlotRenderer.ChooseLegendCorner(new[] { 1.0 }, new[] { 1.0 }));
        Assert.Equal(Alignment.UpperRight,
            PlotRenderer.ChooseLegendCorner(new[] { 0.0, 1.0, 2.0 }, new[] { 5.0, 5.0, 5.0 }));
        // Mismatched lengths are a caller bug, not a crash.
        Assert.Equal(Alignment.UpperRight,
            PlotRenderer.ChooseLegendCorner(new[] { 0.0, 1.0 }, new[] { 5.0 }));
    }

    /// <summary>
    /// NaN and infinity are skipped rather than dragging the range to something meaningless. A PCA of a
    /// cohort with an all-NaN sample would otherwise put every real point in one corner box.
    /// </summary>
    [Fact]
    public void NonFinitePointsAreIgnored()
    {
        var (xs, ys) = Cluster(-100, -100, 5, 20);    // everything in the lower left
        xs.Add(100); ys.Add(100);                     // and one point defining the far corner
        var withJunk = xs.ToList();
        var withJunkY = ys.ToList();
        withJunk.Add(double.NaN); withJunkY.Add(0);
        withJunk.Add(double.PositiveInfinity); withJunkY.Add(double.NaN);

        Assert.Equal(
            PlotRenderer.ChooseLegendCorner(xs, ys),
            PlotRenderer.ChooseLegendCorner(withJunk, withJunkY));
        Assert.NotEqual(Alignment.LowerLeft, PlotRenderer.ChooseLegendCorner(withJunk, withJunkY));
    }
}
