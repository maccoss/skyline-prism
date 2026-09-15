using System;
using System.Collections.Generic;
using System.Linq;
using ScottPlot;
using ScottPlot.Plottables;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.Visualization;
using Xunit;

namespace SkylinePrism.Tests.Visualization;

/// <summary>
/// The Dynamic Range plot is Skyline's Relative Abundance shape, and its points should be the size
/// Skyline draws them - which is not a fixed pixel count. Skyline draws PointSize.normal, 12 points,
/// and ZedGraph scales every symbol by the pane's size, so on a full-width pane the points are about
/// twice what a fixed 9 px marker gave. "The points are still the same small size" was the report.
/// </summary>
public class DynamicRangePointSizeTests
{
    private static List<AbundanceEntry> Entries(int count) =>
        Enumerable.Range(1, count).Select(i =>
        {
            var log10 = 8 - 4.0 * i / count;
            return new AbundanceEntry(
                $"sp|P{i:D5}|G{i}_HUMAN", $"GEN{i}", $"P{i:D5}", $"GEN{i}", $"sp|P{i:D5}|G{i}_HUMAN",
                Math.Pow(10, log10), log10, i, 3);
        }).ToList();

    /// <summary>
    /// ZedGraph's rule, exactly: width over a 576 px base, except that a pane wider than 3:2 uses
    /// 1.5x its height and one taller than 2:3 uses 1.5x its width, and never below 0.1.
    /// </summary>
    [Fact]
    public void TheScaleIsZedGraphsPaneScaleFactor()
    {
        Assert.Equal(1350.0 / 576, PlotRenderer.SkylineSymbolScale(1400, 900), 9); // wider than 3:2
        Assert.Equal(800.0 / 576, PlotRenderer.SkylineSymbolScale(800, 600), 9);   // inside the limits
        Assert.Equal(450.0 / 576, PlotRenderer.SkylineSymbolScale(300, 600), 9);   // taller than 2:3
        Assert.Equal(0.1, PlotRenderer.SkylineSymbolScale(10, 10), 9);             // the floor
        Assert.Equal(1.0, PlotRenderer.SkylineSymbolScale(0, 0), 9);               // not laid out yet
    }

    /// <summary>
    /// The size Skyline would draw on the pane of the screenshot that prompted this: about 19 px,
    /// against the 9 px PRISM drew.
    /// </summary>
    [Fact]
    public void TwelvePointsOnAFullWidthPaneIsAboutNineteenPixels()
    {
        Assert.Equal(12 * 900.0 / 576, PlotRenderer.SkylinePointPixels(1240, 600), 9);
        Assert.InRange(PlotRenderer.SkylinePointPixels(1240, 600), 18.5, 19.0);
    }

    [Fact]
    public void PointsAreDrawnAtTheSizeAskedFor_BackgroundAndListsAlike()
    {
        var entries = Entries(100);
        var plt = new Plot();

        PlotRenderer.DrawDynamicRange(
            plt, entries.Skip(10).ToList(),
            new[] { ("EV markers", "#1f77b4", (IReadOnlyList<AbundanceEntry>)entries.Take(10).ToList()) },
            pointSize: 21);

        var series = plt.GetPlottables<Scatter>().ToList();
        Assert.Equal(2, series.Count);
        Assert.All(series, s => Assert.Equal(21.0, s.MarkerSize, 3));
    }

    /// <summary>Left unspecified, the size is Skyline's for the 1400 x 900 PNG canvas.</summary>
    [Fact]
    public void TheDefaultIsSkylinesSizeForThePngCanvas()
    {
        var plt = new Plot();

        PlotRenderer.DrawDynamicRange(
            plt, Entries(50), Array.Empty<(string, string, IReadOnlyList<AbundanceEntry>)>());

        var dots = Assert.Single(plt.GetPlottables<Scatter>());
        Assert.Equal(PlotRenderer.SkylinePointPixels(1400, 900), dots.MarkerSize, 3);
        Assert.InRange(dots.MarkerSize, 28.0, 28.3);
    }
}
