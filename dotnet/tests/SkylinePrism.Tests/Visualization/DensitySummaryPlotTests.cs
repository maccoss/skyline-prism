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
/// The two summary views of the spectrum-density map. The arithmetic behind them is pinned by
/// <c>DensitySummaryViewsTests</c>; what these cover is that the drawing does not lose or invent
/// anything on the way to the plot.
///
/// <para>The one that matters is the NaN break. <see cref="PrecursorDensityMap.LoadOverTime"/>
/// deliberately returns NaN - not 0 - for a time when no window was firing, so that a gap in a
/// scheduled method reads as a gap. ScottPlot draws a scatter straight through a NaN, so if the
/// renderer hands it one series the plot silently says the instrument was idle across the gap, which
/// is the opposite of what the NaN was for.</para>
/// </summary>
public class DensitySummaryPlotTests
{
    /// <summary>Three abutting 10 Th windows, always on - ordinary DIA.</summary>
    private static IsolationScheme Dia() => new("DIA", new[]
    {
        new IsolationWindow(400, 410),
        new IsolationWindow(410, 420),
        new IsolationWindow(420, 430),
    });

    /// <summary>Two PRM slots firing at different times, with several idle minutes between them.</summary>
    private static IsolationScheme Scheduled() => new("PRM", new[]
    {
        new IsolationWindow(500, 502, 0, RtStart: 0, RtStop: 2),
        new IsolationWindow(600, 602, 0, RtStart: 8, RtStop: 10),
    });

    private static PrecursorDensityMap DiaMap() => PrecursorDensity.Bin(
        new List<DetectedPrecursor>
        {
            new(405, 1.0, 1.2), new(406, 1.0, 1.2), new(407, 1.0, 1.2),
            new(415, 1.0, 1.2),
            new(425, 5.0, 5.2), new(426, 5.0, 5.2),
        },
        Dia(), rtBinMin: 1.0);

    private static PrecursorDensityMap ScheduledMap() => PrecursorDensity.Bin(
        new List<DetectedPrecursor> { new(501, 0.5, 0.9), new(601, 8.5, 8.9) },
        Scheduled(), rtBinMin: 1.0);

    private static IEnumerable<Scatter> Lines(Plot plt) => plt.GetPlottables<Scatter>();

    private static IReadOnlyList<Coordinates> Points(Scatter line) => line.Data.GetScatterPoints();

    /// <summary>
    /// ScottPlot's <c>LinePattern</c> is a struct holding a <c>float[]</c> of dash intervals, so it has
    /// no usable value equality (comparing two would compare array references). Its name is the stable
    /// way to ask which pattern a line was given.
    /// </summary>
    private static bool IsSolid(Scatter line) =>
        line.LinePattern.Name == LinePattern.Solid.Name;

    private static bool IsDashed(Scatter line) =>
        line.LinePattern.Name == LinePattern.Dashed.Name;

    // ---------------------------------------------------------------- histogram

    [Fact]
    public void Histogram_DrawsOneBarPerLoad_WithTheSpectrumCounts()
    {
        var map = DiaMap();
        var plt = new Plot();

        PlotRenderer.DrawPrecursorLoadHistogram(plt, map);

        var bars = Assert.Single(plt.GetPlottables<BarPlot>()).Bars;
        var expected = map.PrecursorsPerSpectrumHistogram();
        Assert.Equal(expected.Length, bars.Count); // 0..MaxCount inclusive, empty bins included

        for (var n = 0; n < expected.Length; n++)
        {
            Assert.Equal(n, bars[n].Position);
            Assert.Equal(expected[n], bars[n].Value);
        }
    }

    /// <summary>
    /// Bin 0 dwarfs the rest for any real acquisition, so the mean is impossible to place by eye - it
    /// is drawn, and it has to be the mean over ACQUIRED spectra, matching the histogram beside it.
    /// </summary>
    [Fact]
    public void Histogram_MarksTheMeanLoadOverTheSpectraItCounted()
    {
        var map = DiaMap();
        var plt = new Plot();

        PlotRenderer.DrawPrecursorLoadHistogram(plt, map);

        var histogram = map.PrecursorsPerSpectrumHistogram();
        var expected = (double)histogram.Select((count, n) => (long)count * n).Sum() / histogram.Sum();

        var line = Assert.Single(plt.GetPlottables<VerticalLine>());
        Assert.Equal(expected, line.X, 6);
        Assert.Contains("mean", line.LegendText, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// The axis has to stop at the data. ScottPlot's default margins pad either side, which on a count
    /// of precursors puts ticks at -2 and -1 - loads that cannot exist.
    /// </summary>
    [Fact]
    public void Histogram_ShowsNoLoadBelowZeroAndNoBinBeyondTheBusiestSpectrum()
    {
        var map = DiaMap();
        var plt = new Plot();

        PlotRenderer.DrawPrecursorLoadHistogram(plt, map);

        var limits = plt.Axes.GetLimits();
        Assert.Equal(-0.5, limits.Left, 6);
        Assert.Equal(map.MaxCount + 0.5, limits.Right, 6);
        Assert.Equal(0, limits.Bottom, 6);
    }

    [Fact]
    public void Histogram_OnAnEmptyMap_SaysSoInsteadOfThrowing()
    {
        var plt = new Plot();

        PlotRenderer.DrawPrecursorLoadHistogram(
            plt, PrecursorDensity.Bin(new List<DetectedPrecursor>(), Dia(), rtBinMin: 1.0));

        Assert.Empty(plt.GetPlottables<BarPlot>());
        Assert.False(string.IsNullOrWhiteSpace(plt.Axes.Title.Label.Text));
    }

    // ---------------------------------------------------------------- load over time

    [Fact]
    public void LoadOverTime_DrawsMeanMinAndMaxAsOneBand()
    {
        var map = DiaMap();
        var plt = new Plot();

        PlotRenderer.DrawPrecursorLoadOverTime(plt, map);

        // Every RT bin was acquired (the windows are always on), so it is one unbroken segment: three
        // lines and one filled band between min and max.
        var lines = Lines(plt).ToList();
        Assert.Equal(3, lines.Count);
        Assert.Single(plt.GetPlottables<FillY>());

        var series = map.LoadOverTime();
        foreach (var line in lines)
            Assert.Equal(series.Count, Points(line).Count);

        // Mean solid, min and max dashed, and all three in one color so it reads as a band rather than
        // three unrelated series.
        Assert.Single(lines, IsSolid);
        Assert.Equal(2, lines.Count(IsDashed));
        Assert.Single(lines.Select(l => l.Color).Distinct());

        // The solid line really is the mean, not one of the bounds.
        var mean = lines.First(IsSolid);
        Assert.Equal(
            series.Select(p => p.Mean).ToArray(),
            Points(mean).Select(c => c.Y).ToArray());
    }

    /// <summary>
    /// The whole reason the renderer splits the series itself: ScottPlot connects across a NaN, so a
    /// single series would draw a line through the minutes when the method was not acquiring at all.
    /// </summary>
    [Fact]
    public void LoadOverTime_BreaksEverySeriesWhereNothingWasAcquired()
    {
        var map = ScheduledMap();
        var plt = new Plot();

        PlotRenderer.DrawPrecursorLoadOverTime(plt, map);

        var series = map.LoadOverTime();
        Assert.Contains(series, p => double.IsNaN(p.Mean)); // the fixture really does have a gap

        var lines = Lines(plt).ToList();
        Assert.Equal(6, lines.Count); // two firing intervals x (mean, min, max)

        // Nothing NaN is ever handed to the plot - that is what would be drawn through.
        foreach (var line in lines)
        {
            Assert.NotEmpty(Points(line));
            Assert.All(Points(line), c => Assert.False(double.IsNaN(c.Y), $"NaN plotted at x={c.X}"));
        }

        // Every acquired point survives the split, exactly once - the break drops the gaps, nothing else.
        var acquired = series.Count(p => !double.IsNaN(p.Mean));
        Assert.Equal(acquired, lines.Where(IsSolid).Sum(l => Points(l).Count));
    }

    /// <summary>
    /// The x axis must span the whole run even when the middle of it was idle, or the gap closes up and
    /// the two firing intervals render as one continuous acquisition.
    /// </summary>
    [Fact]
    public void LoadOverTime_KeepsTheFullRetentionTimeExtent()
    {
        var map = ScheduledMap();
        var plt = new Plot();

        PlotRenderer.DrawPrecursorLoadOverTime(plt, map);

        var limits = plt.Axes.GetLimits();
        Assert.Equal(map.RtLow, limits.Left, 6);
        Assert.Equal(map.RtHigh, limits.Right, 6);
        Assert.Equal(0, limits.Bottom, 6); // counts are non-negative; the axis says so
    }

    [Fact]
    public void LoadOverTime_OnAnEmptyMap_SaysSoInsteadOfThrowing()
    {
        var plt = new Plot();

        PlotRenderer.DrawPrecursorLoadOverTime(
            plt, PrecursorDensity.Bin(new List<DetectedPrecursor>(), Dia(), rtBinMin: 1.0));

        Assert.Empty(Lines(plt));
        Assert.False(string.IsNullOrWhiteSpace(plt.Axes.Title.Label.Text));
    }

    /// <summary>
    /// A slot that fires for less than one RT bin leaves a one-point segment, which has no line to
    /// draw. It gets a marker instead - otherwise a sparse scheduled method renders as a blank plot.
    /// </summary>
    [Fact]
    public void LoadOverTime_ShowsASegmentTooShortToDrawALineThrough()
    {
        // Slot 1 fires for 2 min, slot 2 for 0.5 min - narrower than the 1 min RT bin.
        var scheme = new IsolationScheme("PRM", new[]
        {
            new IsolationWindow(500, 502, 0, RtStart: 0, RtStop: 2),
            new IsolationWindow(600, 602, 0, RtStart: 8, RtStop: 8.5),
        });
        var map = PrecursorDensity.Bin(
            new List<DetectedPrecursor> { new(501, 0.5, 0.9), new(601, 8.1, 8.4) },
            scheme, rtBinMin: 1.0);
        var plt = new Plot();

        PlotRenderer.DrawPrecursorLoadOverTime(plt, map);

        var single = Lines(plt).Where(l => Points(l).Count == 1).ToList();
        Assert.NotEmpty(single);
        Assert.All(single, l => Assert.True(l.MarkerSize > 0,
            "a one-bin segment has no line, so it must be marked or it renders as nothing"));
    }
}
