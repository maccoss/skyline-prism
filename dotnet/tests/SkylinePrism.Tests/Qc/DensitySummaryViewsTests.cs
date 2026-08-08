using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// The two summaries of the spectrum-density map: how the precursor load is distributed across
/// spectra, and how it moves over the gradient.
///
/// <para>The subtle part of both is that a cell can mean three different things - a spectrum that saw
/// nothing, a spectrum that saw several precursors, and <b>no spectrum at all</b>. Only the first two
/// are spectra. Counting the third would put a large spike on "0 precursors" that is purely an
/// artifact of a scheduled method's timing, and would drag the mean down at every time a window was
/// not firing.</para>
/// </summary>
public class DensitySummaryViewsTests
{
    /// <summary>Three abutting 10 Th windows, always on - ordinary DIA.</summary>
    private static IsolationScheme Dia() => new("DIA", new[]
    {
        new IsolationWindow(400, 410),
        new IsolationWindow(410, 420),
        new IsolationWindow(420, 430),
    });

    [Fact]
    public void Histogram_CountsSpectraByHowManyPrecursorsTheyHadToResolve()
    {
        // Window 400-410 gets 3 precursors in the same RT bin; 410-420 gets 1; 420-430 gets none.
        var precursors = new List<DetectedPrecursor>
        {
            new(405, 1.0, 1.2), new(406, 1.0, 1.2), new(407, 1.0, 1.2),
            new(415, 1.0, 1.2),
        };
        var map = PrecursorDensity.Bin(precursors, Dia(), rtBinMin: 1.0);

        var histogram = map.PrecursorsPerSpectrumHistogram();

        Assert.Equal(3, histogram.Length - 1); // indexed 0..max
        Assert.Equal(1, histogram[3]);         // one spectrum resolved 3 precursors
        Assert.Equal(1, histogram[1]);         // one resolved 1
        Assert.Equal(0, histogram[2]);

        // Every acquired cell is counted exactly once, and there are rows x bins of them.
        Assert.Equal(map.MzBins * map.RtBins, histogram.Sum());
    }

    /// <summary>
    /// No precursors at all means no retention-time range, so the map has no cells - there is nothing
    /// to summarize, and both views must say so rather than throwing or inventing a zero bin.
    /// </summary>
    [Fact]
    public void BothViewsAreEmptyForAnEmptyMap()
    {
        var map = PrecursorDensity.Bin(new List<DetectedPrecursor>(), Dia(), rtBinMin: 1.0);

        Assert.True(map.IsEmpty);
        Assert.Empty(map.PrecursorsPerSpectrumHistogram());
        Assert.Empty(map.LoadOverTime());
    }

    /// <summary>
    /// A spectrum that found nothing is still a spectrum, and belongs in bin 0 - that is what makes the
    /// distribution readable as "how many spectra were quiet".
    /// </summary>
    [Fact]
    public void Histogram_CountsSpectraThatDetectedNothing()
    {
        // One precursor in the first window; the other two windows saw nothing at that time.
        var map = PrecursorDensity.Bin(
            new List<DetectedPrecursor> { new(405, 1.0, 1.2) }, Dia(), rtBinMin: 1.0);

        var histogram = map.PrecursorsPerSpectrumHistogram();

        Assert.Equal(1, histogram[1]);
        Assert.Equal(map.MzBins * map.RtBins - 1, histogram[0]);
    }

    [Fact]
    public void LoadOverTime_ReportsMeanMinAndMaxAcrossTheWindowsAtEachTime()
    {
        // At t~1: counts are 3, 1, 0 across the three windows. At t~5: 0, 0, 2.
        var precursors = new List<DetectedPrecursor>
        {
            new(405, 1.0, 1.2), new(406, 1.0, 1.2), new(407, 1.0, 1.2),
            new(415, 1.0, 1.2),
            new(425, 5.0, 5.2), new(426, 5.0, 5.2),
        };
        var map = PrecursorDensity.Bin(precursors, Dia(), rtBinMin: 1.0);

        var series = map.LoadOverTime();

        Assert.Equal(map.RtBins, series.Count);

        var busy = series.First(p => p.Max == 3);
        Assert.Equal(4.0 / 3.0, busy.Mean, 6); // (3 + 1 + 0) / 3
        Assert.Equal(0, busy.Min);
        Assert.Equal(3, busy.Max);

        var later = series.First(p => p.Max == 2);
        Assert.Equal(2.0 / 3.0, later.Mean, 6);
        Assert.Equal(0, later.Min);

        // Times are bin centers, ascending.
        Assert.True(series.Zip(series.Skip(1)).All(p => p.Second.TimeMin > p.First.TimeMin));
    }

    // ---------------------------------------------------------------- scheduled methods

    /// <summary>Two PRM slots that fire at different times, so most cells are "not acquired".</summary>
    private static IsolationScheme Scheduled() => new("PRM", new[]
    {
        new IsolationWindow(500, 502, 0, RtStart: 0, RtStop: 2),
        new IsolationWindow(600, 602, 0, RtStart: 8, RtStop: 10),
    });

    /// <summary>
    /// The whole reason these two methods do not just read <c>Counts</c>: a window that was not firing
    /// is not a spectrum that found nothing. Counting it would pile an enormous, meaningless spike onto
    /// "0 precursors" - here, most of the grid.
    /// </summary>
    [Fact]
    public void Histogram_IgnoresTimesWhenAWindowWasNotFiring()
    {
        var precursors = new List<DetectedPrecursor>
        {
            new(501, 0.5, 0.9),  // in slot 1's interval
            new(601, 8.5, 8.9),  // in slot 2's interval
        };
        var map = PrecursorDensity.Bin(precursors, Scheduled(), rtBinMin: 1.0);

        var histogram = map.PrecursorsPerSpectrumHistogram();

        // Only the cells inside a slot's RT interval count - far fewer than the full grid.
        Assert.True(histogram.Sum() < map.MzBins * map.RtBins,
            "cells outside a scheduled window's interval were counted as spectra");
        Assert.Equal(2, histogram.Skip(1).Sum()); // the two windows that did detect something
    }

    [Fact]
    public void LoadOverTime_ReportsNaNWhenNothingWasAcquiredAtThatTime()
    {
        var map = PrecursorDensity.Bin(
            new List<DetectedPrecursor> { new(501, 0.5, 0.9) }, Scheduled(), rtBinMin: 1.0);

        var series = map.LoadOverTime();

        // Between the two slots (t ~ 4-7) neither window is firing: a gap, not an idle instrument.
        var between = series.Where(p => p.TimeMin > 3 && p.TimeMin < 7).ToList();
        Assert.NotEmpty(between);
        Assert.All(between, p => Assert.True(double.IsNaN(p.Mean), $"t={p.TimeMin} reported {p.Mean}"));

        // While slot 1 was firing there IS a value.
        var firing = series.First(p => p.TimeMin is > 0 and < 2);
        Assert.False(double.IsNaN(firing.Mean));
    }

    [Fact]
    public void BothViewsAgreeWithTheMapTheySummarize()
    {
        var precursors = Enumerable.Range(0, 40)
            .Select(i => new DetectedPrecursor(401 + i % 3 * 10 + i % 7 * 0.1, i % 5, i % 5 + 0.4))
            .ToList();
        var map = PrecursorDensity.Bin(precursors, Dia(), rtBinMin: 1.0);

        var histogram = map.PrecursorsPerSpectrumHistogram();
        var series = map.LoadOverTime();

        // Total precursor-observations must match however they are summed.
        var fromHistogram = histogram.Select((count, n) => (long)count * n).Sum();
        var fromSeries = series.Where(p => !double.IsNaN(p.Mean)).Sum(p => p.Mean * map.MzBins);
        Assert.Equal(fromHistogram, (long)Math.Round(fromSeries));

        // And with the map's own maximum.
        Assert.Equal(map.MaxCount, histogram.Length - 1);
        Assert.Equal(map.MaxCount, series.Where(p => !double.IsNaN(p.Max)).Max(p => p.Max));
    }
}
