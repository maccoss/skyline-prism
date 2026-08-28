using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// Scheduled isolation windows: an m/z range crossed with the RT interval it fires in, rather than a
/// window that is on for the whole gradient.
///
/// <para><b>Why this exists with nothing producing it.</b> PRISM reads real windows by having Skyline
/// import them from a data file, and that importer finds a <i>repeating cycle</i> - so today only plain
/// DIA gets real windows, and every window it produces is always-on. The scheduled model is kept because
/// <b>dynamic DIA is DIA</b> (Pino/Searle: a whole cycle per segment, its windows marching along m/z as
/// the gradient runs), and representing it needs exactly these primitives. Reading them out of an
/// acquisition belongs in Skyline or ProteoWizard, not in this tool.</para>
///
/// <para>The subtle one is <see cref="PrecursorDensityMap.ToDisplayGrid"/> choosing its source window
/// <b>per cell rather than per row</b>: under dynamic DIA the same m/z is covered by different windows at
/// different times, and a per-row choice shows one segment and blanks the rest. These tests pin that,
/// so the behavior cannot quietly rot while unused.</para>
/// </summary>
public class ScheduledWindowTests
{
    /// <summary>
    /// Three scheduled windows. The first is wide and carries two co-eluting precursors at once; the
    /// second and third are narrow and deliberately share an m/z at times 20 minutes apart - the case
    /// that breaks any model without an RT dimension.
    /// </summary>
    private static IsolationScheme Scheduled() => new("Scheduled method", new[]
    {
        new IsolationWindow(498.15, 502.85, 0, RtStart: 10, RtStop: 14),
        new IsolationWindow(699.85, 700.55, 0, RtStart: 20, RtStop: 24),
        new IsolationWindow(699.85, 700.55, 0, RtStart: 40, RtStop: 44),
    });

    [Fact]
    public void AWindowIsAnMzRangeCrossedWithItsFiringInterval()
    {
        var scheme = Scheduled();

        Assert.True(scheme.IsScheduled);
        Assert.Equal(3, scheme.Windows.Count);

        var wide = scheme.Windows[0];
        Assert.Equal(4.7, wide.Width, 6);
        Assert.True(wide.IsScheduled);
        Assert.Equal(10.0, wide.RtStart, 6);
        Assert.Equal(14.0, wide.RtStop, 6);
        Assert.True(wide.IsOnAt(12));
        Assert.False(wide.IsOnAt(30));

        Assert.Contains("RT-scheduled", scheme.Describe());
    }

    [Fact]
    public void ScheduledWindow_OnlyClaimsPrecursorsElutingWhileItFires()
    {
        // The whole point of the RT dimension. Two targets share m/z 700.2 but fire 20 minutes apart;
        // each must be credited to its OWN window, not to both.
        var scheme = Scheduled();
        var early = new DetectedPrecursor(700.2, 21.0, 21.4);
        var late = new DetectedPrecursor(700.2, 41.0, 41.4);

        Assert.Equal(new[] { 1 }, scheme.IndicesCovering(early.Mz, early.RtStart, early.RtStop).ToArray());
        Assert.Equal(new[] { 2 }, scheme.IndicesCovering(late.Mz, late.RtStart, late.RtStop).ToArray());

        // An RT-blind match would put each precursor in both windows - that is the bug this guards.
        Assert.Equal(new[] { 1, 2 }, scheme.IndicesContaining(700.2).ToArray());
    }

    [Fact]
    public void Bin_CountsCoElutingPrecursorsTogetherAndSoloTargetsAlone()
    {
        var scheme = Scheduled();
        var precursors = new[]
        {
            new DetectedPrecursor(499.9, 11.0, 11.5),  // in the wide window
            new DetectedPrecursor(501.2, 11.2, 11.6),  // same window, co-eluting
            new DetectedPrecursor(700.2, 21.0, 21.4),  // the earlier narrow window
            new DetectedPrecursor(700.2, 41.0, 41.4),  // the later one at the same m/z
        };

        var map = PrecursorDensity.Bin(precursors, scheme, rtBinMin: 0.5);

        Assert.Equal(3, map.MzBins);
        Assert.Equal(0, map.PrecursorsOutsideRows);
        Assert.Equal(2, RowMax(map, 0));
        Assert.Equal(1, RowMax(map, 1));
        Assert.Equal(1, RowMax(map, 2));
    }

    [Fact]
    public void Bin_TimeAxisCoversTheWholeScheduleEvenWhereNothingWasDetected()
    {
        // A window that fired but found nothing is exactly what someone opens this plot to spot, so the
        // RT axis must span the schedule, not just the detections.
        var scheme = Scheduled();
        var map = PrecursorDensity.Bin(
            new[] { new DetectedPrecursor(499.9, 11.0, 11.5) }, scheme, rtBinMin: 0.5);

        Assert.True(map.RtLow <= 10.0, $"RT axis starts at {map.RtLow}, after the first window fires");
        Assert.True(map.RtHigh >= 44.0, $"RT axis ends at {map.RtHigh}, before the last window stops");
    }

    [Fact]
    public void DisplayGrid_MarksUnscheduledTimeAsNotAcquiredNotAsZero()
    {
        // A zero must always mean "acquired, nothing detected". Outside a window's firing interval
        // nothing was acquired at all, so those cells are NaN and draw as a gap.
        var scheme = Scheduled();
        var map = PrecursorDensity.Bin(
            new[] { new DetectedPrecursor(499.9, 11.0, 11.5) }, scheme, rtBinMin: 0.5);
        var grid = map.ToDisplayGrid(300);

        var row = RowForMz(map, grid.GetLength(0), 500.5);

        Assert.False(double.IsNaN(grid[row, ColumnForRt(map, 11.25)]), "the window was firing here");
        Assert.True(double.IsNaN(grid[row, ColumnForRt(map, 30.0)]), "the window was NOT firing here");
    }

    [Fact]
    public void UnscheduledWindowsStayAlwaysOn()
    {
        // Plain DIA - what every real scheme is today. No schedule: every window is on for the whole
        // gradient, and nothing about the RT dimension may change that.
        var dia = new IsolationScheme("DIA", new[]
        {
            new IsolationWindow(400, 425), new IsolationWindow(425, 450),
        });
        Assert.False(dia.IsScheduled);
        Assert.True(dia.Windows[0].IsOnAt(0));
        Assert.True(dia.Windows[0].IsOnAt(9999));
        Assert.Equal(new[] { 0 }, dia.IndicesCovering(410, 5, 6).ToArray());

        var map = PrecursorDensity.Bin(
            new[] { new DetectedPrecursor(410, 5, 5.5) }, dia, rtBinMin: 0.1);
        var grid = map.ToDisplayGrid(50);
        // No NaN anywhere: every cell of a DIA map is an acquired spectrum.
        for (var r = 0; r < grid.GetLength(0); r++)
            for (var c = 0; c < grid.GetLength(1); c++)
                Assert.False(double.IsNaN(grid[r, c]));
    }

    [Fact]
    public void ScheduledWindows_SurviveTheRunCatalogRoundTrip()
    {
        // The firing intervals are PRISM's own extension to Skyline's scheme XML, so they have to be
        // written AND read back - otherwise reopening an old output directory silently reverts a
        // scheduled method to always-on.
        var dir = Path.Combine(Path.GetTempPath(), "prism_sched_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var catalog = new IsolationSchemeCatalog();
            catalog.AddDocumentScheme("batch1", Scheduled());
            var path = Path.Combine(dir, IsolationSchemeCatalog.FileName);
            catalog.Save(path);

            var loaded = IsolationSchemeCatalog.Load(path)!;
            var scheme = Assert.Single(loaded.UsableSchemes);
            Assert.True(scheme.IsScheduled);
            Assert.Equal(3, scheme.Windows.Count);
            Assert.Equal(10.0, scheme.Windows[0].RtStart, 6);
            Assert.Equal(14.0, scheme.Windows[0].RtStop, 6);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    private static int RowMax(PrecursorDensityMap map, int row)
    {
        var max = 0;
        for (var c = 0; c < map.RtBins; c++)
            if (map.Counts[row, c] > max)
                max = map.Counts[row, c];
        return max;
    }

    private static int RowForMz(PrecursorDensityMap map, int displayRows, double mz)
    {
        var height = (map.MzHigh - map.MzLow) / displayRows;
        return (int)((map.MzHigh - mz) / height);
    }

    private static int ColumnForRt(PrecursorDensityMap map, double rt) =>
        (int)((rt - map.RtLow) / map.RtBinMin);
}
