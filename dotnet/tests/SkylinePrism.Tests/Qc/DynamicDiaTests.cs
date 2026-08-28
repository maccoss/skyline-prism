using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// Dynamic DIA: a repeating cycle of DIA windows whose m/z positions SHIFT along the gradient, so the
/// method tracks the mass range where peptides are actually eluting (Pino/Searle et al.,
/// PMC10517878 - "8 m/z isolation windows ... covering a variable m/z range of approximately 300 m/z",
/// with "the exact m/z positioning of these spectra adjusted over time").
/// </summary>
/// <remarks>
/// Structurally this is the scheduled-window model with a whole CYCLE per RT segment rather than one slot:
/// several windows share each firing interval, and the same m/z is covered by different windows in
/// different segments. That last property is what distinguishes it from both static DIA (one window per
/// m/z, always on) and PRM (one target per slot), and it is the case that breaks any renderer choosing a
/// window per m/z row instead of per cell.
/// </remarks>
public class DynamicDiaTests
{
    private const double WindowWidth = 8.0;
    private const int WindowsPerCycle = 8;

    /// <summary>
    /// Build a dynamic-DIA scheme: <paramref name="segments"/> RT segments, each a cycle of 8 x 8 m/z
    /// windows, with the cycle's low edge marching upward segment by segment.
    /// </summary>
    private static IsolationScheme DynamicScheme(int segments = 4, double segmentMinutes = 15)
    {
        var windows = new List<IsolationWindow>();
        for (var s = 0; s < segments; s++)
        {
            var cycleLow = 400 + s * 60.0;              // the cycle marches up m/z over the gradient
            var rtStart = s * segmentMinutes;
            var rtStop = (s + 1) * segmentMinutes;
            for (var w = 0; w < WindowsPerCycle; w++)
            {
                var start = cycleLow + w * WindowWidth;
                windows.Add(new IsolationWindow(start, start + WindowWidth, 0, rtStart, rtStop));
            }
        }
        return new IsolationScheme("Dynamic DIA (8 x 8 m/z)", windows.OrderBy(w => w.Start).ToList());
    }

    [Fact]
    public void Scheme_IsACyclePerSegmentNotASingleSlotPerTime()
    {
        var scheme = DynamicScheme();

        Assert.Equal(4 * WindowsPerCycle, scheme.Windows.Count);
        Assert.True(scheme.IsScheduled);
        // Eight windows fire simultaneously in each segment - that is the "cycle", and it is what makes
        // this DIA rather than a targeted method.
        var firingAtStart = scheme.Windows.Count(w => w.IsOnAt(1.0));
        Assert.Equal(WindowsPerCycle, firingAtStart);
    }

    [Fact]
    public void SameMzIsCoveredByDifferentWindowsAtDifferentTimes()
    {
        // The defining property: segments overlap in m/z, so an m/z in the overlap belongs to segment 0's
        // cycle early on and to segment 1's cycle later. Time is what disambiguates them.
        var scheme = DynamicScheme();
        const double mz = 445.0; // inside both segment 0 (400-464) and segment 1 (460-524)? -> check both

        var early = scheme.IndicesCovering(mz, 2.0, 2.4).ToArray();
        var late = scheme.IndicesCovering(mz, 20.0, 20.4).ToArray();

        Assert.Single(early);
        Assert.Empty(late); // segment 1's cycle starts at 460, so 445 is no longer covered after 15 min
        Assert.Equal(0.0, scheme.Windows[early[0]].RtStart);

        // And an m/z inside the segment-0/segment-1 overlap resolves to whichever cycle was firing.
        const double overlap = 462.0; // segment 0: 400-464 covers it; segment 1: 460-524 covers it too
        var inSeg0 = scheme.IndicesCovering(overlap, 5.0, 5.2).Select(i => scheme.Windows[i].RtStart).ToArray();
        var inSeg1 = scheme.IndicesCovering(overlap, 20.0, 20.2).Select(i => scheme.Windows[i].RtStart).ToArray();
        Assert.Equal(new[] { 0.0 }, inSeg0);
        Assert.Equal(new[] { 15.0 }, inSeg1);
    }

    [Fact]
    public void Bin_CreditsPrecursorsToTheCycleThatWasFiring()
    {
        var scheme = DynamicScheme();
        var precursors = new[]
        {
            new DetectedPrecursor(405.0, 2.0, 2.4),    // segment 0 only
            new DetectedPrecursor(462.0, 5.0, 5.4),    // in the m/z overlap, eluting during segment 0
            new DetectedPrecursor(462.0, 20.0, 20.4),  // same m/z, eluting during segment 1
            new DetectedPrecursor(605.0, 47.0, 47.4),  // segment 3
        };

        var map = PrecursorDensity.Bin(precursors, scheme, rtBinMin: 0.5);

        Assert.Equal(0, map.PrecursorsOutsideRows);
        // Every precursor is counted exactly once, in the window whose cycle was running.
        var total = 0;
        for (var r = 0; r < map.MzBins; r++)
            for (var c = 0; c < map.RtBins; c++)
                total += map.Counts[r, c] > 0 ? 1 : 0;
        Assert.True(total >= 4, "each precursor should light up at least one cell");

        // The two 462.0 precursors land in DIFFERENT rows - the segment-0 window and the segment-1 window.
        var rowsFor462 = Enumerable.Range(0, map.MzBins)
            .Where(r => map.Rows[r].Contains(462.0))
            .ToList();
        Assert.Equal(2, rowsFor462.Count);
        var seg0Row = rowsFor462.First(r => map.Rows[r].RtStart == 0.0);
        var seg1Row = rowsFor462.First(r => map.Rows[r].RtStart == 15.0);
        Assert.True(RowTotal(map, seg0Row) > 0);
        Assert.True(RowTotal(map, seg1Row) > 0);
    }

    [Fact]
    public void DisplayGrid_ShowsEverySegmentAtTheSameMz()
    {
        // REGRESSION: the rasterizer used to choose one source window per m/z row, which for dynamic DIA
        // meant a single segment rendered and every other segment's cells came out blank.
        var scheme = DynamicScheme();
        var precursors = new List<DetectedPrecursor>();
        // One detection inside the m/z overlap in each of the first two segments.
        precursors.Add(new DetectedPrecursor(462.0, 5.0, 5.4));
        precursors.Add(new DetectedPrecursor(462.0, 20.0, 20.4));

        var map = PrecursorDensity.Bin(precursors, scheme, rtBinMin: 0.5);
        var grid = map.ToDisplayGrid(400);

        var row = RowForMz(map, 400, 462.0);
        var duringSegment0 = ColumnForRt(map, 5.2);
        var duringSegment1 = ColumnForRt(map, 20.2);

        // Both segments must be visible at this m/z, each showing its own detection.
        Assert.False(double.IsNaN(grid[row, duringSegment0]), "segment 0 should render at this m/z");
        Assert.False(double.IsNaN(grid[row, duringSegment1]), "segment 1 should render at this m/z");
        Assert.True(grid[row, duringSegment0] > 0);
        Assert.True(grid[row, duringSegment1] > 0);
    }

    [Fact]
    public void DisplayGrid_BlanksMzNotCoveredByTheCycleRunningThen()
    {
        // Dynamic DIA's whole point: at any time only ~300 m/z is being covered. The m/z the cycle has
        // already marched past was NOT acquired then, and must not read as an empty spectrum.
        var scheme = DynamicScheme();
        var map = PrecursorDensity.Bin(
            new[] { new DetectedPrecursor(405.0, 2.0, 2.4) }, scheme, rtBinMin: 0.5);
        var grid = map.ToDisplayGrid(400);

        var lowMzRow = RowForMz(map, 400, 405.0);
        Assert.False(double.IsNaN(grid[lowMzRow, ColumnForRt(map, 2.2)]), "covered during segment 0");
        Assert.True(double.IsNaN(grid[lowMzRow, ColumnForRt(map, 50.0)]),
            "the cycle has moved off 405 m/z by then - never acquired, not an empty spectrum");
    }

    [Fact]
    public void Bin_PeakStraddlingASegmentBoundaryCountsInBothCycles()
    {
        // A peak eluting across the switch really was fragmented by both cycles.
        var scheme = DynamicScheme();
        var straddler = new DetectedPrecursor(462.0, 14.6, 15.4); // spans the 15 min segment boundary

        var covering = scheme.IndicesCovering(straddler.Mz, straddler.RtStart, straddler.RtStop).ToList();
        Assert.Equal(2, covering.Count);

        var map = PrecursorDensity.Bin(new[] { straddler }, scheme, rtBinMin: 0.2);
        // Counted in each window only for the bins where that window was firing.
        var seg0Row = covering.First(i => map.Rows[i].RtStart == 0.0);
        var seg1Row = covering.First(i => map.Rows[i].RtStart == 15.0);
        Assert.True(RowTotal(map, seg0Row) > 0);
        Assert.True(RowTotal(map, seg1Row) > 0);
        Assert.True(map.Counts[seg0Row, ColumnForRt(map, 15.2)] == 0,
            "segment 0's window had stopped firing by 15.2 min");
    }

    [Fact]
    public void CatalogRoundTrip_PreservesADynamicDiaCycle()
    {
        // Whatever eventually reads a dynamic-DIA method, the windows have to survive being written to
        // the run's catalog and read back - that is what lets the tab bin on them when it is reopened on
        // an old output directory with no Skyline running. The per-window firing interval is PRISM's own
        // extension to Skyline's scheme XML, so it is the part that can silently go missing.
        var scheme = DynamicScheme(segments: 2, segmentMinutes: 10);
        var dir = Path.Combine(Path.GetTempPath(), "prism_dyndia_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var catalog = new IsolationSchemeCatalog();
            catalog.AddDocumentScheme("batch1", scheme);
            var path = Path.Combine(dir, IsolationSchemeCatalog.FileName);
            catalog.Save(path);

            var loaded = Assert.Single(IsolationSchemeCatalog.Load(path)!.UsableSchemes);

            Assert.Equal(scheme.Windows.Count, loaded.Windows.Count);
            Assert.True(loaded.IsScheduled);
            for (var i = 0; i < scheme.Windows.Count; i++)
            {
                Assert.Equal(scheme.Windows[i].Start, loaded.Windows[i].Start, 6);
                Assert.Equal(scheme.Windows[i].End, loaded.Windows[i].End, 6);
                Assert.Equal(scheme.Windows[i].RtStart, loaded.Windows[i].RtStart, 6);
                Assert.Equal(scheme.Windows[i].RtStop, loaded.Windows[i].RtStop, 6);
            }
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    private static int RowTotal(PrecursorDensityMap map, int row)
    {
        var total = 0;
        for (var c = 0; c < map.RtBins; c++)
            total += map.Counts[row, c];
        return total;
    }

    private static int RowForMz(PrecursorDensityMap map, int displayRows, double mz)
    {
        var height = (map.MzHigh - map.MzLow) / displayRows;
        return Math.Clamp((int)((map.MzHigh - mz) / height), 0, displayRows - 1);
    }

    private static int ColumnForRt(PrecursorDensityMap map, double rt) =>
        Math.Clamp((int)((rt - map.RtLow) / map.RtBinMin), 0, map.RtBins - 1);
}
