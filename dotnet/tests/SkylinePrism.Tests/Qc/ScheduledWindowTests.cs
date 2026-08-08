using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// Scheduled (PRM / MTM) acquisitions, where an isolation window is an m/z range crossed with the RT
/// interval it fires in - Skyline-Cadenza's <c>Slot</c> model. Skyline cannot import these from the data
/// (its importer needs a repeating cycle), so they come from the inclusion list that went to the
/// instrument.
/// </summary>
public class ScheduledWindowTests
{
    // One row per slot, as Cadenza's ThermoCsvWriter emits. Slot 1 is multiplexed (MTM: two co-eluting
    // members share a wider window); slots 2 and 3 are solo PRM-width targets. Slot 3 deliberately
    // repeats slot 2's m/z at a different time - the case that breaks an RT-blind model.
    private const string InclusionCsv = """
        Compound,Formula,Adduct,m/z,z,t start (min),t stop (min),Isolation Window (m/z),HCD Collision Energy
        "PEPTIDEK+2 | ELVISLIVESR+2",,,500.5000,2,10.0000,14.0000,4.7000,27.0
        SAMPLERPEP+2,,,700.2000,2,20.0000,24.0000,0.7000,27.0
        LATEELUTER+2,,,700.2000,2,40.0000,44.0000,0.7000,27.0
        """;

    private static IsolationScheme ParseCsv() =>
        ThermoInclusionList.Parse(InclusionCsv.Split('\n').Select(l => l.TrimEnd('\r')).ToList(), "MTM method");

    [Fact]
    public void InclusionList_BecomesWindowsCenteredOnMzWithTheirFiringInterval()
    {
        var scheme = ParseCsv();

        Assert.Equal(3, scheme.Windows.Count);
        Assert.True(scheme.IsScheduled);

        // Multiplexed slot: center 500.5, width 4.7 -> 498.15-502.85, firing 10-14 min.
        var mtm = scheme.Windows[0];
        Assert.Equal(498.15, mtm.Start, 6);
        Assert.Equal(502.85, mtm.End, 6);
        Assert.Equal(4.7, mtm.Width, 6);
        Assert.True(mtm.IsScheduled);
        Assert.Equal(10.0, mtm.RtStart, 6);
        Assert.Equal(14.0, mtm.RtStop, 6);

        // Solo PRM slot: 0.7 Th around 700.2.
        Assert.Equal(0.7, scheme.Windows[1].Width, 6);
        Assert.Contains("RT-scheduled", scheme.Describe());
    }

    [Fact]
    public void ScheduledWindow_OnlyClaimsPrecursorsElutingWhileItFires()
    {
        // The whole point of the RT dimension. Two targets share m/z 700.2 but are scheduled 20 minutes
        // apart; each must be credited to its OWN slot, not to both.
        var scheme = ParseCsv();
        var early = new DetectedPrecursor(700.2, 21.0, 21.4);
        var late = new DetectedPrecursor(700.2, 41.0, 41.4);

        Assert.Equal(new[] { 1 }, scheme.IndicesCovering(early.Mz, early.RtStart, early.RtStop).ToArray());
        Assert.Equal(new[] { 2 }, scheme.IndicesCovering(late.Mz, late.RtStart, late.RtStop).ToArray());

        // An RT-blind match would put each precursor in both slots - that is the bug this guards.
        Assert.Equal(new[] { 1, 2 }, scheme.IndicesContaining(700.2).ToArray());
    }

    [Fact]
    public void Bin_CountsMultiplexedMembersTogetherAndSoloTargetsAlone()
    {
        var scheme = ParseCsv();
        var precursors = new[]
        {
            new DetectedPrecursor(499.9, 11.0, 11.5),  // MTM member 1
            new DetectedPrecursor(501.2, 11.2, 11.6),  // MTM member 2 - same slot, co-eluting
            new DetectedPrecursor(700.2, 21.0, 21.4),  // solo PRM target
            new DetectedPrecursor(700.2, 41.0, 41.4),  // the later target at the same m/z
        };

        var map = PrecursorDensity.Bin(precursors, scheme, rtBinMin: 0.5);

        Assert.Equal(3, map.MzBins);
        Assert.Equal(0, map.PrecursorsOutsideRows);
        // The multiplexed slot carries 2 precursors at once; the PRM slots carry 1 each.
        Assert.Equal(2, RowMax(map, 0));
        Assert.Equal(1, RowMax(map, 1));
        Assert.Equal(1, RowMax(map, 2));
    }

    [Fact]
    public void Bin_TimeAxisCoversTheWholeScheduleEvenWhereNothingWasDetected()
    {
        // A slot that fired but found nothing is exactly what someone opens this plot to spot, so the RT
        // axis must span the schedule, not just the detections.
        var scheme = ParseCsv();
        var map = PrecursorDensity.Bin(
            new[] { new DetectedPrecursor(499.9, 11.0, 11.5) }, scheme, rtBinMin: 0.5);

        Assert.True(map.RtLow <= 10.0, $"RT axis starts at {map.RtLow}, before the first slot fires");
        Assert.True(map.RtHigh >= 44.0, $"RT axis ends at {map.RtHigh}, after the last slot stops");
    }

    [Fact]
    public void DisplayGrid_MarksUnscheduledTimeAsNotAcquiredNotAsZero()
    {
        // A zero must always mean "acquired, nothing detected". Outside a slot's firing interval nothing
        // was acquired at all, so those cells are NaN and draw as a gap.
        var scheme = ParseCsv();
        var map = PrecursorDensity.Bin(
            new[] { new DetectedPrecursor(499.9, 11.0, 11.5) }, scheme, rtBinMin: 0.5);
        var grid = map.ToDisplayGrid(300);

        // Find a display row inside the multiplexed slot (498.15-502.85).
        var row = RowForMz(map, grid.GetLength(0), 500.5);
        var duringItsWindow = ColumnForRt(map, 11.25);
        var longAfterItStops = ColumnForRt(map, 30.0);

        Assert.False(double.IsNaN(grid[row, duringItsWindow]), "the slot was firing here");
        Assert.True(double.IsNaN(grid[row, longAfterItStops]), "the slot was NOT firing here");
    }

    [Fact]
    public void UnscheduledWindowsStayAlwaysOn()
    {
        // A DIA scheme has no schedule: every window is on for the whole gradient, and nothing about the
        // RT dimension may change that behavior.
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
    public void InclusionList_TolerantOfHeaderSpellingAndJunkRows()
    {
        var scheme = ThermoInclusionList.Parse(new[]
        {
            "Compound,m/z,z,T Start,T Stop,Isolation_Window",  // different spacing/underscores/case
            "A,500.0,2,1.0,2.0,2.0",
            "",                                                 // blank
            "B,not-a-number,2,1.0,2.0,2.0",                     // unparseable m/z -> skipped
            "C,600.0,2,,,3.0",                                  // no schedule -> always-on window
        });

        Assert.Equal(2, scheme.Windows.Count);
        Assert.True(scheme.Windows[0].IsScheduled);   // 500 +/- 1
        Assert.False(scheme.Windows[1].IsScheduled);  // 600 +/- 1.5, no RT columns
        Assert.Equal(499.0, scheme.Windows[0].Start, 6);
        Assert.Equal(598.5, scheme.Windows[1].Start, 6);
    }

    [Fact]
    public void InclusionList_RejectsFilesThatAreNotInclusionLists()
    {
        // A PRISM report, say. Fail with a specific reason rather than a silently empty scheme.
        var ex = Assert.Throws<InvalidDataException>(() => ThermoInclusionList.Parse(new[]
        {
            "Peptide,Protein,Area",
            "PEPTIDEK,P12345,1234",
        }));
        Assert.Contains("m/z", ex.Message);

        // Has m/z but no window width - the widths cannot be invented.
        var noWidth = Assert.Throws<InvalidDataException>(() => ThermoInclusionList.Parse(new[]
        {
            "Compound,m/z,z", "A,500,2",
        }));
        Assert.Contains("Isolation Window", noWidth.Message);

        Assert.Throws<InvalidDataException>(() => ThermoInclusionList.Parse(Array.Empty<string>()));
        // Header only, no rows.
        Assert.Throws<InvalidDataException>(() => ThermoInclusionList.Parse(new[]
        {
            "Compound,m/z,Isolation Window (m/z)",
        }));
    }

    [Fact]
    public void ScheduledWindows_SurviveTheRunCatalogRoundTrip()
    {
        // Reopening an old output directory must still know when each slot fired, or the map silently
        // reverts to treating a scheduled method as always-on.
        var dir = Path.Combine(Path.GetTempPath(), "prism_sched_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var catalog = new IsolationSchemeCatalog();
            catalog.AddLibraryScheme(ParseCsv());
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
