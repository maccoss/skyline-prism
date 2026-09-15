using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Qc;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// The (isolation window x RT) precursor-density map behind the tool's Spectrum density tab: the
/// binning arithmetic, and the query that pulls one run's detected precursors out of a merged report.
/// </summary>
public class PrecursorDensityTests
{
    private static string MergedGolden => Fixtures.Path2("mini", "merge", "merged_data.parquet");

    /// <summary>
    /// 0.01 min, the same bin the ion accounting views use - 0.6 s, about one acquisition cycle, which
    /// is what makes a cell one spectrum.
    /// </summary>
    [Fact]
    public void TheDefaultRtBinIsSixHundredMilliseconds()
    {
        Assert.Equal(0.01, PrecursorDensity.DefaultRtBinMin);
    }

    /// <summary>
    /// A fine bin is actually honored over a real gradient, rather than widened back.
    /// </summary>
    /// <remarks>
    /// The RT axis used to share the m/z axis's 4,000-bin cap, so 0.01 min over an hour - 6,000
    /// bins - came back as 0.015 with nothing but the status line to say so. Making it the default
    /// while that cap stood would have delivered a coarser map than the one asked for.
    /// </remarks>
    [Fact]
    public void AFineBinSurvivesAnHourLongGradient()
    {
        var map = PrecursorDensity.Bin(
            new[] { new DetectedPrecursor(500.4, 1.0, 61.0) },
            mzBinTh: 2.0, rtBinMin: PrecursorDensity.DefaultRtBinMin);

        Assert.Equal(0.01, map.RtBinMin, 6);
        Assert.True(map.RtBins > 5900, $"{map.RtBins} RT bins is coarser than asked for");
        Assert.False(map.RtBinWidened, "an honored bin must not report itself as widened");
    }

    /// <summary>
    /// And it gives way when the grid would not fit, because the map reports the bin it used.
    /// </summary>
    /// <remarks>
    /// The real bound is cells, not bins on one axis: the grid is nMz x nRt ints, so a fine bin on
    /// both at once is what runs a machine out of memory. The RT bin is the one that widens - the
    /// m/z rows are the acquisition's own isolation windows and are not PRISM's to coarsen.
    /// </remarks>
    [Fact]
    public void AVeryWideMzRangeWidensTheRtBinRatherThanExhaustingMemory()
    {
        // 0.05 Th over 400 Th is 8,000 m/z rows; at 0.01 min the full hour would be 48M cells.
        var map = PrecursorDensity.Bin(
            new[] { new DetectedPrecursor(500.4, 1.0, 61.0), new DetectedPrecursor(899.0, 1.0, 61.0) },
            mzBinTh: 0.05, rtBinMin: 0.01);

        Assert.True(map.RtBinMin > 0.01, "the RT bin did not widen");

        // And it SAYS so. A cell that spans several cycles reports the worst of the spectra inside it
        // rather than one spectrum's load, so the histogram and load curve shift right - not something
        // a reader should have to infer from the bin width.
        Assert.True(map.RtBinWidened);
        Assert.Equal(0.01, map.RtBinRequested, 6);
        Assert.True(
            (long)map.MzBins * map.RtBins <= 12_000_000,
            $"{map.MzBins} x {map.RtBins} is over the cell budget");
    }

    /// <summary>
    /// Two peptides that were never in the same spectrum are not counted as though they were.
    /// </summary>
    /// <remarks>
    /// This is the defect the whole cell definition turned on. Each precursor used to add a count to
    /// every bin its peak spanned, so a cell held the UNION of everything that eluted during that
    /// stretch - and one peptide finishing before the next began still came out as two co-detected.
    /// A cell answers "how many peptides did one spectrum have to deal with", and no spectrum ever
    /// saw a union over time.
    /// </remarks>
    [Fact]
    public void PeaksThatDoNotOverlapInTimeAreNotCoDetected()
    {
        var map = PrecursorDensity.Bin(
            new[]
            {
                new DetectedPrecursor(500.4, 10.0, 10.2),   // done before the next starts
                new DetectedPrecursor(500.6, 10.6, 10.8),
            },
            mzBinTh: 2.0, rtBinMin: 1.0);                   // one bin holding both

        Assert.Equal(1, map.MaxCount);
    }

    /// <summary>And two that DO overlap still are.</summary>
    [Fact]
    public void PeaksThatOverlapInTimeAreCoDetected()
    {
        var map = PrecursorDensity.Bin(
            new[]
            {
                new DetectedPrecursor(500.4, 10.0, 10.5),
                new DetectedPrecursor(500.6, 10.4, 10.9),   // overlaps 10.4 - 10.5
            },
            mzBinTh: 2.0, rtBinMin: 1.0);

        Assert.Equal(2, map.MaxCount);
    }

    /// <summary>
    /// The count no longer moves when the bin moves, which is what makes it a measurement.
    /// </summary>
    /// <remarks>
    /// A bin ten times a cycle used to report ten times the co-detection, so the number a reader
    /// quoted depended on a display setting. Sweeping the peak boundaries makes a column report the
    /// worst spectrum inside it: narrowing refines the answer, widening never invents one.
    /// </remarks>
    [Fact]
    public void TheCountDoesNotDependOnTheBinWidth()
    {
        var peaks = new[]
        {
            new DetectedPrecursor(500.4, 10.0, 10.2),
            new DetectedPrecursor(500.5, 10.05, 10.25),
            new DetectedPrecursor(500.6, 11.0, 11.2),
            new DetectedPrecursor(500.7, 12.0, 12.2),
        };

        Assert.Equal(
            PrecursorDensity.Bin(peaks, mzBinTh: 2.0, rtBinMin: 0.01).MaxCount,
            PrecursorDensity.Bin(peaks, mzBinTh: 2.0, rtBinMin: 1.0).MaxCount);
    }

    /// <summary>
    /// Including when the only overlap is a shared instant on a column edge. A peak is a closed
    /// interval, so two that meet at exactly one instant were both in a spectrum acquired then; the
    /// sweep's zero-length segment used to write nothing when that instant fell on an edge, so the
    /// same data gave 1 at one bin and 2 at another.
    /// </summary>
    [Fact]
    public void TouchingPeaksAreConcurrentWhereverTheColumnEdgesFall()
    {
        var peaks = new[]
        {
            new DetectedPrecursor(500.4, 10.0, 11.0),
            new DetectedPrecursor(500.6, 11.0, 12.0),   // meets the first at exactly 11.0
        };

        Assert.Equal(2, PrecursorDensity.Bin(peaks, mzBinTh: 2.0, rtBinMin: 1.0).MaxCount); // 11.0 on an edge
        Assert.Equal(2, PrecursorDensity.Bin(peaks, mzBinTh: 2.0, rtBinMin: 0.3).MaxCount); // 11.0 inside a column
    }

    /// <summary>A peak with no width still happened, and still appears.</summary>
    [Fact]
    public void AZeroWidthPeakIsNotLost()
    {
        var map = PrecursorDensity.Bin(
            new[] { new DetectedPrecursor(500.4, 10.0, 10.0) }, mzBinTh: 2.0, rtBinMin: 0.01);

        Assert.Equal(1, map.MaxCount);
    }

    /// <summary>
    /// Also at the very end of the axis, on a column edge - the one place the old zero-width guard
    /// refused, because the instant mapped to a column index one past the last.
    /// </summary>
    [Fact]
    public void AZeroWidthPeakAtTheEndOfTheAxisIsNotLost()
    {
        var map = PrecursorDensity.Bin(
            new[]
            {
                new DetectedPrecursor(500.4, 10.0, 11.0),
                new DetectedPrecursor(502.5, 12.0, 12.0),   // its own row, at rtHi, on a column edge
            },
            mzBinTh: 2.0, rtBinMin: 1.0);

        Assert.Equal(2, map.RtBins);
        Assert.Equal(1, map.Counts[1, 1]);
    }

    /// <summary>
    /// A scheduled window is credited only while it was firing.
    /// </summary>
    /// <remarks>
    /// The peak is clipped to the window's firing interval before the sweep, and the sweep writes only
    /// columns the window was on for at their center - the same rule the views read with. This is the
    /// path where getting it wrong credits a precursor to a same-m/z window that fired in a different RT
    /// segment, which is what the Covers() check exists to stop.
    /// </remarks>
    [Fact]
    public void AScheduledWindowIsCreditedOnlyWhileItWasFiring()
    {
        var scheme = new IsolationScheme(
            "scheduled", new[] { new IsolationWindow(500.0, 510.0, 0, 10.0, 20.0) });

        // The peak runs on past the end of the window's schedule.
        var map = PrecursorDensity.Bin(
            new[] { new DetectedPrecursor(505.0, 18.0, 25.0) }, scheme, rtBinMin: 1.0);

        Assert.Equal(1, map.Counts[0, map.ColumnAt(19.0)]);
        Assert.Equal(0, map.Counts[0, map.ColumnAt(23.0)]);
    }

    /// <summary>
    /// A count never lands in a cell the views treat as not acquired, even when a window's edge falls
    /// in the middle of a column.
    /// </summary>
    /// <remarks>
    /// The fill and the views used to apply two different rules: the fill wrote every column the clipped
    /// peak touched, while the display grid, histogram and load curve counted only columns whose CENTER
    /// the window was on for. A window edge inside a column then held a count that MaxCount and the
    /// hover readout reported and nothing else drew - the status line said "busiest spectrum 2" over a
    /// map whose visible maximum was 1. One shared rule now, so this asserts the surfaces agree.
    /// </remarks>
    [Fact]
    public void AWindowEdgeInsideAColumnNeverPutsACountWhereNoViewLooks()
    {
        var scheme = new IsolationScheme(
            "scheduled", new[] { new IsolationWindow(500.0, 510.0, 0, 10.0, 20.0) });

        // rtLo is 9.6, so column 10 is [19.6, 20.6) with its center at 20.1 - after the window stopped.
        // The two peaks overlap only in [19.7, 20.0], inside that column.
        var map = PrecursorDensity.Bin(
            new[]
            {
                new DetectedPrecursor(505.0, 9.6, 20.0),
                new DetectedPrecursor(506.0, 19.7, 20.0),
            },
            scheme, rtBinMin: 1.0);

        var histogram = map.PrecursorsPerSpectrumHistogram();
        Assert.Equal(0, map.Counts[0, map.ColumnAt(20.1)]);
        Assert.Equal(1, map.MaxCount);
        Assert.Equal(map.MaxCount, histogram.Length - 1);
        Assert.Equal(10, histogram[1]); // columns 0-9, the ones the window was on for at their center
    }

    /// <summary>And a peak wholly outside the schedule is not credited to it at all.</summary>
    [Fact]
    public void APeakOutsideTheScheduleCountsAsOutsideTheRows()
    {
        var scheme = new IsolationScheme(
            "scheduled", new[] { new IsolationWindow(500.0, 510.0, 0, 10.0, 20.0) });

        var map = PrecursorDensity.Bin(
            new[] { new DetectedPrecursor(505.0, 30.0, 35.0) }, scheme, rtBinMin: 1.0);

        Assert.Equal(1, map.PrecursorsOutsideRows);
    }

    /// <summary>
    /// A column about one cycle wide says WHICH spectra saw the co-detection; a wide one says they all did.
    /// </summary>
    /// <remarks>
    /// The worst case is the same either way - that is the bin-independence the sweep buys. What the
    /// fine column adds is that only a few spectra carried both peptides, where a wide column reports
    /// its worst spectrum for the whole stretch. This is the case that distinguishes "max within the
    /// column" from "concurrency at an instant", and the premise the default bin rests on.
    /// </remarks>
    [Fact]
    public void AFineColumnLocalizesCoDetectionThatAWideOneSpreads()
    {
        var peaks = new[]
        {
            new DetectedPrecursor(500.4, 10.00, 10.30),
            new DetectedPrecursor(500.5, 10.28, 10.60),   // overlaps for 0.02 min
        };

        var fine = PrecursorDensity.Bin(peaks, mzBinTh: 2.0, rtBinMin: 0.01);
        var coarse = PrecursorDensity.Bin(peaks, mzBinTh: 2.0, rtBinMin: 0.3);

        // The worst case is the same either way - that is the bin independence.
        Assert.Equal(2, fine.MaxCount);
        Assert.Equal(2, coarse.MaxCount);

        // What differs is how much of the run is TOLD it was that busy. Asserted as a fraction, and
        // against a coarse grid with more than one column: a single-column coarse map would make the
        // comparison 1 == 1, which any implementation passes, including the one this replaced.
        Assert.True(coarse.RtBins > 1, "a one-column coarse map cannot show the contrast");
        var fineShare = fine.PrecursorsPerSpectrumHistogram()[2] / (double)fine.RtBins;
        var coarseShare = coarse.PrecursorsPerSpectrumHistogram()[2] / (double)coarse.RtBins;

        Assert.True(fineShare < 0.10, $"fine grid flagged {fineShare:P0} of the run");
        Assert.True(coarseShare >= 0.40, $"coarse grid flagged only {coarseShare:P0} of the run");
    }

    /// <summary>
    /// A precursor the window never acquired adds nothing to the concurrency of one it did.
    /// </summary>
    /// <remarks>
    /// <para>The exclusion is Covers(), which drops a precursor whose peak never overlaps the firing
    /// interval at all - and this pins that it survived the move from counting to sweeping, with a
    /// second precursor present so the two are actually combined.</para>
    ///
    /// <para>It does NOT isolate the clip that narrows a surviving peak to the firing interval, and
    /// no test can: for two peaks to overlap each other only outside the window while each still
    /// overlaps the window, they would have to touch it at disjoint ends, which leaves them not
    /// overlapping at all. The clip's effect is on WHICH COLUMNS a peak marks, which is what
    /// AScheduledWindowIsCreditedOnlyWhileItWasFiring asserts.</para>
    /// </remarks>
    [Fact]
    public void APrecursorOutsideTheScheduleAddsNothingToConcurrency()
    {
        var scheme = new IsolationScheme(
            "scheduled", new[] { new IsolationWindow(500.0, 510.0, 0, 10.0, 20.0) });

        var map = PrecursorDensity.Bin(
            new[]
            {
                new DetectedPrecursor(505.0, 12.0, 25.0),
                new DetectedPrecursor(506.0, 22.0, 30.0),   // overlaps the first only after 22 min
            },
            scheme, rtBinMin: 1.0);

        // The window stopped firing at 20; the overlap starts at 22.
        Assert.Equal(1, map.MaxCount);
    }

    /// <summary>A non-finite boundary is dropped, and takes nothing with it.</summary>
    /// <remarks>
    /// Load drops these, but Bin is public. The hazard is not an exception: the event comparator is
    /// consistent with NaN (it sorts below everything), and (int)Math.Floor(NaN) saturates to 0 on this
    /// runtime, so an unguarded NaN start would sort first, open a peak at column 0 and paint phantom
    /// concurrency from the axis start to its stop - wrong numbers, nothing thrown.
    /// </remarks>
    [Fact]
    public void ANonFiniteBoundaryIsDropped()
    {
        var map = PrecursorDensity.Bin(
            new[]
            {
                new DetectedPrecursor(500.4, double.NaN, 10.2),
                new DetectedPrecursor(500.5, 10.0, 10.2),
            },
            mzBinTh: 2.0, rtBinMin: 0.01);

        Assert.Equal(1, map.MaxCount);
        Assert.Equal(10.0, map.RtLow);
    }

    /// <summary>
    /// A peak that ends before it starts is dropped by both overloads alike, and takes nothing from its
    /// neighbors.
    /// </summary>
    /// <remarks>
    /// Fed to the sweep, an inverted peak's close sorts before its open, so live went NEGATIVE across a
    /// neighbor's elution and that stretch read as empty: peaks (10.0-12.0) and (13.0-11.0) gave a row
    /// of [1, 0] with the first peak still eluting across the second column. The scheme overload
    /// happened to skip it and the uniform one did not, so the same input gave two answers.
    /// </remarks>
    [Fact]
    public void AnInvertedPeakIsDroppedAndTakesNothingFromItsNeighbors()
    {
        var peaks = new[]
        {
            new DetectedPrecursor(505.0, 10.0, 12.0),
            new DetectedPrecursor(506.0, 13.0, 11.0),   // ends before it starts
        };

        var uniform = PrecursorDensity.Bin(peaks, mzBinTh: 10.0, rtBinMin: 1.0);
        Assert.Equal(new[] { 1, 1 }, Row(uniform, 0));

        var onScheme = PrecursorDensity.Bin(peaks, Scheme(("dia", 500, 510)), rtBinMin: 1.0);
        Assert.Equal(new[] { 1, 1 }, Row(onScheme, 0));
        Assert.Equal(0, onScheme.PrecursorsOutsideRows);
    }

    /// <summary>Input with nothing placeable is an empty map, not one whose axis starts at infinity.</summary>
    [Fact]
    public void InputWithNoValidPrecursorGivesAnEmptyMap()
    {
        var invalid = new[]
        {
            new DetectedPrecursor(500.4, double.NaN, 10.2),
            new DetectedPrecursor(500.5, 10.0, double.PositiveInfinity),
        };

        Assert.True(PrecursorDensity.Bin(invalid, mzBinTh: 2.0, rtBinMin: 0.01).IsEmpty);
        Assert.True(PrecursorDensity.Bin(invalid, Scheme(("dia", 500, 510)), rtBinMin: 0.01).IsEmpty);
    }

    /// <summary>
    /// The readouts' column lookup floors and refuses the off-axis side, where a plain cast truncated a
    /// cursor just left of the axis onto column 0.
    /// </summary>
    [Fact]
    public void ColumnAt_FloorsAndRejectsAnythingOffTheAxis()
    {
        var map = PrecursorDensity.Bin(
            new[] { new DetectedPrecursor(500.4, 10.0, 10.25) }, mzBinTh: 2.0, rtBinMin: 0.1);

        Assert.Equal(-1, map.ColumnAt(9.95));
        Assert.Equal(0, map.ColumnAt(10.0));
        Assert.Equal(1, map.ColumnAt(10.15));
        Assert.Equal(2, map.ColumnAt(10.25));
        Assert.Equal(-1, map.ColumnAt(10.35));
        Assert.Equal(-1, map.ColumnAt(double.NaN));
    }

    [Fact]
    public void Bin_OnePeakFillsEveryColumnItSpans()
    {
        // One precursor at m/z 500.4, eluting 10.0 -> 10.25 min. With 0.1 min bins the peak covers
        // bins 0, 1 and 2 (the last one partially - a spectrum acquired then still sees the peptide).
        var map = PrecursorDensity.Bin(
            new[] { new DetectedPrecursor(500.4, 10.0, 10.25) }, mzBinTh: 2.0, rtBinMin: 0.1);

        Assert.Equal(500.0, map.MzLow);   // floored to a whole Th
        Assert.Equal(10.0, map.RtLow);
        Assert.Equal(1, map.MzBins);
        Assert.Equal(3, map.RtBins);
        Assert.Equal(new[] { 1, 1, 1 }, Row(map, 0));
        Assert.Equal(1, map.MaxCount);
    }

    [Fact]
    public void Bin_SeparatesPrecursorsByIsolationWindow()
    {
        // Two precursors 6 Th apart, co-eluting. At a 2 Th window they land in different rows (different
        // DIA spectra); at a 10 Th window the same spectrum carries both.
        var precursors = new[]
        {
            new DetectedPrecursor(500.0, 5.0, 5.2),
            new DetectedPrecursor(506.0, 5.0, 5.2),
        };

        var narrow = PrecursorDensity.Bin(precursors, mzBinTh: 2.0, rtBinMin: 0.1);
        Assert.Equal(3, narrow.MzBins);  // 500 -> 506 spans 3 windows of 2 Th
        Assert.Equal(1, narrow.MaxCount); // one precursor each in the bottom and top window

        var wide = PrecursorDensity.Bin(precursors, mzBinTh: 10.0, rtBinMin: 0.1);
        Assert.Equal(1, wide.MzBins);
        Assert.Equal(2, wide.MaxCount); // both precursors in one window = one crowded spectrum
    }

    [Fact]
    public void Bin_CoElutingPrecursorsStackInTheSameCell()
    {
        var map = PrecursorDensity.Bin(
            new[]
            {
                new DetectedPrecursor(700.1, 20.0, 20.1),
                new DetectedPrecursor(700.9, 20.05, 20.3), // same 2 Th window, overlapping peak
                new DetectedPrecursor(700.5, 25.0, 25.1),  // same window, well after the others
            },
            mzBinTh: 2.0, rtBinMin: 0.1);

        Assert.Equal(1, map.MzBins);
        Assert.Equal(2, map.MaxCount);              // the two overlapping peaks, not the third
        Assert.Equal(2, Row(map, 0).First());       // first RT bin holds both
        Assert.Equal(0, Row(map, 0)[10]);           // the quiet stretch between them
    }

    [Fact]
    public void Bin_EmptyInputGivesEmptyMap()
    {
        var map = PrecursorDensity.Bin(Array.Empty<DetectedPrecursor>());
        Assert.True(map.IsEmpty);
        Assert.Equal(0, map.MaxCount);
    }

    [Fact]
    public void Bin_RejectsNonPositiveBins()
    {
        var one = new[] { new DetectedPrecursor(500, 1, 2) };
        Assert.Throws<ArgumentOutOfRangeException>(() => PrecursorDensity.Bin(one, mzBinTh: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => PrecursorDensity.Bin(one, rtBinMin: -1));
    }

    [Fact]
    public void Bin_WidensBinsTooFineToRender()
    {
        // A 1e-6 Th bin over a 500 Th range would be half a billion rows. The grid stays bounded and the
        // map reports the bin size it actually used, so the plot never mislabels its own axes.
        var map = PrecursorDensity.Bin(
            new[] { new DetectedPrecursor(400, 0, 60), new DetectedPrecursor(900, 0, 60) },
            mzBinTh: 1e-6, rtBinMin: 1e-6);

        Assert.True(map.MzBins <= 4000, $"m/z bins: {map.MzBins}");
        Assert.True(map.RtBins <= 4000, $"RT bins: {map.RtBins}");
        Assert.True(map.Rows[0].Width > 1e-6);
        Assert.True(map.RtBinMin > 1e-6);
    }

    [Fact]
    public void Bin_OnRealWindows_UsesTheSchemesOwnEdges()
    {
        // A scheme starting at 400 with 25 Th windows. A precursor at 412 belongs to window 0 (400-425),
        // NOT to a bin whose edges happen to fall where a uniform grid over the observed data put them.
        var scheme = Scheme(("s", 400, 425), ("s", 425, 450), ("s", 450, 475));
        var map = PrecursorDensity.Bin(
            new[]
            {
                new DetectedPrecursor(412.0, 10.0, 10.2),
                new DetectedPrecursor(424.9, 10.0, 10.2), // same window as 412 despite being 13 Th away
                new DetectedPrecursor(425.1, 10.0, 10.2), // just over the edge -> next window
            },
            scheme, rtBinMin: 0.1);

        Assert.Equal(3, map.MzBins);            // the scheme's windows, not the data's range
        Assert.Equal(400, map.MzLow);
        Assert.Equal(475, map.MzHigh);
        Assert.Equal(2, map.Counts[0, 0]);      // 412 + 424.9
        Assert.Equal(1, map.Counts[1, 0]);      // 425.1
        Assert.Equal(0, map.Counts[2, 0]);      // an acquired-but-empty window stays in the map
        Assert.Equal(0, map.PrecursorsOutsideRows);
        Assert.Equal("s", map.RowSource);
    }

    [Fact]
    public void Bin_OnRealWindows_CountsPrecursorsOutsideEveryWindow()
    {
        // The wrong scheme for the data: precursors below its range are reported, never clamped into the
        // nearest window, because that count is how the user sees the scheme does not fit.
        var scheme = Scheme(("s", 600, 625));
        var map = PrecursorDensity.Bin(
            new[]
            {
                new DetectedPrecursor(610.0, 5.0, 5.1),
                new DetectedPrecursor(450.0, 5.0, 5.1),
                new DetectedPrecursor(900.0, 5.0, 5.1),
            },
            scheme, rtBinMin: 0.1);

        Assert.Equal(1, map.MaxCount);
        Assert.Equal(2, map.PrecursorsOutsideRows);
    }

    [Fact]
    public void Bin_OnRealWindows_CountsOverlappingWindowsSeparately()
    {
        // Staggered/overlapping DIA: a precursor in the overlap really was fragmented in both windows.
        var scheme = Scheme(("stagger", 500, 520), ("stagger", 510, 530));
        var map = PrecursorDensity.Bin(
            new[] { new DetectedPrecursor(515.0, 1.0, 1.05) }, scheme, rtBinMin: 0.1);

        Assert.Equal(1, map.Counts[0, 0]);
        Assert.Equal(1, map.Counts[1, 0]);
        Assert.Equal(0, map.PrecursorsOutsideRows);
    }

    [Fact]
    public void Bin_OnRealWindows_RejectsASchemeWithNoWindows()
    {
        // "Results only" is a named scheme with no windows - it cannot be binned on, and must not be
        // silently treated as an empty grid.
        var resultsOnly = new IsolationScheme(IsolationScheme.ResultsOnlyName, Array.Empty<IsolationWindow>());
        Assert.Throws<ArgumentException>(() => PrecursorDensity.Bin(
            new[] { new DetectedPrecursor(500, 1, 2) }, resultsOnly));
    }

    [Fact]
    public void ToDisplayGrid_PreservesVariableWindowWidths()
    {
        // A variable-width scheme: narrow at low m/z, wide at high m/z. The uniform display grid must put
        // the counts at the right m/z, so the wide window occupies proportionally more rows.
        var scheme = Scheme(("vw", 400, 410), ("vw", 410, 500));
        var map = PrecursorDensity.Bin(
            new[]
            {
                new DetectedPrecursor(405.0, 1.0, 1.05),
                new DetectedPrecursor(450.0, 1.0, 1.05),
                new DetectedPrecursor(460.0, 1.0, 1.05),
            },
            scheme, rtBinMin: 0.1);

        var grid = map.ToDisplayGrid(100); // 100 rows over 400-500 -> 1 Th each
        // Row 0 is the TOP (high m/z) = the wide window, which holds 2 precursors.
        Assert.Equal(2, grid[0, 0]);
        // The bottom 10 rows are the narrow 400-410 window, which holds 1.
        Assert.Equal(1, grid[99, 0]);
        // The wide window covers 90% of the m/z range, so ~90 of the 100 display rows show its count.
        var wideRows = 0;
        for (var r = 0; r < 100; r++)
            if (grid[r, 0] == 2)
                wideRows++;
        Assert.InRange(wideRows, 88, 92);
    }

    [Fact]
    public void Coverage_DistinguishesTheRightSchemeFromTheWrongOne()
    {
        var mz = new[] { 405.0, 430.0, 455.0, 470.0 };
        Assert.Equal(1.0, Scheme(("right", 400, 425), ("right", 425, 450), ("right", 450, 475)).Coverage(mz));
        Assert.Equal(0.0, Scheme(("wrong", 700, 725)).Coverage(mz));
    }

    private static IsolationScheme Scheme(params (string Name, double Start, double End)[] windows) =>
        new(windows[0].Name, windows.Select(w => new IsolationWindow(w.Start, w.End)).ToList());

    [Fact]
    public void Resolve_FindsColumnsInBothExportSpellings()
    {
        var csvStyle = PrecursorDensity.Resolve(new[]
        {
            "Sample ID", "Replicate Name", "Peptide Modified Sequence Unimod Ids", "Precursor Charge",
            "Precursor Mz", "Start Time", "End Time", "Detection Q Value",
        });
        Assert.NotNull(csvStyle);
        Assert.Equal("Sample ID", csvStyle!.Sample); // the batch-disambiguated column wins
        Assert.Equal("Detection Q Value", csvStyle.DetectionQValue);

        // Parquet/invariant spelling: no spaces. FindColumn normalizes case, spaces and underscores.
        var parquetStyle = PrecursorDensity.Resolve(new[]
        {
            "ReplicateName", "PeptideModifiedSequenceUnimodIds", "PrecursorCharge", "PrecursorMz",
            "StartTime", "EndTime",
        });
        Assert.NotNull(parquetStyle);
        Assert.Equal("ReplicateName", parquetStyle!.Sample);
        Assert.Null(parquetStyle.DetectionQValue); // optional - the q-value filter is simply unavailable
    }

    [Fact]
    public void Resolve_ReturnsNullWhenBoundariesAreMissing()
    {
        // A report exported without the peak-boundary columns cannot answer this question at all.
        Assert.Null(PrecursorDensity.Resolve(new[]
        {
            "Replicate Name", "Peptide Modified Sequence", "Precursor Charge", "Precursor Mz", "Area",
        }));
    }

    [Fact]
    public void Load_ReadsOneRowPerPrecursorFromTheMergedReport()
    {
        Assert.True(File.Exists(MergedGolden), $"golden fixture missing: {MergedGolden}");
        var cols = PrecursorDensity.Resolve(ParquetTable.ReadColumnNames(MergedGolden).ToHashSet());
        Assert.NotNull(cols);

        var samples = MergedParquetReader.GetSortedSamples(MergedDataset.Open(MergedGolden), cols!.Sample);
        Assert.NotEmpty(samples);

        var precursors = PrecursorDensity.Load(MergedDataset.Open(MergedGolden), cols, samples[0], qValueCutoff: null);
        Assert.NotEmpty(precursors);
        Assert.All(precursors, p =>
        {
            Assert.True(p.Mz > 0, "precursor m/z should be positive");
            Assert.True(p.RtStop >= p.RtStart, "peak must not end before it starts");
            Assert.True(double.IsFinite(p.RtStart) && double.IsFinite(p.RtStop));
        });

        // The report is transition-level; the map is precursor-level, so it must be the smaller of the two.
        var transitionRows = CountRows(MergedGolden, cols.Sample, samples[0]);
        Assert.True(precursors.Count < transitionRows,
            $"{precursors.Count} precursors from {transitionRows} transition rows");

        // And it bins into a usable map.
        var map = PrecursorDensity.Bin(precursors);
        Assert.False(map.IsEmpty);
        Assert.True(map.MaxCount >= 1);
        Assert.True(map.MzHigh > map.MzLow && map.RtHigh > map.RtLow);
    }

    [Fact]
    public void Load_QValueCutoffOnlyKeepsConfidentDetections()
    {
        var cols = PrecursorDensity.Resolve(ParquetTable.ReadColumnNames(MergedGolden).ToHashSet());
        Assert.NotNull(cols);
        Assert.NotNull(cols!.DetectionQValue);
        var sample = MergedParquetReader.GetSortedSamples(MergedDataset.Open(MergedGolden), cols.Sample)[0];

        var all = PrecursorDensity.Load(MergedDataset.Open(MergedGolden), cols, sample, qValueCutoff: null);
        var confident = PrecursorDensity.Load(MergedDataset.Open(MergedGolden), cols, sample, qValueCutoff: 0.01);
        var none = PrecursorDensity.Load(MergedDataset.Open(MergedGolden), cols, sample, qValueCutoff: -1);

        Assert.True(confident.Count <= all.Count);
        Assert.Empty(none); // no q-value is below -1, so nothing counts as detected
    }

    [Fact]
    public void Load_UnknownSampleGivesNoPrecursors()
    {
        var cols = PrecursorDensity.Resolve(ParquetTable.ReadColumnNames(MergedGolden).ToHashSet());
        Assert.Empty(PrecursorDensity.Load(MergedDataset.Open(MergedGolden), cols!, "no such replicate'; --", null));
    }

    private static int[] Row(PrecursorDensityMap map, int row)
    {
        var result = new int[map.RtBins];
        for (var j = 0; j < map.RtBins; j++)
            result[j] = map.Counts[row, j];
        return result;
    }

    private static long CountRows(string parquet, string sampleCol, string sample)
    {
        using var conn = new DuckDB.NET.Data.DuckDBConnection("Data Source=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText =
            $"SELECT COUNT(*) FROM read_parquet('{parquet.Replace("'", "''")}') " +
            $"WHERE \"{sampleCol}\" = '{sample.Replace("'", "''")}'";
        return Convert.ToInt64(cmd.ExecuteScalar());
    }
}
