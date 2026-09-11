using System;
using System.Collections.Generic;
using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// Masking a spectrum by the regions peptides claim. Every expected number here is worked out by
/// hand, because the failure mode of interval arithmetic is a plausible wrong total rather than an
/// exception - and this total is the numerator of the fraction the whole feature reports.
/// </summary>
public class ClaimedSignalIndexTests
{
    private static readonly double[] NoList = Array.Empty<double>();

    private static ClaimedRegion Ms2(
        double mzLow, double mzHigh, double rtStart, double rtStop, int window = 0, uint lists = 0) =>
        new(2, window, mzLow, mzHigh, rtStart, rtStop, lists);

    private static ClaimedRegion Ms1(
        double mzLow, double mzHigh, double rtStart, double rtStop, uint lists = 0) =>
        new(1, ClaimedSignalIndex.AnyWindow, mzLow, mzHigh, rtStart, rtStop, lists);

    private static double Claim(
        ClaimedSignalIndex index, double rt, double[] mz, double[] intensity,
        int level = 2, int window = 0, double[]? perList = null) =>
        index.Claimed(level, window, rt, mz, intensity, perList ?? NoList);

    [Fact]
    public void OnlyPeaksInsideAClaimedWindowCount()
    {
        var index = new ClaimedSignalIndex(new[] { Ms2(100, 110, 0, 10) });

        // 105 and 110 are inside (the high edge is inclusive); 99.9 and 110.1 are not.
        var got = Claim(index, rt: 5, mz: new[] { 99.9, 105.0, 110.0, 110.1 },
            intensity: new[] { 1000.0, 7.0, 3.0, 5000.0 });

        Assert.Equal(10.0, got, 9);
    }

    /// <summary>
    /// The whole reason this works on spectra instead of on peak areas. Two peptides claiming
    /// overlapping windows extract the SAME detector counts; summing their areas would credit the
    /// peak twice and can push assigned past acquired. Merging the claims first makes that
    /// impossible rather than correcting for it afterwards.
    /// </summary>
    [Fact]
    public void SignalClaimedByTwoPeptidesIsCountedOnce()
    {
        var shared = new ClaimedSignalIndex(new[]
        {
            Ms2(100, 106, 0, 10),
            Ms2(104, 110, 0, 10),   // overlaps the first from 104 to 106
        });

        // One peak at 105, inside both claims.
        Assert.Equal(42.0, Claim(shared, 5, new[] { 105.0 }, new[] { 42.0 }), 9);

        // And the merged range spans 100-110, so a peak at either end still counts once.
        Assert.Equal(
            42.0 + 8.0,
            Claim(shared, 5, new[] { 101.0, 109.0 }, new[] { 42.0, 8.0 }), 9);
    }

    [Fact]
    public void TouchingClaimsMergeIntoOneRange()
    {
        // Abutting exactly: a peak on the shared edge is one reading, not two.
        var index = new ClaimedSignalIndex(new[] { Ms2(100, 105, 0, 10), Ms2(105, 110, 0, 10) });
        Assert.Equal(9.0, Claim(index, 5, new[] { 105.0 }, new[] { 9.0 }), 9);
    }

    /// <summary>
    /// A claim only holds while Skyline's peak is open. Outside those boundaries the same m/z is
    /// unclaimed signal, which is the difference between "this peptide was measured here" and "some
    /// ion happened to have this mass".
    /// </summary>
    [Fact]
    public void AClaimCountsOnlyWithinItsPeakBoundaries()
    {
        var index = new ClaimedSignalIndex(new[] { Ms2(100, 110, 4.0, 6.0) });
        var mz = new[] { 105.0 };
        var inten = new[] { 100.0 };

        Assert.Equal(0.0, Claim(index, 3.9, mz, inten), 9);    // before the peak
        Assert.Equal(100.0, Claim(index, 4.0, mz, inten), 9);  // boundaries inclusive
        Assert.Equal(100.0, Claim(index, 5.0, mz, inten), 9);
        Assert.Equal(100.0, Claim(index, 6.0, mz, inten), 9);
        Assert.Equal(0.0, Claim(index, 6.1, mz, inten), 9);    // after it
    }

    /// <summary>
    /// At MS2 two fragments of the same mass in DIFFERENT isolation windows are different signal -
    /// they were never co-isolated - so a scan only sees the claims in its own window.
    /// </summary>
    [Fact]
    public void AnMs2ScanSeesOnlyItsOwnIsolationWindow()
    {
        var index = new ClaimedSignalIndex(new[]
        {
            Ms2(100, 110, 0, 10, window: 0),
            Ms2(100, 110, 0, 10, window: 1),
        });
        var mz = new[] { 105.0 };
        var inten = new[] { 50.0 };

        Assert.Equal(50.0, Claim(index, 5, mz, inten, window: 0), 9);
        Assert.Equal(50.0, Claim(index, 5, mz, inten, window: 1), 9);
        // A window nothing was extracted in contributes nothing.
        Assert.Equal(0.0, Claim(index, 5, mz, inten, window: 7), 9);
    }

    /// <summary>
    /// MS1 has no isolation window - a survey scan measures the whole range at once - so every MS1
    /// claim competes with every other, and sharing there is strictly more likely than at MS2.
    /// </summary>
    [Fact]
    public void Ms1ClaimsShareOneLane()
    {
        var index = new ClaimedSignalIndex(new[]
        {
            Ms1(404.0, 404.5, 0, 10),
            Ms1(404.4, 405.0, 0, 10),   // overlapping isotope envelopes
        });

        var got = index.Claimed(
            1, ClaimedSignalIndex.AnyWindow, 5,
            new[] { 404.45 }, new[] { 21.0 }, NoList);
        Assert.Equal(21.0, got, 9);

        // The MS2 lane is a different lane; an MS1 claim must not answer an MS2 scan.
        Assert.Equal(0.0, Claim(index, 5, new[] { 404.45 }, new[] { 21.0 }), 9);
    }

    /// <summary>
    /// A merged range carries the union of its claims' list bits, so a peak two lists both claim is
    /// credited to each in full while counting once toward the total. The lists nest inside the
    /// total and may overlap each other - that is the documented contract of the bar plot.
    /// </summary>
    [Fact]
    public void OverlappingListsEachGetTheSharedSignalInFull()
    {
        var index = new ClaimedSignalIndex(
            new[] { Ms2(100, 106, 0, 10, lists: 0b01), Ms2(104, 110, 0, 10, lists: 0b10) },
            listCount: 2);

        var perList = new double[2];
        var total = Claim(index, 5, new[] { 105.0 }, new[] { 30.0 }, perList: perList);

        Assert.Equal(30.0, total, 9);
        Assert.Equal(30.0, perList[0], 9);
        Assert.Equal(30.0, perList[1], 9);
    }

    [Fact]
    public void PerListSumsAccumulateAcrossScans()
    {
        var index = new ClaimedSignalIndex(
            new[] { Ms2(100, 110, 0, 10, lists: 0b01) }, listCount: 1);

        var perList = new double[1];
        Claim(index, 4, new[] { 105.0 }, new[] { 5.0 }, perList: perList);
        Claim(index, 5, new[] { 105.0 }, new[] { 7.0 }, perList: perList);
        Assert.Equal(12.0, perList[0], 9);
    }

    /// <summary>
    /// The sweep only moves forward, so a claim that opened and closed must not be re-counted, and
    /// one that has not opened yet must not be. Scans within a window arrive in RT order.
    /// </summary>
    [Fact]
    public void TheSweepOpensAndClosesClaimsAsRetentionTimeAdvances()
    {
        var index = new ClaimedSignalIndex(new[]
        {
            Ms2(100, 101, 1.0, 2.0),
            Ms2(200, 201, 5.0, 6.0),
        });
        var mz = new[] { 100.5, 200.5 };
        var inten = new[] { 3.0, 11.0 };

        Assert.Equal(3.0, Claim(index, 1.5, mz, inten), 9);    // only the early claim is open
        Assert.Equal(0.0, Claim(index, 3.0, mz, inten), 9);    // neither
        Assert.Equal(11.0, Claim(index, 5.5, mz, inten), 9);   // only the late one
    }

    [Fact]
    public void NonPositiveAndNonFiniteIntensitiesAreIgnored()
    {
        var index = new ClaimedSignalIndex(new[] { Ms2(100, 110, 0, 10) });
        var got = Claim(index, 5,
            new[] { 101.0, 102.0, 103.0, 104.0 },
            new[] { 5.0, 0.0, -3.0, double.NaN });
        Assert.Equal(5.0, got, 9);
    }

    [Fact]
    public void DegenerateInputsGiveZeroRatherThanThrowing()
    {
        var index = new ClaimedSignalIndex(new[] { Ms2(100, 110, 0, 10) });

        Assert.Equal(0.0, Claim(index, 5, Array.Empty<double>(), Array.Empty<double>()), 9);
        Assert.Equal(0.0, Claim(index, double.NaN, new[] { 105.0 }, new[] { 1.0 }), 9);
        // Mismatched arrays are a caller bug, but not one worth aborting a cohort read for.
        Assert.Equal(0.0, index.Claimed(2, 0, 5, new[] { 105.0 }, new[] { 1.0, 2.0 }, NoList), 9);
        Assert.Equal(0, new ClaimedSignalIndex(Array.Empty<ClaimedRegion>()).RegionCount);
    }

    /// <summary>
    /// Claims with impossible geometry are dropped at construction rather than producing a range
    /// that swallows everything or nothing.
    /// </summary>
    [Fact]
    public void InvertedRegionsAreDropped()
    {
        var index = new ClaimedSignalIndex(new[]
        {
            new ClaimedRegion(2, 0, 110, 100, 0, 10, 0),   // m/z inverted
            new ClaimedRegion(2, 0, 100, 110, 10, 0, 0),   // RT inverted
        });
        Assert.Equal(0, index.RegionCount);
        Assert.Equal(0.0, Claim(index, 5, new[] { 105.0 }, new[] { 1.0 }), 9);
    }
}
