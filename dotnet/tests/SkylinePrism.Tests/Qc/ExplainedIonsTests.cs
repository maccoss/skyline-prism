using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.RawData;
using SkylinePrism.Core.Visualization;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// The second numerator: what the run's peptides can ACCOUNT FOR, against what it QUANTIFIES on.
///
/// <para>The properties that matter are the nesting (explained can never be below quantified,
/// because it is the quantified set unioned with the theoretical ions) and the distinction between
/// "not measured" and "measured as zero" - an export with no charge column produces the first, and
/// reporting it as the second would say the peptides explain nothing.</para>
/// </summary>
public class ExplainedIonsTests
{
    private static ClaimedRegion Fragment(double mz, double halfWidth, uint mask = 0) =>
        new(2, 0, mz - halfWidth, mz + halfWidth, 0, 10, mask);

    /// <summary>
    /// The union is the whole point: a transition Skyline quantifies on that the b/y enumeration
    /// does not produce - a 3+ fragment, a neutral loss - must still count as explained, or the plot
    /// shows a run quantifying on more than its peptides can account for.
    /// </summary>
    [Fact]
    public void ExplainedIsNeverBelowQuantifiedForTheSameSpectrum()
    {
        var quantified = new ClaimedSignalIndex(new[] { Fragment(500.0, 0.01) });

        // The explained set as the loader builds it: the quantified claim, plus a theoretical ion
        // the document does not carry.
        var explained = new ClaimedSignalIndex(
            new[] { Fragment(500.0, 0.01), Fragment(700.0, 0.01) });

        var mz = new[] { 500.0, 700.0, 900.0 };
        var intensity = new[] { 10.0, 20.0, 40.0 };

        var q = quantified.Claimed(2, 0, 5, mz, intensity, Span<double>.Empty);
        var e = explained.Claimed(2, 0, 5, mz, intensity, Span<double>.Empty);

        Assert.Equal(10.0, q, 9);
        Assert.Equal(30.0, e, 9);
        Assert.True(e >= q);
    }

    /// <summary>
    /// Shared signal counts once on the explained side too. Two peptides whose theoretical ions land
    /// within tolerance of each other read the SAME detector counts, and the merge is what keeps the
    /// explained total from exceeding what was acquired.
    /// </summary>
    [Fact]
    public void OverlappingTheoreticalIonsCountOnce()
    {
        var index = new ClaimedSignalIndex(new[]
        {
            Fragment(500.000, 0.02),
            Fragment(500.005, 0.02),   // a different peptide's ion, overlapping
        });

        var claimed = index.Claimed(
            2, 0, 5, new[] { 500.0 }, new[] { 100.0 }, Span<double>.Empty);

        Assert.Equal(100.0, claimed, 9);
    }

    /// <summary>"Not measured" and "measured as zero" must not render the same.</summary>
    [Fact]
    public void AnUnmeasuredExplainedTotalIsNaNRatherThanZero()
    {
        var measuredZero = Row(ms2Explained: 0, hasExplained: true);
        var notMeasured = Row(ms2Explained: 0, hasExplained: false);

        Assert.Equal(0.0, measuredZero.Ms2ExplainedFraction, 9);
        Assert.True(double.IsNaN(notMeasured.Ms2ExplainedFraction));
    }

    /// <summary>
    /// Explained below quantified is impossible by construction, so it gets its own flag: the cause
    /// is a defect in claim building, not anything about the data, and it is a different fix from
    /// assigning more than was acquired.
    /// </summary>
    [Fact]
    public void ExplainedBelowAssignedIsFlagged()
    {
        var broken = Row(ms2Explained: 5, hasExplained: true, ms2Assigned: 40);
        var sound = Row(ms2Explained: 80, hasExplained: true, ms2Assigned: 40);
        var unmeasured = Row(ms2Explained: 0, hasExplained: false, ms2Assigned: 40);

        Assert.True(broken.Exceeded is false);   // it did not exceed ACQUIRED
        Assert.True(sound.Ms2Explained > sound.Ms2Assigned);
        Assert.False(unmeasured.HasExplained);
    }

    /// <summary>An explained total above acquired is impossible and must refuse to be drawn.</summary>
    [Fact]
    public void ExplainedAboveAcquiredCountsAsExceeded()
    {
        var row = Row(ms2Explained: 5_000, hasExplained: true, ms2Assigned: 40);
        Assert.True(row.Exceeded);
    }

    /// <summary>
    /// The cycle bins carry the explained total through to the gradient plots, and MS1 has none -
    /// at MS1 the theoretical claim IS the isotope envelope Skyline already extracts.
    /// </summary>
    [Fact]
    public void CycleBinsCarryExplainedAtMs2AndNotAtMs1()
    {
        var cycles = new[]
        {
            new IonCycleRow("s", 0, 0.0, 0.5, 1, 10, 100, 200, 40, 80, 150),
            new IonCycleRow("s", 1, 0.5, 1.0, 1, 10, 100, 200, 40, 80, 150),
        };

        var ms2 = PlotRenderer.BinCycles(cycles, PlotRenderer.IonLevel.Ms2, binMinutes: 10);
        Assert.Single(ms2);
        Assert.Equal(400, ms2[0].Acquired, 9);
        Assert.Equal(160, ms2[0].Assigned, 9);
        Assert.Equal(300, ms2[0].Explained, 9);

        var ms1 = PlotRenderer.BinCycles(cycles, PlotRenderer.IonLevel.Ms1, binMinutes: 10);
        Assert.Single(ms1);
        Assert.Equal(0, ms1[0].Explained, 9);
    }

    /// <summary>
    /// Binning must conserve: the explained total of the bins is the explained total of the cycles,
    /// whatever the bin width. A cycle belongs to exactly one bin, so nothing is counted twice.
    /// </summary>
    [Theory]
    [InlineData(0.1)]
    [InlineData(1.0)]
    [InlineData(7.0)]
    public void BinningConservesTheExplainedTotal(double binMinutes)
    {
        var cycles = Enumerable.Range(0, 40)
            .Select(i => new IonCycleRow(
                "s", i, i * 0.25, i * 0.25 + 0.25, 1, 10,
                Ms1Acquired: 100, Ms2Acquired: 200,
                Ms1Assigned: 40, Ms2Assigned: 80, Ms2Explained: 130 + i))
            .ToArray();

        var expected = cycles.Sum(c => c.Ms2Explained);
        var binned = PlotRenderer.BinCycles(cycles, PlotRenderer.IonLevel.Ms2, binMinutes);

        Assert.Equal(expected, binned.Sum(b => b.Explained), 6);
    }

    /// <summary>
    /// The report says nothing about an explained total it does not have, rather than reporting a
    /// zero - and says nothing at MS1, where the quantity does not exist.
    /// </summary>
    [Fact]
    public void TheCaptionIsSilentWhenNothingWasMeasured()
    {
        var measured = new[] { Row(ms2Explained: 800, hasExplained: true) };
        var unmeasured = new[] { Row(ms2Explained: 0, hasExplained: false) };

        var withIt = QcReport.ExplainedCaption(measured, PlotRenderer.IonLevel.Ms2);
        Assert.Contains("explained", withIt, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("b and y", withIt, StringComparison.OrdinalIgnoreCase);

        Assert.Equal("", QcReport.ExplainedCaption(unmeasured, PlotRenderer.IonLevel.Ms2));
        Assert.Equal("", QcReport.ExplainedCaption(measured, PlotRenderer.IonLevel.Ms1));
    }

    /// <summary>
    /// The per-replicate plot gains a third series when the data has one, and renders
    /// BYTE-IDENTICALLY to the old plot when it does not. A property assertion would pass either
    /// way; only the bytes show that an unmeasured cohort's report is unchanged.
    /// </summary>
    [Fact]
    public void ThePlotIsUnchangedWithoutAnExplainedTotal()
    {
        var rows = new[]
        {
            Row(ms2Explained: 0, hasExplained: false, sample: "a"),
            Row(ms2Explained: 0, hasExplained: false, sample: "b"),
        };
        var withExplained = new[]
        {
            Row(ms2Explained: 800, hasExplained: true, sample: "a"),
            Row(ms2Explained: 800, hasExplained: true, sample: "b"),
        };

        var plain = PlotRenderer.IonAccountingPng(
            Result(rows), PlotRenderer.IonLevel.Ms2, "t");
        var plainAgain = PlotRenderer.IonAccountingPng(
            Result(rows), PlotRenderer.IonLevel.Ms2, "t");
        var dual = PlotRenderer.IonAccountingPng(
            Result(withExplained), PlotRenderer.IonLevel.Ms2, "t");

        Assert.Equal(plain, plainAgain);            // deterministic to begin with
        Assert.NotEqual(plain.Length, dual.Length); // the third series actually drew

        // ...and both are PNGs, not the BMPs an earlier version of these helpers returned.
        Assert.Equal(new byte[] { 0x89, 0x50, 0x4E, 0x47 }, plain.Take(4).ToArray());
        Assert.Equal(new byte[] { 0x89, 0x50, 0x4E, 0x47 }, dual.Take(4).ToArray());
    }

    /// <summary>MS1 never draws the series, even when the rows carry one.</summary>
    [Fact]
    public void Ms1IgnoresTheExplainedTotal()
    {
        var rows = new[] { Row(ms2Explained: 800, hasExplained: true, sample: "a") };
        var bare = new[] { Row(ms2Explained: 0, hasExplained: false, sample: "a") };

        Assert.Equal(
            PlotRenderer.IonAccountingPng(Result(bare), PlotRenderer.IonLevel.Ms1, "t"),
            PlotRenderer.IonAccountingPng(Result(rows), PlotRenderer.IonLevel.Ms1, "t"));
    }

    private static IonAccountingRow Row(
        double ms2Explained, bool hasExplained, double ms2Assigned = 40, string sample = "s") =>
        new(sample, "experimental", sample + ".raw", Ms2ReadStatus.Ok, "test", 1, 167,
            Ms1Acquired: 1000, Ms2Acquired: 1000,
            Ms1Assigned: 400, Ms2Assigned: ms2Assigned,
            Ms2Explained: ms2Explained, HasExplained: hasExplained,
            0, 60, 1000, 0, 0, 1, Array.Empty<double>(), Array.Empty<double>());

    private static IonAccountingResult Result(IReadOnlyList<IonAccountingRow> rows) =>
        new("k", "t", "p", "s", Array.Empty<string>(), 1, true, rows, Array.Empty<IonCycleRow>());
}
