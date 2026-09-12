using System;
using System.Collections.Generic;
using System.Linq;
using ScottPlot;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.RawData;
using SkylinePrism.Core.Visualization;
using Xunit;

namespace SkylinePrism.Tests.Visualization;

/// <summary>
/// The ion-accounting plots. Rendering is checked by smoke test like the others, but the two things
/// that can be WRONG rather than merely ugly get their own assertions: whether a fraction is shown
/// at all, and whether cycles are binned without losing or duplicating signal.
/// </summary>
public class IonPlotTests
{
    private static void AssertPng(byte[] png)
    {
        Assert.NotNull(png);
        Assert.True(png.Length > 1000, $"expected a real PNG, got {png.Length} bytes");
        // PNG magic, so a zero-filled buffer cannot pass.
        Assert.Equal(new byte[] { 0x89, 0x50, 0x4E, 0x47 }, png.Take(4).ToArray());
    }

    [Theory]
    [InlineData(PlotRenderer.IonLevel.Ms1)]
    [InlineData(PlotRenderer.IonLevel.Ms2)]
    public void EveryViewRenders(PlotRenderer.IonLevel level)
    {
        var result = Result(Row("s1", 40), Row("s2", 30), Row("s3", 35));
        AssertPng(PlotRenderer.IonAccountingPng(result, level, "Ions per replicate"));

        var cycles = Cycles("s1", count: 240);
        AssertPng(PlotRenderer.IonProfilePng(cycles, level, binMinutes: 1.0, title: "Across the gradient"));
        AssertPng(PlotRenderer.IonFractionProfilePng(cycles, level, binMinutes: 1.0, title: "Share"));
    }

    /// <summary>
    /// Degenerate input draws the empty state rather than throwing or producing an axis to no scale.
    /// </summary>
    [Fact]
    public void NothingToPlotDrawsAnEmptyState()
    {
        var empty = Result();
        AssertPng(PlotRenderer.IonAccountingPng(empty, PlotRenderer.IonLevel.Ms2));
        AssertPng(PlotRenderer.IonProfilePng(
            Array.Empty<IonCycleRow>(), PlotRenderer.IonLevel.Ms2));
        AssertPng(PlotRenderer.IonFractionProfilePng(
            Array.Empty<IonCycleRow>(), PlotRenderer.IonLevel.Ms2));

        // Cycles that acquired nothing have no fraction to take, which is not the same as a
        // fraction of zero - plotting zero would read as "nothing was assigned here".
        var acquiredNothing = new[] { new IonCycleRow("s1", 0, 0, 0.5, 1, 167, 0, 0, 0, 0) };
        AssertPng(PlotRenderer.IonFractionProfilePng(
            acquiredNothing, PlotRenderer.IonLevel.Ms2));
    }

    /// <summary>
    /// The median fraction goes in the title, so a reader gets the headline number without reading
    /// bars off an axis.
    /// </summary>
    [Fact]
    public void TheTitleCarriesTheMedianFraction()
    {
        var plt = new Plot();
        PlotRenderer.DrawIonAccounting(
            plt, Result(Row("s1", 10), Row("s2", 20), Row("s3", 30)),
            PlotRenderer.IonLevel.Ms2, "Ions per replicate");

        // 20% is the median of 10/20/30.
        Assert.Contains("20.0", plt.Axes.Title.Label.Text);
        Assert.Contains("MS2", plt.Axes.Title.Label.Text);
    }

    /// <summary>
    /// A fraction over 1 is impossible, so it is a defect and must be NAMED rather than shown or
    /// silently dropped. This is the assertion that keeps the previous version of this feature's
    /// failure - a plausible-looking fraction that was 7x too large - from being possible to repeat
    /// quietly.
    /// </summary>
    [Fact]
    public void AnImpossibleFractionIsNamedNotDrawn()
    {
        var broken = Result(
            Row("s1", 20),
            new IonAccountingRow(
                "s2", "experimental", "s2.raw", Ms2ReadStatus.Ok, "test", 1, 167,
                Ms1Acquired: 100, Ms2Acquired: 100, Ms1Assigned: 100, Ms2Assigned: 150,
                Ms2Explained: 0, HasExplained: false,
                0, 60, 1000, 0, 0, 1, Array.Empty<double>(), Array.Empty<double>()));

        var plt = new Plot();
        PlotRenderer.DrawIonAccounting(plt, broken, PlotRenderer.IonLevel.Ms2, "Ions per replicate");

        var title = plt.Axes.Title.Label.Text;
        Assert.Contains("exceeds acquired", title);
        Assert.DoesNotContain("median", title);
    }

    /// <summary>
    /// Binning must conserve the totals: every cycle lands in exactly one bin, so the summed bins
    /// equal the summed cycles exactly. A cycle counted in two bins would inflate the trace, and one
    /// counted in none would leave a gap that reads as a stretch the instrument did not acquire.
    /// </summary>
    [Fact]
    public void BinningConservesTheTotals()
    {
        var cycles = Cycles("s1", count: 97);      // deliberately not a multiple of the bin
        var bins = PlotRenderer.BinCycles(cycles, PlotRenderer.IonLevel.Ms2, binMinutes: 1.0);

        Assert.NotEmpty(bins);
        Assert.Equal(cycles.Sum(c => c.Ms2Acquired), bins.Sum(b => b.Acquired), 6);
        Assert.Equal(cycles.Sum(c => c.Ms2Assigned), bins.Sum(b => b.Assigned), 6);

        // 97 cycles a quarter-minute apart span 24 minutes, so a 1-minute bin gives 25 bins.
        Assert.Equal(25, bins.Count);
        // Ascending, and each bin centered in its own minute.
        Assert.Equal(bins.OrderBy(b => b.RtMin).Select(b => b.RtMin), bins.Select(b => b.RtMin));
    }

    /// <summary>
    /// The MS1 and MS2 halves are separate columns, so a level switch must change the numbers and
    /// not merely the labels.
    /// </summary>
    [Fact]
    public void BinningReadsTheLevelItWasAskedFor()
    {
        var cycles = Cycles("s1", count: 20);
        var ms1 = PlotRenderer.BinCycles(cycles, PlotRenderer.IonLevel.Ms1, 1.0);
        var ms2 = PlotRenderer.BinCycles(cycles, PlotRenderer.IonLevel.Ms2, 1.0);

        Assert.Equal(cycles.Sum(c => c.Ms1Acquired), ms1.Sum(b => b.Acquired), 6);
        Assert.Equal(cycles.Sum(c => c.Ms2Acquired), ms2.Sum(b => b.Acquired), 6);
        Assert.NotEqual(ms1.Sum(b => b.Acquired), ms2.Sum(b => b.Acquired), 6);
    }

    /// <summary>
    /// Cycles with no usable retention time cannot be placed, and a non-positive bin width is a
    /// caller bug - neither should throw in a plot renderer.
    /// </summary>
    [Fact]
    public void DegenerateBinningInputsDoNotThrow()
    {
        var noRt = new[] { new IonCycleRow("s1", 0, double.NaN, double.NaN, 1, 167, 5, 5, 1, 1) };
        Assert.Empty(PlotRenderer.BinCycles(noRt, PlotRenderer.IonLevel.Ms2, 1.0));

        var cycles = Cycles("s1", count: 8);
        Assert.NotEmpty(PlotRenderer.BinCycles(cycles, PlotRenderer.IonLevel.Ms2, binMinutes: 0));
        Assert.Empty(PlotRenderer.BinCycles(
            Array.Empty<IonCycleRow>(), PlotRenderer.IonLevel.Ms2, 1.0));
    }

    /// <summary>A replicate whose MS2 assigned share is <paramref name="percent"/>.</summary>
    private static IonAccountingRow Row(string sample, double percent) =>
        new(sample, "experimental", sample + ".raw", Ms2ReadStatus.Ok, "test", 1, 167,
            Ms1Acquired: 1000, Ms2Acquired: 1000,
            Ms1Assigned: percent * 10, Ms2Assigned: percent * 10,
            // No explained total by default, so the existing assertions keep describing exactly
            // what they described before it existed. The explained path has its own rows.
            Ms2Explained: 0, HasExplained: false,
            0, 60, 1000, 0, 0, 1, Array.Empty<double>(), Array.Empty<double>());

    private static IonAccountingResult Result(params IonAccountingRow[] rows) =>
        new("k", "+/-10 ppm (centroided)", "+/-10 ppm (centroided)", "167 windows",
            Array.Empty<string>(), 4321, true, rows, Array.Empty<IonCycleRow>());

    /// <summary>
    /// A run of cycles with a bell-shaped acquired trace, so the profile has a real shape to draw
    /// rather than a flat line.
    /// </summary>
    private static IReadOnlyList<IonCycleRow> Cycles(string sample, int count)
    {
        var rows = new List<IonCycleRow>(count);
        for (var i = 0; i < count; i++)
        {
            var rt = i * 0.25;
            var envelope = Math.Exp(-Math.Pow((i - count / 2.0) / (count / 6.0), 2));
            var acquired = 1e6 * (0.1 + envelope);
            rows.Add(new IonCycleRow(
                sample, i, rt, rt + 0.25, 1, 167,
                acquired * 0.3, acquired, acquired * 0.12, acquired * 0.034));
        }
        return rows;
    }
}
