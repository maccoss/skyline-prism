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
    /// The two quantities are DIFFERENT NUMBERS, not two units for one. A scan's intensity is a
    /// rate; the ion count multiplies it by the injection time and the summed TIC does not. Where
    /// the injection times vary, the assigned fractions differ too - and neither is wrong.
    /// </summary>
    [Fact]
    public void SignalIsNotTheSameQuantityAsIons()
    {
        // Injection time varies across the run, so weighting cannot cancel: an ion fraction and a
        // signal fraction built from the same scans genuinely disagree.
        var rows = new[]
        {
            RowWithSignal("s1", ms2Acquired: 1000, ms2Assigned: 100, ms2Signal: 1000, ms2SignalAssigned: 250),
            RowWithSignal("s2", ms2Acquired: 2000, ms2Assigned: 200, ms2Signal: 1000, ms2SignalAssigned: 250),
        };
        var result = Result(rows);

        var ions = new Plot();
        PlotRenderer.DrawIonAccounting(
            ions, result, PlotRenderer.IonLevel.Ms2, "t", 1.0, PlotRenderer.IonQuantity.Ions);
        var signal = new Plot();
        PlotRenderer.DrawIonAccounting(
            signal, result, PlotRenderer.IonLevel.Ms2, "t", 1.0, PlotRenderer.IonQuantity.Signal);

        // 10% of the ions, 25% of the TIC - the titles must not agree. Asserted without the percent
        // SIGN: P1 renders it as "10.0 %" under the invariant culture and "10.0%" under en-US, so a
        // test that included it passed on the Windows and macOS runners and failed on Linux.
        Assert.Contains("10.0", ions.Axes.Title.Label.Text);
        Assert.Contains("25.0", signal.Axes.Title.Label.Text);
        Assert.NotEqual(ions.Axes.Title.Label.Text, signal.Axes.Title.Label.Text);
    }

    /// <summary>
    /// The axis has to name the quantity drawn. Both are totals of the same scans and they scale
    /// alike, so nothing else on the plot distinguishes them - mislabelling one as the other would
    /// present a sum of rates as a count of ions, which is the error this whole feature is built
    /// around not making.
    /// </summary>
    [Fact]
    public void TheAxisNamesTheQuantityDrawn()
    {
        var result = Result(RowWithSignal("s1", 1000, 100, 1000, 250));

        var ions = new Plot();
        PlotRenderer.DrawIonAccounting(
            ions, result, PlotRenderer.IonLevel.Ms2, null, 1.0, PlotRenderer.IonQuantity.Ions);
        Assert.Contains("ions", ions.Axes.Left.Label.Text);
        Assert.DoesNotContain("TIC", ions.Axes.Left.Label.Text);

        var signal = new Plot();
        PlotRenderer.DrawIonAccounting(
            signal, result, PlotRenderer.IonLevel.Ms2, null, 1.0, PlotRenderer.IonQuantity.Signal);
        Assert.Contains("TIC", signal.Axes.Left.Label.Text);
    }

    /// <summary>
    /// Across the gradient the two quantities bin independently, and the signal bins are the TIC -
    /// not the ion totals relabelled.
    /// </summary>
    [Fact]
    public void BinningKeepsTheTwoQuantitiesApart()
    {
        var cycles = new[]
        {
            new IonCycleRow("s1", 0, 0, 0.5, 1, 10, 100, 200, 10, 20, 30, 1000, 2000, 100, 200, 300),
            new IonCycleRow("s1", 1, 0.5, 1.0, 1, 10, 100, 200, 10, 20, 30, 1000, 2000, 100, 200, 300),
        };

        var ionBins = PlotRenderer.BinCycles(
            cycles, PlotRenderer.IonLevel.Ms2, 10.0, PlotRenderer.IonQuantity.Ions);
        var signalBins = PlotRenderer.BinCycles(
            cycles, PlotRenderer.IonLevel.Ms2, 10.0, PlotRenderer.IonQuantity.Signal);

        Assert.Equal(400, Assert.Single(ionBins).Acquired);
        Assert.Equal(4000, Assert.Single(signalBins).Acquired);
        Assert.Equal(40, ionBins[0].Assigned);
        Assert.Equal(400, signalBins[0].Assigned);
        Assert.Equal(60, ionBins[0].Explained);
        Assert.Equal(600, signalBins[0].Explained);
    }

    /// <summary>
    /// A cache measured before the TIC was recorded has no signal, and zeros are not it. The fraction
    /// reads as NaN - "not measured" - rather than as a replicate that acquired nothing.
    /// </summary>
    [Fact]
    public void ACacheWithoutSignalReportsNotMeasuredRatherThanZero()
    {
        var row = Row("s1", 10);

        Assert.False(row.HasSignal);
        Assert.True(double.IsNaN(row.Ms1SignalFraction));
        Assert.True(double.IsNaN(row.Ms2SignalFraction));
        Assert.True(double.IsNaN(row.Ms2SignalExplainedFraction));
    }

    private static IonAccountingRow RowWithSignal(
        string sample, double ms2Acquired, double ms2Assigned,
        double ms2Signal, double ms2SignalAssigned) =>
        new(sample, "experimental", $"{sample}.raw", Ms2ReadStatus.Ok, "pwiz-sharp",
            10, 100, ms2Acquired, ms2Acquired, ms2Assigned, ms2Assigned,
            ms2Assigned * 1.5, true, 0, 60, 5, 0, 0, 10,
            Array.Empty<double>(), Array.Empty<double>(), null,
            ms2Signal, ms2Signal, ms2SignalAssigned, ms2SignalAssigned,
            ms2SignalAssigned * 1.5, true);

    /// <summary>
    /// Drawing again on the same plot REPLACES what was there. The GUI keeps one Plot for the whole
    /// pane and redraws it every time the view, level, replicate or bin width changes, so anything
    /// that accumulates does so once per interaction.
    ///
    /// <para>Reported from a real run: a legend carrying the bar chart's series, then two more
    /// copies of the gradient profile's, all at once. ScottPlot's <c>Add</c> methods append, and the
    /// legend entries ride on the plottables - so the earlier renders' data was still on the plot
    /// underneath, not merely named in the legend.</para>
    /// </summary>
    [Fact]
    public void RedrawingReplacesTheLastRenderRatherThanStackingOnIt()
    {
        var result = Result(Row("s1", 40), Row("s2", 30), Row("s3", 35));
        var cycles = Cycles("s1", count: 240);
        const PlotRenderer.IonLevel ms2 = PlotRenderer.IonLevel.Ms2;

        // One plot, every view in turn, twice round - which is what a user does in a few clicks.
        var plt = new Plot();
        PlotRenderer.DrawIonAccounting(plt, result, ms2, "bars");
        var afterOneBarDraw = plt.GetPlottables().Count();
        var legendAfterOneBarDraw = LegendEntries(plt);

        PlotRenderer.DrawIonProfile(plt, cycles, ms2, 1.0, "profile");
        PlotRenderer.DrawIonFractionProfile(plt, cycles, ms2, 1.0, "share");
        PlotRenderer.DrawIonAccounting(plt, result, ms2, "bars");

        Assert.Equal(afterOneBarDraw, plt.GetPlottables().Count());
        Assert.Equal(legendAfterOneBarDraw, LegendEntries(plt));

        // And each of the other two views, reached from a different one, is its own render only.
        var fresh = new Plot();
        PlotRenderer.DrawIonProfile(fresh, cycles, ms2, 1.0, "profile");
        var profileOnly = plt.GetPlottables().Count();
        PlotRenderer.DrawIonProfile(plt, cycles, ms2, 1.0, "profile");
        Assert.Equal(fresh.GetPlottables().Count(), plt.GetPlottables().Count());
        Assert.NotEqual(0, profileOnly);

        var share = new Plot();
        PlotRenderer.DrawIonFractionProfile(share, cycles, ms2, 1.0, "share");
        PlotRenderer.DrawIonFractionProfile(plt, cycles, ms2, 1.0, "share");
        Assert.Equal(share.GetPlottables().Count(), plt.GetPlottables().Count());
    }

    /// <summary>
    /// An empty state drawn over a real render leaves nothing of it behind - a message with the
    /// previous plot's bars still under it is worse than either on its own.
    /// </summary>
    [Fact]
    public void AnEmptyStateReplacesWhateverWasDrawnBefore()
    {
        var plt = new Plot();
        PlotRenderer.DrawIonAccounting(
            plt, Result(Row("s1", 40), Row("s2", 30)), PlotRenderer.IonLevel.Ms2, "bars");
        Assert.NotEmpty(plt.GetPlottables());

        PlotRenderer.DrawIonAccounting(plt, Result(), PlotRenderer.IonLevel.Ms2, "nothing");

        Assert.Empty(plt.GetPlottables());
        Assert.Equal(0, LegendEntries(plt));
    }

    /// <summary>Legend entries come from the plottables, so this counts what a reader would see.</summary>
    private static int LegendEntries(Plot plt) =>
        plt.GetPlottables()
            .SelectMany(p => p.LegendItems)
            .Count(item => !string.IsNullOrEmpty(item.LabelText));

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
