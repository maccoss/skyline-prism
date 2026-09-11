using System;
using System.Linq;
using ScottPlot;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.RawData;
using SkylinePrism.Core.Visualization;
using Xunit;

namespace SkylinePrism.Tests.Visualization;

/// <summary>
/// A plot that showed an empty state and then got real data must render it exactly as it would have
/// rendered without the empty state first.
/// </summary>
/// <remarks>
/// <para>This failed silently for two of the three panes that use an empty state.
/// <see cref="PlotRenderer.DrawEmptyState"/> strips the frame and the grid deliberately - a panel
/// with no data must not show numbered axes to no scale - and <c>Plot.Clear()</c> does NOT put them
/// back, so the state persists on the <c>Plot</c> object across draws. Only the QC Plots pane undid
/// it, through the GUI-side <c>QcPlotChrome.Reset</c>; Spectrum density and MS2 signal called
/// DrawEmptyState and never reset, so their first real plot after an empty one was drawn without
/// its chrome. Fixed in <see cref="PlotRenderer.StyleQcPlot"/> rather than at each call site,
/// because every renderer goes through it and a new one would otherwise have to remember.</para>
///
/// <para><b>Asserted on the rendered bytes, not on a property.</b> ScottPlot's frameless state is
/// not reachable through <c>FrameLineStyle</c> - measured: it leaves both the width and the
/// visibility untouched - so a property assertion here would pass while the plot rendered wrongly.
/// Two renders being byte-identical is the claim that actually matters and it cannot be satisfied by
/// looking at the wrong field.</para>
/// </remarks>
public class EmptyStateRecoveryTests
{
    [Fact]
    public void ARealPlotAfterAnEmptyOneRendersIdentically()
    {
        var clean = Render(empty: false);
        var afterEmpty = Render(empty: true);

        Assert.True(clean.Length > 1000, "expected a real image");
        Assert.Equal(clean.Length, afterEmpty.Length);
        Assert.True(
            clean.SequenceEqual(afterEmpty),
            "a plot drawn after an empty state rendered differently from one drawn clean, so the "
            + "empty state's chrome leaked into it");
    }

    /// <summary>
    /// The grid is the half of the empty state that IS observable, so it is pinned directly as well
    /// - a byte comparison says "something differs", and this says which thing.
    /// </summary>
    [Fact]
    public void TheGridComesBackWhenAPlotIsStyled()
    {
        var plt = new Plot();
        PlotRenderer.DrawEmptyState(plt, "Nothing to plot yet");
        Assert.False(plt.Grid.IsVisible);

        PlotRenderer.StyleQcPlot(plt);
        Assert.True(plt.Grid.IsVisible);
    }

    /// <summary>
    /// And the ordering inside DrawEmptyState still leaves an empty state bare: it styles first and
    /// strips afterwards, so the fix above must not fight it.
    /// </summary>
    [Fact]
    public void AnEmptyStateAfterARealPlotIsStillBare()
    {
        var plt = new Plot();
        PlotRenderer.StyleQcPlot(plt);
        PlotRenderer.DrawEmptyState(plt, "Now there is nothing");

        Assert.False(plt.Grid.IsVisible);
        Assert.Equal("Now there is nothing", plt.Axes.Title.Label.Text);
    }

    private static byte[] Render(bool empty)
    {
        var plt = new Plot();
        if (empty)
            PlotRenderer.DrawEmptyState(plt, "Nothing to plot yet");

        PlotRenderer.DrawIonAccounting(plt, Result(), PlotRenderer.IonLevel.Ms2, "Ions");
        return plt.GetImageBytes(600, 400, ImageFormat.Png);
    }

    private static IonAccountingResult Result()
    {
        var rows = new[] { Row("s1", 10), Row("s2", 20), Row("s3", 30) };
        return new IonAccountingResult(
            "k", "+/-10 ppm", "+/-10 ppm", "167 windows",
            Array.Empty<string>(), 4321, true, rows, Array.Empty<IonCycleRow>());
    }

    private static IonAccountingRow Row(string sample, double percent) =>
        new(sample, "experimental", sample + ".raw", Ms2ReadStatus.Ok, "test", 1, 167,
            Ms1Acquired: 1000, Ms2Acquired: 1000,
            Ms1Assigned: percent * 10, Ms2Assigned: percent * 10,
            0, 60, 1000, 0, 0, 1, Array.Empty<double>(), Array.Empty<double>());
}
