using ScottPlot;
using SkylinePrism.App;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The QC tab draws every plot kind onto ONE reused <see cref="Plot"/>, and <c>Clear()</c> removes only
/// the plottables. Everything else - tick generators, axis labels, the title, the legend - survives into
/// whatever is drawn next.
///
/// <para>That shipped, and it was not subtle: switching to PCA after viewing the marker plots left EV
/// protein names down the y axis, sample-type names along the x, and the marker loadings' title above a
/// scatter of samples. Every label was confidently describing a different plot.</para>
///
/// <para>These are real assertions rather than source checks because plot chrome is plain ScottPlot
/// state with no window or dispatcher behind it.</para>
/// </summary>
public class QcPlotChromeTests
{
    /// <summary>A plot left in the state the marker plots leave it in.</summary>
    private static Plot Dirtied()
    {
        var plt = new Plot();
        plt.Axes.Left.TickGenerator = new ScottPlot.TickGenerators.NumericManual(
            new[] { new Tick(0, "CD9"), new Tick(1, "CD63") });
        plt.Axes.Bottom.TickGenerator = new ScottPlot.TickGenerators.NumericManual(
            new[] { new Tick(0, "Quality Control"), new Tick(1, "Unknown") });
        plt.YLabel("Marker score (PC1)");
        plt.XLabel("PC1 loading");
        plt.Title("18 markers, 4 opposing");
        plt.ShowLegend();
        return plt;
    }

    [Fact]
    public void ResetRestoresAutomaticTicks()
    {
        var plt = Dirtied();

        QcPlotChrome.Reset(plt);

        // Categorical ticks from another plot kind are the worst of these: they put a protein name
        // beside a principal-component coordinate and read as though it meant something.
        Assert.IsType<ScottPlot.TickGenerators.NumericAutomatic>(plt.Axes.Left.TickGenerator);
        Assert.IsType<ScottPlot.TickGenerators.NumericAutomatic>(plt.Axes.Bottom.TickGenerator);
    }

    [Fact]
    public void ResetClearsBothAxisLabelsAndTheTitle()
    {
        var plt = Dirtied();

        QcPlotChrome.Reset(plt);

        Assert.True(string.IsNullOrEmpty(plt.Axes.Left.Label.Text));
        Assert.True(string.IsNullOrEmpty(plt.Axes.Bottom.Label.Text));
        Assert.True(string.IsNullOrEmpty(plt.Axes.Title.Label.Text));
    }

    /// <summary>
    /// A legend belongs to the plot that created it. Left up, it labels colors that the next plot uses
    /// for something else entirely.
    /// </summary>
    [Fact]
    public void ResetHidesTheLegend()
    {
        var plt = Dirtied();

        QcPlotChrome.Reset(plt);

        Assert.False(plt.Legend.IsVisible);
    }

    /// <summary>Idempotent, and safe on a plot that was never dirtied - it runs before every render.</summary>
    [Fact]
    public void ResetIsSafeOnAFreshPlot()
    {
        var plt = new Plot();

        QcPlotChrome.Reset(plt);
        QcPlotChrome.Reset(plt);

        Assert.True(string.IsNullOrEmpty(plt.Axes.Title.Label.Text));
        Assert.IsType<ScottPlot.TickGenerators.NumericAutomatic>(plt.Axes.Left.TickGenerator);
    }
}
