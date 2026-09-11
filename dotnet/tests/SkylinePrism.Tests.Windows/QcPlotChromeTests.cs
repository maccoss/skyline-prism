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

    /// <summary>
    /// The two marker-score layout rules, pinned at their boundaries. Both are pure functions of a group
    /// count, and both change what the user sees: one refuses to draw, the other moves the per-group n
    /// from the legend onto the tick labels.
    /// </summary>
    [Theory]
    [InlineData(1, false)]
    [InlineData(12, false)]   // the cap itself still draws
    [InlineData(13, true)]    // one past it refuses
    [InlineData(45, true)]    // grouping by subject, the case that prompted the rule
    public void TooManyGroupsRefusesOnlyPastTheCap(int groups, bool refused)
        => Assert.Equal(refused, QcPlotChrome.TooManyMarkerScoreGroups(groups));

    [Theory]
    [InlineData(1, true)]
    [InlineData(6, true)]     // the threshold itself keeps the legend
    [InlineData(7, false)]    // one past it moves the counts to the ticks
    [InlineData(12, false)]
    public void TheLegendAppearsOnlyForAFewGroups(int groups, bool legend)
        => Assert.Equal(legend, QcPlotChrome.ShowMarkerScoreLegend(groups));

    /// <summary>
    /// The legend threshold must sit below the cap, or the band between them - where the counts move
    /// onto the tick labels - would not exist and the counts would simply be lost.
    /// </summary>
    [Fact]
    public void TheLegendThresholdIsBelowTheGroupCap()
        => Assert.True(QcPlotChrome.MaxMarkerScoreLegendGroups < QcPlotChrome.MaxMarkerScoreGroups);

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

    /// <summary>
    /// The hover hit test, which the PCA and marker-score plots share. Pure pixel arithmetic, so it is
    /// testable where the handler around it - which needs a window, a dispatcher and a laid-out plot to
    /// project data coordinates into pixels - is not.
    /// </summary>
    [Fact]
    public void NearestPointFindsThePointUnderTheCursor()
    {
        var points = new[] { new Pixel(10, 10), new Pixel(100, 100), new Pixel(300, 50) };

        Assert.Equal(0, QcPlotChrome.NearestPoint(points, new Pixel(12, 8)));
        Assert.Equal(1, QcPlotChrome.NearestPoint(points, new Pixel(104, 96)));
        Assert.Equal(2, QcPlotChrome.NearestPoint(points, new Pixel(300, 50)));
    }

    /// <summary>
    /// The case that hides the readout again when the cursor leaves a point. Returning the nearest
    /// point regardless would leave a label pinned to the plot for as long as the mouse stayed inside
    /// it - so this is the half of the contract worth pinning.
    /// </summary>
    [Fact]
    public void NearestPointReturnsNothingOverEmptySpace()
    {
        var points = new[] { new Pixel(10, 10) };

        Assert.Equal(-1, QcPlotChrome.NearestPoint(points, new Pixel(400, 400)));

        // Just outside the radius, on the diagonal, and just inside it on one axis.
        Assert.Equal(-1, QcPlotChrome.NearestPoint(
            points, new Pixel(10 + QcPlotChrome.HoverRadiusPx + 1, 10)));
        Assert.Equal(0, QcPlotChrome.NearestPoint(
            points, new Pixel(10 + QcPlotChrome.HoverRadiusPx - 1, 10)));
    }

    /// <summary>An empty plot must not report a hit; the handler runs on every mouse move.</summary>
    [Fact]
    public void NearestPointHandlesNoPoints()
    {
        Assert.Equal(-1, QcPlotChrome.NearestPoint(System.Array.Empty<Pixel>(), new Pixel(0, 0)));
    }
}
