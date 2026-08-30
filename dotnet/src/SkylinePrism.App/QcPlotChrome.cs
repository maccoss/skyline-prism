using ScottPlot;

namespace SkylinePrism.App;

/// <summary>
/// Chrome shared by every QC plot. Its own class because it is pure plot state with no window
/// state, which makes it the one part of the QC tab that can be tested without a dispatcher.
/// </summary>
public static class QcPlotChrome
{
    /// <summary>
    /// Return the shared plot to a blank slate. One <see cref="Plot"/> is reused for every plot kind, and
    /// <c>Clear()</c> removes only the plottables - a tick generator, an axis label, the legend and the
    /// title all survive into whatever is drawn next.
    ///
    /// <para>That is not cosmetic when the kinds disagree about what an axis MEANS. The marker plots
    /// replace both tick generators with categorical ones, so the PCA came back with EV protein names
    /// down its y axis and sample types along its x, under the marker loadings' title - three plots'
    /// worth of chrome over one set of points, each label confidently describing the wrong thing.</para>
    ///
    /// <para>Every draw method is therefore free to set only what it uses, and may assume anything it
    /// does not set is absent.</para>
    /// </summary>
    public static void Reset(Plot plt)
    {
        plt.Axes.Left.TickGenerator = new ScottPlot.TickGenerators.NumericAutomatic();
        plt.Axes.Bottom.TickGenerator = new ScottPlot.TickGenerators.NumericAutomatic();
        plt.Axes.Left.Label.Text = "";
        plt.Axes.Bottom.Label.Text = "";
        plt.Title("");
        plt.HideLegend();
    }
}
