using ScottPlot;

namespace SkylinePrism.App;

/// <summary>
/// Chrome shared by every QC plot. Its own class because it is pure plot state with no window
/// state, which makes it the one part of the QC tab that can be tested without a dispatcher.
/// </summary>
internal static class QcPlotChrome
{
    /// <summary>
    /// Above this many Group-by values the marker-score strip plot stops answering its question - "do
    /// the study's groups separate?" cannot be read off 45 columns of two. Sized to fit a study design
    /// (arms, timepoints, batches) and to exclude an identifier column such as subject.
    /// </summary>
    public const int MaxMarkerScoreGroups = 12;

    /// <summary>
    /// Above this many groups the legend costs more space than it earns, because the x ticks already
    /// name every group. Deliberately below <see cref="MaxMarkerScoreGroups"/>: between the two the
    /// per-group counts move onto the tick labels rather than disappearing with the legend.
    /// </summary>
    public const int MaxMarkerScoreLegendGroups = 6;

    /// <summary>Whether to refuse the strip plot and name a better column instead.</summary>
    public static bool TooManyMarkerScoreGroups(int groups) => groups > MaxMarkerScoreGroups;

    /// <summary>Whether a legend earns its space, or the counts should go on the tick labels.</summary>
    public static bool ShowMarkerScoreLegend(int groups) => groups <= MaxMarkerScoreLegendGroups;

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
        // Alignment, not just visibility: ShowLegend(Alignment) sets it permanently, and HideLegend()
        // only flips IsVisible. The CV plot asks for UpperRight to clear its bars, so without this the
        // next plot's legend appears upper-right - over the point cloud, and by exactly the mechanism
        // this method exists to stop.
        plt.Legend.Alignment = Alignment.LowerRight;
        plt.HideLegend();
    }
}
