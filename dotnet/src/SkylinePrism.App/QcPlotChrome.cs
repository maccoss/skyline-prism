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
    /// How near the cursor has to be, in pixels, to read as hovering a point. Comfortably larger than
    /// the 12 px markers, because the target is a mouse pointer rather than a click.
    /// </summary>
    public const double HoverRadiusPx = 18;

    /// <summary>
    /// Index of the plotted point nearest <paramref name="cursor"/> and within
    /// <see cref="HoverRadiusPx"/> of it, or -1 when the cursor is over empty space.
    /// </summary>
    /// <remarks>
    /// Pixels rather than data coordinates, and squared distances rather than distances: a fixed
    /// pixel radius is what "near the cursor" means on screen, and it cannot be expressed in data
    /// units on a plot whose two axes carry different quantities at different scales - a radius that
    /// looked right on the PCA would be a sliver on the marker-score plot, whose x is a group index
    /// and whose y is a PC1 score.
    ///
    /// <para>The -1 case is the one that matters and the easy one to lose: it is what hides the
    /// readout again when the cursor moves off a point. Returning the nearest point regardless would
    /// leave a label stuck to the plot for as long as the mouse stayed inside it.</para>
    /// </remarks>
    public static int NearestPoint(IReadOnlyList<Pixel> points, Pixel cursor)
    {
        var best = double.MaxValue;
        var bestIdx = -1;
        for (var i = 0; i < points.Count; i++)
        {
            double dx = points[i].X - cursor.X, dy = points[i].Y - cursor.Y;
            var d2 = dx * dx + dy * dy;
            if (d2 < best)
            {
                best = d2;
                bestIdx = i;
            }
        }
        return best <= HoverRadiusPx * HoverRadiusPx ? bestIdx : -1;
    }

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
        // Undo the empty state before anything else. PlotRenderer.DrawEmptyState strips the axes and
        // grid so a panel with no data cannot be misread as a flat measurement - and Clear() does not
        // put them back, so without this the first REAL plot after an empty one renders with no axes
        // at all. Exactly the class of leftover this method exists for.
        plt.Axes.Frameless(false);
        plt.ShowGrid();

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
