using System;
using System.Collections.Generic;
using System.Linq;
using ScottPlot;
using SkylinePrism.Core.Qc;

namespace SkylinePrism.Core.Visualization;

/// <summary>
/// Static PNG plot rendering via ScottPlot (SkiaSharp) for the QC report. Visually
/// equivalent to the Python matplotlib plots, not pixel-identical. On headless Linux the
/// SkiaSharp backend needs libfontconfig1 / libfreetype6 installed.
///
/// Fixed sample-type colors match the Python palette: experimental #1f77b4, qc #ff7f0e,
/// reference #d62728, unknown #7f7f7f.
/// </summary>
public static class PlotRenderer
{
    // Canvas for the static QC-report PNGs. Sized so the (deliberately large) fonts below stay in
    // proportion and the image is still crisp when dropped into a figure or slide.
    private const int Width = 1100;
    private const int Height = 780;

    /// <summary>
    /// Font family used by every plot. Pinned rather than left to the backend's default because the
    /// default resolves per machine (whatever SkiaSharp finds first), which is why the same plot came out
    /// with visibly different text on different computers. First installed family wins; the last entry is
    /// the backend default, so this always resolves to something.
    /// </summary>
    public static readonly string PlotFontName = FirstInstalledFont(
        "Segoe UI",        // Windows 10/11
        "Helvetica",       // macOS
        "DejaVu Sans",     // most Linux distributions
        "Liberation Sans", // RHEL/Fedora family
        "Arial");

    /// <summary>
    /// CSS <c>font-family</c> stack for the HTML reports. The family the plots actually resolved to
    /// comes FIRST, so the report's own text matches the text inside the plot images it embeds -
    /// otherwise the page renders in whatever the browser picks and the headings sit in a visibly
    /// different typeface from the axis labels right below them. The rest is the same candidate list
    /// <see cref="PlotFontName"/> searches, for the case where a browser cannot use the resolved one.
    /// </summary>
    public static string HtmlFontStack
    {
        get
        {
            var families = new List<string> { PlotFontName }
                .Concat(new[] { "Segoe UI", "Helvetica", "DejaVu Sans", "Liberation Sans", "Arial" })
                .Select(f => f.Replace("'", ""))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .Select(f => $"'{f}'");
            return string.Join(", ", families) + ", sans-serif";
        }
    }

    private static string FirstInstalledFont(params string[] candidates)
    {
        foreach (var name in candidates)
        {
            try
            {
                // Skia hands back a substitute rather than null for an unknown family, so compare the
                // family it actually resolved to - otherwise every candidate would "exist".
                var typeface = Fonts.GetTypeface(name, false, false);
                if (string.Equals(typeface?.FamilyName, name, StringComparison.OrdinalIgnoreCase))
                    return name;
            }
            catch (Exception)
            {
                // A backend without font enumeration must not stop a plot from rendering.
            }
        }
        return Fonts.Default;
    }

    public static readonly IReadOnlyDictionary<string, string> TypeColors = new Dictionary<string, string>
    {
        ["experimental"] = "#1f77b4",
        ["qc"] = "#ff7f0e",
        ["reference"] = "#d62728",
        ["unknown"] = "#7f7f7f",
    };

    // ColorBrewer RdBu reversed (low -> high: blue -> white -> red) = matplotlib "RdBu_r": a diverging
    // map with white in the middle, as the Python control-correlation heatmap uses.
    private static readonly Color[] RdBuReversed =
    {
        Color.FromHex("#053061"), Color.FromHex("#2166ac"), Color.FromHex("#4393c3"),
        Color.FromHex("#92c5de"), Color.FromHex("#d1e5f0"), Color.FromHex("#f7f7f7"),
        Color.FromHex("#fddbc7"), Color.FromHex("#f4a582"), Color.FromHex("#d6604d"),
        Color.FromHex("#b2182b"), Color.FromHex("#67001f"),
    };

    /// <summary>
    /// Axis/legend styling shared by every QC plot (static PNGs and the interactive tool): thick left +
    /// bottom axes only (top + right hidden), a pinned font family, and text sized for figures and slides.
    /// </summary>
    /// <remarks>
    /// These plots are routinely saved or copied straight into a paper or a talk, where the axis labels
    /// have to survive being scaled down to a journal column or projected across a room - so the text is
    /// deliberately large relative to the plot rather than dashboard-sized. Scale everything together with
    /// <paramref name="fontScale"/> instead of changing individual sizes.
    /// </remarks>
    public static void StyleQcPlot(Plot plt, double fontScale = 1.0)
    {
        float Pt(double points) => (float)(points * fontScale);

        plt.Axes.Title.Label.FontName = PlotFontName;
        plt.Axes.Left.Label.FontName = PlotFontName;
        plt.Axes.Bottom.Label.FontName = PlotFontName;
        plt.Axes.Left.TickLabelStyle.FontName = PlotFontName;
        plt.Axes.Bottom.TickLabelStyle.FontName = PlotFontName;
        plt.Legend.FontName = PlotFontName;

        plt.Axes.Title.Label.FontSize = Pt(28);
        plt.Axes.Title.Label.Bold = true;
        plt.Axes.Left.Label.FontSize = Pt(30);
        plt.Axes.Bottom.Label.FontSize = Pt(30);
        plt.Axes.Left.Label.Bold = true;
        plt.Axes.Bottom.Label.Bold = true;
        plt.Axes.Left.TickLabelStyle.FontSize = Pt(24);
        plt.Axes.Bottom.TickLabelStyle.FontSize = Pt(24);
        plt.Legend.FontSize = Pt(22);

        // Anything added to the plot AFTER this call carries the backend's default font unless it is
        // routed through StyleColorBar / StyleTextLabel - ScottPlot styles those per item, and
        // StyleQcPlot cannot reach them.

        // Only the left + bottom axes, thick enough to stay visible when the figure is scaled down;
        // hide the top + right frame lines.
        plt.Axes.Left.FrameLineStyle.Width = 4;
        plt.Axes.Bottom.FrameLineStyle.Width = 4;
        plt.Axes.Right.FrameLineStyle.Width = 0;
        plt.Axes.Top.FrameLineStyle.Width = 0;
        plt.Axes.Left.MajorTickStyle.Width = 3;
        plt.Axes.Bottom.MajorTickStyle.Width = 3;
        plt.Axes.Left.MajorTickStyle.Length = 8;
        plt.Axes.Bottom.MajorTickStyle.Length = 8;
    }

    /// <summary>
    /// Style a color bar to match the axes. A color bar is a plot item, not part of the axis
    /// system, so <see cref="StyleQcPlot"/> does not reach it and it would otherwise render in the
    /// backend's default family next to axes that do not.
    /// </summary>
    public static void StyleColorBar(
        ScottPlot.Panels.ColorBar bar, string? label = null, double fontScale = 1.0)
    {
        if (label is not null)
            bar.Label = label;
        bar.LabelStyle.FontName = PlotFontName;
        bar.LabelStyle.FontSize = (float)(28 * fontScale);
        bar.LabelStyle.Bold = true;
        bar.Axis.TickLabelStyle.FontName = PlotFontName;
        bar.Axis.TickLabelStyle.FontSize = (float)(24 * fontScale);
    }

    /// <summary>
    /// Style a text annotation to match the axes. Same reason as <see cref="StyleColorBar"/>: text
    /// added to a plot carries its own font, so every call site has to opt in or the annotation ends
    /// up in a different typeface from the labels around it.
    /// </summary>
    public static void StyleTextLabel(
        ScottPlot.Plottables.Text text, double fontSize, bool bold = false)
    {
        text.LabelFontName = PlotFontName;
        text.LabelFontSize = (float)fontSize;
        text.LabelBold = bold;
    }

    /// <summary>
    /// The precursor-density map: retention time across, precursor m/z up, color = how many peptide
    /// precursors were eluting in that isolation window at that time (one cell = one DIA spectrum).
    /// </summary>
    /// <remarks>
    /// Sized for figures and slides, not for a dense QC dashboard: this plot is meant to be saved or
    /// copied straight out of the tool into a paper/presentation, where the QC-report font sizes are
    /// unreadable once the image is scaled down to a column or projected. Scale every font together with
    /// <paramref name="fontScale"/> rather than tweaking sizes individually.
    /// </remarks>
    public static void DrawPrecursorDensity(
        Plot plt, PrecursorDensityMap map, IColormap? colormap = null, string? title = null,
        double fontScale = 1.0)
    {
        // Publication-scale baselines (points at the default plot size), all scaled by fontScale.
        float Pt(double points) => (float)(points * fontScale);

        if (map.IsEmpty)
        {
            plt.Title(title ?? "No precursors to map");
            StyleQcPlot(plt, fontScale);
            return;
        }

        // Real isolation windows are not obliged to be equal-height, but a heatmap's cells are - so the
        // map rasterizes itself onto a uniform display grid that preserves the true m/z extents (row 0
        // at the top, matching ScottPlot's orientation).
        var heat = plt.Add.Heatmap(map.ToDisplayGrid());
        heat.Colormap = colormap ?? new ScottPlot.Colormaps.Viridis();
        heat.NaNCellColor = Colors.Transparent; // m/z not covered by any window: a gap, not a zero
        heat.Extent = new CoordinateRect(
            left: map.RtLow, right: map.RtHigh, bottom: map.MzLow, top: map.MzHigh);

        plt.XLabel("Retention time (min)");
        plt.YLabel("Precursor m/z");
        StyleQcPlot(plt, fontScale);

        // The color bar IS this plot's legend - it carries the number the reader cares about, so it gets
        // the same treatment as the axes.
        StyleColorBar(plt.Add.ColorBar(heat), "Precursors per spectrum", fontScale);

        if (title is not null)
        {
            // Run names are long and ScottPlot does not wrap titles, so shrink rather than clip.
            plt.Title(title);
            plt.Axes.Title.Label.FontSize = title.Length > 45 ? Pt(20) : Pt(26);
        }
        plt.Axes.SetLimits(map.RtLow, map.RtHigh, map.MzLow, map.MzHigh);
    }

    /// <summary>
    /// PNG of <see cref="DrawPrecursorDensity"/>, for headless use. Defaults to a figure-sized canvas
    /// (the fonts are scaled for that size, so a much smaller canvas will look crowded).
    /// </summary>
    public static byte[] PrecursorDensityPng(
        PrecursorDensityMap map, IColormap? colormap = null, string? title = null,
        int width = 1400, int height = 900, double fontScale = 1.0)
    {
        var plt = new Plot();
        DrawPrecursorDensity(plt, map, colormap, title, fontScale);
        return plt.GetImageBytes(width, height, ImageFormat.Png);
    }

    /// <summary>
    /// Dynamic range: log10 abundance against abundance rank - Skyline's Relative Abundance shape. Points
    /// not claimed by a protein list are drawn first in grey so the highlighted lists sit on top of them.
    /// </summary>
    /// <param name="highlights">
    /// Ordered groups drawn over the background, each with its own color and legend entry.
    /// </param>
    public static void DrawDynamicRange(
        Plot plt,
        IReadOnlyList<AbundanceEntry> background,
        IReadOnlyList<(string Label, string ColorHex, IReadOnlyList<AbundanceEntry> Entries)> highlights,
        string yLabel = "Log10 abundance",
        string xLabel = "Rank",
        double fontScale = 1.0)
    {
        if (background.Count > 0)
        {
            var dots = plt.Add.ScatterPoints(
                background.Select(e => (double)e.Rank).ToArray(),
                background.Select(e => e.Log10Abundance).ToArray());
            dots.Color = Color.FromHex("#9e9e9e").WithAlpha(0.55);
            dots.MarkerSize = 6;
        }

        foreach (var (label, colorHex, entries) in highlights)
        {
            if (entries.Count == 0)
                continue;
            var marks = plt.Add.ScatterPoints(
                entries.Select(e => (double)e.Rank).ToArray(),
                entries.Select(e => e.Log10Abundance).ToArray());
            marks.Color = Color.FromHex(colorHex);
            marks.MarkerSize = 13;
            marks.LegendText = $"{label} ({entries.Count})";
        }

        plt.XLabel(xLabel);
        plt.YLabel(yLabel);
        StyleQcPlot(plt, fontScale);
        if (highlights.Any(h => h.Entries.Count > 0))
            plt.ShowLegend(Alignment.UpperRight);
    }

    /// <summary>PNG of <see cref="DrawDynamicRange"/>, for headless use.</summary>
    public static byte[] DynamicRangePng(
        IReadOnlyList<AbundanceEntry> background,
        IReadOnlyList<(string Label, string ColorHex, IReadOnlyList<AbundanceEntry> Entries)> highlights,
        string yLabel = "Log10 abundance", string xLabel = "Rank",
        int width = 1400, int height = 900)
    {
        var plt = new Plot();
        DrawDynamicRange(plt, background, highlights, yLabel, xLabel);
        return plt.GetImageBytes(width, height, ImageFormat.Png);
    }

    /// <summary>Histogram of per-feature CVs with a median line, for one sample-type group.</summary>
    public static byte[] CvHistogram(double[] cvs, string title, string colorHex, double medianCv)
    {
        var plt = new Plot();
        if (cvs.Length > 0)
        {
            var maxCv = Math.Min(cvs.Max(), 100.0);
            const int bins = 30;
            var binWidth = Math.Max(maxCv / bins, 1e-6);
            var counts = new double[bins];
            var centers = new double[bins];
            for (var b = 0; b < bins; b++)
                centers[b] = (b + 0.5) * binWidth;
            foreach (var cv in cvs)
            {
                var b = (int)Math.Min(cv / binWidth, bins - 1);
                if (b >= 0)
                    counts[b]++;
            }
            var bars = plt.Add.Bars(centers, counts);
            bars.Color = Color.FromHex(colorHex);

            var med = plt.Add.VerticalLine(medianCv);
            med.Color = Colors.Black;
            med.LineWidth = 4;
            med.LinePattern = LinePattern.Dashed;
            med.LegendText = $"median {medianCv:0.0}%";
            plt.ShowLegend(Alignment.UpperRight);
        }
        plt.Title(title);
        plt.XLabel("CV (%)");
        plt.YLabel("Count");
        plt.Axes.Margins(bottom: 0); // bars sit on the x-axis (y starts at 0)
        StyleQcPlot(plt);
        return plt.GetImageBytes(Width, Height, ImageFormat.Png);
    }

    /// <summary>
    /// Overlaid before/after CV histograms (before in grey, after in color) with both median
    /// lines - the comparative-CV plot users expect from the Python report.
    /// </summary>
    public static byte[] CvComparison(double[] beforeCvs, double[] afterCvs, string title, string afterColorHex)
    {
        var plt = new Plot();

        void AddHist(double[] cvs, Color color, string label)
        {
            if (cvs.Length == 0)
                return;
            var maxCv = Math.Min(cvs.Max(), 100.0);
            const int bins = 30;
            var binWidth = Math.Max(maxCv / bins, 1e-6);
            var counts = new double[bins];
            var centers = new double[bins];
            for (var b = 0; b < bins; b++)
                centers[b] = (b + 0.5) * binWidth;
            foreach (var cv in cvs)
            {
                var b = (int)Math.Min(cv / binWidth, bins - 1);
                if (b >= 0)
                    counts[b]++;
            }
            var bars = plt.Add.Bars(centers, counts);
            bars.Color = color.WithAlpha((byte)140);

            var med = Numerics.Stats.NanMedian(cvs);
            var line = plt.Add.VerticalLine(med);
            line.Color = color;
            line.LineWidth = 4;
            line.LinePattern = LinePattern.Dashed;
            line.LegendText = $"{label} median {med:0.0}%";
        }

        AddHist(beforeCvs, Color.FromHex("#7f7f7f"), "Before");
        AddHist(afterCvs, Color.FromHex(afterColorHex), "After");
        plt.ShowLegend(Alignment.UpperRight);
        plt.Title(title);
        plt.XLabel("CV (%)");
        plt.YLabel("Count");
        plt.Axes.Margins(bottom: 0); // bars sit on the x-axis (y starts at 0)
        StyleQcPlot(plt);
        return plt.GetImageBytes(Width, Height, ImageFormat.Png);
    }

    /// <summary>2-D PCA scatter colored by sample type.</summary>
    public static byte[] PcaScatter(double[,] scores2d, IReadOnlyList<string> sampleTypes, string title)
    {
        var plt = new Plot();
        var byType = new Dictionary<string, (List<double> X, List<double> Y)>();
        for (var i = 0; i < sampleTypes.Count; i++)
        {
            var t = sampleTypes[i];
            if (!byType.TryGetValue(t, out var lists))
                byType[t] = lists = (new List<double>(), new List<double>());
            lists.X.Add(scores2d[i, 0]);
            lists.Y.Add(scores2d[i, 1]);
        }

        foreach (var (type, lists) in byType.OrderBy(kv => kv.Key, StringComparer.Ordinal))
        {
            var markers = plt.Add.Markers(lists.X.ToArray(), lists.Y.ToArray());
            markers.Color = Color.FromHex(TypeColors.GetValueOrDefault(type, "#7f7f7f"));
            markers.MarkerSize = 15; // sized to match the figure-scale axis text
            markers.LegendText = type;
        }
        plt.ShowLegend();
        plt.Title(title);
        plt.XLabel("PC1");
        plt.YLabel("PC2");
        StyleQcPlot(plt);
        return plt.GetImageBytes(Width, Height, ImageFormat.Png);
    }

    /// <summary>
    /// Correlation heatmap among the given (control) sample columns - the "did batch correction
    /// tighten the controls" diagnostic (Pearson over features present in both samples).
    /// </summary>
    public static byte[] CorrelationHeatmap(
        double[,] featuresBySamples, IReadOnlyList<int> cols, string title,
        IReadOnlyList<string>? colTypes = null)
    {
        var plt = new Plot();
        DrawCorrelationHeatmap(plt, featuresBySamples, cols, title, colTypes);
        return plt.GetImageBytes(Width, Height, ImageFormat.Png);
    }

    /// <summary>
    /// <see cref="CorrelationHeatmap"/> onto a caller-supplied plot. Split out so the styling is
    /// inspectable by tests - this plot is dense enough to need its own tick sizes, which is exactly
    /// how it drifted into a different font family from every other plot in the report.
    /// </summary>
    public static void DrawCorrelationHeatmap(
        Plot plt, double[,] featuresBySamples, IReadOnlyList<int> cols, string title,
        IReadOnlyList<string>? colTypes = null)
    {
        var n = cols.Count;
        var corr = new double[n, n];
        for (var a = 0; a < n; a++)
            for (var b = 0; b < n; b++)
                corr[a, b] = a == b ? 1.0 : Pearson(featuresBySamples, cols[a], cols[b]);

        // Reorder by average-linkage hierarchical clustering of (1 - corr), like the Python report
        // (scipy linkage(method="average") + leaves_list) - no dendrogram, just a similarity ordering.
        var order = n >= 3 ? AverageLinkageOrder(corr) : Enumerable.Range(0, n).ToArray();
        var rc = new double[n, n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
                rc[i, j] = corr[order[i], order[j]];
        var labels = TypeLabels(order, colTypes);

        // Color range: 1.0 = best (red), the min off-diagonal correlation = worst (blue).
        var vmin = 1.0;
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
                if (i != j && !double.IsNaN(rc[i, j]) && rc[i, j] < vmin)
                    vmin = rc[i, j];
        vmin = Math.Max(-1.0, vmin);

        // ScottPlot draws array row 0 at the TOP (matplotlib/Python orientation), so pass rc directly:
        // the 1.0 diagonal runs top-left -> bottom-right. Cell rc[i,j] center is (j+0.5, n-0.5-i).
        var hm = plt.Add.Heatmap(rc);
        hm.Colormap = new ScottPlot.Colormaps.CustomInterpolated(RdBuReversed);
        hm.ManualRange = new ScottPlot.Range(vmin, 1.0);
        hm.Position = new ScottPlot.CoordinateRect(0, n, 0, n);
        // The color bar carries only ~15 labels however many samples there are, so it does not have
        // to shrink with the grid - keep it near the size the other plots' tick labels use.
        StyleColorBar(plt.Add.ColorBar(hm), fontScale: HeatmapColorBarScale);

        // Cell value annotations (skip when too many samples to stay readable, like seaborn annot<=15).
        if (n <= 15)
            for (var i = 0; i < n; i++)
                for (var j = 0; j < n; j++)
                {
                    if (double.IsNaN(rc[i, j]))
                        continue;
                    var t = plt.Add.Text(rc[i, j].ToString("0.00"), j + 0.5, (n - 1 - i) + 0.5);
                    t.LabelAlignment = Alignment.MiddleCenter;
                    StyleTextLabel(t, 13);
                    // white text on the dark (low/high) ends, dark text near the white middle.
                    var mid = (vmin + 1.0) / 2.0;
                    t.LabelFontColor = Math.Abs(rc[i, j] - mid) > (1.0 - vmin) * 0.28 ? Colors.White : Colors.Black;
                }

        // Ref_001 / QC_001 / ... tick labels at cell centers.
        var pos = new double[n];
        for (var i = 0; i < n; i++)
            pos[i] = i + 0.5;
        plt.Axes.Bottom.TickGenerator = new ScottPlot.TickGenerators.NumericManual(pos, labels.ToArray());
        var yLabels = new string[n];
        for (var i = 0; i < n; i++)
            yLabels[i] = labels[n - 1 - i]; // y is flipped
        plt.Axes.Left.TickGenerator = new ScottPlot.TickGenerators.NumericManual(pos, yLabels);

        plt.Title(title);

        // Same font family and title treatment as every other plot in the report; only the things a
        // dense grid genuinely needs differently are overridden below. Going through StyleQcPlot is
        // what keeps the family shared - setting sizes here and nothing else is how this plot ended
        // up in the backend's default typeface while its neighbours were in Segoe UI.
        StyleQcPlot(plt);

        // A heatmap needs no axis frame or tick marks - the cells are the grid. Labels are centered on
        // the cells (y-labels on rows; x-labels rotated, right-aligned so they read up into their
        // column). Keeps the labels but drops the L-shaped axis lines.
        plt.Axes.Left.FrameLineStyle.Width = 0;
        plt.Axes.Bottom.FrameLineStyle.Width = 0;
        plt.Axes.Right.FrameLineStyle.Width = 0;
        plt.Axes.Top.FrameLineStyle.Width = 0;
        plt.Axes.Left.MajorTickStyle.Length = 0;
        plt.Axes.Bottom.MajorTickStyle.Length = 0;

        // One tick label per sample, up to ~30 of them: these have to be small to fit, which is the
        // one place this plot cannot follow the shared sizes.
        plt.Axes.Left.TickLabelStyle.FontSize = 13;
        plt.Axes.Bottom.TickLabelStyle.FontSize = 13;
        plt.Axes.Left.TickLabelStyle.Alignment = Alignment.MiddleRight;
        plt.Axes.Bottom.TickLabelStyle.Rotation = 45;
        plt.Axes.Bottom.TickLabelStyle.Alignment = Alignment.MiddleRight;
    }

    /// <summary>
    /// The correlation heatmap's color bar, relative to the shared sizes. Smaller than the other
    /// plots because it sits beside a dense grid, but nowhere near as small as that grid's own tick
    /// labels have to be - the bar's label count does not grow with the cohort.
    /// </summary>
    private const double HeatmapColorBarScale = 0.7;

    // Short per-sample labels grouped by type: Ref_001, QC_001, Exp_001 (falls back to S_001).
    private static List<string> TypeLabels(int[] order, IReadOnlyList<string>? colTypes)
    {
        var counters = new Dictionary<string, int>();
        var labels = new List<string>(order.Length);
        foreach (var idx in order)
        {
            var type = colTypes is not null && idx < colTypes.Count ? colTypes[idx] : "";
            var prefix = type switch
            {
                "reference" => "Ref",
                "qc" => "QC",
                "experimental" => "Exp",
                "blank" => "Blank",
                _ => "S",
            };
            counters.TryGetValue(prefix, out var c);
            counters[prefix] = c + 1;
            labels.Add($"{prefix}_{c + 1:000}");
        }
        return labels;
    }

    // Leaf order from average-linkage (UPGMA) agglomerative clustering of the distance matrix 1 - corr.
    private static int[] AverageLinkageOrder(double[,] corr)
    {
        var n = corr.GetLength(0);
        var dist = new double[n, n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
            {
                var c = corr[i, j];
                dist[i, j] = i == j ? 0.0 : (double.IsNaN(c) ? 1.0 : 1.0 - c);
            }

        var active = new List<int>();          // active cluster ids
        var members = new Dictionary<int, List<int>>();
        var size = new Dictionary<int, int>();
        var children = new Dictionary<int, (int, int)>();
        for (var i = 0; i < n; i++)
        {
            active.Add(i);
            members[i] = new List<int> { i };
            size[i] = 1;
        }
        var d = new Dictionary<(int, int), double>();
        for (var i = 0; i < n; i++)
            for (var j = i + 1; j < n; j++)
                d[(i, j)] = dist[i, j];
        double Get(int a, int b) => a < b ? d[(a, b)] : d[(b, a)];

        var next = n;
        while (active.Count > 1)
        {
            // closest pair
            int bi = active[0], bj = active[1];
            var best = double.MaxValue;
            for (var x = 0; x < active.Count; x++)
                for (var y = x + 1; y < active.Count; y++)
                {
                    var dd = Get(active[x], active[y]);
                    if (dd < best) { best = dd; bi = active[x]; bj = active[y]; }
                }

            var id = next++;
            var si = size[bi];
            var sj = size[bj];
            children[id] = (bi, bj);
            members[id] = new List<int>(members[bi]);
            members[id].AddRange(members[bj]);
            size[id] = si + sj;
            // UPGMA update: d(new,k) = (si*d(bi,k) + sj*d(bj,k)) / (si+sj)
            foreach (var k in active)
            {
                if (k == bi || k == bj) continue;
                var nd = (si * Get(bi, k) + sj * Get(bj, k)) / (si + sj);
                var key = id < k ? (id, k) : (k, id);
                d[key] = nd;
            }
            active.Remove(bi);
            active.Remove(bj);
            active.Add(id);
        }

        // leaf order = in-order traversal of the merge tree
        var result = new List<int>(n);
        void Walk(int node)
        {
            if (node < n) { result.Add(node); return; }
            var (l, r) = children[node];
            Walk(l);
            Walk(r);
        }
        Walk(active[0]);
        return result.ToArray();
    }

    private static double Pearson(double[,] m, int ca, int cb)
    {
        var nF = m.GetLength(0);
        double sa = 0, sb = 0;
        var cnt = 0;
        for (var f = 0; f < nF; f++)
        {
            var va = m[f, ca];
            var vb = m[f, cb];
            if (double.IsNaN(va) || double.IsNaN(vb))
                continue;
            sa += va;
            sb += vb;
            cnt++;
        }
        if (cnt < 2)
            return double.NaN;
        var ma = sa / cnt;
        var mb = sb / cnt;
        double num = 0, da = 0, db = 0;
        for (var f = 0; f < nF; f++)
        {
            var va = m[f, ca];
            var vb = m[f, cb];
            if (double.IsNaN(va) || double.IsNaN(vb))
                continue;
            var xa = va - ma;
            var xb = vb - mb;
            num += xa * xb;
            da += xa * xa;
            db += xb * xb;
        }
        var den = Math.Sqrt(da * db);
        return den > 0 ? num / den : double.NaN;
    }

    /// <summary>
    /// Per-sample LOWESS curves of LOG2 abundance vs mean RT, colored by sample type - the
    /// diagnostic for RT-lowess normalization (before curves spread; after curves collapse).
    /// </summary>
    public static byte[] RtLowessCurves(
        double[,] featuresBySamples, double[] meanRt, IReadOnlyList<string> types, string title)
    {
        var nF = featuresBySamples.GetLength(0);
        var nS = featuresBySamples.GetLength(1);
        var plt = new Plot();

        double rtMin = double.PositiveInfinity, rtMax = double.NegativeInfinity;
        foreach (var rt in meanRt)
        {
            if (double.IsNaN(rt)) continue;
            if (rt < rtMin) rtMin = rt;
            if (rt > rtMax) rtMax = rt;
        }
        if (rtMin < rtMax)
        {
            const int g = 80;
            var grid = new double[g];
            var step = (rtMax - rtMin) / (g - 1);
            for (var k = 0; k < g; k++)
                grid[k] = rtMin + k * step;

            // Color by the passed label (sample type in the report, Group-by value in the tool): a
            // distinct color per distinct value, so N groups -> N colors.
            var groupColors = new Dictionary<string, Color>(StringComparer.Ordinal);
            var gci = 0;
            foreach (var lab in types.Distinct().OrderBy(x => x, StringComparer.Ordinal))
                groupColors[lab] = GroupColor(lab, gci++);
            var seenType = new HashSet<string>();
            for (var s = 0; s < nS; s++)
            {
                var xs = new List<double>(nF);
                var ys = new List<double>(nF);
                for (var f = 0; f < nF; f++)
                {
                    var v = featuresBySamples[f, s];
                    var rt = meanRt[f];
                    if (!double.IsNaN(v) && !double.IsNaN(rt))
                    {
                        xs.Add(rt);
                        ys.Add(v);
                    }
                }
                if (xs.Count < 20)
                    continue;

                var order = Numerics.Stats.ArgSort(xs.ToArray());
                var xSorted = new double[order.Length];
                var ySorted = new double[order.Length];
                for (var k = 0; k < order.Length; k++)
                {
                    xSorted[k] = xs[order[k]];
                    ySorted[k] = ys[order[k]];
                }

                // Fit with delta = 1% of the RT range (statsmodels' speedup, same as the normalization):
                // over tens of thousands of peptides an un-delta'd LOWESS is O(n * frac*n * iters) and
                // dominated the whole QC report; delta fits anchors and interpolates between them.
                var yfit = Numerics.Lowess.Fit(xSorted, ySorted, 0.3, delta: (rtMax - rtMin) * 0.01);

                var gx = new List<double>(g);
                var gy = new List<double>(g);
                for (var k = 0; k < g; k++)
                {
                    if (grid[k] < xSorted[0] || grid[k] > xSorted[^1])
                        continue;
                    gx.Add(grid[k]);
                    gy.Add(Numerics.Stats.Interp(grid[k], xSorted, yfit));
                }
                if (gx.Count < 2)
                    continue;

                var type = types[s];
                var sc = plt.Add.Scatter(gx.ToArray(), gy.ToArray());
                sc.MarkerSize = 0;
                sc.LineWidth = 3;
                sc.Color = groupColors[type].WithAlpha((byte)110);
                if (seenType.Add(type))
                    sc.LegendText = type;
            }
            plt.ShowLegend();
        }

        plt.Title(title);
        plt.XLabel("Retention time");
        plt.YLabel("log2 abundance");
        StyleQcPlot(plt);
        return plt.GetImageBytes(Width, Height, ImageFormat.Png);
    }

    /// <summary>
    /// RT-binned median CV before vs after, for one control group (grouped bars: light = before,
    /// solid = after). Ports plot_rt_bin_cv_comparison for a single panel.
    /// </summary>
    public static byte[] RtBinCv(
        double[,] raw, double[,] corrected, double[] meanRt, IReadOnlyList<int> controlIdx,
        string title, string colorHex, int nBins = 8)
    {
        var plt = new Plot();

        double rtMin = double.PositiveInfinity, rtMax = double.NegativeInfinity;
        foreach (var rt in meanRt)
        {
            if (double.IsNaN(rt)) continue;
            if (rt < rtMin) rtMin = rt;
            if (rt > rtMax) rtMax = rt;
        }

        if (rtMin < rtMax && controlIdx.Count >= 2)
        {
            var step = (rtMax - rtMin) / nBins;
            var before = Color.FromHex(colorHex).WithAlpha((byte)110);
            var after = Color.FromHex(colorHex);
            var bars = new List<ScottPlot.Bar>();
            for (var i = 0; i < nBins; i++)
            {
                var lo = rtMin + i * step;
                var hi = i == nBins - 1 ? double.PositiveInfinity : rtMin + (i + 1) * step;
                var cvB = MedianBinCv(raw, meanRt, controlIdx, lo, hi);
                var cvA = MedianBinCv(corrected, meanRt, controlIdx, lo, hi);
                bars.Add(new ScottPlot.Bar { Position = i - 0.2, Value = double.IsNaN(cvB) ? 0 : cvB, Size = 0.38, FillColor = before });
                bars.Add(new ScottPlot.Bar { Position = i + 0.2, Value = double.IsNaN(cvA) ? 0 : cvA, Size = 0.38, FillColor = after });
            }
            plt.Add.Bars(bars);
        }

        plt.Title(title + " (light = before, solid = after)");
        plt.XLabel("RT bin");
        plt.YLabel("Median CV (%)");
        plt.Axes.Margins(bottom: 0); // bars sit on the x-axis (y starts at 0)
        StyleQcPlot(plt);
        return plt.GetImageBytes(Width, Height, ImageFormat.Png);
    }

    /// <summary>
    /// Abundance (LOG2) distribution per RT bin as Tukey boxplots - ports
    /// plot_rt_bin_boxplot_comparison for a single panel (all samples pooled per bin).
    /// </summary>
    public static byte[] RtBinBoxplot(
        double[,] featuresBySamples, double[] meanRt, string title, string colorHex, int nBins = 8)
    {
        var nF = featuresBySamples.GetLength(0);
        var nS = featuresBySamples.GetLength(1);
        var plt = new Plot();

        double rtMin = double.PositiveInfinity, rtMax = double.NegativeInfinity;
        foreach (var rt in meanRt)
        {
            if (double.IsNaN(rt)) continue;
            if (rt < rtMin) rtMin = rt;
            if (rt > rtMax) rtMax = rt;
        }

        if (rtMin < rtMax)
        {
            var step = (rtMax - rtMin) / nBins;
            var fill = Color.FromHex(colorHex).WithAlpha((byte)120);
            var boxes = new List<ScottPlot.Box>();
            for (var i = 0; i < nBins; i++)
            {
                var lo = rtMin + i * step;
                var hi = i == nBins - 1 ? double.PositiveInfinity : rtMin + (i + 1) * step;
                var vals = new List<double>();
                for (var f = 0; f < nF; f++)
                {
                    var rt = meanRt[f];
                    if (double.IsNaN(rt) || rt < lo || rt >= hi)
                        continue;
                    for (var s = 0; s < nS; s++)
                    {
                        var v = featuresBySamples[f, s];
                        if (!double.IsNaN(v))
                            vals.Add(v);
                    }
                }
                if (vals.Count < 6)
                    continue;

                var arr = vals.ToArray();
                var q1 = Numerics.Stats.PercentileLinear(arr, 25);
                var med = Numerics.Stats.PercentileLinear(arr, 50);
                var q3 = Numerics.Stats.PercentileLinear(arr, 75);
                var iqr = q3 - q1;
                double dataMin = double.PositiveInfinity, dataMax = double.NegativeInfinity;
                foreach (var v in arr) { if (v < dataMin) dataMin = v; if (v > dataMax) dataMax = v; }
                boxes.Add(new ScottPlot.Box
                {
                    Position = i,
                    Width = 0.7,
                    BoxMin = q1,
                    BoxMiddle = med,
                    BoxMax = q3,
                    WhiskerMin = Math.Max(dataMin, q1 - 1.5 * iqr),
                    WhiskerMax = Math.Min(dataMax, q3 + 1.5 * iqr),
                    FillColor = fill, // Box.Fill is obsolete in ScottPlot 5; only the color is set here
                });
            }
            if (boxes.Count > 0)
                plt.Add.Boxes(boxes);
        }

        plt.Title(title);
        plt.XLabel("RT bin");
        plt.YLabel("log2 abundance");
        StyleQcPlot(plt);
        return plt.GetImageBytes(Width, Height, ImageFormat.Png);
    }

    private static double MedianBinCv(
        double[,] log2Matrix, double[] meanRt, IReadOnlyList<int> cols, double lo, double hi)
    {
        var nF = log2Matrix.GetLength(0);
        var cvs = new List<double>();
        var buf = new double[cols.Count];
        for (var f = 0; f < nF; f++)
        {
            var rt = meanRt[f];
            if (double.IsNaN(rt) || rt < lo || rt >= hi)
                continue;
            var cnt = 0;
            for (var k = 0; k < cols.Count; k++)
            {
                var v = log2Matrix[f, cols[k]];
                if (!double.IsNaN(v))
                    buf[cnt++] = Math.Pow(2, v);
            }
            if (cnt < 2)
                continue;
            var mean = 0.0;
            for (var k = 0; k < cnt; k++) mean += buf[k];
            mean /= cnt;
            if (mean <= 0) continue;
            var ss = 0.0;
            for (var k = 0; k < cnt; k++) { var d = buf[k] - mean; ss += d * d; }
            var std = Math.Sqrt(ss / (cnt - 1));
            cvs.Add(std / mean * 100.0);
        }
        return cvs.Count == 0 ? double.NaN : Numerics.Stats.NanMedian(cvs.ToArray());
    }

    // The 10 distinct matplotlib tab10 hues first (so a few groups get well-separated colors), then
    // their tab20 light variants (so many overlaid samples still each get a color).
    private static readonly string[] Tab20 =
    {
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
        "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    };

    /// <summary>Cycled per-sample color for overlaid density curves.</summary>
    public static Color SampleColor(int index) => Color.FromHex(Tab20[((index % Tab20.Length) + Tab20.Length) % Tab20.Length]);

    /// <summary>
    /// Standardized color for a group label, consistent across every plot. Sample types map to fixed
    /// colors in any spelling (Skyline "Standard"/"Quality Control"/"Unknown" or the PRISM
    /// reference/qc/experimental names); any other value (e.g. a Condition annotation) gets a cycled color.
    /// </summary>
    public static Color GroupColor(string? label, int cycleIndex = 0)
    {
        var key = (label ?? "").Trim().ToLowerInvariant().Replace(" ", "");
        return key switch
        {
            "reference" or "standard" or "std" => Color.FromHex(TypeColors["reference"]),
            "qc" or "qualitycontrol" => Color.FromHex(TypeColors["qc"]),
            "experimental" or "unknown" => Color.FromHex(TypeColors["experimental"]),
            "blank" or "solvent" or "doubleblank" => Color.FromHex(TypeColors["unknown"]),
            _ => SampleColor(cycleIndex),
        };
    }

    /// <summary>
    /// Overlay one Gaussian-KDE density curve per sample of the (features x samples) LOG2 matrix onto
    /// <paramref name="plt"/> - the Python "Intensity Distribution" plot. Shared by the static PNG in the
    /// HTML report and the interactive tool, which must look the same.
    ///
    /// <para>With <paramref name="groupLabels"/> every sample in a group shares a color (via
    /// <see cref="GroupColor"/>, so a type keeps its color in any spelling) and the plot gets a legend.
    /// Without them each sample gets its own cycled color and there is no legend - only appropriate when
    /// there is genuinely no grouping to show.</para>
    /// </summary>
    public static void DrawIntensityDensity(Plot plt, double[,] log2Matrix, IReadOnlyList<string>? groupLabels = null)
    {
        var nF = log2Matrix.GetLength(0);
        var nS = log2Matrix.GetLength(1);
        var perSample = new List<double>[nS];
        double gmin = double.PositiveInfinity, gmax = double.NegativeInfinity;
        for (var s = 0; s < nS; s++)
        {
            var vals = new List<double>(nF);
            for (var f = 0; f < nF; f++)
            {
                var v = log2Matrix[f, s];
                if (!double.IsNaN(v) && !double.IsInfinity(v))
                {
                    vals.Add(v);
                    if (v < gmin) gmin = v;
                    if (v > gmax) gmax = v;
                }
            }
            perSample[s] = vals;
        }
        if (gmin >= gmax)
            return;

        const int gN = 256;
        var grid = new double[gN];
        var step = (gmax - gmin) / (gN - 1);
        for (var i = 0; i < gN; i++)
            grid[i] = gmin + i * step;

        // Color by group value if given (every sample in a group shares a color; N groups -> N colors),
        // else one color per sample. Legend + a distinct color per distinct group.
        Dictionary<string, Color>? groupColors = null;
        if (groupLabels is not null)
        {
            groupColors = new Dictionary<string, Color>(StringComparer.Ordinal);
            var ci = 0;
            foreach (var lab in groupLabels.Distinct().OrderBy(x => x, StringComparer.Ordinal))
                groupColors[lab] = GroupColor(lab, ci++);
        }
        var seen = new HashSet<string>(StringComparer.Ordinal);

        for (var s = 0; s < nS; s++)
        {
            if (perSample[s].Count < 2)
                continue;
            var density = Numerics.Kde.Estimate(perSample[s], grid);
            var line = plt.Add.Scatter(grid, density);
            line.MarkerSize = 0;
            line.LineWidth = 3.0f; // thick, like the RT-lowess curves
            if (groupColors is not null && groupLabels is not null)
            {
                var lab = groupLabels[s];
                line.Color = groupColors[lab].WithAlpha((byte)150);
                if (seen.Add(lab))
                    line.LegendText = lab;
            }
            else
            {
                line.Color = SampleColor(s).WithAlpha((byte)130);
            }
        }
        if (groupColors is not null)
            plt.ShowLegend();

        // Self-style, like the other Draw* methods. Callers style again afterwards (it is idempotent)
        // but must not have to: a draw method that leaves its legend in the backend's default font is
        // how the report ends up with two typefaces in it.
        StyleQcPlot(plt);
    }

    /// <summary>
    /// LOG2 intensity density curves, one per sample, colored by sample type (the Python "Intensity
    /// Distribution" plot).
    ///
    /// <para><paramref name="sampleTypes"/> must be passed through to the draw call: it is what makes the
    /// HTML report match the tool's interactive plot, and the PCA and RT-lowess plots in this same report,
    /// all of which color by group. Dropping it falls back to a per-sample color cycle - a rainbow that
    /// carries no information and silently disagrees with every other view of the same data.</para>
    /// </summary>
    public static byte[] IntensityDistribution(
        double[,] log2Matrix, IReadOnlyList<string> sampleTypes, string title)
    {
        var plt = new Plot();
        DrawIntensityDensity(plt, log2Matrix, sampleTypes);
        plt.Title(title);
        plt.XLabel("Log2 Abundance");
        plt.YLabel("Density");
        StyleQcPlot(plt);
        return plt.GetImageBytes(Width, Height, ImageFormat.Png);
    }
}
