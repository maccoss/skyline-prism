using System;
using System.Collections.Generic;
using System.Linq;
using System.Globalization;
using System.Windows;
using ScottPlot;
using SkylinePrism.Core.Numerics;
using SkylinePrism.Core.Visualization;

namespace SkylinePrism.App;

/// <summary>
/// A per-feature detail popup: a boxplot of one feature's log2 abundance split by the two contrast
/// groups, with jittered per-sample points overlaid. Opened by clicking a point on the Volcano view,
/// mirroring the explorer's "Feature detail" panel. Reused across clicks (one window, redrawn).
/// </summary>
public partial class FeatureDetailWindow : Window
{
    private const string AColor = "#1f77b4"; // group A: blue
    private const string BColor = "#d62728"; // group B: red

    /// <summary>
    /// Each drawn point's location and the replicate it came from, for the hover readout. A boxplot
    /// hides which sample is which by construction, so an outlier in the swarm is not identifiable
    /// without this - and identifying one is usually the reason for opening this window.
    /// </summary>
    private readonly List<(Coordinates Loc, string Replicate)> _points = new();

    private ScottPlot.Plottables.Marker? _hoverMarker;
    private ScottPlot.Plottables.Text? _hoverText;

    public FeatureDetailWindow()
    {
        InitializeComponent();
        DetailPlot.MouseMove += OnDetailPlotMouseMove;
    }

    /// <summary>Draw (or redraw) the boxplot for one feature.</summary>
    /// <param name="aReplicates">
    /// Replicate names for <paramref name="aValues"/>, same order and length - the caller filters
    /// non-finite values out of both together, so the two stay aligned.
    /// </param>
    /// <param name="bReplicates">Replicate names for <paramref name="bValues"/>.</param>
    public void ShowFeature(string label, string featureId, double logFc, double q,
        string aName, IReadOnlyList<double> aValues, IReadOnlyList<string> aReplicates,
        string bName, IReadOnlyList<double> bValues, IReadOnlyList<string> bReplicates)
    {
        var inv = CultureInfo.InvariantCulture;
        var fc = double.IsFinite(logFc) ? logFc.ToString("0.###", inv) : "n/a";
        var qs = double.IsFinite(q) ? q.ToString("0.##e0", inv) : "n/a";
        ShowGroups(label, featureId,
            // "non-missing", never "detected": a Skyline export integrates imputed peak boundaries
            // for every replicate, so the corrected matrix is dense and a finite value here may
            // carry no real signal at all. Calling these counts detections would contradict the
            // pane's own caveat and overstate what the boxplot shows.
            $"log2FC (B/A) = {fc}   adj.P = {qs}   -   {aName}: n={aValues.Count}, "
            + $"{bName}: n={bValues.Count} (non-missing values)",
            new[] { new DetailGroup(aName, aValues, aReplicates), new DetailGroup(bName, bValues, bReplicates) });
    }

    /// <summary>One box: its name, its values and the replicate behind each value, index-aligned.</summary>
    public sealed record DetailGroup(string Name, IReadOnlyList<double> Values, IReadOnlyList<string> Replicates);

    /// <summary>
    /// Colors past the Volcano's two arms, for the Dynamic Range pane's sample types. Tableau's, so a
    /// third group reads as one more of the same family rather than as an alarm.
    /// </summary>
    private static readonly string[] MoreColors = { "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#7f7f7f" };

    /// <summary>
    /// Draw (or redraw) one feature as a box per group with its per-sample points jittered over it -
    /// the Volcano's two arms, or the Dynamic Range pane's sample types. An empty group keeps its
    /// place and its name on the axis, so the colors mean the same group whatever is missing.
    /// </summary>
    public void ShowGroups(string label, string featureId, string subtitle, IReadOnlyList<DetailGroup> groups)
    {
        _points.Clear();
        HeaderText.Text = label == featureId ? label : $"{label} ({featureId})";
        SubText.Text = subtitle;

        var plt = DetailPlot.Plot;
        plt.Clear();

        var shown = groups;
        var boxes = new List<ScottPlot.Box>();
        for (var i = 0; i < shown.Count; i++)
            AddBox(boxes, i, shown[i].Values, ColorFor(i));
        if (boxes.Count > 0)
            plt.Add.Boxes(boxes);

        // Jittered per-sample points, seeded so the layout is stable across redraws of the same feature.
        var rng = new Random(HashCode.Combine(featureId, shown.Sum(g => g.Values.Count), shown.Count) & 0x7fffffff);
        for (var i = 0; i < shown.Count; i++)
            AddJitter(plt, i, shown[i].Values, shown[i].Replicates, ColorFor(i), rng);
        AddHoverOverlay(plt);

        plt.Axes.Bottom.TickGenerator = new ScottPlot.TickGenerators.NumericManual(
            Enumerable.Range(0, shown.Count).Select(i => (double)i).ToArray(),
            shown.Select(g => g.Name).ToArray());
        plt.Axes.SetLimitsX(-0.6, Math.Max(1, shown.Count - 1) + 0.6);
        plt.XLabel("group");
        plt.YLabel("log2 abundance");
        PlotRenderer.StyleQcPlot(plt);
        DetailPlot.Refresh();

        static string ColorFor(int i) => i switch
        {
            0 => AColor,
            1 => BColor,
            _ => MoreColors[(i - 2) % MoreColors.Length],
        };
    }

    /// <summary>
    /// The per-feature view under a TREND design: log2 abundance against the trend column, with the
    /// fitted line, and - when subjects are given - one faint line per subject.
    /// </summary>
    /// <remarks>
    /// <para>A boxplot needs two groups to box, and a trend has none. The scatter is the honest
    /// equivalent, and the per-subject lines are the point of a within-subject design made visible:
    /// each subject sits at its own level while moving in a consistent direction, which is exactly
    /// the pattern a fixed-effect subject block extracts and an independent fit cannot see.</para>
    ///
    /// <para>The fitted line is drawn from <paramref name="logFc"/>, the modeled change across the
    /// observed range, rather than refitted here - so the line a reader sees is the one the
    /// reported number came from, and the two cannot drift.</para>
    /// </remarks>
    public void ShowTrendFeature(string label, string featureId, double logFc, double q,
        string trendColumn, IReadOnlyList<double> x, IReadOnlyList<double> values,
        IReadOnlyList<string> replicates, IReadOnlyList<string>? subjects)
    {
        _points.Clear();
        var inv = CultureInfo.InvariantCulture;
        var fc = double.IsFinite(logFc) ? logFc.ToString("0.###", inv) : "n/a";
        var qs = double.IsFinite(q) ? q.ToString("0.##e0", inv) : "n/a";
        HeaderText.Text = label == featureId ? label : $"{label} ({featureId})";

        var n = Math.Min(x.Count, values.Count);
        var finite = new List<int>(n);
        for (var i = 0; i < n; i++)
            if (double.IsFinite(x[i]) && double.IsFinite(values[i]))
                finite.Add(i);

        var subjectCount = subjects is null
            ? 0
            : finite.Select(i => i < subjects.Count ? subjects[i] : null)
                .Where(v => !string.IsNullOrEmpty(v))
                .Distinct(StringComparer.Ordinal).Count();
        var who = subjectCount > 0 ? $", {subjectCount} subjects" : string.Empty;
        SubText.Text =
            $"log2 change across {trendColumn} = {fc}   adj.P = {qs}   -   n={finite.Count}"
            + $"{who} (non-missing values)";

        var plt = DetailPlot.Plot;
        plt.Clear();

        // One faint line per subject, drawn first so the points sit on top of it.
        if (subjects is not null && subjectCount > 0)
        {
            var bySubject = finite
                .Where(i => i < subjects.Count && !string.IsNullOrEmpty(subjects[i]))
                .GroupBy(i => subjects[i], StringComparer.Ordinal);
            foreach (var group in bySubject)
            {
                var ordered = group.OrderBy(i => x[i]).ToList();
                if (ordered.Count < 2)
                    continue;
                var line = plt.Add.ScatterLine(
                    ordered.Select(i => x[i]).ToArray(),
                    ordered.Select(i => values[i]).ToArray());
                line.Color = ScottPlot.Color.FromHex("#b8c4d0").WithAlpha(0.55);
                line.LineWidth = 1;
                line.LegendText = string.Empty;
            }
        }

        var xs = finite.Select(i => x[i]).ToArray();
        var ys = finite.Select(i => values[i]).ToArray();
        foreach (var i in finite)
            _points.Add((new Coordinates(x[i], values[i]),
                Describe(i, replicates, subjects, trendColumn, x)));

        if (xs.Length > 0)
        {
            var markers = plt.Add.Markers(xs, ys);
            markers.Color = ScottPlot.Color.FromHex(AColor);
            markers.MarkerSize = 8;
            markers.LegendText = string.Empty;

            // The fitted line, reconstructed from the reported change so the drawing and the number
            // agree by construction: it passes through the feature's mean at the mean of x, and
            // rises by exactly logFc across the observed span.
            var xMin = xs.Min();
            var xMax = xs.Max();
            var span = xMax - xMin;
            if (span > 0 && double.IsFinite(logFc))
            {
                var xMean = xs.Average();
                var yMean = ys.Average();
                var slope = logFc / span;
                var fitted = plt.Add.Line(
                    xMin, yMean + slope * (xMin - xMean),
                    xMax, yMean + slope * (xMax - xMean));
                fitted.Color = ScottPlot.Color.FromHex(BColor);
                fitted.LineWidth = 2;
            }
        }

        AddHoverOverlay(plt);
        plt.XLabel(trendColumn);
        plt.YLabel("log2 abundance");
        PlotRenderer.StyleQcPlot(plt);
        DetailPlot.Refresh();
    }

    /// <summary>The hover readout for one trend point: which replicate, which subject, which x.</summary>
    private static string Describe(int i, IReadOnlyList<string> replicates,
        IReadOnlyList<string>? subjects, string trendColumn, IReadOnlyList<double> x)
    {
        var parts = new List<string>();
        if (i < replicates.Count && !string.IsNullOrEmpty(replicates[i]))
            parts.Add(replicates[i]);
        if (subjects is not null && i < subjects.Count && !string.IsNullOrEmpty(subjects[i]))
            parts.Add(subjects[i]);
        parts.Add($"{trendColumn} = {x[i].ToString("0.###", CultureInfo.InvariantCulture)}");
        return string.Join("  -  ", parts);
    }

    private static void AddBox(List<ScottPlot.Box> boxes, double position, IReadOnlyList<double> values,
        string colorHex)
    {
        if (values.Count == 0)
            return;
        var arr = new double[values.Count];
        for (var i = 0; i < values.Count; i++)
            arr[i] = values[i];

        var q1 = Stats.PercentileLinear(arr, 25);
        var med = Stats.PercentileLinear(arr, 50);
        var q3 = Stats.PercentileLinear(arr, 75);
        var iqr = q3 - q1;
        double dataMin = double.PositiveInfinity, dataMax = double.NegativeInfinity;
        foreach (var v in arr)
        {
            if (v < dataMin) dataMin = v;
            if (v > dataMax) dataMax = v;
        }

        boxes.Add(new ScottPlot.Box
        {
            Position = position,
            Width = 0.6,
            BoxMin = q1,
            BoxMiddle = med,
            BoxMax = q3,
            WhiskerMin = Math.Max(dataMin, q1 - 1.5 * iqr),
            WhiskerMax = Math.Min(dataMax, q3 + 1.5 * iqr),
            FillColor = ScottPlot.Color.FromHex(colorHex).WithAlpha((byte)90),
            LineColor = ScottPlot.Color.FromHex(colorHex),
        });
    }

    private void AddJitter(ScottPlot.Plot plt, double center, IReadOnlyList<double> values,
        IReadOnlyList<string> replicates, string colorHex, Random rng)
    {
        if (values.Count == 0)
            return;
        var xs = new double[values.Count];
        var ys = new double[values.Count];
        for (var i = 0; i < values.Count; i++)
        {
            xs[i] = center - 0.28 + rng.NextDouble() * 0.56;
            ys[i] = values[i];
            // The jittered x is what was DRAWN, so it is what the cursor has to be matched against.
            _points.Add((new Coordinates(xs[i], ys[i]),
                i < replicates.Count ? replicates[i] : string.Empty));
        }

        var m = plt.Add.Markers(xs, ys);
        m.Color = ScottPlot.Color.FromHex(colorHex);
        m.MarkerSize = 8;
    }

    /// <summary>The hidden hover readout: a ring and a label, positioned when a point is hovered.</summary>
    private void AddHoverOverlay(ScottPlot.Plot plt)
    {
        var marker = plt.Add.Marker(0, 0, MarkerShape.OpenCircle, 19, Colors.Black);
        marker.IsVisible = false;
        _hoverMarker = marker;

        var text = plt.Add.Text(" ", 0, 0);
        PlotRenderer.StyleTextLabel(text, 18, bold: true);
        text.LabelFontColor = Colors.Black;
        text.LabelBackgroundColor = Colors.White.WithAlpha(0.85);
        text.LabelAlignment = Alignment.LowerLeft;
        text.IsVisible = false;
        _hoverText = text;
    }

    /// <summary>
    /// Name the replicate under the cursor. Uses the same radius as the other panes' readouts so
    /// hovering feels the same everywhere.
    /// </summary>
    private void OnDetailPlotMouseMove(object sender, System.Windows.Input.MouseEventArgs e)
    {
        if (_hoverMarker is null || _hoverText is null || _points.Count == 0)
            return;

        var plt = DetailPlot.Plot;
        var pos = e.GetPosition(DetailPlot);
        var scale = DetailPlot.DisplayScale;
        var cursor = new Pixel(pos.X * scale, pos.Y * scale);

        var best = -1;
        var bestDistance = double.MaxValue;
        for (var i = 0; i < _points.Count; i++)
        {
            var px = plt.GetPixel(_points[i].Loc);
            double dx = px.X - cursor.X, dy = px.Y - cursor.Y;
            var d2 = dx * dx + dy * dy;
            if (d2 < bestDistance)
            {
                bestDistance = d2;
                best = i;
            }
        }

        if (best >= 0 && bestDistance <= QcPlotChrome.HoverRadiusPx * QcPlotChrome.HoverRadiusPx
                      && !string.IsNullOrEmpty(_points[best].Replicate))
        {
            var point = _points[best];
            _hoverMarker.Location = point.Loc;
            _hoverMarker.IsVisible = true;
            _hoverText.Location = point.Loc;
            _hoverText.LabelText = point.Replicate;
            _hoverText.IsVisible = true;
            DetailPlot.Refresh();
        }
        else if (_hoverMarker.IsVisible)
        {
            _hoverMarker.IsVisible = false;
            _hoverText.IsVisible = false;
            DetailPlot.Refresh();
        }
    }
}
