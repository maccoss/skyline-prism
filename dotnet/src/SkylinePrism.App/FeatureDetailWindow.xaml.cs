using System;
using System.Collections.Generic;
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

    public FeatureDetailWindow()
    {
        InitializeComponent();
    }

    /// <summary>Draw (or redraw) the boxplot for one feature.</summary>
    public void ShowFeature(string label, string featureId, double logFc, double q,
        string aName, IReadOnlyList<double> aValues, string bName, IReadOnlyList<double> bValues)
    {
        var inv = CultureInfo.InvariantCulture;
        var fc = double.IsFinite(logFc) ? logFc.ToString("0.###", inv) : "n/a";
        var qs = double.IsFinite(q) ? q.ToString("0.##e0", inv) : "n/a";
        HeaderText.Text = label == featureId ? label : $"{label} ({featureId})";
        SubText.Text =
            $"log2FC (B/A) = {fc}   adj.P = {qs}   -   {aName}: n={aValues.Count}, "
            + $"{bName}: n={bValues.Count} (detected, non-missing)";

        var plt = DetailPlot.Plot;
        plt.Clear();

        var boxes = new List<ScottPlot.Box>();
        AddBox(boxes, 0, aValues, AColor);
        AddBox(boxes, 1, bValues, BColor);
        if (boxes.Count > 0)
            plt.Add.Boxes(boxes);

        // Jittered per-sample points, seeded so the layout is stable across redraws of the same feature.
        var rng = new Random(HashCode.Combine(featureId, aValues.Count, bValues.Count) & 0x7fffffff);
        AddJitter(plt, 0, aValues, AColor, rng);
        AddJitter(plt, 1, bValues, BColor, rng);

        plt.Axes.Bottom.TickGenerator = new ScottPlot.TickGenerators.NumericManual(
            new double[] { 0, 1 }, new[] { aName, bName });
        plt.Axes.SetLimitsX(-0.6, 1.6);
        plt.XLabel("group");
        plt.YLabel("log2 abundance");
        PlotRenderer.StyleQcPlot(plt);
        DetailPlot.Refresh();
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

    private static void AddJitter(ScottPlot.Plot plt, double center, IReadOnlyList<double> values,
        string colorHex, Random rng)
    {
        if (values.Count == 0)
            return;
        var xs = new double[values.Count];
        var ys = new double[values.Count];
        for (var i = 0; i < values.Count; i++)
        {
            xs[i] = center - 0.28 + rng.NextDouble() * 0.56;
            ys[i] = values[i];
        }

        var m = plt.Add.Markers(xs, ys);
        m.Color = ScottPlot.Color.FromHex(colorHex);
        m.MarkerSize = 6;
    }
}
