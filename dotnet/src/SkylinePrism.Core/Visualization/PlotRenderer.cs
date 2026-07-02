using System;
using System.Collections.Generic;
using System.Linq;
using ScottPlot;

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
    private const int Width = 700;
    private const int Height = 500;

    public static readonly IReadOnlyDictionary<string, string> TypeColors = new Dictionary<string, string>
    {
        ["experimental"] = "#1f77b4",
        ["qc"] = "#ff7f0e",
        ["reference"] = "#d62728",
        ["unknown"] = "#7f7f7f",
    };

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
            med.LineWidth = 2;
            med.LinePattern = LinePattern.Dashed;
            med.LegendText = $"median {medianCv:0.0}%";
            plt.ShowLegend();
        }
        plt.Title(title);
        plt.XLabel("CV (%)");
        plt.YLabel("Count");
        return plt.GetImageBytes(Width, Height, ImageFormat.Png);
    }

    /// <summary>
    /// Overlaid before/after CV histograms (before in grey, after in colour) with both median
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
            line.LineWidth = 2;
            line.LinePattern = LinePattern.Dashed;
            line.LegendText = $"{label} median {med:0.0}%";
        }

        AddHist(beforeCvs, Color.FromHex("#7f7f7f"), "Before");
        AddHist(afterCvs, Color.FromHex(afterColorHex), "After");
        plt.ShowLegend();
        plt.Title(title);
        plt.XLabel("CV (%)");
        plt.YLabel("Count");
        return plt.GetImageBytes(Width, Height, ImageFormat.Png);
    }

    /// <summary>2-D PCA scatter coloured by sample type.</summary>
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
            markers.MarkerSize = 7;
            markers.LegendText = type;
        }
        plt.ShowLegend();
        plt.Title(title);
        plt.XLabel("PC1");
        plt.YLabel("PC2");
        return plt.GetImageBytes(Width, Height, ImageFormat.Png);
    }

    /// <summary>Box-like distribution of per-sample median LOG2 intensity, grouped by sample type.</summary>
    public static byte[] IntensityDistribution(
        double[,] log2Matrix, IReadOnlyList<string> sampleTypes, string title)
    {
        var nFeatures = log2Matrix.GetLength(0);
        var nSamples = log2Matrix.GetLength(1);

        var plt = new Plot();
        var xs = new List<double>();
        var ys = new List<double>();
        var colors = new List<Color>();
        var buf = new double[nFeatures];
        for (var j = 0; j < nSamples; j++)
        {
            var n = 0;
            for (var f = 0; f < nFeatures; f++)
            {
                var v = log2Matrix[f, j];
                if (!double.IsNaN(v))
                    buf[n++] = v;
            }
            if (n == 0)
                continue;
            var med = Numerics.Stats.NanMedian(buf.AsSpan(0, n));
            xs.Add(j);
            ys.Add(med);
            colors.Add(Color.FromHex(TypeColors.GetValueOrDefault(sampleTypes[j], "#7f7f7f")));
        }

        var markers = plt.Add.Markers(xs.ToArray(), ys.ToArray());
        markers.MarkerSize = 4;
        plt.Title(title);
        plt.XLabel("Sample index");
        plt.YLabel("Median log2 intensity");
        return plt.GetImageBytes(Width, Height, ImageFormat.Png);
    }
}
