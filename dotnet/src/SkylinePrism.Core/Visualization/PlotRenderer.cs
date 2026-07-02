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

    /// <summary>
    /// Correlation heatmap among the given (control) sample columns - the "did batch correction
    /// tighten the controls" diagnostic (Pearson over features present in both samples).
    /// </summary>
    public static byte[] CorrelationHeatmap(double[,] featuresBySamples, IReadOnlyList<int> cols, string title)
    {
        var n = cols.Count;
        var corr = new double[n, n];
        for (var a = 0; a < n; a++)
            for (var b = 0; b < n; b++)
                corr[a, b] = a == b ? 1.0 : Pearson(featuresBySamples, cols[a], cols[b]);

        var plt = new Plot();
        var hm = plt.Add.Heatmap(corr);
        hm.Colormap = new ScottPlot.Colormaps.Turbo();
        plt.Add.ColorBar(hm);
        plt.Title(title);
        plt.XLabel("Control sample");
        plt.YLabel("Control sample");
        return plt.GetImageBytes(Width, Height, ImageFormat.Png);
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
    /// Per-sample LOWESS curves of LOG2 abundance vs mean RT, coloured by sample type - the
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
                var yfit = Numerics.Lowess.Fit(xSorted, ySorted, 0.3);

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
                sc.LineWidth = 1;
                sc.Color = Color.FromHex(TypeColors.GetValueOrDefault(type, "#7f7f7f")).WithAlpha((byte)110);
                if (seenType.Add(type))
                    sc.LegendText = type;
            }
            plt.ShowLegend();
        }

        plt.Title(title);
        plt.XLabel("Retention time");
        plt.YLabel("log2 abundance");
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
                    Fill = new ScottPlot.FillStyle { Color = fill },
                });
            }
            if (boxes.Count > 0)
                plt.Add.Boxes(boxes);
        }

        plt.Title(title);
        plt.XLabel("RT bin");
        plt.YLabel("log2 abundance");
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
