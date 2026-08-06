using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.Normalization;

/// <summary>
/// Global normalization expressed as per-sample factors that can be applied to one cell at a time.
/// <para>
/// Every method <see cref="Normalizer"/> implements except <c>quantile</c> is separable this way:
/// what a cell becomes depends only on its own value, its sample's statistics, and (for
/// <c>rt_lowess</c>) its feature's retention time - never on other cells in the same row. That is
/// what lets Stage 2b run without the feature x sample matrix in memory: the statistics come from a
/// column-at-a-time pass (O(rows)), and the application then happens row group by row group.
/// </para>
/// <para>
/// <c>quantile</c> is deliberately absent. Mapping a cell needs its whole column's rank
/// distribution at apply time, so doing it for every column is the full matrix again;
/// <see cref="Compute"/> returns null for it and the caller keeps the in-memory path.
/// </para>
/// </summary>
internal sealed class NormalizationFactors
{
    /// <summary>The method actually applied, which is not always the one requested - see <see cref="Compute"/>.</summary>
    public required string Method { get; init; }

    /// <summary>median: value - Offsets[sample].</summary>
    public double[]? Offsets { get; init; }

    /// <summary>vsn: asinh(VsnScale[sample] * 2^value).</summary>
    public double[]? VsnScale { get; init; }

    /// <summary>rt_lowess: per-sample fitted curve on <see cref="RtGrid"/>, null where unfittable.</summary>
    public double[]?[]? SampleCurves { get; init; }

    public double[]? RtGrid { get; init; }

    /// <summary>rt_lowess: the across-sample median curve the per-sample curves are aligned to.</summary>
    public double[]? GlobalCurve { get; init; }

    /// <summary>
    /// Normalize one cell. <paramref name="rt"/> is only read by rt_lowess and may be NaN.
    /// Mirrors <see cref="Normalizer"/> exactly, including its no-op cases (NaN in, NaN out;
    /// a value whose curve or interpolation is undefined is left alone).
    /// </summary>
    public double Apply(int sample, double value, double rt)
    {
        switch (Method)
        {
            case "vsn":
                return double.IsNaN(value) ? double.NaN : Math.Asinh(VsnScale![sample] * Math.Pow(2, value));

            case "rt_lowess":
            {
                var curve = SampleCurves![sample];
                if (curve is null || double.IsNaN(rt) || double.IsNaN(value))
                    return value;
                var sampleVal = Stats.Interp(rt, RtGrid!, curve);
                var globalVal = Stats.Interp(rt, RtGrid!, GlobalCurve!);
                if (double.IsNaN(sampleVal) || double.IsNaN(globalVal))
                    return value;
                return value - (sampleVal - globalVal);
            }

            case "none":
                return value;

            default: // median
                return value - Offsets![sample]; // NaN - x = NaN
        }
    }

    /// <summary>
    /// Compute the factors by reading <paramref name="wideParquet"/> one sample column at a time.
    /// Returns null when the requested method cannot be applied cell-wise (<c>quantile</c>).
    /// <para>
    /// All-NaN feature rows are dropped downstream, but they cannot change a median, a VSN scale or
    /// a LOWESS fit - every statistic here skips NaN - so this pass does not need the kept-row set.
    /// The one exception is the rt_lowess GRID, whose span comes from the kept rows' retention
    /// times; that is why rt_lowess costs a second pass.
    /// </para>
    /// </summary>
    public static NormalizationFactors? Compute(
        string wideParquet,
        IReadOnlyList<string> samples,
        string method,
        string? rtColumn = null,
        double rtLowessFrac = 0.3,
        int rtLowessGridPoints = 100)
    {
        if (method == "quantile")
            return null;

        using var reader = ParquetColumnReader.Open(wideParquet);

        // rt_lowess silently degrades to median when the file has no RT column - the in-memory path
        // does the same (its `rtKept is not null` guard falls through to the method switch, whose
        // default is median), and this stage's job is to match it.
        if (method == "rt_lowess" && (rtColumn is null || !reader.HasColumn(rtColumn)))
            method = "median";

        switch (method)
        {
            case "none":
                return new NormalizationFactors { Method = "none" };

            case "vsn":
            {
                var scale = new double[samples.Count];
                for (var j = 0; j < samples.Count; j++)
                {
                    var col = reader.ReadDoubles(samples[j]);
                    var positives = new List<double>(col.Length);
                    foreach (var v in col)
                        if (!double.IsNaN(v))
                            positives.Add(Math.Pow(2, v)); // 2^x is always > 0
                    var median = positives.Count > 0 ? Stats.NanMedian(positives.ToArray()) : 0.0;
                    scale[j] = median > 0 ? 1.0 / median : 1.0;
                }
                return new NormalizationFactors { Method = "vsn", VsnScale = scale };
            }

            case "rt_lowess":
                return ComputeRtLowess(reader, samples, rtColumn!, rtLowessFrac, rtLowessGridPoints);

            default:
            {
                var medians = new double[samples.Count];
                for (var j = 0; j < samples.Count; j++)
                    medians[j] = Stats.NanMedian(reader.ReadDoubles(samples[j]));
                var globalMedian = Stats.NanMedian(medians);
                var offsets = medians.Select(m => m - globalMedian).ToArray();
                return new NormalizationFactors { Method = "median", Offsets = offsets };
            }
        }
    }

    private static NormalizationFactors ComputeRtLowess(
        ParquetColumnReader reader, IReadOnlyList<string> samples, string rtColumn,
        double frac, int nGridPoints)
    {
        var nRows = reader.RowCount;

        // Pass 1: which rows survive the all-NaN filter. Only needed for the grid span, but the span
        // has to exist before any curve is fitted (it sets both the grid and the LOWESS delta), so it
        // cannot share the pass that fits them.
        var keep = new bool[nRows];
        for (var j = 0; j < samples.Count; j++)
        {
            var col = reader.ReadDoubles(samples[j]);
            for (var i = 0; i < nRows; i++)
                if (!double.IsNaN(col[i]))
                    keep[i] = true;
        }

        var rtAll = reader.ReadDoubles(rtColumn);
        double rtMin = double.PositiveInfinity, rtMax = double.NegativeInfinity;
        for (var i = 0; i < nRows; i++)
        {
            if (!keep[i] || double.IsNaN(rtAll[i]))
                continue;
            if (rtAll[i] < rtMin) rtMin = rtAll[i];
            if (rtAll[i] > rtMax) rtMax = rtAll[i];
        }
        if (rtMin > rtMax || nGridPoints < 2)
            return new NormalizationFactors { Method = "none" }; // no usable RT -> identity

        var rtGrid = new double[nGridPoints];
        var step = (rtMax - rtMin) / (nGridPoints - 1);
        for (var g = 0; g < nGridPoints; g++)
            rtGrid[g] = rtMin + g * step;

        // Pass 2: one LOWESS curve per sample, from that sample's own (rt, value) points.
        var curves = new double[]?[samples.Count];
        for (var j = 0; j < samples.Count; j++)
        {
            var col = reader.ReadDoubles(samples[j]);
            var xs = new List<double>(nRows);
            var ys = new List<double>(nRows);
            for (var i = 0; i < nRows; i++)
            {
                if (!keep[i])
                    continue;
                if (!double.IsNaN(col[i]) && !double.IsNaN(rtAll[i]))
                {
                    xs.Add(rtAll[i]);
                    ys.Add(col[i]);
                }
            }
            if (xs.Count < 20)
            {
                curves[j] = null;
                continue;
            }

            var order = Stats.ArgSort(xs.ToArray());
            var xSorted = new double[order.Length];
            var ySorted = new double[order.Length];
            for (var k = 0; k < order.Length; k++)
            {
                xSorted[k] = xs[order[k]];
                ySorted[k] = ys[order[k]];
            }

            // delta = 1% of the RT range (statsmodels' speedup, matching the Python pipeline).
            var yfit = Lowess.Fit(xSorted, ySorted, frac, delta: (rtMax - rtMin) * 0.01);
            var curve = new double[nGridPoints];
            for (var g = 0; g < nGridPoints; g++)
                curve[g] = InterpOrNaN(rtGrid[g], xSorted, yfit); // NaN outside the sample's RT range
            curves[j] = curve;
        }

        var globalCurve = new double[nGridPoints];
        var buf = new List<double>(samples.Count);
        for (var g = 0; g < nGridPoints; g++)
        {
            buf.Clear();
            foreach (var c in curves)
                if (c != null && !double.IsNaN(c[g]))
                    buf.Add(c[g]);
            globalCurve[g] = buf.Count == 0 ? double.NaN : Stats.NanMedian(buf.ToArray());
        }

        return new NormalizationFactors
        {
            Method = "rt_lowess",
            SampleCurves = curves,
            RtGrid = rtGrid,
            GlobalCurve = globalCurve,
        };
    }

    // np.interp with left/right = NaN (undefined outside the fitted range).
    private static double InterpOrNaN(double x, double[] xp, double[] fp)
    {
        var n = xp.Length;
        if (n == 0 || x < xp[0] || x > xp[n - 1])
            return double.NaN;
        return Stats.Interp(x, xp, fp);
    }
}
