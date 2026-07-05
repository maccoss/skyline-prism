using System;
using System.Collections.Generic;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.Normalization;

/// <summary>
/// Global (sample-level) normalization on LOG2 matrices, ported from the inline Stage 2b/4b
/// logic in cli.py. Matrices are [nFeatures, nSamples]; NaN is preserved.
/// </summary>
public static class Normalizer
{
    /// <summary>
    /// Median normalization (cli.py ~1908): for each sample column subtract
    /// (median(col) - median(all sample medians)). Equalizes every sample's median to the
    /// global median. Operates on LOG2. pandas median skips NaN.
    /// </summary>
    public static double[,] MedianNormalize(double[,] log2Matrix)
    {
        var nRows = log2Matrix.GetLength(0);
        var nCols = log2Matrix.GetLength(1);

        var sampleMedians = new double[nCols];
        var colBuf = new double[nRows];
        for (var j = 0; j < nCols; j++)
        {
            for (var i = 0; i < nRows; i++)
                colBuf[i] = log2Matrix[i, j];
            sampleMedians[j] = Stats.NanMedian(colBuf);
        }

        var globalMedian = Stats.NanMedian(sampleMedians);

        var result = new double[nRows, nCols];
        for (var j = 0; j < nCols; j++)
        {
            var normFactor = sampleMedians[j] - globalMedian;
            for (var i = 0; i < nRows; i++)
                result[i, j] = log2Matrix[i, j] - normFactor; // NaN - x = NaN
        }
        return result;
    }

    /// <summary>
    /// RT-lowess normalization (normalization.py:apply_rt_lowess_normalization). For each sample,
    /// fit a LOWESS curve of LOG2 abundance vs mean_rt, take the global median curve across
    /// samples, and subtract (sample_curve(rt) - global_curve(rt)) from each value - removing
    /// RT-dependent systematic variation (ion suppression, gradient drift). Functional parity.
    /// </summary>
    public static double[,] RtLowessNormalize(
        double[,] log2Matrix, double[] meanRt, double frac = 0.3, int nGridPoints = 100)
    {
        var nRows = log2Matrix.GetLength(0);
        var nCols = log2Matrix.GetLength(1);
        var result = (double[,])log2Matrix.Clone();

        double rtMin = double.PositiveInfinity, rtMax = double.NegativeInfinity;
        foreach (var rt in meanRt)
        {
            if (double.IsNaN(rt))
                continue;
            if (rt < rtMin) rtMin = rt;
            if (rt > rtMax) rtMax = rt;
        }
        if (rtMin > rtMax || nGridPoints < 2)
            return result; // no usable RT

        var rtGrid = new double[nGridPoints];
        var step = (rtMax - rtMin) / (nGridPoints - 1);
        for (var g = 0; g < nGridPoints; g++)
            rtGrid[g] = rtMin + g * step;

        // Per-sample LOWESS curve on the RT grid (null when too few points).
        var sampleCurves = new double[nCols][];
        for (var s = 0; s < nCols; s++)
        {
            var xs = new List<double>(nRows);
            var ys = new List<double>(nRows);
            for (var i = 0; i < nRows; i++)
            {
                var v = log2Matrix[i, s];
                var rt = meanRt[i];
                if (!double.IsNaN(v) && !double.IsNaN(rt))
                {
                    xs.Add(rt);
                    ys.Add(v);
                }
            }
            if (xs.Count < 20)
            {
                sampleCurves[s] = null!;
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

            // delta = 1% of the RT range (statsmodels' speedup, matching the Python pipeline): fit
            // anchor points and interpolate between them instead of fitting every peptide.
            var yfit = Lowess.Fit(xSorted, ySorted, frac, delta: (rtMax - rtMin) * 0.01);
            var curve = new double[nGridPoints];
            for (var g = 0; g < nGridPoints; g++)
                curve[g] = InterpOrNaN(rtGrid[g], xSorted, yfit); // NaN outside the sample's RT range
            sampleCurves[s] = curve;
        }

        // Global median curve across samples at each grid point.
        var globalCurve = new double[nGridPoints];
        var buf = new List<double>(nCols);
        for (var g = 0; g < nGridPoints; g++)
        {
            buf.Clear();
            for (var s = 0; s < nCols; s++)
            {
                var c = sampleCurves[s];
                if (c != null && !double.IsNaN(c[g]))
                    buf.Add(c[g]);
            }
            globalCurve[g] = buf.Count == 0 ? double.NaN : Stats.NanMedian(buf.ToArray());
        }

        // Correction = sample_curve(rt) - global_curve(rt), subtracted from each value.
        for (var s = 0; s < nCols; s++)
        {
            var curve = sampleCurves[s];
            if (curve == null)
                continue;
            for (var i = 0; i < nRows; i++)
            {
                var rt = meanRt[i];
                if (double.IsNaN(rt) || double.IsNaN(log2Matrix[i, s]))
                    continue;
                var sampleVal = Stats.Interp(rt, rtGrid, curve);
                var globalVal = Stats.Interp(rt, rtGrid, globalCurve);
                if (double.IsNaN(sampleVal) || double.IsNaN(globalVal))
                    continue;
                result[i, s] = log2Matrix[i, s] - (sampleVal - globalVal);
            }
        }
        return result;
    }

    // np.interp with left/right = NaN (undefined outside the fitted range).
    private static double InterpOrNaN(double x, double[] xp, double[] fp)
    {
        var n = xp.Length;
        if (n == 0 || x < xp[0] || x > xp[n - 1])
            return double.NaN;
        return Stats.Interp(x, xp, fp);
    }

    /// <summary>
    /// Quantile normalization (cli.py Stage 2b): force identical distributions across samples.
    /// Reference = per-rank mean of the sorted columns; each value is mapped to the reference by
    /// interpolating its average rank. Operates on LOG2; NaN preserved.
    /// </summary>
    public static double[,] QuantileNormalize(double[,] log2Matrix)
    {
        var nRows = log2Matrix.GetLength(0);
        var nCols = log2Matrix.GetLength(1);

        var ranks = new double[nRows, nCols];
        var sortedCols = new double[nCols][];
        for (var j = 0; j < nCols; j++)
        {
            var validIdx = new List<int>(nRows);
            for (var i = 0; i < nRows; i++)
                if (!double.IsNaN(log2Matrix[i, j]))
                    validIdx.Add(i);

            var validVals = new double[validIdx.Count];
            for (var k = 0; k < validIdx.Count; k++)
                validVals[k] = log2Matrix[validIdx[k], j];
            var r = Stats.RankAverage(validVals); // 1-based average ranks over valid entries
            for (var i = 0; i < nRows; i++)
                ranks[i, j] = double.NaN;
            for (var k = 0; k < validIdx.Count; k++)
                ranks[validIdx[k], j] = r[k];

            var col = new double[nRows];
            for (var i = 0; i < nRows; i++)
                col[i] = log2Matrix[i, j];
            Array.Sort(col, NanLast); // ascending, NaN last (np.sort)
            sortedCols[j] = col;
        }

        var reference = new double[nRows];
        for (var i = 0; i < nRows; i++)
        {
            double sum = 0;
            var cnt = 0;
            for (var j = 0; j < nCols; j++)
            {
                var v = sortedCols[j][i];
                if (!double.IsNaN(v))
                {
                    sum += v;
                    cnt++;
                }
            }
            reference[i] = cnt > 0 ? sum / cnt : double.NaN;
        }

        var result = new double[nRows, nCols];
        for (var j = 0; j < nCols; j++)
        {
            var maxRank = 0;
            for (var i = 0; i < nRows; i++)
                if (!double.IsNaN(log2Matrix[i, j]))
                    maxRank++;
            var xp = new double[maxRank];
            var fp = new double[maxRank];
            for (var k = 0; k < maxRank; k++)
            {
                xp[k] = k + 1;
                fp[k] = reference[k];
            }
            for (var i = 0; i < nRows; i++)
                result[i, j] = double.IsNaN(log2Matrix[i, j]) || maxRank == 0
                    ? double.NaN
                    : Stats.Interp(ranks[i, j], xp, fp);
        }
        return result;
    }

    /// <summary>
    /// VSN (variance-stabilizing normalization) via arcsinh (cli.py Stage 2b): per sample,
    /// arcsinh(linear / median(linear)). Input LOG2 -> linear internally; OUTPUT IS ON THE
    /// ARCSINH SCALE, not log2 (negative values are expected). Parameter optimization is off by
    /// default in the pipeline and is not ported.
    /// </summary>
    public static double[,] VsnNormalize(double[,] log2Matrix)
    {
        var nRows = log2Matrix.GetLength(0);
        var nCols = log2Matrix.GetLength(1);
        var result = new double[nRows, nCols];
        for (var j = 0; j < nCols; j++)
        {
            var pos = new List<double>(nRows);
            for (var i = 0; i < nRows; i++)
            {
                var v = log2Matrix[i, j];
                if (!double.IsNaN(v))
                    pos.Add(Math.Pow(2, v)); // 2^x is always > 0
            }
            var median = pos.Count > 0 ? Stats.NanMedian(pos.ToArray()) : 0.0;
            var a = median > 0 ? 1.0 / median : 1.0;
            for (var i = 0; i < nRows; i++)
            {
                var v = log2Matrix[i, j];
                result[i, j] = double.IsNaN(v) ? double.NaN : Math.Asinh(a * Math.Pow(2, v));
            }
        }
        return result;
    }

    private static readonly Comparison<double> NanLast = (a, b) =>
    {
        if (double.IsNaN(a))
            return double.IsNaN(b) ? 0 : 1;
        if (double.IsNaN(b))
            return -1;
        return a.CompareTo(b);
    };
}
