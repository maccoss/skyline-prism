using System;
using System.Collections.Generic;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.Normalization;

/// <summary>
/// One-sided (low-signal only) sample outlier detection, ported from cli.py Stage 2b
/// (lines 1704-1783). Operates on a LOG2 matrix but computes all statistics on the LINEAR
/// scale (per-sample median of 2^value over non-NaN cells), matching the Python code.
/// </summary>
public static class OutlierDetector
{
    public enum Method { Iqr, FoldMedian }

    public sealed record Result(IReadOnlyList<string> Outliers, double OverallMedian);

    /// <summary>
    /// Returns the sample names flagged as low-signal outliers. The caller decides whether
    /// to report (keep) or exclude them.
    /// </summary>
    public static Result Detect(
        double[,] log2Matrix,
        IReadOnlyList<string> samples,
        Method method = Method.Iqr,
        double iqrMultiplier = 1.5,
        double foldThreshold = 0.1)
    {
        var nRows = log2Matrix.GetLength(0);
        var nCols = log2Matrix.GetLength(1);

        // Per-sample LINEAR median (drop NaN, convert to 2^, median).
        var linearMedians = new double[nCols];
        var buf = new List<double>(nRows);
        for (var j = 0; j < nCols; j++)
        {
            buf.Clear();
            for (var i = 0; i < nRows; i++)
            {
                var v = log2Matrix[i, j];
                if (!double.IsNaN(v))
                    buf.Add(Math.Pow(2.0, v));
            }
            linearMedians[j] = buf.Count == 0 ? double.NaN : Stats.NanMedian(buf.ToArray());
        }

        var overallMedian = Stats.NanMedian(linearMedians);

        double lowerBound;
        if (method == Method.Iqr)
        {
            var q1 = Stats.PercentileLinear(linearMedians, 25);
            var q3 = Stats.PercentileLinear(linearMedians, 75);
            var iqr = q3 - q1;
            lowerBound = q1 - iqrMultiplier * iqr;
        }
        else
        {
            lowerBound = foldThreshold * overallMedian;
        }

        var outliers = new List<string>();
        for (var j = 0; j < nCols; j++)
        {
            if (linearMedians[j] < lowerBound)
                outliers.Add(samples[j]);
        }
        return new Result(outliers, overallMedian);
    }
}
