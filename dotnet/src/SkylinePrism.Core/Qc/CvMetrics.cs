using System;
using System.Collections.Generic;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// Coefficient-of-variation metrics for the QC report, ported from validation.py:calc_median_cv.
/// CVs are ALWAYS computed on the LINEAR scale: per feature, cv = std(ddof=1)/mean*100 over the
/// selected sample columns, then the median CV across features. Input matrices are LOG2.
/// </summary>
public static class CvMetrics
{
    /// <summary>
    /// Median CV (%) across features for the given sample columns. <paramref name="log2Matrix"/>
    /// is [nFeatures, nSamples]; <paramref name="sampleIndices"/> selects the columns.
    /// </summary>
    public static double MedianCv(double[,] log2Matrix, IReadOnlyList<int> sampleIndices)
    {
        var nFeatures = log2Matrix.GetLength(0);
        var cvs = new List<double>(nFeatures);
        var row = new double[log2Matrix.GetLength(1)];

        for (var f = 0; f < nFeatures; f++)
        {
            for (var s = 0; s < row.Length; s++)
                row[s] = log2Matrix[f, s];
            if (TryFeatureCv(row, sampleIndices, out var cv))
                cvs.Add(cv);
        }

        return MedianOfCvs(cvs);
    }

    /// <summary>
    /// One feature's CV (%) over the given sample columns, from that feature's LOG2 row alone.
    /// Returns false for a feature the median must skip (fewer than 2 of the selected samples
    /// present). Row-local by construction, so a streaming caller that never holds the matrix can
    /// accumulate exactly the same set of CVs <see cref="MedianCv"/> would.
    /// </summary>
    public static bool TryFeatureCv(
        ReadOnlySpan<double> log2Row, IReadOnlyList<int> sampleIndices, out double cv)
    {
        // CV over the feature's NON-NaN samples (pandas std/mean skipna), not all-or-nothing:
        // features with some missing values are common and must still count.
        var linear = new List<double>(sampleIndices.Count);
        for (var k = 0; k < sampleIndices.Count; k++)
        {
            var v = log2Row[sampleIndices[k]];
            if (!double.IsNaN(v))
                linear.Add(Math.Pow(2.0, v));
        }
        if (linear.Count < 2)
        {
            cv = double.NaN;
            return false;
        }

        var arr = linear.ToArray();
        var mean = Stats.Mean(arr);
        var std = Math.Sqrt(Stats.Var(arr, ddof: 1));
        cv = std / mean * 100.0;
        return true;
    }

    /// <summary>Median of the per-feature CVs collected by <see cref="TryFeatureCv"/>.</summary>
    public static double MedianOfCvs(IReadOnlyList<double> cvs)
        => cvs.Count == 0 ? double.NaN : Stats.NanMedian(cvs.ToArray());

    /// <summary>Per-feature CV (%) over the given sample columns (linear scale). NaN-free features only.</summary>
    public static double[] PerFeatureCvs(double[,] log2Matrix, IReadOnlyList<int> sampleIndices)
    {
        var nFeatures = log2Matrix.GetLength(0);
        var cvs = new List<double>(nFeatures);
        var linear = new List<double>(sampleIndices.Count);
        for (var f = 0; f < nFeatures; f++)
        {
            // CV over the feature's non-NaN samples (pandas skipna); needs >= 2 present.
            linear.Clear();
            for (var k = 0; k < sampleIndices.Count; k++)
            {
                var v = log2Matrix[f, sampleIndices[k]];
                if (!double.IsNaN(v))
                    linear.Add(Math.Pow(2.0, v));
            }
            if (linear.Count < 2)
                continue;
            var arr = linear.ToArray();
            var mean = Stats.Mean(arr);
            var std = Math.Sqrt(Stats.Var(arr, ddof: 1));
            cvs.Add(std / mean * 100.0);
        }
        return cvs.ToArray();
    }

    /// <summary>Median CV before/after for one sample-type group (needs >= 2 samples).</summary>
    public readonly record struct BeforeAfter(double Before, double After)
    {
        public double ImprovementPercent => Before > 0 ? (Before - After) / Before * 100.0 : 0.0;
    }

    public static BeforeAfter? Compute(
        double[,] before, double[,] after, IReadOnlyList<int> sampleIndices)
    {
        if (sampleIndices.Count < 2)
            return null;
        return new BeforeAfter(MedianCv(before, sampleIndices), MedianCv(after, sampleIndices));
    }
}
