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
        var linear = new double[sampleIndices.Count];

        for (var f = 0; f < nFeatures; f++)
        {
            var anyNaN = false;
            for (var k = 0; k < sampleIndices.Count; k++)
            {
                var v = log2Matrix[f, sampleIndices[k]];
                if (double.IsNaN(v))
                {
                    anyNaN = true;
                    break;
                }
                linear[k] = Math.Pow(2.0, v);
            }
            if (anyNaN)
            {
                // pandas std/mean over a row with NaN would skip them; features with NaN are
                // rare here (imputed upstream). Skip to keep the median well-defined.
                continue;
            }

            var mean = Stats.Mean(linear);
            var std = Math.Sqrt(Stats.Var(linear, ddof: 1));
            cvs.Add(std / mean * 100.0);
        }

        return cvs.Count == 0 ? double.NaN : Stats.NanMedian(cvs.ToArray());
    }

    /// <summary>Per-feature CV (%) over the given sample columns (linear scale). NaN-free features only.</summary>
    public static double[] PerFeatureCvs(double[,] log2Matrix, IReadOnlyList<int> sampleIndices)
    {
        var nFeatures = log2Matrix.GetLength(0);
        var cvs = new List<double>(nFeatures);
        var linear = new double[sampleIndices.Count];
        for (var f = 0; f < nFeatures; f++)
        {
            var anyNaN = false;
            for (var k = 0; k < sampleIndices.Count; k++)
            {
                var v = log2Matrix[f, sampleIndices[k]];
                if (double.IsNaN(v)) { anyNaN = true; break; }
                linear[k] = Math.Pow(2.0, v);
            }
            if (anyNaN)
                continue;
            var mean = Stats.Mean(linear);
            var std = Math.Sqrt(Stats.Var(linear, ddof: 1));
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
