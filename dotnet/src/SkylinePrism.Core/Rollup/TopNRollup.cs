using System;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.Rollup;

/// <summary>
/// Transition-&gt;peptide Top-N rollup, porting transition_rollup.rollup_peptide_topn. The SAME
/// top-N transitions are used for every sample, selected either by mean LINEAR intensity
/// ("intensity") or by median shape correlation across samples ("correlation", which needs the
/// shape-correlation matrix). The selected transitions are combined by a weighted linear sum
/// ("sum" = equal weights; "sqrt" = weighted by sqrt of mean intensity), back to log2.
/// Input/output LOG2.
/// </summary>
public sealed class TopNRollup : IRollupMethod
{
    private readonly int _n;
    private readonly int _minTransitions;
    private readonly string _selection;
    private readonly string _weighting;

    public TopNRollup(int n, int minTransitions, string selection = "intensity", string weighting = "sum")
    {
        _n = n;
        _minTransitions = minTransitions;
        _selection = selection;
        _weighting = weighting;
    }

    /// <summary>IRollupMethod path (intensity selection; correlation needs the shape matrix).</summary>
    public double[] Aggregate(double[,] log2Matrix)
        => Compute(log2Matrix, null, _n, _minTransitions, _selection, _weighting);

    public static double[] Compute(
        double[,] log2Matrix, double[,]? shapeCorr, int n, int minTransitions, string selection, string weighting)
    {
        var nT = log2Matrix.GetLength(0);
        var nCols = log2Matrix.GetLength(1);
        var result = new double[nCols];
        if (nT < minTransitions)
        {
            for (var j = 0; j < nCols; j++)
                result[j] = double.NaN;
            return result;
        }

        var useCorr = selection == "correlation" && shapeCorr is not null;
        var scores = new double[nT];
        var buf = new double[nCols];
        for (var t = 0; t < nT; t++)
        {
            if (useCorr)
            {
                // median shape correlation across samples (missing filled with 0 upstream).
                for (var j = 0; j < nCols; j++)
                    buf[j] = shapeCorr![t, j];
                scores[t] = Stats.NanMedian(buf);
            }
            else
            {
                // mean LINEAR intensity across samples.
                double sum = 0;
                var cnt = 0;
                for (var j = 0; j < nCols; j++)
                {
                    var v = log2Matrix[t, j];
                    if (!double.IsNaN(v))
                    {
                        sum += Math.Pow(2, v);
                        cnt++;
                    }
                }
                scores[t] = cnt > 0 ? sum / cnt : double.NaN;
            }
        }

        var order = new int[nT];
        for (var t = 0; t < nT; t++)
            order[t] = t;
        Array.Sort(order, (a, b) =>
        {
            var na = double.IsNaN(scores[a]);
            var nb = double.IsNaN(scores[b]);
            if (na || nb)
                return na == nb ? a.CompareTo(b) : (na ? 1 : -1);
            var c = scores[b].CompareTo(scores[a]); // descending
            return c != 0 ? c : a.CompareTo(b);
        });

        var nSel = Math.Min(n, nT);
        var weights = new double[nSel];
        if (weighting == "sqrt")
        {
            for (var k = 0; k < nSel; k++)
            {
                double sum = 0;
                var cnt = 0;
                for (var j = 0; j < nCols; j++)
                {
                    var v = log2Matrix[order[k], j];
                    if (!double.IsNaN(v)) { sum += Math.Pow(2, v); cnt++; }
                }
                var meanInt = cnt > 0 ? sum / cnt : 1.0;
                weights[k] = Math.Sqrt(Math.Max(meanInt, 1.0));
            }
        }
        else
        {
            for (var k = 0; k < nSel; k++)
                weights[k] = 1.0;
        }

        // Normalize weights to sum to n_select (preserve sum magnitude).
        double wsum = 0;
        for (var k = 0; k < nSel; k++)
            wsum += weights[k];
        if (!double.IsFinite(wsum) || wsum <= 0)
            for (var k = 0; k < nSel; k++)
                weights[k] = 1.0;
        else
            for (var k = 0; k < nSel; k++)
                weights[k] *= nSel / wsum;

        for (var j = 0; j < nCols; j++)
        {
            double weightedSum = 0;
            var valid = 0;
            for (var k = 0; k < nSel; k++)
            {
                var v = log2Matrix[order[k], j];
                if (double.IsNaN(v))
                    continue;
                weightedSum += weights[k] * Math.Pow(2, v);
                valid++;
            }
            result[j] = valid < minTransitions ? double.NaN : Math.Log2(Math.Max(weightedSum, 1.0));
        }
        return result;
    }
}
