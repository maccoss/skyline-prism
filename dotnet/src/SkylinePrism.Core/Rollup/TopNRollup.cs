using System;

namespace SkylinePrism.Core.Rollup;

/// <summary>
/// Transition-&gt;peptide Top-N rollup, porting transition_rollup.rollup_peptide_topn with
/// selection="intensity", weighting="sum" (the path that needs only the intensity matrix; the
/// correlation-selection / sqrt-weighting variants need the shape-correlation matrix, not yet
/// plumbed through the C# rollup). The SAME top-N transitions - highest mean LINEAR intensity
/// across samples - are used for every sample; the peptide value is log2(sum of the selected
/// transitions' linear intensities). Input/output LOG2.
/// </summary>
public sealed class TopNRollup : IRollupMethod
{
    private readonly int _n;
    private readonly int _minTransitions;

    public TopNRollup(int n, int minTransitions)
    {
        _n = n;
        _minTransitions = minTransitions;
    }

    public double[] Aggregate(double[,] log2Matrix)
    {
        var nT = log2Matrix.GetLength(0);
        var nCols = log2Matrix.GetLength(1);
        var result = new double[nCols];

        if (nT < _minTransitions)
        {
            for (var j = 0; j < nCols; j++)
                result[j] = double.NaN;
            return result;
        }

        // Score each transition by its mean LINEAR intensity across samples (NaN-skipping).
        var scores = new double[nT];
        for (var t = 0; t < nT; t++)
        {
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

        var order = new int[nT];
        for (var t = 0; t < nT; t++)
            order[t] = t;
        Array.Sort(order, (a, b) =>
        {
            var na = double.IsNaN(scores[a]);
            var nb = double.IsNaN(scores[b]);
            if (na || nb)
                return na == nb ? a.CompareTo(b) : (na ? 1 : -1); // NaN scores last
            var c = scores[b].CompareTo(scores[a]); // descending
            return c != 0 ? c : a.CompareTo(b);
        });

        var nSel = Math.Min(_n, nT);
        for (var j = 0; j < nCols; j++)
        {
            double sum = 0;
            var valid = 0;
            for (var k = 0; k < nSel; k++)
            {
                var v = log2Matrix[order[k], j];
                if (!double.IsNaN(v))
                {
                    sum += Math.Pow(2, v);
                    valid++;
                }
            }
            // Python requires at least min_transitions of the selected present in each sample.
            result[j] = valid < _minTransitions ? double.NaN : Math.Log2(Math.Max(sum, 1.0));
        }
        return result;
    }
}
