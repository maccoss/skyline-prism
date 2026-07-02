using System;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.Rollup;

/// <summary>
/// Consensus transition-&gt;peptide rollup, porting transition_rollup.rollup_peptide_consensus.
/// All transitions of a peptide should share the same fold-change pattern across samples; a
/// two-way-median decomposition (transition offset + sample effect) yields per-transition
/// residuals, and each transition is inverse-variance weighted so ones that deviate from the
/// consensus are down-weighted. The peptide value is the weighted sum in LINEAR space, back to
/// log2 (scaled like the sum method). Input/output LOG2.
/// </summary>
public sealed class ConsensusRollup : IRollupMethod
{
    private readonly int _minTransitions;
    private readonly double _regularization;

    public ConsensusRollup(int minTransitions, double regularization = 0.1)
    {
        _minTransitions = minTransitions;
        _regularization = regularization;
    }

    public double[] Aggregate(double[,] log2Matrix)
    {
        var nT = log2Matrix.GetLength(0);
        var nS = log2Matrix.GetLength(1);
        var result = new double[nS];
        if (nT < _minTransitions)
        {
            for (var j = 0; j < nS; j++)
                result[j] = double.NaN;
            return result;
        }

        // alpha_i: transition offsets = row medians.
        var rowMed = new double[nT];
        var rowBuf = new double[nS];
        for (var t = 0; t < nT; t++)
        {
            for (var s = 0; s < nS; s++)
                rowBuf[s] = log2Matrix[t, s];
            rowMed[t] = Stats.NanMedian(rowBuf);
        }

        // beta_j: sample effects = column medians of the row-centered matrix.
        var colMed = new double[nS];
        var colBuf = new double[nT];
        for (var s = 0; s < nS; s++)
        {
            for (var t = 0; t < nT; t++)
                colBuf[t] = log2Matrix[t, s] - rowMed[t];
            colMed[s] = Stats.NanMedian(colBuf);
        }

        // Per-transition inverse-variance weights from the residuals (NaN variance -> weight 0).
        var weights = new double[nT];
        var resBuf = new double[nS];
        for (var t = 0; t < nT; t++)
        {
            for (var s = 0; s < nS; s++)
                resBuf[s] = log2Matrix[t, s] - rowMed[t] - colMed[s];
            var v = Stats.NanVar(resBuf, ddof: 0);
            weights[t] = double.IsNaN(v) ? 0.0 : 1.0 / (v + _regularization);
        }

        // Normalize weights to sum to n_transitions (preserve sum magnitude).
        double wsum = 0;
        for (var t = 0; t < nT; t++)
            wsum += weights[t];
        if (!double.IsFinite(wsum) || wsum <= 0)
            for (var t = 0; t < nT; t++)
                weights[t] = 1.0;
        else
            for (var t = 0; t < nT; t++)
                weights[t] *= nT / wsum;

        // Weighted sum in linear space per sample.
        for (var j = 0; j < nS; j++)
        {
            double weightedSum = 0;
            var valid = 0;
            for (var t = 0; t < nT; t++)
            {
                var lv = log2Matrix[t, j];
                if (double.IsNaN(lv))
                    continue;
                weightedSum += weights[t] * Math.Pow(2, lv);
                valid++;
            }
            result[j] = valid < _minTransitions ? double.NaN : Math.Log2(Math.Max(weightedSum, 1.0));
        }
        return result;
    }
}
