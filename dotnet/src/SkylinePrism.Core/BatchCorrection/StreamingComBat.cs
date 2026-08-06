using System;
using System.Collections.Generic;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.BatchCorrection;

/// <summary>
/// Per-(batch, feature) summary of the standardized data, and the per-feature terms needed to
/// standardize and un-standardize it. Everything <see cref="ComBat"/>'s empirical-Bayes stage does
/// can be driven from this, so the standardized matrix itself never has to exist.
/// <para>
/// Sized O(features x batches), not O(features x samples): ~10 MB at 100k features x 10 batches,
/// against gigabytes for the matrix.
/// </para>
/// </summary>
internal sealed class ComBatSufficientStats
{
    public required IReadOnlyList<List<int>> Batches { get; init; }

    /// <summary>Number of active (correctable) features summarized.</summary>
    public required int NFeatures { get; init; }

    /// <summary>Per feature: the across-batch weighted mean that standardization subtracts.</summary>
    public required double[] GrandMean { get; init; }

    /// <summary>Per feature: sqrt(var_pooled), the scale standardization divides by.</summary>
    public required double[] StdPooled { get; init; }

    /// <summary>
    /// [batch][feature]: how many of the batch's samples OBSERVED this feature. sva's per-row
    /// <c>n &lt;- rowSums(!is.na(sdat))</c>; equals the batch size when nothing is missing.
    /// </summary>
    public required int[][] Counts { get; init; }

    /// <summary>[batch][feature]: mean of the batch's OBSERVED standardized values.</summary>
    public required double[][] GammaHat { get; init; }

    /// <summary>
    /// [batch][feature]: sum of squared deviations of those observed values about
    /// <see cref="GammaHat"/> - exactly the numerator <c>Stats.Var(row, ddof: 1)</c> forms.
    /// </summary>
    public required double[][] SumSq { get; init; }

    /// <summary>
    /// The scale effect the iteration starts from, and whether the data supports it at all. Fewer
    /// than 2 observations, or no spread among them, means there is no scale to estimate: the
    /// placeholder 1.0 is returned but flagged, so it is neither corrected by nor fed into the prior.
    /// </summary>
    public (double[] DeltaHat, bool[] Estimable) DeltaHat(int batch)
    {
        var counts = Counts[batch];
        var sumSq = SumSq[batch];
        var delta = new double[NFeatures];
        var estimable = new bool[NFeatures];
        for (var f = 0; f < NFeatures; f++)
        {
            if (counts[f] < 2)
            {
                delta[f] = 1.0;
                continue;
            }
            // Same division Stats.Var(..., ddof: 1) performs, on the same numerator.
            var v = sumSq[f] / (counts[f] - 1);
            if (!ComBat.IsSpreadResolvable(v, GammaHat[batch][f]))
            {
                delta[f] = 1.0;
                continue;
            }
            delta[f] = v;
            estimable[f] = true;
        }
        return (delta, estimable);
    }
}

/// <summary>
/// ComBat's empirical-Bayes stage (priors + the <c>ItSol</c> fixed point), run against
/// <see cref="ComBatSufficientStats"/> instead of the standardized matrix.
/// <para>
/// The only thing <c>ItSol</c> needs the matrix for is, per feature and batch,
/// <c>sum over the batch's observed samples of (sData - gammaStar)^2</c> - and that is a shift of
/// the sum of squares about the batch mean:
/// </para>
/// <code>
/// sum (x_k - g)^2  =  sum (x_k - m)^2  +  n * (m - g)^2      where m = mean(x) = gammaHat
/// </code>
/// <para>
/// So <c>SumSq</c>, <c>GammaHat</c> and <c>Counts</c> are sufficient. Note the identity is used in
/// this form deliberately rather than the algebraically equivalent <c>S2 - 2*g*S1 + n*g^2</c>: for a
/// batch whose standardized values sit far from zero the latter subtracts two large nearly-equal
/// quantities and loses most of its significant digits, while the form above only ever adds two
/// terms of the same order as the answer.
/// </para>
/// </summary>
internal static class StreamingComBat
{
    /// <summary>
    /// Shrunken batch effects, matching <c>ComBat.Run</c>'s parametric, not-mean-only branch (the
    /// only one the pipeline uses). Returns [batch][feature] arrays, and the number of (batch,
    /// feature) scales the data did not support.
    /// </summary>
    public static (double[][] GammaStar, double[][] DeltaStar, long UnestimableScales) Estimate(
        ComBatSufficientStats stats)
    {
        var nBatch = stats.Batches.Count;
        var nf = stats.NFeatures;

        var gammaStar = new double[nBatch][];
        var deltaStar = new double[nBatch][];
        long unestimable = 0;

        for (var i = 0; i < nBatch; i++)
        {
            var gammaHat = stats.GammaHat[i];
            var (deltaHat, estimable) = stats.DeltaHat(i);
            foreach (var ok in estimable)
                if (!ok)
                    unestimable++;

            // --- _compute_priors (per batch, across features) ---
            var gammaBar = Stats.Mean(gammaHat);
            var t2 = Stats.Var(gammaHat, ddof: 1);

            // Only the deltas the data supports - a placeholder 1.0 in here would bias the
            // shrinkage of every feature in the batch.
            var kept = new double[nf];
            var nKept = 0;
            for (var f = 0; f < nf; f++)
                if (estimable[f])
                    kept[nKept++] = deltaHat[f];
            var m = Stats.Mean(kept.AsSpan(0, nKept));
            var v = Stats.Var(kept.AsSpan(0, nKept), ddof: 1);

            double aPrior, bPrior;
            if (v > 0 && m > 0)
            {
                aPrior = (m * m / v) + 2;
                bPrior = m * ((m * m / v) + 1);
            }
            else
            {
                aPrior = 1.0;
                bPrior = 1.0;
            }

            var (g, d) = ItSol(
                stats.Counts[i], gammaHat, deltaHat, stats.SumSq[i], estimable,
                gammaBar, t2, aPrior, bPrior, nf);
            gammaStar[i] = g;
            deltaStar[i] = d;
        }

        return (gammaStar, deltaStar, unestimable);
    }

    /// <summary>
    /// <c>ComBat.ItSol</c> with the residual sum of squares reconstructed from the sufficient
    /// statistics. Same starting values, same per-feature observation count, same convergence test,
    /// same iteration cap.
    /// </summary>
    private static (double[] GammaStar, double[] DeltaStar) ItSol(
        int[] counts, double[] gammaHat, double[] deltaHat, double[] sumSq, bool[] estimable,
        double gBar, double t2, double a, double b, int nf,
        double conv = 1e-4, int maxIter = 100)
    {
        var gOld = (double[])gammaHat.Clone();
        var dOld = (double[])deltaHat.Clone();
        var gNew = new double[nf];
        var dNew = new double[nf];

        for (var iter = 0; iter < maxIter; iter++)
        {
            for (var f = 0; f < nf; f++)
                gNew[f] = ComBat.PostMean(gammaHat[f], gBar, counts[f], dOld[f], t2);

            for (var f = 0; f < nf; f++)
            {
                if (!estimable[f])
                {
                    dNew[f] = 1.0; // scale not supported by the data - leave this one unscaled
                    continue;
                }
                var shift = gammaHat[f] - gNew[f];
                dNew[f] = ComBat.PostVar(sumSq[f] + counts[f] * shift * shift, counts[f], a, b);
            }

            var gChange = ComBat.MaxRelChange(gNew, gOld);
            var dChange = ComBat.MaxRelChange(dNew, dOld);
            if (gChange < conv && dChange < conv)
                break;

            Array.Copy(gNew, gOld, nf);
            Array.Copy(dNew, dOld, nf);
        }

        return (gNew, dNew);
    }
}
