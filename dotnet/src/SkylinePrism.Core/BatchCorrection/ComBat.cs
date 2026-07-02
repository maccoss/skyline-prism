using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.BatchCorrection;

/// <summary>
/// ComBat empirical-Bayes batch correction (Johnson et al. 2007), ported 1:1 from
/// batch_correction.py:combat and its helpers. Standard path only: one-hot batch design,
/// no covariates, no reference batch (the pipeline's reference_anchored=false case for
/// both the peptide Stage 2c and protein Stage 4c corrections). Operates on a LOG2
/// [nFeatures, nSamples] matrix. This branch is deterministic, so it is an exact-parity
/// target (~1e-9).
/// </summary>
public static class ComBat
{
    /// <summary>
    /// Batch-correct <paramref name="data"/> (features x samples) given per-sample batch
    /// labels. Returns a new corrected matrix on the same scale.
    /// </summary>
    public static double[,] Run(double[,] data, IReadOnlyList<string> batchLabels,
        bool parPrior = true, bool meanOnly = false)
    {
        var nFeatures = data.GetLength(0);
        var nSamples = data.GetLength(1);
        if (batchLabels.Count != nSamples)
            throw new ArgumentException("batchLabels length must equal number of samples.");

        // Zero-variance features (np.var ddof=0 across samples) are held out and restored.
        var zeroVar = new bool[nFeatures];
        var activeRows = new List<int>(nFeatures);
        for (var f = 0; f < nFeatures; f++)
        {
            if (RowVarPopulation(data, f, nSamples) == 0.0)
                zeroVar[f] = true;
            else
                activeRows.Add(f);
        }

        var nf = activeRows.Count;
        // Active data submatrix [nf, nSamples].
        var d = new double[nf, nSamples];
        for (var a = 0; a < nf; a++)
            for (var s = 0; s < nSamples; s++)
                d[a, s] = data[activeRows[a], s];

        // Batches: sorted unique labels (np.unique), sample indices per batch.
        var uniqueBatches = batchLabels.Distinct().OrderBy(x => x, StringComparer.Ordinal).ToList();
        var nBatch = uniqueBatches.Count;
        var batchOf = uniqueBatches.Select((b, i) => (b, i)).ToDictionary(x => x.b, x => x.i, StringComparer.Ordinal);
        var batches = new List<int>[nBatch];
        for (var i = 0; i < nBatch; i++)
            batches[i] = new List<int>();
        for (var s = 0; s < nSamples; s++)
            batches[batchOf[batchLabels[s]]].Add(s);

        if (batches.Any(b => b.Count == 1))
            meanOnly = true;

        // --- _calculate_mean_var ---
        // One-hot design: XtX = diag(batch sizes); B_hat[i,f] = batch-i mean of feature f.
        var bHat = new double[nBatch, nf];
        for (var i = 0; i < nBatch; i++)
        {
            var idx = batches[i];
            var buf = new double[idx.Count];
            for (var f = 0; f < nf; f++)
            {
                for (var k = 0; k < idx.Count; k++)
                    buf[k] = d[f, idx[k]];
                bHat[i, f] = NumpyMath.PairwiseSum(buf) / idx.Count;
            }
        }

        var grandMean = new double[nf];
        for (var f = 0; f < nf; f++)
        {
            double gm = 0.0;
            for (var i = 0; i < nBatch; i++)
                gm += ((double)batches[i].Count / nSamples) * bHat[i, f];
            grandMean[f] = gm;
        }

        // Residuals = data.T - design@B_hat; predicted[s,f] = bHat[batch(s), f].
        // var_pooled[f] = var over samples of residual (ddof=1), zero -> median of positives.
        var varPooled = new double[nf];
        for (var f = 0; f < nf; f++)
        {
            var res = new double[nSamples];
            for (var i = 0; i < nBatch; i++)
                foreach (var s in batches[i])
                    res[s] = d[f, s] - bHat[i, f];
            varPooled[f] = Stats.Var(res, ddof: 1);
        }
        ReplaceZeroWithMedianOfPositive(varPooled);

        // --- _standardize_data --- (no covariates => stand_mean[f,s] = grandMean[f])
        var sData = new double[nf, nSamples];
        var stdPooled = new double[nf];
        for (var f = 0; f < nf; f++)
        {
            stdPooled[f] = Math.Sqrt(varPooled[f]);
            for (var s = 0; s < nSamples; s++)
                sData[f, s] = (d[f, s] - grandMean[f]) / stdPooled[f];
        }

        // --- _fit_batch_effects ---
        // gamma_hat[i,f] = mean over batch-i samples of sData[f,s] (one-hot solve).
        var gammaHat = new double[nBatch, nf];
        for (var i = 0; i < nBatch; i++)
        {
            var idx = batches[i];
            var buf = new double[idx.Count];
            for (var f = 0; f < nf; f++)
            {
                for (var k = 0; k < idx.Count; k++)
                    buf[k] = sData[f, idx[k]];
                gammaHat[i, f] = NumpyMath.PairwiseSum(buf) / idx.Count;
            }
        }

        var deltaHat = new double[nBatch, nf];
        for (var i = 0; i < nBatch; i++)
        {
            var idx = batches[i];
            for (var f = 0; f < nf; f++)
            {
                if (meanOnly)
                {
                    deltaHat[i, f] = 1.0;
                }
                else
                {
                    var row = new double[idx.Count];
                    for (var k = 0; k < idx.Count; k++)
                        row[k] = sData[f, idx[k]];
                    var v = Stats.Var(row, ddof: 1);
                    deltaHat[i, f] = v == 0.0 ? 1.0 : v;
                }
            }
        }

        // --- _compute_priors ---
        var gammaBar = new double[nBatch];
        var t2 = new double[nBatch];
        var aPrior = new double[nBatch];
        var bPrior = new double[nBatch];
        for (var i = 0; i < nBatch; i++)
        {
            var gRow = Row(gammaHat, i, nf);
            gammaBar[i] = Stats.Mean(gRow);
            t2[i] = Stats.Var(gRow, ddof: 1);
            if (meanOnly)
            {
                aPrior[i] = 1.0;
                bPrior[i] = 1.0;
            }
            else
            {
                var dRow = Row(deltaHat, i, nf);
                var m = Stats.Mean(dRow);
                var v = Stats.Var(dRow, ddof: 1);
                if (v > 0 && m > 0)
                {
                    aPrior[i] = (m * m / v) + 2;
                    bPrior[i] = m * ((m * m / v) + 1);
                }
                else
                {
                    aPrior[i] = 1.0;
                    bPrior[i] = 1.0;
                }
            }
        }

        // --- EB estimation ---
        var gammaStar = new double[nBatch, nf];
        var deltaStar = new double[nBatch, nf];
        for (var i = 0; i < nBatch; i++)
        {
            if (parPrior)
            {
                if (meanOnly)
                {
                    var n = batches[i].Count;
                    for (var f = 0; f < nf; f++)
                    {
                        gammaStar[i, f] = PostMean(gammaHat[i, f], gammaBar[i], n, 1.0, t2[i]);
                        deltaStar[i, f] = 1.0;
                    }
                }
                else
                {
                    ItSol(sData, batches[i], i, gammaHat, deltaHat, gammaBar[i], t2[i], aPrior[i], bPrior[i],
                        gammaStar, deltaStar, nf);
                }
            }
            else
            {
                IntEprior(i, gammaHat, deltaHat, gammaStar, deltaStar, nf, meanOnly);
            }
        }

        // --- _adjust_data ---
        var bayes = (double[,])sData.Clone();
        for (var i = 0; i < nBatch; i++)
        {
            foreach (var s in batches[i])
                for (var f = 0; f < nf; f++)
                    bayes[f, s] = (bayes[f, s] - gammaStar[i, f]) / Math.Sqrt(deltaStar[i, f]);
        }
        for (var f = 0; f < nf; f++)
            for (var s = 0; s < nSamples; s++)
                bayes[f, s] = bayes[f, s] * stdPooled[f] + grandMean[f];

        // Scatter active rows back; zero-variance features unchanged.
        var result = new double[nFeatures, nSamples];
        for (var f = 0; f < nFeatures; f++)
            if (zeroVar[f])
                for (var s = 0; s < nSamples; s++)
                    result[f, s] = data[f, s];
        for (var a = 0; a < nf; a++)
            for (var s = 0; s < nSamples; s++)
                result[activeRows[a], s] = bayes[a, s];

        return result;
    }

    internal static void ItSol(
        double[,] sData, List<int> batchIdx, int i,
        double[,] gammaHat, double[,] deltaHat, double gBar, double t2, double a, double b,
        double[,] gammaStar, double[,] deltaStar, int nf,
        double conv = 1e-4, int maxIter = 100)
    {
        var n = batchIdx.Count;
        var gHat = Row(gammaHat, i, nf);
        var gOld = (double[])gHat.Clone();
        var dOld = Row(deltaHat, i, nf);

        var gNew = new double[nf];
        var dNew = new double[nf];
        var sqBuf = new double[n];
        for (var iter = 0; iter < maxIter; iter++)
        {
            for (var f = 0; f < nf; f++)
                gNew[f] = PostMean(gHat[f], gBar, n, dOld[f], t2);

            for (var f = 0; f < nf; f++)
            {
                for (var k = 0; k < n; k++)
                {
                    var r = sData[f, batchIdx[k]] - gNew[f];
                    sqBuf[k] = r * r;
                }
                dNew[f] = PostVar(NumpyMath.PairwiseSum(sqBuf), n, a, b);
            }

            var gChange = MaxRelChange(gNew, gOld);
            var dChange = MaxRelChange(dNew, dOld);
            if (gChange < conv && dChange < conv)
                break;

            Array.Copy(gNew, gOld, nf);
            Array.Copy(dNew, dOld, nf);
        }

        for (var f = 0; f < nf; f++)
        {
            gammaStar[i, f] = gNew[f];
            deltaStar[i, f] = dNew[f];
        }
    }

    internal static void IntEprior(int i, double[,] gammaHat, double[,] deltaHat,
        double[,] gammaStar, double[,] deltaStar, int nf, bool meanOnly)
    {
        var gHat = Row(gammaHat, i, nf);
        var dHat = meanOnly ? Enumerable.Repeat(1.0, nf).ToArray() : Row(deltaHat, i, nf);
        var gBar = Stats.Mean(gHat);
        var dBar = Stats.Mean(dHat);
        var gVar = Stats.Var(gHat, ddof: 1);
        var dVar = Stats.Var(dHat, ddof: 1);
        var shrinkG = gVar > 0 ? gVar / (gVar + 1) : 0.5;
        var shrinkD = dVar > 0 ? dVar / (dVar + 1) : 0.5;
        for (var f = 0; f < nf; f++)
        {
            gammaStar[i, f] = shrinkG * gHat[f] + (1 - shrinkG) * gBar;
            deltaStar[i, f] = shrinkD * dHat[f] + (1 - shrinkD) * dBar;
        }
    }

    internal static double PostMean(double gHat, double gBar, int n, double dStar, double t2)
        => (t2 * n * gHat + dStar * gBar) / (t2 * n + dStar);

    private static double PostVar(double sumSq, int n, double a, double b)
        => (0.5 * sumSq + b) / (0.5 * n + a - 1);

    private static double MaxRelChange(double[] newV, double[] oldV)
    {
        double max = 0.0;
        for (var f = 0; f < newV.Length; f++)
        {
            var c = Math.Abs(newV[f] - oldV[f]) / (Math.Abs(oldV[f]) + 1e-10);
            if (c > max)
                max = c;
        }
        return max;
    }

    private static double[] Row(double[,] m, int i, int nf)
    {
        var r = new double[nf];
        for (var f = 0; f < nf; f++)
            r[f] = m[i, f];
        return r;
    }

    private static double RowVarPopulation(double[,] data, int f, int nSamples)
    {
        var row = new double[nSamples];
        for (var s = 0; s < nSamples; s++)
            row[s] = data[f, s];
        return Stats.Var(row, ddof: 0);
    }

    private static void ReplaceZeroWithMedianOfPositive(double[] v)
    {
        var positives = v.Where(x => x > 0).ToArray();
        if (positives.Length == 0)
            return;
        var med = Stats.NanMedian(positives);
        for (var i = 0; i < v.Length; i++)
            if (v[i] == 0.0)
                v[i] = med;
    }
}
