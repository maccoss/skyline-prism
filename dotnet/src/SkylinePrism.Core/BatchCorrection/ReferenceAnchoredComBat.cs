using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.BatchCorrection;

/// <summary>
/// Reference-anchored ComBat, ported 1:1 from batch_correction.py:combat_reference_anchored.
/// Unlike standard ComBat (which aligns each batch to the across-batch grand mean, assuming equal
/// biological composition per batch), this estimates each batch's technical effect from the
/// REFERENCE samples only - identical material run in every batch - using ALL peptides in those
/// references, then applies the correction to every sample. Because the reference material is
/// identical, per-batch differences in it are purely technical, so no biology is removed. The
/// additive (gamma) and multiplicative (delta) effects are empirical-Bayes shrunk across features
/// within each batch, reusing ComBat's EB core. LOG2 [nFeatures, nSamples] in and out.
/// </summary>
public static class ReferenceAnchoredComBat
{
    public static double[,] Run(
        double[,] data,
        IReadOnlyList<string> batchLabels,
        IReadOnlyList<bool> referenceMask,
        bool parPrior = true,
        string noReferenceBatch = "fallback")
    {
        var nFeatures = data.GetLength(0);
        var nSamples = data.GetLength(1);
        if (batchLabels.Count != nSamples)
            throw new ArgumentException("batchLabels length must equal number of samples.");
        if (referenceMask.Count != nSamples)
            throw new ArgumentException("referenceMask length must equal number of samples.");
        if (noReferenceBatch is not ("fallback" or "skip" or "error"))
            throw new ArgumentException("noReferenceBatch must be 'fallback', 'skip', or 'error'.");
        if (!referenceMask.Any(m => m))
            throw new ArgumentException(
                "Reference-anchored ComBat requires at least one reference sample.");

        // First-seen batch order (dict.fromkeys), NOT sorted.
        var uniqueBatches = new List<string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var b in batchLabels)
            if (seen.Add(b))
                uniqueBatches.Add(b);
        var nBatch = uniqueBatches.Count;
        if (nBatch < 2)
            return (double[,])data.Clone();

        // Hold out zero-variance features (nanvar ddof=0), pass them through unchanged.
        var zeroVar = new bool[nFeatures];
        var activeRows = new List<int>(nFeatures);
        var rowBuf = new double[nSamples];
        for (var f = 0; f < nFeatures; f++)
        {
            for (var s = 0; s < nSamples; s++)
                rowBuf[s] = data[f, s];
            var v = Stats.NanVar(rowBuf, ddof: 0);
            if (v == 0.0 || double.IsNaN(v))
                zeroVar[f] = true;
            else
                activeRows.Add(f);
        }
        var nWork = activeRows.Count;
        var work = new double[nWork, nSamples];
        for (var a = 0; a < nWork; a++)
            for (var s = 0; s < nSamples; s++)
                work[a, s] = data[activeRows[a], s];

        // Per-batch sample / reference column indices.
        var batchIndices = new List<int>[nBatch];
        var batchRefIndices = new List<int>[nBatch];
        for (var bi = 0; bi < nBatch; bi++)
        {
            batchIndices[bi] = new List<int>();
            batchRefIndices[bi] = new List<int>();
        }
        var batchOf = uniqueBatches.Select((b, i) => (b, i)).ToDictionary(x => x.b, x => x.i, StringComparer.Ordinal);
        for (var s = 0; s < nSamples; s++)
        {
            var bi = batchOf[batchLabels[s]];
            batchIndices[bi].Add(s);
            if (referenceMask[s])
                batchRefIndices[bi].Add(s);
        }
        var nRefPerBatch = batchRefIndices.Select(r => r.Count).ToArray();

        if (nRefPerBatch.Any(n => n == 0) && noReferenceBatch == "error")
            throw new ArgumentException(
                "Some batches have no reference samples; cannot reference-anchor. "
                + "Set noReferenceBatch='fallback'/'skip' or provide references in every batch.");

        var allRefCols = Enumerable.Range(0, nSamples).Where(s => referenceMask[s]).ToList();

        // alpha[g]: pooled reference level (nanmean over all reference columns); if a feature is
        // NaN in every reference, fall back to its all-sample mean.
        var alpha = new double[nWork];
        for (var g = 0; g < nWork; g++)
        {
            alpha[g] = NanMeanCols(work, g, allRefCols);
            if (double.IsNaN(alpha[g]))
                alpha[g] = NanMeanCols(work, g, AllCols(nSamples));
        }

        // Raw additive offsets per batch (log2).
        var gammaRaw = new double[nBatch, nWork];
        for (var bi = 0; bi < nBatch; bi++)
        {
            List<int>? cols = nRefPerBatch[bi] >= 1 ? batchRefIndices[bi]
                : noReferenceBatch == "fallback" ? batchIndices[bi] : null;
            for (var g = 0; g < nWork; g++)
            {
                var val = cols is null ? 0.0 : NanMeanCols(work, g, cols) - alpha[g];
                gammaRaw[bi, g] = double.IsNaN(val) ? 0.0 : val;
            }
        }

        // var_pooled: pooled within-batch reference-replicate variance (batches with >=2 refs).
        var sumsq = new double[nWork];
        var pooledDf = 0;
        for (var bi = 0; bi < nBatch; bi++)
        {
            if (nRefPerBatch[bi] < 2)
                continue;
            var cols = batchRefIndices[bi];
            for (var g = 0; g < nWork; g++)
            {
                var mean = NanMeanCols(work, g, cols);
                foreach (var s in cols)
                {
                    var r = work[g, s] - mean;
                    if (!double.IsNaN(r))
                        sumsq[g] += r * r;
                }
            }
            pooledDf += nRefPerBatch[bi] - 1;
        }

        var varPooled = new double[nWork];
        var scaleEstimable = pooledDf > 0;
        if (scaleEstimable)
        {
            for (var g = 0; g < nWork; g++)
                varPooled[g] = sumsq[g] / pooledDf;
        }
        else
        {
            // Fallback homoscedastic scale for the location EB prior (cancels on back-transform).
            var resBuf = new double[nSamples];
            for (var g = 0; g < nWork; g++)
            {
                for (var bi = 0; bi < nBatch; bi++)
                    foreach (var s in batchIndices[bi])
                        resBuf[s] = work[g, s] - alpha[g] - gammaRaw[bi, g];
                varPooled[g] = Stats.NanVar(resBuf, ddof: 1);
            }
        }

        // Fill non-positive / non-finite var with the median of the positive vars.
        var positives = varPooled.Where(x => x > 0 && !double.IsNaN(x) && !double.IsInfinity(x)).ToArray();
        var fill = positives.Length > 0 ? Stats.NanMedian(positives) : 1.0;
        for (var g = 0; g < nWork; g++)
            if (!(varPooled[g] > 0 && !double.IsNaN(varPooled[g]) && !double.IsInfinity(varPooled[g])))
                varPooled[g] = fill;
        var stdPooled = varPooled.Select(Math.Sqrt).ToArray();

        // Standardized data + standardized additive effects.
        var sData = new double[nWork, nSamples];
        for (var g = 0; g < nWork; g++)
            for (var s = 0; s < nSamples; s++)
                sData[g, s] = (work[g, s] - alpha[g]) / stdPooled[g];

        var gammaHat = new double[nBatch, nWork];
        for (var bi = 0; bi < nBatch; bi++)
            for (var g = 0; g < nWork; g++)
            {
                var v = gammaRaw[bi, g] / stdPooled[g];
                gammaHat[bi, g] = double.IsNaN(v) ? 0.0 : v;
            }

        // Scale effects: variance of standardized reference replicates per batch (>=2 refs).
        var deltaHat = new double[nBatch, nWork];
        for (var bi = 0; bi < nBatch; bi++)
            for (var g = 0; g < nWork; g++)
                deltaHat[bi, g] = 1.0;
        if (scaleEstimable)
        {
            for (var bi = 0; bi < nBatch; bi++)
            {
                if (nRefPerBatch[bi] < 2)
                    continue;
                var cols = batchRefIndices[bi];
                for (var g = 0; g < nWork; g++)
                {
                    var dv = NanVarCols(sData, g, cols, ddof: 1);
                    if (double.IsNaN(dv) || dv <= 0)
                        dv = 1.0;
                    deltaHat[bi, g] = dv;
                }
            }
        }

        // EB priors per batch (from that batch's own features).
        var gammaBar = new double[nBatch];
        var t2 = new double[nBatch];
        var aPrior = new double[nBatch];
        var bPrior = new double[nBatch];
        for (var bi = 0; bi < nBatch; bi++)
        {
            var gRow = Row(gammaHat, bi, nWork);
            gammaBar[bi] = Stats.Mean(gRow);
            t2[bi] = Stats.Var(gRow, ddof: 1);
            var dRow = Row(deltaHat, bi, nWork);
            var m = Stats.Mean(dRow);
            var v = Stats.Var(dRow, ddof: 1);
            if (v > 0 && m > 0)
            {
                aPrior[bi] = m * m / v + 2;
                bPrior[bi] = m * (m * m / v + 1);
            }
            else
            {
                aPrior[bi] = 1.0;
                bPrior[bi] = 1.0;
            }
        }

        var gammaStar = new double[nBatch, nWork];
        var deltaStar = new double[nBatch, nWork];
        for (var bi = 0; bi < nBatch; bi++)
            for (var g = 0; g < nWork; g++)
                deltaStar[bi, g] = 1.0;

        for (var bi = 0; bi < nBatch; bi++)
        {
            var nRef = nRefPerBatch[bi];
            if (nRef == 0 && noReferenceBatch == "skip")
                continue; // gamma_star 0, delta_star 1

            if (nRef >= 2)
            {
                if (parPrior)
                    ComBat.ItSol(sData, batchRefIndices[bi], bi, gammaHat, deltaHat,
                        gammaBar[bi], t2[bi], aPrior[bi], bPrior[bi], gammaStar, deltaStar, nWork);
                else
                    ComBat.IntEprior(bi, gammaHat, deltaHat, gammaStar, deltaStar, nWork, meanOnly: false);
            }
            else
            {
                // Location-only: single-reference batch (n_ref=1) or grand-mean fallback (n_ref=0).
                var nEff = nRef >= 1 ? nRef : batchIndices[bi].Count;
                for (var g = 0; g < nWork; g++)
                    gammaStar[bi, g] = ComBat.PostMean(gammaHat[bi, g], gammaBar[bi], nEff, 1.0, t2[bi]);
            }
        }

        // Apply to ALL samples: (s_data - gamma*)/sqrt(delta*) * std_pooled + alpha.
        var bayes = (double[,])sData.Clone();
        for (var bi = 0; bi < nBatch; bi++)
            foreach (var s in batchIndices[bi])
                for (var g = 0; g < nWork; g++)
                    bayes[g, s] = (bayes[g, s] - gammaStar[bi, g]) / Math.Sqrt(deltaStar[bi, g]);
        for (var g = 0; g < nWork; g++)
            for (var s = 0; s < nSamples; s++)
                bayes[g, s] = bayes[g, s] * stdPooled[g] + alpha[g];

        // Restore zero-variance features.
        var result = new double[nFeatures, nSamples];
        for (var f = 0; f < nFeatures; f++)
            if (zeroVar[f])
                for (var s = 0; s < nSamples; s++)
                    result[f, s] = data[f, s];
        for (var a = 0; a < nWork; a++)
            for (var s = 0; s < nSamples; s++)
                result[activeRows[a], s] = bayes[a, s];
        return result;
    }

    private static IReadOnlyList<int> AllCols(int n) => Enumerable.Range(0, n).ToList();

    private static double NanMeanCols(double[,] m, int f, IReadOnlyList<int> cols)
    {
        var buf = new double[cols.Count];
        for (var k = 0; k < cols.Count; k++)
            buf[k] = m[f, cols[k]];
        return Stats.NanMean(buf);
    }

    private static double NanVarCols(double[,] m, int f, IReadOnlyList<int> cols, int ddof)
    {
        var buf = new double[cols.Count];
        for (var k = 0; k < cols.Count; k++)
            buf[k] = m[f, cols[k]];
        return Stats.NanVar(buf, ddof);
    }

    private static double[] Row(double[,] m, int i, int nf)
    {
        var r = new double[nf];
        for (var f = 0; f < nf; f++)
            r[f] = m[i, f];
        return r;
    }
}
