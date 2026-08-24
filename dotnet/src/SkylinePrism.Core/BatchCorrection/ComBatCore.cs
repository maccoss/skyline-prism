using System;
using System.Collections.Generic;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.BatchCorrection;

/// <summary>What the pooled per-feature scale is divided by. See <see cref="ComBatPlan"/>.</summary>
internal enum PooledScaleRule
{
    /// <summary>
    /// Variance of the residuals about each batch's fit mean, over the whole fit set, at
    /// <see cref="ComBatPlan.ResidualDdof"/>. This is standard ComBat's <c>var_pooled</c>.
    /// </summary>
    ResidualVariance,

    /// <summary>
    /// Pooled WITHIN-batch variance: the residual sum of squares over the fit set divided by
    /// <c>sum(n_i - 1)</c>. This is the unbiased pooled variance of repeated measurements of the same
    /// material, which is what reference-anchored ComBat's fit set is - so it, not
    /// <see cref="ResidualVariance"/>, is the honest scale there.
    /// </summary>
    PooledWithinBatch,
}

/// <summary>
/// Which samples a batch's effect is estimated FROM, which samples it is applied TO, and how the
/// per-feature scale is formed. This is the whole of the difference between standard and
/// reference-anchored ComBat.
/// <para>
/// <b>Standard:</b> fit = apply = every sample in the batch. Each batch is aligned to the across-batch
/// grand mean, which assumes the batches are biologically comparable.
/// </para>
/// <para>
/// <b>Reference-anchored:</b> fit = the batch's REFERENCE samples only (identical material run in
/// every batch), apply = every sample in the batch. Differences between batches in that material are
/// purely technical, so nothing biological is removed and the batches need not be comparable.
/// </para>
/// </summary>
internal sealed class ComBatPlan
{
    /// <summary>Per batch: the samples the correction is applied to.</summary>
    public required IReadOnlyList<List<int>> Apply { get; init; }

    /// <summary>
    /// Per batch: the samples its effect is estimated from. EMPTY means this batch is left alone
    /// (reference-anchored's <c>no_reference_batch: "skip"</c>) - it is neither corrected nor allowed
    /// to hold a feature out.
    /// </summary>
    public required IReadOnlyList<List<int>> Fit { get; init; }

    /// <summary>Batch index per sample, over the <see cref="Apply"/> sets.</summary>
    public required int[] BatchOfSample { get; init; }

    /// <summary>
    /// Per batch: estimate the location effect only, never a scale. Set where the fit set is too
    /// small for a scale to mean anything (a batch with one reference), and - importantly - where
    /// the fit set is not repeated measurements of one material, so its spread is biological rather
    /// than technical (reference-anchored's <c>no_reference_batch: "fallback"</c>, which falls back
    /// to the batch's own samples). Estimating a scale from biological spread would shrink real
    /// biology, so it is not done.
    /// </summary>
    public required bool[] LocationOnly { get; init; }

    public required PooledScaleRule PooledScale { get; init; }

    /// <summary>Denominator for <see cref="PooledScaleRule.ResidualVariance"/>.</summary>
    public int ResidualDdof { get; init; } = ComBat.VarPooledDdof;

    public bool MeanOnly { get; init; }

    public bool ParPrior { get; init; } = true;

    public int BatchCount => Apply.Count;

    /// <summary>Batch indices whose effect is actually estimated (a non-empty fit set).</summary>
    public IEnumerable<int> Fitted
    {
        get
        {
            for (var i = 0; i < BatchCount; i++)
                if (Fit[i].Count > 0)
                    yield return i;
        }
    }

    /// <summary>Total fit-set size, the denominator of the centre's weights.</summary>
    public int TotalFit
    {
        get
        {
            var n = 0;
            foreach (var i in Fitted)
                n += Fit[i].Count;
            return n;
        }
    }

    /// <summary>
    /// Degrees of freedom for <see cref="PooledScaleRule.PooledWithinBatch"/>: <c>sum(n_i - 1)</c>
    /// over the batches with replicates to pool. NOMINAL, not per-feature-observed, so every feature
    /// gets the same denominator and their scales stay comparable. Zero means no fit set has
    /// replicates at all, and the caller falls back to a homoscedastic scale.
    /// </summary>
    public int PooledWithinBatchDf
    {
        get
        {
            var df = 0;
            foreach (var i in Fitted)
                if (!LocationOnly[i] && Fit[i].Count >= 2)
                    df += Fit[i].Count - 1;
            return df;
        }
    }

    /// <summary>
    /// Standard ComBat: every sample in a batch both estimates that batch's effect and receives it.
    /// </summary>
    public static ComBatPlan Standard(
        IReadOnlyList<List<int>> batches, int[] batchOfSample,
        bool meanOnly = false, bool parPrior = true, int residualDdof = ComBat.VarPooledDdof)
        => new()
        {
            Apply = batches,
            Fit = batches,
            BatchOfSample = batchOfSample,
            LocationOnly = new bool[batches.Count],
            PooledScale = PooledScaleRule.ResidualVariance,
            ResidualDdof = residualDdof,
            MeanOnly = meanOnly,
            ParPrior = parPrior,
        };

    /// <summary>
    /// Reference-anchored ComBat: each batch's effect comes from its REFERENCE samples and is applied
    /// to all of them. Shared by the in-memory and streaming paths so they cannot end up fitting
    /// different columns.
    /// </summary>
    public static ComBatPlan ReferenceAnchored(
        IReadOnlyList<List<int>> batches, int[] batchOfSample, IReadOnlyList<bool> referenceMask,
        string noReferenceBatch, bool parPrior = true)
    {
        var nBatch = batches.Count;
        var references = new List<int>[nBatch];
        for (var i = 0; i < nBatch; i++)
            references[i] = batches[i].Where(s => referenceMask[s]).ToList();

        if (noReferenceBatch == "error" && references.Any(r => r.Count == 0))
            throw new ArgumentException(
                "Some batches have no reference samples; cannot reference-anchor. "
                + "Set noReferenceBatch='fallback'/'skip' or provide references in every batch.");

        var fit = new List<int>[nBatch];
        var locationOnly = new bool[nBatch];
        for (var i = 0; i < nBatch; i++)
        {
            if (references[i].Count > 0)
            {
                fit[i] = references[i];
                // One reference is a level, not a spread: correct where it sits, do not rescale.
                locationOnly[i] = references[i].Count < 2;
            }
            else if (noReferenceBatch == "fallback")
            {
                // No anchor: fall back to the batch's own centre. Its spread is biological, so this
                // is a location correction only - the "assume comparable biology" assumption that
                // reference anchoring exists to avoid, taken deliberately and only for this batch.
                fit[i] = batches[i];
                locationOnly[i] = true;
            }
            else
            {
                fit[i] = new List<int>(); // "skip": left exactly as it came in
            }
        }

        return new ComBatPlan
        {
            Apply = batches,
            Fit = fit,
            BatchOfSample = batchOfSample,
            LocationOnly = locationOnly,
            PooledScale = PooledScaleRule.PooledWithinBatch,
            MeanOnly = false,
            ParPrior = parPrior,
        };
    }
}

/// <summary>
/// The one ComBat implementation. Both <see cref="ComBat"/> and
/// <see cref="ReferenceAnchoredComBat"/> are thin wrappers that build a <see cref="ComBatPlan"/> and
/// call this, so a fix to the estimator - NaN handling, an unestimable scale, the spread floor -
/// lands in both instead of in whichever one happened to be looked at.
///
/// <para><b>Missing values.</b> Every reduction ignores NaN, as <c>sva::ComBat</c> does. It is done
/// by compacting a feature's observed values and running the ORDINARY pairwise reductions on them,
/// not by NaN-skipping accumulators: with no missing values the compacted buffer IS the original
/// buffer, so a dense cohort is bit-identical to a version that could not handle NaN at all.</para>
///
/// <para><b>What is not invented.</b> A feature is held out entirely (returned unchanged) when it has
/// no variance, or when some fitted batch never observed it - that batch's effect on it is then
/// undefined. A (batch, feature) SCALE is skipped, keeping its location correction, when the fit set
/// has fewer than 2 observations or no resolvable spread; such a scale is also excluded from that
/// batch's <c>aPrior</c>/<c>bPrior</c>, because a placeholder 1.0 inside a mean taken ACROSS features
/// lets one unestimable feature perturb the shrinkage of every other feature in the batch.</para>
/// </summary>
internal static class ComBatCore
{
    public static double[,] Run(double[,] data, ComBatPlan plan, ComBatDiagnostics? diagnostics = null)
    {
        var nFeatures = data.GetLength(0);
        var nSamples = data.GetLength(1);
        var nBatch = plan.BatchCount;

        // Batches with an empty fit set are passengers: not corrected, and - crucially - not allowed
        // to veto a feature in the screening below.
        var fitted = plan.Fitted.ToList();

        // ---- hold out what the data does not determine ----
        var heldOut = new bool[nFeatures];
        var activeRows = new List<int>(nFeatures);
        var scratch = new double[nSamples];
        for (var f = 0; f < nFeatures; f++)
        {
            if (IsCorrectable(data, f, nSamples, plan, fitted, scratch))
                activeRows.Add(f);
            else
                heldOut[f] = true;
        }

        var nf = activeRows.Count;
        var d = new double[nf, nSamples];
        for (var a = 0; a < nf; a++)
            for (var s = 0; s < nSamples; s++)
                d[a, s] = data[activeRows[a], s];

        // ---- per-batch fit means, and the center they are measured against ----
        var bHat = new double[nBatch, nf];
        var totalFit = plan.TotalFit;

        for (var i = 0; i < nBatch; i++)
        {
            var idx = plan.Fit[i];
            if (idx.Count == 0)
                continue;
            var buf = new double[idx.Count];
            for (var f = 0; f < nf; f++)
            {
                var n = Observed(d, f, idx, buf);
                bHat[i, f] = NumpyMath.PairwiseSum(buf, 0, n) / n;
            }
        }

        var center = new double[nf];
        for (var f = 0; f < nf; f++)
        {
            var c = 0.0;
            foreach (var i in fitted)
                c += ((double)plan.Fit[i].Count / totalFit) * bHat[i, f];
            center[f] = c;
        }

        var varPooled = PooledScale(d, plan, fitted, bHat, center, nf, nSamples);
        ComBat.ReplaceUnusableWithMedianOfPositive(varPooled);
        var stdPooled = new double[nf];
        for (var f = 0; f < nf; f++)
            stdPooled[f] = Math.Sqrt(varPooled[f]);

        // ---- standardize ----
        var sData = new double[nf, nSamples];
        for (var f = 0; f < nf; f++)
            for (var s = 0; s < nSamples; s++)
                sData[f, s] = (d[f, s] - center[f]) / stdPooled[f]; // NaN stays NaN, and stays local

        // ---- batch effects, from the fit set only ----
        var gammaHat = new double[nBatch, nf];
        var deltaHat = new double[nBatch, nf];
        var scaleEstimable = new bool[nBatch][];
        long unestimableScales = 0;

        for (var i = 0; i < nBatch; i++)
        {
            scaleEstimable[i] = new bool[nf];
            var idx = plan.Fit[i];
            if (idx.Count == 0)
                continue;

            var buf = new double[idx.Count];
            for (var f = 0; f < nf; f++)
            {
                var n = Observed(sData, f, idx, buf);
                gammaHat[i, f] = NumpyMath.PairwiseSum(buf, 0, n) / n;
            }

            for (var f = 0; f < nf; f++)
            {
                if (plan.MeanOnly || plan.LocationOnly[i])
                {
                    deltaHat[i, f] = 1.0;
                    continue;
                }
                var n = Observed(sData, f, idx, buf);
                var v = n >= 2 ? Stats.Var(buf.AsSpan(0, n), ddof: 1) : double.NaN;
                if (n < 2 || !ComBat.IsSpreadResolvable(v, gammaHat[i, f]))
                {
                    deltaHat[i, f] = 1.0; // no spread to estimate from -> do not scale this one
                    unestimableScales++;
                }
                else
                {
                    deltaHat[i, f] = v;
                    scaleEstimable[i][f] = true;
                }
            }
        }

        // ---- empirical-Bayes shrinkage, per batch across features ----
        var gammaStar = new double[nBatch, nf];
        var deltaStar = new double[nBatch, nf];
        for (var i = 0; i < nBatch; i++)
            for (var f = 0; f < nf; f++)
                deltaStar[i, f] = 1.0;

        for (var i = 0; i < nBatch; i++)
        {
            if (plan.Fit[i].Count == 0)
                continue; // untouched: gamma* = 0, delta* = 1

            var gRow = Row(gammaHat, i, nf);
            var gammaBar = Stats.Mean(gRow);
            var t2 = Stats.Var(gRow, ddof: 1);

            if (plan.MeanOnly || plan.LocationOnly[i])
            {
                // No scale: shrink the location effect alone. n is the fit set's size, which is what
                // decides how much this batch's own estimate is trusted against the prior.
                var n = plan.Fit[i].Count;
                for (var f = 0; f < nf; f++)
                    gammaStar[i, f] = ComBat.PostMean(gammaHat[i, f], gammaBar, n, 1.0, t2);
                continue;
            }

            if (!plan.ParPrior)
            {
                ComBat.IntEprior(i, gammaHat, deltaHat, gammaStar, deltaStar, nf, meanOnly: false);
                continue;
            }

            var (aPrior, bPrior) = Priors(deltaHat, scaleEstimable[i], i, nf);
            ComBat.ItSol(sData, plan.Fit[i], i, gammaHat, deltaHat, gammaBar, t2, aPrior, bPrior,
                gammaStar, deltaStar, nf, scaleEstimable[i]);
        }

        // ---- apply to every sample of each batch ----
        var bayes = (double[,])sData.Clone();
        for (var i = 0; i < nBatch; i++)
            foreach (var s in plan.Apply[i])
                for (var f = 0; f < nf; f++)
                    bayes[f, s] = (bayes[f, s] - gammaStar[i, f]) / Math.Sqrt(deltaStar[i, f]);
        for (var f = 0; f < nf; f++)
            for (var s = 0; s < nSamples; s++)
                bayes[f, s] = bayes[f, s] * stdPooled[f] + center[f];

        var result = new double[nFeatures, nSamples];
        for (var f = 0; f < nFeatures; f++)
            if (heldOut[f])
                for (var s = 0; s < nSamples; s++)
                    result[f, s] = data[f, s];
        for (var a = 0; a < nf; a++)
            for (var s = 0; s < nSamples; s++)
                result[activeRows[a], s] = bayes[a, s];

        if (diagnostics is not null)
        {
            diagnostics.HeldOutFeatures = nFeatures - nf;
            diagnostics.UnestimableScales = unestimableScales;

            // Both estimators reach here - standard and reference-anchored differ only in the plan -
            // so populating this once covers both.
            var activeOfRow = new int[nFeatures];
            Array.Fill(activeOfRow, -1);
            for (var a = 0; a < activeRows.Count; a++)
                activeOfRow[activeRows[a]] = a;

            var batchOfSample = new int[nSamples];
            Array.Fill(batchOfSample, -1);
            for (var i = 0; i < nBatch; i++)
                foreach (var sample in plan.Apply[i])
                    batchOfSample[sample] = i;

            diagnostics.Scaling = new ComBatScaling
            {
                DeltaStar = deltaStar,
                ActiveOfRow = activeOfRow,
                BatchOfSample = batchOfSample,
            };
        }
        return result;
    }

    /// <summary>
    /// The per-feature scale that standardization divides by. Both rules are the same residual sum of
    /// squares about the batch fit means; they differ in what it is divided by, and in which batches
    /// contribute (<see cref="PooledScaleRule.PooledWithinBatch"/> can only use batches whose fit set
    /// has replicates to pool).
    /// </summary>
    private static double[] PooledScale(
        double[,] d, ComBatPlan plan, List<int> fitted, double[,] bHat, double[] center,
        int nf, int nSamples)
    {
        var varPooled = new double[nf];

        if (plan.PooledScale == PooledScaleRule.ResidualVariance)
        {
            // Residuals are collected in SAMPLE order, not batch order: the pairwise summation inside
            // Stats.Var is order-sensitive, and this is the order the pre-unification code used.
            var residual = new double[nSamples];
            var inFit = new bool[nSamples];
            foreach (var i in fitted)
                foreach (var s in plan.Fit[i])
                    inFit[s] = true;

            for (var f = 0; f < nf; f++)
            {
                var n = 0;
                for (var s = 0; s < nSamples; s++)
                {
                    if (!inFit[s])
                        continue;
                    var v = d[f, s];
                    if (!double.IsNaN(v))
                        residual[n++] = v - bHat[plan.BatchOfSample[s], f];
                }
                varPooled[f] = Stats.Var(residual.AsSpan(0, n), plan.ResidualDdof);
            }
            return varPooled;
        }

        // Pooled within-batch: sum of squares about each batch's own fit mean, over sum(n_i - 1).
        var df = plan.PooledWithinBatchDf;
        foreach (var i in fitted)
        {
            if (plan.LocationOnly[i] || plan.Fit[i].Count < 2)
                continue; // no replicates here to pool
            for (var f = 0; f < nf; f++)
            {
                foreach (var s in plan.Fit[i])
                {
                    var v = d[f, s];
                    if (double.IsNaN(v))
                        continue;
                    var r = v - bHat[i, f];
                    varPooled[f] += r * r;
                }
            }
        }

        if (df > 0)
        {
            for (var f = 0; f < nf; f++)
                varPooled[f] /= df;
            return varPooled;
        }

        // No batch has replicates in its fit set, so there is no technical scale to measure. Fall
        // back to a homoscedastic one: it only sets how strongly the location prior shrinks, and it
        // cancels on the back-transform.
        var resBuf = new double[nSamples];
        for (var f = 0; f < nf; f++)
        {
            for (var s = 0; s < nSamples; s++)
            {
                var b = plan.BatchOfSample[s];
                resBuf[s] = d[f, s] - center[f] - (plan.Fit[b].Count > 0 ? bHat[b, f] - center[f] : 0.0);
            }
            varPooled[f] = Stats.NanVar(resBuf, ddof: 1);
        }
        return varPooled;
    }

    /// <summary>
    /// Whether anything can be estimated for this feature: every FITTED batch must have observed it
    /// at least once in its fit set (otherwise that batch's effect on it is undefined), and it must
    /// vary somewhere (otherwise there is nothing to standardize by).
    /// </summary>
    private static bool IsCorrectable(
        double[,] data, int f, int nSamples, ComBatPlan plan, List<int> fitted, double[] scratch)
    {
        foreach (var i in fitted)
        {
            var seen = false;
            foreach (var s in plan.Fit[i])
            {
                if (!double.IsNaN(data[f, s]))
                {
                    seen = true;
                    break;
                }
            }
            if (!seen)
                return false;
        }

        var n = 0;
        for (var s = 0; s < nSamples; s++)
        {
            var v = data[f, s];
            if (!double.IsNaN(v))
                scratch[n++] = v;
        }
        return n > 0 && Stats.Var(scratch.AsSpan(0, n), ddof: 0) != 0.0;
    }

    private static (double APrior, double BPrior) Priors(
        double[,] deltaHat, bool[] estimable, int i, int nf)
    {
        var kept = new double[nf];
        var n = 0;
        for (var f = 0; f < nf; f++)
            if (estimable[f])
                kept[n++] = deltaHat[i, f];

        var m = Stats.Mean(kept.AsSpan(0, n));
        var v = Stats.Var(kept.AsSpan(0, n), ddof: 1);
        if (v > 0 && m > 0)
            return ((m * m / v) + 2, m * ((m * m / v) + 1));
        return (1.0, 1.0);
    }

    /// <summary>
    /// Copy a feature's OBSERVED values for the given samples into <paramref name="buffer"/> (in
    /// sample order), returning how many there were.
    /// </summary>
    private static int Observed(double[,] m, int f, IReadOnlyList<int> samples, double[] buffer)
    {
        var n = 0;
        for (var k = 0; k < samples.Count; k++)
        {
            var v = m[f, samples[k]];
            if (!double.IsNaN(v))
                buffer[n++] = v;
        }
        return n;
    }

    private static double[] Row(double[,] m, int i, int nf)
    {
        var r = new double[nf];
        for (var f = 0; f < nf; f++)
            r[f] = m[i, f];
        return r;
    }
}
