using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.BatchCorrection;

/// <summary>
/// What ComBat could not estimate from the data, so a caller can report it rather than have it
/// disappear silently. Both the in-memory and the streaming implementations fill this identically.
/// </summary>
public sealed class ComBatDiagnostics
{
    /// <summary>
    /// Features passed through uncorrected: no variance at all, or no observation in some batch, so
    /// that batch's effect on them is undefined.
    /// </summary>
    public int HeldOutFeatures { get; internal set; }

    /// <summary>
    /// (batch, feature) pairs whose SCALE effect was not estimable - fewer than 2 observations in
    /// that batch, or no spread among them. These keep their location correction and are left
    /// unscaled; see <see cref="ComBat"/> remarks.
    /// </summary>
    public long UnestimableScales { get; internal set; }
}

/// <summary>
/// ComBat empirical-Bayes batch correction (Johnson et al. 2007). Operates on a LOG2
/// [nFeatures, nSamples] matrix. Standard path only: one-hot batch design, no covariates, no
/// reference batch.
///
/// <para><b>Missing values.</b> Every reduction here ignores NaN, as <c>sva::ComBat</c> does
/// (<c>Beta.NA</c>, <c>rowVars(na.rm = TRUE)</c>, and a per-row <c>n</c> inside <c>it.sol</c>). It is
/// done by compacting a feature's observed values and then running the ORDINARY pairwise reductions
/// on them, rather than by NaN-skipping accumulators: with no missing values the compacted buffer is
/// the original buffer, so a dense cohort produces bit-identical results to the version that could
/// not handle NaN at all.</para>
///
/// <para><b>What cannot be estimated.</b> A quantity that the data does not determine is not
/// invented:</para>
/// <list type="bullet">
/// <item><b>Feature held out entirely</b> when it has no variance, or no observation at all in some
/// batch (that batch's location effect is then undefined). Such features are returned unchanged,
/// matching sva's treatment of its zero-variance rows.</item>
/// <item><b>Scale not estimated</b> for a (batch, feature) with fewer than 2 observations, or with
/// no spread among them. Its location effect IS estimable, so it is still applied - only the scale
/// correction is skipped (<c>delta* = 1</c>). Note this is deliberately NOT sva's behavior: sva
/// drops the whole feature if any batch is constant in it, discarding a location correction that the
/// data supports.</item>
/// </list>
/// <para>Crucially, a scale that could not be estimated is also excluded from that batch's
/// <c>aPrior</c>/<c>bPrior</c>. Feeding a placeholder 1.0 into a mean and variance taken ACROSS
/// features - which is what the previous implementation did - let one unestimable feature perturb
/// the shrinkage of every other feature in the batch.</para>
///
/// <para><b>The one remaining difference from sva</b> is <c>var_pooled</c>'s denominator, and it is
/// sva that is inconsistent: its dense path computes <c>sum(residual^2) / n</c> (ddof 0) while its
/// missing-value path computes <c>rowVars(..., na.rm = TRUE)</c> (ddof 1). PRISM uses <b>ddof 0
/// throughout</b> (<see cref="VarPooledDdof"/>), which means:</para>
/// <list type="bullet">
/// <item>On a dense matrix - the normal case for PRISM, because Skyline integrates imputed peak
/// boundaries for every replicate (see "Data Density" in CLAUDE.md) - PRISM reproduces
/// <c>sva::ComBat</c> to floating-point noise.</item>
/// <item>On an input that DOES contain missing values, PRISM differs from sva by ~0.3%, because sva
/// switches denominator there and PRISM does not.</item>
/// </list>
/// <para>Following sva's switch was considered and rejected: it would mean that one peptide missing
/// from one document shifts every corrected value in the cohort by ~0.3%, a discontinuity that is
/// worse than the divergence it removes. <c>SvaGoldenTests</c> pins both halves by running each
/// denominator against the R goldens.</para>
/// </summary>
public static class ComBat
{
    /// <summary>
    /// Batch-correct <paramref name="data"/> (features x samples) given per-sample batch
    /// labels. Returns a new corrected matrix on the same scale.
    /// </summary>
    /// <summary>
    /// The <c>var_pooled</c> denominator: <c>sum(residual^2) / n</c>, matching what
    /// <c>sva::ComBat</c> does when the input has no missing values - which for PRISM is the normal
    /// case (see "Data Density" in CLAUDE.md). Shared so the in-memory and streaming paths cannot
    /// drift apart on it.
    /// </summary>
    internal const int VarPooledDdof = 0;

    public static double[,] Run(double[,] data, IReadOnlyList<string> batchLabels,
        bool parPrior = true, bool meanOnly = false, ComBatDiagnostics? diagnostics = null)
        => Run(data, batchLabels, parPrior, meanOnly, diagnostics, VarPooledDdof);

    /// <summary>
    /// As <see cref="Run(double[,], IReadOnlyList{string}, bool, bool, ComBatDiagnostics)"/>, with
    /// the pooled-variance denominator exposed. It exists ONLY so the sva golden tests can
    /// demonstrate that <c>var_pooled</c> is the single remaining difference from sva - see the
    /// remarks on the public overload. Production always uses <see cref="VarPooledDdof"/>.
    /// </summary>
    internal static double[,] Run(double[,] data, IReadOnlyList<string> batchLabels,
        bool parPrior, bool meanOnly, ComBatDiagnostics? diagnostics, int varPooledDdof)
    {
        var nSamples = data.GetLength(1);
        if (batchLabels.Count != nSamples)
            throw new ArgumentException("batchLabels length must equal number of samples.");

        var (uniqueBatches, batches, batchOfSample) = SortedBatches(batchLabels);
        ValidateBatchSizes(uniqueBatches, batches);

        // Standard ComBat is the core with the fit set equal to the apply set: every sample in a
        // batch both estimates that batch's effect and receives it.
        return ComBatCore.Run(data, new ComBatPlan
        {
            Apply = batches,
            Fit = batches,
            BatchOfSample = batchOfSample,
            LocationOnly = new bool[batches.Length],
            PooledScale = PooledScaleRule.ResidualVariance,
            ResidualDdof = varPooledDdof,
            MeanOnly = meanOnly,
            ParPrior = parPrior,
        }, diagnostics);
    }

    /// <summary>
    /// Batches in np.unique order (sorted unique labels), with each batch's sample indices ascending.
    /// Shared so the two entry points cannot disagree about which batch is which.
    /// </summary>
    internal static (List<string> Unique, List<int>[] Batches, int[] BatchOfSample) SortedBatches(
        IReadOnlyList<string> batchLabels)
    {
        var unique = batchLabels.Distinct().OrderBy(x => x, StringComparer.Ordinal).ToList();
        var indexOf = unique.Select((b, i) => (b, i))
            .ToDictionary(x => x.b, x => x.i, StringComparer.Ordinal);
        var batches = new List<int>[unique.Count];
        for (var i = 0; i < unique.Count; i++)
            batches[i] = new List<int>();
        var batchOfSample = new int[batchLabels.Count];
        for (var s = 0; s < batchLabels.Count; s++)
        {
            var b = indexOf[batchLabels[s]];
            batchOfSample[s] = b;
            batches[b].Add(s);
        }
        return (unique, batches, batchOfSample);
    }

    /// <summary>
    /// Spread below a trillionth of the values' own magnitude is rounding, not a scale estimate.
    /// <para>
    /// Testing <c>variance == 0</c> exactly - which is what sva and every previous version of this
    /// code did - is knife-edge. The same 82 replicates gave a within-batch variance of exactly 0.0
    /// through Python and 7.99e-31 through C#, which flipped the feature between "no scale to
    /// estimate" and "a scale of 8e-31", and the two engines then produced answers 3% apart. The
    /// floor sits far above accumulated rounding (~1e-15 relative) and far below any real
    /// measurement (a standardized variance of 1e-6 is values agreeing to a part in a thousand),
    /// so nothing genuine is caught by it.
    /// </para>
    /// </summary>
    internal static bool IsSpreadResolvable(double variance, double mean)
    {
        if (double.IsNaN(variance))
            return false;
        const double relativeFloor = 1e-12;
        var scale = Math.Max(Math.Abs(mean), 1.0);
        return Math.Sqrt(variance) > relativeFloor * scale;
    }

    internal static void ItSol(
        double[,] sData, List<int> batchIdx, int i,
        double[,] gammaHat, double[,] deltaHat, double gBar, double t2, double a, double b,
        double[,] gammaStar, double[,] deltaStar, int nf,
        bool[]? scaleEstimable = null,
        double conv = 1e-4, int maxIter = 100)
    {
        var gHat = Row(gammaHat, i, nf);
        var gOld = (double[])gHat.Clone();
        var dOld = Row(deltaHat, i, nf);

        // Per-FEATURE observation count, as sva's `n <- rowSums(!is.na(sdat))`. With no missing
        // values this is the batch size for every feature, i.e. the scalar the old code used.
        var nObs = new int[nf];
        for (var f = 0; f < nf; f++)
        {
            var n = 0;
            for (var k = 0; k < batchIdx.Count; k++)
                if (!double.IsNaN(sData[f, batchIdx[k]]))
                    n++;
            nObs[f] = n;
        }

        var gNew = new double[nf];
        var dNew = new double[nf];
        var sqBuf = new double[batchIdx.Count];
        for (var iter = 0; iter < maxIter; iter++)
        {
            for (var f = 0; f < nf; f++)
                gNew[f] = PostMean(gHat[f], gBar, nObs[f], dOld[f], t2);

            for (var f = 0; f < nf; f++)
            {
                if (scaleEstimable is not null && !scaleEstimable[f])
                {
                    dNew[f] = 1.0; // scale not supported by the data - leave this one unscaled
                    continue;
                }
                var m = 0;
                for (var k = 0; k < batchIdx.Count; k++)
                {
                    var v = sData[f, batchIdx[k]];
                    if (double.IsNaN(v))
                        continue;
                    var r = v - gNew[f];
                    sqBuf[m++] = r * r;
                }
                dNew[f] = PostVar(NumpyMath.PairwiseSum(sqBuf, 0, m), nObs[f], a, b);
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

    /// <summary>
    /// NOTE: this is NOT sva's <c>int.eprior</c> (a leave-one-out Monte-Carlo integration over the
    /// empirical prior) but the much cheaper fixed shrinkage the Python implementation uses. It is
    /// unreachable from the pipeline - <c>par_prior</c> is not exposed in either engine's config and
    /// the callers never pass false - and is kept only so the two engines match.
    /// </summary>
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

    /// <summary>
    /// A batch with a single sample has no within-batch spread to estimate its effect; abort like
    /// Python's _check_inputs rather than silently degrading to a mean-only correction. Shared with
    /// the streaming path so both refuse the same cohorts with the same message.
    /// </summary>
    internal static void ValidateBatchSizes(
        IReadOnlyList<string> uniqueBatches, IReadOnlyList<List<int>> batches)
    {
        var singletonBatches = uniqueBatches.Where((_, i) => batches[i].Count == 1).ToList();
        if (singletonBatches.Count > 0)
            throw new InvalidOperationException(
                "ComBat cannot correct batch(es) with a single sample (each batch needs >= 2 samples to "
                + "estimate its effect): " + string.Join(", ", singletonBatches)
                + ". Relabel/merge the batch, drop the sample, or turn off batch correction.");
    }

    internal static double PostVar(double sumSq, int n, double a, double b)
        => (0.5 * sumSq + b) / (0.5 * n + a - 1);

    internal static double MaxRelChange(double[] newV, double[] oldV)
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

    /// <summary>
    /// A scale of zero (or a non-finite one) cannot be divided by. Substitute the median of the
    /// scales that ARE usable, or 1.0 when not one feature has a usable scale - which leaves the
    /// standardization a no-op instead of producing infinities.
    /// <para>
    /// Shared by the in-memory core and the streaming path so the two cannot disagree about which
    /// scales are usable.
    /// </para>
    /// </summary>
    internal static void ReplaceUnusableWithMedianOfPositive(double[] v)
    {
        var positives = v.Where(Usable).ToArray();
        var fill = positives.Length > 0 ? Stats.NanMedian(positives) : 1.0;
        for (var i = 0; i < v.Length; i++)
            if (!Usable(v[i]))
                v[i] = fill;

        static bool Usable(double x) => x > 0 && !double.IsNaN(x) && !double.IsInfinity(x);
    }
}
