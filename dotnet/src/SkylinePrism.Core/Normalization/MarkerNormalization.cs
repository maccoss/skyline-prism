using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Double;

namespace SkylinePrism.Core.Normalization;

/// <summary>
/// Normalization against a defined set of marker proteins: estimate one per-sample score from how
/// those markers move together, then remove from every feature the part that tracks it.
///
/// <para><b>What it answers.</b> A capture-based experiment measures whatever the beads caught. If the
/// captured amount of the thing you care about varies between samples - extracellular vesicles, say -
/// then every protein's share moves with it, and a plain loading normalization cannot tell that apart
/// from biology, because it makes total signal equal by construction. Residualizing on a marker score
/// turns "what changed in what was captured" into "what changed per unit of the marked material".</para>
///
/// <para><b>Why PC1 and not the mean of the markers.</b> Markers do not have to move as one block. In
/// the EV panel this was built from, PC1 explains ~70% of marker variance and four of the eighteen
/// load with the OPPOSITE sign to the rest - a mean therefore partially cancels and blunts the
/// estimate. PC1 weights each marker by its contribution and handles the sign structure, and it
/// transfers to a set whose dominant axis is driven by different members. The mean is offered as
/// <see cref="MarkerScoreMethod.Mean"/> for comparison; on the source cohort the two correlate at
/// r = 0.95 while PC1 gives the more conservative answer.</para>
///
/// <para><b>Where it belongs.</b> AFTER the ordinary normalization, never instead of it. The score has
/// to be estimated from data whose per-sample loading is already removed; computed on raw abundances,
/// PC1 loads on injection volume, and residualizing then quietly re-does the loading step using
/// eighteen proteins' worth of noise instead of everything.</para>
/// </summary>
public static class MarkerNormalization
{
    /// <summary>Smallest number of markers that can define a score. Below this, refuse rather than guess.</summary>
    public const int MinMarkers = 3;

    /// <summary>
    /// The per-sample score and what it was built from - reported so a run can be judged, not just
    /// trusted. <paramref name="VarianceExplained"/> below ~0.4 means the markers are not moving
    /// together and the score is a weak summary of them.
    /// </summary>
    /// <param name="Score">One value per sample; higher means more of the marked material.</param>
    /// <param name="Loadings">PC1 loading per marker, in <paramref name="MarkerNames"/> order.</param>
    public sealed record MarkerScore(
        double[] Score,
        double[] Loadings,
        IReadOnlyList<string> MarkerNames,
        double VarianceExplained,
        double CorrelationWithMean);

    /// <summary>
    /// Estimate the per-sample score from a LOG2 marker block [markers, samples].
    /// <para>
    /// Each marker is z-scored across samples first, so a high-abundance marker does not dominate the
    /// axis purely by scale, and PC1 is taken by SVD. The sign of a principal component is arbitrary,
    /// so the result is oriented to correlate positively with the mean z-scored marker profile:
    /// without that, "higher score" would mean more marked material or less, at random, from run to run.
    /// </para>
    /// </summary>
    public static MarkerScore ComputeScore(
        double[,] markerLog2, IReadOnlyList<string> markerNames,
        MarkerScoreMethod method = MarkerScoreMethod.Pc1)
    {
        var nMarkers = markerLog2.GetLength(0);
        var nSamples = markerLog2.GetLength(1);
        if (nMarkers < MinMarkers)
            throw new ArgumentException(
                $"A marker score needs at least {MinMarkers} markers; got {nMarkers}.", nameof(markerLog2));
        if (nSamples < 2)
            throw new ArgumentException("A marker score needs at least 2 samples.", nameof(markerLog2));

        // z-score each marker across samples (sample standard deviation, ddof=1).
        var z = new double[nMarkers, nSamples];
        for (var i = 0; i < nMarkers; i++)
        {
            var mean = 0.0;
            for (var j = 0; j < nSamples; j++)
                mean += markerLog2[i, j];
            mean /= nSamples;

            var ss = 0.0;
            for (var j = 0; j < nSamples; j++)
            {
                var d = markerLog2[i, j] - mean;
                ss += d * d;
            }
            var sd = Math.Sqrt(ss / (nSamples - 1));
            // A marker that never moves carries no information about the axis; leaving it at zero keeps
            // it in the block without letting a divide-by-zero decide the score.
            for (var j = 0; j < nSamples; j++)
                z[i, j] = sd > 0 ? (markerLog2[i, j] - mean) / sd : 0.0;
        }

        // The mean z-scored profile: the sign reference, and the Mean method's score.
        var meanProfile = new double[nSamples];
        for (var j = 0; j < nSamples; j++)
        {
            var acc = 0.0;
            for (var i = 0; i < nMarkers; i++)
                acc += z[i, j];
            meanProfile[j] = acc / nMarkers;
        }

        double[] score;
        double[] loadings;
        double varianceExplained;
        if (method == MarkerScoreMethod.Mean)
        {
            score = (double[])meanProfile.Clone();
            loadings = Enumerable.Repeat(1.0 / nMarkers, nMarkers).ToArray();
            varianceExplained = double.NaN;
        }
        else
        {
            // A dense SVD is right here and nowhere else in this codebase: the marker block is a
            // couple of dozen rows, not the tens of thousands of features Pca has to avoid
            // materializing. Same decomposition numpy's svd(full_matrices=False) gives.
            var svd = DenseMatrix.OfArray(z).Svd(computeVectors: true);
            score = new double[nSamples];
            for (var j = 0; j < nSamples; j++)
                score[j] = svd.VT[0, j] * svd.S[0];
            loadings = new double[nMarkers];
            for (var i = 0; i < nMarkers; i++)
                loadings[i] = svd.U[i, 0];

            var total = svd.S.Sum(v => v * v);
            varianceExplained = total > 0 ? svd.S[0] * svd.S[0] / total : double.NaN;
        }

        // Orient: higher score = more marked material.
        var r = Correlation(score, meanProfile);
        if (r < 0)
        {
            for (var j = 0; j < nSamples; j++)
                score[j] = -score[j];
            for (var i = 0; i < nMarkers; i++)
                loadings[i] = -loadings[i];
            r = -r;
        }

        return new MarkerScore(score, loadings, markerNames.ToList(), varianceExplained, r);
    }

    /// <summary>
    /// Remove the part of every feature that tracks <paramref name="score"/>, in place, on a LOG2
    /// matrix [features, samples].
    /// <para>
    /// Per feature this is an ordinary least-squares fit of its profile on <c>[1, score]</c>, keeping
    /// the residual. The intercept is added back, so the feature keeps its own abundance level and only
    /// the score-dependent part is taken out - the residual alone would put every feature at zero in
    /// log space, i.e. an abundance of 1, which is not a quantity anything downstream can use.
    /// </para>
    /// <para>
    /// A feature with a missing value in some sample is fitted on the samples it has. Fewer than 3
    /// observations leaves it untouched: a two-point regression through a two-parameter model has no
    /// residual to speak of, and zeroing it would fabricate a result.
    /// </para>
    /// </summary>
    public static void Residualize(double[,] log2Matrix, IReadOnlyList<double> score)
    {
        var nFeatures = log2Matrix.GetLength(0);
        var nSamples = log2Matrix.GetLength(1);
        if (score.Count != nSamples)
            throw new ArgumentException(
                $"Score has {score.Count} values but the matrix has {nSamples} samples.", nameof(score));

        for (var i = 0; i < nFeatures; i++)
        {
            // Sums over the samples this feature was actually observed in.
            double n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
            for (var j = 0; j < nSamples; j++)
            {
                var y = log2Matrix[i, j];
                if (double.IsNaN(y))
                    continue;
                var x = score[j];
                n++;
                sx += x;
                sy += y;
                sxx += x * x;
                sxy += x * y;
            }
            if (n < 3)
                continue;

            var denom = n * sxx - sx * sx;
            if (Math.Abs(denom) < 1e-12)
                continue; // the score is constant across this feature's samples: nothing to remove
            var slope = (n * sxy - sx * sy) / denom;

            // y - slope*x, i.e. the residual with the intercept added back.
            for (var j = 0; j < nSamples; j++)
                if (!double.IsNaN(log2Matrix[i, j]))
                    log2Matrix[i, j] -= slope * score[j];
        }
    }

    private static double Correlation(IReadOnlyList<double> a, IReadOnlyList<double> b)
    {
        var n = a.Count;
        double ma = 0, mb = 0;
        for (var i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
        ma /= n; mb /= n;
        double sab = 0, saa = 0, sbb = 0;
        for (var i = 0; i < n; i++)
        {
            var da = a[i] - ma;
            var db = b[i] - mb;
            sab += da * db;
            saa += da * da;
            sbb += db * db;
        }
        return saa > 0 && sbb > 0 ? sab / Math.Sqrt(saa * sbb) : 0.0;
    }
}

/// <summary>How the per-sample marker score is summarized from the marker block.</summary>
public enum MarkerScoreMethod
{
    /// <summary>First principal component of the z-scored markers (default).</summary>
    Pc1,

    /// <summary>Plain mean of the z-scored markers - blunter when the markers do not share a sign.</summary>
    Mean,
}
