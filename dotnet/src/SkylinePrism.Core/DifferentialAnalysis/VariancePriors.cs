using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.DifferentialAnalysis;

/// <summary>
/// Per-feature prior variances for the moderated t, in the estimator the lab's
/// <c>proteomics-toolkit</c> uses rather than limma's.
/// </summary>
/// <remarks>
/// <para><b>Why this exists beside <see cref="EmpiricalBayes.SqueezeVarTrend"/>.</b> Both are called
/// an "intensity trend" and they are different estimators. limma's <c>trend=TRUE</c> fits a natural
/// cubic spline of log(residual variance) against mean <b>log2</b> expression and re-estimates the
/// prior degrees of freedom from that fit. The toolkit fits a LOWESS of log(within-group variance)
/// against log(within-group mean) on <b>raw, pre-log</b> intensities, one point per (feature, group),
/// and converts back to log space by the delta method. They differ in the smoother, the space and what
/// contributes a point. What they share is that the prior's level and degrees of freedom are fitted to
/// the residuals around the trend: limma does it inside fitFDist, and this trend is handed to
/// <c>Differential.WithCalibratedLevel</c>, which does the same thing with the curve supplied. So what
/// this class returns is a SHAPE, not a finished prior. The two agree to a median 0.9% on p-values on
/// the trend golden (<c>docs/differential-analysis.md</c>), but they are not interchangeable, and a
/// result has to say which one produced it.</para>
/// <para><b>Why raw intensities are free here.</b> The toolkit has to carry a separate pre-log copy
/// of the matrix because its dispatcher log-transforms in place. PRISM's matrix is log2 of a LINEAR
/// parquet, so the raw intensity is just <c>2^x</c> - exact, since that is the transform that
/// produced it.</para>
/// </remarks>
internal static class VariancePriors
{
    /// <summary>The toolkit's minimum: fewer points than this and the trend is not a fit, it is noise.</summary>
    private const int MinTrendPoints = 5;

    /// <summary>
    /// The LOWESS both priors use: statsmodels' parameters, plus the interpolation distance
    /// statsmodels itself recommends for large inputs.
    /// </summary>
    /// <remarks>
    /// <para><b>delta is not optional here.</b> With delta = 0 - statsmodels' default, and what the
    /// reference implementation passes - every point gets its own weighted regression, which is
    /// O(n * frac*n * iters). Measured on this repo: 43 ms at n = 2,000, 256 ms at 6,000, 2.9 s at
    /// 20,000. The intensity trend contributes one point per (feature, group), so a peptide-level
    /// contrast on a 75,000-peptide cohort is n = 150,000, which extrapolates to roughly three
    /// MINUTES - on a pane that re-runs whenever a selector changes, and with this prior as the
    /// default. With delta at 1% of the x range the same 20,000 points take 23 ms, and a whole
    /// contrast over 75,000 features x 100 samples - prior, fit, moderation and BH - returns in
    /// 0.62 s.</para>
    /// <para>The cost is a small approximation: points closer together than delta are linearly
    /// interpolated rather than individually fitted, which moved the fitted values by at most ~3e-5
    /// relative in the same measurement, far below anything that changes a conclusion. On a real
    /// cohort (3,595 serum proteins, prior fitted on 12 reference and QC injections) it is the only
    /// thing left between PRISM and the toolkit: it moves the per-feature prior SCALE by at most
    /// 1.4e-3 relative (median 2.5e-4) and p-values by at most 8e-4, and handing the toolkit the same
    /// delta closes the gap to ~1e-12. On inputs the
    /// size of the goldens it makes NO difference at all - 1% of the x range there is narrower than
    /// the spacing between points, so nothing is interpolated and the fit is identical, which is why
    /// those still assert against the reference at 1e-9. The approximation only ever engages where
    /// the exact fit would not return.</para>
    /// <para>Every other LOWESS call in PRISM already passes this same 1% delta
    /// (<c>Normalizer</c>, <c>NormalizationFactors</c>, <c>PlotRenderer</c>); these two were the
    /// exception.</para>
    /// </remarks>
    private static double[] SmoothTrend(double[] x, double[] y)
    {
        var span = x[^1] - x[0]; // x is sorted ascending by every caller
        return Lowess.Fit(x, y, frac: 0.5, iterations: 3, delta: span > 0 ? span * 0.01 : 0.0);
    }

    /// <summary>One (feature, group) observation: the raw-scale mean, sd and sample count.</summary>
    private readonly record struct GroupStat(int Feature, double Mean, double Sd, int N);

    /// <summary>
    /// The toolkit's <c>intensity_trend</c> prior: per-feature prior variance in LOG2 space, or null
    /// when there are too few usable points to fit a trend (the caller then falls back to the global
    /// prior and says so).
    /// </summary>
    /// <param name="exprLog2">
    /// The FULL LOG2 matrix, <c>[feature, sample]</c> - not the fitted submatrix. The prior may be
    /// fitted on replicates that take no part in the contrast (dedicated QC or reference
    /// injections), and those columns exist only here.
    /// </param>
    /// <param name="testedRows">
    /// The features the fit covers, in the order their variances were given. The returned array is
    /// parallel to this, not to <paramref name="exprLog2"/>'s rows.
    /// </param>
    /// <param name="groups">
    /// ABSOLUTE sample-column indices, one list per prior group. Normally the two contrast arms;
    /// with the QC/reference override, whichever replicates were nominated.
    /// </param>
    public static double[]? IntensityTrend(
        double[,] exprLog2, IReadOnlyList<int> testedRows, IReadOnlyList<IReadOnlyList<int>> groups)
    {
        var nFeatures = testedRows.Count;
        var stats = CollectGroupStats(exprLog2, testedRows, groups);

        // The trend is fitted only on points that can carry one; every point is then PREDICTED from
        // it, including the ones excluded from the fit, which is what the reference does.
        var fitPoints = stats
            .Where(s => s.N >= 2 && double.IsFinite(s.Mean) && double.IsFinite(s.Sd) && s.Mean > 0 && s.Sd > 0)
            .ToList();
        if (fitPoints.Count < MinTrendPoints)
            return null;

        // LOWESS wants x ascending, and the interpolation below needs the sorted curve anyway.
        var ordered = fitPoints.OrderBy(s => Math.Log(s.Mean)).ToList();
        var x = ordered.Select(s => Math.Log(s.Mean)).ToArray();
        var y = ordered.Select(s => 2.0 * Math.Log(s.Sd)).ToArray(); // log(variance) = 2*log(sd)
        var yhat = SmoothTrend(x, y);

        // Sample-size-weighted mean of the per-group predicted log-space variance, per feature.
        var sum = new double[nFeatures];
        var weight = new double[nFeatures];
        foreach (var s in stats)
        {
            if (!(s.N > 0) || !double.IsFinite(s.Mean) || !(s.Mean > 0))
                continue;
            // Edge-clamped interpolation onto the fitted curve, matching numpy.interp with
            // left=ys[0], right=ys[-1]: outside the fitted range the trend is held flat rather than
            // extrapolated, which a LOWESS has no basis to do.
            var logVarRaw = Stats.Interp(Math.Log(s.Mean), x, yhat);
            // Delta method for x -> log2(x): var_log2 = var_raw / mean^2 / (ln 2)^2.
            var varLog2 = Math.Exp(logVarRaw) / (s.Mean * s.Mean) / (Ln2 * Ln2);
            if (!double.IsFinite(varLog2) || !(varLog2 > 0))
                continue;
            sum[s.Feature] += varLog2 * s.N;
            weight[s.Feature] += s.N;
        }

        var prior = new double[nFeatures];
        for (var f = 0; f < nFeatures; f++)
            prior[f] = weight[f] > 0 ? sum[f] / weight[f] : double.NaN;

        // A feature with no usable group takes the mean of those that had one. Leaving it NaN would
        // silently drop the feature from the result at the p-value step.
        var known = prior.Where(double.IsFinite).ToArray();
        if (known.Length == 0)
            return null;
        var fallback = known.Average();
        for (var f = 0; f < nFeatures; f++)
            if (!double.IsFinite(prior[f]) || !(prior[f] > 0))
                prior[f] = fallback;

        return prior;
    }


    /// <summary>
    /// DEqMS (Zhu 2020): a LOWESS of log(residual variance) on log(peptide count), giving a
    /// per-feature prior scale. Null when too few features carry a usable count.
    /// </summary>
    /// <remarks>
    /// <para>The insight DEqMS adds over an intensity trend is that a protein rolled up from many
    /// peptides is better determined than one rolled up from few, at the SAME intensity. That is
    /// information the abundance alone does not carry, and it is why the two priors are worth having
    /// separately rather than one standing in for the other.</para>
    /// <para>Protein level only: a peptide has no peptide count. The caller reports that rather than
    /// offering the option where it cannot mean anything.</para>
    /// <para>Unlike <see cref="IntensityTrend"/> this fits in the SAME space the model was fitted in
    /// - the residual variances are already log2-scale - so there is no delta-method conversion.</para>
    /// </remarks>
    public static double[]? PeptideCountTrend(
        ReadOnlySpan<double> variances, IReadOnlyList<double> peptideCounts)
    {
        var n = variances.Length;
        if (peptideCounts.Count != n)
            return null;

        // counts >= 1, not > 0: the reference requires a whole peptide, and a fractional count
        // would otherwise be fitted as though it were real.
        var usable = new List<int>(n);
        for (var i = 0; i < n; i++)
            if (double.IsFinite(variances[i]) && variances[i] > 0
                && double.IsFinite(peptideCounts[i]) && peptideCounts[i] >= 1)
                usable.Add(i);

        if (usable.Count < MinTrendPoints)
            return null;

        var ordered = usable.OrderBy(i => Math.Log(peptideCounts[i])).ToList();
        var x = ordered.Select(i => Math.Log(peptideCounts[i])).ToArray();
        var y = new double[ordered.Count];
        for (var k = 0; k < ordered.Count; k++)
            y[k] = Math.Log(variances[ordered[k]]);

        var yhat = SmoothTrend(x, y);

        // The fallback is the mean of the valid LOG VARIANCES, not the mean of the fitted curve -
        // the reference's `global_log_s0`. The two are close but not equal, and a feature with no
        // count takes this one.
        var fallback = Math.Exp(y.Average());

        var prior = new double[n];
        for (var i = 0; i < n; i++)
        {
            if (!double.IsFinite(peptideCounts[i]) || !(peptideCounts[i] >= 1))
            {
                prior[i] = fallback;
                continue;
            }

            prior[i] = Math.Exp(Stats.Interp(Math.Log(peptideCounts[i]), x, yhat));
            if (!double.IsFinite(prior[i]) || !(prior[i] > 0))
                prior[i] = fallback;
        }

        return prior;
    }

    private const double Ln2 = 0.6931471805599453;

    /// <summary>
    /// Per (feature, group) raw-scale mean, sd (ddof = 1) and count, from the log2 matrix. Groups with
    /// fewer than two samples contribute nothing - an sd needs two.
    /// </summary>
    private static List<GroupStat> CollectGroupStats(
        double[,] exprLog2, IReadOnlyList<int> testedRows, IReadOnlyList<IReadOnlyList<int>> groups)
    {
        var nFeatures = testedRows.Count;
        var stats = new List<GroupStat>(nFeatures * Math.Max(groups.Count, 1));
        var buffer = new List<double>();

        foreach (var cols in groups)
        {
            if (cols.Count < 2)
                continue;
            for (var f = 0; f < nFeatures; f++)
            {
                buffer.Clear();
                foreach (var c in cols)
                {
                    var v = exprLog2[testedRows[f], c];
                    if (double.IsFinite(v))
                        buffer.Add(Math.Pow(2.0, v)); // back to the linear scale the parquet held
                }

                if (buffer.Count < 2)
                {
                    stats.Add(new GroupStat(f, double.NaN, double.NaN, buffer.Count));
                    continue;
                }

                var mean = 0.0;
                foreach (var v in buffer)
                    mean += v;
                mean /= buffer.Count;

                var ss = 0.0;
                foreach (var v in buffer)
                {
                    var d = v - mean;
                    ss += d * d;
                }

                stats.Add(new GroupStat(f, mean, Math.Sqrt(ss / (buffer.Count - 1)), buffer.Count));
            }
        }

        return stats;
    }
}
