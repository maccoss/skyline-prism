using System;
using System.Collections.Generic;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra.Factorization;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.DifferentialAnalysis;

/// <summary>Result of <see cref="EmpiricalBayes.SqueezeVarGlobal"/>.</summary>
public sealed class SqueezeVarResult
{
    internal SqueezeVarResult(double[] varPost, double[] varPrior, double dfPrior,
        IReadOnlyList<string> warnings)
    {
        VarPost = varPost;
        VarPrior = varPrior;
        DfPrior = dfPrior;
        Warnings = warnings;
    }

    /// <summary>Posterior (shrunk) variances, one per input variance.</summary>
    public double[] VarPost { get; }

    /// <summary>
    /// Prior scale <c>s0^2</c> of the fitted scaled-F distribution. Length 1 for the global prior; the
    /// (not-yet-implemented) intensity-trend prior returns one value per feature, which is why this is
    /// an array rather than a scalar.
    /// </summary>
    public double[] VarPrior { get; }

    /// <summary>Prior degrees of freedom <c>d0</c>. May be <see cref="double.PositiveInfinity"/>.</summary>
    public double DfPrior { get; }

    /// <summary>
    /// Diagnostic messages matching inmoose/limma's fitFDist warnings (e.g. residual variances mostly
    /// zero, or zero variances offset away from zero). Empty when the fit was unremarkable.
    /// </summary>
    public IReadOnlyList<string> Warnings { get; }
}

/// <summary>
/// limma empirical-Bayes variance moderation, ported from inmoose's squeezeVar/fitFDist (limma
/// 3.55.1). This is the <c>global</c> (no-covariate) prior: a single scaled-F is moment-matched to the
/// residual variances, giving a common prior scale and prior df, and each feature's variance is
/// shrunk toward that prior. The intensity-trend prior (covariate spline) is a separate path.
/// Degrees of freedom are constant across features in the differential path (shared design), so this
/// takes a scalar <c>df_residual</c>.
/// </summary>
public static class EmpiricalBayes
{
    /// <summary>
    /// Argument above which the digamma-family asymptotic series is used directly; below it the
    /// recurrence pushes the argument up first. 30 keeps <see cref="Trigamma"/> and
    /// <see cref="Tetragamma"/> at ~1e-14 relative accuracy, which matters because
    /// <see cref="TrigammaInverse"/> amplifies absolute error in its argument by roughly y^2 - a
    /// lower threshold (e.g. 12) leaves ~1e-12 relative error that shows up as ~1e-8 error on a large
    /// <c>df_prior</c>.
    /// </summary>
    private const double AsymptoticThreshold = 30.0;

    /// <summary>
    /// Squeeze residual variances toward a global scaled-F prior. <paramref name="variances"/> are the
    /// per-feature residual variances (<c>sigma^2</c>); <paramref name="dfResidual"/> is the shared
    /// residual degrees of freedom (> 0). Throws when no variance is usable (e.g. all NaN); note the
    /// Python reference instead returns NaN there, so callers must not rely on that value.
    /// </summary>
    public static SqueezeVarResult SqueezeVarGlobal(ReadOnlySpan<double> variances, double dfResidual)
    {
        var n = variances.Length;
        if (n == 0)
            throw new ArgumentException("variances is empty", nameof(variances));
        if (!(dfResidual > 1e-15) || double.IsInfinity(dfResidual))
            throw new ArgumentException("dfResidual must be a finite value > 0", nameof(dfResidual));

        var warnings = new List<string>();

        // fitFDist: keep finite, non-negative variances (x > -1e-15). df is a positive scalar, so its
        // own ok-check passes for every feature.
        var kept = new double[n];
        var nok = 0;
        for (var i = 0; i < n; i++)
        {
            var v = variances[i];
            if (!double.IsNaN(v) && !double.IsInfinity(v) && v > -1e-15)
                kept[nok++] = v;
        }

        double s20, df2;
        if (nok == 0)
        {
            throw new InvalidOperationException("Could not estimate prior df: no usable variances");
        }
        else if (nok == 1)
        {
            s20 = kept[0];
            df2 = 0.0;
        }
        else
        {
            // Clip negatives to 0 BEFORE the median, then floor at 1e-5 * median (matching fitFDist).
            var x = new double[nok];
            for (var i = 0; i < nok; i++)
                x[i] = kept[i] < 0.0 ? 0.0 : kept[i];

            var m = Stats.NanMedian(x.AsSpan(0, nok));
            var anyZero = false;
            for (var i = 0; i < nok; i++)
                if (x[i] == 0.0)
                {
                    anyZero = true;
                    break;
                }

            if (m == 0.0)
            {
                m = 1.0;
                warnings.Add("More than half of residual variances are exactly zero: eBayes unreliable");
            }
            else if (anyZero)
            {
                warnings.Add("Zero sample variances detected, have been offset away from zero");
            }

            var floor = 1e-5 * m;
            for (var i = 0; i < nok; i++)
                if (x[i] < floor)
                    x[i] = floor;

            // Work on log(F). df is a scalar so digamma/trigamma(df/2) are constants.
            var halfDf = dfResidual / 2.0;
            var offset = SpecialFunctions.DiGamma(halfDf) - Math.Log(halfDf);
            var e = new double[nok];
            for (var i = 0; i < nok; i++)
                e[i] = Math.Log(x[i]) - offset;

            var emean = NumpyMath.Mean(e);
            var dev2 = new double[nok];
            for (var i = 0; i < nok; i++)
            {
                var d = e[i] - emean;
                dev2[i] = d * d;
            }
            var evar = NumpyMath.PairwiseSum(dev2) / (nok - 1);
            evar -= Trigamma(halfDf);

            if (evar > 0.0)
            {
                df2 = 2.0 * TrigammaInverse(evar);
                s20 = Math.Exp(emean + SpecialFunctions.DiGamma(df2 / 2.0) - Math.Log(df2 / 2.0));
            }
            else
            {
                df2 = double.PositiveInfinity;
                // MLE of the scale in this case is the pooled (floored) variance mean.
                s20 = NumpyMath.Mean(x);
            }
        }

        // Posterior variances over EVERY input variance.
        var varPost = new double[n];
        var dfFinite = !double.IsInfinity(df2);
        for (var i = 0; i < n; i++)
        {
            varPost[i] = dfFinite
                ? (dfResidual * variances[i] + df2 * s20) / (dfResidual + df2)
                : s20;
        }

        return new SqueezeVarResult(varPost, new[] { s20 }, df2, warnings);
    }

    /// <summary>
    /// Squeeze residual variances toward an intensity-dependent prior (limma-trend, Sartor 2006): the
    /// prior scale follows a natural-spline trend in <paramref name="covariate"/> (mean log-intensity)
    /// rather than a single global value. Ported from inmoose fitFDist's covariate path. The prior df is
    /// still a single value; the prior scale is per feature. Assumes finite variances and covariate
    /// (the caller falls back to the global prior otherwise), and falls back to global when the spline
    /// degrees of freedom collapse below 2.
    /// </summary>
    public static SqueezeVarResult SqueezeVarTrend(ReadOnlySpan<double> variances, double dfResidual,
        double[] covariate)
    {
        var n = variances.Length;
        if (n == 0)
            throw new ArgumentException("variances is empty", nameof(variances));
        if (!(dfResidual > 1e-15) || double.IsInfinity(dfResidual))
            throw new ArgumentException("dfResidual must be a finite value > 0", nameof(dfResidual));
        if (covariate.Length != n)
            throw new ArgumentException("covariate length must match variances", nameof(covariate));

        // The trend path needs every feature usable; otherwise defer to the global prior.
        for (var i = 0; i < n; i++)
            if (double.IsNaN(variances[i]) || double.IsInfinity(variances[i]) || variances[i] <= -1e-15
                || !double.IsFinite(covariate[i]))
                return SqueezeVarGlobal(variances, dfResidual);

        var distinctCovariate = new HashSet<double>();
        foreach (var c in covariate)
            distinctCovariate.Add(c);
        var splineDf = 1 + (n >= 3 ? 1 : 0) + (n >= 6 ? 1 : 0) + (n >= 30 ? 1 : 0);
        splineDf = Math.Min(splineDf, distinctCovariate.Count);
        if (splineDf < 2)
            return SqueezeVarGlobal(variances, dfResidual);

        // Floor the variances away from zero (as fitFDist does) and move to log(F).
        var x = new double[n];
        for (var i = 0; i < n; i++)
            x[i] = variances[i] < 0.0 ? 0.0 : variances[i];
        var m = Stats.NanMedian(x);
        if (m == 0.0)
            m = 1.0;
        var floor = 1e-5 * m;
        for (var i = 0; i < n; i++)
            if (x[i] < floor)
                x[i] = floor;

        var halfDf = dfResidual / 2.0;
        var offset = SpecialFunctions.DiGamma(halfDf) - Math.Log(halfDf);
        var e = new double[n];
        for (var i = 0; i < n; i++)
            e[i] = Math.Log(x[i]) - offset;

        // Fit e on the natural-spline trend; the fitted values are the per-feature trend (emean) and the
        // residual mean square feeds the prior df.
        var design = NaturalSplineBasis.Build(covariate, splineDf, includeIntercept: true);
        var d = DenseMatrix.OfArray(design);
        var beta = d.QR(QRMethod.Thin).Solve(DenseVector.OfArray(e));
        var fitted = d * beta;
        double rss = 0;
        for (var i = 0; i < n; i++)
        {
            var r = e[i] - fitted[i];
            rss += r * r;
        }

        var evar = rss / (n - splineDf) - Trigamma(halfDf);

        double df2;
        var varPrior = new double[n];
        if (evar > 0.0)
        {
            df2 = 2.0 * TrigammaInverse(evar);
            var shift = SpecialFunctions.DiGamma(df2 / 2.0) - Math.Log(df2 / 2.0);
            for (var i = 0; i < n; i++)
                varPrior[i] = Math.Exp(fitted[i] + shift);
        }
        else
        {
            df2 = double.PositiveInfinity;
            for (var i = 0; i < n; i++)
                varPrior[i] = Math.Exp(fitted[i]);
        }

        var dfFinite = !double.IsInfinity(df2);
        var varPost = new double[n];
        for (var i = 0; i < n; i++)
            varPost[i] = dfFinite
                ? (dfResidual * variances[i] + df2 * varPrior[i]) / (dfResidual + df2)
                : varPrior[i];

        return new SqueezeVarResult(varPost, varPrior, df2, Array.Empty<string>());
    }

    /// <summary>
    /// Solve <c>trigamma(y) = x</c> for <c>y</c>, ported from inmoose's trigammaInverse (Newton on the
    /// convex, near-linear <c>1/trigamma</c>). Uses <see cref="Tetragamma"/> for the derivative since
    /// MathNet does not expose polygamma of order 2.
    /// </summary>
    internal static double TrigammaInverse(double x)
    {
        if (double.IsNaN(x))
            return x;
        if (x < 0.0)
            return double.NaN;
        if (x > 1e7)
            return 1.0 / Math.Sqrt(x);
        if (x < 1e-6)
            return 1.0 / x;

        var y = 0.5 + 1.0 / x;
        for (var it = 0; it < 50; it++)
        {
            var tri = Trigamma(y);
            var dif = tri * (1.0 - tri / x) / Tetragamma(y);
            y += dif;
            if (-dif / y < 1e-8)
                break;
        }
        return y;
    }

    /// <summary>
    /// Trigamma function (first derivative of digamma, polygamma order 1), matching
    /// scipy.special.polygamma(1, .) to ~1e-14 relative for x &gt; 0. MathNet 5.0.0 does not expose it.
    /// Recurrence up to <see cref="AsymptoticThreshold"/> then an asymptotic (Bernoulli) series.
    /// </summary>
    internal static double Trigamma(double x)
    {
        // psi'(x) = psi'(x+1) + 1/x^2, so accumulate the correction while pushing x up.
        double correction = 0.0;
        while (x < AsymptoticThreshold)
        {
            correction += 1.0 / (x * x);
            x += 1.0;
        }

        var inv = 1.0 / x;
        var a = inv * inv;      // 1/x^2
        var b3 = a * inv;       // 1/x^3
        var b5 = b3 * a;        // 1/x^5
        var b7 = b5 * a;        // 1/x^7
        var b9 = b7 * a;        // 1/x^9
        // psi'(x) ~ 1/x + 1/(2 x^2) + 1/(6 x^3) - 1/(30 x^5) + 1/(42 x^7) - 1/(30 x^9)
        var asym = inv + 0.5 * a + (1.0 / 6.0) * b3 - (1.0 / 30.0) * b5
                   + (1.0 / 42.0) * b7 - (1.0 / 30.0) * b9;
        return asym + correction;
    }

    /// <summary>
    /// Tetragamma function (second derivative of digamma, polygamma order 2), matching
    /// scipy.special.polygamma(2, .) to ~1e-14 relative for x &gt; 0. Recurrence up to
    /// <see cref="AsymptoticThreshold"/> then an asymptotic (Bernoulli) series.
    /// </summary>
    internal static double Tetragamma(double x)
    {
        // psi''(x) = psi''(x+1) - 2/x^3, so accumulate the correction while pushing x up.
        double correction = 0.0;
        while (x < AsymptoticThreshold)
        {
            correction -= 2.0 / (x * x * x);
            x += 1.0;
        }

        var inv = 1.0 / x;
        var a = inv * inv;      // 1/x^2
        var b = a * inv;        // 1/x^3
        var c = b * inv;        // 1/x^4
        var e6 = c * a;         // 1/x^6
        var e8 = e6 * a;        // 1/x^8
        var e10 = e8 * a;       // 1/x^10
        // psi''(x) ~ -(1/x^2 + 1/x^3 + 1/(2 x^4) - 1/(6 x^6) + 1/(6 x^8) - 3/(10 x^10))
        var asym = -(a + b + 0.5 * c - (1.0 / 6.0) * e6 + (1.0 / 6.0) * e8 - 0.3 * e10);
        return asym + correction;
    }
}
