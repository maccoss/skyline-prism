using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
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
    /// trend priors return one value per feature, which is why this is an array rather than a scalar.
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
        if (!(dfResidual > 1e-15) || double.IsInfinity(dfResidual))
            throw new ArgumentException("dfResidual must be a finite value > 0", nameof(dfResidual));
        var df = new double[variances.Length];
        Array.Fill(df, dfResidual);
        return SqueezeVarGlobal(variances, df);
    }

    /// <summary>
    /// <see cref="SqueezeVarGlobal(ReadOnlySpan{double}, double)"/> with a residual df per feature, as
    /// features with missing values have (<see cref="LinearModel.FitObserved"/>).
    /// </summary>
    /// <remarks>
    /// This is <c>fitFDist</c>'s moment fit over unequal df: each log variance is centered by its own
    /// <c>digamma(df/2) - log(df/2)</c>, and the sampling variance subtracted is the MEAN of
    /// <c>trigamma(df/2)</c>. limma 3.58 and later switch to <c>fitFDistUnequalDF1</c> when the df
    /// differ; this keeps <c>fitFDist</c>, as limma did before 3.58 (<c>eBayes(legacy = TRUE)</c>), as
    /// <c>proteomics-toolkit</c> does, and as <c>limma_ebayes.json</c>'s missing-value cases pin. With
    /// every df equal it is the scalar computation exactly. A feature whose df is not positive (one
    /// that could not be fitted) takes no part in the fit, and its posterior is the prior's when
    /// <c>d0</c> is infinite and NaN otherwise, as limma's <c>squeezeVar</c> gives it.
    /// </remarks>
    public static SqueezeVarResult SqueezeVarGlobal(ReadOnlySpan<double> variances, ReadOnlySpan<double> dfResidual)
    {
        var n = variances.Length;
        if (n == 0)
            throw new ArgumentException("variances is empty", nameof(variances));
        if (dfResidual.Length != n)
            throw new ArgumentException("dfResidual length must match variances", nameof(dfResidual));

        var warnings = new List<string>();

        // fitFDist: keep finite, non-negative variances (x > -1e-15) with a finite df > 1e-15.
        var kept = new double[n];
        var keptDf = new double[n];
        var nok = 0;
        for (var i = 0; i < n; i++)
        {
            var v = variances[i];
            var d = dfResidual[i];
            if (!double.IsNaN(v) && !double.IsInfinity(v) && v > -1e-15 && d > 1e-15 && double.IsFinite(d))
            {
                kept[nok] = v;
                keptDf[nok++] = d;
            }
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

            // Work on log(F), each value centered by its own df (one constant when the df are shared).
            var e = new double[nok];
            for (var i = 0; i < nok; i++)
                e[i] = Math.Log(x[i]) - LogChiSquareOffset(keptDf[i]);

            var emean = NumpyMath.Mean(e);
            var dev2 = new double[nok];
            for (var i = 0; i < nok; i++)
            {
                var d = e[i] - emean;
                dev2[i] = d * d;
            }
            var evar = NumpyMath.PairwiseSum(dev2) / (nok - 1);
            evar -= MeanTrigamma(keptDf.AsSpan(0, nok));

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
                ? (dfResidual[i] * variances[i] + df2 * s20) / (dfResidual[i] + df2)
                : s20;
        }

        return new SqueezeVarResult(varPost, new[] { s20 }, df2, warnings);
    }

    /// <summary>
    /// The expected log of a chi-square variance at <paramref name="df"/> degrees of freedom, relative to
    /// its log mean: <c>digamma(df/2) - log(df/2)</c>, which fitFDist subtracts from each log variance.
    /// </summary>
    private static double LogChiSquareOffset(double df)
    {
        var half = df / 2.0;
        return SpecialFunctions.DiGamma(half) - Math.Log(half);
    }

    /// <summary>
    /// fitFDist's sampling variance of a log chi-square, <c>mean(trigamma(df/2))</c>: one trigamma when
    /// every df is the same, so a shared-design fit is the scalar computation to the bit.
    /// </summary>
    private static double MeanTrigamma(ReadOnlySpan<double> df)
    {
        var shared = true;
        for (var i = 1; i < df.Length; i++)
            if (df[i] != df[0])
            {
                shared = false;
                break;
            }

        if (shared)
            return Trigamma(df[0] / 2.0);
        var values = new double[df.Length];
        for (var i = 0; i < df.Length; i++)
            values[i] = Trigamma(df[i] / 2.0);
        return NumpyMath.Mean(values);
    }

    /// <summary>
    /// The empirical-Bayes posterior for a prior whose SCALE is already known per feature and whose
    /// degrees of freedom are not re-estimated: <c>(d0*s0_i^2 + d*s_i^2) / (d0 + d)</c>, collapsing to
    /// <c>s0_i^2</c> as <c>d0 -> inf</c>.
    /// </summary>
    /// <remarks>
    /// Exists so the priors whose scale comes from <see cref="VariancePriors"/> compose the posterior
    /// through the same arithmetic as every other path rather than restating it. Restating it is how
    /// two priors end up shrinking by subtly different amounts. The <paramref name="dfPrior"/> is the
    /// caller's: the peptide-count prior passes limma's global one, as DEqMS does, and the intensity
    /// trend passes the one fitted around its scaled trend (<c>Differential.WithCalibratedLevel</c>).
    /// </remarks>
    /// <param name="varPrior">Per-feature prior scale, one per entry of <paramref name="variances"/>.</param>
    /// <param name="dfPrior">The prior degrees of freedom to pair with it; may be infinite.</param>
    public static SqueezeVarResult SqueezeVarWithScale(
        ReadOnlySpan<double> variances, double dfResidual, double[] varPrior, double dfPrior,
        IReadOnlyList<string>? warnings = null)
    {
        var df = new double[variances.Length];
        Array.Fill(df, dfResidual);
        return SqueezeVarWithScale(variances, df, varPrior, dfPrior, warnings);
    }

    /// <summary>
    /// <see cref="SqueezeVarWithScale(ReadOnlySpan{double}, double, double[], double, IReadOnlyList{string})"/>
    /// with a residual df per feature.
    /// </summary>
    public static SqueezeVarResult SqueezeVarWithScale(
        ReadOnlySpan<double> variances, ReadOnlySpan<double> dfResidual, double[] varPrior, double dfPrior,
        IReadOnlyList<string>? warnings = null)
    {
        var n = variances.Length;
        if (varPrior.Length != n)
            throw new ArgumentException("varPrior length must match variances", nameof(varPrior));
        if (dfResidual.Length != n)
            throw new ArgumentException("dfResidual length must match variances", nameof(dfResidual));

        var varPost = new double[n];
        for (var i = 0; i < n; i++)
            varPost[i] = double.IsInfinity(dfPrior)
                ? varPrior[i]
                : (dfPrior * varPrior[i] + dfResidual[i] * variances[i]) / (dfPrior + dfResidual[i]);

        return new SqueezeVarResult(varPost, varPrior, dfPrior, warnings ?? Array.Empty<string>());
    }

    /// <summary>
    /// Squeeze residual variances toward an intensity-dependent prior (limma-trend, Sartor 2006): the
    /// prior scale follows a natural-spline trend in <paramref name="covariate"/> (mean log-intensity)
    /// rather than a single global value. Ported from inmoose fitFDist's covariate path. The prior df is
    /// still a single value; the prior scale is per feature. Without peptide counts it falls back to
    /// the global prior on a non-finite variance or covariate, or when the spline degrees of freedom
    /// collapse below 2, as limma does; with them it refuses instead (see
    /// <paramref name="logPeptideCounts"/>).
    /// </summary>
    /// <param name="variances">Residual variance per feature.</param>
    /// <param name="dfResidual">The residual df, shared by every feature.</param>
    /// <param name="covariate">Mean log2 expression per feature (limma's Amean).</param>
    /// <param name="logPeptideCounts">
    /// Log peptide count per feature, for <see cref="VariancePrior.LimmaTrendIntensityNPeptides"/>: the
    /// same regression with a second covariate block (<see cref="PeptideCountBasis"/>) beside the
    /// intensity spline. limma's fitFDist takes one covariate; this is its algebra with the second
    /// block, nothing else changed. With it, nothing falls back: a feature the fit cannot use, or too
    /// few features for both covariates, is refused, because limma's global-prior fallback would drop
    /// the counts without a word.
    /// </param>
    public static SqueezeVarResult SqueezeVarTrend(ReadOnlySpan<double> variances, double dfResidual,
        double[] covariate, double[]? logPeptideCounts = null)
    {
        if (!(dfResidual > 1e-15) || double.IsInfinity(dfResidual))
            throw new ArgumentException("dfResidual must be a finite value > 0", nameof(dfResidual));
        var df = new double[variances.Length];
        Array.Fill(df, dfResidual);
        return SqueezeVarTrend(variances, df, covariate, logPeptideCounts);
    }

    /// <summary>
    /// <see cref="SqueezeVarTrend(ReadOnlySpan{double}, double, double[], double[])"/> with a residual
    /// df per feature, as features with missing values have. Each log variance is centered by its own
    /// df and the sampling variance subtracted is the mean trigamma, as in
    /// <see cref="SqueezeVarGlobal(ReadOnlySpan{double}, ReadOnlySpan{double})"/>; with every df equal it
    /// is the scalar computation exactly.
    /// </summary>
    public static SqueezeVarResult SqueezeVarTrend(ReadOnlySpan<double> variances, ReadOnlySpan<double> dfResidual,
        double[] covariate, double[]? logPeptideCounts = null)
    {
        var n = variances.Length;
        if (n == 0)
            throw new ArgumentException("variances is empty", nameof(variances));
        if (dfResidual.Length != n)
            throw new ArgumentException("dfResidual length must match variances", nameof(dfResidual));
        if (covariate.Length != n)
            throw new ArgumentException("covariate length must match variances", nameof(covariate));
        var withCounts = logPeptideCounts is not null;
        if (withCounts && logPeptideCounts!.Length != n)
            throw new ArgumentException("logPeptideCounts length must match variances", nameof(logPeptideCounts));

        // The trend path needs every feature usable; otherwise defer to the global prior - or, with
        // counts, refuse, since the global prior would drop them without a word.
        for (var i = 0; i < n; i++)
            if (double.IsNaN(variances[i]) || double.IsInfinity(variances[i]) || variances[i] <= -1e-15
                || !(dfResidual[i] > 1e-15) || !double.IsFinite(dfResidual[i])
                || !double.IsFinite(covariate[i]) || (withCounts && !double.IsFinite(logPeptideCounts![i])))
            {
                if (withCounts)
                    throw new ArgumentException(
                        "The limma-trend-intensity-n-peptides prior needs a finite residual variance, mean "
                        + "expression and peptide count for every tested protein.");
                return SqueezeVarGlobal(variances, dfResidual);
            }

        var splineDf = TrendSplineDf(covariate);
        if (withCounts && !CanFitIntensityAndCounts(covariate, logPeptideCounts!))
            throw new ArgumentException(
                $"Only {n} proteins have a usable residual variance: too few to fit a prior on intensity "
                + "and peptide count. Use the limma-trend or global prior instead (--prior limma-trend, or "
                + "Intensity trend (limma) in the pane).");
        var countBlock = withCounts ? PeptideCountBasis(logPeptideCounts!) : null;
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

        var e = new double[n];
        for (var i = 0; i < n; i++)
            e[i] = Math.Log(x[i]) - LogChiSquareOffset(dfResidual[i]);

        // Fit e on the natural-spline trend; the fitted values are the per-feature trend (emean) and the
        // residual mean square feeds the prior df.
        Matrix<double> d = DenseMatrix.OfArray(NaturalSplineBasis.Build(covariate, splineDf, includeIntercept: true));
        if (countBlock is not null)
        {
            d = d.Append(DenseMatrix.OfArray(countBlock));
            // The intensity spline is full rank by construction (its df never exceed the distinct
            // values), but log count could in principle lie in its span, and an unpivoted QR would
            // then return a prior of NaNs rather than an error. lm.fit pivots and drops the column;
            // refusing says what happened instead.
            if (LinAlg.MatrixRank(d.ToArray()) < d.ColumnCount)
                throw new ArgumentException(
                    "Peptide count is collinear with mean expression across the proteins tested, so the "
                    + "limma-trend-intensity-n-peptides prior cannot separate the two. Use the limma-trend "
                    + "prior instead.");
        }

        var beta = d.QR(QRMethod.Thin).Solve(DenseVector.OfArray(e));
        var fitted = d * beta;
        double rss = 0;
        for (var i = 0; i < n; i++)
        {
            var r = e[i] - fitted[i];
            rss += r * r;
        }

        // lm.fit's mean(effects[-(1:rank)]^2): the residual mean square on n - rank df, and the design
        // is full rank here (checked above), so the rank is its column count.
        var evar = rss / (n - d.ColumnCount) - MeanTrigamma(dfResidual);

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
                ? (dfResidual[i] * variances[i] + df2 * varPrior[i]) / (dfResidual[i] + df2)
                : varPrior[i];

        return new SqueezeVarResult(varPost, varPrior, df2, Array.Empty<string>());
    }

    /// <summary>
    /// limma's spline df for the intensity trend over these features:
    /// <c>1 + (n &gt;= 3) + (n &gt;= 6) + (n &gt;= 30)</c>, capped at the number of distinct values.
    /// </summary>
    private static int TrendSplineDf(double[] covariate)
    {
        var n = covariate.Length;
        var splineDf = 1 + (n >= 3 ? 1 : 0) + (n >= 6 ? 1 : 0) + (n >= 30 ? 1 : 0);
        return Math.Min(splineDf, covariate.Distinct().Count());
    }

    /// <summary>
    /// Whether these features can fit <see cref="VariancePrior.LimmaTrendIntensityNPeptides"/>: an
    /// intensity spline of at least two columns, a peptide count that varies, and more features than
    /// the two blocks have columns, so the spread around the curve - which sets the prior df - has a
    /// degree of freedom to be measured on. <c>proteomics-toolkit</c>'s
    /// <c>_can_fit_intensity_and_counts</c> is the same rule.
    /// </summary>
    internal static bool CanFitIntensityAndCounts(double[] covariate, double[] logPeptideCounts)
    {
        var splineDf = TrendSplineDf(covariate);
        if (splineDf < 2 || logPeptideCounts.Distinct().Count() < 2)
            return false;
        return covariate.Length > splineDf + PeptideCountBasis(logPeptideCounts).GetLength(1);
    }

    /// <summary>
    /// The peptide-count columns <see cref="VariancePrior.LimmaTrendIntensityNPeptides"/> adds to the
    /// intensity spline: R's <c>splines::ns(log(n), df = 2)</c> - a natural cubic spline in log count
    /// with one interior knot, at the median, and no constant column (the intensity spline carries
    /// one). With only two distinct counts no spline can be placed, so the block is log count itself,
    /// a straight line; the same holds when more than half the proteins share the smallest or largest
    /// count, putting the median knot on the end of the range. This is <c>proteomics-toolkit</c>'s
    /// <c>_peptide_count_basis</c>, which builds the same span in another basis.
    /// </summary>
    internal static double[,] PeptideCountBasis(double[] logCounts)
    {
        var distinct = logCounts.Distinct().Count();
        if (distinct < 2)
            throw new ArgumentException(
                "Every protein has the same peptide count, so peptide count cannot inform the prior.");
        var median = Stats.PercentileLinear(logCounts, 50.0);
        if (distinct >= 3 && median > logCounts.Min() && median < logCounts.Max())
            return NaturalSplineBasis.Build(logCounts, 2, includeIntercept: false);

        var line = new double[logCounts.Length, 1];
        for (var i = 0; i < logCounts.Length; i++)
            line[i, 0] = logCounts[i];
        return line;
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
