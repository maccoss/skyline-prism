using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Double;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.DifferentialAnalysis;

/// <summary>Result of <see cref="DuplicateCorrelation.Estimate"/>.</summary>
public sealed class DuplicateCorrelationResult
{
    internal DuplicateCorrelationResult(double consensus, double[] atanhCorrelations, string? degenerate)
    {
        Consensus = consensus;
        AtanhCorrelations = atanhCorrelations;
        Degenerate = degenerate;
    }

    /// <summary>
    /// The intra-block correlation shared by every feature: <c>tanh</c> of the 15%-trimmed mean of
    /// <see cref="AtanhCorrelations"/>. NaN when no feature yielded an estimate.
    /// </summary>
    public double Consensus { get; }

    /// <summary>
    /// Each feature's REML correlation on the atanh scale, after limma's bounds; NaN where limma
    /// estimates none (too few observations, too few blocks, or a fit that failed).
    /// </summary>
    public double[] AtanhCorrelations { get; }

    /// <summary>
    /// Why the correlation was set to 0 without being estimated, or null when it was estimated:
    /// every block has one sample, or the blocks are already encoded in the design.
    /// </summary>
    public string? Degenerate { get; }
}

/// <summary>
/// limma's <c>duplicateCorrelation(M, design, block = block)</c>: one correlation between samples of
/// the same block (subject), shared by every feature. Each feature gets a REML estimate of its
/// block and residual variance components; their ratio is bounded, moved to the atanh scale, and
/// averaged with 15% trimmed from each end. Smyth, Michaud &amp; Scott 2005, Bioinformatics
/// 21(9):2067-2075.
/// </summary>
/// <remarks>
/// <para><b>This replays limma's arithmetic, not just its model.</b> limma asks statmod's
/// <c>mixedModel2Fit</c> for the variance components with <c>maxit = 20</c> and statmod's default
/// tolerance of 1e-6 on the score step, so its per-feature values are NOT the REML optimum: on the
/// committed golden they sit up to 2.4e-3 from it on the atanh scale where the likelihood is flat.
/// An optimizer that reached the true optimum by another route would therefore disagree with limma
/// by that much. So the steps are statmod's own: project the response and the block indicators
/// onto the residual space of the design, diagonalize, regress the squared projections on
/// <c>[1, d]</c>, then refine with statmod's Levenberg-damped gamma GLM (<c>glmgam.fit</c>),
/// stopping where it stops. <c>blocked.json</c> pins every quantity.</para>
/// <para>One shortcut, and why it changes nothing: statmod rotates the residual space with the
/// Householder Q of <c>lm.fit</c> and then takes an SVD; this takes any orthonormal basis of the
/// same space (from an SVD of the design). Where two singular values are equal the individual
/// squared projections then differ between the two, but the least squares start, the score, the
/// information and every comparison between two deviances depend on them only through their sum
/// within each run of equal values, which is basis-free (the <c>log y</c> terms cancel in a
/// difference). Two things read a single deviance or a single response and so are not basis-free:
/// the stop on <c>deviance / max(mu) &lt; 1e-15</c>, and the zero test in <see cref="Deviance"/>.
/// Both need an exact fit or a zero residual, which real data does not produce; on the golden the
/// per-feature values agree with limma to 4e-14.</para>
/// <para>Missing values are handled as limma handles them: each feature is fitted on its finite
/// values only. The differential path never sends one, because it tests complete features only.</para>
/// </remarks>
public static class DuplicateCorrelation
{
    /// <summary>The fraction trimmed from EACH end before averaging, limma's default.</summary>
    public const double Trim = 0.15;

    /// <summary>limma's upper bound on a per-feature correlation.</summary>
    public const double RhoMax = 0.99;

    // statmod's defaults as duplicateCorrelation calls them: maxit is passed as 20, tol left at 1e-6.
    private const int MaxIterations = 20;
    private const double Tolerance = 1e-6;

    /// <summary>
    /// Estimate the consensus intra-block correlation of a LOG2 <paramref name="exprFeaturesBySamples"/>
    /// under <paramref name="designSamplesByCoef"/>, with <paramref name="block"/> naming each sample's
    /// block (subject). NaN entries are left out of that feature's fit.
    /// </summary>
    public static DuplicateCorrelationResult Estimate(
        double[,] exprFeaturesBySamples, double[,] designSamplesByCoef, IReadOnlyList<string> block)
    {
        var nFeatures = exprFeaturesBySamples.GetLength(0);
        var nSamples = exprFeaturesBySamples.GetLength(1);
        var nBeta = designSamplesByCoef.GetLength(1);
        if (designSamplesByCoef.GetLength(0) != nSamples || block.Count != nSamples)
            throw new ArgumentException("The design, the block labels and the data must cover the same samples.");

        var maxBlockSize = block.GroupBy(b => b, StringComparer.Ordinal).Max(g => g.Count());
        if (maxBlockSize == 1)
            return Zero(nFeatures, "every block (subject) has one sample, so there is no correlation to estimate");

        // limma: if the block indicators (all but the first level) lie in the column space of the
        // design, the design has already absorbed the blocks - the paired design is the case.
        var all = Enumerable.Range(0, nSamples).ToArray();
        var complement = ResidualBasis(designSamplesByCoef, all);
        var levels = block.Distinct(StringComparer.Ordinal).OrderBy(b => b, StringComparer.Ordinal).ToArray();
        var maxResidual = 0.0;
        for (var l = 1; l < levels.Length; l++)
            for (var k = 0; k < complement.GetLength(1); k++)
            {
                double dot = 0;
                for (var s = 0; s < nSamples; s++)
                    if (block[s] == levels[l])
                        dot += complement[s, k];
                maxResidual = Math.Max(maxResidual, Math.Abs(dot));
            }
        if (maxResidual < 1e-8)
            return Zero(nFeatures, "the blocks are already encoded in the design (as in a paired design)");

        // Complete features share one design and one block layout, so their projection is computed
        // once. A feature with missing values gets its own.
        Projection? shared = null;
        var rho = new double[nFeatures];
        var y = new double[nSamples];
        for (var f = 0; f < nFeatures; f++)
        {
            var observed = new List<int>(nSamples);
            for (var s = 0; s < nSamples; s++)
                if (double.IsFinite(exprFeaturesBySamples[f, s]))
                    observed.Add(s);

            rho[f] = double.NaN;
            var nObs = observed.Count;
            var nBlocks = observed.Select(s => block[s]).Distinct(StringComparer.Ordinal).Count();
            if (!(nObs > nBeta + 2 && nBlocks > 1 && nBlocks < nObs - 1))
                continue;

            Projection projection;
            if (nObs == nSamples)
                projection = shared ??= Projection.For(designSamplesByCoef, block, all);
            else
                projection = Projection.For(designSamplesByCoef, block, observed.ToArray());

            for (var i = 0; i < nObs; i++)
                y[i] = exprFeaturesBySamples[f, observed[i]];
            var varcomp = projection.VarianceComponents(y);
            if (varcomp is { } vc)
                rho[f] = vc.Block / (vc.Residual + vc.Block);
        }

        // limma's bounds. It applies each only when some estimate crosses it, which is the same as
        // applying it to every value.
        var rhoMin = 1.0 / (1 - maxBlockSize) + 0.01;
        var atanh = new double[nFeatures];
        var finite = new List<double>(nFeatures);
        for (var f = 0; f < nFeatures; f++)
        {
            var r = rho[f];
            if (double.IsNaN(r))
            {
                atanh[f] = double.NaN;
                continue;
            }

            r = Math.Min(Math.Max(r, rhoMin), RhoMax);
            atanh[f] = Math.Atanh(r);
            finite.Add(atanh[f]);
        }

        return new DuplicateCorrelationResult(Math.Tanh(TrimmedMean(finite, Trim)), atanh, null);
    }

    private static DuplicateCorrelationResult Zero(int nFeatures, string why)
        => new(0.0, new double[nFeatures], why);

    /// <summary>
    /// R's <c>mean(x, trim = trim)</c>: sort, drop <c>floor(n * trim)</c> values from each end,
    /// average the rest. NaN for no values.
    /// </summary>
    internal static double TrimmedMean(List<double> values, double trim)
    {
        var n = values.Count;
        if (n == 0)
            return double.NaN;
        var sorted = values.OrderBy(v => v).ToArray();
        var lo = (int)Math.Floor(n * trim);
        var kept = new double[n - 2 * lo];
        Array.Copy(sorted, lo, kept, 0, kept.Length);
        return NumpyMath.Mean(kept);
    }

    /// <summary>
    /// An orthonormal basis (columns) of the space orthogonal to the design's columns, restricted
    /// to <paramref name="rows"/> - what statmod reaches through the trailing rows of <c>lm.fit</c>'s
    /// effects.
    /// </summary>
    private static double[,] ResidualBasis(double[,] design, int[] rows)
    {
        var n = rows.Length;
        var p = design.GetLength(1);
        var x = new DenseMatrix(n, p);
        for (var i = 0; i < n; i++)
            for (var c = 0; c < p; c++)
                x[i, c] = design[rows[i], c];

        var svd = x.Svd(computeVectors: true);
        // Rank by the same rule as LinAlg.MatrixRank (numpy's: max singular value x max(n, p) x eps),
        // because that is the guard LinearModel.Fit applies before the GLS. R's lm.fit uses a pivoted
        // QR at 1e-7 instead, and the two differ only on a nearly collinear design; there a looser
        // cut here would estimate the correlation for a model one column smaller than the one fitted.
        var smax = svd.S.Count > 0 ? svd.S.Maximum() : 0.0;
        var tol = smax * Math.Max(n, p) * 2.220446049250313e-16;
        var rank = svd.S.Count(s => s > tol);
        var u = svd.U;
        var basis = new double[n, n - rank];
        for (var i = 0; i < n; i++)
            for (var k = rank; k < n; k++)
                basis[i, k - rank] = u[i, k];
        return basis;
    }

    /// <summary>
    /// statmod's <c>mixedModel2Fit</c> for one block factor, up to the per-feature response: the
    /// residual space of the design, rotated so the block's contribution is diagonal.
    /// </summary>
    private sealed class Projection
    {
        private readonly double[,] _rotation; // (n_obs - rank) x n_obs: rows are U' Q2'
        private readonly double[] _d;          // squared singular values, padded with zeros
        private readonly bool _refine;

        private Projection(double[,] rotation, double[] d)
        {
            _rotation = rotation;
            _d = d;
            var mq = d.Length;
            var nonzero = d.Count(v => Math.Abs(v) > 1e-15);
            // statmod: refine with the gamma GLM only when mq > 2, more than one non-zero d, and
            // the d values vary. var() is R's sample variance.
            _refine = mq > 2 && nonzero > 1 && NumpyMath.Var(d, ddof: 1) > 1e-15;
        }

        public static Projection For(double[,] design, IReadOnlyList<string> block, int[] rows)
        {
            var q2 = ResidualBasis(design, rows);
            var n = rows.Length;
            var mq = q2.GetLength(1);

            // Z: one indicator per block level present in these rows (model.matrix(~0 + A)).
            var levels = rows.Select(r => block[r]).Distinct(StringComparer.Ordinal)
                .OrderBy(b => b, StringComparer.Ordinal).ToArray();
            var qtz = new DenseMatrix(mq, levels.Length);
            for (var l = 0; l < levels.Length; l++)
                for (var k = 0; k < mq; k++)
                {
                    double dot = 0;
                    for (var i = 0; i < n; i++)
                        if (block[rows[i]] == levels[l])
                            dot += q2[i, k];
                    qtz[k, l] = dot;
                }

            var d = new double[mq];
            var rotation = new double[mq, n];
            if (mq == 0)
                return new Projection(rotation, d);

            var svd = qtz.Svd(computeVectors: true);
            for (var k = 0; k < svd.S.Count && k < mq; k++)
                d[k] = svd.S[k] * svd.S[k];

            // rotation = U' Q2', so rotation . y is statmod's crossprod(s$u, Q2' y).
            var u = svd.U;
            for (var j = 0; j < mq; j++)
                for (var i = 0; i < n; i++)
                {
                    double v = 0;
                    for (var k = 0; k < mq; k++)
                        v += u[k, j] * q2[i, k];
                    rotation[j, i] = v;
                }

            return new Projection(rotation, d);
        }

        /// <summary>The (residual, block) variance components, or null where statmod's would be NA.</summary>
        public (double Residual, double Block)? VarianceComponents(double[] y)
        {
            var mq = _d.Length;
            if (mq == 0)
                return null;

            var dy = new double[mq];
            for (var j = 0; j < mq; j++)
            {
                double v = 0;
                for (var i = 0; i < _rotation.GetLength(1); i++)
                    v += _rotation[j, i] * y[i];
                dy[j] = v * v;
            }

            // lm.fit(cbind(1, d), dy). R pivots out a column that is (numerically) a multiple of
            // the first and reports NA for it, which makes the correlation NA too.
            var start = LeastSquaresStart(_d, dy);
            if (start is null)
                return null;

            if (!_refine)
                return start;

            var (b0, b1) = start.Value;
            var allNonNegative = true;
            for (var j = 0; j < mq; j++)
                if (b0 + b1 * _d[j] < 0)
                    allNonNegative = false;
            if (!allNonNegative)
                (b0, b1) = (NumpyMath.Mean(dy), 0.0);

            return GammaGlm(_d, dy, b0, b1);
        }

        private static (double Residual, double Block)? LeastSquaresStart(double[] d, double[] dy)
        {
            var n = d.Length;
            var meanD = NumpyMath.Mean(d);
            double ssd = 0, normD = 0;
            for (var j = 0; j < n; j++)
            {
                ssd += (d[j] - meanD) * (d[j] - meanD);
                normD += d[j] * d[j];
            }

            // dqrdc2's test: the second column is deficient when what is left of it after removing
            // the first is under 1e-7 of its original norm.
            if (n < 2 || !(Math.Sqrt(ssd) >= 1e-7 * Math.Sqrt(normD)) || normD == 0)
                return null;

            var meanY = NumpyMath.Mean(dy);
            double sxy = 0;
            for (var j = 0; j < n; j++)
                sxy += (d[j] - meanD) * (dy[j] - meanY);
            var slope = sxy / ssd;
            return (meanY - slope * meanD, slope);
        }
    }

    /// <summary>
    /// statmod's <c>glmgam.fit(cbind(1, d), dy, coef.start)</c>: a gamma GLM with identity link,
    /// fitted by Levenberg-damped Fisher scoring, with statmod's damping schedule and stopping rule.
    /// </summary>
    private static (double Residual, double Block)? GammaGlm(double[] d, double[] y, double b0, double b1)
    {
        var n = d.Length;
        var mu = new double[n];
        void Fitted(double c0, double c1)
        {
            for (var j = 0; j < n; j++)
                mu[j] = c0 + c1 * d[j];
        }

        Fitted(b0, b1);
        if (mu.Any(m => m < 0))
            return null; // "Starting values give negative fitted values"

        var dev = Deviance(y, mu);
        var lambda = 0.0;
        var v = new double[n];
        for (var iter = 1; ; iter++)
        {
            var maxV = 0.0;
            for (var j = 0; j < n; j++)
            {
                v[j] = mu[j] * mu[j];
                maxV = Math.Max(maxV, v[j]);
            }
            for (var j = 0; j < n; j++)
                v[j] = Math.Max(v[j], maxV / 1e3);

            // XVX = X' diag(1/v) X and dl = X' ((y - mu) / v), X = [1, d].
            double a00 = 0, a01 = 0, a11 = 0, g0 = 0, g1 = 0;
            for (var j = 0; j < n; j++)
            {
                var w = 1.0 / v[j];
                a00 += w;
                a01 += w * d[j];
                a11 += w * d[j] * d[j];
                var r = (y[j] - mu[j]) / v[j];
                g0 += r;
                g1 += r * d[j];
            }

            var maxInfo = Math.Max(a00, a11);
            if (iter == 1)
                lambda = Math.Abs((a00 + a11) / 2.0) / 2.0;

            var (old0, old1, devOld) = (b0, b1, dev);
            double db0 = 0, db1 = 0;
            var lev = 0;
            while (true)
            {
                lev++;
                // (XVX + lambda I) dbeta = dl, by Cholesky as statmod does.
                var m00 = a00 + lambda;
                var m11 = a11 + lambda;
                var l00 = Math.Sqrt(m00);
                var l10 = a01 / l00;
                var l11 = Math.Sqrt(m11 - l10 * l10);
                if (!(l11 > 0) || double.IsNaN(l00))
                    return null; // chol() refuses a matrix that is not positive definite
                var z0 = g0 / l00;
                var z1 = (g1 - l10 * z0) / l11;
                db1 = z1 / l11;
                db0 = (z0 - l10 * db1) / l00;

                b0 = old0 + db0;
                b1 = old1 + db1;
                Fitted(b0, b1);
                dev = Deviance(y, mu);
                if (dev <= devOld || dev / mu.Max() < 1e-15)
                    break;
                if (lambda / maxInfo > 1e15)
                {
                    (b0, b1) = (old0, old1);
                    break;
                }

                lambda *= 2;
            }

            // Checked here as well as inside, as statmod does: a step accepted after the damping has
            // already passed the limit still ends the fit, keeping that step.
            if (lambda / maxInfo > 1e15)
                break;
            if (lev == 1)
                lambda /= 10;
            if (g0 * db0 + g1 * db1 < Tolerance || dev / mu.Max() < 1e-15)
                break;
            if (iter > MaxIterations)
                break;
        }

        return (b0, b1);
    }

    /// <summary>statmod's <c>deviance.gamma</c>, including its handling of zero responses.</summary>
    private static double Deviance(double[] y, double[] mu)
    {
        if (mu.Any(m => m < 0))
            return double.PositiveInfinity;

        double dev = 0;
        var any = false;
        var all = true;
        for (var j = 0; j < y.Length; j++)
        {
            if (y[j] < 1e-15 && mu[j] < 1e-15)
            {
                any = true;
                continue;
            }

            all = false;
            dev += (y[j] - mu[j]) / mu[j] - Math.Log(y[j] / mu[j]);
        }

        return any && all ? 0.0 : 2 * dev;
    }
}
