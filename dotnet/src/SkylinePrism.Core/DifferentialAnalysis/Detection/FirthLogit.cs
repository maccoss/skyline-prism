using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace SkylinePrism.Core.DifferentialAnalysis.Detection;

/// <summary>Result of a <see cref="FirthLogit"/> fit.</summary>
public sealed record FirthResult(double[] Beta, double PenalizedLogLik, bool Converged);

/// <summary>
/// Firth-penalized logistic regression (Jeffreys-prior bias reduction), ported from the PRISM
/// Differential Explorer's <c>_firth_logit</c>. Gives finite coefficient estimates under
/// complete/quasi-separation (common at small n) where the ordinary MLE diverges. Newton steps on the
/// penalized log-likelihood with overshoot damping; the returned log-likelihood is the penalized one
/// evaluated at the final coefficients.
/// </summary>
public static class FirthLogit
{
    /// <summary>
    /// Fit <paramref name="y"/> (0/1) on the design <paramref name="x"/> (rows = observations, columns =
    /// predictors, an intercept column included by the caller). Returns the coefficients, the penalized
    /// log-likelihood at those coefficients, and whether the iteration converged.
    /// </summary>
    public static FirthResult Fit(double[,] x, double[] y, int maxIter = 500, double tol = 1e-7)
    {
        var n = x.GetLength(0);
        var p = x.GetLength(1);
        if (y.Length != n)
            throw new ArgumentException($"y has {y.Length} entries but x has {n} rows.", nameof(y));

        var xm = DenseMatrix.OfArray(x);
        var yv = DenseVector.OfArray(y);
        Vector<double> beta = new DenseVector(p);
        var llPrev = double.NegativeInfinity;
        var converged = false;

        for (var it = 0; it < maxIter; it++)
        {
            var (pr, w, inv, ll) = PenalizedLogLik(xm, yv, beta);
            if (it > 0 && Math.Abs(ll - llPrev) < tol)
            {
                converged = true;
                break;
            }

            llPrev = ll;

            // Hat-matrix diagonal h_i = w_i * x_i^T inv x_i.
            var adj = new double[n];
            for (var i = 0; i < n; i++)
            {
                var xi = xm.Row(i);
                var hi = w[i] * xi.DotProduct(inv * xi);
                adj[i] = yv[i] - pr[i] + hi * (0.5 - pr[i]);
            }

            // step = inv (X^T adj), the Firth-modified Newton step.
            var step = inv * xm.TransposeThisAndMultiply(DenseVector.OfArray(adj));
            if (!step.Enumerate().All(double.IsFinite))
                break;

            var maxAbs = step.AbsoluteMaximum();
            if (maxAbs > 5.0)
                step = step * (5.0 / maxAbs);

            beta += step;
        }

        var (_, _, _, finalLl) = PenalizedLogLik(xm, yv, beta);
        return new FirthResult(beta.ToArray(), finalLl, converged);
    }

    private static (DenseVector Pr, DenseVector W, Matrix<double> Inv, double Ll) PenalizedLogLik(
        Matrix<double> x, Vector<double> y, Vector<double> beta)
    {
        var n = x.RowCount;
        var eta = x * beta;
        var pr = new DenseVector(n);
        var w = new DenseVector(n);
        for (var i = 0; i < n; i++)
        {
            var p = 1.0 / (1.0 + Math.Exp(-eta[i]));
            if (p < 1e-12)
                p = 1e-12;
            else if (p > 1.0 - 1e-12)
                p = 1.0 - 1e-12;
            pr[i] = p;
            w[i] = p * (1.0 - p);
        }

        // XtWX = X^T diag(w) X.
        var scaled = DenseMatrix.Create(n, x.ColumnCount, (i, j) => x[i, j] * w[i]);
        var xtwx = x.TransposeThisAndMultiply(scaled);

        var inv = xtwx.Inverse();
        if (!inv.Enumerate().All(double.IsFinite))
            inv = xtwx.PseudoInverse();

        // np.linalg.slogdet returns log|det| regardless of sign; only an exactly-singular matrix is -inf.
        var det = xtwx.Determinant();
        var logdet = det == 0.0 ? double.NegativeInfinity : Math.Log(Math.Abs(det));

        double ll = 0;
        for (var i = 0; i < n; i++)
            ll += y[i] * Math.Log(pr[i]) + (1.0 - y[i]) * Math.Log(1.0 - pr[i]);
        ll += 0.5 * logdet;

        return (pr, w, inv, ll);
    }
}
