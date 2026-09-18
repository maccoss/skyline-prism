using System;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra.Factorization;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.DifferentialAnalysis;

/// <summary>
/// Result of a shared-design per-feature ordinary least-squares fit (limma's <c>lmFit</c>).
/// Because the design matrix is shared across features (the differential path pre-filters to
/// features observed in every selected sample), the unscaled standard errors depend only on the
/// design and are the same for every feature, so they are stored once.
/// </summary>
public sealed class LinearModelFit
{
    internal LinearModelFit(double[,] coefficients, double[] sigma, double[] amean,
        double[] stdevUnscaled, double dfResidual)
    {
        Coefficients = coefficients;
        Sigma = sigma;
        Amean = amean;
        StdevUnscaled = stdevUnscaled;
        DfResidual = dfResidual;
    }

    /// <summary>Least-squares coefficients, <c>[feature, coefficient]</c>.</summary>
    public double[,] Coefficients { get; }

    /// <summary>Residual standard deviation per feature: <c>sqrt(SSR / df_residual)</c>.</summary>
    public double[] Sigma { get; }

    /// <summary>Row (feature) mean of the response, limma's <c>Amean</c>.</summary>
    public double[] Amean { get; }

    /// <summary>
    /// Unscaled standard error of each coefficient: <c>sqrt(diag((X^T X)^-1))</c>. Shared across
    /// features because the design is shared.
    /// </summary>
    public double[] StdevUnscaled { get; }

    /// <summary>
    /// Residual degrees of freedom, <c>n_samples - n_coef</c>. The design is required full column
    /// rank, so <c>rank(design) == n_coef</c> and this equals <c>n_samples - rank(design)</c>.
    /// </summary>
    public double DfResidual { get; }
}

/// <summary>
/// Per-feature ordinary least squares with a design matrix shared across features - the
/// <c>lmFit</c> step of the limma moderated-t pipeline ported from the PRISM Differential Explorer
/// (prism_diff_explorer.py). Operates on a LOG2 abundance matrix. The differential path filters to
/// features that are observed (non-NaN) in every selected sample before fitting, so rows are
/// expected to be complete; a NaN in the response will propagate to that feature's outputs.
/// </summary>
public static class LinearModel
{
    /// <summary>
    /// Fit every feature (row) of <paramref name="dataFeaturesBySamples"/> (LOG2) against the shared
    /// <paramref name="designSamplesByCoef"/>. The design must be full column rank; a rank-deficient
    /// design throws (matching the Python reference, which refuses before fitting rather than
    /// returning NaN/Infinity). <c>df_residual</c> is <c>n_samples - n_coef</c>. The solve uses a thin
    /// QR of the design (as R limma's lmFit does) so it stays accurate on ill-conditioned covariate
    /// designs, where forming the normal equations would square the condition number.
    /// </summary>
    public static LinearModelFit Fit(double[,] dataFeaturesBySamples, double[,] designSamplesByCoef)
    {
        var nFeatures = dataFeaturesBySamples.GetLength(0);
        var nSamples = dataFeaturesBySamples.GetLength(1);
        var nCoef = designSamplesByCoef.GetLength(1);
        if (designSamplesByCoef.GetLength(0) != nSamples)
            throw new ArgumentException(
                $"Design has {designSamplesByCoef.GetLength(0)} rows but the data has {nSamples} samples.",
                nameof(designSamplesByCoef));
        if (nSamples <= nCoef)
            throw new ArgumentException(
                $"Need more samples ({nSamples}) than coefficients ({nCoef}) for a residual fit.",
                nameof(designSamplesByCoef));
        if (LinAlg.MatrixRank(designSamplesByCoef) < nCoef)
            throw new ArgumentException(
                "Design matrix is rank-deficient (not full column rank).",
                nameof(designSamplesByCoef));

        var x = DenseMatrix.OfArray(designSamplesByCoef);
        // Thin QR: X = Q R, Q is n_samples x n_coef, R is n_coef x n_coef upper-triangular.
        var qr = x.QR(QRMethod.Thin);
        var rInv = qr.R.Inverse();

        // stdev_unscaled[c] = sqrt(diag((X^T X)^-1))[c]; (X^T X)^-1 = R^-1 R^-T, so its c-th diagonal
        // element is the squared norm of row c of R^-1.
        var stdevUnscaled = new double[nCoef];
        for (var c = 0; c < nCoef; c++)
        {
            double ss = 0;
            for (var k = 0; k < nCoef; k++)
                ss += rInv[c, k] * rInv[c, k];
            stdevUnscaled[c] = Math.Sqrt(ss);
        }

        // P = R^-1 Q^T  (n_coef x n_samples): beta_feature = P . y_feature.
        var p = (rInv * qr.Q.Transpose()).ToArray();

        var dfResidual = (double)(nSamples - nCoef);
        var coefficients = new double[nFeatures, nCoef];
        var sigma = new double[nFeatures];
        var amean = new double[nFeatures];

        var beta = new double[nCoef];
        var y = new double[nSamples];
        for (var i = 0; i < nFeatures; i++)
        {
            for (var s = 0; s < nSamples; s++)
                y[s] = dataFeaturesBySamples[i, s];

            for (var c = 0; c < nCoef; c++)
            {
                double b = 0;
                for (var s = 0; s < nSamples; s++)
                    b += p[c, s] * y[s];
                beta[c] = b;
                coefficients[i, c] = b;
            }

            double ssr = 0;
            for (var s = 0; s < nSamples; s++)
            {
                double fitted = 0;
                for (var c = 0; c < nCoef; c++)
                    fitted += designSamplesByCoef[s, c] * beta[c];
                var r = y[s] - fitted;
                ssr += r * r;
            }

            sigma[i] = Math.Sqrt(ssr / dfResidual);
            amean[i] = NumpyMath.Mean(y);
        }

        return new LinearModelFit(coefficients, sigma, amean, stdevUnscaled, dfResidual);
    }
}
