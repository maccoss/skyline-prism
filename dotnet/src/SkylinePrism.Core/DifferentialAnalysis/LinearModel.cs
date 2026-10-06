using System;
using System.Collections.Generic;
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

    /// <summary>
    /// Generalized least squares with correlated samples within each block - limma's
    /// <c>lmFit(M, design, block = block, correlation = correlation)</c> on complete data. Samples in
    /// the same block are given correlation <paramref name="correlation"/> and samples in different
    /// blocks none; every sample has the same variance.
    /// </summary>
    /// <remarks>
    /// <para>Fitted the way limma fits it: with <c>V</c> that correlation matrix and <c>V = L L'</c>
    /// its Cholesky factor, the data and the design are both multiplied by <c>L^-1</c>, which makes the
    /// samples uncorrelated, and the result goes through <see cref="Fit"/>. So the coefficients,
    /// <c>sigma</c> (from the whitened residuals) and the unscaled standard errors
    /// (<c>sqrt(diag((X' V^-1 X)^-1))</c>) are GLS ones, and the residual degrees of freedom are still
    /// <c>n - p</c>.</para>
    /// <para><see cref="LinearModelFit.Amean"/> is the mean of the ORIGINAL data, not the whitened
    /// data, as in limma - it is the intensity a trend prior is fitted against.</para>
    /// <para>Rows must be complete, as for <see cref="Fit"/>. limma refits a feature with missing
    /// values on the correlation matrix of its observed samples; the differential path never needs
    /// that, because it tests complete features only.</para>
    /// </remarks>
    /// <param name="dataFeaturesBySamples">LOG2 abundances, features x samples.</param>
    /// <param name="designSamplesByCoef">The design, samples x coefficients.</param>
    /// <param name="block">Each sample's block (subject).</param>
    /// <param name="correlation">The intra-block correlation, strictly between -1 and 1.</param>
    public static LinearModelFit FitBlocked(double[,] dataFeaturesBySamples, double[,] designSamplesByCoef,
        IReadOnlyList<string> block, double correlation)
    {
        var nFeatures = dataFeaturesBySamples.GetLength(0);
        var nSamples = dataFeaturesBySamples.GetLength(1);
        var nCoef = designSamplesByCoef.GetLength(1);
        if (block.Count != nSamples)
            throw new ArgumentException(
                $"{block.Count} block labels for {nSamples} samples.", nameof(block));
        if (!(Math.Abs(correlation) < 1))
            throw new ArgumentException(
                $"The intra-block correlation is {correlation}; it must be strictly between -1 and 1.",
                nameof(correlation));

        var v = new DenseMatrix(nSamples, nSamples);
        for (var i = 0; i < nSamples; i++)
            for (var j = 0; j < nSamples; j++)
                v[i, j] = i == j ? 1.0 : string.Equals(block[i], block[j], StringComparison.Ordinal) ? correlation : 0.0;

        // A negative correlation is not positive definite for every block size: limma bounds each
        // feature's estimate at 1/(1 - largest block) + 0.01 for exactly this reason, but a caller
        // can still pass something outside it.
        Cholesky<double> chol;
        try
        {
            chol = v.Cholesky();
        }
        catch (ArgumentException)
        {
            throw new ArgumentException(
                $"An intra-block correlation of {correlation} is not valid for blocks this large "
                + "(the correlation matrix is not positive definite).", nameof(correlation));
        }

        var lower = chol.Factor.ToArray();
        var whitenedDesign = ForwardSubstitute(lower, designSamplesByCoef, rowsAreSamples: true);
        var whitenedData = ForwardSubstitute(lower, dataFeaturesBySamples, rowsAreSamples: false);

        var fit = Fit(whitenedData, whitenedDesign);
        var amean = new double[nFeatures];
        var row = new double[nSamples];
        for (var i = 0; i < nFeatures; i++)
        {
            for (var s = 0; s < nSamples; s++)
                row[s] = dataFeaturesBySamples[i, s];
            amean[i] = NumpyMath.Mean(row);
        }

        return new LinearModelFit(fit.Coefficients, fit.Sigma, amean, fit.StdevUnscaled, fit.DfResidual);
    }

    /// <summary>
    /// <c>L^-1 B</c> for lower-triangular <paramref name="lower"/>, by forward substitution, one
    /// column of <c>B</c> at a time. <paramref name="b"/> holds samples along its rows when
    /// <paramref name="rowsAreSamples"/> (a design), or along its columns (a features x samples
    /// matrix, returned in the same orientation).
    /// </summary>
    private static double[,] ForwardSubstitute(double[,] lower, double[,] b, bool rowsAreSamples)
    {
        var n = lower.GetLength(0);
        var k = rowsAreSamples ? b.GetLength(1) : b.GetLength(0);
        var result = new double[b.GetLength(0), b.GetLength(1)];
        var x = new double[n];
        for (var j = 0; j < k; j++)
        {
            for (var i = 0; i < n; i++)
            {
                var sum = rowsAreSamples ? b[i, j] : b[j, i];
                for (var m = 0; m < i; m++)
                    sum -= lower[i, m] * x[m];
                x[i] = sum / lower[i, i];
            }

            for (var i = 0; i < n; i++)
                if (rowsAreSamples)
                    result[i, j] = x[i];
                else
                    result[j, i] = x[i];
        }

        return result;
    }
}
