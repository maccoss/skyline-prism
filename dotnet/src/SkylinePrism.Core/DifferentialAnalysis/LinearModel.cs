using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra.Factorization;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.DifferentialAnalysis;

/// <summary>
/// Result of a shared-design per-feature ordinary least-squares fit (limma's <c>lmFit</c>) on
/// complete rows. Because the design matrix is shared across them, the unscaled standard errors
/// depend only on the design and are the same for every feature, so they are stored once. Features
/// with missing values go through <see cref="LinearModel.FitObserved"/>, which groups them by their
/// missing-value pattern and fits each group through <see cref="LinearModel.Fit"/>.
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
/// Result of <see cref="LinearModel.FitObserved"/>: every feature fitted on the samples where it was
/// observed, as limma's <c>lmFit</c> fits a feature with missing values. One entry per feature, and
/// only the TESTED coefficient is kept, since that is all the moderated t reads.
/// </summary>
public sealed class ObservedLinearModelFit
{
    internal ObservedLinearModelFit(double[] coefficient, double[] stdevUnscaled, double[] sigma,
        double[] dfResidual, double[] amean)
    {
        Coefficient = coefficient;
        StdevUnscaled = stdevUnscaled;
        Sigma = sigma;
        DfResidual = dfResidual;
        Amean = amean;
    }

    /// <summary>
    /// The tested coefficient; NaN where it cannot be estimated from the feature's observed samples
    /// (an arm entirely missing) or the feature has no residual degrees of freedom.
    /// </summary>
    public double[] Coefficient { get; }

    /// <summary>Unscaled standard error of the tested coefficient; NaN where it is.</summary>
    public double[] StdevUnscaled { get; }

    /// <summary>
    /// Residual standard deviation; NaN where the feature has no residual degrees of freedom. A
    /// feature whose tested coefficient is NaN still has one, and it still informs the prior.
    /// </summary>
    public double[] Sigma { get; }

    /// <summary>
    /// Residual degrees of freedom, <c>n_observed - rank</c> of the design on the observed samples;
    /// 0 for a feature that could not be fitted, as limma records it.
    /// </summary>
    public double[] DfResidual { get; }

    /// <summary>Mean of the observed values, limma's <c>Amean</c>; NaN when none was observed.</summary>
    public double[] Amean { get; }
}

/// <summary>
/// Per-feature ordinary least squares with a design matrix shared across features - the
/// <c>lmFit</c> step of the limma moderated-t pipeline ported from the PRISM Differential Explorer
/// (prism_diff_explorer.py). Operates on a LOG2 abundance matrix. <see cref="Fit"/> expects complete
/// rows - a NaN in the response propagates to that feature's outputs - and the differential path
/// reaches it through <see cref="FitObserved"/>, which hands it each missing-value pattern's observed
/// samples, as limma's lmFit fits a feature with missing values.
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
    /// Fit every feature on the samples where it was observed, as limma's <c>lmFit</c> fits a
    /// feature with missing values, and keep the coefficient in column <paramref name="testedColumn"/>.
    /// </summary>
    /// <remarks>
    /// <para>For each feature the design is restricted to its observed samples, and the columns those
    /// samples can estimate are kept left to right, as R's <c>lm.fit</c> pivots
    /// (<see cref="EstimableColumns"/>). A subject with no value for a feature leaves its subject column
    /// empty - or, for the subject the intercept stands for, makes the other subject columns sum to the
    /// intercept - so a column is dropped and the feature is fitted on the remaining subjects. The
    /// residual df is the observed sample count less the kept columns.</para>
    /// <para>The tested coefficient is reported only when its column is outside the span of every
    /// other column on those samples (<see cref="TestedIsEstimable"/>): an arm entirely missing makes
    /// the group column the intercept, and there is then no contrast. Such a feature still has a
    /// residual variance and df, which go into the prior and the pooled df, as in limma.</para>
    /// <para>Features sharing a missing-value pattern share a design, and are fitted together through
    /// <see cref="Fit"/> on that design - with complete data, every feature in one call, which is the
    /// computation PRISM made before it fitted missing values at all.</para>
    /// <para><c>proteomics-toolkit</c>'s <c>_fit_moderated_t</c> is the same rule, and both are pinned
    /// to R limma 3.66.0 by <c>limma_ebayes.json</c>.</para>
    /// </remarks>
    /// <param name="exprLog2">LOG2 abundances, <c>[feature, matrix column]</c>.</param>
    /// <param name="columns">The matrix columns the design's rows stand for, in design order.</param>
    /// <param name="designSamplesByCoef">The design over those columns; full column rank.</param>
    /// <param name="testedColumn">The design column whose coefficient is tested.</param>
    public static ObservedLinearModelFit FitObserved(
        double[,] exprLog2, IReadOnlyList<int> columns, double[,] designSamplesByCoef, int testedColumn)
    {
        var nFeatures = exprLog2.GetLength(0);
        var nSamples = columns.Count;
        var nCoef = designSamplesByCoef.GetLength(1);
        if (designSamplesByCoef.GetLength(0) != nSamples)
            throw new ArgumentException(
                $"Design has {designSamplesByCoef.GetLength(0)} rows for {nSamples} sample columns.",
                nameof(designSamplesByCoef));

        var coefficient = new double[nFeatures];
        var stdevUnscaled = new double[nFeatures];
        var sigma = new double[nFeatures];
        var dfResidual = new double[nFeatures];
        var amean = new double[nFeatures];
        Array.Fill(coefficient, double.NaN);
        Array.Fill(stdevUnscaled, double.NaN);
        Array.Fill(sigma, double.NaN);
        Array.Fill(amean, double.NaN);

        // Group the features by which samples they were observed in. A value that is not finite is
        // missing, as limma's lmFit takes it (is.finite): an infinity is not a measurement.
        var byPattern = new Dictionary<string, List<int>>(StringComparer.Ordinal);
        var pattern = new char[nSamples];
        for (var f = 0; f < nFeatures; f++)
        {
            for (var s = 0; s < nSamples; s++)
                pattern[s] = double.IsFinite(exprLog2[f, columns[s]]) ? '1' : '0';
            var key = new string(pattern);
            if (!byPattern.TryGetValue(key, out var members))
                byPattern[key] = members = new List<int>();
            members.Add(f);
        }

        foreach (var (key, members) in byPattern)
        {
            var observed = new List<int>(nSamples);
            for (var s = 0; s < nSamples; s++)
                if (key[s] == '1')
                    observed.Add(s);
            if (observed.Count == 0)
                continue;

            var data = new double[members.Count, observed.Count];
            for (var i = 0; i < members.Count; i++)
                for (var k = 0; k < observed.Count; k++)
                    data[i, k] = exprLog2[members[i], columns[observed[k]]];

            // limma's Amean: over every observed value, whether or not the feature can be fitted.
            var row = new double[observed.Count];
            for (var i = 0; i < members.Count; i++)
            {
                for (var k = 0; k < observed.Count; k++)
                    row[k] = data[i, k];
                amean[members[i]] = NumpyMath.Mean(row);
            }

            var observedDesign = new double[observed.Count, nCoef];
            for (var k = 0; k < observed.Count; k++)
                for (var c = 0; c < nCoef; c++)
                    observedDesign[k, c] = designSamplesByCoef[observed[k], c];
            var kept = EstimableColumns(observedDesign);
            if (observed.Count - kept.Count <= 0)
                continue;

            var keptDesign = new double[observed.Count, kept.Count];
            for (var k = 0; k < observed.Count; k++)
                for (var c = 0; c < kept.Count; c++)
                    keptDesign[k, c] = observedDesign[k, kept[c]];
            var fit = Fit(data, keptDesign);

            var at = kept.IndexOf(testedColumn);
            var estimable = at >= 0 && TestedIsEstimable(observedDesign, testedColumn);
            for (var i = 0; i < members.Count; i++)
            {
                var f = members[i];
                sigma[f] = fit.Sigma[i];
                dfResidual[f] = fit.DfResidual;
                if (estimable)
                {
                    coefficient[f] = fit.Coefficients[i, at];
                    stdevUnscaled[f] = fit.StdevUnscaled[at];
                }
            }
        }

        return new ObservedLinearModelFit(coefficient, stdevUnscaled, sigma, dfResidual, amean);
    }

    /// <summary>
    /// The design columns a least-squares fit can estimate, kept left to right as R's <c>lm.fit</c>
    /// (LINPACK <c>dqrdc2</c>) keeps them: a column whose part outside the span of the columns kept
    /// before it is below <paramref name="tolerance"/> of its own norm is aliased and dropped.
    /// </summary>
    /// <remarks>
    /// Order matters - of an aliased pair the LATER column goes - which is why every design here puts
    /// the intercept first and the tested coefficient second. <c>proteomics-toolkit</c>'s
    /// <c>_estimable_columns</c> is the same rule.
    /// </remarks>
    internal static List<int> EstimableColumns(double[,] design, double tolerance = 1e-7)
    {
        var n = design.GetLength(0);
        var p = design.GetLength(1);
        var kept = new List<int>(p);
        var basis = new List<double[]>(p);
        var column = new double[n];
        for (var j = 0; j < p; j++)
        {
            double norm = 0;
            for (var i = 0; i < n; i++)
            {
                column[i] = design[i, j];
                norm += column[i] * column[i];
            }

            norm = Math.Sqrt(norm);
            if (norm == 0.0)
                continue;

            // Gram-Schmidt against the kept columns, twice, for orthogonality in floating point.
            var remainder = (double[])column.Clone();
            for (var pass = 0; pass < 2; pass++)
                foreach (var q in basis)
                {
                    double dot = 0;
                    for (var i = 0; i < n; i++)
                        dot += q[i] * remainder[i];
                    for (var i = 0; i < n; i++)
                        remainder[i] -= dot * q[i];
                }

            double size = 0;
            for (var i = 0; i < n; i++)
                size += remainder[i] * remainder[i];
            size = Math.Sqrt(size);
            if (size <= tolerance * norm)
                continue;

            for (var i = 0; i < n; i++)
                remainder[i] /= size;
            basis.Add(remainder);
            kept.Add(j);
        }

        return kept;
    }

    /// <summary>
    /// Whether the tested coefficient can be estimated from these samples: its column must lie outside
    /// the span of every other column, not only of the columns before it.
    /// </summary>
    /// <remarks>
    /// <see cref="EstimableColumns"/> alone is not enough. It keeps columns left to right and the tested
    /// column sits second, so it always survives; when it is aliased with a LATER column - a covariate
    /// equal to the group on a feature's observed samples - that column is dropped instead and the
    /// tested coefficient silently absorbs it. Neither tool reports a contrast there.
    /// <c>proteomics-toolkit</c>'s <c>_tested_is_estimable</c> is the same rule.
    /// </remarks>
    internal static bool TestedIsEstimable(double[,] design, int testedColumn)
    {
        var n = design.GetLength(0);
        var p = design.GetLength(1);
        var reordered = new double[n, p];
        var target = 0;
        for (var j = 0; j < p; j++)
        {
            if (j == testedColumn)
                continue;
            for (var i = 0; i < n; i++)
                reordered[i, target] = design[i, j];
            target++;
        }

        for (var i = 0; i < n; i++)
            reordered[i, p - 1] = design[i, testedColumn];
        var kept = EstimableColumns(reordered);
        return kept.Count > 0 && kept[^1] == p - 1;
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
