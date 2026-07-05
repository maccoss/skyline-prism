using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace SkylinePrism.Core.Numerics;

/// <summary>
/// Simple 2-component PCA for QC scatter plots (approximates sklearn PCA + StandardScaler).
/// Features are standardized (mean 0, unit population std); scores = U * S.
/// Not a parity target - used only for visualization, where sign/rotation are immaterial.
///
/// IMPORTANT: proteomics matrices have far more features (peptides/proteins) than samples, so we
/// must NEVER form the full feature-space SVD - MathNet's dense SVD would allocate an
/// (nFeatures x nFeatures) V matrix (e.g. 40k x 40k ~ 13 GB), which hangs/OOMs the machine.
/// Instead we eigendecompose the small (nSamples x nSamples) Gram matrix X.Xᵀ = U S² Uᵀ, so cost
/// scales with the sample count, not the feature count.
/// </summary>
public static class Pca
{
    /// <summary>
    /// Compute 2-D PCA scores from a [nSamples, nFeatures] matrix. NaN cells are imputed to the
    /// feature mean (0 after standardization); constant / all-NaN features are dropped. Returns
    /// [nSamples, 2] scores (zeros if too few usable features).
    /// </summary>
    public static double[,] Fit2D(double[,] samplesByFeatures)
    {
        var nSamples = samplesByFeatures.GetLength(0);
        var nFeatures = samplesByFeatures.GetLength(1);

        // Standardize each feature over its observed (non-NaN) values; impute NaN to the mean
        // (=> 0 after centering) rather than dropping the whole feature, which would discard
        // almost everything once there are many samples.
        var keptCols = new List<double[]>();
        for (var j = 0; j < nFeatures; j++)
        {
            var col = new double[nSamples];
            double sum = 0;
            var cnt = 0;
            for (var i = 0; i < nSamples; i++)
            {
                col[i] = samplesByFeatures[i, j];
                if (!double.IsNaN(col[i]))
                {
                    sum += col[i];
                    cnt++;
                }
            }
            if (cnt < 2)
                continue;

            var mean = sum / cnt;
            double ss = 0;
            for (var i = 0; i < nSamples; i++)
                if (!double.IsNaN(col[i]))
                {
                    var d = col[i] - mean;
                    ss += d * d;
                }
            var std = Math.Sqrt(ss / cnt);
            if (std == 0.0)
                continue;

            for (var i = 0; i < nSamples; i++)
                col[i] = double.IsNaN(col[i]) ? 0.0 : (col[i] - mean) / std;
            keptCols.Add(col);
        }

        var scores = new double[nSamples, 2];
        if (keptCols.Count < 2 || nSamples < 2)
            return scores;

        var x = new DenseMatrix(nSamples, keptCols.Count);
        for (var j = 0; j < keptCols.Count; j++)
            for (var i = 0; i < nSamples; i++)
                x[i, j] = keptCols[j][i];

        // Gram matrix X.Xᵀ (nSamples x nSamples). X = U S Vᵀ  =>  X.Xᵀ = U S² Uᵀ, so the symmetric
        // eigendecomposition gives U (eigenvectors) and S² (eigenvalues); scores = U * S.
        var gram = x.TransposeAndMultiply(x);
        var evd = gram.Evd(Symmetricity.Symmetric);
        var evals = evd.EigenValues;   // ascending
        var evecs = evd.EigenVectors;  // columns are the eigenvectors (= U)
        var n = nSamples;

        var s0 = Math.Sqrt(Math.Max(evals[n - 1].Real, 0.0));
        var s1 = Math.Sqrt(Math.Max(evals[n - 2].Real, 0.0));
        for (var i = 0; i < n; i++)
        {
            scores[i, 0] = evecs[i, n - 1] * s0;
            scores[i, 1] = evecs[i, n - 2] * s1;
        }
        return scores;
    }
}
