using System;
using MathNet.Numerics.LinearAlgebra.Double;

namespace SkylinePrism.Core.Numerics;

/// <summary>
/// Simple 2-component PCA for QC scatter plots (approximates sklearn PCA + StandardScaler).
/// Features are standardized (mean 0, unit population std); scores = U * S from the SVD.
/// Not a parity target - used only for visualization, where sign/rotation are immaterial.
/// </summary>
public static class Pca
{
    /// <summary>
    /// Compute 2-D PCA scores from a [nSamples, nFeatures] matrix. Features that are constant
    /// or contain NaN are dropped. Returns [nSamples, 2] scores (zeros if too few features).
    /// </summary>
    public static double[,] Fit2D(double[,] samplesByFeatures)
    {
        var nSamples = samplesByFeatures.GetLength(0);
        var nFeatures = samplesByFeatures.GetLength(1);

        // Standardize each feature; drop NaN / constant features.
        var keptCols = new System.Collections.Generic.List<double[]>();
        for (var j = 0; j < nFeatures; j++)
        {
            var col = new double[nSamples];
            var ok = true;
            for (var i = 0; i < nSamples; i++)
            {
                col[i] = samplesByFeatures[i, j];
                if (double.IsNaN(col[i]))
                {
                    ok = false;
                    break;
                }
            }
            if (!ok)
                continue;

            var mean = Stats.Mean(col);
            var std = Math.Sqrt(Stats.Var(col, ddof: 0));
            if (std == 0.0)
                continue;
            for (var i = 0; i < nSamples; i++)
                col[i] = (col[i] - mean) / std;
            keptCols.Add(col);
        }

        var scores = new double[nSamples, 2];
        if (keptCols.Count < 2 || nSamples < 2)
            return scores;

        var x = new double[nSamples, keptCols.Count];
        for (var j = 0; j < keptCols.Count; j++)
            for (var i = 0; i < nSamples; i++)
                x[i, j] = keptCols[j][i];

        var m = DenseMatrix.OfArray(x);
        var svd = m.Svd(computeVectors: true);
        var u = svd.U;      // nSamples x nSamples
        var s = svd.S;      // singular values

        for (var i = 0; i < nSamples; i++)
        {
            scores[i, 0] = u[i, 0] * s[0];
            scores[i, 1] = s.Count > 1 ? u[i, 1] * s[1] : 0.0;
        }
        return scores;
    }
}
