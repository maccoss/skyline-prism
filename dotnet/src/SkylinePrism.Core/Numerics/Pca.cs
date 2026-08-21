using System;
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
    /// <summary>Features per Gram-accumulation block; bounds the working matrix to nSamples x this.</summary>
    private const int FeatureBlock = 4096;

    /// <summary>
    /// Compute 2-D PCA scores from a [nSamples, nFeatures] matrix. NaN cells are imputed to the
    /// feature mean (0 after standardization); constant / all-NaN features are dropped. Returns
    /// [nSamples, 2] scores (zeros if too few usable features).
    /// </summary>
    public static double[,] Fit2D(double[,] samplesByFeatures)
        => Fit2D(samplesByFeatures, transposed: false);

    /// <summary>
    /// The same PCA on a <b>[nFeatures, nSamples]</b> matrix - the orientation every PRISM matrix is
    /// already in - without materializing its transpose.
    /// <para>
    /// Worth having rather than calling <c>Transpose</c> first: the transpose is a full second copy of
    /// the biggest object in the pipeline (5.7 GB on a 100-document peptide matrix), it lands on the
    /// large object heap where it inflates the working set well past the point it is dropped, and the
    /// loop below reads the matrix one FEATURE at a time anyway - which is the untransposed layout.
    /// </para>
    /// </summary>
    public static double[,] Fit2DOfFeaturesBySamples(double[,] featuresBySamples)
        => Fit2D(featuresBySamples, transposed: true);

    private static double[,] Fit2D(double[,] matrix, bool transposed)
    {
        var nSamples = transposed ? matrix.GetLength(1) : matrix.GetLength(0);
        var nFeatures = transposed ? matrix.GetLength(0) : matrix.GetLength(1);
        var samplesByFeatures = matrix;

        // Each feature is standardized over its observed (non-NaN) values below, imputing NaN to the
        // mean (=> 0 after centering) rather than dropping the whole feature, which would discard
        // almost everything once there are many samples.
        //
        // The Gram matrix is accumulated a BLOCK OF FEATURES at a time rather than by materializing
        // the standardized matrix. X is nSamples x nFeatures, and on a 100-document cohort that is
        // 9,600 x 75,000 - 5.7 GB, which used to be built twice over (a list of per-feature arrays,
        // then a DenseMatrix copy of it) purely to produce a Gram matrix of nSamples x nSamples, a few
        // tens of MB. Blocking keeps the optimized multiply while holding only
        // nSamples x FeatureBlock at a time (~150 MB at 9,600 samples).
        var gram = new DenseMatrix(nSamples, nSamples);
        var block = new DenseMatrix(nSamples, FeatureBlock);
        var col = new double[nSamples];
        var kept = 0;
        var inBlock = 0;

        void FlushBlock()
        {
            if (inBlock == 0)
                return;
            // Xb.Xbᵀ summed over blocks IS X.Xᵀ - the product is a sum over features either way.
            var used = inBlock == FeatureBlock
                ? (Matrix<double>)block
                : block.SubMatrix(0, nSamples, 0, inBlock);
            gram.Add(used.TransposeAndMultiply(used), gram);
            inBlock = 0;
        }

        for (var j = 0; j < nFeatures; j++)
        {
            double sum = 0;
            var cnt = 0;
            for (var i = 0; i < nSamples; i++)
            {
                col[i] = transposed ? samplesByFeatures[j, i] : samplesByFeatures[i, j];
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
            {
                var v = double.IsNaN(col[i]) ? 0.0 : (col[i] - mean) / std;
                block[i, inBlock] = v;
            }
            kept++;
            if (++inBlock == FeatureBlock)
                FlushBlock();
        }
        FlushBlock();

        var scores = new double[nSamples, 2];
        if (kept < 2 || nSamples < 2)
            return scores;

        // Gram matrix X.Xᵀ (nSamples x nSamples). X = U S Vᵀ  =>  X.Xᵀ = U S² Uᵀ, so the symmetric
        // eigendecomposition gives U (eigenvectors) and S² (eigenvalues); scores = U * S.
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
