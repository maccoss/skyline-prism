using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace SkylinePrism.Core.Numerics;

/// <summary>
/// Dense linear-algebra helpers backed by MathNet.Numerics, wrapping the numpy calls
/// used by ComBat (np.linalg.solve, np.linalg.matrix_rank). Design matrices in ComBat
/// are tiny (n_batch x n_batch), so LU solve agrees with LAPACK dgesv to well under the
/// 1e-9 parity target.
/// </summary>
public static class LinAlg
{
    /// <summary>
    /// Solve A x = b for a square matrix A (numpy.linalg.solve). A is
    /// row-major [n, n]; b is length n; returns x of length n.
    /// </summary>
    public static double[] Solve(double[,] a, double[] b)
    {
        var n = b.Length;
        var matA = DenseMatrix.OfArray(a);
        var vecB = DenseVector.OfArray(b);
        var x = matA.Solve(vecB);
        var result = new double[n];
        for (var i = 0; i < n; i++)
            result[i] = x[i];
        return result;
    }

    /// <summary>
    /// Solve A X = B with multiple right-hand sides (numpy.linalg.solve on a matrix B).
    /// A is [n, n]; b is [n, k]; returns [n, k].
    /// </summary>
    public static double[,] Solve(double[,] a, double[,] b)
    {
        var matA = DenseMatrix.OfArray(a);
        var matB = DenseMatrix.OfArray(b);
        var x = matA.Solve(matB);
        return x.ToArray();
    }

    /// <summary>
    /// numpy.linalg.matrix_rank via SVD with numpy's default tolerance
    /// (max singular value * max(rows, cols) * eps).
    /// </summary>
    public static int MatrixRank(double[,] a)
    {
        var m = DenseMatrix.OfArray(a);
        var svd = m.Svd(computeVectors: false);
        var s = svd.S;
        double smax = 0.0;
        for (var i = 0; i < s.Count; i++)
            smax = Math.Max(smax, s[i]);
        var tol = smax * Math.Max(a.GetLength(0), a.GetLength(1)) * 2.220446049250313e-16;
        var rank = 0;
        for (var i = 0; i < s.Count; i++)
        {
            if (s[i] > tol)
                rank++;
        }
        return rank;
    }
}
