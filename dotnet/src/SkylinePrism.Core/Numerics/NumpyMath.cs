using System;

namespace SkylinePrism.Core.Numerics;

/// <summary>
/// Reductions that reproduce numpy's summation algorithm. numpy's add.reduce uses
/// pairwise summation (recursive split with an 8-accumulator base case for blocks &lt;= 128),
/// which is more accurate than naive left-to-right summation and, crucially, is what
/// np.mean / np.var / np.sum use. Matching it keeps ComBat parity tight for well-conditioned
/// features and minimizes divergence for ill-conditioned ones.
/// </summary>
public static class NumpyMath
{
    private const int BlockSize = 128;

    /// <summary>numpy-compatible pairwise sum over a[offset .. offset+n).</summary>
    public static double PairwiseSum(double[] a, int offset, int n)
    {
        if (n < 8)
        {
            double res0 = 0.0;
            for (var i = 0; i < n; i++)
                res0 += a[offset + i];
            return res0;
        }

        if (n <= BlockSize)
        {
            // 8 independent accumulators, matching numpy's base case.
            double r0 = a[offset + 0], r1 = a[offset + 1], r2 = a[offset + 2], r3 = a[offset + 3];
            double r4 = a[offset + 4], r5 = a[offset + 5], r6 = a[offset + 6], r7 = a[offset + 7];
            int i;
            for (i = 8; i < n - (n % 8); i += 8)
            {
                r0 += a[offset + i + 0];
                r1 += a[offset + i + 1];
                r2 += a[offset + i + 2];
                r3 += a[offset + i + 3];
                r4 += a[offset + i + 4];
                r5 += a[offset + i + 5];
                r6 += a[offset + i + 6];
                r7 += a[offset + i + 7];
            }
            var res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
            for (; i < n; i++)
                res += a[offset + i];
            return res;
        }

        // Recurse: split at n/2 rounded down to a multiple of 8.
        var n2 = n / 2;
        n2 -= n2 % 8;
        return PairwiseSum(a, offset, n2) + PairwiseSum(a, offset + n2, n - n2);
    }

    public static double PairwiseSum(double[] a) => PairwiseSum(a, 0, a.Length);

    /// <summary>numpy.mean.</summary>
    public static double Mean(double[] a) => a.Length == 0 ? double.NaN : PairwiseSum(a) / a.Length;

    /// <summary>
    /// numpy.var: mean via pairwise reduce, then pairwise reduce of squared deviations
    /// divided by (n - ddof).
    /// </summary>
    public static double Var(double[] a, int ddof = 0)
    {
        var n = a.Length;
        if (n - ddof <= 0)
            return double.NaN;
        var mean = PairwiseSum(a) / n;
        var sq = new double[n];
        for (var i = 0; i < n; i++)
        {
            var dd = a[i] - mean;
            sq[i] = dd * dd;
        }
        return PairwiseSum(sq) / (n - ddof);
    }
}
