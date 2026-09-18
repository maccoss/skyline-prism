using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra.Factorization;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.DifferentialAnalysis;

/// <summary>
/// Natural cubic-spline basis, a port of R/inmoose <c>splines::ns</c> used by the limma-trend variance
/// prior. Builds the clamped cubic B-spline basis (De Boor), imposes the natural (linear-beyond-boundary)
/// constraint by projecting onto the null space of the second derivatives at the boundary knots, and
/// returns the reduced basis. The exact basis is defined only up to an orthogonal rotation (the QR
/// choice), but its column span - all the trend fit depends on - is fixed.
/// </summary>
internal static class NaturalSplineBasis
{
    private const int Order = 4; // cubic
    private const int Degree = 3;

    /// <summary>Build the <c>[n, df]</c> natural-spline basis for <paramref name="x"/>.</summary>
    public static double[,] Build(double[] x, int df, bool includeIntercept)
    {
        var min = x.Min();
        var max = x.Max();
        var boundary = new[] { min, max };

        var nIknots = df - 1 - (includeIntercept ? 1 : 0);
        if (nIknots < 0)
            nIknots = 0;

        var inner = new double[nIknots];
        for (var i = 0; i < nIknots; i++)
            inner[i] = Stats.PercentileLinear(x, (i + 1) * 100.0 / (nIknots + 1));

        var aknotsList = new List<double>();
        for (var i = 0; i < Order; i++)
        {
            aknotsList.Add(min);
            aknotsList.Add(max);
        }

        aknotsList.AddRange(inner);
        aknotsList.Sort();
        var aknots = aknotsList.ToArray();
        var nbases = aknots.Length - Order;

        var basis = SplineDesign(aknots, x, null);              // n x nbases
        var constMat = SplineDesign(aknots, boundary, new[] { 2, 2 }); // 2 x nbases

        var start = includeIntercept ? 0 : 1;
        var cols = nbases - start;

        // const.T (cols x 2); complete QR gives Q (cols x cols) whose columns span R^cols.
        var constT = new double[cols, 2];
        for (var c = 0; c < cols; c++)
        {
            constT[c, 0] = constMat[0, c + start];
            constT[c, 1] = constMat[1, c + start];
        }

        var q = DenseMatrix.OfArray(constT).QR(QRMethod.Full).Q; // cols x cols

        var basisM = DenseMatrix.Create(x.Length, cols, (i, c) => basis[i, c + start]);
        // rotated = (Q^T basis^T)^T = basis Q  (n x cols); drop the first 2 columns (the constrained ones).
        var rotated = basisM * q; // n x cols
        var result = new double[x.Length, cols - 2];
        for (var i = 0; i < x.Length; i++)
        for (var c = 2; c < cols; c++)
            result[i, c - 2] = rotated[i, c];

        return result;
    }

    /// <summary>
    /// Evaluate the cubic B-spline basis (or a derivative per point) defined by <paramref name="knots"/>
    /// at each of <paramref name="xs"/>. Row j, column i is B_i^(der[j])(xs[j]); a null
    /// <paramref name="derivsPerPoint"/> means value (derivative 0). Mirrors inmoose <c>spline_design</c>.
    /// </summary>
    private static double[,] SplineDesign(double[] knots, double[] xs, int[]? derivsPerPoint)
    {
        var nbases = knots.Length - Order;
        var res = new double[xs.Length, nbases];
        for (var j = 0; j < xs.Length; j++)
        {
            var d = derivsPerPoint is not null && j < derivsPerPoint.Length ? derivsPerPoint[j] : 0;
            var span = FindSpan(nbases - 1, xs[j], knots);
            var ders = DersBasisFuns(span, xs[j], d, knots);
            for (var r = 0; r <= Degree; r++)
                res[j, span - Degree + r] = ders[d][r];
        }

        return res;
    }

    /// <summary>Knot span containing u for a clamped cubic (NURBS book A2.1); n is the last basis index.</summary>
    private static int FindSpan(int n, double u, double[] u_)
    {
        if (u >= u_[n + 1])
            return n;
        if (u <= u_[Degree])
            return Degree;

        int low = Degree, high = n + 1, mid = (low + high) / 2;
        while (u < u_[mid] || u >= u_[mid + 1])
        {
            if (u < u_[mid])
                high = mid;
            else
                low = mid;
            mid = (low + high) / 2;
        }

        return mid;
    }

    /// <summary>
    /// Nonzero basis functions and their derivatives up to order <paramref name="nDeriv"/> at u in the
    /// given span (NURBS book A2.3). ders[k][r] = N_{span-degree+r}^{(k)}(u).
    /// </summary>
    private static double[][] DersBasisFuns(int span, double u, int nDeriv, double[] knots)
    {
        var ndu = new double[Degree + 1][];
        for (var i = 0; i <= Degree; i++)
            ndu[i] = new double[Degree + 1];
        var left = new double[Degree + 1];
        var right = new double[Degree + 1];

        ndu[0][0] = 1.0;
        for (var j = 1; j <= Degree; j++)
        {
            left[j] = u - knots[span + 1 - j];
            right[j] = knots[span + j] - u;
            var saved = 0.0;
            for (var r = 0; r < j; r++)
            {
                ndu[j][r] = right[r + 1] + left[j - r];
                var temp = ndu[r][j - 1] / ndu[j][r];
                ndu[r][j] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }

            ndu[j][j] = saved;
        }

        var ders = new double[nDeriv + 1][];
        for (var k = 0; k <= nDeriv; k++)
            ders[k] = new double[Degree + 1];
        for (var j = 0; j <= Degree; j++)
            ders[0][j] = ndu[j][Degree];

        var a = new double[2][];
        a[0] = new double[Degree + 1];
        a[1] = new double[Degree + 1];
        for (var r = 0; r <= Degree; r++)
        {
            int s1 = 0, s2 = 1;
            a[0][0] = 1.0;
            for (var k = 1; k <= nDeriv; k++)
            {
                var dd = 0.0;
                var rk = r - k;
                var pk = Degree - k;
                if (r >= k)
                {
                    a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
                    dd = a[s2][0] * ndu[rk][pk];
                }

                var j1 = rk >= -1 ? 1 : -rk;
                var j2 = r - 1 <= pk ? k - 1 : Degree - r;
                for (var jj = j1; jj <= j2; jj++)
                {
                    a[s2][jj] = (a[s1][jj] - a[s1][jj - 1]) / ndu[pk + 1][rk + jj];
                    dd += a[s2][jj] * ndu[rk + jj][pk];
                }

                if (r <= pk)
                {
                    a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
                    dd += a[s2][k] * ndu[r][pk];
                }

                ders[k][r] = dd;
                (s1, s2) = (s2, s1);
            }
        }

        var factor = Degree;
        for (var k = 1; k <= nDeriv; k++)
        {
            for (var j = 0; j <= Degree; j++)
                ders[k][j] *= factor;
            factor *= Degree - k;
        }

        return ders;
    }
}
