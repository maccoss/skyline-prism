using System;

namespace SkylinePrism.Core.Numerics;

/// <summary>
/// LOWESS (locally weighted scatterplot smoothing), a hand-port of Cleveland's algorithm as
/// used by statsmodels.nonparametric.lowess. Local linear regression with tricube distance
/// weights over the frac*n nearest neighbours, plus bisquare robustifying iterations. This is
/// a FUNCTIONAL-parity primitive (RT-lowess normalization), not an exact-parity target: the
/// delta speedup is omitted (every point is fit), so the smoothed curve matches statsmodels
/// closely but not bit-for-bit.
/// </summary>
public static class Lowess
{
    /// <summary>
    /// Smooth y against x (x MUST be sorted ascending), returning the fitted value at each x.
    /// <paramref name="frac"/> is the neighbourhood fraction; <paramref name="iterations"/> is
    /// the number of robustifying reweightings (statsmodels default 3).
    /// </summary>
    public static double[] Fit(double[] x, double[] y, double frac, int iterations = 3)
    {
        var n = x.Length;
        var yfit = new double[n];
        if (n == 0)
            return yfit;
        if (n == 1)
        {
            yfit[0] = y[0];
            return yfit;
        }

        var r = Math.Max(2, Math.Min(n, (int)Math.Ceiling(frac * n)));
        var robust = new double[n];
        for (var i = 0; i < n; i++)
            robust[i] = 1.0;
        var absResid = new double[n];

        for (var iter = 0; iter <= iterations; iter++)
        {
            int left = 0, right = r - 1;
            for (var i = 0; i < n; i++)
            {
                // Slide the window so [left, right] holds the r nearest points to x[i].
                while (right < n - 1 && (x[i] - x[left]) > (x[right + 1] - x[i]))
                {
                    left++;
                    right++;
                }
                var h = Math.Max(x[i] - x[left], x[right] - x[i]);

                double sw = 0, swx = 0, swy = 0, swxx = 0, swxy = 0;
                for (var j = left; j <= right; j++)
                {
                    double w;
                    if (h <= 0)
                    {
                        w = 1.0;
                    }
                    else
                    {
                        var u = Math.Abs(x[j] - x[i]) / h;
                        if (u >= 1.0)
                        {
                            w = 0.0;
                        }
                        else
                        {
                            var t = 1.0 - u * u * u; // tricube
                            w = t * t * t;
                        }
                    }
                    w *= robust[j];
                    sw += w;
                    swx += w * x[j];
                    swy += w * y[j];
                    swxx += w * x[j] * x[j];
                    swxy += w * x[j] * y[j];
                }

                var denom = sw * swxx - swx * swx;
                if (sw <= 0)
                    yfit[i] = y[i];
                else if (Math.Abs(denom) < 1e-12 * (sw * swxx + 1))
                    yfit[i] = swy / sw; // near-singular window -> weighted mean
                else
                {
                    var b = (sw * swxy - swx * swy) / denom;
                    var a = (swy - b * swx) / sw;
                    yfit[i] = a + b * x[i];
                }
            }

            if (iter < iterations)
            {
                for (var i = 0; i < n; i++)
                    absResid[i] = Math.Abs(y[i] - yfit[i]);
                var s = Stats.NanMedian(absResid);
                if (s <= 0)
                    break; // essentially exact fit; no further reweighting
                for (var i = 0; i < n; i++)
                {
                    var e = absResid[i] / (6.0 * s);
                    if (e >= 1.0)
                    {
                        robust[i] = 0.0;
                    }
                    else
                    {
                        var t = 1.0 - e * e; // bisquare
                        robust[i] = t * t;
                    }
                }
            }
        }
        return yfit;
    }
}
