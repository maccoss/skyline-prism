using System;

namespace SkylinePrism.Core.Numerics;

/// <summary>
/// LOWESS (locally weighted scatterplot smoothing), a hand-port of Cleveland's algorithm as
/// used by statsmodels.nonparametric.lowess. Local linear regression with tricube distance
/// weights over the frac*n nearest neighbours, plus bisquare robustifying iterations.
///
/// Supports the statsmodels <c>delta</c> speedup: local regressions are computed only at anchor
/// points spaced &gt; delta apart on x, and the points between two anchors are filled by linear
/// interpolation. statsmodels/PRISM use delta = 0.01 * range(x), which turns the cost from
/// O(n * frac*n * iters) into roughly O((range/delta) * frac*n * iters) - orders of magnitude faster
/// on tens of thousands of points, with a visually identical curve. delta = 0 fits every point.
/// </summary>
public static class Lowess
{
    /// <summary>
    /// Smooth y against x (x MUST be sorted ascending), returning the fitted value at each x.
    /// <paramref name="frac"/> is the neighbourhood fraction; <paramref name="iterations"/> is the
    /// number of robustifying reweightings (statsmodels default 3); <paramref name="delta"/> is the
    /// x-distance within which points are linearly interpolated instead of individually fit
    /// (0 = fit every point; statsmodels/PRISM use 0.01 * range(x)).
    /// </summary>
    public static double[] Fit(double[] x, double[] y, double frac, int iterations = 3, double delta = 0.0)
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

        // Neighbourhood size = statsmodels' int(frac*n + 1e-10) (truncation), not ceil.
        var r = Math.Max(2, Math.Min(n, (int)(frac * n + 1e-10)));
        var robust = new double[n];
        for (var i = 0; i < n; i++)
            robust[i] = 1.0;
        var absResid = new double[n];

        for (var iter = 0; iter <= iterations; iter++)
        {
            int left = 0, right = r - 1;
            int i = 0, last = -1;
            while (i < n)
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

                // Fill points skipped since the previous anchor by linear interpolation (delta speedup).
                if (last < i - 1)
                {
                    var span = x[i] - x[last];
                    if (span > 0)
                        for (var j = last + 1; j < i; j++)
                        {
                            var alpha = (x[j] - x[last]) / span;
                            yfit[j] = alpha * yfit[i] + (1.0 - alpha) * yfit[last];
                        }
                    else
                        for (var j = last + 1; j < i; j++)
                            yfit[j] = yfit[i];
                }
                last = i;

                // Advance to the next anchor: the last point within delta of x[last] (duplicates copy
                // the anchor's fit). delta <= 0 falls through to fitting every point (i = last + 1).
                if (delta > 0)
                {
                    var cut = x[last] + delta;
                    var ni = last + 1;
                    while (ni < n && x[ni] <= cut)
                    {
                        if (x[ni] == x[last])
                        {
                            yfit[ni] = yfit[last];
                            last = ni;
                        }
                        ni++;
                    }
                    i = Math.Max(last + 1, ni - 1);
                }
                else
                {
                    i = last + 1;
                }
            }

            if (iter < iterations)
            {
                for (var k = 0; k < n; k++)
                    absResid[k] = Math.Abs(y[k] - yfit[k]);
                var s = Stats.NanMedian(absResid);
                if (s <= 0)
                    break; // essentially exact fit; no further reweighting
                for (var k = 0; k < n; k++)
                {
                    var e = absResid[k] / (6.0 * s);
                    if (e >= 1.0)
                    {
                        robust[k] = 0.0;
                    }
                    else
                    {
                        var t = 1.0 - e * e; // bisquare
                        robust[k] = t * t;
                    }
                }
            }
        }
        return yfit;
    }
}
