using System;
using System.Collections.Generic;

namespace SkylinePrism.Core.Numerics;

/// <summary>
/// Deterministic numeric primitives that reproduce the numpy / scipy behavior the
/// Python PRISM pipeline relies on. Every method here is a PARITY KEYSTONE: subtle
/// differences (even-count median averaging, percentile interpolation, ddof) compound
/// downstream through rollup and ComBat, so the semantics must match numpy exactly.
///
/// Scale note: these are pure math helpers and are scale-agnostic. Callers are
/// responsible for the LINEAR vs LOG2 convention documented in the pipeline.
/// </summary>
public static class Stats
{
    /// <summary>
    /// numpy.nanmedian over a 1-D sequence: NaNs are ignored. For an even count of
    /// non-NaN values the result is the mean of the two middle order statistics.
    /// Returns NaN when there are no non-NaN values.
    /// </summary>
    public static double NanMedian(ReadOnlySpan<double> values)
    {
        // Collect non-NaN values.
        var buf = new List<double>(values.Length);
        foreach (var v in values)
        {
            if (!double.IsNaN(v))
                buf.Add(v);
        }

        if (buf.Count == 0)
            return double.NaN;

        return MedianOfSorted(SortedCopy(buf));
    }

    /// <summary>
    /// numpy.median over a 1-D sequence: NaN propagates (any NaN =&gt; NaN result),
    /// matching numpy semantics. Use <see cref="NanMedian"/> to skip NaNs.
    /// </summary>
    public static double Median(ReadOnlySpan<double> values)
    {
        var buf = new double[values.Length];
        for (var i = 0; i < values.Length; i++)
        {
            if (double.IsNaN(values[i]))
                return double.NaN;
            buf[i] = values[i];
        }

        if (buf.Length == 0)
            return double.NaN;

        Array.Sort(buf);
        return MedianOfSorted(buf);
    }

    private static double MedianOfSorted(double[] sorted)
    {
        var n = sorted.Length;
        var mid = n / 2;
        if ((n & 1) == 1)
            return sorted[mid];
        // Even count: numpy averages the two middle values as (a + b) / 2.
        return (sorted[mid - 1] + sorted[mid]) / 2.0;
    }

    /// <summary>
    /// numpy.percentile with the default "linear" interpolation over a 1-D sequence
    /// that contains no NaNs. <paramref name="p"/> is in [0, 100]. The virtual index
    /// is p/100 * (n - 1); the result linearly interpolates between the bracketing
    /// order statistics.
    /// </summary>
    public static double PercentileLinear(ReadOnlySpan<double> values, double p)
    {
        if (values.Length == 0)
            return double.NaN;
        if (values.Length == 1)
            return values[0];

        var sorted = new double[values.Length];
        values.CopyTo(sorted);
        Array.Sort(sorted);

        var rank = (p / 100.0) * (sorted.Length - 1);
        var lo = (int)Math.Floor(rank);
        if (lo >= sorted.Length - 1)
            return sorted[sorted.Length - 1];
        if (lo < 0)
            return sorted[0];
        var frac = rank - lo;
        return sorted[lo] + frac * (sorted[lo + 1] - sorted[lo]);
    }

    /// <summary>numpy.nanpercentile ("linear"): ignores NaNs.</summary>
    public static double NanPercentileLinear(ReadOnlySpan<double> values, double p)
    {
        var buf = new List<double>(values.Length);
        foreach (var v in values)
        {
            if (!double.IsNaN(v))
                buf.Add(v);
        }
        if (buf.Count == 0)
            return double.NaN;
        return PercentileLinear(buf.ToArray(), p);
    }

    /// <summary>
    /// numpy.var with the given delta-degrees-of-freedom. Uses numpy-compatible pairwise
    /// summation (mean then mean of squared deviations) so ComBat's variance estimates
    /// match numpy to the last bits even for near-singular features.
    /// </summary>
    public static double Var(ReadOnlySpan<double> values, int ddof = 0)
    {
        var n = values.Length;
        if (n - ddof <= 0)
            return double.NaN;
        return NumpyMath.Var(values.ToArray(), ddof);
    }

    /// <summary>numpy.nanvar with the given ddof: ignores NaNs.</summary>
    public static double NanVar(ReadOnlySpan<double> values, int ddof = 0)
    {
        double sum = 0.0;
        var n = 0;
        foreach (var v in values)
        {
            if (double.IsNaN(v))
                continue;
            sum += v;
            n++;
        }
        if (n - ddof <= 0)
            return double.NaN;
        var mean = sum / n;
        double ss = 0.0;
        foreach (var v in values)
        {
            if (double.IsNaN(v))
                continue;
            var d = v - mean;
            ss += d * d;
        }
        return ss / (n - ddof);
    }

    /// <summary>Arithmetic mean; ignores NaNs (numpy.nanmean). NaN if all NaN.</summary>
    public static double NanMean(ReadOnlySpan<double> values)
    {
        double sum = 0.0;
        var n = 0;
        foreach (var v in values)
        {
            if (double.IsNaN(v))
                continue;
            sum += v;
            n++;
        }
        return n == 0 ? double.NaN : sum / n;
    }

    /// <summary>
    /// <see cref="NanMean"/> that does not depend on the order of <paramref name="values"/>.
    /// <para>
    /// A mean is permutation-invariant in exact arithmetic but not in floating point, because
    /// addition is not associative. That matters wherever the input order is not guaranteed: the
    /// rollup's per-peptide rows come out of DuckDB with no ordering promised WITHIN a peptide, so
    /// a plain sequential sum made <c>mean_rt</c> vary in the last bits between two runs of the
    /// same binary on the same input. Summing a sorted copy makes the evaluation order a function
    /// of the multiset alone, so the result is reproducible however the rows arrive.
    /// </para>
    /// <para>Sorting also sums small magnitudes first, so this is no less accurate than the
    /// sequential version. Inputs here are one peptide's values, so the cost is negligible.</para>
    /// </summary>
    public static double NanMeanOrderInvariant(ReadOnlySpan<double> values)
    {
        var buf = new double[values.Length];
        var n = 0;
        foreach (var v in values)
        {
            if (!double.IsNaN(v))
                buf[n++] = v;
        }
        if (n == 0)
            return double.NaN;

        Array.Sort(buf, 0, n);
        double sum = 0.0;
        for (var i = 0; i < n; i++)
            sum += buf[i];
        return sum / n;
    }

    /// <summary>Arithmetic mean over all values (numpy.mean), pairwise summation.</summary>
    public static double Mean(ReadOnlySpan<double> values)
    {
        if (values.Length == 0)
            return double.NaN;
        return NumpyMath.Mean(values.ToArray());
    }

    /// <summary>
    /// scipy.stats.rankdata(method="average"): 1-based ranks with ties assigned the
    /// average of the ranks they span. Result length equals input length.
    /// </summary>
    public static double[] RankAverage(ReadOnlySpan<double> values)
    {
        var n = values.Length;
        var order = new int[n];
        for (var i = 0; i < n; i++)
            order[i] = i;

        var vals = new double[n];
        values.CopyTo(vals);
        // Stable sort by value so ties keep original order (irrelevant for average
        // ranks but keeps behavior deterministic).
        Array.Sort(order, (a, b) =>
        {
            var c = vals[a].CompareTo(vals[b]);
            return c != 0 ? c : a.CompareTo(b);
        });

        var ranks = new double[n];
        var i2 = 0;
        while (i2 < n)
        {
            var j = i2;
            // Extend the tie group [i2, j].
            while (j + 1 < n && vals[order[j + 1]] == vals[order[i2]])
                j++;
            // Average of 1-based ranks (i2+1) .. (j+1).
            var avg = ((i2 + 1) + (j + 1)) / 2.0;
            for (var k = i2; k <= j; k++)
                ranks[order[k]] = avg;
            i2 = j + 1;
        }
        return ranks;
    }

    /// <summary>
    /// numpy.interp: piecewise-linear interpolation of <paramref name="x"/> onto the
    /// data points (<paramref name="xp"/>, <paramref name="fp"/>). <paramref name="xp"/>
    /// must be monotonically increasing. Values of x outside the range clamp to the
    /// endpoint fp values (numpy's default left/right behavior).
    /// </summary>
    public static double Interp(double x, double[] xp, double[] fp)
    {
        var n = xp.Length;
        if (n == 0)
            return double.NaN;
        if (x <= xp[0])
            return fp[0];
        if (x >= xp[n - 1])
            return fp[n - 1];

        // Binary search for the interval xp[hi-1] <= x < xp[hi].
        var lo = 0;
        var hi = n - 1;
        while (hi - lo > 1)
        {
            var mid = (lo + hi) >> 1;
            if (xp[mid] <= x)
                lo = mid;
            else
                hi = mid;
        }
        var t = (x - xp[lo]) / (xp[hi] - xp[lo]);
        return fp[lo] + t * (fp[hi] - fp[lo]);
    }

    /// <summary>
    /// Stable argsort ascending (mergesort semantics): returns indices that would sort
    /// the input, with ties broken by original index. Matches numpy argsort(kind="stable").
    /// </summary>
    public static int[] ArgSort(ReadOnlySpan<double> values)
    {
        var n = values.Length;
        var idx = new int[n];
        for (var i = 0; i < n; i++)
            idx[i] = i;
        var vals = new double[n];
        values.CopyTo(vals);
        Array.Sort(idx, (a, b) =>
        {
            var c = vals[a].CompareTo(vals[b]);
            return c != 0 ? c : a.CompareTo(b);
        });
        return idx;
    }

    /// <summary>
    /// Indices of the <paramref name="n"/> largest values, descending by value, ties
    /// broken by original (lower) index first. Reproduces pandas Series.nlargest with
    /// keep="first" used by the TopN rollup for deterministic peptide selection.
    /// </summary>
    public static int[] NLargestIndices(ReadOnlySpan<double> values, int n)
    {
        var count = values.Length;
        var idx = new int[count];
        for (var i = 0; i < count; i++)
            idx[i] = i;
        var vals = new double[count];
        values.CopyTo(vals);
        // Sort descending by value; ties -> lower original index first.
        Array.Sort(idx, (a, b) =>
        {
            var c = vals[b].CompareTo(vals[a]);
            return c != 0 ? c : a.CompareTo(b);
        });
        var take = Math.Min(n, count);
        var result = new int[take];
        Array.Copy(idx, result, take);
        return result;
    }

    private static double[] SortedCopy(List<double> buf)
    {
        var arr = buf.ToArray();
        Array.Sort(arr);
        return arr;
    }
}
