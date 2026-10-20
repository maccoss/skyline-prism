using System;
using System.Collections.Generic;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.Library;

/// <summary>
/// One precursor's noise-gated ratio fit, per sample: the LINEAR abundance, the raw area of the
/// fragments the scale was estimated from, and how many there were.
/// </summary>
public sealed record NoiseGatedFit(double[] Abundance, double[] Retained, int[] Kept);

/// <summary>
/// Library-assisted transition rollup. The spectral library fixes each fragment's share of a
/// precursor's signal, so a sample's abundance is one scale times the library's total, and a fragment
/// reading far above its share is treated as interference and left out of the scale.
/// <para>
/// Two fits. <see cref="NoiseGatedRatio"/> (<c>library_fitting_method: noise_gated_ratio</c>, the
/// default) is the one to use; <see cref="MedianPolish"/> (<c>median_polish</c>) is the original and
/// the earlier default, kept so earlier results reproduce. Measured on FLARE-Extended (93 runs, 7
/// reference + 7 QC pool injections), the median fit was 43% noisier than the signal it kept allows:
/// its 2x removal fires on fragments that are only noise-high at low intensity, the fragments left
/// behind read low by chance, and the median of their log ratios lands on floored zero areas - a scale
/// biased low by an amount that changes from injection to injection.
/// </para>
/// </summary>
public static class LibraryRollup
{
    /// <summary>
    /// The noise-gated ratio fit, per sample j over the library-matched rows k (lib finite and &gt; 0):
    /// <code>
    /// kept = every matched row; up to maxRemovals times:
    ///     c = sum_kept o / sum_kept lib
    ///     stop if removal is off or |kept| &lt;= minFragments
    ///     pred_k = c lib_k ; nr_k = (o_k - pred_k) / pred_k
    ///     flagged_k = nr_k &gt; outlierThreshold AND o_k - pred_k &gt; noiseSigma * sd(pred_k)
    ///     drop the flagged row with the largest nr, or stop if none
    /// abundance = (sum_kept o / sum_kept lib) * sum_matched lib
    /// </code>
    /// Equivalently, abundance = the observed areas of the kept fragments plus c * lib for each dropped
    /// one: with nothing dropped it is the plain sum of the matched fragments, which is why its precision
    /// stays close to sum's. The noise test is what keeps noise from being removed as interference: at
    /// an expected area of 1,000 on FLARE, a fragment had to read 3.2x its prediction to go, against
    /// 2x for the ratio test alone. It is off when <paramref name="fragmentSd"/> is null or
    /// <paramref name="noiseSigma"/> &lt;= 0.
    /// </summary>
    /// <param name="observed">(T x S) LINEAR areas as fitted - PRISM passes the same floored values
    /// <see cref="Rollup.SumRollup"/> sums. NaN and negative cells count as 0.</param>
    /// <param name="lib">(T) library intensities; NaN or &lt;= 0 marks an unmatched row.</param>
    /// <param name="retainedSource">(T x S) areas the retained signal is counted from - the raw areas,
    /// when <paramref name="observed"/> has been floored. Null counts <paramref name="observed"/>.</param>
    /// <returns>Abundance NaN for every sample when fewer than <paramref name="minFragments"/> rows match.</returns>
    public static NoiseGatedFit NoiseGatedRatio(
        double[,] observed,
        double[] lib,
        int minFragments,
        double outlierThreshold,
        double noiseSigma,
        Func<double, double>? fragmentSd,
        bool removeOutliers = true,
        double[,]? retainedSource = null,
        int maxRemovals = 5)
    {
        var t = observed.GetLength(0);
        var s = observed.GetLength(1);
        var abundance = new double[s];
        var retained = new double[s];
        var kept = new int[s];

        var matched = new List<int>(t);
        for (var k = 0; k < t; k++)
            if (!double.IsNaN(lib[k]) && lib[k] > 0)
                matched.Add(k);
        if (matched.Count < minFragments)
        {
            Array.Fill(abundance, double.NaN);
            return new NoiseGatedFit(abundance, retained, kept);
        }

        double libSum = 0;
        foreach (var k in matched)
            libSum += lib[k];
        var gate = fragmentSd is not null && noiseSigma > 0;
        var src = retainedSource ?? observed;

        var inc = new List<int>(matched.Count);
        for (var j = 0; j < s; j++)
        {
            inc.Clear();
            inc.AddRange(matched);
            for (var iter = 0; iter < maxRemovals; iter++)
            {
                var c = Scale(observed, lib, inc, j);
                if (!removeOutliers || inc.Count <= minFragments)
                    break;
                var worst = -1;
                var worstNr = double.NegativeInfinity;
                foreach (var k in inc)
                {
                    var o = Clean(observed[k, j]);
                    var pred = lib[k] * c;
                    var nr = pred > 0 ? (o - pred) / pred : 0.0;
                    var flagged = nr > outlierThreshold && (!gate || o - pred > noiseSigma * fragmentSd!(pred));
                    if (flagged && nr > worstNr)
                    {
                        worst = k;
                        worstNr = nr;
                    }
                }
                if (worst < 0)
                    break;
                inc.Remove(worst);
            }
            abundance[j] = Scale(observed, lib, inc, j) * libSum;
            double r = 0;
            foreach (var k in inc)
                r += Clean(src[k, j]);
            retained[j] = r;
            kept[j] = inc.Count;
        }
        return new NoiseGatedFit(abundance, retained, kept);
    }

    private static double Scale(double[,] observed, double[] lib, List<int> rows, int j)
    {
        double so = 0, sl = 0;
        foreach (var k in rows)
        {
            so += Clean(observed[k, j]);
            sl += lib[k];
        }
        return so / sl;
    }

    private static double Clean(double v) => double.IsNaN(v) ? 0.0 : Math.Max(v, 0.0);

    /// <summary>
    /// Each transition's library intensity, matched by product m/z; NaN where nothing matches. Null when
    /// the peptide/charge is not in the library at all (the caller falls back to a plain sum).
    /// </summary>
    public static double[]? MatchFragments(
        SpectralLibrary library, string modifiedSequence, int charge, double[] productMz,
        double mzTolerance, out int nMatched)
    {
        nMatched = 0;
        var spectrum = library.GetSpectrum(modifiedSequence, charge);
        if (spectrum is null)
            return null;
        var lib = new double[productMz.Length];
        for (var i = 0; i < productMz.Length; i++)
        {
            var m = SpectralLibrary.MatchByMz(spectrum, productMz[i], mzTolerance);
            if (m.HasValue)
            {
                lib[i] = m.Value;
                nMatched++;
            }
            else
            {
                lib[i] = double.NaN;
            }
        }
        return lib;
    }

    /// <summary>
    /// Median-polish fit with the library as a fixed prior. <paramref name="observed"/> is a
    /// (T x S) LINEAR matrix; <paramref name="lib"/> is the (T) library intensity vector (NaN or
    /// &lt;=0 marks an unmatched transition, excluded from the fit). Returns per-sample LINEAR
    /// abundance = exp(scale_s) * sum(library).
    /// </summary>
    public static double[] MedianPolish(
        double[,] observed,
        double[] lib,
        int minFragments = 2,
        double outlierThreshold = 1.0,
        int maxIterations = 5,
        bool removeOutliers = true)
    {
        var t = observed.GetLength(0);
        var s = observed.GetLength(1);
        var result = new double[s];

        var validRows = new List<int>(t);
        for (var i = 0; i < t; i++)
            if (!double.IsNaN(lib[i]) && lib[i] > 0)
                validRows.Add(i);
        var nv = validRows.Count;
        if (nv < minFragments)
        {
            Array.Fill(result, double.NaN);
            return result;
        }

        var libV = new double[nv];
        var logLib = new double[nv];
        double libSum = 0;
        for (var k = 0; k < nv; k++)
        {
            libV[k] = lib[validRows[k]];
            libSum += libV[k];
            logLib[k] = Math.Log(libV[k]);
        }

        var obsV = new double[nv, s];
        var logObs = new double[nv, s];
        var included = new bool[nv, s];
        for (var k = 0; k < nv; k++)
        {
            for (var j = 0; j < s; j++)
            {
                var o = observed[validRows[k], j];
                obsV[k, j] = o;
                var safe = !double.IsNaN(o) && o > 0 ? o : double.NaN;
                logObs[k, j] = double.IsNaN(safe) ? double.NaN : Math.Log(safe);
                included[k, j] = !double.IsNaN(logObs[k, j]);
            }
        }

        var beta = new double[s];
        for (var iter = 0; iter < maxIterations; iter++)
        {
            ComputeBeta(beta, logObs, logLib, included, nv, s);
            if (!removeOutliers)
                break;

            var anyOutlier = false;
            for (var j = 0; j < s; j++)
            {
                var scale = Math.Exp(beta[j]);
                var nIncluded = 0;
                for (var k = 0; k < nv; k++)
                    if (included[k, j])
                        nIncluded++;
                if (nIncluded <= minFragments)
                    continue;

                var worstRes = double.NegativeInfinity;
                var worstK = -1;
                for (var k = 0; k < nv; k++)
                {
                    if (!included[k, j])
                        continue;
                    var pred = libV[k] * scale;
                    var obs = double.IsNaN(obsV[k, j]) ? 0.0 : obsV[k, j];
                    var nr = pred > 0 ? (obs - pred) / pred : 0.0;
                    if (nr > worstRes)
                    {
                        worstRes = nr;
                        worstK = k;
                    }
                }
                if (worstK >= 0 && worstRes > outlierThreshold)
                {
                    included[worstK, j] = false;
                    anyOutlier = true;
                }
            }
            if (!anyOutlier)
                break;
        }

        ComputeBeta(beta, logObs, logLib, included, nv, s);
        for (var j = 0; j < s; j++)
        {
            var nIncluded = 0;
            var nObs = 0;
            for (var k = 0; k < nv; k++)
            {
                if (included[k, j])
                    nIncluded++;
                if (!double.IsNaN(logObs[k, j]))
                    nObs++;
            }
            result[j] = nIncluded < minFragments || nObs == 0 || double.IsNaN(beta[j])
                ? double.NaN
                : Math.Exp(beta[j]) * libSum;
        }
        return result;
    }

    private static void ComputeBeta(double[] beta, double[,] logObs, double[] logLib, bool[,] included, int nv, int s)
    {
        var buf = new List<double>(nv);
        for (var j = 0; j < s; j++)
        {
            buf.Clear();
            for (var k = 0; k < nv; k++)
            {
                if (!included[k, j])
                    continue;
                var d = logObs[k, j] - logLib[k];
                if (!double.IsNaN(d))
                    buf.Add(d);
            }
            beta[j] = buf.Count == 0 ? double.NaN : Stats.NanMedian(buf.ToArray());
        }
    }

    /// <summary>
    /// Roll up one peptide charge state: match each transition to the library by product m/z,
    /// then median-polish. Returns per-sample LINEAR abundance. If the peptide is not in the
    /// library, falls back to a simple per-sample sum (matching rollup_peptide's fallback).
    /// </summary>
    public static double[] RollupCharge(
        SpectralLibrary library,
        string modifiedSequence,
        int charge,
        double[] productMz,
        double[,] observedLinear,
        int minFragments,
        double mzTolerance,
        double outlierThreshold,
        bool removeOutliers = true)
    {
        var t = observedLinear.GetLength(0);
        var s = observedLinear.GetLength(1);
        var lib = MatchFragments(library, modifiedSequence, charge, productMz, mzTolerance, out var nMatched);

        if (lib is null)
        {
            var sum = new double[s];
            for (var j = 0; j < s; j++)
            {
                double acc = 0;
                var any = false;
                for (var i = 0; i < t; i++)
                {
                    var v = observedLinear[i, j];
                    if (!double.IsNaN(v) && v > 0)
                    {
                        acc += v;
                        any = true;
                    }
                }
                sum[j] = any ? acc : double.NaN;
            }
            return sum;
        }

        if (nMatched < minFragments)
        {
            var nan = new double[s];
            Array.Fill(nan, double.NaN);
            return nan;
        }

        return MedianPolish(observedLinear, lib, minFragments, outlierThreshold, removeOutliers: removeOutliers);
    }
}
