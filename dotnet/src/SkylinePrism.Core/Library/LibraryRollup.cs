using System;
using System.Collections.Generic;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.Library;

/// <summary>
/// Library-assisted transition rollup, porting spectral_library.
/// library_median_polish_rollup_vectorized + library_assisted_rollup_peptide. Uses the spectral
/// library as a prior for transition (row) effects and estimates each sample's scale by the MEDIAN
/// of log(observed) - log(library), which is robust to interfered transitions. High positive
/// residuals (observed &gt; expected) are iteratively removed as interference.
/// </summary>
public static class LibraryRollup
{
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
        var spectrum = library.GetSpectrum(modifiedSequence, charge);

        if (spectrum is null)
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

        var lib = new double[t];
        var nMatched = 0;
        for (var i = 0; i < t; i++)
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
        if (nMatched < minFragments)
        {
            var nan = new double[s];
            Array.Fill(nan, double.NaN);
            return nan;
        }

        return MedianPolish(observedLinear, lib, minFragments, outlierThreshold, removeOutliers: removeOutliers);
    }
}
