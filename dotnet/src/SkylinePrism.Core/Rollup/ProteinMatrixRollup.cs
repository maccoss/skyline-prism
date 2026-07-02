using System;
using System.Collections.Generic;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.Rollup;

public enum ProteinRollupMethod
{
    MedianPolish,
    Sum,
    TopN,
    MaxLfq,
    Ibaq,
}

/// <summary>
/// Peptide-&gt;protein matrix rollup, porting rollup.py:rollup_protein_matrix (the single
/// source of truth). Dispatch by peptide count:
///   0 -&gt; NaN, 1 -&gt; peptide directly, &lt; min_peptides -&gt; sum_linear, else the method.
/// The median-polish branch uses col_effects WITHOUT the +log2(n) offset (unlike the
/// peptide stage). Input/output LOG2.
/// </summary>
public static class ProteinMatrixRollup
{
    public static double[] Aggregate(
        double[,] log2Matrix, ProteinRollupMethod method, int minPeptides, int topN = 3, int nTheoretical = -1)
    {
        var nPep = log2Matrix.GetLength(0);
        var nCols = log2Matrix.GetLength(1);

        if (nPep == 0)
        {
            var nan = new double[nCols];
            for (var j = 0; j < nCols; j++)
                nan[j] = double.NaN;
            return nan;
        }

        if (nPep == 1)
        {
            var row = new double[nCols];
            for (var j = 0; j < nCols; j++)
                row[j] = log2Matrix[0, j];
            return row;
        }

        // sum_linear = log2(clip(sum(2^m), 1)); SumRollup computes exactly this.
        if (nPep < minPeptides)
            return new SumRollup().Aggregate(log2Matrix);

        return method switch
        {
            ProteinRollupMethod.MedianPolish =>
                new MedianPolishRollup(addLog2NOffset: false).Aggregate(log2Matrix),
            ProteinRollupMethod.Sum => new SumRollup().Aggregate(log2Matrix),
            ProteinRollupMethod.TopN => TopN(log2Matrix, topN),
            ProteinRollupMethod.MaxLfq => MaxLfq(log2Matrix),
            ProteinRollupMethod.Ibaq => Ibaq(log2Matrix, nTheoretical > 0 ? nTheoretical : nPep),
            _ => throw new NotSupportedException($"Unsupported protein method {method}"),
        };
    }

    /// <summary>
    /// iBAQ (rollup.py:rollup_ibaq): log2(sum of LINEAR peptide intensities / n_theoretical_peptides).
    /// n_theoretical falls back to the observed peptide count when no FASTA is available.
    /// </summary>
    private static double[] Ibaq(double[,] m, int nTheoretical)
    {
        var nPep = m.GetLength(0);
        var nCols = m.GetLength(1);
        var result = new double[nCols];
        for (var j = 0; j < nCols; j++)
        {
            double sum = 0;
            for (var p = 0; p < nPep; p++)
            {
                var v = m[p, j];
                if (!double.IsNaN(v))
                    sum += Math.Pow(2, v);
            }
            result[j] = nTheoretical > 0 && sum > 0 ? Math.Log2(sum / nTheoretical) : double.NaN;
        }
        return result;
    }

    /// <summary>
    /// rollup.py:rollup_top_n (default "median_abundance" selection): pick the top-N peptides by
    /// median LOG2 abundance across samples (freq as tie-break) - the SAME peptides for all samples
    /// - then the per-sample mean (NaN-skipping) of those peptides.
    /// </summary>
    private static double[] TopN(double[,] m, int n)
    {
        var nPep = m.GetLength(0);
        var nCols = m.GetLength(1);

        var medianAbund = new double[nPep];
        var freq = new int[nPep];
        var rowBuf = new double[nCols];
        for (var p = 0; p < nPep; p++)
        {
            var cnt = 0;
            for (var j = 0; j < nCols; j++)
            {
                var v = m[p, j];
                if (!double.IsNaN(v))
                    rowBuf[cnt++] = v;
            }
            freq[p] = cnt;
            medianAbund[p] = cnt == 0 ? double.NaN : Stats.NanMedian(rowBuf.AsSpan(0, cnt));
        }

        var order = new int[nPep];
        for (var p = 0; p < nPep; p++)
            order[p] = p;
        Array.Sort(order, (a, b) =>
        {
            var na = double.IsNaN(medianAbund[a]);
            var nb = double.IsNaN(medianAbund[b]);
            if (na || nb)
                return na == nb ? a.CompareTo(b) : (na ? 1 : -1); // NaN-median peptides last
            var c = medianAbund[b].CompareTo(medianAbund[a]); // median desc
            if (c != 0)
                return c;
            var f = freq[b].CompareTo(freq[a]); // frequency desc
            return f != 0 ? f : a.CompareTo(b); // stable
        });

        var nUse = Math.Min(n, nPep);
        var abund = new double[nCols];
        for (var j = 0; j < nCols; j++)
        {
            double sum = 0;
            var cnt = 0;
            for (var k = 0; k < nUse; k++)
            {
                var v = m[order[k], j];
                if (!double.IsNaN(v))
                {
                    sum += v;
                    cnt++;
                }
            }
            abund[j] = cnt > 0 ? sum / cnt : double.NaN;
        }
        return abund;
    }

    /// <summary>
    /// rollup.py:rollup_maxlfq: pairwise median peptide log-ratios between samples, reconstruct the
    /// per-sample profile as the row-mean of the ratio matrix, center to its median, and re-anchor
    /// to the overall level (median of per-sample medians). Input/output LOG2.
    /// </summary>
    private static double[] MaxLfq(double[,] m)
    {
        var nPep = m.GetLength(0);
        var nSamples = m.GetLength(1);

        // Per-sample median over peptides (NaN-skipping).
        var perSampleMed = new double[nSamples];
        var buf = new double[nPep];
        for (var j = 0; j < nSamples; j++)
        {
            var cnt = 0;
            for (var p = 0; p < nPep; p++)
                if (!double.IsNaN(m[p, j]))
                    buf[cnt++] = m[p, j];
            perSampleMed[j] = cnt == 0 ? double.NaN : Stats.NanMedian(buf.AsSpan(0, cnt));
        }

        if (nSamples < 2)
            return perSampleMed;

        // ratio[i,j] = median over peptides of (col_i - col_j), 0 when no shared peptides.
        var ratio = new double[nSamples, nSamples];
        var diffs = new List<double>(nPep);
        for (var i = 0; i < nSamples; i++)
        {
            for (var j = 0; j < nSamples; j++)
            {
                if (i == j)
                    continue;
                diffs.Clear();
                for (var p = 0; p < nPep; p++)
                {
                    var a = m[p, i];
                    var b = m[p, j];
                    if (!double.IsNaN(a) && !double.IsNaN(b))
                        diffs.Add(a - b);
                }
                if (diffs.Count > 0)
                    ratio[i, j] = Stats.NanMedian(diffs.ToArray());
            }
        }

        // abundance[i] = mean over j (incl. the 0 diagonal) of ratio[i,j].
        var abund = new double[nSamples];
        for (var i = 0; i < nSamples; i++)
        {
            double sum = 0;
            for (var j = 0; j < nSamples; j++)
                sum += ratio[i, j];
            abund[i] = sum / nSamples;
        }

        var center = Stats.NanMedian(abund);
        var overall = Stats.NanMedian(perSampleMed);
        for (var i = 0; i < nSamples; i++)
            abund[i] = abund[i] - center + overall;
        return abund;
    }
}
