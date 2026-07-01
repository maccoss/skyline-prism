using System;
using System.Collections.Generic;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.Rollup;

/// <summary>
/// Stage 2 transition-&gt;peptide rollup driver, porting chunked_processing.rollup_transitions_sorted
/// (single-threaded path) + _process_single_peptide. Streams the merged parquet peptide-by-peptide,
/// builds the transition x sample matrix (pivot first-non-null), imputes + log2, aggregates per the
/// configured method, and writes the wide LOG2 peptides_rollup parquet.
/// </summary>
public sealed class TransitionRollup
{
    public sealed record Result(int NPeptides, int NFiltered, IReadOnlyList<string> Samples);

    public static Result Run(
        string mergedParquet,
        SkylineColumns cols,
        TransitionRollupConfig cfg,
        string outputPath,
        IReadOnlyList<string>? samples = null)
    {
        samples ??= MergedParquetReader.GetSortedSamples(mergedParquet, cols.Sample);
        var sampleIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var i = 0; i < samples.Count; i++)
            sampleIndex[samples[i]] = i;

        IRollupMethod method = cfg.Method switch
        {
            TransitionRollupMethod.Sum => new SumRollup(),
            TransitionRollupMethod.MedianPolish =>
                new MedianPolishRollup(addLog2NOffset: true, minTransitions: cfg.MinTransitions),
            _ => throw new NotSupportedException($"Unsupported method {cfg.Method}"),
        };

        var rows = new List<(string Pep, long Nt, double Rt, double[] Vals)>();
        var nFiltered = 0;

        foreach (var block in MergedParquetReader.StreamPeptideBlocks(mergedParquet, cols))
        {
            var res = ProcessPeptide(block, cfg, samples.Count, sampleIndex, method);
            if (res is null)
            {
                nFiltered++;
                continue;
            }
            rows.Add(res.Value);
        }

        WriteOutput(outputPath, cols.Peptide, samples, rows);
        return new Result(rows.Count, nFiltered, samples);
    }

    private static (string Pep, long Nt, double Rt, double[] Vals)? ProcessPeptide(
        PeptideBlock block,
        TransitionRollupConfig cfg,
        int nSamples,
        IReadOnlyDictionary<string, int> sampleIndex,
        IRollupMethod method)
    {
        // Non-precursor rows + distinct transition ids (first-appearance order).
        var tidIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        var rowIdxs = new List<int>(block.RowCount);
        for (var i = 0; i < block.RowCount; i++)
        {
            if (cfg.ExcludePrecursor && block.Ion[i].StartsWith("precursor", StringComparison.Ordinal))
                continue;
            rowIdxs.Add(i);
            var tid = TransitionId(block, i);
            if (!tidIndex.ContainsKey(tid))
                tidIndex[tid] = tidIndex.Count;
        }

        var nt = tidIndex.Count;
        if (nt < cfg.MinTransitions)
            return null;

        // pivot first-non-null: transition x sample linear matrix.
        var matrix = new double[nt, nSamples];
        var filled = new bool[nt, nSamples];
        for (var a = 0; a < nt; a++)
            for (var b = 0; b < nSamples; b++)
                matrix[a, b] = double.NaN;

        var rtBuf = new List<double>(rowIdxs.Count);
        foreach (var i in rowIdxs)
        {
            var ti = tidIndex[TransitionId(block, i)];
            if (sampleIndex.TryGetValue(block.Sample[i], out var si))
            {
                var area = block.Area[i];
                if (!filled[ti, si] && !double.IsNaN(area))
                {
                    matrix[ti, si] = area;
                    filled[ti, si] = true;
                }
            }
            rtBuf.Add(block.RetentionTime[i]);
        }

        var meanRt = Stats.NanMean(rtBuf.ToArray());
        var pre = RollupPreprocess.ImputeAndLog2(matrix, cfg.LogTransform);
        var vals = method.Aggregate(pre.Log2Matrix);
        return (block.Peptide, nt, meanRt, vals);
    }

    private static string TransitionId(PeptideBlock block, int i)
        => block.Ion[i] + "_z" + block.PrecursorCharge[i] + "_" + block.ProductCharge[i];

    private static void WriteOutput(
        string outputPath,
        string peptideCol,
        IReadOnlyList<string> samples,
        List<(string Pep, long Nt, double Rt, double[] Vals)> rows)
    {
        var n = rows.Count;
        var pepArr = new string[n];
        var ntArr = new long[n];
        var rtArr = new double[n];
        var sampleCols = new double[samples.Count][];
        for (var s = 0; s < samples.Count; s++)
            sampleCols[s] = new double[n];

        for (var r = 0; r < n; r++)
        {
            pepArr[r] = rows[r].Pep;
            ntArr[r] = rows[r].Nt;
            rtArr[r] = rows[r].Rt;
            var vals = rows[r].Vals;
            for (var s = 0; s < samples.Count; s++)
                sampleCols[s][r] = vals[s];
        }

        var meta = new List<ParquetWideWriter.MetaColumn>
        {
            ParquetWideWriter.Strings(peptideCol, pepArr),
            ParquetWideWriter.Longs("n_transitions", ntArr),
            ParquetWideWriter.Doubles("mean_rt", rtArr),
        };
        ParquetWideWriter.Write(outputPath, meta, samples, sampleCols, n);
    }
}
