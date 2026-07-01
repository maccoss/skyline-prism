using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Library;
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

        var isLibrary = cfg.Method == TransitionRollupMethod.LibraryAssist;
        IRollupMethod? method = cfg.Method switch
        {
            TransitionRollupMethod.Sum => new SumRollup(),
            TransitionRollupMethod.MedianPolish =>
                new MedianPolishRollup(addLog2NOffset: true, minTransitions: cfg.MinTransitions),
            TransitionRollupMethod.LibraryAssist => null,
            _ => throw new NotSupportedException($"Unsupported method {cfg.Method}"),
        };

        SpectralLibrary? library = null;
        if (isLibrary)
        {
            if (string.IsNullOrWhiteSpace(cfg.LibraryPath))
                throw new InvalidOperationException(
                    "Library-assisted rollup requires a spectral library (.blib) path.");
            library = SpectralLibrary.LoadBlib(cfg.LibraryPath);
        }

        var captureResiduals = !isLibrary && cfg.Method == TransitionRollupMethod.MedianPolish
            && !string.IsNullOrEmpty(cfg.ResidualsPath);
        var sink = captureResiduals ? new ResidualSink() : null;

        var rows = new List<(string Pep, long Nt, double Rt, double[] Vals)>();
        var nFiltered = 0;

        foreach (var block in MergedParquetReader.StreamPeptideBlocks(mergedParquet, cols, includeProductMz: isLibrary))
        {
            var res = isLibrary
                ? ProcessPeptideLibrary(block, cfg, samples.Count, sampleIndex, library!)
                : ProcessPeptide(block, cfg, samples.Count, sampleIndex, method!, sink);
            if (res is null)
            {
                nFiltered++;
                continue;
            }
            rows.Add(res.Value);
        }

        WriteOutput(outputPath, cols.Peptide, samples, rows);
        if (sink is not null)
            WriteResiduals(cfg.ResidualsPath!, cols.Peptide, samples, sink);
        return new Result(rows.Count, nFiltered, samples);
    }

    private static (string Pep, long Nt, double Rt, double[] Vals)? ProcessPeptide(
        PeptideBlock block,
        TransitionRollupConfig cfg,
        int nSamples,
        IReadOnlyDictionary<string, int> sampleIndex,
        IRollupMethod method,
        ResidualSink? residuals = null)
    {
        // Non-precursor rows + distinct transition ids (first-appearance order).
        var tidIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        var tidNames = new List<string>();
        var rowIdxs = new List<int>(block.RowCount);
        for (var i = 0; i < block.RowCount; i++)
        {
            if (cfg.ExcludePrecursor && block.Ion[i].StartsWith("precursor", StringComparison.Ordinal))
                continue;
            rowIdxs.Add(i);
            var tid = TransitionId(block, i);
            if (!tidIndex.ContainsKey(tid))
            {
                tidIndex[tid] = tidIndex.Count;
                tidNames.Add(tid);
            }
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

        if (residuals is not null)
        {
            // Median-polish residuals path: capture per-transition residuals (interference /
            // proteoform signal) alongside the peptide abundance (col effects + log2(n)).
            var polish = TukeyMedianPolish.Run(pre.Log2Matrix);
            var scale = Math.Log2(nt);
            var vals2 = new double[nSamples];
            for (var j = 0; j < nSamples; j++)
                vals2[j] = polish.ColEffects[j] + scale;
            for (var r = 0; r < nt; r++)
            {
                var row = new double[nSamples];
                for (var j = 0; j < nSamples; j++)
                    row[j] = polish.Residuals[r, j];
                residuals.Peptide.Add(block.Peptide);
                residuals.TransitionId.Add(tidNames[r]);
                residuals.Values.Add(row);
            }
            return (block.Peptide, nt, meanRt, vals2);
        }

        var vals = method.Aggregate(pre.Log2Matrix);
        return (block.Peptide, nt, meanRt, vals);
    }

    private sealed class ResidualSink
    {
        public readonly List<string> Peptide = new();
        public readonly List<string> TransitionId = new();
        public readonly List<double[]> Values = new();
    }

    private static void WriteResiduals(
        string path, string peptideCol, IReadOnlyList<string> samples, ResidualSink sink)
    {
        var n = sink.Peptide.Count;
        var sampleCols = new double[samples.Count][];
        for (var s = 0; s < samples.Count; s++)
        {
            sampleCols[s] = new double[n];
            for (var r = 0; r < n; r++)
                sampleCols[s][r] = sink.Values[r][s];
        }
        var meta = new List<ParquetWideWriter.MetaColumn>
        {
            ParquetWideWriter.Strings(peptideCol, sink.Peptide.ToArray()),
            ParquetWideWriter.Strings("transition_id", sink.TransitionId.ToArray()),
        };
        ParquetWideWriter.Write(path, meta, samples, sampleCols, n);
    }

    /// <summary>
    /// Library-assisted per-peptide rollup, porting the chunked_processing "library-assisted"
    /// branch: impute to LINEAR, group transitions by precursor charge, match each to the library
    /// by product m/z, median-polish fit per charge, sum charge abundances (LINEAR), then log2.
    /// </summary>
    private static (string Pep, long Nt, double Rt, double[] Vals)? ProcessPeptideLibrary(
        PeptideBlock block,
        TransitionRollupConfig cfg,
        int nSamples,
        IReadOnlyDictionary<string, int> sampleIndex,
        SpectralLibrary library)
    {
        var tidIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        var rowIdxs = new List<int>(block.RowCount);
        var tidCharge = new List<int>();
        var tidMz = new List<double>();
        for (var i = 0; i < block.RowCount; i++)
        {
            if (cfg.ExcludePrecursor && block.Ion[i].StartsWith("precursor", StringComparison.Ordinal))
                continue;
            rowIdxs.Add(i);
            var tid = TransitionId(block, i);
            if (!tidIndex.ContainsKey(tid))
            {
                tidIndex[tid] = tidIndex.Count;
                tidCharge.Add(ParseChargeOrDefault(block.PrecursorCharge[i]));
                tidMz.Add(i < block.ProductMz.Count ? block.ProductMz[i] : double.NaN);
            }
        }

        var nt = tidIndex.Count;
        if (nt < cfg.MinTransitions)
            return null;

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

        // Impute in log2 then back to linear = 2**(imputed log2 matrix), as in Python.
        var pre = RollupPreprocess.ImputeAndLog2(matrix, logTransform: true);
        var linear = new double[nt, nSamples];
        for (var a = 0; a < nt; a++)
            for (var b = 0; b < nSamples; b++)
                linear[a, b] = Math.Pow(2, pre.Log2Matrix[a, b]);

        var charges = tidCharge.Where(c => c > 0).Distinct().OrderBy(c => c).ToList();
        if (charges.Count == 0)
            charges.Add(2);

        var final = new double[nSamples];
        var hasValue = new bool[nSamples];
        foreach (var charge in charges)
        {
            var idxs = new List<int>();
            for (var tt = 0; tt < nt; tt++)
                if (tidCharge[tt] == charge)
                    idxs.Add(tt);
            if (idxs.Count == 0)
                continue;

            var obs = new double[idxs.Count, nSamples];
            var mz = new double[idxs.Count];
            for (var r = 0; r < idxs.Count; r++)
            {
                mz[r] = tidMz[idxs[r]];
                for (var b = 0; b < nSamples; b++)
                    obs[r, b] = linear[idxs[r], b];
            }

            var abund = LibraryRollup.RollupCharge(
                library, block.Peptide, charge, mz, obs,
                cfg.LibraryMinFragments, cfg.LibraryMzTolerance, cfg.LibraryOutlierThreshold);
            for (var b = 0; b < nSamples; b++)
            {
                var v = abund[b];
                if (!double.IsNaN(v) && v > 0)
                {
                    final[b] += v;
                    hasValue[b] = true;
                }
            }
        }

        var vals = new double[nSamples];
        for (var b = 0; b < nSamples; b++)
            vals[b] = hasValue[b] ? Math.Log2(final[b]) : double.NaN;
        return (block.Peptide, nt, meanRt, vals);
    }

    private static int ParseChargeOrDefault(string s)
    {
        if (int.TryParse(s, NumberStyles.Any, CultureInfo.InvariantCulture, out var c))
            return c;
        return double.TryParse(s, NumberStyles.Any, CultureInfo.InvariantCulture, out var d) ? (int)d : 0;
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
