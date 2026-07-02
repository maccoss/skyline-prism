using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Library;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.Rollup;

/// <summary>
/// Stage 2 transition-&gt;peptide rollup driver, porting chunked_processing.rollup_transitions_sorted
/// + _process_single_peptide. A single DuckDB reader streams the peptide-sorted merged parquet
/// (one <see cref="PeptideBlock"/> per peptide); blocks are processed by a bounded pool of worker
/// threads (per-peptide rollup is pure/thread-safe) and results are streamed to the wide LOG2
/// parquet via a single writer thread that flushes row groups periodically. This keeps CPU busy
/// while holding only a bounded number of in-flight peptides in memory (no accumulate-all-then-write).
/// Output row order is not the input peptide order under parallelism; downstream stages key by
/// peptide, so this is immaterial.
/// </summary>
public sealed class TransitionRollup
{
    public sealed record Result(int NPeptides, int NFiltered, IReadOnlyList<string> Samples);

    private sealed record PeptideResult(
        string Pep, long Nt, double Rt, double[] Vals, List<(string Tid, double[] Res)>? Residuals);

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
            TransitionRollupMethod.TopN =>
                new TopNRollup(cfg.TopNCount, cfg.MinTransitions, cfg.TopNSelection, cfg.TopNWeighting),
            TransitionRollupMethod.Consensus => new ConsensusRollup(cfg.MinTransitions, cfg.ConsensusRegularization),
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
        var topNCorr = cfg.Method == TransitionRollupMethod.TopN
            && cfg.TopNSelection == "correlation" && cols.ShapeCorrelation is not null;

        var pepMeta = new (string, Type)[]
        {
            (cols.Peptide, typeof(string)), ("n_transitions", typeof(long)), ("mean_rt", typeof(double)),
        };
        var pepWriter = StreamingWideWriter.Create(outputPath, pepMeta, samples);
        var resWriter = captureResiduals
            ? StreamingWideWriter.Create(cfg.ResidualsPath!,
                new (string, Type)[] { (cols.Peptide, typeof(string)), ("transition_id", typeof(string)) }, samples)
            : null;

        try
        {
            var sink = new PeptideStreamSink(pepWriter, resWriter, samples.Count, Math.Max(1, cfg.FlushRows));
            var nFiltered = 0;
            var dop = ResolveDop(cfg.MaxDegreeOfParallelism);

            PeptideResult? Process(PeptideBlock block) => isLibrary
                ? ProcessPeptideLibrary(block, cfg, samples.Count, sampleIndex, library!)
                : topNCorr
                    ? ProcessPeptideTopNCorr(block, cfg, samples.Count, sampleIndex)
                    : ProcessPeptide(block, cfg, samples.Count, sampleIndex, method!, captureResiduals);

            if (dop <= 1)
            {
                foreach (var block in MergedParquetReader.StreamPeptideBlocks(
                    mergedParquet, cols, includeProductMz: isLibrary, includeShapeCorr: topNCorr))
                {
                    var r = Process(block);
                    if (r is null)
                        nFiltered++;
                    else
                        sink.Add(r);
                }
            }
            else
            {
                RunParallel(mergedParquet, cols, isLibrary, topNCorr, dop, Process, sink, ref nFiltered);
            }

            sink.FlushAll();
            return new Result((int)sink.NPeptides, nFiltered, samples);
        }
        finally
        {
            pepWriter.Dispose();
            resWriter?.Dispose();
        }
    }

    private static void RunParallel(
        string mergedParquet, SkylineColumns cols, bool isLibrary, bool topNCorr, int dop,
        Func<PeptideBlock, PeptideResult?> process, PeptideStreamSink sink, ref int nFiltered)
    {
        // Single producer -> bounded queue -> N consumers -> single writer (this thread). The
        // bounded capacities cap the number of in-flight peptides so RAM stays flat.
        using var inputQ = new BlockingCollection<PeptideBlock>(dop * 4);
        using var outputQ = new BlockingCollection<PeptideResult>(dop * 4);
        Exception? error = null;
        var filtered = 0;

        var producer = Task.Run(() =>
        {
            try
            {
                foreach (var b in MergedParquetReader.StreamPeptideBlocks(
                    mergedParquet, cols, includeProductMz: isLibrary, includeShapeCorr: topNCorr))
                    inputQ.Add(b);
            }
            catch (Exception ex) { Interlocked.CompareExchange(ref error, ex, null); }
            finally { inputQ.CompleteAdding(); }
        });

        var consumers = new Task[dop];
        for (var i = 0; i < dop; i++)
        {
            consumers[i] = Task.Run(() =>
            {
                try
                {
                    foreach (var block in inputQ.GetConsumingEnumerable())
                    {
                        var r = process(block);
                        if (r is null)
                            Interlocked.Increment(ref filtered);
                        else
                            outputQ.Add(r);
                    }
                }
                catch (Exception ex) { Interlocked.CompareExchange(ref error, ex, null); }
            });
        }

        var closer = Task.Run(() =>
        {
            try { Task.WaitAll(consumers); }
            finally { outputQ.CompleteAdding(); }
        });

        foreach (var r in outputQ.GetConsumingEnumerable())
            sink.Add(r);

        Task.WaitAll(producer, closer);
        nFiltered += filtered;
        if (error is not null)
            throw new InvalidOperationException("Transition rollup failed: " + error.Message, error);
    }

    private static int ResolveDop(int configured)
    {
        var cores = Math.Max(1, Environment.ProcessorCount);
        return configured <= 0 ? cores : Math.Min(configured, cores);
    }

    private static PeptideResult? ProcessPeptide(
        PeptideBlock block,
        TransitionRollupConfig cfg,
        int nSamples,
        IReadOnlyDictionary<string, int> sampleIndex,
        IRollupMethod method,
        bool captureResiduals)
    {
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

        if (captureResiduals)
        {
            // Median-polish residuals path: capture per-transition residuals (interference /
            // proteoform signal) alongside the peptide abundance (col effects + log2(n)).
            var polish = TukeyMedianPolish.Run(pre.Log2Matrix);
            var scale = Math.Log2(nt);
            var vals = new double[nSamples];
            for (var j = 0; j < nSamples; j++)
                vals[j] = polish.ColEffects[j] + scale;

            var residuals = new List<(string, double[])>(nt);
            for (var r = 0; r < nt; r++)
            {
                var row = new double[nSamples];
                for (var j = 0; j < nSamples; j++)
                    row[j] = polish.Residuals[r, j];
                residuals.Add((tidNames[r], row));
            }
            return new PeptideResult(block.Peptide, nt, meanRt, vals, residuals);
        }

        var v = method.Aggregate(pre.Log2Matrix);
        return new PeptideResult(block.Peptide, nt, meanRt, v, null);
    }

    /// <summary>
    /// Top-N rollup with CORRELATION selection: builds the transition x sample shape-correlation
    /// matrix (first-non-null, missing = 0) alongside the imputed intensity matrix and selects the
    /// top-N transitions by median shape correlation.
    /// </summary>
    private static PeptideResult? ProcessPeptideTopNCorr(
        PeptideBlock block, TransitionRollupConfig cfg, int nSamples, IReadOnlyDictionary<string, int> sampleIndex)
    {
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

        var matrix = new double[nt, nSamples];
        var shape = new double[nt, nSamples]; // default 0 = low correlation
        var filled = new bool[nt, nSamples];
        var shapeFilled = new bool[nt, nSamples];
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
                if (i < block.ShapeCorrelation.Count)
                {
                    var sc = block.ShapeCorrelation[i];
                    if (!shapeFilled[ti, si] && !double.IsNaN(sc))
                    {
                        shape[ti, si] = sc;
                        shapeFilled[ti, si] = true;
                    }
                }
            }
            rtBuf.Add(block.RetentionTime[i]);
        }

        var meanRt = Stats.NanMean(rtBuf.ToArray());
        var pre = RollupPreprocess.ImputeAndLog2(matrix, cfg.LogTransform);
        var vals = TopNRollup.Compute(
            pre.Log2Matrix, shape, cfg.TopNCount, cfg.MinTransitions, "correlation", cfg.TopNWeighting);
        return new PeptideResult(block.Peptide, nt, meanRt, vals, null);
    }

    /// <summary>
    /// Library-assisted per-peptide rollup: impute to LINEAR, group transitions by precursor
    /// charge, match each to the library by product m/z, median-polish fit per charge, sum charge
    /// abundances (LINEAR), then log2.
    /// </summary>
    private static PeptideResult? ProcessPeptideLibrary(
        PeptideBlock block,
        TransitionRollupConfig cfg,
        int nSamples,
        IReadOnlyDictionary<string, int> sampleIndex,
        SpectralLibrary library)
    {
        var tidIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        var tidCharge = new List<int>();
        var tidMz = new List<double>();
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
                var val = abund[b];
                if (!double.IsNaN(val) && val > 0)
                {
                    final[b] += val;
                    hasValue[b] = true;
                }
            }
        }

        var vals = new double[nSamples];
        for (var b = 0; b < nSamples; b++)
            vals[b] = hasValue[b] ? Math.Log2(final[b]) : double.NaN;
        return new PeptideResult(block.Peptide, nt, meanRt, vals, null);
    }

    private static int ParseChargeOrDefault(string s)
    {
        if (int.TryParse(s, NumberStyles.Any, CultureInfo.InvariantCulture, out var c))
            return c;
        return double.TryParse(s, NumberStyles.Any, CultureInfo.InvariantCulture, out var d) ? (int)d : 0;
    }

    private static string TransitionId(PeptideBlock block, int i)
        => block.Ion[i] + "_z" + block.PrecursorCharge[i] + "_" + block.ProductCharge[i];

    /// <summary>
    /// Batches peptide (and residual) rows and flushes them to the streaming writers every
    /// <c>flushRows</c> peptides. Driven by ONE thread (the writer), so it needs no locking.
    /// </summary>
    private sealed class PeptideStreamSink
    {
        private readonly StreamingWideWriter _pep;
        private readonly StreamingWideWriter? _res;
        private readonly int _nSamples;
        private readonly int _flush;

        private readonly List<string> _bPep = new();
        private readonly List<long> _bNt = new();
        private readonly List<double> _bRt = new();
        private readonly List<double>[] _bSamp;

        private readonly List<string> _rPep = new();
        private readonly List<string> _rTid = new();
        private readonly List<double>[] _rSamp;

        public long NPeptides { get; private set; }

        public PeptideStreamSink(StreamingWideWriter pep, StreamingWideWriter? res, int nSamples, int flush)
        {
            _pep = pep;
            _res = res;
            _nSamples = nSamples;
            _flush = flush;
            _bSamp = new List<double>[nSamples];
            _rSamp = new List<double>[nSamples];
            for (var s = 0; s < nSamples; s++)
            {
                _bSamp[s] = new List<double>();
                _rSamp[s] = new List<double>();
            }
        }

        public void Add(PeptideResult r)
        {
            _bPep.Add(r.Pep);
            _bNt.Add(r.Nt);
            _bRt.Add(r.Rt);
            for (var s = 0; s < _nSamples; s++)
                _bSamp[s].Add(r.Vals[s]);
            NPeptides++;

            if (_res is not null && r.Residuals is not null)
            {
                foreach (var (tid, res) in r.Residuals)
                {
                    _rPep.Add(r.Pep);
                    _rTid.Add(tid);
                    for (var s = 0; s < _nSamples; s++)
                        _rSamp[s].Add(res[s]);
                }
            }

            if (_bPep.Count >= _flush)
                FlushPep();
            if (_rPep.Count >= _flush)
                FlushRes();
        }

        public void FlushAll()
        {
            FlushPep();
            FlushRes();
        }

        private void FlushPep()
        {
            if (_bPep.Count == 0)
                return;
            var meta = new Array[] { _bPep.ToArray(), _bNt.ToArray(), _bRt.ToArray() };
            var samp = new double[_nSamples][];
            for (var s = 0; s < _nSamples; s++)
                samp[s] = _bSamp[s].ToArray();
            _pep.WriteRowGroup(meta, samp);

            _bPep.Clear();
            _bNt.Clear();
            _bRt.Clear();
            for (var s = 0; s < _nSamples; s++)
                _bSamp[s].Clear();
        }

        private void FlushRes()
        {
            if (_res is null || _rPep.Count == 0)
                return;
            var meta = new Array[] { _rPep.ToArray(), _rTid.ToArray() };
            var samp = new double[_nSamples][];
            for (var s = 0; s < _nSamples; s++)
                samp[s] = _rSamp[s].ToArray();
            _res.WriteRowGroup(meta, samp);

            _rPep.Clear();
            _rTid.Clear();
            for (var s = 0; s < _nSamples; s++)
                _rSamp[s].Clear();
        }
    }
}
