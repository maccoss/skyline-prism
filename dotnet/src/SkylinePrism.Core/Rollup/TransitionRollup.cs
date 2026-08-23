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
/// + _process_single_peptide. A DuckDB reader streams the merged dataset one partition at a time
/// (one <see cref="PeptideBlock"/> per peptide); blocks are processed by a
/// bounded pool of worker threads (per-peptide rollup is pure/thread-safe) and results are streamed to
/// the wide LOG2 parquet via a single writer thread that flushes row groups periodically. This keeps
/// CPU busy while holding only a bounded number of in-flight peptides in memory (no
/// accumulate-all-then-write).
/// <para>
/// The single reader is this stage's bottleneck (62 minutes at 1.2 of 32 cores on a 20-document
/// cohort). Reading partitions concurrently is the obvious fix and is blocked by a use-after-free in
/// the DuckDB binding, not by anything in the data model - see RunParallel.
/// </para>
/// Output row order IS the input peptide order, at any worker count, and that is a correctness
/// requirement rather than tidiness. This comment used to say the opposite - that order did not
/// matter because downstream stages key by peptide. They do, but ComBat's cross-feature reductions
/// sum over rows in FILE order, and floating-point addition is not associative, so a varying row
/// order changed the reported quantities in their last bits: on a 2-plate cohort only 17% of
/// corrected_proteins cells were bit-identical between two runs of the same binary on the same
/// input. See the reorder buffer in RunParallel, and Stats.NanMeanOrderInvariant for the matching
/// hazard WITHIN a peptide (DuckDB promises no row order there either).
/// </summary>
public sealed class TransitionRollup
{
    public sealed record Result(int NPeptides, int NFiltered, IReadOnlyList<string> Samples);

    private sealed record PeptideResult(
        string Pep, long Nt, double Rt, double[] Vals, List<(string Tid, double[] Res)>? Residuals);

    public static Result Run(
        MergedDataset dataset,
        SkylineColumns cols,
        TransitionRollupConfig cfg,
        string outputPath,
        IReadOnlyList<string>? samples = null,
        CancellationToken cancellationToken = default)
    {
        samples ??= MergedParquetReader.GetSortedSamples(dataset, cols.Sample, cfg.MemoryBudgetMb);
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
                    "Library-assisted rollup requires a spectral library path (.blib or Carafe/DIA-NN .tsv).");
            library = SpectralLibrary.Load(cfg.LibraryPath);
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
            var sink = new PeptideStreamSink(
                pepWriter, resWriter, samples.Count, FlushRowsFor(cfg.FlushRows, samples.Count));
            var nFiltered = 0;
            var dop = ResolveDop(cfg.MaxDegreeOfParallelism);

            PeptideResult? Process(PeptideBlock block) => isLibrary
                ? ProcessPeptideLibrary(block, cfg, samples.Count, library!)
                : topNCorr
                    ? ProcessPeptideTopNCorr(block, cfg, samples.Count)
                    : ProcessPeptide(block, cfg, samples.Count, method!, captureResiduals);

            if (dop <= 1)
            {
                foreach (var block in MergedParquetReader.StreamPeptideBlocks(
                    dataset, cols, samples, includeProductMz: isLibrary, includeShapeCorr: topNCorr,
                    memoryBudgetMb: cfg.MemoryBudgetMb))
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var r = Process(block);
                    if (r is null)
                        nFiltered++;
                    else
                        sink.Add(r);
                }
            }
            else
            {
                RunParallel(dataset, cols, samples, isLibrary, topNCorr, dop, cfg.MemoryBudgetMb,
                    Process, sink, ref nFiltered, cancellationToken);
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
        MergedDataset dataset, SkylineColumns cols, IReadOnlyList<string> samples, bool isLibrary,
        bool topNCorr, int dop, int memoryBudgetMb, Func<PeptideBlock, PeptideResult?> process,
        PeptideStreamSink sink, ref int nFiltered, CancellationToken cancellationToken)
    {
        // ONE producer -> bounded queue -> N consumers -> single writer (this thread). The bounded
        // capacities cap the number of in-flight peptides so RAM stays flat.
        //
        // The single producer is the stage's bottleneck and is known to be so: reading a partition is a
        // DuckDB sort followed by a row-at-a-time walk of the result, and that walk is serial, so on a
        // 20-document cohort this stage ran 62 minutes at 1.2 of 32 cores with the workers starved.
        // Partitions are independent, so reading several at once is the obvious fix and was tried.
        //
        // It is NOT SAFE with this DuckDB binding, and the failure is memory corruption rather than an
        // exception. Four configurations were tried and all crash:
        //   * a connection per partition, concurrent      -> AccessViolationException, ~2 runs in 3
        //   * every connection opened up front and none closed mid-stage, with a keepalive connection
        //     pinning the instance refcount above zero    -> segfault
        //   * genuinely isolated FILE-BACKED databases,
        //     one per reader (distinct cache keys)        -> segfault
        //   * DuckDB.NET 1.5.5 rather than 1.5.3          -> segfault
        // The first suggested the cause was Connection.ConnectionManager tearing a shared refcounted
        // instance down under a live reader; the second and third rule that out, since it fails with no
        // teardown possible and no shared instance at all. Concurrent streaming readers are simply
        // unsafe here, and a version bump does not fix it.
        //
        // Not worked around, because this is the stage that computes the reported quantities and the
        // failure mode is wrong numbers as readily as a crash.
        //
        // dotnet/STAGE2_THROUGHPUT.md has the plan for lifting this ceiling - including the cheap
        // measurement that should come before any of it, and the bar this has to clear to ship.
        // Results are emitted in the PRODUCER's order, not in completion order, via the reorder
        // buffer below. This is a correctness requirement, not tidiness: consumers finish out of
        // order, so writing as they finish made peptides_rollup.parquet's row order vary run to run.
        // Values were stable, but ComBat's cross-feature reductions sum over rows in file order and
        // floating-point addition is not associative, so identical inputs produced outputs that
        // differed in the last bits - and through two ComBat passes only 17% of corrected_proteins
        // cells were bit-identical between two runs of the same binary on the same input. Measured
        // on a 2-plate cohort; n_workers=1 was byte-identical, which is what localized it here.
        using var inputQ = new BlockingCollection<(long Seq, PeptideBlock Block)>(dop * 4);
        using var outputQ = new BlockingCollection<(long Seq, PeptideResult? Result)>(dop * 4);

        // Sliding window over IN-FLIGHT peptides. The queue capacities alone do NOT bound the
        // reorder buffer: if the block holding the next sequence number is a straggler, the writer
        // keeps draining outputQ into `pending` (it must - the missing result may be behind others
        // in the queue), which frees consumers to keep running, which lets the producer keep
        // reading. Nothing then stops `pending` from growing to the whole cohort, and each entry is
        // a peptide x samples row. The producer takes a slot per block and the writer returns it
        // once that sequence has been emitted, so in-flight work - and therefore `pending` - is
        // capped regardless of how uneven the per-peptide cost is.
        using var window = new SemaphoreSlim(dop * 8);
        Exception? error = null;
        var filtered = 0;

        var producer = Task.Run(() =>
        {
            try
            {
                long seq = 0;
                foreach (var b in MergedParquetReader.StreamPeptideBlocks(
                    dataset, cols, samples, includeProductMz: isLibrary, includeShapeCorr: topNCorr,
                    memoryBudgetMb: memoryBudgetMb))
                {
                    // Stopping the producer drains the queue and ends the consumers, so one check here
                    // stops the whole stage rather than needing one per worker.
                    cancellationToken.ThrowIfCancellationRequested();

                    // Poll rather than block outright: if a consumer has died, no slot will ever be
                    // returned, and waiting forever would hang the stage instead of surfacing the
                    // error the consumer recorded.
                    while (!window.Wait(100, cancellationToken))
                    {
                        if (Volatile.Read(ref error) is not null)
                            return;
                    }
                    inputQ.Add((seq++, b));
                }
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
                    foreach (var (seq, block) in inputQ.GetConsumingEnumerable())
                    {
                        var r = process(block);
                        if (r is null)
                            Interlocked.Increment(ref filtered);
                        // Filtered peptides still carry their sequence number through, so the
                        // reorder buffer can advance past them instead of waiting forever.
                        outputQ.Add((seq, r));
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

        // Reorder buffer: emit strictly in producer order. Bounded by the in-flight count, which the
        // two queue capacities already cap at (dop * 4) + dop, so this cannot grow with the cohort.
        var pending = new Dictionary<long, PeptideResult?>();
        long next = 0;
        foreach (var (seq, result) in outputQ.GetConsumingEnumerable())
        {
            pending[seq] = result;
            while (pending.Remove(next, out var ready))
            {
                if (ready is not null)
                    sink.Add(ready);
                next++;
                window.Release();
            }
        }

        Task.WaitAll(producer, closer);
        if (pending.Count > 0 && error is null)
            throw new InvalidOperationException(
                $"Transition rollup lost ordering: {pending.Count} result(s) never became emittable "
                + $"(next expected sequence {next}). This would silently reorder the output.");
        nFiltered += filtered;
        if (error is not null)
            throw new InvalidOperationException("Transition rollup failed: " + error.Message, error);
    }

    private static int ResolveDop(int configured)
    {
        var cores = Math.Max(1, Environment.ProcessorCount);
        return configured <= 0 ? cores : Math.Min(configured, cores);
    }

    /// <summary>
    /// Peptides to buffer per output row group. <c>processing.peptide_batch_size</c> counts PEPTIDES,
    /// but the buffer is peptides x samples doubles, so its size also depends on how many runs were
    /// merged - and the sample count is exactly what grows when a cohort goes from two documents to a
    /// hundred. Capping the product keeps the write buffer flat instead of letting a batch size chosen
    /// for a small cohort turn into hundreds of MB on a large one. Small cohorts are unaffected: 2,000
    /// peptides x 192 samples is well under the cap, so the configured value stands.
    /// </summary>
    internal static int FlushRowsFor(int configured, int sampleCount)
    {
        const int maxCellsPerFlush = 4_000_000; // ~32 MB of doubles
        var requested = Math.Max(1, configured);
        if (sampleCount <= 0)
            return requested;
        return Math.Max(1, Math.Min(requested, maxCellsPerFlush / sampleCount));
    }

    private static PeptideResult? ProcessPeptide(
        PeptideBlock block,
        TransitionRollupConfig cfg,
        int nSamples,
        IRollupMethod method,
        bool captureResiduals)
    {
        var tidIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        var tidNames = new List<string>();
        var rowIdxs = new List<int>(block.RowCount);
        for (var i = 0; i < block.RowCount; i++)
        {
            if (cfg.ExcludePrecursor && block.IsPrecursor[i])
                continue;
            rowIdxs.Add(i);
            var tid = block.TransitionId[i];
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
            var ti = tidIndex[block.TransitionId[i]];
            var si = block.SampleIndex[i];
            var area = block.Area[i];
            if (!filled[ti, si] && !double.IsNaN(area))
            {
                matrix[ti, si] = area;
                filled[ti, si] = true;
            }
            rtBuf.Add(block.RetentionTime[i]);
        }

        var meanRt = Stats.NanMeanOrderInvariant(rtBuf.ToArray());
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
        PeptideBlock block, TransitionRollupConfig cfg, int nSamples)
    {
        var tidIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        var rowIdxs = new List<int>(block.RowCount);
        for (var i = 0; i < block.RowCount; i++)
        {
            if (cfg.ExcludePrecursor && block.IsPrecursor[i])
                continue;
            rowIdxs.Add(i);
            var tid = block.TransitionId[i];
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
            var ti = tidIndex[block.TransitionId[i]];
            var si = block.SampleIndex[i];
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
            rtBuf.Add(block.RetentionTime[i]);
        }

        var meanRt = Stats.NanMeanOrderInvariant(rtBuf.ToArray());
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
        SpectralLibrary library)
    {
        var tidIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        var tidCharge = new List<int>();
        var tidMz = new List<double>();
        var rowIdxs = new List<int>(block.RowCount);
        for (var i = 0; i < block.RowCount; i++)
        {
            if (cfg.ExcludePrecursor && block.IsPrecursor[i])
                continue;
            rowIdxs.Add(i);
            var tid = block.TransitionId[i];
            if (!tidIndex.ContainsKey(tid))
            {
                tidIndex[tid] = tidIndex.Count;
                tidCharge.Add(block.PrecursorCharge[i]);
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
            var ti = tidIndex[block.TransitionId[i]];
            var si = block.SampleIndex[i];
            var area = block.Area[i];
            if (!filled[ti, si] && !double.IsNaN(area))
            {
                matrix[ti, si] = area;
                filled[ti, si] = true;
            }
            rtBuf.Add(block.RetentionTime[i]);
        }
        var meanRt = Stats.NanMeanOrderInvariant(rtBuf.ToArray());

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
                cfg.LibraryMinFragments, cfg.LibraryMzTolerance, cfg.LibraryOutlierThreshold,
                cfg.LibraryRemoveOutliers);
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

    // ParseChargeOrDefault and TransitionId used to live here, composing the id and parsing the charge
    // from three string columns on every transition row. Both moved into the reader's SQL, where they
    // cost one expression per row instead of three string allocations - see
    // MergedParquetReader.TransitionIdSql for why the rendering has to stay byte-identical.

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
