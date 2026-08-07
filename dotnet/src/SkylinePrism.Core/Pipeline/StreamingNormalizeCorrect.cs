using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.BatchCorrection;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Normalization;
using SkylinePrism.Core.Numerics;
using SkylinePrism.Core.Qc;
using System.Threading;

namespace SkylinePrism.Core.Pipeline;

/// <summary>
/// Stage 2b/2c without the feature x sample matrix: the same normalization and ComBat correction as
/// <see cref="NormalizeCorrectStage.RunInMemory"/>, computed from a column pass and a few row-group
/// passes instead of from a resident matrix.
/// <para>
/// Peak memory drops from O(features x samples), several copies over, to
/// O(rowGroupRows x samples) + O(features x batches). On a 5,000-sample cohort that is the
/// difference between ~5.6 GB per matrix copy and ~80 MB per row-group buffer.
/// </para>
/// <para>
/// What runs here and what falls back is decided by <see cref="CanHandle"/>. The fallback is always
/// correct, just memory-hungry, so an unsupported case is a performance question, never a
/// correctness one.
/// </para>
/// </summary>
internal static class StreamingNormalizeCorrect
{
    /// <summary>
    /// Whether this implementation covers the request. The excluded cases:
    /// <list type="bullet">
    /// <item><c>quantile</c> normalization - mapping a cell needs its whole column's rank
    /// distribution at apply time, so streaming it row-wise is the full matrix again.</item>
    /// <item>a non-parquet corrected output - CSV/TSV of a cohort this large is not a real
    /// scenario, and the delimited writer is not incremental.</item>
    /// <item>reference-anchored ComBat - a different estimator (NaN-aware, anchored on the
    /// reference replicates) that has not been given the streaming treatment yet.</item>
    /// </list>
    /// </summary>
    public static bool CanHandle(NormalizeCorrectRequest r) => Eligibility(r).CanStream;

    /// <summary>
    /// <see cref="CanHandle"/> plus why, so the pipeline log can say which implementation ran and -
    /// when it is the memory-hungry one - what would have to change.
    /// </summary>
    public static (bool CanStream, string Reason) Eligibility(NormalizeCorrectRequest r)
    {
        if (r.NormMethod == "quantile")
            return (false, "in-memory (quantile normalization cannot be applied cell-wise)");
        if (!r.CorrectedLinearPath.EndsWith(".parquet", StringComparison.OrdinalIgnoreCase))
            return (false, $"in-memory (output format '{Path.GetExtension(r.CorrectedLinearPath).TrimStart('.')}' "
                + "is not written incrementally)");
        if (r.CombatEnabled && r.ReferenceAnchored && r.ReferenceMask is not null && r.ReferenceMask.Any(m => m))
            return (false, "in-memory (reference-anchored ComBat is not streamed yet)");
        return (true, "streaming (bounded memory)");
    }

    /// <summary>Run the stage. Returns the number of features written.</summary>
    public static int Run(NormalizeCorrectRequest r)
    {
        var samples = r.Samples;
        var factors = NormalizationFactors.Compute(
            r.WideParquet, samples, r.NormMethod, r.RtColumn, r.RtLowessFrac, r.RtLowessGridPoints)
            ?? throw new InvalidOperationException(
                $"Normalization method '{r.NormMethod}' has no cell-wise factors; "
                + "CanHandle should have routed this to the in-memory path.");

        using var reader = ParquetColumnReader.Open(r.WideParquet);
        var rtColumn = factors.Method == "rt_lowess" ? r.RtColumn : null;

        // Pass 1: the control CVs of the RAW data, plus - when correcting - the per-feature terms
        // ComBat standardizes with. Also settles which rows survive the all-NaN filter.
        var beforeRefCvs = new List<double>();
        var beforeQcCvs = new List<double>();
        var wantRefCv = r.RefIdx.Count >= 2;
        var wantQcCv = r.QcIdx.Count >= 2;

        Batching? batching = null;
        var zeroVar = new List<bool>();
        var grandMeans = new List<double>();
        var varPooled = new List<double>();
        if (r.CombatEnabled)
            batching = Batching.Build(r.BatchLabels);

        var nKept = ForEachRow(reader, samples, factors, rtColumn, (raw, normalized) =>
        {
            if (wantRefCv && CvMetrics.TryFeatureCv(raw, r.RefIdx, out var refCv))
                beforeRefCvs.Add(refCv);
            if (wantQcCv && CvMetrics.TryFeatureCv(raw, r.QcIdx, out var qcCv))
                beforeQcCvs.Add(qcCv);

            if (batching is null)
                return;

            // ComBat holds features it cannot estimate out and passes them through untouched.
            if (!batching.IsCorrectable(normalized))
            {
                zeroVar.Add(true);
                return;
            }
            zeroVar.Add(false);
            var (grandMean, pooled) = batching.RowMeanAndPooledVar(normalized);
            grandMeans.Add(grandMean);
            varPooled.Add(pooled);
        }, r.CancellationToken);

        var beforeRefCv = CvMetrics.MedianOfCvs(beforeRefCvs);
        var beforeQcCv = CvMetrics.MedianOfCvs(beforeQcCvs);

        double[][]? gammaStar = null;
        double[][]? deltaStar = null;
        double[]? stdPooled = null;
        double[]? grandMeanArray = null;
        int[]? activeOfKept = null;
        var reverted = false;
        BatchRevertDecision decision = default;

        if (batching is not null)
        {
            // var_pooled's zero-fill is a median across ALL features, so standardization cannot
            // start until pass 1 has seen every row.
            var pooledArray = varPooled.ToArray();
            ComBat.ReplaceZeroWithMedianOfPositive(pooledArray);
            stdPooled = pooledArray.Select(Math.Sqrt).ToArray();
            grandMeanArray = grandMeans.ToArray();

            // Kept row -> index among the active (non zero-variance) features, or -1.
            activeOfKept = new int[zeroVar.Count];
            var active = 0;
            for (var i = 0; i < zeroVar.Count; i++)
                activeOfKept[i] = zeroVar[i] ? -1 : active++;

            // Pass 2: the per-(batch, feature) sufficient statistics of the standardized data.
            var stats = Accumulate(
                reader, samples, factors, rtColumn, batching, activeOfKept, grandMeanArray, stdPooled,
                active, r.CancellationToken);
            (gammaStar, deltaStar, var unestimableScales) = StreamingComBat.Estimate(stats);
            NormalizeCorrectStage.ReportComBatDiagnostics(r.Report, zeroVar.Count - active, unestimableScales);

            if (r.AutoRevert)
            {
                // Pass 3: the decision needs the corrected control CVs, and reverting means writing
                // the uncorrected values - so it has to be settled before anything is written.
                decision = Evaluate(
                    reader, samples, factors, rtColumn, batching, activeOfKept, grandMeanArray,
                    stdPooled, gammaStar, deltaStar, r.QcIdx, r.RefIdx, r.CancellationToken);
                if (decision.OverfittingWarning is not null)
                    r.Report($"  WARNING: ComBat {decision.OverfittingWarning}");
                if (decision.Revert)
                {
                    r.Report($"  ComBat REVERTED: {decision.ControlName} CV worsened "
                        + $"{decision.ControlCvBefore:F1}% -> {decision.ControlCvAfter:F1}% (>10%); "
                        + "keeping uncorrected data.");
                    reverted = true;
                }
            }
        }

        // Final pass: correct and write, one row group at a time.
        var applyCorrection = batching is not null && !reverted;
        var afterRefCvs = new List<double>();
        var afterQcCvs = new List<double>();
        var written = WriteOutputs(
            reader, r, factors, rtColumn, applyCorrection ? batching : null,
            activeOfKept, grandMeanArray, stdPooled, gammaStar, deltaStar,
            afterRefCvs, afterQcCvs, wantRefCv, wantQcCv);

        if (wantRefCv)
            r.Report($"  Reference CV (median): {beforeRefCv:F1}% -> "
                + $"{CvMetrics.MedianOfCvs(afterRefCvs):F1}% (before -> after)");
        if (wantQcCv)
            r.Report($"  QC CV (median): {beforeQcCv:F1}% -> "
                + $"{CvMetrics.MedianOfCvs(afterQcCvs):F1}% (before -> after)");

        if (written != nKept)
            throw new InvalidOperationException(
                $"Streaming Stage 2b/2c wrote {written} rows but kept {nKept} - row filtering diverged "
                + "between passes.");
        return written;
    }

    // ------------------------------------------------------------------ passes

    /// <summary>
    /// Walk every row group, normalize each surviving row in place, and hand it to
    /// <paramref name="action"/>. Rows that are NaN in every sample are dropped here, exactly as the
    /// in-memory path drops them before it does anything else. Returns the number of rows kept.
    /// </summary>
    private static int ForEachRow(
        ParquetColumnReader reader, IReadOnlyList<string> samples, NormalizationFactors factors,
        string? rtColumn, Action<double[], double[]> action,
        CancellationToken cancellationToken = default)
    {
        var nS = samples.Count;
        var raw = new double[nS];
        var normalized = new double[nS];
        var kept = 0;

        for (var rg = 0; rg < reader.RowGroupCount; rg++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            using var group = reader.OpenRowGroup(rg);
            var columns = ReadSampleColumns(group, samples);
            var rt = rtColumn is not null ? group.ReadDoubles(rtColumn) : null;

            for (var i = 0; i < group.RowCount; i++)
            {
                var any = false;
                for (var j = 0; j < nS; j++)
                {
                    raw[j] = columns[j][i];
                    any |= !double.IsNaN(raw[j]);
                }
                if (!any)
                    continue;

                kept++;
                var rowRt = rt is null ? double.NaN : rt[i];
                for (var j = 0; j < nS; j++)
                    normalized[j] = factors.Apply(j, raw[j], rowRt);
                action(raw, normalized);
            }
        }
        return kept;
    }

    /// <summary>Pass 2: the standardized data's per-(batch, feature) summary.</summary>
    private static ComBatSufficientStats Accumulate(
        ParquetColumnReader reader, IReadOnlyList<string> samples, NormalizationFactors factors,
        string? rtColumn, Batching batching, int[] activeOfKept,
        double[] grandMean, double[] stdPooled, int nActive, CancellationToken cancellationToken)
    {
        var nBatch = batching.Batches.Count;
        var gammaHat = new double[nBatch][];
        var sumSq = new double[nBatch][];
        var counts = new int[nBatch][];
        for (var b = 0; b < nBatch; b++)
        {
            gammaHat[b] = new double[nActive];
            sumSq[b] = new double[nActive];
            counts[b] = new int[nActive];
        }

        var sData = new double[samples.Count];
        var keptOrdinal = 0;
        ForEachRow(reader, samples, factors, rtColumn, (_, normalized) =>
        {
            var a = activeOfKept[keptOrdinal++];
            if (a < 0)
                return; // zero-variance feature: ComBat leaves it alone

            for (var s = 0; s < sData.Length; s++)
                sData[s] = (normalized[s] - grandMean[a]) / stdPooled[a];

            for (var b = 0; b < nBatch; b++)
            {
                var (count, mean, ss) = batching.Summarize(sData, b);
                counts[b][a] = count;
                gammaHat[b][a] = mean;
                sumSq[b][a] = ss;
            }
        }, cancellationToken);

        return new ComBatSufficientStats
        {
            Batches = batching.Batches,
            NFeatures = nActive,
            GrandMean = grandMean,
            StdPooled = stdPooled,
            Counts = counts,
            GammaHat = gammaHat,
            SumSq = sumSq,
        };
    }

    /// <summary>Pass 3 (auto-revert only): the control CVs before and after correction.</summary>
    private static BatchRevertDecision Evaluate(
        ParquetColumnReader reader, IReadOnlyList<string> samples, NormalizationFactors factors,
        string? rtColumn, Batching batching, int[] activeOfKept,
        double[] grandMean, double[] stdPooled, double[][] gammaStar, double[][] deltaStar,
        IReadOnlyList<int> qcIdx, IReadOnlyList<int> refIdx, CancellationToken cancellationToken)
    {
        var hasQc = qcIdx.Count >= 2;
        var hasRef = refIdx.Count >= 2;
        List<double> qcBefore = new(), qcAfter = new(), refBefore = new(), refAfter = new();

        var corrected = new double[samples.Count];
        var keptOrdinal = 0;
        ForEachRow(reader, samples, factors, rtColumn, (_, normalized) =>
        {
            Correct(normalized, corrected, activeOfKept[keptOrdinal++],
                batching, grandMean, stdPooled, gammaStar, deltaStar);

            if (hasQc)
            {
                if (CvMetrics.TryFeatureCv(normalized, qcIdx, out var b)) qcBefore.Add(b);
                if (CvMetrics.TryFeatureCv(corrected, qcIdx, out var a)) qcAfter.Add(a);
            }
            if (hasRef)
            {
                if (CvMetrics.TryFeatureCv(normalized, refIdx, out var b)) refBefore.Add(b);
                if (CvMetrics.TryFeatureCv(corrected, refIdx, out var a)) refAfter.Add(a);
            }
        }, cancellationToken);

        return BatchCorrectionEvaluator.Decide(
            CvMetrics.MedianOfCvs(qcBefore), CvMetrics.MedianOfCvs(qcAfter),
            CvMetrics.MedianOfCvs(refBefore), CvMetrics.MedianOfCvs(refAfter),
            hasQc, hasRef);
    }

    /// <summary>
    /// Final pass: normalize, correct, and append each row group to both outputs. This is where the
    /// in-memory path's two full transposes disappear - the row group's columns ARE the buffers.
    /// </summary>
    private static int WriteOutputs(
        ParquetColumnReader reader, NormalizeCorrectRequest r, NormalizationFactors factors,
        string? rtColumn, Batching? batching, int[]? activeOfKept,
        double[]? grandMean, double[]? stdPooled, double[][]? gammaStar, double[][]? deltaStar,
        List<double> afterRefCvs, List<double> afterQcCvs, bool wantRefCv, bool wantQcCv)
    {
        var samples = r.Samples;
        var nS = samples.Count;
        var metaTypes = r.MetaSpec.Select(m => (m.Name, Type: ElementType(m.Type))).ToList();
        var derived = r.DerivedMeta is { Count: > 0 } && r.DerivedKeyColumn is not null
                      && reader.HasColumn(r.DerivedKeyColumn)
            ? r.DerivedMeta
            : null;

        using var internalWriter = r.InternalLog2Path is null
            ? null
            : StreamingWideWriter.Create(r.InternalLog2Path, metaTypes, samples);
        using var correctedWriter = StreamingWideWriter.Create(
            r.CorrectedLinearPath,
            derived is null
                ? metaTypes
                : metaTypes.Concat(derived.Select(d => (d.Name, Type: typeof(string)))).ToList(),
            samples);

        var raw = new double[nS];
        var normalized = new double[nS];
        var corrected = new double[nS];
        var keptOrdinal = 0;
        var written = 0;

        for (var rg = 0; rg < reader.RowGroupCount; rg++)
        {
            // Stopping mid-write leaves a partial parquet, which is fine: it is an intermediate of a
            // run that did not finish, and the next run overwrites it.
            r.CancellationToken.ThrowIfCancellationRequested();
            using var group = reader.OpenRowGroup(rg);
            var columns = ReadSampleColumns(group, samples);
            var rt = rtColumn is not null ? group.ReadDoubles(rtColumn) : null;
            var metaRaw = r.MetaSpec.Select(m => group.ReadRaw(m.Name)).ToList();
            var keyRaw = derived is null ? null : group.ReadRaw(r.DerivedKeyColumn!);

            // Which rows of this group survive; sized once so the output buffers are exact.
            var keptRows = new List<int>(group.RowCount);
            for (var i = 0; i < group.RowCount; i++)
                for (var j = 0; j < nS; j++)
                    if (!double.IsNaN(columns[j][i]))
                    {
                        keptRows.Add(i);
                        break;
                    }
            if (keptRows.Count == 0)
                continue;

            var log2Out = new double[nS][];
            var linearOut = new double[nS][];
            for (var j = 0; j < nS; j++)
            {
                log2Out[j] = new double[keptRows.Count];
                linearOut[j] = new double[keptRows.Count];
            }

            for (var k = 0; k < keptRows.Count; k++)
            {
                var i = keptRows[k];
                for (var j = 0; j < nS; j++)
                    raw[j] = columns[j][i];
                var rowRt = rt is null ? double.NaN : rt[i];
                for (var j = 0; j < nS; j++)
                    normalized[j] = factors.Apply(j, raw[j], rowRt);

                double[] output;
                if (batching is null)
                {
                    output = normalized;
                }
                else
                {
                    Correct(normalized, corrected, activeOfKept![keptOrdinal],
                        batching, grandMean!, stdPooled!, gammaStar!, deltaStar!);
                    output = corrected;
                }
                keptOrdinal++;

                if (wantRefCv && CvMetrics.TryFeatureCv(output, r.RefIdx, out var refCv))
                    afterRefCvs.Add(refCv);
                if (wantQcCv && CvMetrics.TryFeatureCv(output, r.QcIdx, out var qcCv))
                    afterQcCvs.Add(qcCv);

                for (var j = 0; j < nS; j++)
                {
                    log2Out[j][k] = output[j];
                    linearOut[j][k] = Math.Pow(2.0, output[j]); // the published output is LINEAR
                }
            }

            var meta = new List<Array>(metaTypes.Count);
            for (var m = 0; m < r.MetaSpec.Count; m++)
                meta.Add(FilterMeta(metaRaw[m], keptRows, r.MetaSpec[m].Type));

            internalWriter?.WriteRowGroup(meta, log2Out);

            // Derived (parsimony) columns go on the CORRECTED output only: the internal log2 file
            // feeds ProteinRollup and QcReport, which treat any undeclared column as a sample.
            var correctedMeta = meta;
            if (derived is not null)
            {
                correctedMeta = new List<Array>(meta);
                foreach (var (_, value) in derived)
                {
                    var values = new string[keptRows.Count];
                    for (var k = 0; k < keptRows.Count; k++)
                        values[k] = value(keyRaw!.GetValue(keptRows[k])?.ToString() ?? "");
                    correctedMeta.Add(values);
                }
            }
            correctedWriter.WriteRowGroup(correctedMeta, linearOut);
            written += keptRows.Count;
        }

        return written;
    }

    // ------------------------------------------------------------------ helpers

    /// <summary>
    /// ComBat's <c>_adjust_data</c> for one feature: de-batch the standardized value, then put it
    /// back on the original scale. A zero-variance feature (<paramref name="active"/> &lt; 0) is
    /// passed through untouched, as the in-memory path's zero-variance holdout does.
    /// </summary>
    private static void Correct(
        double[] normalized, double[] corrected, int active, Batching batching,
        double[] grandMean, double[] stdPooled, double[][] gammaStar, double[][] deltaStar)
    {
        if (active < 0)
        {
            Array.Copy(normalized, corrected, normalized.Length);
            return;
        }

        for (var s = 0; s < normalized.Length; s++)
        {
            var b = batching.BatchOfSample[s];
            var sData = (normalized[s] - grandMean[active]) / stdPooled[active];
            var bayes = (sData - gammaStar[b][active]) / Math.Sqrt(deltaStar[b][active]);
            corrected[s] = bayes * stdPooled[active] + grandMean[active];
        }
    }

    private static double[][] ReadSampleColumns(
        ParquetColumnReader.RowGroup group, IReadOnlyList<string> samples)
    {
        var columns = new double[samples.Count][];
        for (var j = 0; j < samples.Count; j++)
            columns[j] = group.ReadDoubles(samples[j]);
        return columns;
    }

    private static Array FilterMeta(Array raw, List<int> keptRows, MetaType type)
    {
        switch (type)
        {
            case MetaType.Str:
            {
                var v = new string[keptRows.Count];
                for (var k = 0; k < keptRows.Count; k++)
                    v[k] = raw.GetValue(keptRows[k])?.ToString() ?? "";
                return v;
            }
            case MetaType.Long:
            {
                var v = new long[keptRows.Count];
                for (var k = 0; k < keptRows.Count; k++)
                    v[k] = ParquetTable.CoerceLong(raw.GetValue(keptRows[k]));
                return v;
            }
            case MetaType.Bool:
            {
                var v = new bool[keptRows.Count];
                for (var k = 0; k < keptRows.Count; k++)
                    v[k] = ParquetTable.CoerceBool(raw.GetValue(keptRows[k]));
                return v;
            }
            default:
            {
                var v = new double[keptRows.Count];
                for (var k = 0; k < keptRows.Count; k++)
                    v[k] = ParquetTable.CoerceDouble(raw.GetValue(keptRows[k])) ?? double.NaN;
                return v;
            }
        }
    }

    private static Type ElementType(MetaType t) => t switch
    {
        MetaType.Str => typeof(string),
        MetaType.Long => typeof(long),
        MetaType.Bool => typeof(bool),
        _ => typeof(double),
    };

    /// <summary>
    /// The sample-to-batch layout, plus the row-local reductions ComBat performs over it. The
    /// summation order here is the in-memory implementation's, so the per-feature terms it produces
    /// are bit-identical rather than merely close.
    /// </summary>
    private sealed class Batching
    {
        public required IReadOnlyList<List<int>> Batches { get; init; }
        public required int[] BatchOfSample { get; init; }
        public required int NSamples { get; init; }

        // Scratch reused across rows: this runs once per feature, and allocating here would put
        // millions of short-lived arrays through the GC on a real cohort. Single-threaded by design.
        private double[] _buffer = Array.Empty<double>();
        private double[] _residual = Array.Empty<double>();

        public static Batching Build(IReadOnlyList<string> batchLabels)
        {
            // np.unique order: sorted unique labels.
            var unique = batchLabels.Distinct().OrderBy(x => x, StringComparer.Ordinal).ToList();
            var indexOf = unique.Select((b, i) => (b, i))
                .ToDictionary(x => x.b, x => x.i, StringComparer.Ordinal);
            var batches = new List<int>[unique.Count];
            for (var i = 0; i < unique.Count; i++)
                batches[i] = new List<int>();
            var batchOfSample = new int[batchLabels.Count];
            for (var s = 0; s < batchLabels.Count; s++)
            {
                var b = indexOf[batchLabels[s]];
                batchOfSample[s] = b;
                batches[b].Add(s);
            }
            ComBat.ValidateBatchSizes(unique, batches);
            return new Batching
            {
                Batches = batches,
                BatchOfSample = batchOfSample,
                NSamples = batchLabels.Count,
                _buffer = new double[batchLabels.Count],
                _residual = new double[batchLabels.Count],
            };
        }

        /// <summary>
        /// Whether ComBat can estimate anything for this feature - <c>ComBat.IsCorrectable</c> for
        /// one row: every batch must have observed it at least once, and it must vary somewhere.
        /// </summary>
        public bool IsCorrectable(double[] row)
        {
            foreach (var batch in Batches)
            {
                var seen = false;
                foreach (var s in batch)
                {
                    if (!double.IsNaN(row[s]))
                    {
                        seen = true;
                        break;
                    }
                }
                if (!seen)
                    return false;
            }

            var n = Observed(row, null);
            return n > 0 && Stats.Var(_buffer.AsSpan(0, n), ddof: 0) != 0.0;
        }

        /// <summary>
        /// One feature's grand mean (batch-size-weighted mean of the batch means) and pooled
        /// residual variance (ddof=1) over its OBSERVED values - <c>_calculate_mean_var</c> for one
        /// row. Residuals are collected in SAMPLE order because the pairwise summation inside
        /// <c>Stats.Var</c> is order-sensitive and that is the order the in-memory path uses.
        /// </summary>
        public (double GrandMean, double VarPooled) RowMeanAndPooledVar(double[] row)
        {
            var nBatch = Batches.Count;
            var bHat = new double[nBatch];
            for (var i = 0; i < nBatch; i++)
            {
                var n = Observed(row, Batches[i]);
                bHat[i] = NumpyMath.PairwiseSum(_buffer, 0, n) / n;
            }

            var grandMean = 0.0;
            for (var i = 0; i < nBatch; i++)
                grandMean += ((double)Batches[i].Count / NSamples) * bHat[i];

            var m = 0;
            for (var s = 0; s < NSamples; s++)
            {
                var v = row[s];
                if (!double.IsNaN(v))
                    _residual[m++] = v - bHat[BatchOfSample[s]];
            }

            return (grandMean, Stats.Var(_residual.AsSpan(0, m), ComBat.VarPooledDdof));
        }

        /// <summary>
        /// How many of the batch's samples observed this feature, the mean of those values, and
        /// their sum of squared deviations about that mean - the three numbers
        /// <c>Stats.Var(observed, ddof: 1)</c> is built from, kept separately so the EB iteration
        /// can shift the sum to any other centre.
        /// </summary>
        public (int Count, double Mean, double SumSq) Summarize(double[] values, int batch)
        {
            var n = Observed(values, Batches[batch]);
            if (n == 0)
                return (0, double.NaN, double.NaN);
            var mean = NumpyMath.PairwiseSum(_buffer, 0, n) / n;
            for (var k = 0; k < n; k++)
            {
                var d = _buffer[k] - mean;
                _buffer[k] = d * d;
            }
            return (n, mean, NumpyMath.PairwiseSum(_buffer, 0, n));
        }

        /// <summary>
        /// Compact a row's observed values into the shared buffer, in sample order. Compacting and
        /// then using the ordinary pairwise reductions - rather than NaN-skipping accumulators -
        /// keeps a dense cohort bit-identical to the version that could not handle NaN.
        /// </summary>
        private int Observed(double[] row, IReadOnlyList<int>? samples)
        {
            var n = 0;
            if (samples is null)
            {
                for (var s = 0; s < NSamples; s++)
                {
                    var v = row[s];
                    if (!double.IsNaN(v))
                        _buffer[n++] = v;
                }
            }
            else
            {
                for (var k = 0; k < samples.Count; k++)
                {
                    var v = row[samples[k]];
                    if (!double.IsNaN(v))
                        _buffer[n++] = v;
                }
            }
            return n;
        }
    }
}
