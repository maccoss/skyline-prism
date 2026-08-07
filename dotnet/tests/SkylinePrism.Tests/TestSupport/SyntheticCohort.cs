using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.IO;

namespace SkylinePrism.Tests.TestSupport;

/// <summary>
/// A synthetic wide LOG2 cohort shaped like <c>peptides_rollup.parquet</c>, for tests that need a
/// realistic Stage 2b/2c input rather than a committed golden: string / long / double / bool meta
/// columns (one of each MetaType) followed by one float64 column per sample.
/// <para>
/// Values carry a per-batch offset (so ComBat has something to remove), an RT-dependent per-sample
/// drift (so rt_lowess does), and tight replicates in the reference / QC columns (so the control-CV
/// metrics and the auto-revert decision are meaningful).
/// </para>
/// </summary>
public sealed record SyntheticCohort
{
    public const string KeyColumn = "peptide";
    public const string RtColumn = "mean_rt";

    public required string InputPath { get; init; }
    public required string[] Samples { get; init; }
    public required string[] BatchLabels { get; init; }
    public required bool[] ReferenceMask { get; init; }
    public required int[] RefIdx { get; init; }
    public required int[] QcIdx { get; init; }

    /// <summary>Rows written, including the deliberately all-NaN ones.</summary>
    public required int NRows { get; init; }

    /// <summary>Rows that survive the stage's all-NaN filter.</summary>
    public required int KeptRows { get; init; }

    /// <param name="missingFraction">Chance a cell is missing at random (column 0 is never missing,
    /// so a row is only ever all-NaN deliberately).</param>
    /// <param name="allNanEvery">Make every Nth feature all-NaN, to exercise the row filter. 0 = none.</param>
    /// <param name="batchSpread">Per-batch multiplier on the noise, for constructing batches whose
    /// scale ComBat will try to equalize.</param>
    /// <param name="qcOnlyInBatch">Confine the QC replicates to one batch (-1 = QC in every batch).</param>
    /// <param name="allNanRows">Explicit all-NaN feature indices, overriding
    /// <paramref name="allNanEvery"/> - use it to blank a whole row group.</param>
    /// <param name="rowGroupRows">Split the file into row groups of this size instead of writing one
    /// group. Real <c>peptides_rollup.parquet</c> files are written in 2,000-row groups, and a
    /// streaming reader that works on a single-group file can still be wrong at group boundaries.</param>
    public static SyntheticCohort Write(
        string dir,
        int nFeatures = 80,
        int nBatches = 3,
        int samplesPerBatch = 8,
        double missingFraction = 0.0,
        int allNanEvery = 0,
        int seed = 20260806,
        double[]? batchSpread = null,
        int qcOnlyInBatch = -1,
        int[]? allNanRows = null,
        int rowGroupRows = 0)
    {
        Directory.CreateDirectory(dir);
        var rng = new Rng(seed);
        var spread = batchSpread ?? Enumerable.Repeat(1.0, nBatches).ToArray();

        var samples = new List<string>();
        var batchLabels = new List<string>();
        var refMask = new List<bool>();
        var refIdx = new List<int>();
        var qcIdx = new List<int>();
        for (var b = 0; b < nBatches; b++)
        for (var k = 0; k < samplesPerBatch; k++)
        {
            var isRef = k < 2;
            var isQc = k is 2 or 3 && (qcOnlyInBatch < 0 || qcOnlyInBatch == b);
            samples.Add($"rep{k}__@__batch{b}");
            batchLabels.Add($"batch{b}");
            refMask.Add(isRef);
            if (isRef)
                refIdx.Add(samples.Count - 1);
            if (isQc)
                qcIdx.Add(samples.Count - 1);
        }
        var nSamples = samples.Count;
        var isControl = Enumerable.Range(0, nSamples).Select(s => refMask[s] || qcIdx.Contains(s)).ToArray();

        // Per-sample RT drift amplitude - what rt_lowess is there to remove.
        var drift = new double[nSamples];
        for (var s = 0; s < nSamples; s++)
            drift[s] = 0.6 * rng.NextGaussian();

        var keys = new string[nFeatures];
        var nTransitions = new long[nFeatures];
        var meanRt = new double[nFeatures];
        var lowConfidence = new bool[nFeatures];
        var columns = new double[nSamples][];
        for (var s = 0; s < nSamples; s++)
            columns[s] = new double[nFeatures];

        var blankRows = allNanRows is null ? null : new HashSet<int>(allNanRows);
        var kept = 0;
        for (var f = 0; f < nFeatures; f++)
        {
            keys[f] = $"PEP{f:D5}";
            nTransitions[f] = 3 + (f % 6);
            meanRt[f] = 5.0 + 90.0 * rng.NextDouble();
            lowConfidence[f] = f % 5 == 0;

            var allNan = blankRows is not null
                ? blankRows.Contains(f)
                : allNanEvery > 0 && f % allNanEvery == 0;
            if (!allNan)
                kept++;

            var baseAbundance = 15.0 + 6.0 * rng.NextDouble();
            var batchOffset = new double[nBatches];
            for (var b = 0; b < nBatches; b++)
                batchOffset[b] = 0.4 * rng.NextGaussian();

            for (var s = 0; s < nSamples; s++)
            {
                if (allNan)
                {
                    columns[s][f] = double.NaN;
                    continue;
                }
                var b = s / samplesPerBatch;
                var noise = (isControl[s] ? 0.08 : 0.5) * spread[b] * rng.NextGaussian();
                var value = baseAbundance + batchOffset[b] + noise
                            + drift[s] * Math.Sin(meanRt[f] / 20.0);
                columns[s][f] = s > 0 && missingFraction > 0 && rng.NextDouble() < missingFraction
                    ? double.NaN
                    : value;
            }
        }

        var path = Path.Combine(dir, "wide_log2.parquet");
        if (rowGroupRows > 0)
        {
            using var writer = StreamingWideWriter.Create(
                path,
                new (string, Type)[]
                {
                    (KeyColumn, typeof(string)), ("n_transitions", typeof(long)),
                    (RtColumn, typeof(double)), ("low_confidence", typeof(bool)),
                },
                samples);
            for (var start = 0; start < nFeatures; start += rowGroupRows)
            {
                var count = Math.Min(rowGroupRows, nFeatures - start);
                var meta = new Array[]
                {
                    keys[start..(start + count)],
                    nTransitions[start..(start + count)],
                    meanRt[start..(start + count)],
                    lowConfidence[start..(start + count)],
                };
                var slice = new double[nSamples][];
                for (var s = 0; s < nSamples; s++)
                    slice[s] = columns[s][start..(start + count)];
                writer.WriteRowGroup(meta, slice);
            }
        }
        else
        {
            ParquetWideWriter.Write(
                path,
                new[]
                {
                    ParquetWideWriter.Strings(KeyColumn, keys),
                    ParquetWideWriter.Longs("n_transitions", nTransitions),
                    ParquetWideWriter.Doubles(RtColumn, meanRt),
                    ParquetWideWriter.Bools("low_confidence", lowConfidence),
                },
                samples, columns, nFeatures);
        }

        return new SyntheticCohort
        {
            InputPath = path,
            Samples = samples.ToArray(),
            BatchLabels = batchLabels.ToArray(),
            ReferenceMask = refMask.ToArray(),
            RefIdx = refIdx.ToArray(),
            QcIdx = qcIdx.ToArray(),
            NRows = nFeatures,
            KeptRows = kept,
        };
    }

    /// <summary>
    /// The kept (non-all-NaN) rows as a [rows, samples] matrix plus their retention times - what the
    /// in-memory stage works on, so a test can compare a streamed result against it.
    /// </summary>
    public (double[,] Matrix, double[] Rt, int[] KeptIndices) LoadKeptMatrix()
    {
        var table = ParquetTable.Load(InputPath);
        var nAll = table.RowCount;
        var all = new double[nAll, Samples.Length];
        for (var j = 0; j < Samples.Length; j++)
        {
            var col = table.GetDouble(Samples[j]);
            for (var i = 0; i < nAll; i++)
                all[i, j] = col[i] ?? double.NaN;
        }

        var keep = new List<int>(nAll);
        for (var i = 0; i < nAll; i++)
        {
            var any = false;
            for (var j = 0; j < Samples.Length && !any; j++)
                any = !double.IsNaN(all[i, j]);
            if (any)
                keep.Add(i);
        }

        var rtAll = table.GetDouble(RtColumn);
        var matrix = new double[keep.Count, Samples.Length];
        var rt = new double[keep.Count];
        for (var r = 0; r < keep.Count; r++)
        {
            rt[r] = rtAll[keep[r]] ?? double.NaN;
            for (var j = 0; j < Samples.Length; j++)
                matrix[r, j] = all[keep[r], j];
        }
        return (matrix, rt, keep.ToArray());
    }

    /// <summary>
    /// Small deterministic PRNG. Not System.Random: fixtures must be identical on every runtime and
    /// platform, or a parity failure could come from the data rather than the code.
    /// </summary>
    private sealed class Rng
    {
        private ulong _state;

        public Rng(int seed) => _state = (ulong)(uint)seed * 6364136223846793005UL + 1442695040888963407UL;

        public double NextDouble()
        {
            _state = _state * 6364136223846793005UL + 1442695040888963407UL;
            return ((_state >> 11) & ((1UL << 53) - 1)) / (double)(1UL << 53);
        }

        public double NextGaussian()
        {
            var u1 = Math.Max(NextDouble(), 1e-12);
            var u2 = NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }
    }
}
