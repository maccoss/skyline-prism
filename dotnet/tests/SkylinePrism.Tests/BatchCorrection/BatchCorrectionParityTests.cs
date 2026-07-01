using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.BatchCorrection;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Normalization;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.BatchCorrection;

/// <summary>
/// Layers 4+5 parity: starting from the golden peptides_rollup (LOG2), median-normalize
/// then apply ComBat and reproduce the golden peptides_log2_internal.parquet (the LOG2
/// matrix the Python pipeline saves after Stage 2c). Exercises the deterministic ComBat
/// empirical-Bayes path end-to-end on real data (5 features x 166 samples, 2 batches).
/// </summary>
public class BatchCorrectionParityTests
{
    private const string PepCol = "Peptide Modified Sequence Unimod Ids";
    private static string E2eDir => Fixtures.Path2("mini", "e2e-sum", "output");

    [Fact]
    public void MedianNorm_Then_ComBat_MatchesPeptidesLog2Internal()
    {
        var rollup = ParquetTable.Load(Path.Combine(E2eDir, "peptides_rollup.parquet"));
        var golden = ParquetTable.Load(Path.Combine(E2eDir, "peptides_log2_internal.parquet"));

        var sampleCols = Fixtures.SampleColumns(rollup, PepCol, "n_transitions", "mean_rt");
        var (matrix, rollupKeys) = Fixtures.WideMatrix(rollup, PepCol, sampleCols);

        // Batch label per sample column = the suffix after "__@__" in the Sample ID.
        var batchLabels = sampleCols.Select(BatchFromSampleId).ToList();

        var normalized = Normalizer.MedianNormalize(matrix);
        var corrected = ComBat.Run(normalized, batchLabels);

        // Golden aligned by peptide + same sample columns.
        var goldenKeys = golden.GetString(PepCol);
        var goldenIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var i = 0; i < goldenKeys.Length; i++)
            goldenIndex[goldenKeys[i]!] = i;
        var goldenCols = sampleCols.ToDictionary(c => c, golden.GetDouble);

        Assert.Equal(golden.RowCount, rollup.RowCount);

        for (var i = 0; i < rollupKeys.Length; i++)
        {
            var gi = goldenIndex[rollupKeys[i]];
            for (var j = 0; j < sampleCols.Count; j++)
            {
                var expected = goldenCols[sampleCols[j]][gi]!.Value;
                var actual = corrected[i, j];
                var diff = Math.Abs(expected - actual);
                var tol = 1e-9 + 1e-9 * Math.Abs(expected);
                Assert.True(diff <= tol,
                    $"ComBat mismatch at {rollupKeys[i]}/{sampleCols[j]}: {expected} vs {actual} (|d|={diff})");
            }
        }
    }

    private static string BatchFromSampleId(string sampleId)
    {
        const string sep = "__@__";
        var idx = sampleId.IndexOf(sep, StringComparison.Ordinal);
        return idx >= 0 ? sampleId[(idx + sep.Length)..] : sampleId;
    }
}
