using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Rollup;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.Rollup;

/// <summary>
/// End-to-end Layer 3 parity: the C# streaming transition-&gt;peptide rollup must reproduce
/// the Python peptides_rollup.parquet golden (e2e-sum fixture) for the sum method.
/// Validates precursor exclusion (DSTVAVVVYDITNVNSFQQTTK is precursor-only -> dropped),
/// n_transitions, mean_rt, and per-sample LOG2 abundances.
/// </summary>
public class TransitionRollupParityTests
{
    private static string E2eDir => Fixtures.Path2("mini", "e2e-sum", "output");

    [Fact]
    public void SumRollup_MatchesPythonGolden()
    {
        var merged = Path.Combine(E2eDir, "merged_data.parquet");
        var golden = ParquetTable.Load(Path.Combine(E2eDir, "peptides_rollup.parquet"));

        var cols = SkylineColumns.Detect(GetMergedColumns(merged).ToHashSet());
        // The peptide column is detected from the merged data; assert it is the expected one.
        Assert.Equal("Peptide Modified Sequence Unimod Ids", cols.Peptide);
        Assert.Equal("Sample ID", cols.Sample);

        var cfg = new TransitionRollupConfig
        {
            Method = TransitionRollupMethod.Sum,
            MinTransitions = 1,
            UseMs1 = false,
        };

        var tempOut = Path.Combine(
            Path.GetTempPath(), "prism_rollup_" + Guid.NewGuid().ToString("N"), "peptides_rollup.parquet");
        try
        {
            var result = TransitionRollup.Run(MergedDataset.Open(merged), cols, cfg, tempOut);
            Assert.Equal(1, result.NFiltered); // precursor-only peptide dropped
            var actual = ParquetTable.Load(tempOut);

            Assert.Equal(golden.RowCount, actual.RowCount);
            Assert.Equal(5, actual.RowCount);

            // Same sample columns.
            Assert.Equal(
                golden.ColumnNames.OrderBy(x => x, StringComparer.Ordinal),
                actual.ColumnNames.OrderBy(x => x, StringComparer.Ordinal));

            var pep = cols.Peptide;
            var goldenByPep = IndexByKey(golden, pep);
            var actualByPep = IndexByKey(actual, pep);
            Assert.Equal(goldenByPep.Keys.OrderBy(x => x), actualByPep.Keys.OrderBy(x => x));

            var gN = golden.GetDouble("n_transitions");
            var aN = actual.GetDouble("n_transitions");
            var gRt = golden.GetDouble("mean_rt");
            var aRt = actual.GetDouble("mean_rt");

            var sampleCols = golden.ColumnNames
                .Where(c => c != pep && c != "n_transitions" && c != "mean_rt").ToList();
            var goldenSamples = sampleCols.ToDictionary(c => c, golden.GetDouble);
            var actualSamples = sampleCols.ToDictionary(c => c, actual.GetDouble);

            foreach (var peptide in goldenByPep.Keys)
            {
                var gi = goldenByPep[peptide];
                var ai = actualByPep[peptide];

                Assert.Equal(gN[gi]!.Value, aN[ai]!.Value, 9); // n_transitions exact
                Assert.Equal(gRt[gi]!.Value, aRt[ai]!.Value, 6); // mean_rt

                foreach (var col in sampleCols)
                {
                    var g = goldenSamples[col][gi];
                    var a = actualSamples[col][ai];
                    AssertClose(g, a, peptide, col);
                }
            }
        }
        finally
        {
            var dir = Path.GetDirectoryName(tempOut);
            if (dir is not null && Directory.Exists(dir))
                Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void SerialAndParallel_ProduceIdenticalRollup()
    {
        // The bounded-parallel path must give bit-identical per-peptide results to the serial path
        // (per-peptide work is deterministic; only row order differs, and we key by peptide).
        var merged = Path.Combine(E2eDir, "merged_data.parquet");
        var cols = SkylineColumns.Detect(GetMergedColumns(merged).ToHashSet());

        string RunWith(int workers)
        {
            var cfg = new TransitionRollupConfig
            {
                Method = TransitionRollupMethod.MedianPolish, // exercise the polish + residual-free path
                MinTransitions = 1,
                UseMs1 = false,
                MaxDegreeOfParallelism = workers,
            };
            var outPath = Path.Combine(
                Path.GetTempPath(), "prism_dop_" + Guid.NewGuid().ToString("N"), "p.parquet");
            TransitionRollup.Run(MergedDataset.Open(merged), cols, cfg, outPath);
            return outPath;
        }

        var serialPath = RunWith(1);
        var parallelPath = RunWith(8);
        try
        {
            var serial = ParquetTable.Load(serialPath);
            var parallel = ParquetTable.Load(parallelPath);
            Assert.Equal(serial.RowCount, parallel.RowCount);

            var pep = cols.Peptide;
            var sByPep = IndexByKey(serial, pep);
            var pByPep = IndexByKey(parallel, pep);
            Assert.Equal(sByPep.Keys.OrderBy(x => x), pByPep.Keys.OrderBy(x => x));

            var sampleCols = serial.ColumnNames
                .Where(c => c != pep && c != "n_transitions" && c != "mean_rt").ToList();
            var sS = sampleCols.ToDictionary(c => c, serial.GetDouble);
            var pS = sampleCols.ToDictionary(c => c, parallel.GetDouble);
            foreach (var peptide in sByPep.Keys)
            {
                var si = sByPep[peptide];
                var pi = pByPep[peptide];
                foreach (var col in sampleCols)
                    Assert.True(Nullable.Equals(sS[col][si], pS[col][pi]),
                        $"serial/parallel mismatch at {peptide}/{col}");
            }
        }
        finally
        {
            foreach (var p in new[] { serialPath, parallelPath })
            {
                var dir = Path.GetDirectoryName(p);
                if (dir is not null && Directory.Exists(dir))
                    Directory.Delete(dir, recursive: true);
            }
        }
    }

    private static void AssertClose(double? expected, double? actual, string peptide, string col)
    {
        Assert.True(expected.HasValue == actual.HasValue,
            $"null mismatch at {peptide}/{col}: {expected} vs {actual}");
        if (!expected.HasValue)
            return;
        var e = expected.Value;
        var a = actual!.Value;
        var diff = Math.Abs(e - a);
        var tol = 1e-9 + 1e-9 * Math.Abs(e);
        Assert.True(diff <= tol, $"value mismatch at {peptide}/{col}: {e} vs {a} (|d|={diff})");
    }

    private static Dictionary<string, int> IndexByKey(ParquetTable t, string keyCol)
    {
        var keys = t.GetString(keyCol);
        var map = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var i = 0; i < keys.Length; i++)
            map[keys[i]!] = i;
        return map;
    }

    private static IEnumerable<string> GetMergedColumns(string mergedParquet)
        => ParquetTable.Load(mergedParquet).ColumnNames;
}
