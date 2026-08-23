using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using DuckDB.NET.Data;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Rollup;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.Rollup;

/// <summary>
/// The rollup must emit peptides in the same order every run, at any worker count.
/// <para>
/// This is a correctness requirement, not cosmetics. The parallel path used to write results in
/// COMPLETION order, so <c>peptides_rollup.parquet</c>'s row order varied run to run. Per-peptide
/// values were stable and the old serial-vs-parallel test keyed by peptide, explicitly noting that
/// "only row order differs" - judging that harmless. It is not: ComBat's cross-feature reductions
/// sum over rows in file order, floating-point addition is not associative, and through two ComBat
/// passes only <b>17%</b> of <c>corrected_proteins</c> cells came back bit-identical between two
/// runs of the same binary on the same input (2-plate cohort, 192 samples; max 105 ulp). At
/// <c>n_workers=1</c> the same runs were byte-identical, which is what localized it to the
/// completion-order write.
/// </para>
/// <para>
/// The mini fixture has too few peptides for completion order to vary reliably, so these tests
/// build a wider transition-level dataset first - with enough peptides that out-of-order completion
/// is essentially certain without the reorder buffer.
/// </para>
/// </summary>
public class RollupDeterminismTests : IDisposable
{
    private readonly string _dir = Path.Combine(
        Path.GetTempPath(), "prism_det_" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_dir))
            Directory.Delete(_dir, recursive: true);
    }

    /// <summary>
    /// Widen the mini merged fixture to <paramref name="copies"/> distinct peptides by suffixing the
    /// peptide column, so there is enough work for the workers to finish out of order.
    /// </summary>
    private string WideMergedDataset(int copies = 200)
    {
        Directory.CreateDirectory(_dir);
        var src = Path.Combine(Fixtures.Path2("mini", "merge"), "merged_data.parquet").Replace('\\', '/');
        var outPath = Path.Combine(_dir, "wide.parquet").Replace('\\', '/');

        using var conn = new DuckDBConnection("Data Source=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText =
            $"COPY (SELECT * REPLACE (\"Peptide Modified Sequence Unimod Ids\" || '_c' || c.i "
            + "AS \"Peptide Modified Sequence Unimod Ids\") "
            + $"FROM read_parquet('{src}'), (SELECT unnest(range({copies})) AS i) c) "
            + $"TO '{outPath}' (FORMAT PARQUET)";
        cmd.ExecuteNonQuery();
        return outPath;
    }

    private static string RunRollup(string merged, string outDir, int workers)
    {
        var cols = SkylineColumns.Detect(MergedColumns(merged).ToHashSet());
        var cfg = new TransitionRollupConfig
        {
            Method = TransitionRollupMethod.MedianPolish,
            MinTransitions = 1,
            UseMs1 = false,
            MaxDegreeOfParallelism = workers,
        };
        Directory.CreateDirectory(outDir);
        var outPath = Path.Combine(outDir, "p.parquet");
        TransitionRollup.Run(MergedDataset.Open(merged), cols, cfg, outPath);
        return outPath;
    }

    private static List<string> MergedColumns(string path)
    {
        using var conn = new DuckDBConnection("Data Source=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText =
            $"SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{path.Replace('\\', '/')}'))";
        using var r = cmd.ExecuteReader();
        var cols = new List<string>();
        while (r.Read())
            cols.Add(r.GetString(0));
        return cols;
    }

    /// <summary>Compare two outputs as data: same column order, same row order, same exact bits.</summary>
    private static void AssertSameTable(string pathA, string pathB)
    {
        var a = ParquetTable.Load(pathA);
        var b = ParquetTable.Load(pathB);
        Assert.Equal(a.ColumnNames, b.ColumnNames);
        Assert.Equal(a.RowCount, b.RowCount);
        var diffs = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var col in a.ColumnNames)
        {
            var ca = a.Column(col);
            var cb = b.Column(col);
            for (var i = 0; i < a.RowCount; i++)
            {
                var x = ca.GetValue(i);
                var y = cb.GetValue(i);
                var same = x is double dx && y is double dy
                    ? BitConverter.DoubleToInt64Bits(dx) == BitConverter.DoubleToInt64Bits(dy)
                    : Equals(x, y);
                if (!same) { diffs.TryGetValue(col, out var c); diffs[col] = c + 1; }
            }
        }
        Assert.True(diffs.Count == 0,
            $"{diffs.Values.Sum():n0} differing cells in {diffs.Count} column(s) of "
            + $"{a.ColumnNames.Count}: " + string.Join(", ", diffs.OrderByDescending(d => d.Value)
                .Take(6).Select(d => $"{d.Key}={d.Value}")));
    }

    /// <summary>Repeated runs at the same worker count must produce byte-identical files.</summary>
    [Fact]
    public void RepeatedParallelRuns_AreByteIdentical()
    {
        var merged = WideMergedDataset();
        var a = RunRollup(merged, Path.Combine(_dir, "a"), workers: 8);
        var b = RunRollup(merged, Path.Combine(_dir, "b"), workers: 8);
        AssertSameTable(a, b);
        Assert.Equal(File.ReadAllBytes(a), File.ReadAllBytes(b));
    }

    /// <summary>
    /// The worker count must be a performance knob only. Byte equality (not just per-peptide value
    /// equality) is the assertion, because row ORDER is what leaked into downstream reductions.
    /// </summary>
    [Theory]
    [InlineData(2)]
    [InlineData(8)]
    public void WorkerCount_DoesNotChangeOutput(int workers)
    {
        var merged = WideMergedDataset();
        var serial = RunRollup(merged, Path.Combine(_dir, "s" + workers), workers: 1);
        var parallel = RunRollup(merged, Path.Combine(_dir, "p" + workers), workers);
        Assert.Equal(File.ReadAllBytes(serial), File.ReadAllBytes(parallel));
    }

    /// <summary>
    /// The peptide sequence itself, spelled out - so a failure says "the order changed" rather than
    /// "the bytes changed", which is the part that matters and the part a future refactor is most
    /// likely to break.
    /// </summary>
    [Fact]
    public void ParallelRollup_PreservesProducerOrder()
    {
        var merged = WideMergedDataset();
        var serial = ParquetTable.Load(RunRollup(merged, Path.Combine(_dir, "os"), workers: 1));
        var parallel = ParquetTable.Load(RunRollup(merged, Path.Combine(_dir, "op"), workers: 8));

        var key = serial.ColumnNames[0];
        Assert.True(serial.RowCount > 100, $"fixture too small to detect reordering ({serial.RowCount} rows)");
        Assert.Equal(serial.GetString(key), parallel.GetString(key));
    }
}
