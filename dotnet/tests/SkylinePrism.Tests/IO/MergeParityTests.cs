using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.IO;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.IO;

/// <summary>
/// Cross-language parity for Layer 2: the C# DuckDbMerge must reproduce the Python
/// merge_and_sort_streaming output (fixtures/mini/merge/merged_data.parquet) row-for-row
/// as a multiset. This validates the synthesized Batch / Source Document / Sample ID
/// columns (incl. the "__@__" join), the UNION ALL, and column/type handling.
/// </summary>
public class MergeParityTests
{
    private static string MergeDir => Fixtures.Path2("mini", "merge");

    [Fact]
    public void Merge_HandlesManyReports_Streaming()
    {
        // Proxy for the ~200-PRISM-report command-line case: N distinct report files stream
        // through one UNION ALL -> COPY, giving N x the single-file row count.
        var input1 = Path.Combine(MergeDir, "mini_plate1.csv");
        Assert.True(File.Exists(input1));

        const int n = 30;
        var dir = Path.Combine(Path.GetTempPath(), "prism_merge_many_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var paths = new List<string>();
            for (var i = 0; i < n; i++)
            {
                var p = Path.Combine(dir, $"report_{i:D3}.csv"); // distinct stems -> distinct batches
                File.Copy(input1, p);
                paths.Add(p);
            }

            var singleRows = DuckDbMerge.Merge(new[] { input1 }, Path.Combine(dir, "single.parquet")).TotalRows;
            var result = DuckDbMerge.Merge(paths, Path.Combine(dir, "many.parquet"));

            Assert.Equal(singleRows * n, result.TotalRows);
        }
        finally
        {
            if (Directory.Exists(dir))
                Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void CSharpMerge_MatchesPythonGolden_RowContent()
    {
        var input1 = Path.Combine(MergeDir, "mini_plate1.csv");
        var input2 = Path.Combine(MergeDir, "mini_plate2.csv");
        var golden = Path.Combine(MergeDir, "merged_data.parquet");
        Assert.True(File.Exists(golden), $"golden fixture missing: {golden}");

        var tempOut = Path.Combine(
            Path.GetTempPath(), "prism_merge_test_" + Guid.NewGuid().ToString("N"), "merged.parquet");
        try
        {
            // Default batch names = file stems (mini_plate1, mini_plate2), matching Python.
            var result = DuckDbMerge.Merge(new[] { input1, input2 }, tempOut);

            var goldenTable = ParquetTable.Load(golden);
            var actualTable = Fixtures.LoadMerged(result.OutputPath);

            // Same columns (as a set) and same row count.
            Assert.Equal(
                goldenTable.ColumnNames.OrderBy(x => x, StringComparer.Ordinal),
                actualTable.ColumnNames.OrderBy(x => x, StringComparer.Ordinal));
            Assert.Equal(goldenTable.RowCount, actualTable.RowCount);
            Assert.Equal(4000, actualTable.RowCount);

            // Synthesized metadata columns present.
            foreach (var col in new[] { "Batch", "Source Document", "Sample ID" })
                Assert.Contains(col, actualTable.ColumnNames);

            // Order-independent multiset comparison of full-row content.
            var columnOrder = goldenTable.ColumnNames.OrderBy(x => x, StringComparer.Ordinal).ToList();
            var goldenKeys = Fixtures.RowKeys(goldenTable, columnOrder);
            var actualKeys = Fixtures.RowKeys(actualTable, columnOrder);
            Array.Sort(goldenKeys, StringComparer.Ordinal);
            Array.Sort(actualKeys, StringComparer.Ordinal);

            for (var i = 0; i < goldenKeys.Length; i++)
            {
                if (!string.Equals(goldenKeys[i], actualKeys[i], StringComparison.Ordinal))
                    Assert.Fail(
                        $"Row content mismatch at sorted index {i}.\n" +
                        $"golden: {goldenKeys[i]}\nactual: {actualKeys[i]}");
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
    public void SampleId_UsesDoubleAtJoin()
    {
        // Sample ID = "<Replicate Name>__@__<batch>", where batch defaults to the file stem.
        var golden = ParquetTable.Load(Path.Combine(MergeDir, "merged_data.parquet"));
        var sampleIds = golden.GetString("Sample ID");
        var replicates = golden.GetString("Replicate Name");
        var batches = golden.GetString("Batch");

        for (var i = 0; i < golden.RowCount; i++)
            Assert.Equal($"{replicates[i]}__@__{batches[i]}", sampleIds[i]);

        Assert.Contains(golden.GetString("Batch"), b => b == "mini_plate1");
        Assert.Contains(golden.GetString("Batch"), b => b == "mini_plate2");
    }
}
