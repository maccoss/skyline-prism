using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.IO;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.IO;

/// <summary>
/// One input skips the unsorted intermediate and sorts in a single COPY. The saving is a full write
/// plus a full read of the whole dataset - most of Stage 1's wall clock when the report lives on a
/// network share - so what has to be proven is that the shortcut changes nothing but the timing.
/// </summary>
public class MergeSingleInputFastPathTests
{
    private static string MergeDir => Fixtures.Path2("mini", "merge");

    /// <summary>
    /// The fast path against the two-stage path, on the same rows.
    /// <para>
    /// Merging [plate1] alone takes the fast path; merging [plate1, plate2] takes the two-stage
    /// path, and its plate1 rows went through the intermediate. Both carry the same synthesized
    /// Batch / Source Document / Sample ID, so the plate1 halves must agree exactly - which is a
    /// real comparison of the two implementations rather than a restatement of what one of them did.
    /// </para>
    /// </summary>
    [Fact]
    public void FastPath_ProducesTheSameRowsAsTheTwoStagePath()
    {
        var input1 = Path.Combine(MergeDir, "mini_plate1.csv");
        var input2 = Path.Combine(MergeDir, "mini_plate2.csv");
        var dir = Path.Combine(Path.GetTempPath(), "prism_fastpath_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var single = DuckDbMerge.MergeAndSort(new[] { input1 }, Path.Combine(dir, "one.parquet"));
            var both = DuckDbMerge.MergeAndSort(new[] { input1, input2 }, Path.Combine(dir, "two.parquet"));

            Assert.True(single.SingleInputFastPath);
            Assert.False(both.SingleInputFastPath);

            var singleTable = ParquetTable.Load(single.OutputPath);
            var bothTable = ParquetTable.Load(both.OutputPath);

            var columnOrder = singleTable.ColumnNames.OrderBy(x => x, StringComparer.Ordinal).ToList();
            Assert.Equal(columnOrder, bothTable.ColumnNames.OrderBy(x => x, StringComparer.Ordinal));

            var fastKeys = Sorted(Fixtures.RowKeys(singleTable, columnOrder));

            // The plate1 slice of the two-stage merge.
            var allKeys = Fixtures.RowKeys(bothTable, columnOrder);
            var batches = bothTable.GetString("Batch");
            var twoStageKeys = Sorted(allKeys.Where((_, i) => batches[i] == "mini_plate1").ToArray());

            Assert.Equal(twoStageKeys.Length, fastKeys.Length);
            for (var i = 0; i < fastKeys.Length; i++)
                if (!string.Equals(fastKeys[i], twoStageKeys[i], StringComparison.Ordinal))
                    Assert.Fail($"Row mismatch at sorted index {i}.\n"
                        + $"fast path: {fastKeys[i]}\ntwo-stage: {twoStageKeys[i]}");
        }
        finally
        {
            Cleanup(dir);
        }
    }

    /// <summary>
    /// The output still has to be SORTED by the peptide column - the whole point of the stage, and
    /// what the streamed transition rollup downstream relies on to see one peptide at a time.
    /// </summary>
    [Fact]
    public void FastPath_StillSortsByThePeptideColumn()
    {
        var input1 = Path.Combine(MergeDir, "mini_plate1.csv");
        var dir = Path.Combine(Path.GetTempPath(), "prism_fastsort_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var result = DuckDbMerge.MergeAndSort(new[] { input1 }, Path.Combine(dir, "one.parquet"));
            var table = ParquetTable.Load(result.OutputPath);
            var peptides = table.GetString(result.SortColumn);

            for (var i = 1; i < peptides.Length; i++)
                Assert.True(
                    string.CompareOrdinal(peptides[i - 1], peptides[i]) <= 0,
                    $"row {i} ('{peptides[i]}') sorts before row {i - 1} ('{peptides[i - 1]}')");
        }
        finally
        {
            Cleanup(dir);
        }
    }

    /// <summary>The intermediate must not be left behind - the fast path never writes one.</summary>
    [Fact]
    public void FastPath_LeavesNoUnsortedIntermediate()
    {
        var input1 = Path.Combine(MergeDir, "mini_plate1.csv");
        var dir = Path.Combine(Path.GetTempPath(), "prism_fastclean_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            DuckDbMerge.MergeAndSort(new[] { input1 }, Path.Combine(dir, "one.parquet"));

            Assert.False(File.Exists(Path.Combine(dir, "one.unsorted.parquet")));
            Assert.Equal(
                new[] { "one.parquet" },
                Directory.GetFiles(dir).Select(Path.GetFileName).OrderBy(x => x, StringComparer.Ordinal));
        }
        finally
        {
            Cleanup(dir);
        }
    }

    private static string[] Sorted(string[] keys)
    {
        var copy = (string[])keys.Clone();
        Array.Sort(copy, StringComparer.Ordinal);
        return copy;
    }

    private static void Cleanup(string dir)
    {
        try
        {
            if (Directory.Exists(dir))
                Directory.Delete(dir, recursive: true);
        }
        catch (IOException)
        {
            // A just-closed DuckDB connection can still hold a handle; not worth failing a test over.
        }
    }
}
