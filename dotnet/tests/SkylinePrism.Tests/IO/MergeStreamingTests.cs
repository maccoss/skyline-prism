using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using DuckDB.NET.Data;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Rollup;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.IO;

/// <summary>
/// The merge is one streaming COPY into peptide-hash partitions: no ORDER BY, no intermediate, whatever
/// the input count. Three things have to hold, and each is a way the change could have gone wrong:
/// the row CONTENT is unchanged (the parity contract), no peptide is split across partitions (or the
/// rollup would silently emit it twice from a fraction of its transitions), and the reader still hands
/// back one complete block per peptide.
/// </summary>
public class MergeStreamingTests
{
    private static string MergeDir => Fixtures.Path2("mini", "merge");
    private static string Plate1 => Path.Combine(MergeDir, "mini_plate1.csv");
    private static string Plate2 => Path.Combine(MergeDir, "mini_plate2.csv");

    /// <summary>
    /// One input against two, on the same rows.
    /// <para>
    /// Merging [plate1] alone and merging [plate1, plate2] both go through the same single COPY, and
    /// both synthesize Batch / Source Document / Sample ID, so the plate1 halves must agree exactly.
    /// </para>
    /// </summary>
    [Fact]
    public void OneInputProducesTheSameRowsAsPartOfMany()
    {
        var dir = NewDir("merge1");
        try
        {
            var single = DuckDbMerge.Merge(new[] { Plate1 }, Path.Combine(dir, "one"));
            var both = DuckDbMerge.Merge(new[] { Plate1, Plate2 }, Path.Combine(dir, "two"));

            var singleTable = LoadAll(single, dir, "one_flat");
            var bothTable = LoadAll(both, dir, "two_flat");

            var columnOrder = singleTable.ColumnNames.OrderBy(x => x, StringComparer.Ordinal).ToList();
            Assert.Equal(columnOrder, bothTable.ColumnNames.OrderBy(x => x, StringComparer.Ordinal));

            var oneKeys = Sorted(Fixtures.RowKeys(singleTable, columnOrder));

            // The plate1 slice of the two-input merge.
            var allKeys = Fixtures.RowKeys(bothTable, columnOrder);
            var batches = bothTable.GetString("Batch");
            var manyKeys = Sorted(allKeys.Where((_, i) => batches[i] == "mini_plate1").ToArray());

            Assert.Equal(manyKeys.Length, oneKeys.Length);
            for (var i = 0; i < oneKeys.Length; i++)
                if (!string.Equals(oneKeys[i], manyKeys[i], StringComparison.Ordinal))
                    Assert.Fail($"Row mismatch at sorted index {i}.\n"
                        + $"one input: {oneKeys[i]}\nmany inputs: {manyKeys[i]}");
        }
        finally
        {
            Cleanup(dir);
        }
    }

    /// <summary>
    /// The bucket column is a partitioning device, not data: it lives in the directory names and must
    /// not appear as a column, or every <c>SELECT *</c> consumer would see a column the single-file
    /// layout never had.
    /// </summary>
    [Fact]
    public void BucketColumnIsNotPartOfTheSchema()
    {
        var dir = NewDir("mergeschema");
        try
        {
            var merge = DuckDbMerge.Merge(new[] { Plate1 }, Path.Combine(dir, "merged"));
            var dataset = merge.Dataset();

            Assert.True(dataset.IsPartitioned);
            Assert.DoesNotContain(
                MergedDataset.BucketColumn,
                ParquetTable.ReadColumnNames(dataset.RepresentativeFile()));
            Assert.Contains(merge.PeptideColumn,
                ParquetTable.ReadColumnNames(dataset.RepresentativeFile()));
        }
        finally
        {
            Cleanup(dir);
        }
    }

    /// <summary>
    /// The property the whole partitioning scheme rests on: a peptide's rows are all in ONE partition.
    /// If this breaks, the rollup still runs and still produces peptides - it just quietly computes each
    /// split peptide twice, from half its transitions each time. Nothing downstream would notice.
    /// </summary>
    [Fact]
    public void NoPeptideIsSplitAcrossPartitions()
    {
        var dir = NewDir("mergesplit");
        try
        {
            // Forced to 4 so the fixture actually spans partitions; the real sizing would give it 1.
            var merge = DuckDbMerge.Merge(
                new[] { Plate1, Plate2 }, Path.Combine(dir, "merged"), partitionsOverride: 4);
            var dataset = merge.Dataset();
            // >= 2, not == 4: hashing need not fill every bucket, and an empty one writes no directory.
            Assert.True(dataset.Partitions.Count >= 2,
                $"fixture landed in {dataset.Partitions.Count} partition(s); nothing spans a boundary");

            var seen = new Dictionary<string, int>(StringComparer.Ordinal);
            for (var i = 0; i < dataset.Partitions.Count; i++)
                foreach (var pep in DistinctPeptides(dataset.Partitions[i], merge.PeptideColumn))
                {
                    Assert.False(seen.TryGetValue(pep, out var first),
                        $"peptide '{pep}' appears in partition {(seen.TryGetValue(pep, out var f) ? f : -1)} "
                        + $"and again in {i}");
                    seen[pep] = i;
                    _ = first;
                }

            Assert.NotEmpty(seen);
        }
        finally
        {
            Cleanup(dir);
        }
    }

    /// <summary>
    /// No intermediate, for any input count. The old two-stage form wrote a full unsorted copy of the
    /// dataset and read it back purely to feed the sort; both are gone, and a stray one would be
    /// gigabytes on a real cohort. Only the partition directory may be left behind.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void LeavesNoIntermediate(int inputCount)
    {
        var inputs = new[] { Plate1, Plate2 }.Take(inputCount).ToArray();
        var dir = NewDir("mergeclean");
        try
        {
            DuckDbMerge.Merge(inputs, Path.Combine(dir, "out"));

            Assert.Empty(Directory.GetFiles(dir)); // everything lives under out/
            Assert.Equal(
                new[] { "out" },
                Directory.GetDirectories(dir).Select(Path.GetFileName)
                    .OrderBy(x => x, StringComparer.Ordinal));
        }
        finally
        {
            Cleanup(dir);
        }
    }

    /// <summary>
    /// The invariant the dropped sort used to carry, asserted where it now lives: whatever order the
    /// merge wrote, the reader hands the rollup one block per peptide holding ALL of that peptide's
    /// rows - across partition boundaries, which is where a block could most easily be cut in half.
    /// </summary>
    [Fact]
    public void ReaderGroupsByPeptideAcrossPartitions()
    {
        var dir = NewDir("mergegrp");
        try
        {
            // Forced to 4: a block cut in half at a partition boundary is the failure this guards.
            var merge = DuckDbMerge.Merge(
                new[] { Plate1, Plate2 }, Path.Combine(dir, "merged"), partitionsOverride: 4);
            var dataset = merge.Dataset();
            // >= 2, not == 4: hashing need not fill every bucket, and an empty one writes no directory.
            Assert.True(dataset.Partitions.Count >= 2,
                $"fixture landed in {dataset.Partitions.Count} partition(s); nothing spans a boundary");
            var cols = SkylineColumns.Detect(
                ParquetTable.ReadColumnNames(dataset.RepresentativeFile()).ToHashSet());

            var blocks = MergedParquetReader.StreamPeptideBlocks(
                dataset, cols, MergedParquetReader.GetSortedSamples(dataset, cols.Sample)).ToList();

            Assert.NotEmpty(blocks);
            // One block per peptide - the failure a split peptide would produce.
            var peptides = blocks.Select(b => b.Peptide).ToList();
            Assert.Equal(peptides.Count, peptides.Distinct(StringComparer.Ordinal).Count());
            // And every row is accounted for.
            Assert.Equal(merge.TotalRows, blocks.Sum(b => (long)b.Area.Count));
        }
        finally
        {
            Cleanup(dir);
        }
    }

    /// <summary>
    /// Partitions are sized so one sorts inside the memory budget, then made no smaller - the count is
    /// a consequence, not a target. Partitions cost time and memory in the merge (256 of them ran 2.5x
    /// slower and 3x heavier than 16 on the same rows), so over-partitioning is a real regression, not
    /// merely untidy.
    /// </summary>
    [Theory]
    // 8 GB budget, an eighth of it per partition sort -> ~16.8M rows per partition.
    [InlineData(0, 8192, 1)]
    [InlineData(1_000, 8192, 1)]
    [InlineData(186_000_000, 8192, 12)]         // two documents
    [InlineData(1_860_000_000, 8192, 111)]      // twenty
    [InlineData(9_300_000_000, 8192, MergedDataset.MaxPartitions)]  // a hundred: hits the cap
    // A small budget makes partitions smaller still, down to the 12M-row floor.
    [InlineData(1_860_000_000, 2048, 155)]
    // ...and the count is capped however extreme the ratio gets.
    [InlineData(400_000_000_000, 2048, MergedDataset.MaxPartitions)]
    public void PartitionCountIsDerivedFromTheBudget(long rows, int budgetMb, int expected)
        => Assert.Equal(expected, MergedDataset.PartitionCountFor(rows, budgetMb));

    /// <summary>
    /// An output directory written by an earlier release holds a single merged_data.parquet. It must
    /// still open - the QC report and the density tab are routinely pointed at old runs.
    /// </summary>
    [Fact]
    public void OpensALegacySingleFileDataset()
    {
        var dir = NewDir("mergelegacy");
        var legacy = Path.Combine(dir, "merged_data.parquet");
        try
        {
            // A single parquet standing in for what an older release wrote.
            var merge = DuckDbMerge.Merge(new[] { Plate1 }, Path.Combine(dir, "src"));
            Flatten(merge.Dataset(), legacy);

            var dataset = MergedDataset.Open(legacy);

            Assert.False(dataset.IsPartitioned);
            Assert.Single(dataset.Partitions);
            Assert.Equal(legacy, dataset.RepresentativeFile());
            var cols = SkylineColumns.Detect(
                ParquetTable.ReadColumnNames(dataset.RepresentativeFile()).ToHashSet());
            Assert.NotEmpty(MergedParquetReader.StreamPeptideBlocks(
                dataset, cols, MergedParquetReader.GetSortedSamples(dataset, cols.Sample)).ToList());
        }
        finally
        {
            Cleanup(dir);
        }
    }

    // --- helpers -------------------------------------------------------------------------------

    private static IEnumerable<string> DistinctPeptides(string partitionGlob, string peptideColumn)
    {
        using var conn = new DuckDBConnection("Data Source=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText =
            $"SELECT DISTINCT \"{peptideColumn}\" AS p FROM {MergedParquetReader.Scan(partitionGlob)}";
        using var reader = cmd.ExecuteReader();
        var result = new List<string>();
        while (reader.Read())
            result.Add(reader.IsDBNull(0) ? "" : reader.GetString(0));
        return result;
    }

    /// <summary>Collapse a partitioned dataset into one parquet, so ParquetTable can load it.</summary>
    private static void Flatten(MergedDataset dataset, string destination)
    {
        using var conn = new DuckDBConnection("Data Source=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText =
            $"COPY (SELECT * FROM {MergedParquetReader.Scan(dataset.ScanTarget)}) "
            + $"TO '{destination.Replace("'", "''")}' (FORMAT PARQUET)";
        cmd.ExecuteNonQuery();
    }

    private static ParquetTable LoadAll(DuckDbMerge.MergeResult merge, string dir, string name)
    {
        var flat = Path.Combine(dir, name + ".parquet");
        Flatten(merge.Dataset(), flat);
        return ParquetTable.Load(flat);
    }

    private static string NewDir(string tag)
    {
        var dir = Path.Combine(Path.GetTempPath(), $"prism_{tag}_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
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
