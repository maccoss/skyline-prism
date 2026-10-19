using System;
using System.Globalization;
using System.IO;
using System.Linq;
using SkylinePrism.Core.IO;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.IO;

/// <summary>
/// <c>merged_data/</c> must hold exactly the rows the merge that wrote it counted. Every reader globs
/// the whole directory, so a partition the merge did not write is read as more data, with nothing to
/// say so.
///
/// <para>Found on a real 96-run cohort. Its output directory had been copied over an older copy of
/// itself, and the copy replaced files but removed none. That kept <c>_pep_bucket=2</c> from a
/// 3-partition merge beside the 2-partition merge that replaced it: 9.9M rows on top of 29.9M, every
/// one a duplicate. The partition count is chosen partly from FREE memory, so two runs of one cohort
/// can differ in it, and a copy that only adds files is all it takes. The run's own rollup was clean,
/// since it never saw the extra bucket. A QC section that globbed the directory counted 49,139
/// precursors instead of 36,937.</para>
///
/// <para>PRISM's own publish cannot produce this: it renames the previous tree aside and moves a
/// freshly built one into place (<see cref="MergePublishTests"/>). What these pin is that the damage
/// is caught when the directory is opened, whatever made it.</para>
/// </summary>
public class MergedDatasetConsistencyTests
{
    private static string MergeDir => Fixtures.Path2("mini", "merge");
    private static string Plate1 => Path.Combine(MergeDir, "mini_plate1.csv");
    private static string Plate2 => Path.Combine(MergeDir, "mini_plate2.csv");

    private static string NewDir(string label)
    {
        var dir = Path.Combine(Path.GetTempPath(), $"prism_consist_{label}_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static int[] BucketsIn(string root) => Directory
        .GetDirectories(root, MergedDataset.BucketColumn + "=*")
        .Select(d => int.Parse(Path.GetFileName(d).Split('=')[1], CultureInfo.InvariantCulture))
        .OrderBy(b => b)
        .ToArray();

    /// <summary>
    /// What a copy that merges into an existing folder does: replaces the files it brings, and leaves
    /// everything else where it was.
    /// </summary>
    private static void CopyOver(string source, string target)
    {
        foreach (var dir in Directory.GetDirectories(source, "*", SearchOption.AllDirectories))
            Directory.CreateDirectory(Path.Combine(target, Path.GetRelativePath(source, dir)));
        Directory.CreateDirectory(target);
        foreach (var file in Directory.GetFiles(source, "*", SearchOption.AllDirectories))
            File.Copy(file, Path.Combine(target, Path.GetRelativePath(source, file)), overwrite: true);
    }

    /// <summary>The sidecar the pipeline writes beside the merge, with the merge's own row count.</summary>
    private static void RecordCount(string target, DuckDbMerge.MergeResult merge) =>
        SourceFingerprint.Write(
            target + ".cache.json",
            new SourceFingerprint.CacheEntry("fingerprint", merge.TotalRows, merge.PeptideColumn));

    private static void Cleanup(string dir)
    {
        try { Directory.Delete(dir, recursive: true); } catch (IOException) { }
    }

    /// <summary>
    /// The case that was found, reproduced the way it happened: a merge with more partitions, the
    /// re-run that replaced it with fewer, and a copy of each made into one folder in turn.
    /// </summary>
    [Fact]
    public void OpenRefusesPartitionsLeftByAnEarlierMergeWithMore()
    {
        var dir = NewDir("stale");
        try
        {
            var older = DuckDbMerge.Merge(
                new[] { Plate1, Plate2 }, Path.Combine(dir, "older", "merged_data"), partitionsOverride: 16);
            var newer = DuckDbMerge.Merge(
                new[] { Plate1, Plate2 }, Path.Combine(dir, "newer", "merged_data"), partitionsOverride: 1);
            Assert.Equal(new[] { 0 }, BucketsIn(newer.OutputPath));
            // Precondition, not the property: the fixture's few peptides must hash to more than one of
            // sixteen buckets, or nothing is left over for the copy to keep.
            var leftovers = BucketsIn(older.OutputPath).Where(b => b != 0).ToArray();
            Assert.NotEmpty(leftovers);

            var target = Path.Combine(dir, "copy", "merged_data");
            CopyOver(older.OutputPath, target);
            CopyOver(newer.OutputPath, target);
            RecordCount(target, newer);

            var ex = Assert.Throws<InvalidDataException>(() => MergedDataset.Open(target));
            foreach (var b in leftovers)
                Assert.Contains($"{MergedDataset.BucketColumn}={b} ", ex.Message);
            Assert.Contains(newer.TotalRows.ToString("N0"), ex.Message);
            // Bucket 0 is the newer merge's, and is what the recorded count is made of. Naming it as
            // a leftover would send someone to delete the one partition that is right.
            Assert.Contains("left over from an earlier merge", ex.Message);
            Assert.DoesNotContain($"{MergedDataset.BucketColumn}=0 (", ex.Message);
        }
        finally
        {
            Cleanup(dir);
        }
    }

    /// <summary>
    /// The other direction: a copy that stopped part way, or a partition removed by hand. Fewer rows
    /// than were merged is as wrong as more - the QC report would describe part of the cohort as all
    /// of it.
    /// </summary>
    [Fact]
    public void OpenRefusesAMergeMissingAPartition()
    {
        var dir = NewDir("missing");
        try
        {
            var target = Path.Combine(dir, "merged_data");
            var merge = DuckDbMerge.Merge(new[] { Plate1, Plate2 }, target, partitionsOverride: 16);
            RecordCount(target, merge);
            var buckets = BucketsIn(target);
            Assert.True(buckets.Length > 1, "the fixture must spread over more than one bucket");
            Directory.Delete(
                Path.Combine(target, $"{MergedDataset.BucketColumn}={buckets[^1]}"), recursive: true);

            var ex = Assert.Throws<InvalidDataException>(() => MergedDataset.Open(target));
            Assert.Contains(merge.TotalRows.ToString("N0"), ex.Message);
        }
        finally
        {
            Cleanup(dir);
        }
    }

    /// <summary>
    /// A file in a partition that is not parquet at all reads as no rows, and is named for what it
    /// is rather than reported as a bare count that does not add up.
    /// </summary>
    [Fact]
    public void OpenNamesAPartitionFileThatIsNotParquet()
    {
        var dir = NewDir("unreadable");
        try
        {
            var target = Path.Combine(dir, "merged_data");
            var merge = DuckDbMerge.Merge(new[] { Plate1, Plate2 }, target);
            RecordCount(target, merge);
            var stray = Path.Combine(target, $"{MergedDataset.BucketColumn}=99");
            Directory.CreateDirectory(stray);
            File.WriteAllText(Path.Combine(stray, "data_0.parquet"), "not parquet");

            var ex = Assert.Throws<InvalidDataException>(() => MergedDataset.Open(target));
            Assert.Contains($"{MergedDataset.BucketColumn}=99", ex.Message);
            Assert.Contains("could not be read", ex.Message);
        }
        finally
        {
            Cleanup(dir);
        }
    }

    /// <summary>
    /// No false alarms: an intact merge opens, and so does one with no count to check against - a
    /// merge written by <c>prism merge</c>, or a sidecar from before the count was recorded. Refusing
    /// those would break every directory the guard has nothing to say about.
    /// </summary>
    [Fact]
    public void OpenAcceptsAMatchingCountAndChecksOnlyARecordedOne()
    {
        var dir = NewDir("intact");
        try
        {
            var target = Path.Combine(dir, "merged_data");
            var merge = DuckDbMerge.Merge(new[] { Plate1, Plate2 }, target, partitionsOverride: 16);

            Assert.Equal(BucketsIn(target).Length, MergedDataset.Open(target).Partitions.Count);

            RecordCount(target, merge);
            Assert.Equal(BucketsIn(target).Length, MergedDataset.Open(target).Partitions.Count);

            File.WriteAllText(target + ".cache.json", "{}");
            Assert.Equal(BucketsIn(target).Length, MergedDataset.Open(target).Partitions.Count);
        }
        finally
        {
            Cleanup(dir);
        }
    }
}
