using System;
using System.IO;
using System.Linq;
using System.Threading;
using DuckDB.NET.Data;
using SkylinePrism.Core.IO;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.IO;

/// <summary>
/// The merge is built in a staging directory and renamed into place, rather than deleting the previous
/// one and writing over the same path.
///
/// <para>Reported from a lab share: <c>Cannot open file "W:\...\merged_data\_pep_bucket=1\
/// data_0.parquet": The system cannot find the path specified</c> - ERROR_PATH_NOT_FOUND, a missing
/// DIRECTORY rather than a missing file, raised only after <c>_pep_bucket=0</c> had already been
/// written. <c>Directory.Delete</c> only STARTS a removal: on Windows a directory survives until its
/// last handle closes, and over SMB neither the server-side removal nor the client's directory cache
/// is synchronous with the return, so DuckDB recreated the tree into a path that was still going
/// away.</para>
///
/// <para>The race itself cannot be staged here - it needs a real share and unlucky timing - so what
/// these pin is the structure that removes it: the target path is only ever produced by a rename, and
/// a merge that fails leaves the previous one alone rather than a truncated stand-in for it.</para>
/// </summary>
public class MergePublishTests
{
    private static string MergeDir => Fixtures.Path2("mini", "merge");
    private static string Plate1 => Path.Combine(MergeDir, "mini_plate1.csv");
    private static string Plate2 => Path.Combine(MergeDir, "mini_plate2.csv");

    private static string NewDir(string label)
    {
        var dir = Path.Combine(Path.GetTempPath(), $"prism_pub_{label}_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static void SeedPartitions(string root, params int[] buckets)
    {
        foreach (var b in buckets)
        {
            var dir = Path.Combine(root, $"{MergedDataset.BucketColumn}={b}");
            Directory.CreateDirectory(dir);
            File.WriteAllText(Path.Combine(dir, "data_0.parquet"), $"bucket {b}");
        }
    }

    private static string[] StagingDirsIn(string dir, string stem) =>
        Directory.GetDirectories(dir, stem + DuckDbMerge.StagingSuffix + "*");

    private static int[] BucketsIn(string root) => Directory
        .GetDirectories(root, MergedDataset.BucketColumn + "=*")
        .Select(d => int.Parse(Path.GetFileName(d).Split('=')[1]))
        .OrderBy(b => b)
        .ToArray();

    /// <summary>
    /// A merge that does not finish must leave the previous one alone.
    ///
    /// <para>This is the behavioral difference the staging directory buys, and the reachable half of
    /// the reported bug. The old code deleted the target FIRST and wrote to the same path, so any
    /// failure between those two points - a cancelled run, a full disk, the share going away - left
    /// the output directory with either nothing or a partial set of buckets. Partial is the worse
    /// outcome: MergedDataset.Exists is true for a directory with one _pep_bucket=* in it and Open
    /// globs whatever it finds, so the Spectrum density pane would plot a fraction of the cohort with
    /// nothing to say it was incomplete.</para>
    ///
    /// <para>Cancellation is used to provoke it because it is deterministic - Exec checks the token
    /// before issuing the COPY - and because it is the failure a user actually causes, by pressing
    /// Stop while Stage 1 runs.</para>
    /// </summary>
    [Fact]
    public void ACancelledMergeLeavesThePreviousOneIntact()
    {
        var dir = NewDir("cancel");
        try
        {
            var target = Path.Combine(dir, "merged_data");
            var good = DuckDbMerge.Merge(new[] { Plate1, Plate2 }, target);
            var bucketsBefore = BucketsIn(target);
            Assert.NotEmpty(bucketsBefore);

            using var cancelled = new CancellationTokenSource();
            cancelled.Cancel();
            Assert.ThrowsAny<OperationCanceledException>(() => DuckDbMerge.Merge(
                new[] { Plate1 }, target, cancellationToken: cancelled.Token));

            // Untouched: same partitions, same row count, still readable.
            Assert.Equal(bucketsBefore, BucketsIn(target));
            using var conn = new DuckDBConnection("Data Source=:memory:");
            conn.Open();
            using var cmd = conn.CreateCommand();
            cmd.CommandText = "SELECT COUNT(*) FROM read_parquet('"
                + MergedDataset.Open(target).ScanTarget.Replace("'", "''")
                + "', hive_partitioning=false)";
            Assert.Equal(good.TotalRows, Convert.ToInt64(cmd.ExecuteScalar()));

            Assert.Empty(StagingDirsIn(dir, "merged_data"));
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch (IOException) { }
        }
    }

    /// <summary>
    /// The invariant the old code stated in a comment and the new code gets structurally: a rebuild
    /// must not leave the previous run's partitions behind. MergedDataset.Open globs whatever
    /// _pep_bucket=* it finds, so a shorter re-merge over a longer old one would read the leftovers as
    /// extra data - stale rows from a previous cohort, silently mixed into the table.
    /// </summary>
    [Fact]
    public void PublishLeavesOnlyTheNewPartitions()
    {
        var dir = NewDir("replace");
        try
        {
            var target = Path.Combine(dir, "merged_data");
            SeedPartitions(target, 0, 1, 2, 3);
            var staging = target + DuckDbMerge.StagingSuffix + "unittest";
            SeedPartitions(staging, 0, 1);

            DuckDbMerge.Publish(staging, target);

            Assert.Equal(new[] { 0, 1 }, BucketsIn(target));
            Assert.Equal("bucket 0", File.ReadAllText(
                Path.Combine(target, $"{MergedDataset.BucketColumn}=0", "data_0.parquet")));
            Assert.False(Directory.Exists(staging), "staging should have been renamed away");
            Assert.Empty(Directory.GetDirectories(dir, "merged_data" + DuckDbMerge.AsideSuffix + "*"));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// An output directory written by a release before the merge was partitioned holds a single
    /// merged_data FILE at the same path a directory now goes. Publish has to retire either.
    /// </summary>
    [Fact]
    public void PublishReplacesTheLegacySingleFileLayout()
    {
        var dir = NewDir("legacy");
        try
        {
            var target = Path.Combine(dir, "merged_data");
            File.WriteAllText(target, "a single-file merge from an older release");
            var staging = target + DuckDbMerge.StagingSuffix + "unittest";
            SeedPartitions(staging, 0);

            DuckDbMerge.Publish(staging, target);

            Assert.True(Directory.Exists(target));
            Assert.Equal(new[] { 0 }, BucketsIn(target));
            Assert.Empty(Directory.GetFiles(dir, "merged_data" + DuckDbMerge.AsideSuffix + "*"));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// The sweep runs before a merge on a share that may be unwell, so it must be narrow: only
    /// directories this class named itself, never the target, the staging directory, or a neighbour
    /// that merely starts with the same letters.
    /// </summary>
    [Fact]
    public void TheSweepRemovesOnlyItsOwnLeftovers()
    {
        var dir = NewDir("sweep");
        try
        {
            var target = Path.Combine(dir, "merged_data");
            Directory.CreateDirectory(target);
            Directory.CreateDirectory(target + DuckDbMerge.AsideSuffix + "aaaaaaaa");
            Directory.CreateDirectory(target + DuckDbMerge.AsideSuffix + "bbbbbbbb");
            Directory.CreateDirectory(Path.Combine(dir, "merged_data_other"));
            File.WriteAllText(Path.Combine(dir, "merged_data.cache.json"), "{}");

            // A staging directory being written RIGHT NOW by another window pointed at the same output
            // directory. Sweeping this would turn a tidy-up into a second run failing mid-write, so
            // recency is what protects it.
            var live = target + DuckDbMerge.StagingSuffix + "live0000";
            Directory.CreateDirectory(live);

            // One abandoned by a process that died without running its finally. Backdated past the
            // bound, which is the only thing that distinguishes it from the live one above.
            var orphan = target + DuckDbMerge.StagingSuffix + "orph0000";
            Directory.CreateDirectory(orphan);
            Directory.SetLastWriteTimeUtc(orphan, DateTime.UtcNow - TimeSpan.FromDays(3));

            DuckDbMerge.SweepLeftovers(target);

            Assert.True(Directory.Exists(target), "the target itself must never be swept");
            Assert.True(Directory.Exists(live), "a live concurrent merge's staging must survive");
            Assert.False(Directory.Exists(orphan), "an abandoned staging directory should be removed");
            Assert.True(Directory.Exists(Path.Combine(dir, "merged_data_other")));
            Assert.True(File.Exists(Path.Combine(dir, "merged_data.cache.json")));
            Assert.Empty(Directory.GetDirectories(dir, "merged_data" + DuckDbMerge.AsideSuffix + "*"));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// End to end, through DuckDB: re-merging into a directory that already holds a merge must leave
    /// exactly the new dataset, with none of the staging or aside scaffolding on show. A user looking
    /// at their output directory should see what they saw before.
    /// </summary>
    [Fact]
    public void ReMergingIntoTheSameDirectoryLeavesNoScaffolding()
    {
        var dir = NewDir("remerge");
        try
        {
            var target = Path.Combine(dir, "merged_data");

            var first = DuckDbMerge.Merge(new[] { Plate1, Plate2 }, target, partitionsOverride: 4);
            var firstBuckets = BucketsIn(target);
            Assert.NotEmpty(firstBuckets);

            // A bucket the merge itself cannot produce, standing in for one left by a previous run with
            // a different partition count. Seeded rather than produced because the count of REAL
            // buckets is not controllable: DuckDB materializes only partitions that receive rows, so
            // partitionsOverride: 4 gives this fixture's handful of peptides just 2 directories.
            //
            // This is the leftover that matters. MergedDataset.Open globs every _pep_bucket=* it
            // finds, so surviving it would be read back as extra rows from a previous cohort.
            var leftover = Path.Combine(target, $"{MergedDataset.BucketColumn}=99");
            Directory.CreateDirectory(leftover);
            File.WriteAllText(Path.Combine(leftover, "data_0.parquet"), "stale");

            var second = DuckDbMerge.Merge(new[] { Plate1 }, target, partitionsOverride: 2);

            Assert.DoesNotContain(99, BucketsIn(target));
            Assert.False(Directory.Exists(leftover), "a stale partition survived the rebuild");
            Assert.True(second.TotalRows < first.TotalRows, "the second merge is one plate, not two");

            // Read the PUBLISHED directory back and count it, rather than trusting the row count the
            // merge reported against its staging copy - that is the half the rename could break.
            using var conn = new DuckDBConnection("Data Source=:memory:");
            conn.Open();
            using var cmd = conn.CreateCommand();
            cmd.CommandText = "SELECT COUNT(*) FROM read_parquet('"
                + MergedDataset.Open(target).ScanTarget.Replace("'", "''")
                + "', hive_partitioning=false)";
            Assert.Equal(second.TotalRows, Convert.ToInt64(cmd.ExecuteScalar()));

            Assert.Empty(StagingDirsIn(dir, "merged_data"));
            Assert.Empty(Directory.GetDirectories(dir, "merged_data" + DuckDbMerge.AsideSuffix + "*"));
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch (IOException) { }
        }
    }
}
