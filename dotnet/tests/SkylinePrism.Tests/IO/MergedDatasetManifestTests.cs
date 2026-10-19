using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using DuckDB.NET.Data;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Pipeline;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.IO;

/// <summary>
/// merged_data/ carries a manifest of the partitions its merge wrote, and nothing reads a directory that
/// no longer matches it.
///
/// <para>Found on a real output directory. Verapamil's merged_data held three partitions where its
/// merge had written two: a run on another machine produced the folder, and it was copied to this one
/// twice by a copy that added and overwrote files but never deleted any. The third partition was the
/// first copy's, and every one of its 10,494 peptides was also in the other two, so every reader that
/// globbed merged_data/** - the density view, detection, ion accounting - was handed 9.9 million
/// duplicated rows as real ones, with nothing to say so. PRISM's own merge cannot leave a partition behind (it
/// builds in a staging directory and renames it into place; see MergePublishTests), but a folder that
/// moves between machines can, and the glob had no way to tell.</para>
///
/// <para>The tests below reproduce that with real partitions copied from the merge itself, so the
/// stray rows are genuine duplicates of the kept ones rather than placeholder files.</para>
/// </summary>
public class MergedDatasetManifestTests
{
    private static string MergeDir => Fixtures.Path2("mini", "merge");
    private static string Plate1 => Path.Combine(MergeDir, "mini_plate1.csv");
    private static string Plate2 => Path.Combine(MergeDir, "mini_plate2.csv");

    private static string NewDir(string label)
    {
        var dir = Path.Combine(Path.GetTempPath(), $"prism_manifest_{label}_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static void Cleanup(string dir)
    {
        try { Directory.Delete(dir, recursive: true); } catch (IOException) { }
    }

    private static string[] PartitionDirs(string root) => Directory
        .GetDirectories(root, MergedDataset.BucketColumn + "=*")
        .OrderBy(d => d, StringComparer.Ordinal)
        .ToArray();

    /// <summary>A merge with at least two real partitions, so one can be copied and one kept.</summary>
    private static DuckDbMerge.MergeResult MergeInto(string target)
    {
        var merge = DuckDbMerge.Merge(new[] { Plate1, Plate2 }, target, partitionsOverride: 4);
        Assert.True(PartitionDirs(target).Length >= 2, "the fixture needs two partitions to copy one");
        return merge;
    }

    /// <summary>
    /// What a non-mirroring copy leaves: a whole partition directory the current merge did not write,
    /// holding real rows that duplicate peptides the kept partitions already have.
    /// </summary>
    private static string CopyInStalePartition(string target)
    {
        var source = PartitionDirs(target)[0];
        var stale = Path.Combine(target, $"{MergedDataset.BucketColumn}=99");
        Directory.CreateDirectory(stale);
        foreach (var file in Directory.GetFiles(source, "*.parquet"))
            File.Copy(file, Path.Combine(stale, Path.GetFileName(file)));
        return stale;
    }

    private static long CountRows(string readParquetTarget)
    {
        using var conn = new DuckDBConnection("Data Source=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = "SELECT COUNT(*) FROM read_parquet('"
            + readParquetTarget.Replace('\\', '/').Replace("'", "''")
            + "', hive_partitioning=false)";
        return Convert.ToInt64(cmd.ExecuteScalar());
    }

    [Fact]
    public void AMergeWritesAManifestThatMatchesWhatIsOnDisk()
    {
        var dir = NewDir("writes");
        try
        {
            var target = Path.Combine(dir, "merged_data");
            var merge = MergeInto(target);

            Assert.True(File.Exists(Path.Combine(target, MergedDataset.ManifestFileName)));
            var manifest = MergedDataset.ReadManifest(target);
            Assert.NotNull(manifest);

            Assert.Equal(
                PartitionDirs(target).Select(Path.GetFileName),
                manifest!.Partitions.Select(p => p.Name).OrderBy(n => n, StringComparer.Ordinal));

            // Every file is listed with what is really in it: its bytes on disk, and the rows DuckDB
            // reads back from it - not just a total that could hide two compensating errors.
            foreach (var partition in manifest.Partitions)
            {
                var onDisk = Directory.GetFiles(Path.Combine(target, partition.Name), "*.parquet")
                    .Select(Path.GetFileName).OrderBy(n => n, StringComparer.Ordinal);
                Assert.Equal(onDisk, partition.Files.Select(f => f.Name).OrderBy(n => n, StringComparer.Ordinal));
                foreach (var file in partition.Files)
                {
                    var path = Path.Combine(target, partition.Name, file.Name);
                    Assert.Equal(new FileInfo(path).Length, file.Bytes);
                    Assert.Equal(CountRows(path), file.Rows);
                }
            }

            Assert.Equal(merge.TotalRows, manifest.TotalRows);
            Assert.Equal(manifest.TotalRows, manifest.Partitions.SelectMany(p => p.Files).Sum(f => f.Rows));
            Assert.Empty(MergedDataset.Inconsistencies(target));
        }
        finally
        {
            Cleanup(dir);
        }
    }

    [Fact]
    public void AConsistentDatasetOpensAndReadsEveryRow()
    {
        var dir = NewDir("opens");
        try
        {
            var target = Path.Combine(dir, "merged_data");
            var merge = MergeInto(target);

            var dataset = MergedDataset.Open(target);

            Assert.Equal(MergedDataset.ReadManifest(target)!.Partitions.Count, dataset.Partitions.Count);
            Assert.Equal(merge.TotalRows, CountRows(dataset.ScanTarget));
        }
        finally
        {
            Cleanup(dir);
        }
    }

    /// <summary>The Verapamil case: a whole partition that the merge did not write.</summary>
    [Fact]
    public void APartitionCopiedInFromAnOlderMergeIsRefusedByName()
    {
        var dir = NewDir("stalepart");
        try
        {
            var target = Path.Combine(dir, "merged_data");
            MergeInto(target);
            CopyInStalePartition(target);

            var ex = Assert.Throws<MergedDatasetMismatchException>(() => MergedDataset.Open(target));

            Assert.Contains(ex.Problems, p => p.Contains($"{MergedDataset.BucketColumn}=99", StringComparison.Ordinal));
            // The message has to say what to do, not only what is wrong: the person reading it is
            // looking at a QC pane that will not open.
            Assert.Contains("--force-reprocess", ex.Message, StringComparison.Ordinal);
            Assert.Contains("mirror", ex.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            Cleanup(dir);
        }
    }

    /// <summary>
    /// A copy can just as well leave a stray FILE inside a partition the merge did write - DuckDB names
    /// them data_0, data_1, ..., and a run that wrote fewer of them leaves the rest behind.
    /// </summary>
    [Fact]
    public void AnExtraFileInsideAListedPartitionIsRefused()
    {
        var dir = NewDir("stalefile");
        try
        {
            var target = Path.Combine(dir, "merged_data");
            MergeInto(target);
            var partition = PartitionDirs(target)[0];
            var source = Directory.GetFiles(partition, "*.parquet").OrderBy(f => f, StringComparer.Ordinal).First();
            var stray = Path.Combine(partition, "data_9.parquet");
            File.Copy(source, stray);

            var ex = Assert.Throws<MergedDatasetMismatchException>(() => MergedDataset.Open(target));
            Assert.Contains(ex.Problems, p => p.Contains("data_9.parquet", StringComparison.Ordinal));
        }
        finally
        {
            Cleanup(dir);
        }
    }

    /// <summary>
    /// A listed file whose bytes changed: overwritten by a different run's file of the same name, or
    /// cut short by an interrupted copy. Size is the check, so it costs a stat rather than a read.
    /// </summary>
    [Fact]
    public void AListedFileWhoseSizeChangedIsRefused()
    {
        var dir = NewDir("size");
        try
        {
            var target = Path.Combine(dir, "merged_data");
            MergeInto(target);
            var file = Directory.GetFiles(PartitionDirs(target)[0], "*.parquet").First();
            File.AppendAllText(file, "x");

            var ex = Assert.Throws<MergedDatasetMismatchException>(() => MergedDataset.Open(target));
            Assert.Contains(ex.Problems, p => p.Contains(Path.GetFileName(file), StringComparison.Ordinal));
        }
        finally
        {
            Cleanup(dir);
        }
    }

    /// <summary>
    /// The manifest's names are joined onto the merged_data path, so one that climbs out of it must not
    /// be followed - a hand-edited or damaged manifest is a broken one, not a different directory.
    /// </summary>
    [Fact]
    public void AManifestNamingSomethingOutsideMergedDataIsRefused()
    {
        var dir = NewDir("traversal");
        try
        {
            var target = Path.Combine(dir, "merged_data");
            MergeInto(target);
            var manifestPath = Path.Combine(target, MergedDataset.ManifestFileName);
            var first = Path.GetFileName(PartitionDirs(target)[0]);
            File.WriteAllText(manifestPath, File.ReadAllText(manifestPath).Replace($"\"{first}\"", "\"..\""));

            var ex = Assert.Throws<MergedDatasetMismatchException>(() => MergedDataset.Open(target));
            Assert.Contains(ex.Problems, p => p.Contains("not a partition or file inside merged_data", StringComparison.Ordinal));
        }
        finally
        {
            Cleanup(dir);
        }
    }

    [Fact]
    public void AMissingPartitionIsRefused()
    {
        var dir = NewDir("missing");
        try
        {
            var target = Path.Combine(dir, "merged_data");
            MergeInto(target);
            var gone = PartitionDirs(target)[0];
            Directory.Delete(gone, recursive: true);

            var ex = Assert.Throws<MergedDatasetMismatchException>(() => MergedDataset.Open(target));
            Assert.Contains(ex.Problems, p => p.Contains(Path.GetFileName(gone), StringComparison.Ordinal));
        }
        finally
        {
            Cleanup(dir);
        }
    }

    /// <summary>
    /// A merge written before manifests existed. Its only record of what it held is TotalRows in the
    /// merged_data.cache.json the pipeline writes beside it, so that is what the partitions are counted
    /// against - which is exactly the check that would have caught Verapamil (39,789,408 rows on disk
    /// against 29,910,240 recorded).
    /// </summary>
    [Fact]
    public void WithoutAManifestTheRowsAreCheckedAgainstTheMergeCache()
    {
        var dir = NewDir("legacy");
        try
        {
            var target = Path.Combine(dir, "merged_data");
            var merge = MergeInto(target);
            File.Delete(Path.Combine(target, MergedDataset.ManifestFileName));
            SourceFingerprint.Write(target + ".cache.json",
                new SourceFingerprint.CacheEntry("fingerprint", merge.TotalRows, merge.PeptideColumn));

            Assert.Empty(MergedDataset.Inconsistencies(target));
            MergedDataset.Open(target);

            CopyInStalePartition(target);
            var ex = Assert.Throws<MergedDatasetMismatchException>(() => MergedDataset.Open(target));
            Assert.Contains(merge.TotalRows.ToString("N0", CultureInfo.InvariantCulture), ex.Message, StringComparison.Ordinal);
            Assert.Contains(ex.Problems, p => p.Contains($"{MergedDataset.BucketColumn}=99", StringComparison.Ordinal));
        }
        finally
        {
            Cleanup(dir);
        }
    }

    /// <summary>
    /// No manifest and no cache - a merge written by `prism merge` from an older release, say - leaves
    /// nothing to compare against. That directory still opens: refusing it would break every pre-manifest
    /// output directory that has done nothing wrong, to guard against a case that cannot be detected.
    /// Pinned so the limitation is a decision rather than an accident.
    /// </summary>
    [Fact]
    public void WithoutAManifestOrACacheThereIsNothingToCheckAgainst()
    {
        var dir = NewDir("nobasis");
        try
        {
            var target = Path.Combine(dir, "merged_data");
            MergeInto(target);
            File.Delete(Path.Combine(target, MergedDataset.ManifestFileName));
            var partitionsBefore = PartitionDirs(target).Length;
            CopyInStalePartition(target);

            Assert.Empty(MergedDataset.Inconsistencies(target));
            Assert.Equal(partitionsBefore + 1, MergedDataset.Open(target).Partitions.Count);
        }
        finally
        {
            Cleanup(dir);
        }
    }

    /// <summary>
    /// `prism run` reusing a cached merge must not read a directory that no longer matches its manifest.
    /// Rebuilding is the fix the error message would otherwise ask for, so the pipeline does it, and
    /// says why.
    /// </summary>
    [Fact]
    public void ARerunRebuildsAMergeThatNoLongerMatchesItsManifest()
    {
        var dir = NewDir("pipeline");
        try
        {
            var inputs = new[] { Plate1, Plate2 };
            var config = PrismConfig.Load(Path.Combine(Fixtures.Path2("mini", "e2e-sum"), "config.yaml"));
            PrismPipeline.Run(inputs, dir, config);

            var merged = Path.Combine(dir, MergedDataset.DirectoryName);
            Assert.False(ExistingResults.Inspect(dir, config, inputs).MergedDataMismatch);
            var stale = CopyInStalePartition(merged);
            Assert.NotEmpty(MergedDataset.Inconsistencies(merged));

            // The pre-run check the GUI shows has to predict the same rebuild, and give the reason as
            // itself: same version, same inputs, same settings - "different settings" would send the
            // reader to a config diff with nothing in it.
            var existing = ExistingResults.Inspect(dir, config, inputs);
            Assert.True(existing.MergedDataMismatch);
            Assert.Equal(StageDependencies.Merge, existing.Recomputed[0]);
            Assert.Contains("no longer matches", existing.Warning(), StringComparison.Ordinal);

            var log = new List<string>();
            PrismPipeline.Run(inputs, dir, config, log: log.Add);

            Assert.Contains(log, line => line.Contains("does not match", StringComparison.Ordinal)
                                         && line.Contains("rebuilding", StringComparison.OrdinalIgnoreCase));
            Assert.DoesNotContain(log, line => line.Contains("Reusing cached merge", StringComparison.Ordinal));
            Assert.False(Directory.Exists(stale), "the rebuilt merge still holds the copied-in partition");
            Assert.Empty(MergedDataset.Inconsistencies(merged));
        }
        finally
        {
            Cleanup(dir);
        }
    }
}
