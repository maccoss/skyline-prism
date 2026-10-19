using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace SkylinePrism.Core.IO;

/// <summary>
/// The merged transition-level data on disk, and the two ways the pipeline reads it: as one dataset
/// (for the order-independent scans - parsimony, batch map, density) or one partition at a time (for
/// the transition rollup).
/// <para>
/// <b>Why it is partitioned.</b> The rollup has to see all of a peptide's transitions together. Doing
/// that with a global <c>ORDER BY</c> means a blocking sort over the entire cohort, whose spill grows
/// linearly with it: measured at ~64 bytes per row, a 100-document cohort (~9 billion transition rows)
/// would spill ~600 GB and re-read it several times during the external merge. Hashing rows into
/// buckets on the peptide column instead puts every row of a given peptide in exactly one partition,
/// so the rollup can sort and stream one partition at a time and never holds more than one partition's
/// worth. <b>The rollup's</b> peak is then a function of the partition SIZE, not of how many documents
/// were merged - which is the whole point, and the property to preserve.
/// </para>
/// <para>
/// Note the scope of that claim: it makes Stage 1 and Stage 2 flat in cohort size, but the pipeline as
/// a whole is not. The later stages hold a feature x sample matrix, and while the feature count is
/// fixed by the target list, the sample count does grow with every document added. Those matrices are
/// read a column at a time (<see cref="ParquetColumnReader"/>) so they cost 8 bytes per cell rather
/// than the 24 a whole-table load costs, but they remain the term that grows.
/// </para>
/// <para>
/// The bucket column lives only in the directory names (parquet <c>PARTITION_BY</c> does not write it
/// into the files), and every read passes <c>hive_partitioning=false</c>, so the columns PRISM sees
/// are exactly the columns it saw when this was a single file.
/// </para>
/// </summary>
public sealed class MergedDataset
{
    /// <summary>The hive partition key. Underscore-prefixed so it cannot collide with a report column.</summary>
    public const string BucketColumn = "_pep_bucket";

    /// <summary>
    /// Bytes the rollup's sort carries per transition row - peptide, fragment ion, sample id, two
    /// charges and three doubles - measured at ~64 on a real cohort. Used to size partitions against
    /// the memory budget.
    /// </summary>
    private const long SortBytesPerRow = 64;

    /// <summary>
    /// Fraction of the budget a partition's sort may claim.
    /// <para>
    /// An eighth, which is far more conservative than the arithmetic alone suggests, because the
    /// arithmetic was wrong in the optimistic direction when measured. Sizing partitions at half the
    /// budget (~66M rows) made the ROLLUP 2.4x slower than ~12M-row partitions - 62 min against 26 on
    /// a 20-document cohort - swamping the 25 min it saved in the merge. The nominal payload is only
    /// part of what a DuckDB sort wants, and once it spills, the reader streaming out of it slows down
    /// far more than the spill volume implies. Small partitions stream; large ones grind.
    /// </para>
    /// </summary>
    private const double BudgetFractionForSort = 0.125;

    /// <summary>Floor and ceiling on partition size, whatever the budget works out to.</summary>
    private const long MinRowsPerPartition = 12_000_000;
    private const long MaxRowsPerPartition = 100_000_000;

    /// <summary>
    /// Ceiling on the bucket count. Every partition is held open by the writer for the whole pass (the
    /// rows are hash-assigned, so all of them are live at once), and each open partition carries a
    /// buffer - so this bounds the writer's file handles and, with
    /// <c>DuckDbTuning.ApplyPartitionedWrite</c>, its memory. Past this the rollup's per-partition sort
    /// simply spills a little, which is bounded and cheap; an unbounded writer is neither.
    /// </summary>
    public const int MaxPartitions = 256;

    /// <summary>The name this release writes: a directory of <c>_pep_bucket=N</c> partitions.</summary>
    public const string DirectoryName = "merged_data";

    /// <summary>The single file earlier releases wrote instead.</summary>
    public const string LegacyFileName = "merged_data.parquet";

    /// <summary>
    /// The merged data inside an output directory, whichever layout is there, or null when neither
    /// is.
    /// </summary>
    /// <remarks>
    /// Every caller must go through this rather than composing the path itself. Composing
    /// <c>Path.Combine(dir, "merged_data")</c> silently misses the legacy file - it is not a
    /// directory and does not carry that name - so a feature written that way reports "no merged
    /// data" on a directory that plainly has some. All three QC features did exactly that until a
    /// real 82-replicate directory from an older release was pointed at them.
    /// </remarks>
    public static string? Locate(string? outputDir)
    {
        if (string.IsNullOrWhiteSpace(outputDir))
            return null;
        foreach (var name in new[] { DirectoryName, LegacyFileName })
        {
            var candidate = Path.Combine(outputDir, name);
            if (Exists(candidate))
                return candidate;
        }
        return null;
    }

    /// <summary>Directory holding the <c>_pep_bucket=N</c> subdirectories, or a legacy single file.</summary>
    public string Root { get; }

    /// <summary>False for a legacy single-file <c>merged_data.parquet</c> from an older release.</summary>
    public bool IsPartitioned { get; }

    /// <summary>
    /// One entry per partition, each a <c>read_parquet</c> target. Iterate these to process the cohort
    /// in bounded pieces; a peptide never spans two of them.
    /// </summary>
    public IReadOnlyList<string> Partitions { get; }

    /// <summary>A <c>read_parquet</c> target covering every row, for scans that do not need grouping.</summary>
    public string ScanTarget { get; }

    private MergedDataset(string root, bool partitioned, IReadOnlyList<string> partitions, string scanTarget)
    {
        Root = root;
        IsPartitioned = partitioned;
        Partitions = partitions;
        ScanTarget = scanTarget;
    }

    /// <summary>
    /// Describe what is actually on disk at <paramref name="root"/>. Accepts both the partitioned
    /// directory this release writes and the single <c>merged_data.parquet</c> written by earlier ones,
    /// so an existing output directory still opens (in the QC report, the density tab, and the merge
    /// cache) without being re-merged.
    /// </summary>
    public static MergedDataset Open(string root)
    {
        if (File.Exists(root))
            return new MergedDataset(root, false, new[] { root }, root);

        if (!Directory.Exists(root))
            throw new DirectoryNotFoundException($"No merged data at {root}.");

        // Refused rather than read: a stray partition's rows would otherwise be counted as real ones by
        // every reader, with nothing downstream to show it. See ManifestFileName.
        var problems = Inconsistencies(root);
        if (problems.Count > 0)
            throw new MergedDatasetMismatchException(root, problems);

        // Checked above to be exactly the manifest's partitions when there is one, so listing them from
        // disk reads the same set without parsing the manifest a second time.
        var partitions = Directory
            .GetDirectories(root, BucketColumn + "=*")
            .OrderBy(d => d, StringComparer.Ordinal)
            .Select(d => Glob(d))
            .ToList();

        if (partitions.Count == 0)
            throw new InvalidOperationException(
                $"{root} has no {BucketColumn}=* partitions; it is not a merged PRISM dataset.");

        return new MergedDataset(root, true, partitions, Glob(root, recursive: true));
    }

    /// <summary>True when a merged dataset (either layout) already exists at this path.</summary>
    public static bool Exists(string root) =>
        File.Exists(root)
        || (Directory.Exists(root) && Directory.GetDirectories(root, BucketColumn + "=*").Length > 0);

    /// <summary>Delete whichever layout is there, so a rebuild never mixes old and new partitions.</summary>
    public static void Delete(string root)
    {
        if (File.Exists(root))
            File.Delete(root);
        else if (Directory.Exists(root))
            Directory.Delete(root, recursive: true);
    }

    /// <summary>
    /// How many buckets to hash into, given the cohort size and the memory budget the rollup will have.
    /// <para>
    /// This trades the two stages against each other and the trade was measured, in both directions,
    /// because it is not obvious. Partitions cost the MERGE: on 186M rows with buffered rows held
    /// constant, 16 partitions took 0.84 min / 2.4 GB, 64 -> 1.08 / 3.3, 155 -> 1.74 / 5.5,
    /// 256 -> 2.07 / 7.8, since every open partition writer carries its own row-group buffer. That
    /// argues for few, large partitions.
    /// </para>
    /// <para>
    /// Trying it settled the argument the other way. On a 20-document cohort, ~66M-row partitions
    /// (28 of them) cut the merge from 39.4 min to 14.7 - and pushed the ROLLUP from 26.0 min to 62.2,
    /// for 82.9 min end to end against 71.9. A big partition's sort stops fitting, and the reader
    /// streaming out of a spilling sort degrades much faster than the extra spill suggests. So
    /// partitions are sized small - an eighth of the budget, ~16.8M rows at 8 GB - and the merge pays
    /// for it: ~12 partitions for two documents, ~111 for twenty, the 256 cap beyond ~4.3 billion rows.
    /// </para>
    /// <para>
    /// Both measurements come from ONE cohort on ONE machine, and the optimum is somewhere between the
    /// two points rather than at either. Re-measure end to end - not just the stage being tuned -
    /// before moving this.
    /// </para>
    /// </summary>
    public static int PartitionCountFor(long estimatedRows, int memoryBudgetMb)
    {
        if (estimatedRows <= 0)
            return 1;

        var budgetBytes = Math.Max(1L, memoryBudgetMb) * 1024L * 1024L;
        var rowsPerPartition = Math.Clamp(
            (long)(budgetBytes * BudgetFractionForSort / SortBytesPerRow),
            MinRowsPerPartition,
            MaxRowsPerPartition);

        var n = (estimatedRows + rowsPerPartition - 1) / rowsPerPartition;
        return (int)Math.Clamp(n, 1, MaxPartitions);
    }

    /// <summary>
    /// One real parquet file, for the schema-only reads that go through Parquet.Net rather than DuckDB.
    /// Every partition carries the full schema, so any of them will do.
    /// </summary>
    public string RepresentativeFile()
    {
        if (!IsPartitioned)
            return Root;
        var file = Directory
            .EnumerateFiles(Root, "*.parquet", SearchOption.AllDirectories)
            .OrderBy(f => f, StringComparer.Ordinal)
            .FirstOrDefault();
        return file ?? throw new InvalidOperationException($"{Root} contains no parquet files.");
    }

    private static string Glob(string dir, bool recursive = false) =>
        Path.Combine(dir, recursive ? "**" : "", "*.parquet").Replace('\\', '/');

    // ---- The manifest ---------------------------------------------------------------------------

    /// <summary>
    /// The record a merge leaves inside merged_data/ of every partition and file it wrote. JSON, not
    /// parquet, so no <c>**/*.parquet</c> read can pick it up as data.
    /// </summary>
    /// <remarks>
    /// <para><b>Why it exists.</b> A merged_data directory is read by globbing whatever
    /// <c>_pep_bucket=*</c> partitions it holds, which is right only while the directory holds exactly
    /// what one merge wrote. PRISM's own merge guarantees that (it builds in a staging directory and
    /// renames it into place), but an output folder copied between machines does not: a copy that adds
    /// and overwrites files but never deletes them leaves behind any partition the newer run no longer
    /// wrote. Verapamil's merged_data was found holding three partitions where its merge had written
    /// two, and every peptide in the third was also in the other two - 9.9 million duplicated rows that
    /// every reader of the glob took for real ones. What each made of them varies (ion accounting
    /// merges claims before counting and may have absorbed them; that was not measured), but none of
    /// them could tell a stray partition from a real one.</para>
    ///
    /// <para>The manifest is that missing record. It travels with the folder, so a copy that brings a
    /// newer merge brings its manifest too, and whatever the copy failed to remove shows up as not
    /// listed.</para>
    /// </remarks>
    public const string ManifestFileName = "_manifest.json";

    /// <summary>One parquet file of a partition: its name, the rows in its footer, its size on disk.</summary>
    public sealed record ManifestFile(string Name, long Rows, long Bytes);

    /// <summary>One <c>_pep_bucket=N</c> directory and the files the merge wrote into it.</summary>
    public sealed record ManifestPartition(string Name, IReadOnlyList<ManifestFile> Files);

    /// <summary>Everything one merge wrote.</summary>
    public sealed record Manifest(int Version, long TotalRows, IReadOnlyList<ManifestPartition> Partitions);

    private const int ManifestVersion = 1;

    private static readonly JsonSerializerOptions ManifestJson = new() { WriteIndented = true };

    /// <summary>
    /// Record what is in <paramref name="root"/> now. Called by the merge on its staging directory,
    /// after the COPY has closed its files and before the directory is renamed into place - so the
    /// manifest describes exactly that merge, and arrives with it.
    /// </summary>
    internal static Manifest WriteManifest(string root)
    {
        var partitions = new List<ManifestPartition>();
        foreach (var dir in Directory.GetDirectories(root, BucketColumn + "=*").OrderBy(d => d, StringComparer.Ordinal))
        {
            var files = Directory.GetFiles(dir, "*.parquet")
                .OrderBy(f => f, StringComparer.Ordinal)
                .Select(f => new ManifestFile(Path.GetFileName(f), ParquetColumnReader.RowCountOf(f), new FileInfo(f).Length))
                .ToList();
            partitions.Add(new ManifestPartition(Path.GetFileName(dir), files));
        }
        var manifest = new Manifest(ManifestVersion, partitions.SelectMany(p => p.Files).Sum(f => f.Rows), partitions);
        File.WriteAllText(Path.Combine(root, ManifestFileName), JsonSerializer.Serialize(manifest, ManifestJson));
        return manifest;
    }

    /// <summary>
    /// The manifest in <paramref name="root"/>, or null when there is none (a merge written before
    /// manifests existed, or the legacy single file).
    /// </summary>
    /// <exception cref="InvalidDataException">The manifest is there but cannot be read.</exception>
    public static Manifest? ReadManifest(string root)
    {
        var path = Path.Combine(root, ManifestFileName);
        if (!File.Exists(path))
            return null;
        try
        {
            // Shared for write and delete, not File.ReadAllText: the GUI reads this while a run in
            // another window may be publishing a re-merge, and Publish renames the whole directory -
            // refused while ANY handle in it denies sharing (CLAUDE.md, "A reader must never take a
            // file hostage").
            using var fs = ParquetColumnIo.OpenRead(path);
            using var sr = new StreamReader(fs);
            var m = JsonSerializer.Deserialize<Manifest>(sr.ReadToEnd());
            if (m?.Partitions is null)
                throw new InvalidDataException($"{ManifestFileName} lists no partitions.");
            // Names are joined onto the merged_data path, so anything but a bare partition or file
            // name - a separator, "..", an absolute path - would point the check outside it.
            foreach (var p in m.Partitions)
                if (!IsBareName(p.Name) || !p.Name.StartsWith(BucketColumn + "=", StringComparison.Ordinal)
                    || p.Files is null || p.Files.Any(f => !IsBareName(f.Name)))
                    throw new InvalidDataException(
                        $"{ManifestFileName} lists an entry that is not a partition or file inside merged_data: {p.Name}");
            return m;
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException($"{ManifestFileName} cannot be read: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Everything about <paramref name="root"/> that does not match what its merge wrote, as sentences
    /// a person can act on; empty when it matches, or when there is nothing to compare against.
    /// </summary>
    /// <remarks>
    /// <para>With a manifest, the directory must hold exactly the listed partitions and files, each at
    /// its recorded size. Size rather than rows because it costs a stat rather than a read - and a file
    /// overwritten by another run's, or cut short by an interrupted copy, will not keep its size.</para>
    ///
    /// <para>Without one - a merge written before this release - the only record is the row count the
    /// pipeline stores beside it in <c>merged_data.cache.json</c>, so the partitions' footers are
    /// summed against that. It would have caught Verapamil: 39,789,408 rows on disk against 29,910,240
    /// recorded. With neither there is nothing to check, and the directory is read as it is; refusing
    /// it would break every older output directory that has done nothing wrong.</para>
    /// </remarks>
    public static IReadOnlyList<string> Inconsistencies(string root)
    {
        if (!Directory.Exists(root))
            return Array.Empty<string>(); // absent, or the legacy single file: one file cannot hold a stray partition

        Manifest? manifest;
        try
        {
            manifest = ReadManifest(root);
        }
        catch (Exception ex) when (ex is InvalidDataException or IOException or UnauthorizedAccessException)
        {
            return new[] { ex.Message };
        }
        return manifest is null ? AgainstCache(root) : AgainstManifest(root, manifest);
    }

    private static List<string> AgainstManifest(string root, Manifest manifest)
    {
        var problems = new List<string>();
        var listed = manifest.Partitions.ToDictionary(p => p.Name, StringComparer.Ordinal);

        foreach (var dir in Directory.GetDirectories(root, BucketColumn + "=*").OrderBy(d => d, StringComparer.Ordinal))
        {
            var name = Path.GetFileName(dir);
            if (listed.ContainsKey(name))
                continue;
            var files = Directory.GetFiles(dir, "*.parquet", SearchOption.AllDirectories);
            problems.Add(string.Create(CultureInfo.InvariantCulture,
                $"{name}: not written by this merge ({files.Length} file(s), {files.Sum(f => new FileInfo(f).Length):N0} bytes, last written {LastWritten(files)})"));
        }

        foreach (var partition in manifest.Partitions)
        {
            var dir = Path.Combine(root, partition.Name);
            if (!Directory.Exists(dir))
            {
                problems.Add($"{partition.Name}: written by this merge but missing");
                continue;
            }
            var expected = partition.Files.ToDictionary(f => f.Name, StringComparer.Ordinal);
            foreach (var file in Directory.GetFiles(dir, "*.parquet", SearchOption.AllDirectories).OrderBy(f => f, StringComparer.Ordinal))
            {
                var rel = Path.GetRelativePath(dir, file).Replace('\\', '/');
                if (!expected.TryGetValue(rel, out var want))
                {
                    problems.Add(string.Create(CultureInfo.InvariantCulture,
                        $"{partition.Name}/{rel}: not written by this merge (last written {LastWritten(new[] { file })})"));
                    continue;
                }
                var bytes = new FileInfo(file).Length;
                if (bytes != want.Bytes)
                    problems.Add(string.Create(CultureInfo.InvariantCulture,
                        $"{partition.Name}/{rel}: {bytes:N0} bytes, but the merge wrote {want.Bytes:N0}"));
            }
            foreach (var want in partition.Files)
                if (!File.Exists(Path.Combine(dir, want.Name)))
                    problems.Add($"{partition.Name}/{want.Name}: written by this merge but missing");
        }

        foreach (var file in OutsidePartitions(root))
            problems.Add($"{Path.GetRelativePath(root, file).Replace('\\', '/')}: not written by this merge");
        return problems;
    }

    /// <summary>
    /// Parquet files the recursive <see cref="ScanTarget"/> glob would read that sit in no partition
    /// directory - at the top of merged_data, or in some other folder inside it.
    /// </summary>
    private static IEnumerable<string> OutsidePartitions(string root) =>
        Directory.GetFiles(root, "*.parquet", SearchOption.TopDirectoryOnly)
            .Concat(Directory.GetDirectories(root)
                .Where(d => !Path.GetFileName(d).StartsWith(BucketColumn + "=", StringComparison.Ordinal))
                .SelectMany(d => Directory.GetFiles(d, "*.parquet", SearchOption.AllDirectories)))
            .OrderBy(f => f, StringComparer.Ordinal);

    private static List<string> AgainstCache(string root)
    {
        var problems = new List<string>();
        var cache = SourceFingerprint.TryRead(CachePathFor(root));
        if (cache is null || cache.TotalRows <= 0)
            return problems;

        // Everything the recursive glob reads: each partition, plus anything outside them as one group.
        var groups = Directory.GetDirectories(root, BucketColumn + "=*")
            .OrderBy(d => d, StringComparer.Ordinal)
            .Select(d => (Name: Path.GetFileName(d), Files: Directory.GetFiles(d, "*.parquet", SearchOption.AllDirectories)))
            .ToList();
        var outside = OutsidePartitions(root).ToArray();
        if (outside.Length > 0)
            groups.Add(("outside any partition", outside));

        var perPartition = new List<(string Name, long Rows, string[] Files)>();
        long total = 0;
        foreach (var (name, files) in groups)
        {
            long rows = 0;
            foreach (var f in files)
            {
                try
                {
                    rows += ParquetColumnReader.RowCountOf(f);
                }
                catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException
                                               or NotSupportedException or Parquet.ParquetException)
                {
                    problems.Add($"{name}/{Path.GetFileName(f)}: cannot be read as parquet ({ex.Message})");
                }
            }
            perPartition.Add((name, rows, files));
            total += rows;
        }

        if (problems.Count == 0 && total == cache.TotalRows)
            return problems;

        problems.Insert(0, string.Create(CultureInfo.InvariantCulture,
            $"the partitions hold {total:N0} rows, but {Path.GetFileName(CachePathFor(root))} records {cache.TotalRows:N0} for this merge"));
        // Every partition with its rows and date: a stray one usually stands out as the odd date.
        foreach (var (name, rows, files) in perPartition)
            problems.Add(string.Create(CultureInfo.InvariantCulture,
                $"{name}: {rows:N0} rows in {files.Length} file(s), last written {LastWritten(files)}"));
        return problems;
    }

    private static bool IsBareName(string? name) =>
        !string.IsNullOrEmpty(name) && name != "." && name != ".."
        && name.IndexOfAny(new[] { '/', '\\', ':' }) < 0 && name.IndexOfAny(Path.GetInvalidFileNameChars()) < 0;

    /// <summary>The <c>merged_data.cache.json</c> the pipeline writes beside a merge.</summary>
    internal static string CachePathFor(string root) =>
        Path.TrimEndingDirectorySeparator(root) + ".cache.json";

    private static string LastWritten(IEnumerable<string> files)
    {
        var latest = files.Select(File.GetLastWriteTime).DefaultIfEmpty().Max();
        return latest == default ? "never" : latest.ToString("yyyy-MM-dd HH:mm", CultureInfo.InvariantCulture);
    }
}

/// <summary>
/// A merged_data directory that no longer holds what its merge wrote. Thrown by
/// <see cref="MergedDataset.Open"/> rather than reading it, because every reader would otherwise
/// count whatever is extra as real rows - with nothing in any output to say so.
/// </summary>
public sealed class MergedDatasetMismatchException : InvalidOperationException
{
    public MergedDatasetMismatchException(string root, IReadOnlyList<string> problems)
        : base(Describe(root, problems))
    {
        Root = root;
        Problems = problems;
    }

    /// <summary>The merged_data directory that was refused.</summary>
    public string Root { get; }

    /// <summary>What does not match, one sentence each.</summary>
    public IReadOnlyList<string> Problems { get; }

    private static string Describe(string root, IReadOnlyList<string> problems) =>
        $"{root} does not match what its merge wrote, so it was not read:"
        + string.Concat(problems.Select(p => Environment.NewLine + "  - " + p))
        + Environment.NewLine
        + "This usually means the output folder was copied over an older copy of itself by a copy that adds "
        + "and overwrites files but never deletes them. Copy it again with a mirroring copy (for example "
        + "robocopy /MIR), or rebuild merged_data by running prism run on this folder again: a run that "
        + "finds this rebuilds it, and --force-reprocess rebuilds everything.";
}
