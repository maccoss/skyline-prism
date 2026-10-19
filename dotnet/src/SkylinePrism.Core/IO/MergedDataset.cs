using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Parquet;

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
    /// The sidecar the pipeline writes beside the merged data, holding the inputs' fingerprint and the
    /// row count the merge took of what it wrote (<see cref="SourceFingerprint.CacheEntry"/>). The same
    /// rule for both layouts: <c>merged_data.cache.json</c>, and <c>merged_data.parquet.cache.json</c>
    /// beside the single file earlier releases wrote.
    /// </summary>
    public static string SidecarPathFor(string root) =>
        Path.TrimEndingDirectorySeparator(root) + ".cache.json";

    /// <summary>
    /// Describe what is actually on disk at <paramref name="root"/>, and refuse it if it no longer
    /// holds the rows the merge that wrote it counted. Accepts both the partitioned directory this
    /// release writes and the single <c>merged_data.parquet</c> written by earlier ones, so an existing
    /// output directory still opens (in the QC report, the density tab, and the merge cache) without
    /// being re-merged.
    /// </summary>
    /// <remarks>
    /// <para>The check is against the count in <see cref="SidecarPathFor"/>, and it is what stops a
    /// damaged directory being read as a whole one. Every reader scans the directory recursively, so a
    /// partition the merge did not write is read as more data. One was found beside a real 96-run
    /// cohort: a copy of its output directory made over an older copy, replacing files and removing
    /// none, kept <c>_pep_bucket=2</c> of a 3-partition merge beside the 2-partition merge that
    /// replaced it. That was 9.9M duplicated rows on 29.9M, and a QC section counted a third more
    /// precursors than there were. PRISM's own publish cannot leave one (see
    /// <c>DuckDbMerge.Publish</c>), but the partition count follows free memory, so two runs of one
    /// cohort can write different numbers of buckets, and a copy that only adds is all it takes.</para>
    ///
    /// <para>The cost is one footer read per file - no column data - which is small against the read
    /// that follows. With no sidecar, or one that recorded no count, there is nothing to check
    /// against and the directory opens as it is.</para>
    /// </remarks>
    /// <exception cref="InvalidDataException">The files do not add up to the recorded count, or one
    /// of them is not readable parquet. The message names the partitions responsible where it can.</exception>
    public static MergedDataset Open(string root)
    {
        var dataset = OpenUnverified(root);
        var recorded = SourceFingerprint.TryRead(SidecarPathFor(root))?.TotalRows ?? 0;
        if (recorded > 0 && dataset.DescribeMismatch(recorded) is { } problem)
            throw new InvalidDataException(
                problem + " Re-running PRISM into this output directory rebuilds it from the inputs.");
        return dataset;
    }

    /// <summary>
    /// <see cref="Open"/> without the check against the recorded count. For a caller that holds the
    /// count itself and has already checked it (the pipeline's merge-cache reuse), or that has just
    /// counted the directory (a merge in its own staging directory).
    /// </summary>
    internal static MergedDataset OpenUnverified(string root)
    {
        if (File.Exists(root))
            return new MergedDataset(root, false, new[] { root }, root);

        if (!Directory.Exists(root))
            throw new DirectoryNotFoundException($"No merged data at {root}.");

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

    /// <summary>How many partitions a mismatch message names before it summarizes the rest.</summary>
    private const int NamedInMessage = 8;

    /// <summary>
    /// Null when this dataset's files hold exactly <paramref name="recordedRows"/> rows between them,
    /// and every one of them is readable. Otherwise a sentence or two saying how they differ, naming
    /// the partitions responsible where that can be told.
    /// </summary>
    /// <remarks>
    /// <para>Counted from the parquet footers, never from the data. The files counted are the ones
    /// <see cref="ScanTarget"/> reads, recursively, because that scan is what a stray file corrupts.</para>
    ///
    /// <para>How the culprit is told: a merge writes buckets <c>0..n-1</c>, so an earlier merge with
    /// MORE buckets leaves its extras at the top of the range. When the lowest-numbered partitions add
    /// up to exactly the recorded count, the ones above them are what the merge did not write. That is
    /// the case that was found. Anything else - rows missing, or extra rows that are not a tail of
    /// whole partitions - is reported with every partition's count, since there is nothing better to
    /// say about it.</para>
    /// </remarks>
    public string? DescribeMismatch(long recordedRows)
    {
        var parts = CountPartitions();
        var found = parts.Sum(p => p.Rows);
        var unreadable = parts.Where(p => p.Unreadable is not null).ToList();
        if (found == recordedRows && unreadable.Count == 0)
            return null;

        var name = Path.GetFileName(Path.TrimEndingDirectorySeparator(Root));
        var sidecar = Path.GetFileName(SidecarPathFor(Root));
        var message = new StringBuilder();
        message.Append(found == recordedRows
            ? $"{name} holds a file its merge did not write."
            : $"{name} holds {found:N0} rows, but the merge that wrote it counted {recordedRows:N0} "
              + $"(recorded in {sidecar}).");
        foreach (var p in unreadable)
            message.Append($" {p.Name} could not be read as parquet ({p.Unreadable}).");

        // The longest run of lowest-numbered partitions that is still within the count. Kept only when
        // it lands on the count exactly and leaves something above it.
        var ordered = parts.OrderBy(p => p.Bucket).ThenBy(p => p.Name, StringComparer.Ordinal).ToList();
        long prefix = 0;
        var k = 0;
        while (k < ordered.Count && prefix + ordered[k].Rows <= recordedRows && prefix < recordedRows)
            prefix += ordered[k++].Rows;

        if (IsPartitioned && prefix == recordedRows && k > 0 && k < ordered.Count)
        {
            var extras = ordered.Skip(k).ToList();
            message.Append(k == 1
                ? $" {ordered[0].Name} alone holds exactly the {recordedRows:N0} rows it counted, so "
                : $" {ordered[0].Name} to {ordered[k - 1].Name} hold exactly the {recordedRows:N0} rows "
                  + "it counted, so ");
            message.Append(Describe(extras));
            message.Append(extras.Count == 1 ? " is" : " are");
            message.Append(" left over from an earlier merge with more partitions - most likely a copy "
                + "or sync that added files over an older copy of this directory without removing any. "
                + "Read whole, the directory counts those rows on top of the merge's own.");
        }
        else if (found != recordedRows && IsPartitioned)
        {
            message.Append(found < recordedRows
                ? " Rows are missing: a partition may have been removed, or a copy stopped part way."
                : " It holds rows the merge did not write.");
            message.Append(" Partitions: ").Append(Describe(ordered)).Append('.');
        }
        return message.ToString();
    }

    /// <summary>One partition's footer count, for <see cref="DescribeMismatch"/>.</summary>
    /// <param name="Bucket">Its bucket number, or <see cref="int.MaxValue"/> for anything else.</param>
    /// <param name="Unreadable">Why a file in it could not be counted, or null.</param>
    private sealed record PartitionCount(
        string Name, int Bucket, long Rows, DateTime LastWrite, string? Unreadable);

    private IReadOnlyList<PartitionCount> CountPartitions()
    {
        if (!IsPartitioned)
            return new[] { Count(Path.GetFileName(Root), int.MaxValue, new[] { Root }) };

        // Grouped by the directory directly under Root - the partition - with a file sitting in Root
        // itself as its own entry, since the recursive scan reads that too.
        return Directory
            .EnumerateFiles(Root, "*.parquet", SearchOption.AllDirectories)
            .GroupBy(f => Path.GetRelativePath(Root, f)
                .Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)[0],
                StringComparer.Ordinal)
            .Select(g => Count(g.Key, BucketOf(g.Key), g.ToList()))
            .ToList();
    }

    private static PartitionCount Count(string name, int bucket, IReadOnlyList<string> files)
    {
        long rows = 0;
        string? unreadable = null;
        var lastWrite = DateTime.MinValue;
        foreach (var file in files.OrderBy(f => f, StringComparer.Ordinal))
        {
            lastWrite = Max(lastWrite, File.GetLastWriteTime(file));
            try
            {
                rows += FooterRows(file);
            }
            catch (Exception ex)
            {
                // Broad on purpose: Parquet.Net reports a file that is not parquet, or a torn footer,
                // as anything from an IOException to an index error, and every one of them means the
                // same thing here - a file the merge did not write. Recorded, never thrown, so the
                // message can still say what the rest of the directory holds.
                unreadable ??= $"{Path.GetFileName(file)}: {ex.Message}";
            }
        }
        return new PartitionCount(name, bucket, rows, lastWrite, unreadable);
    }

    private static DateTime Max(DateTime a, DateTime b) => a > b ? a : b;

    /// <summary>Rows in one parquet file, summed over its row groups' footer counts.</summary>
    private static long FooterRows(string path)
    {
        // Not ParquetColumnReader.RowCountOf, which counts in an int: a legacy single-file merge of a
        // large cohort is past int.MaxValue rows.
        using var fs = ParquetColumnIo.OpenRead(path);
        var reader = ParquetReader
            .CreateAsync(fs, ParquetColumnIo.Options(), leaveStreamOpen: true)
            .GetAwaiter().GetResult();
        try
        {
            long rows = 0;
            for (var rg = 0; rg < reader.RowGroupCount; rg++)
            {
                using var group = reader.OpenRowGroupReader(rg);
                rows += group.RowCount;
            }
            return rows;
        }
        finally
        {
            // Sync facade over an IAsyncDisposable reader, as ParquetColumnReader.Dispose does.
            reader.DisposeAsync().AsTask().GetAwaiter().GetResult();
        }
    }

    private static int BucketOf(string partitionName) =>
        partitionName.StartsWith(BucketColumn + "=", StringComparison.Ordinal)
        && int.TryParse(partitionName.AsSpan(BucketColumn.Length + 1),
            NumberStyles.None, CultureInfo.InvariantCulture,
            out var n)
            ? n
            : int.MaxValue;

    private static string Describe(IReadOnlyList<PartitionCount> parts)
    {
        var named = parts.Take(NamedInMessage).Select(p =>
            $"{p.Name} ({p.Rows:N0} rows, written "
            + p.LastWrite.ToString("yyyy-MM-dd HH:mm", CultureInfo.InvariantCulture)
            + ")");
        var text = string.Join(", ", named);
        return parts.Count > NamedInMessage
            ? $"{text} and {parts.Count - NamedInMessage:N0} more"
            : text;
    }

    private static string Glob(string dir, bool recursive = false) =>
        Path.Combine(dir, recursive ? "**" : "", "*.parquet").Replace('\\', '/');
}
