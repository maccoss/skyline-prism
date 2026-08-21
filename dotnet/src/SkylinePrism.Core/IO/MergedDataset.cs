using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

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
}
