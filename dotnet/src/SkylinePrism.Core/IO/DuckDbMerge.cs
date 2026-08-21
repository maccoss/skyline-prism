using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using DuckDB.NET.Data;

namespace SkylinePrism.Core.IO;

/// <summary>
/// Streaming CSV/parquet merge, ported from data_io.py:merge_and_sort_streaming.
/// Issues the same DuckDB SQL as the Python pipeline (via the DuckDB.NET binding to the
/// same libduckdb engine), so the merged parquet has identical row CONTENT: the
/// synthesized Batch / Source Document / Sample ID columns (with the "__@__" join),
/// UNION ALL of all files, zstd output. Row content is the parity contract; row ORDER
/// is not, and deliberately differs from Python - see below.
///
/// <para>
/// <b>This stage does not sort, by design.</b> It used to close with
/// <c>ORDER BY &lt;peptide column&gt;</c> over the full union, on the theory that the
/// downstream transition rollup needs to see one peptide at a time. It does - but it gets
/// that from its OWN <c>ORDER BY</c> in
/// <see cref="MergedParquetReader.StreamPeptideBlocks"/>, which DuckDB issues regardless of
/// how the file is laid out (parquet carries no sortedness the optimizer will trust). So the
/// rows were being sorted twice, and the sort here was the expensive one: it is a blocking
/// operator over EVERY column of a transition-level report - ~26 of them, half wide repeated
/// strings (protein name, accession, gene, peptide, modified sequence, fragment ion,
/// replicate, file name, and the synthesized Sample ID) - while the reader's sort projects
/// the 8 narrow columns the rollup actually reads. Measured on a 2-plate cohort
/// (3.7 GB of parquet in): the wide sort peaked at ~35 GB RSS and spilled 31 GB; the
/// identical ordering, done narrow in the reader, fits in ~8 GB.
/// </para>
/// <para>
/// Dropping it makes this stage a single streaming <c>COPY</c> - readers feed the parquet
/// writer with no pipeline breaker in between - so memory is bounded by the reader and
/// writer buffers rather than by the size of the cohort, and the unsorted intermediate
/// (a full extra write plus a full extra read of the whole dataset) disappears with it.
/// Python still sorts here because its rollup consumes the file positionally
/// (<c>rollup_transitions_sorted(pre_sorted=True)</c>) and so genuinely depends on the
/// order; the C# reader does not. Do not "restore parity" by adding the sort back without
/// first removing the reader's.
/// </para>
/// <para>
/// What the write DOES do is hash each row into a peptide bucket and let parquet
/// <c>PARTITION_BY</c> fan it out - free here, since it rides on a pass that was happening
/// anyway, and it is what removes the last unbounded operator downstream. See
/// <see cref="MergedDataset"/> for why the rollup needs it.
/// </para>
/// </summary>
public static class DuckDbMerge
{
    // Peptide column preference order (data_io.py:1216). First match wins.
    private static readonly string[] PeptideColumnNames =
    {
        "Peptide Modified Sequence",
        "Peptide_Modified_Sequence",
        "Peptide Modified Sequence Unimod Ids",
        "Peptide_Modified_Sequence_Unimod_Ids",
        "Modified Sequence",
        "Modified_Sequence",
        "Peptide Sequence",
        "Peptide_Sequence",
        "Peptide",
    };

    private static readonly string[] ReplicateColumnNames =
    {
        "Replicate Name", "Replicate_Name", "ReplicateName",
    };

    private static readonly HashSet<string> SynthMetadataCols = new()
    {
        "Batch", "Source Document", "Sample ID",
    };

    /// <param name="PeptideColumn">
    /// The peptide column, which rows are hashed on to choose a partition. Nothing SORTS by it here
    /// (see the type remarks) - but it is what makes the partitioning correct, since every row of a
    /// peptide must land in the same bucket for the rollup to see them together.
    /// </param>
    /// <param name="Partitions">Number of buckets written; 1 on a cohort small enough not to need more.</param>
    public sealed record MergeResult(
        string OutputPath, string PeptideColumn, long TotalRows, string TempDirectory = "",
        int MemoryBudgetMb = 0, int Partitions = 1)
    {
        /// <summary>The merged data, ready to scan whole or one partition at a time.</summary>
        public MergedDataset Dataset() => MergedDataset.Open(OutputPath);
    }

    /// <summary>
    /// Floor for the DuckDB budget. Below this the reader buffers alone do not fit and the merge
    /// fails outright instead of spilling, so a busy machine gets a small budget, never a broken one.
    /// </summary>
    internal const int MinMemoryBudgetMb = 2048;

    /// <summary>
    /// Ceiling for the AUTOMATIC budget (an explicit <c>processing.merge_memory_mb</c> is honoured
    /// above it). Since the merge stopped sorting, nothing here scales with the size of the cohort:
    /// the buffer pool holds per-thread reader buffers and the parquet writer's in-flight row groups,
    /// which a few GB covers on any machine. Handing DuckDB a fraction of a large machine's RAM would
    /// just let it cache the whole scan for no benefit - and at real cost, because that memory is
    /// taken from the Skyline instance PRISM was launched from, which is sitting right there holding
    /// the documents being processed.
    /// </summary>
    internal const int MaxAutoMemoryBudgetMb = 8192;

    /// <summary>
    /// Override for DuckDB's spill directory. Set it when the automatic choice picks badly - the
    /// merge writes gigabytes of spill here on a large cohort, so it wants a fast local disk with
    /// room, not a quota'd or synced one.
    /// </summary>
    public const string TempDirEnvVar = "PRISM_TEMP_DIR";

    /// <summary>
    /// Where DuckDB spills the sort. Beside the output when that is a local disk (same volume, so
    /// nothing crosses a filesystem and cleanup is obvious), but NOT when the output is on a network
    /// share: PRISM output routinely lives on a mapped drive, and spilling a multi-gigabyte sort over
    /// SMB is slow enough to look like a hang - and fails outright on some servers. In that case fall
    /// back to the machine's own temp directory.
    /// </summary>
    internal static string ResolveTempDirectory(string outputPath)
    {
        var overridden = Environment.GetEnvironmentVariable(TempDirEnvVar);
        if (!string.IsNullOrWhiteSpace(overridden))
            return Path.Combine(overridden, "prism-duckdb");

        var beside = Path.Combine(
            Path.GetDirectoryName(Path.GetFullPath(outputPath))!, ".duckdb_temp");
        return IsNetworkPath(beside)
            ? Path.Combine(Path.GetTempPath(), "prism-duckdb")
            : beside;
    }

    /// <summary>
    /// How much memory to let DuckDB use, when the caller has not said.
    /// <para>
    /// Three bounds, whichever is smallest: <b>25% of total RAM</b>, <b>50% of free RAM</b>, and the
    /// flat <see cref="MaxAutoMemoryBudgetMb"/> ceiling - floored at <see cref="MinMemoryBudgetMb"/>
    /// so a busy machine gets a small budget, never a broken one. The free-RAM bound is what keeps a
    /// half-full machine honest: DuckDB's buffer pool is native memory outside the GC's view, so a
    /// budget written against total RAM alone does not spill on a loaded machine, it pages.
    /// </para>
    /// <para>
    /// These fractions used to be 75% of total / 80% of free, from when this stage sorted the whole
    /// cohort and a bigger buffer pool meant less spilling. It no longer sorts (see the type
    /// remarks), so there is nothing here for a large budget to buy - and the old one actively hurt:
    /// on a 62 GB workstation with Skyline holding two 13 GB documents, it let the merge take ~35 GB,
    /// which Windows found by paging Skyline out. The whole system swapped for 40 minutes.
    /// </para>
    /// <para>
    /// All three are ceilings on the buffer pool, not reservations - work beyond the budget spills to
    /// <see cref="ResolveTempDirectory"/>. So a small budget is slower, never wrong, which is why
    /// they can be applied without knowing how big the cohort is.
    /// </para>
    /// </summary>
    internal static int AutoMemoryBudgetMb()
    {
        const long mb = 1024L * 1024L;
        var budgetMb = SystemMemory.TotalPhysicalBytes / mb / 4;

        var available = SystemMemory.AvailablePhysicalBytes();
        if (available is > 0)
            budgetMb = Math.Min(budgetMb, available.Value / mb / 2);

        budgetMb = Math.Min(budgetMb, MaxAutoMemoryBudgetMb);
        return (int)Math.Max(MinMemoryBudgetMb, budgetMb);
    }

    /// <summary>UNC path, or a drive the OS reports as a network mount.</summary>
    private static bool IsNetworkPath(string path)
    {
        try
        {
            var full = Path.GetFullPath(path);
            if (full.StartsWith(@"\\", StringComparison.Ordinal))
                return true;
            var root = Path.GetPathRoot(full);
            return !string.IsNullOrEmpty(root)
                   && new DriveInfo(root).DriveType == DriveType.Network;
        }
        catch (Exception)
        {
            // Unknown/unreachable volume: keep the old behavior rather than refuse to merge.
            return false;
        }
    }

    /// <param name="partitionsOverride">
    /// Force the bucket count instead of deriving it from the cohort size and budget. For tests: the
    /// mini fixtures are a few thousand rows, so the real sizing gives them one partition, and the
    /// invariants that matter most - a peptide never split across partitions, blocks still whole across
    /// a partition boundary - are vacuous unless several can be forced. 0 = derive normally.
    /// </param>
    public static MergeResult Merge(
        IReadOnlyList<string> reportPaths,
        string outputPath,
        string? peptideColumn = null,
        IReadOnlyList<string>? batchNames = null,
        int memoryBudgetMb = 0,
        string? replicateColumn = null,
        CancellationToken cancellationToken = default,
        int partitionsOverride = 0)
    {
        if (reportPaths.Count == 0)
            throw new ArgumentException("No report paths provided.", nameof(reportPaths));

        if (memoryBudgetMb <= 0)
            memoryBudgetMb = AutoMemoryBudgetMb();

        batchNames ??= reportPaths.Select(p => Path.GetFileNameWithoutExtension(p)).ToList();
        if (batchNames.Count != reportPaths.Count)
            throw new ArgumentException("batchNames must have same length as reportPaths.");

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath))!);

        // Read per-file headers.
        var fileHeaders = reportPaths.Select(ReadHeader).ToList();
        var firstHeader = fileHeaders[0];

        // Auto-detect the peptide column. Nothing here sorts by it, but a transition report without
        // one cannot be rolled up, and failing now beats failing after a full merge.
        peptideColumn ??= PeptideColumnNames.FirstOrDefault(firstHeader.Contains)
            ?? throw new InvalidOperationException(
                "Could not find a peptide column. Looked for: "
                + string.Join(", ", PeptideColumnNames));

        // Validate data columns match across files (synth metadata cols allowed to differ).
        var firstDataCols = firstHeader.Where(c => !SynthMetadataCols.Contains(c)).ToHashSet();
        for (var i = 1; i < reportPaths.Count; i++)
        {
            var hdrData = fileHeaders[i].Where(c => !SynthMetadataCols.Contains(c)).ToHashSet();
            var missing = firstDataCols.Except(hdrData).OrderBy(c => c).ToList();
            if (missing.Count > 0)
                throw new InvalidOperationException(
                    $"{Path.GetFileName(reportPaths[i])} is missing columns present in "
                    + $"{Path.GetFileName(reportPaths[0])}: {string.Join(", ", missing.Take(10))}");
        }

        // Build the per-file SELECTs and UNION ALL.
        var unionParts = new List<string>();
        for (var i = 0; i < reportPaths.Count; i++)
        {
            unionParts.Add(BuildFileSelect(reportPaths[i], batchNames[i], fileHeaders[i], replicateColumn));
        }
        var unionQuery = string.Join(" UNION ALL ", unionParts);

        var tempDir = ResolveTempDirectory(outputPath);
        Directory.CreateDirectory(tempDir);

        var fullOutput = Path.GetFullPath(outputPath);
        var outputEsc = SqlEscape(fullOutput);
        // A stale layout underneath would be read back as extra partitions, so the target is always
        // cleared rather than written over. Callers only reach here when they mean to rebuild.
        MergedDataset.Delete(fullOutput);

        long totalRows;
        int partitions;
        try
        {
            using var conn = new DuckDBConnection("Data Source=:memory:");
            conn.Open();
            DuckDbTuning.Apply(conn, memoryBudgetMb, tempDir);

            partitions = partitionsOverride > 0
                ? partitionsOverride
                : MergedDataset.PartitionCountFor(
                    EstimateTotalRows(conn, reportPaths, fileHeaders), memoryBudgetMb);
            DuckDbTuning.ApplyPartitionedWrite(conn, partitions);

            // Hash on the peptide column so all of a peptide's rows share a bucket - that is what lets
            // the rollup process one partition at a time. COALESCE because a NULL peptide would
            // otherwise land in DuckDB's default-partition directory rather than a numbered bucket.
            var bucket = $"(hash(COALESCE(CAST(\"{peptideColumn}\" AS VARCHAR), '')) % {partitions})"
                       + $" AS \"{MergedDataset.BucketColumn}\"";

            // One streaming pass: readers -> partitioned parquet writer, no pipeline breaker between
            // them, so peak memory is the reader buffers plus the writer's in-flight row groups no
            // matter how big the cohort is. DuckDB bounds the partition writers itself (it keeps a
            // fixed number open and rotates the rest), so a high partition count costs files, not RAM.
            Exec(conn, cancellationToken, $@"
                COPY (
                    SELECT *, {bucket} FROM ({unionQuery})
                ) TO '{outputEsc}' (
                    FORMAT PARQUET,
                    COMPRESSION ZSTD,
                    ROW_GROUP_SIZE {RowGroupSize},
                    PARTITION_BY ({MergedDataset.BucketColumn})
                )");

            // Cheap: parquet row counts come from the file footers, so this reads metadata rather
            // than scanning the data back.
            using var cmd = conn.CreateCommand();
            cmd.CommandText =
                $"SELECT COUNT(*) FROM read_parquet('{SqlEscape(MergedDataset.Open(fullOutput).ScanTarget)}', "
                + "hive_partitioning=false)";
            totalRows = Convert.ToInt64(cmd.ExecuteScalar());
        }
        finally
        {
            // In a finally because the spill directory may live outside the output directory, where a
            // failed run would otherwise leave gigabytes behind with nothing pointing at it.
            try { Directory.Delete(tempDir, recursive: true); } catch (IOException) { }
        }

        return new MergeResult(
            fullOutput, peptideColumn, totalRows, tempDir, memoryBudgetMb, partitions);
    }

    /// <summary>
    /// Roughly how many rows the merge is about to write, to choose a partition count. Parquet inputs
    /// answer exactly from their footers; text inputs are estimated from file size over the mean length
    /// of a sample of lines. Only the order of magnitude matters - the partition count is clamped, and
    /// being one bucket out changes the rollup's footprint by a few percent, so an estimate that costs
    /// nothing beats an exact count that costs a full scan.
    /// </summary>
    private static long EstimateTotalRows(
        DuckDBConnection conn, IReadOnlyList<string> reportPaths, IReadOnlyList<List<string>> headers)
    {
        long total = 0;
        for (var i = 0; i < reportPaths.Count; i++)
        {
            var path = reportPaths[i];
            try
            {
                if (Path.GetExtension(path).Equals(".parquet", StringComparison.OrdinalIgnoreCase))
                {
                    using var cmd = conn.CreateCommand();
                    cmd.CommandText =
                        $"SELECT COUNT(*) FROM read_parquet('{SqlEscape(Path.GetFullPath(path))}')";
                    total += Convert.ToInt64(cmd.ExecuteScalar());
                }
                else
                {
                    total += EstimateTextRows(path);
                }
            }
            catch (Exception)
            {
                // An unreadable input fails properly in the COPY below, with a better message than
                // anything an estimator could raise. Here it just means one fewer input counted.
            }
        }
        return total;
    }

    private static long EstimateTextRows(string path)
    {
        var info = new FileInfo(path);
        if (!info.Exists || info.Length == 0)
            return 0;

        using var sr = new StreamReader(path);
        long sampled = 0;
        var lines = 0;
        for (; lines < 200; lines++)
        {
            var line = sr.ReadLine();
            if (line is null)
                break;
            sampled += line.Length + 1; // + the newline the reader stripped
        }
        if (lines == 0 || sampled == 0)
            return 0;
        return info.Length / Math.Max(1, sampled / lines);
    }

    /// <summary>
    /// Rows per parquet row group. DuckDB buffers a whole row group per writing thread before it
    /// flushes, so this is multiplied by the thread count in peak memory: at the previous 1,000,000
    /// it was ~26 columns x 1M rows in flight per thread, which on a many-core box is gigabytes of
    /// write buffer on its own. DuckDB's default is a deliberate balance for exactly this reason, and
    /// smaller groups also give the downstream per-sample scans (the density map) finer row-group
    /// skipping, so nothing downstream wants the larger value either.
    /// </summary>
    private const int RowGroupSize = 122880;

    private static string BuildFileSelect(
        string filePath, string batchName, IReadOnlyList<string> header, string? replicateColumn = null)
    {
        var suffix = Path.GetExtension(filePath).ToLowerInvariant();
        var isParquet = suffix == ".parquet";
        var stem = Path.GetFileNameWithoutExtension(filePath);
        var pathStr = SqlEscape(Path.GetFullPath(filePath));
        var batchEsc = batchName.Replace("'", "''");
        var stemEsc = stem.Replace("'", "''");

        // Prefer the configured replicate column (data.sample_column), matched case/space/underscore-
        // insensitively, then the standard names. This is the INPUT column used to synthesize the
        // batch-disambiguated "Sample ID"; the output sample column is always that synthesized Sample ID.
        var fileReplicateCol =
            (replicateColumn is not null ? SkylineColumns.FindColumn(header.ToList(), replicateColumn) : null)
            ?? ReplicateColumnNames.FirstOrDefault(header.Contains)
            ?? throw new InvalidOperationException(
                $"Could not find replicate column in {Path.GetFileName(filePath)}.");

        var addCols = new List<string>();
        if (!header.Contains("Batch"))
            addCols.Add($"'{batchEsc}' AS \"Batch\"");
        if (!header.Contains("Source Document"))
            addCols.Add($"'{stemEsc}' AS \"Source Document\"");
        if (!header.Contains("Sample ID"))
            addCols.Add($"\"{fileReplicateCol}\" || '__@__' || '{batchEsc}' AS \"Sample ID\"");

        var selectClause = addCols.Count > 0
            ? "SELECT *, " + string.Join(", ", addCols)
            : "SELECT *";

        if (isParquet)
            return $"{selectClause} FROM read_parquet('{pathStr}')";

        var sqlDelim = suffix is ".tsv" or ".txt" ? "\\t" : ",";
        return $"{selectClause} FROM read_csv('{pathStr}', "
             + $"header=true, delim='{sqlDelim}', ignore_errors=true, all_varchar=false)";
    }

    private static List<string> ReadHeader(string filePath)
    {
        var suffix = Path.GetExtension(filePath).ToLowerInvariant();
        if (suffix == ".parquet")
        {
            // DESCRIBE gives column names without materializing rows.
            using var conn = new DuckDBConnection("Data Source=:memory:");
            conn.Open();
            using var cmd = conn.CreateCommand();
            cmd.CommandText =
                $"SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{SqlEscape(Path.GetFullPath(filePath))}'))";
            using var reader = cmd.ExecuteReader();
            var cols = new List<string>();
            while (reader.Read())
                cols.Add(reader.GetString(0));
            return cols;
        }

        var delimiter = suffix is ".tsv" or ".txt" ? '\t' : ',';
        using var sr = new StreamReader(filePath);
        var line = sr.ReadLine() ?? string.Empty;
        return line.Split(delimiter).Select(s => s.Trim()).ToList();
    }

    /// <summary>
    /// Run one statement, interruptible. The merge's two COPY statements are the longest single
    /// operations in the pipeline - many minutes on a large cohort - so a Stop that only took effect
    /// between them would not feel like a stop at all. DuckDB's own interrupt is what makes the
    /// in-flight query abandon its work.
    /// </summary>
    private static void Exec(DuckDBConnection conn, string sql) => Exec(conn, default, sql);

    /// <inheritdoc cref="Exec(DuckDBConnection, string)"/>
    private static void Exec(DuckDBConnection conn, CancellationToken cancellationToken, string sql)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        using var registration = cancellationToken.Register(() =>
        {
            try { cmd.Cancel(); }
            catch (Exception) { /* already finished, or a backend without interrupt support */ }
        });
        try
        {
            cmd.ExecuteNonQuery();
        }
        catch (Exception) when (cancellationToken.IsCancellationRequested)
        {
            // The interrupt surfaces as an ordinary query error; report it as the cancellation it is.
            cancellationToken.ThrowIfCancellationRequested();
            throw;
        }
    }

    private static string SqlEscape(string path) => path.Replace("'", "''");
}
