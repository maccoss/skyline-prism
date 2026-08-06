using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using DuckDB.NET.Data;

namespace SkylinePrism.Core.IO;

/// <summary>
/// Streaming CSV/parquet merge + sort, ported from data_io.py:merge_and_sort_streaming.
/// Issues the SAME DuckDB SQL as the Python pipeline (via the DuckDB.NET binding to the
/// same libduckdb engine), so the merged parquet has identical row content: the
/// synthesized Batch / Source Document / Sample ID columns (with the "__@__" join),
/// UNION ALL of all files, ORDER BY the peptide column, zstd output.
///
/// Row TIE order under ORDER BY is engine/thread dependent, but that only affects the
/// downstream pivot aggfunc="first" when a (peptide, transition, sample) key is
/// duplicated -- which clean Skyline exports never are -- so the merged CONTENT is the
/// parity contract.
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

    public sealed record MergeResult(
        string OutputPath, string SortColumn, long TotalRows, string TempDirectory = "");

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
            // Unknown/unreachable volume: keep the old behaviour rather than refuse to merge.
            return false;
        }
    }

    public static MergeResult MergeAndSort(
        IReadOnlyList<string> reportPaths,
        string outputPath,
        string? sortColumn = null,
        IReadOnlyList<string>? batchNames = null,
        int sortBufferMb = 0,
        string? replicateColumn = null)
    {
        if (reportPaths.Count == 0)
            throw new ArgumentException("No report paths provided.", nameof(reportPaths));

        // Use the machine: budget = 75% of RAM so all worker threads' read/decompress buffers fit, with
        // ~25% headroom for the .NET host + OS (DuckDB runs in-process), and temp_directory spill for any
        // sort working set beyond the budget - so we get full throughput without OOM. The original bug was
        // a fixed 8 GB limit that was smaller than the upfront per-thread read buffers when DuckDB ran one
        // thread per core, so it OOM'd in seconds before sorting - not too little memory for the data, just
        // too little to hold the parallel read. 0 = auto.
        if (sortBufferMb <= 0)
        {
            var availableMb = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / (1024L * 1024L);
            sortBufferMb = (int)Math.Max(2048, availableMb * 3 / 4);
        }

        batchNames ??= reportPaths.Select(p => Path.GetFileNameWithoutExtension(p)).ToList();
        if (batchNames.Count != reportPaths.Count)
            throw new ArgumentException("batchNames must have same length as reportPaths.");

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath))!);

        // Read per-file headers.
        var fileHeaders = reportPaths.Select(ReadHeader).ToList();
        var firstHeader = fileHeaders[0];

        // Auto-detect sort (peptide) column.
        sortColumn ??= PeptideColumnNames.FirstOrDefault(firstHeader.Contains)
            ?? throw new InvalidOperationException(
                "Could not find a peptide column for sorting. Looked for: "
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

        var unsortedPath = Path.ChangeExtension(outputPath, ".unsorted.parquet");
        var outputEsc = SqlEscape(Path.GetFullPath(outputPath));
        var unsortedEsc = SqlEscape(Path.GetFullPath(unsortedPath));

        long totalRows;
        try
        {
            using var conn = new DuckDBConnection("Data Source=:memory:");
            conn.Open();
            // Use all cores; the 75%-of-RAM budget above is sized so all threads' read buffers fit, and
            // the sort spills to temp_directory beyond it.
            Exec(conn, $"SET memory_limit='{sortBufferMb}MB'");
            Exec(conn, $"SET temp_directory='{SqlEscape(tempDir)}'");
            Exec(conn, "SET preserve_insertion_order=false");
            Exec(conn, $"SET threads={Environment.ProcessorCount}");

            // Stage A: union -> unsorted intermediate (snappy).
            Exec(conn, $@"
                COPY (
                    {unionQuery}
                ) TO '{unsortedEsc}' (
                    FORMAT PARQUET,
                    COMPRESSION SNAPPY
                )");

            // Stage B: single-source ORDER BY -> zstd output.
            Exec(conn, $@"
                COPY (
                    SELECT * FROM read_parquet('{unsortedEsc}')
                    ORDER BY ""{sortColumn}""
                ) TO '{outputEsc}' (
                    FORMAT PARQUET,
                    COMPRESSION ZSTD,
                    ROW_GROUP_SIZE 1000000
                )");

            using var cmd = conn.CreateCommand();
            cmd.CommandText = $"SELECT COUNT(*) FROM read_parquet('{outputEsc}')";
            totalRows = Convert.ToInt64(cmd.ExecuteScalar());
        }
        finally
        {
            // In a finally because the spill directory may now live outside the output directory,
            // where a failed run would otherwise leave gigabytes behind with nothing pointing at it.
            try { File.Delete(unsortedPath); } catch (IOException) { }
            try { Directory.Delete(tempDir, recursive: true); } catch (IOException) { }
        }

        return new MergeResult(Path.GetFullPath(outputPath), sortColumn, totalRows, tempDir);
    }

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

    private static void Exec(DuckDBConnection conn, string sql)
    {
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        cmd.ExecuteNonQuery();
    }

    private static string SqlEscape(string path) => path.Replace("'", "''");
}
