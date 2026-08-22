using System;
using System.Collections.Generic;
using System.IO;
using DuckDB.NET.Data;
using Parquet;

namespace Stage2Bench;

/// <summary>What a strategy produced, so arms can be checked against each other rather than just timed.</summary>
public readonly record struct ReadResult(long Rows, long Peptides, long Transitions);

/// <summary>
/// The candidate ways Stage 2 could read one partition. Every arm must return the same
/// <see cref="ReadResult"/>; a faster arm that reads different data is not a faster arm.
/// </summary>
public static class Strategies
{
    public const string Peptide = "PeptideModifiedSequenceUnimodIds";
    public const string Ion = "FragmentIon";
    public const string PrecursorCharge = "PrecursorCharge";
    public const string ProductCharge = "ProductCharge";
    public const string Sample = "Sample ID";
    public const string Area = "Area";
    public const string Rt = "RetentionTime";

    /// <summary>Narrow projection, in the shape Stage 2 actually consumes.</summary>
    private static string Select(string scan, bool ordered) =>
        $"SELECT \"{Peptide}\" AS pep, "
        + $"(COALESCE(\"{Ion}\",'nan') || '_z' || COALESCE(CAST(\"{PrecursorCharge}\" AS VARCHAR),'nan')"
        + $" || '_' || COALESCE(CAST(\"{ProductCharge}\" AS VARCHAR),'nan')) AS tid, "
        + $"\"{Sample}\" AS samp, \"{Area}\" AS area, \"{Rt}\" AS rt "
        + $"FROM {scan}" + (ordered ? $" ORDER BY \"{Peptide}\"" : "");

    private static string Scan(string glob) => $"read_parquet('{glob}', hive_partitioning=false)";

    /// <summary>
    /// A: what ships today. DuckDB sorts the partition, the result is walked a row at a time through
    /// the ADO.NET reader, and blocks close when the peptide changes.
    /// </summary>
    public static ReadResult DuckDbSortedStream(string glob, string scratch, int budgetMb)
    {
        using var conn = Open(scratch, budgetMb);
        using var cmd = conn.CreateCommand();
        cmd.CommandText = Select(Scan(glob), ordered: true);
        cmd.UseStreamingMode = true;
        using var r = cmd.ExecuteReader();

        long rows = 0, peptides = 0;
        string? cur = null;
        var tids = new HashSet<string>(StringComparer.Ordinal);
        while (r.Read())
        {
            rows++;
            var pep = r.IsDBNull(0) ? "" : r.GetString(0);
            if (cur is null || !string.Equals(cur, pep, StringComparison.Ordinal))
            {
                cur = pep;
                peptides++;
            }
            tids.Add(r.IsDBNull(1) ? "" : r.GetString(1));
            _ = r.IsDBNull(2) ? "" : r.GetString(2);
            _ = r.IsDBNull(3) ? double.NaN : r.GetDouble(3);
            _ = r.IsDBNull(4) ? double.NaN : r.GetDouble(4);
        }
        return new ReadResult(rows, peptides, tids.Count);
    }

    /// <summary>
    /// B: no sort at all. Parquet.Net reads the partition as it lies - parquet is columnar, so only the
    /// narrow columns are touched - and rows are grouped by peptide in a dictionary.
    /// <para>
    /// The hypothesis worth testing: the sort is ~2.7 s of A's ~4.7 s, so removing it should beat both
    /// other arms, with no temp file and nothing to clean up. The risk is that the grouping dictionary
    /// allocates as badly as the per-row strings that PR #40 just removed, which is exactly what this
    /// measures - watch the allocated column, not only the seconds.
    /// </para>
    /// </summary>
    public static ReadResult ParquetNoSortGrouped(string glob, string scratch, int budgetMb)
    {
        long rows = 0;
        var groups = new Dictionary<string, List<int>>(StringComparer.Ordinal);
        var tids = new HashSet<string>(StringComparer.Ordinal);

        foreach (var file in ResolveFiles(glob))
        {
            using var fs = File.OpenRead(file);
            using var reader = ParquetReader.CreateAsync(fs).GetAwaiter().GetResult();
            var byName = new Dictionary<string, Parquet.Schema.DataField>(StringComparer.Ordinal);
            foreach (var f in reader.Schema.DataFields)
                byName[f.Name] = f;

            for (var rg = 0; rg < reader.RowGroupCount; rg++)
            {
                using var g = reader.OpenRowGroupReader(rg);
                var pep = (string?[])g.ReadColumnAsync(byName[Peptide]).GetAwaiter().GetResult().Data;
                var ion = (string?[])g.ReadColumnAsync(byName[Ion]).GetAwaiter().GetResult().Data;
                var samp = (string?[])g.ReadColumnAsync(byName[Sample]).GetAwaiter().GetResult().Data;
                _ = g.ReadColumnAsync(byName[Area]).GetAwaiter().GetResult().Data;
                _ = g.ReadColumnAsync(byName[Rt]).GetAwaiter().GetResult().Data;

                for (var i = 0; i < pep.Length; i++)
                {
                    rows++;
                    var key = pep[i] ?? "";
                    if (!groups.TryGetValue(key, out var list))
                    {
                        list = new List<int>();
                        groups[key] = list;
                    }
                    list.Add(i);
                    tids.Add(ion[i] ?? "");
                    _ = samp[i];
                }
            }
        }
        return new ReadResult(rows, groups.Count, tids.Count);
    }

    /// <summary>
    /// C: the sketched two-phase design. DuckDB writes the sorted narrow projection to a temp parquet
    /// (phase 1), which is then read back a row group at a time (phase 2). Phase 2 is pure managed code
    /// and parallelizes; phase 1 is a DuckDB sort and does not, so it is the floor.
    /// </summary>
    public static ReadResult CopyThenParquetRead(string glob, string scratch, int budgetMb)
    {
        var tmp = Path.Combine(scratch, "bench_narrow.parquet");
        try { File.Delete(tmp); } catch (IOException) { }

        using (var conn = Open(scratch, budgetMb))
        using (var cmd = conn.CreateCommand())
        {
            cmd.CommandText =
                $"COPY ({Select(Scan(glob), ordered: true)}) TO '{tmp.Replace('\\', '/')}' "
                + "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)";
            cmd.ExecuteNonQuery();
        }

        long rows = 0, peptides = 0;
        string? cur = null;
        var tids = new HashSet<string>(StringComparer.Ordinal);
        using (var fs = File.OpenRead(tmp))
        {
            using var reader = ParquetReader.CreateAsync(fs).GetAwaiter().GetResult();
            var f = reader.Schema.DataFields;
            for (var rg = 0; rg < reader.RowGroupCount; rg++)
            {
                using var g = reader.OpenRowGroupReader(rg);
                var pep = (string?[])g.ReadColumnAsync(f[0]).GetAwaiter().GetResult().Data;
                var tid = (string?[])g.ReadColumnAsync(f[1]).GetAwaiter().GetResult().Data;
                _ = g.ReadColumnAsync(f[2]).GetAwaiter().GetResult().Data;
                _ = g.ReadColumnAsync(f[3]).GetAwaiter().GetResult().Data;
                _ = g.ReadColumnAsync(f[4]).GetAwaiter().GetResult().Data;
                for (var i = 0; i < pep.Length; i++)
                {
                    rows++;
                    if (cur is null || !string.Equals(cur, pep[i], StringComparison.Ordinal))
                    {
                        cur = pep[i];
                        peptides++;
                    }
                    tids.Add(tid[i] ?? "");
                }
            }
        }
        try { File.Delete(tmp); } catch (IOException) { }
        return new ReadResult(rows, peptides, tids.Count);
    }

    /// <summary>
    /// B and C count transitions from different columns (raw ion vs composed id), so the arms are
    /// comparable on rows and peptides but not on that third figure. Kept separate rather than fudged.
    /// </summary>
    public static bool ComparableTransitions(string arm) => arm != "nosort-managed";

    private static IEnumerable<string> ResolveFiles(string glob)
    {
        var dir = Path.GetDirectoryName(glob)!;
        var pattern = Path.GetFileName(glob);
        return Directory.EnumerateFiles(dir, pattern, SearchOption.AllDirectories);
    }

    private static DuckDBConnection Open(string scratch, int budgetMb)
    {
        Directory.CreateDirectory(scratch);
        var c = new DuckDBConnection("Data Source=:memory:");
        c.Open();
        Exec(c, $"SET memory_limit='{budgetMb}MB'");
        Exec(c, $"SET temp_directory='{scratch.Replace('\\', '/')}'");
        Exec(c, "SET preserve_insertion_order=false");
        Exec(c, $"SET threads={Environment.ProcessorCount}");
        return c;
    }

    private static void Exec(DuckDBConnection c, string sql)
    {
        using var cmd = c.CreateCommand();
        cmd.CommandText = sql;
        cmd.ExecuteNonQuery();
    }
}
