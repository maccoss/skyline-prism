using System;
using System.Collections.Generic;
using System.IO;
using DuckDB.NET.Data;
using Parquet;

namespace Stage2Bench;

/// <summary>What a strategy produced, so arms can be checked against each other rather than just timed.</summary>
public readonly record struct ReadResult(long Rows, long Peptides, long Values);

/// <summary>
/// The candidate ways Stage 2 could read one partition.
/// <para>
/// Every arm must not only return the same <see cref="ReadResult"/> but do the same WORK: accumulate
/// each peptide's transition values into per-peptide lists, exactly as <c>PeptideBlock</c> does. An
/// earlier version of this file had the no-sort arm store a row-group-local index and throw the values
/// away, which made it look 3.12x faster than a path that was materializing everything. A faster arm
/// that reads the same rows but does less with them is not a faster arm either - <see cref="ReadResult.Values"/>
/// is counted so that shortcut cannot pass unnoticed again.
/// </para>
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

    /// <summary>One peptide's rows, as the rollup would consume them.</summary>
    private sealed class Block
    {
        public readonly List<string> Tid = new();
        public readonly List<string> Samp = new();
        public readonly List<double> Area = new();
        public readonly List<double> Rt = new();
        public int Count => Tid.Count;
    }

    private static string Select(string scan, bool ordered) =>
        $"SELECT \"{Peptide}\" AS pep, "
        + $"(COALESCE(\"{Ion}\",'nan') || '_z' || COALESCE(CAST(\"{PrecursorCharge}\" AS VARCHAR),'nan')"
        + $" || '_' || COALESCE(CAST(\"{ProductCharge}\" AS VARCHAR),'nan')) AS tid, "
        + $"\"{Sample}\" AS samp, \"{Area}\" AS area, \"{Rt}\" AS rt "
        + $"FROM {scan}" + (ordered ? $" ORDER BY \"{Peptide}\"" : "");

    private static string Scan(string glob) => $"read_parquet('{glob}', hive_partitioning=false)";

    /// <summary>
    /// A: what ships today. DuckDB sorts, the result is walked a row at a time through the ADO.NET
    /// reader, and a block closes when the peptide changes - so only one block is ever live.
    /// </summary>
    public static ReadResult DuckDbSortedStream(string glob, string scratch, int budgetMb)
    {
        using var conn = Open(scratch, budgetMb);
        using var cmd = conn.CreateCommand();
        cmd.CommandText = Select(Scan(glob), ordered: true);
        cmd.UseStreamingMode = true;
        using var r = cmd.ExecuteReader();

        long rows = 0, peptides = 0, values = 0;
        string? cur = null;
        var block = new Block();
        while (r.Read())
        {
            rows++;
            var pep = r.IsDBNull(0) ? "" : r.GetString(0);
            if (cur is null || !string.Equals(cur, pep, StringComparison.Ordinal))
            {
                values += Close(block);
                peptides++;
                cur = pep;
            }
            block.Tid.Add(r.IsDBNull(1) ? "" : r.GetString(1));
            block.Samp.Add(r.IsDBNull(2) ? "" : r.GetString(2));
            block.Area.Add(r.IsDBNull(3) ? double.NaN : r.GetDouble(3));
            block.Rt.Add(r.IsDBNull(4) ? double.NaN : r.GetDouble(4));
        }
        values += Close(block);
        return new ReadResult(rows, peptides, values);
    }

    /// <summary>
    /// B: no sort. Parquet.Net reads the partition as it lies and rows are grouped by peptide in a
    /// dictionary.
    /// <para>
    /// The trade this measures is not only speed. Without a sort, every peptide's block is live at once
    /// - the whole partition sits in managed memory until the partition is finished - where the sorted
    /// arms hold exactly one. Watch peak working set alongside the seconds; partition size is the knob
    /// that would bound it.
    /// </para>
    /// </summary>
    public static ReadResult ParquetNoSortGrouped(string glob, string scratch, int budgetMb)
    {
        long rows = 0, values = 0;
        var groups = new Dictionary<string, Block>(StringComparer.Ordinal);

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
                var pz = g.ReadColumnAsync(byName[PrecursorCharge]).GetAwaiter().GetResult().Data;
                var zz = g.ReadColumnAsync(byName[ProductCharge]).GetAwaiter().GetResult().Data;
                var samp = (string?[])g.ReadColumnAsync(byName[Sample]).GetAwaiter().GetResult().Data;
                var area = g.ReadColumnAsync(byName[Area]).GetAwaiter().GetResult().Data;
                var rt = g.ReadColumnAsync(byName[Rt]).GetAwaiter().GetResult().Data;

                for (var i = 0; i < pep.Length; i++)
                {
                    rows++;
                    var key = pep[i] ?? "";
                    if (!groups.TryGetValue(key, out var block))
                    {
                        block = new Block();
                        groups[key] = block;
                    }
                    // The transition id is composed here rather than in SQL, because there is no SQL in
                    // this arm - that composition is part of its cost and must not be omitted.
                    block.Tid.Add((ion[i] ?? "nan") + "_z" + Str(pz, i) + "_" + Str(zz, i));
                    block.Samp.Add(samp[i] ?? "");
                    block.Area.Add(Dbl(area, i));
                    block.Rt.Add(Dbl(rt, i));
                }
            }
        }
        foreach (var b in groups.Values)
            values += b.Count;
        return new ReadResult(rows, groups.Count, values);
    }

    /// <summary>
    /// C: the sketched two-phase design. DuckDB writes the sorted narrow projection to a temp parquet
    /// (phase 1), read back a row group at a time (phase 2). Sorted, so one block is live at a time.
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

        long rows = 0, peptides = 0, values = 0;
        string? cur = null;
        var block = new Block();
        using (var fs = File.OpenRead(tmp))
        {
            using var reader = ParquetReader.CreateAsync(fs).GetAwaiter().GetResult();
            var f = reader.Schema.DataFields;
            for (var rg = 0; rg < reader.RowGroupCount; rg++)
            {
                using var g = reader.OpenRowGroupReader(rg);
                var pep = (string?[])g.ReadColumnAsync(f[0]).GetAwaiter().GetResult().Data;
                var tid = (string?[])g.ReadColumnAsync(f[1]).GetAwaiter().GetResult().Data;
                var samp = (string?[])g.ReadColumnAsync(f[2]).GetAwaiter().GetResult().Data;
                var area = g.ReadColumnAsync(f[3]).GetAwaiter().GetResult().Data;
                var rt = g.ReadColumnAsync(f[4]).GetAwaiter().GetResult().Data;
                for (var i = 0; i < pep.Length; i++)
                {
                    rows++;
                    if (cur is null || !string.Equals(cur, pep[i], StringComparison.Ordinal))
                    {
                        values += Close(block);
                        peptides++;
                        cur = pep[i];
                    }
                    block.Tid.Add(tid[i] ?? "");
                    block.Samp.Add(samp[i] ?? "");
                    block.Area.Add(Dbl(area, i));
                    block.Rt.Add(Dbl(rt, i));
                }
            }
        }
        values += Close(block);
        try { File.Delete(tmp); } catch (IOException) { }
        return new ReadResult(rows, peptides, values);
    }

    /// <summary>Count a finished block's values and reset it, as emitting to the rollup would.</summary>
    private static long Close(Block b)
    {
        var n = b.Count;
        b.Tid.Clear();
        b.Samp.Clear();
        b.Area.Clear();
        b.Rt.Clear();
        return n;
    }

    private static string Str(Array a, int i)
    {
        var v = a.GetValue(i);
        return v?.ToString() ?? "nan";
    }

    private static double Dbl(Array a, int i)
    {
        var v = a.GetValue(i);
        return v is null ? double.NaN : Convert.ToDouble(v);
    }

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
