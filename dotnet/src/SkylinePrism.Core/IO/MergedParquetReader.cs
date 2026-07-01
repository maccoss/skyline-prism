using System;
using System.Collections.Generic;
using System.Globalization;
using DuckDB.NET.Data;
using SkylinePrism.Core.Rollup;

namespace SkylinePrism.Core.IO;

/// <summary>
/// Streams the merged transition-level parquet grouped by peptide, using DuckDB.NET to
/// read rows ordered by the peptide column (mirroring chunked_processing.rollup_transitions_sorted,
/// which sorts by peptide then streams peptide-by-peptide). Genuinely streaming: the
/// DuckDB data reader yields rows without materializing the whole file.
/// </summary>
public static class MergedParquetReader
{
    /// <summary>Distinct, sorted, non-null values of the sample column (the wide output columns).</summary>
    public static List<string> GetSortedSamples(string parquetPath, string sampleCol)
    {
        using var conn = new DuckDBConnection("Data Source=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText =
            $"SELECT DISTINCT \"{sampleCol}\" AS s FROM read_parquet('{Esc(parquetPath)}') " +
            "WHERE s IS NOT NULL ORDER BY s";
        using var reader = cmd.ExecuteReader();
        var samples = new List<string>();
        while (reader.Read())
            samples.Add(reader.GetString(0));
        return samples;
    }

    /// <summary>
    /// Stream <see cref="PeptideBlock"/>s ordered by peptide. Consecutive rows sharing a
    /// peptide value are grouped into one block.
    /// </summary>
    public static IEnumerable<PeptideBlock> StreamPeptideBlocks(string parquetPath, SkylineColumns cols)
    {
        using var conn = new DuckDBConnection("Data Source=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText =
            "SELECT " +
            $"\"{cols.Peptide}\" AS pep, " +
            $"\"{cols.Transition}\" AS ion, " +
            $"\"{cols.PrecursorCharge}\" AS pz, " +
            $"\"{cols.ProductCharge}\" AS zz, " +
            $"\"{cols.Sample}\" AS samp, " +
            $"\"{cols.Abundance}\" AS area, " +
            $"\"{cols.RetentionTime}\" AS rt " +
            $"FROM read_parquet('{Esc(parquetPath)}') " +
            $"ORDER BY \"{cols.Peptide}\"";
        using var reader = cmd.ExecuteReader();

        PeptideBlock? current = null;
        while (reader.Read())
        {
            var pep = reader.IsDBNull(0) ? string.Empty : reader.GetString(0);
            if (current is null || !string.Equals(current.Peptide, pep, StringComparison.Ordinal))
            {
                if (current is not null)
                    yield return current;
                current = new PeptideBlock { Peptide = pep };
            }

            current.Ion.Add(reader.IsDBNull(1) ? string.Empty : reader.GetString(1));
            current.PrecursorCharge.Add(FormatKey(reader.GetValue(2)));
            current.ProductCharge.Add(FormatKey(reader.GetValue(3)));
            current.Sample.Add(reader.IsDBNull(4) ? string.Empty : reader.GetString(4));
            current.Area.Add(ToDouble(reader.GetValue(5)));
            current.RetentionTime.Add(ToDouble(reader.GetValue(6)));
        }
        if (current is not null)
            yield return current;
    }

    // Format a charge value for the transition-id key. Only distinctness matters for the
    // transition count, and (ion, prec, prod) -> string is injective, so plain invariant
    // formatting is sufficient.
    private static string FormatKey(object? v) => v switch
    {
        null => "nan",
        DBNull => "nan",
        long l => l.ToString(CultureInfo.InvariantCulture),
        int n => n.ToString(CultureInfo.InvariantCulture),
        short s => s.ToString(CultureInfo.InvariantCulture),
        double d => d.ToString(CultureInfo.InvariantCulture),
        _ => Convert.ToString(v, CultureInfo.InvariantCulture) ?? "nan",
    };

    private static double ToDouble(object? v) => v switch
    {
        null or DBNull => double.NaN,
        double d => d,
        float f => f,
        long l => l,
        int n => n,
        short s => s,
        decimal m => (double)m,
        _ => Convert.ToDouble(v, CultureInfo.InvariantCulture),
    };

    private static string Esc(string path) => path.Replace("'", "''");
}
