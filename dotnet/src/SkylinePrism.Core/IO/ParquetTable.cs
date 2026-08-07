using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Parquet;
using Parquet.Data;
using Parquet.Schema;

namespace SkylinePrism.Core.IO;

/// <summary>
/// A simple in-memory columnar table read from a parquet file via Parquet.Net. Columns
/// are kept as their raw arrays (which may be nullable value-type arrays). Helper
/// accessors coerce to double / string for parity comparisons and downstream use.
/// </summary>
public sealed class ParquetTable
{
    private readonly Dictionary<string, Array> _columns;

    public IReadOnlyList<string> ColumnNames { get; }
    public int RowCount { get; }

    private ParquetTable(List<string> names, Dictionary<string, Array> columns, int rowCount)
    {
        ColumnNames = names;
        _columns = columns;
        RowCount = rowCount;
    }

    public bool HasColumn(string name) => _columns.ContainsKey(name);

    /// <summary>
    /// Drop columns that are no longer needed so their arrays can be collected, without releasing the
    /// whole table. The wide matrices are the memory wall of the pipeline: a corrected-peptide stage on a
    /// large cohort holds the loaded sample columns AND the working matrices at the same time, and the
    /// loaded columns are the bigger of the two (parquet nullable columns arrive as double?[], 16 bytes
    /// per cell against the matrix's 8). Once the values have been copied into the working matrix nothing
    /// reads them again.
    /// <para><see cref="ColumnNames"/> is left untouched, so it still describes the file that was read.
    /// Reading a released column throws <see cref="KeyNotFoundException"/>, same as an absent one.</para>
    /// </summary>
    public void ReleaseColumns(IEnumerable<string> names)
    {
        foreach (var name in names)
            _columns.Remove(name);
    }

    public static ParquetTable Load(string path)
        => LoadAsync(path).GetAwaiter().GetResult();

    /// <summary>
    /// Read only the column names from the parquet footer/schema - no row data is materialized.
    /// Use this for column detection on large files (e.g. the merged report) instead of Load,
    /// which pulls every column x row into memory.
    /// </summary>
    public static IReadOnlyList<string> ReadColumnNames(string path)
    {
        using var fs = File.OpenRead(path);
        using var reader = ParquetReader.CreateAsync(fs).GetAwaiter().GetResult();
        return reader.Schema.DataFields.Select(f => f.Name).ToList();
    }

    public static async Task<ParquetTable> LoadAsync(string path)
    {
        await using var fs = File.OpenRead(path);
        using var reader = await ParquetReader.CreateAsync(fs);
        var dataFields = reader.Schema.DataFields;

        // Accumulate each field's data across all row groups.
        var accum = dataFields.ToDictionary(f => f.Name, _ => new List<Array>());
        for (var rg = 0; rg < reader.RowGroupCount; rg++)
        {
            using var rgReader = reader.OpenRowGroupReader(rg);
            foreach (var field in dataFields)
            {
                var col = await rgReader.ReadColumnAsync(field);
                accum[field.Name].Add(col.Data);
            }
        }

        var columns = new Dictionary<string, Array>();
        var rowCount = 0;
        foreach (var field in dataFields)
        {
            var merged = Concat(accum[field.Name]);
            columns[field.Name] = merged;
            rowCount = merged.Length;
        }

        return new ParquetTable(dataFields.Select(f => f.Name).ToList(), columns, rowCount);
    }

    private static Array Concat(List<Array> parts)
    {
        if (parts.Count == 1)
            return parts[0];
        var total = parts.Sum(p => p.Length);
        var elementType = parts[0].GetType().GetElementType()!;
        var result = Array.CreateInstance(elementType, total);
        var offset = 0;
        foreach (var part in parts)
        {
            Array.Copy(part, 0, result, offset, part.Length);
            offset += part.Length;
        }
        return result;
    }

    /// <summary>Raw column array (may be a nullable value-type array or string[]).</summary>
    public Array Column(string name) =>
        _columns.TryGetValue(name, out var arr)
            ? arr
            : throw new KeyNotFoundException($"Column '{name}' not found. Available: {string.Join(", ", ColumnNames)}");

    /// <summary>Column coerced to double?[] (null preserved), from any numeric source type.</summary>
    public double?[] GetDouble(string name)
    {
        var arr = Column(name);
        var result = new double?[arr.Length];
        for (var i = 0; i < arr.Length; i++)
            result[i] = CoerceDouble(arr.GetValue(i));
        return result;
    }

    /// <summary>
    /// The scalar coercions the Get* accessors apply, exposed so a streaming reader that never
    /// materializes a <see cref="ParquetTable"/> converts metadata cells identically - a meta column
    /// that differs between the two paths shows up as a corrupt output file, not a rounding error.
    /// </summary>
    internal static double? CoerceDouble(object? v) => v switch
    {
        null => (double?)null,
        double d => d,
        float f => f,
        long l => l,
        int n => n,
        short s => s,
        decimal m => (double)m,
        _ => Convert.ToDouble(v),
    };

    internal static long CoerceLong(object? v)
    {
        var d = CoerceDouble(v);
        return d.HasValue ? (long)d.Value : 0L;
    }

    internal static bool CoerceBool(object? v) => v switch
    {
        null => false,
        bool b => b,
        _ => Convert.ToBoolean(v),
    };

    /// <summary>Column coerced to long[] (nulls -&gt; 0), from any integer/float source.</summary>
    public long[] GetLong(string name)
    {
        var arr = Column(name);
        var result = new long[arr.Length];
        for (var i = 0; i < arr.Length; i++)
            result[i] = CoerceLong(arr.GetValue(i));
        return result;
    }

    /// <summary>Column coerced to bool[] (nulls -&gt; false).</summary>
    public bool[] GetBool(string name)
    {
        var arr = Column(name);
        var result = new bool[arr.Length];
        for (var i = 0; i < arr.Length; i++)
            result[i] = CoerceBool(arr.GetValue(i));
        return result;
    }

    /// <summary>Column coerced to string?[] (null preserved).</summary>
    public string?[] GetString(string name)
    {
        var arr = Column(name);
        var result = new string?[arr.Length];
        for (var i = 0; i < arr.Length; i++)
        {
            var v = arr.GetValue(i);
            result[i] = v?.ToString();
        }
        return result;
    }
}
