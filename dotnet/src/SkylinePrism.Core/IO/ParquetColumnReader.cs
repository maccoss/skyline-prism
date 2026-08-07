using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Parquet;
using Parquet.Schema;

namespace SkylinePrism.Core.IO;

/// <summary>
/// Reads a wide parquet file a piece at a time instead of all at once, which is what
/// <see cref="ParquetTable.Load"/> does. Two access patterns, both bounded:
/// <list type="bullet">
/// <item>ONE column across every row group (<see cref="ReadDoubles(string)"/>) - O(rows), the
/// orientation per-sample statistics want (medians, LOWESS curves).</item>
/// <item>ONE row group across many columns (<see cref="OpenRowGroup"/>) - O(rowGroupRows x columns),
/// the orientation row-wise arithmetic wants.</item>
/// </list>
/// A feature x sample matrix is the pipeline's memory wall, and parquet is columnar, so neither
/// pattern ever needs the whole matrix resident.
/// <para>Not thread-safe: the underlying reader holds one file stream.</para>
/// </summary>
internal sealed class ParquetColumnReader : IDisposable
{
    private readonly FileStream _fs;
    private readonly ParquetReader _reader;
    private readonly Dictionary<string, DataField> _fields;

    private ParquetColumnReader(FileStream fs, ParquetReader reader)
    {
        _fs = fs;
        _reader = reader;
        _fields = reader.Schema.DataFields.ToDictionary(f => f.Name, f => f, StringComparer.Ordinal);
        ColumnNames = reader.Schema.DataFields.Select(f => f.Name).ToList();
        RowCount = 0;
        for (var rg = 0; rg < reader.RowGroupCount; rg++)
        {
            using var rgReader = reader.OpenRowGroupReader(rg);
            RowCount += (int)rgReader.RowCount;
        }
    }

    public IReadOnlyList<string> ColumnNames { get; }

    /// <summary>Total rows across all row groups.</summary>
    public int RowCount { get; }

    public int RowGroupCount => _reader.RowGroupCount;

    public static ParquetColumnReader Open(string path)
    {
        var fs = File.OpenRead(path);
        try
        {
            var reader = ParquetReader.CreateAsync(fs).GetAwaiter().GetResult();
            return new ParquetColumnReader(fs, reader);
        }
        catch
        {
            fs.Dispose();
            throw;
        }
    }

    public bool HasColumn(string name) => _fields.ContainsKey(name);

    /// <summary>One whole column as doubles (null -&gt; NaN), concatenated across row groups.</summary>
    public double[] ReadDoubles(string name)
    {
        var field = Field(name);
        var result = new double[RowCount];
        var offset = 0;
        for (var rg = 0; rg < _reader.RowGroupCount; rg++)
        {
            using var rgReader = _reader.OpenRowGroupReader(rg);
            var data = rgReader.ReadColumnAsync(field).GetAwaiter().GetResult().Data;
            CopyAsDoubles(data, result, offset);
            offset += data.Length;
        }
        return result;
    }

    /// <summary>A row group's columns, read on demand. Dispose before opening the next one.</summary>
    public RowGroup OpenRowGroup(int index) => new(_reader.OpenRowGroupReader(index), this);

    public void Dispose()
    {
        _reader.Dispose();
        _fs.Dispose();
    }

    private DataField Field(string name) =>
        _fields.TryGetValue(name, out var f)
            ? f
            : throw new KeyNotFoundException(
                $"Column '{name}' not found. Available: {string.Join(", ", ColumnNames)}");

    /// <summary>Coerce a parquet column chunk into a caller-owned double buffer (null -&gt; NaN).</summary>
    private static void CopyAsDoubles(Array data, double[] destination, int offset)
    {
        // The common case: the writer emitted non-nullable doubles, so this is a memcpy.
        if (data is double[] fast)
        {
            Array.Copy(fast, 0, destination, offset, fast.Length);
            return;
        }
        for (var i = 0; i < data.Length; i++)
        {
            var v = data.GetValue(i);
            destination[offset + i] = v switch
            {
                null => double.NaN,
                double d => d,
                float f => f,
                long l => l,
                int n => n,
                short s => s,
                decimal m => (double)m,
                _ => Convert.ToDouble(v),
            };
        }
    }

    /// <summary>One row group, from which individual columns can be pulled.</summary>
    internal sealed class RowGroup : IDisposable
    {
        private readonly ParquetRowGroupReader _reader;
        private readonly ParquetColumnReader _owner;

        internal RowGroup(ParquetRowGroupReader reader, ParquetColumnReader owner)
        {
            _reader = reader;
            _owner = owner;
            RowCount = (int)reader.RowCount;
        }

        public int RowCount { get; }

        /// <summary>This row group's slice of a column, as doubles (null -&gt; NaN).</summary>
        public double[] ReadDoubles(string name)
        {
            var data = ReadRaw(name);
            var result = new double[data.Length];
            CopyAsDoubles(data, result, 0);
            return result;
        }

        /// <summary>This row group's slice of a column, in its stored type (string[], long[], ...).</summary>
        public Array ReadRaw(string name)
            => _reader.ReadColumnAsync(_owner.Field(name)).GetAwaiter().GetResult().Data;

        public void Dispose() => _reader.Dispose();
    }
}
