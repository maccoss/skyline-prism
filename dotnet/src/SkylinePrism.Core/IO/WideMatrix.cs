using System;
using System.Collections.Generic;
using System.Linq;

namespace SkylinePrism.Core.IO;

/// <summary>
/// One of PRISM's wide matrices (features x samples) held in memory so a later stage can adjust it and
/// write it back, metadata columns intact.
///
/// <para>Existing readers take what they need and drop the rest, which is right for reporting and
/// wrong for rewriting: a marker normalization computed after both arms have finished has to reproduce
/// the file it edits, not a projection of it. The metadata columns are read by declared name and kept
/// with their original types, so a peptide matrix comes back with its protein-group columns and a
/// protein matrix with its counts and flags.</para>
///
/// <para>Whole-matrix, so it is for the small end of the pipeline - the corrected outputs, not the
/// merged transition table. It allocates one double per cell plus the metadata.</para>
/// </summary>
public sealed class WideMatrix
{
    private readonly List<ParquetWideWriter.MetaColumn> _meta;

    private WideMatrix(
        List<ParquetWideWriter.MetaColumn> meta, List<string> samples, double[,] values)
    {
        _meta = meta;
        Samples = samples;
        Values = values;
    }

    /// <summary>[feature, sample], in the file's own order.</summary>
    public double[,] Values { get; }

    public IReadOnlyList<string> Samples { get; }

    public int RowCount => Values.GetLength(0);

    public int SampleCount => Values.GetLength(1);

    /// <summary>
    /// Read a wide parquet. <paramref name="metaNames"/> are the non-sample columns to preserve; any
    /// other non-numeric column is preserved too (a string column is never a replicate), and every
    /// remaining numeric column is treated as a sample.
    /// </summary>
    public static WideMatrix Read(string path, IReadOnlyList<string> metaNames)
    {
        using var reader = ParquetColumnReader.Open(path);
        var declared = new HashSet<string>(metaNames, StringComparer.Ordinal);

        var meta = new List<ParquetWideWriter.MetaColumn>();
        var samples = new List<string>();
        foreach (var name in reader.ColumnNames)
        {
            if (declared.Contains(name) || !reader.IsNumericColumn(name))
                meta.Add(ReadMeta(reader, name));
            else
                samples.Add(name);
        }

        var values = new double[reader.RowCount, samples.Count];
        for (var j = 0; j < samples.Count; j++)
        {
            var col = reader.ReadDoubles(samples[j]);
            for (var i = 0; i < reader.RowCount; i++)
                values[i, j] = col[i];
        }
        return new WideMatrix(meta, samples, values);
    }

    private static ParquetWideWriter.MetaColumn ReadMeta(ParquetColumnReader reader, string name)
    {
        if (!reader.IsNumericColumn(name))
            return ParquetWideWriter.Strings(name, reader.ReadStrings(name));

        // A numeric metadata column (n_peptides, n_transitions, mean_rt): keep whole counts whole
        // rather than turning them into floats on the way through.
        var values = reader.ReadDoubles(name);
        var whole = values.All(v => double.IsNaN(v) || v == Math.Floor(v));
        return whole
            ? ParquetWideWriter.Longs(name, values.Select(v => double.IsNaN(v) ? 0L : (long)v).ToArray())
            : ParquetWideWriter.Doubles(name, values);
    }

    /// <summary>A preserved string metadata column, or null when the file has no such column.</summary>
    public string[]? MetaStrings(string name) =>
        _meta.FirstOrDefault(m => string.Equals(m.Name, name, StringComparison.Ordinal))?.Values
            as string[];

    /// <summary>Add or replace a boolean metadata column, appended after the existing metadata.</summary>
    public void SetFlag(string name, bool[] values)
    {
        if (values.Length != RowCount)
            throw new ArgumentException(
                $"Flag '{name}' has {values.Length} values but the matrix has {RowCount} rows.",
                nameof(values));
        _meta.RemoveAll(m => string.Equals(m.Name, name, StringComparison.Ordinal));
        _meta.Add(ParquetWideWriter.Bools(name, values));
    }

    /// <summary>Write the matrix back, metadata columns first, in the order they are held.</summary>
    public void Write(string path)
    {
        var columns = new List<double[]>(SampleCount);
        for (var j = 0; j < SampleCount; j++)
        {
            var col = new double[RowCount];
            for (var i = 0; i < RowCount; i++)
                col[i] = Values[i, j];
            columns.Add(col);
        }
        ParquetWideWriter.Write(path, _meta, Samples, columns, RowCount);
    }
}
