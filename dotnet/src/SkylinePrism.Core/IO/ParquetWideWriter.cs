using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Parquet;
using Parquet.Data;
using Parquet.Schema;

namespace SkylinePrism.Core.IO;

/// <summary>
/// Writes wide feature x sample matrices to parquet (Parquet.Net), matching the schemas
/// the Python pipeline emits. A wide table is a set of leading metadata columns followed
/// by one float64 column per sample.
/// </summary>
public static class ParquetWideWriter
{
    /// <summary>A metadata (non-sample) column: name + typed values (string[]/long[]/double[]/bool[]).</summary>
    public sealed record MetaColumn(string Name, Array Values, Type ElementType);

    public static MetaColumn Strings(string name, string[] values) => new(name, values, typeof(string));
    public static MetaColumn Longs(string name, long[] values) => new(name, values, typeof(long));
    public static MetaColumn Doubles(string name, double[] values) => new(name, values, typeof(double));
    public static MetaColumn Bools(string name, bool[] values) => new(name, values, typeof(bool));

    /// <summary>
    /// Write a wide table. <paramref name="metaColumns"/> are the leading columns (each with
    /// <paramref name="rowCount"/> values); <paramref name="sampleNames"/> are the trailing
    /// float64 columns whose values are <paramref name="sampleColumns"/>[sampleIndex][row].
    /// </summary>
    public static void Write(
        string path,
        IReadOnlyList<MetaColumn> metaColumns,
        IReadOnlyList<string> sampleNames,
        IReadOnlyList<double[]> sampleColumns,
        int rowCount)
        => WriteAsync(path, metaColumns, sampleNames, sampleColumns, rowCount).GetAwaiter().GetResult();

    public static async Task WriteAsync(
        string path,
        IReadOnlyList<MetaColumn> metaColumns,
        IReadOnlyList<string> sampleNames,
        IReadOnlyList<double[]> sampleColumns,
        int rowCount)
    {
        var fields = new List<Field>(metaColumns.Count + sampleNames.Count);
        foreach (var mc in metaColumns)
            fields.Add(MakeField(mc.Name, mc.ElementType));
        foreach (var s in sampleNames)
            fields.Add(new DataField<double>(s));

        var schema = new ParquetSchema(fields);

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        await using var fs = File.Create(path);
        using var writer = await ParquetWriter.CreateAsync(schema, fs);
        writer.CompressionMethod = CompressionMethod.Snappy;

        using var rg = writer.CreateRowGroup();
        var fieldIndex = 0;
        foreach (var mc in metaColumns)
            await rg.WriteColumnAsync(new DataColumn((DataField)fields[fieldIndex++], mc.Values));
        for (var s = 0; s < sampleNames.Count; s++)
            await rg.WriteColumnAsync(new DataColumn((DataField)fields[fieldIndex++], sampleColumns[s]));

        _ = rowCount; // row count is implied by the column lengths
    }

    private static Field MakeField(string name, Type elementType)
    {
        if (elementType == typeof(string)) return new DataField<string>(name);
        if (elementType == typeof(long)) return new DataField<long>(name);
        if (elementType == typeof(double)) return new DataField<double>(name);
        if (elementType == typeof(bool)) return new DataField<bool>(name);
        throw new NotSupportedException($"Unsupported meta column type {elementType}");
    }
}
