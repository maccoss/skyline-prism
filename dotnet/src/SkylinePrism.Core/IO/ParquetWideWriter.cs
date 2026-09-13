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
        await using var fs = await OpenWriteWithRetryAsync(path);
        await using var writer =
            await ParquetWriter.CreateAsync(schema, fs, ParquetColumnIo.Options());

        using var rg = writer.CreateRowGroup();
        var fieldIndex = 0;
        foreach (var mc in metaColumns)
            await ParquetColumnIo.WriteColumnAsync(rg, (DataField)fields[fieldIndex++], mc.Values);
        for (var s = 0; s < sampleNames.Count; s++)
            await ParquetColumnIo.WriteColumnAsync(rg, (DataField)fields[fieldIndex++], sampleColumns[s]);

        _ = rowCount; // row count is implied by the column lengths
    }

    /// <summary>
    /// Open the output file for writing, retrying on transient IO locks. New parquet files in
    /// watched folders (e.g. Downloads) are briefly locked by Windows Defender / the search
    /// indexer / cloud sync; a short backoff clears those.
    ///
    /// <para>A persistent lock throws after the retries. Do not assume it is local: on a share the
    /// holder can be another machine, or a dead session the server has not timed out yet, and the
    /// old message sent people to look in a Downloads folder for a file on a lab drive.</para>
    /// </summary>
    private static async Task<FileStream> OpenWriteWithRetryAsync(
        string path, int maxAttempts = 15, int delayMs = 300)
    {
        IOException? last = null;
        for (var attempt = 1; attempt <= maxAttempts; attempt++)
        {
            try
            {
                // Read, not None. None is refused while ANY other handle is open, so a
                // reader anywhere in this process - the GUI showing the very file the run is
                // updating - blocked the write outright. Sharing Read still excludes a second
                // WRITER, because that would need Write access this handle does not grant.
                return new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
            }
            catch (IOException ex)
            {
                last = ex;
                await Task.Delay(delayMs);
            }
        }
        throw new IOException(
            $"Could not write '{path}' after {maxAttempts} attempts - another process is holding it "
            + "open. On a network share that can be a program on ANOTHER machine, and the lock can "
            + "outlive it: a client that was killed or lost its connection leaves the server holding "
            + "the file until the session times out. Locally it is usually the file open in a viewer, "
            + "cloud sync, or antivirus. Close whatever has it - 'openfiles /query' on the file "
            + "server names the holder - or write to a directory that is not being watched.",
            last);
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
