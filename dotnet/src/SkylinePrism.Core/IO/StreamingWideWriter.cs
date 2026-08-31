using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using Parquet;
using Parquet.Data;
using Parquet.Schema;

namespace SkylinePrism.Core.IO;

/// <summary>
/// Incremental wide-parquet writer: opens the file once and appends one row group per
/// <see cref="WriteRowGroup"/> call, so producers can flush batches to disk with bounded memory
/// instead of accumulating every row and writing once. Snappy-compressed, matching the Stage-2
/// streaming writers. Not thread-safe - drive it from a single writer thread.
/// </summary>
public sealed class StreamingWideWriter : IDisposable
{
    private readonly FileStream _fs;
    private readonly ParquetWriter _writer;
    private readonly List<Field> _fields;

    private StreamingWideWriter(FileStream fs, ParquetWriter writer, List<Field> fields)
    {
        _fs = fs;
        _writer = writer;
        _fields = fields;
    }

    public static StreamingWideWriter Create(
        string path,
        IReadOnlyList<(string Name, Type ElementType)> metaColumns,
        IReadOnlyList<string> sampleNames)
    {
        var fields = new List<Field>(metaColumns.Count + sampleNames.Count);
        foreach (var (name, elementType) in metaColumns)
            fields.Add(MakeField(name, elementType));
        foreach (var s in sampleNames)
            fields.Add(new DataField<double>(s));

        var schema = new ParquetSchema(fields);
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        var fs = OpenWriteWithRetry(path);
        // Compression moved from a writer property to the options in Parquet.Net 6.
        var writer = ParquetWriter
            .CreateAsync(schema, fs, ParquetColumnIo.Options())
            .GetAwaiter().GetResult();
        return new StreamingWideWriter(fs, writer, fields);
    }

    /// <summary>Append one row group. Meta arrays + sample columns must all share the row count.</summary>
    public void WriteRowGroup(IReadOnlyList<Array> metaColumnData, IReadOnlyList<double[]> sampleColumnData)
    {
        var rowCount = metaColumnData.Count > 0 ? metaColumnData[0].Length
            : sampleColumnData.Count > 0 ? sampleColumnData[0].Length : 0;
        if (rowCount == 0)
            return;

        using var rg = _writer.CreateRowGroup();
        var idx = 0;
        foreach (var arr in metaColumnData)
            ParquetColumnIo.WriteColumnAsync((ParquetRowGroupWriter)rg, (DataField)_fields[idx++], arr)
                .GetAwaiter().GetResult();
        foreach (var col in sampleColumnData)
            ParquetColumnIo.WriteColumnAsync((ParquetRowGroupWriter)rg, (DataField)_fields[idx++], col)
                .GetAwaiter().GetResult();
    }

    public void Dispose()
    {
        // ParquetWriter is IAsyncDisposable only in Parquet.Net 6; this class presents a sync facade to
        // callers that are themselves sync, so the disposal is blocked on here rather than pushed onto
        // them. The writer's own row-group writer is still sync-disposable.
        _writer.DisposeAsync().AsTask().GetAwaiter().GetResult();
        _fs.Dispose();
    }

    private static Field MakeField(string name, Type t)
    {
        if (t == typeof(string)) return new DataField<string>(name);
        if (t == typeof(long)) return new DataField<long>(name);
        if (t == typeof(bool)) return new DataField<bool>(name);
        return new DataField<double>(name);
    }

    // New parquet files in watched folders (Downloads, cloud sync) get briefly locked; retry.
    private static FileStream OpenWriteWithRetry(string path, int maxAttempts = 15, int delayMs = 300)
    {
        IOException? last = null;
        for (var attempt = 1; attempt <= maxAttempts; attempt++)
        {
            try
            {
                return new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
            }
            catch (IOException ex)
            {
                last = ex;
                Thread.Sleep(delayMs);
            }
        }
        throw new IOException(
            $"Could not write '{path}' after {maxAttempts} attempts - it is locked by another process "
            + "(antivirus, cloud sync, or open in a viewer).", last);
    }
}
