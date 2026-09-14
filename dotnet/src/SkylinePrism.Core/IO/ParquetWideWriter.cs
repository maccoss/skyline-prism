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
    /// Add one row group to <paramref name="path"/>, creating the file if it is not there yet.
    /// </summary>
    /// <remarks>
    /// <para><b>For output that accumulates over a long run.</b> Writing the whole table again after
    /// every unit of work is O(n^2) in bytes: measured on a real 48-replicate ion accounting cache,
    /// 883 MB written to persist 36 MB, and about 92 GB projected to persist 376 MB at 500
    /// replicates. Appending writes each row once.</para>
    ///
    /// <para><b>The file is complete after every append.</b> Parquet.Net rewrites the footer on
    /// close, so a run killed between appends leaves a valid file holding everything up to the last
    /// one - which is the property that matters here, and is why this is worth more than the bytes.
    /// Verified over SMB: 500 appends, each closing and reopening the file, readable with the
    /// correct cumulative row count throughout.</para>
    ///
    /// <para><b>The schema must match</b> what is already in the file - same names, same types, same
    /// order. Appending a different shape is a defect, and parquet will not detect it for you.</para>
    ///
    /// <para>NOT thread-safe, deliberately: two appends at once would interleave footers. The caller
    /// serializes, which <c>IonAccountingRun</c> already does for its progress saves.</para>
    /// </remarks>
    /// <param name="replace">
    /// Start the file over rather than adding to it. The truncation happens in the OPEN, so it
    /// cannot half-succeed: there is no delete-then-open window in which a holder releases and the
    /// new rows land on top of an older measurement carrying a different settings key.
    /// </param>
    public static void Append(
        string path, IReadOnlyList<MetaColumn> metaColumns, bool replace = false) =>
        AppendAsync(path, metaColumns, replace).GetAwaiter().GetResult();

    /// <inheritdoc cref="Append"/>
    public static async Task AppendAsync(
        string path, IReadOnlyList<MetaColumn> metaColumns, bool replace = false)
    {
        var fields = new List<Field>(metaColumns.Count);
        foreach (var mc in metaColumns)
            fields.Add(MakeField(mc.Name, mc.ElementType));
        var schema = new ParquetSchema(fields);

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);

        // Appendable means the file HAS A FOOTER, not merely that it is there. A first append killed
        // between creating the file and flushing leaves zero bytes, and asking parquet to append to
        // that throws - once per unit of work, for the rest of the run, with nothing ever repairing
        // it. Starting such a file over loses nothing, because there is nothing in it.
        var info = new FileInfo(path);
        var mightAppend = !replace && info.Exists && info.Length > 0;

        // Appending needs the existing footer read back, so the stream is ReadWrite rather than the
        // write-only one Write uses. Sharing Read for the same reason every writer here does: a
        // reader must be able to look while a run is in progress - see ParquetColumnIo.OpenRead.
        await using (var fs = await OpenAppendWithRetryAsync(path, mightAppend))
        {
            // Decided from the OPEN file, never from the stat above: between that stat and this open
            // the file can be replaced or emptied, and asking parquet to append to a file with no
            // footer throws.
            var append = !replace && fs.Length > 0;

            // An append is NOT atomic, and the file it damages is the whole file rather than the row
            // group being written. Parquet keeps its metadata at the END, so appending overwrites
            // the existing footer with the new row group and writes a fresh footer after it; between
            // those two the file has no footer at all, and a reader gets nothing - not "everything
            // except the replicate in flight", nothing. Measured: truncating a four-replicate file
            // at the footer offset, which is exactly what the first write of an append does, takes
            // ReadCycles from 4,000 rows to 0.
            //
            // So the bytes about to be overwritten are kept first, and TryRepair puts them back. The
            // footer is ~1,577 bytes per row group (measured across 8 to 120 groups, linear), so the
            // whole protection costs about 188 MB over a 500-replicate run against 376 MB of data -
            // as against the 92 GB that rewriting the table each time would have cost.
            //
            // Read through the stream this append OWNS, rather than reopening the path: taken before
            // the open, the copy describes a state another writer can leave behind in between, and a
            // later repair would then truncate away a row group it never saw.
            if (append)
                SaveFooter(fs, FooterBackupOf(path));
            else
                TryDelete(FooterBackupOf(path));

            await using var writer = await ParquetWriter.CreateAsync(
                schema, fs, ParquetColumnIo.Options(), append: append);

            using var rg = writer.CreateRowGroup();
            for (var i = 0; i < metaColumns.Count; i++)
                await ParquetColumnIo.WriteColumnAsync(rg, (DataField)fields[i], metaColumns[i].Values);
        }
    }

    /// <summary>Where the bytes an append is about to overwrite are kept.</summary>
    public static string FooterBackupOf(string path) => path + ".footer";

    /// <summary>
    /// Copy the footer of an existing parquet file - the region the next append overwrites - beside
    /// it, as [8-byte offset][footer bytes].
    /// </summary>
    /// <remarks>
    /// Silent on anything unexpected: this is protection, and failing to take it is not a reason to
    /// refuse to write. A torn backup is harmless on its own - it is only ever read when the file it
    /// describes will not parse, and <see cref="TryRepair"/> verifies the result.
    /// </remarks>
    private static void SaveFooter(FileStream fs, string backup)
    {
        var resume = fs.Position;
        try
        {
            if (fs.Length < 12)
                return;

            var tail = new byte[8];
            fs.Seek(-8, SeekOrigin.End);
            fs.ReadExactly(tail);
            if (tail[4] != (byte)'P' || tail[5] != (byte)'A' || tail[6] != (byte)'R' || tail[7] != (byte)'1')
                return;

            var footerLength = BitConverter.ToInt32(tail, 0);
            var offset = fs.Length - 8 - footerLength;
            if (footerLength <= 0 || offset < 4)
                return;

            var blob = new byte[8 + footerLength + 8];
            BitConverter.TryWriteBytes(blob.AsSpan(0, 8), offset);
            fs.Seek(offset, SeekOrigin.Begin);
            fs.ReadExactly(blob.AsSpan(8));
            File.WriteAllBytes(backup, blob);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
        }
        finally
        {
            // Handed straight to ParquetWriter next, which expects to decide its own position. Its
            // own failure must never replace the one on its way out: a dropped share faults the read
            // above AND this seek, and the seek is the less useful of the two diagnoses.
            try
            {
                fs.Seek(resume, SeekOrigin.Begin);
            }
            catch (Exception ex) when (ex is IOException or ObjectDisposedException
                                           or NotSupportedException)
            {
            }
        }
    }

    /// <summary>
    /// Put back the footer an interrupted append destroyed, returning true only if the file parses
    /// afterwards.
    /// </summary>
    /// <remarks>
    /// <para>What this recovers is everything written before the interrupted append - the row groups
    /// themselves are untouched, and only the metadata that describes them was overwritten. The
    /// replicate that was in flight is lost, which is right: it was never finished.</para>
    ///
    /// <para>Refuses while a writer is live (it opens exclusively), so a run in progress is never
    /// rewound by a reader that happened to look during an append.</para>
    /// </remarks>
    public static bool TryRepair(string path, Func<string, bool> parses)
    {
        var backup = FooterBackupOf(path);
        if (!File.Exists(path) || !File.Exists(backup))
            return false;
        try
        {
            var blob = File.ReadAllBytes(backup);
            if (!IsWellFormedFooter(blob))
                return false;
            var offset = BitConverter.ToInt64(blob, 0);
            if (offset < 4 || offset > new FileInfo(path).Length)
                return false;

            using (var fs = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None))
            {
                fs.SetLength(offset);
                fs.Seek(offset, SeekOrigin.Begin);
                fs.Write(blob, 8, blob.Length - 8);
            }
            return parses(path);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>
    /// Whether a saved footer is shaped like one, before it is written over a file.
    /// </summary>
    /// <remarks>
    /// The repair truncates to the recorded offset, so acting on a backup that was itself torn -
    /// <see cref="SaveFooter"/> writes it in one go, but a machine can stop mid-write - would
    /// discard bytes in exchange for a footer that describes nothing. The blob is
    /// [8-byte offset][metadata][4-byte length][PAR1], so its own declared length has to account for
    /// exactly what is there.
    /// </remarks>
    private static bool IsWellFormedFooter(byte[] blob)
    {
        if (blob.Length < 24)
            return false;
        if (blob[^4] != (byte)'P' || blob[^3] != (byte)'A' || blob[^2] != (byte)'R' || blob[^1] != (byte)'1')
            return false;
        return BitConverter.ToInt32(blob, blob.Length - 8) == blob.Length - 16;
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
                File.Delete(path);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
        }
    }

    /// <inheritdoc cref="OpenWriteWithRetryAsync"/>
    private static async Task<FileStream> OpenAppendWithRetryAsync(
        string path, bool exists, int maxAttempts = 15, int delayMs = 300)
    {
        IOException? last = null;
        for (var attempt = 1; attempt <= maxAttempts; attempt++)
        {
            try
            {
                // Create truncates an existing file in the same operation that opens it, which is
                // what makes "start over" safe: a separate delete can be refused and then succeed a
                // moment later, leaving the next open to find the file and append to it.
                //
                // OpenOrCreate rather than Open for the other case, because `exists` came from a stat
                // that is already stale: a file deleted in between would make Open throw
                // FileNotFoundException - an IOException, so the retry would sit on it for 4.5
                // seconds and then report it as another process holding the file, sending the next
                // investigation to the wrong place. Creating it instead lets the caller's length
                // check decide correctly that there is nothing to append to.
                return new FileStream(
                    path, exists ? FileMode.OpenOrCreate : FileMode.Create,
                    FileAccess.ReadWrite, FileShare.Read);
            }
            catch (IOException ex)
            {
                last = ex;
                await Task.Delay(delayMs);
            }
        }
        throw new IOException(
            $"Could not append to '{path}' after {maxAttempts} attempts - another process is holding "
            + "it open. On a network share that can be a program on ANOTHER machine, and the lock can "
            + "outlive it: a client that was killed or lost its connection leaves the server holding "
            + "the file until the session times out.",
            last);
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
