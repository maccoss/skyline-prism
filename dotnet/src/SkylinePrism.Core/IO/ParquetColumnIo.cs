using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Parquet;
using Parquet.Schema;

namespace SkylinePrism.Core.IO;

/// <summary>
/// The one place that knows how Parquet.Net wants a column handed over, so the four readers and writers
/// do not each carry it.
///
/// <para><b>Why this exists.</b> Parquet.Net 6 removed <c>DataColumn</c>, the wrapper that used to take a
/// bare <see cref="Array"/> and work out its element type internally. Its replacements -
/// <c>WriteAsync&lt;T&gt;(field, ReadOnlyMemory&lt;T&gt;)</c> and
/// <c>ReadAsync&lt;T&gt;(field, Memory&lt;T&gt;)</c> - need T at COMPILE time, while PRISM's meta columns
/// are an <c>IReadOnlyList&lt;Array&gt;</c> whose element types are decided at run time. Something has to
/// dispatch on the runtime type; doing it here keeps it out of four files and gives the closed set of
/// supported types a single home.</para>
///
/// <para><b>Reading returns an <see cref="Array"/> on purpose.</b> That is the shape
/// <c>DataColumn.Data</c> had, so every caller's coercion downstream - null to NaN, int widened to
/// double, null string to empty - is untouched by the upgrade. The migration then cannot alter a value
/// by accident, which matters because these files are the bit-exact quantity contract.</para>
/// </summary>
internal static class ParquetColumnIo
{
    /// <summary>
    /// Element types PRISM actually writes, matching the schema built by the writers' MakeField. A type
    /// outside this set throws rather than being silently coerced: parquet would accept a wrong-typed
    /// column and the error would surface much later as bad data.
    /// </summary>
    private static readonly HashSet<Type> Supported =
        new()
        {
            typeof(string), typeof(long), typeof(bool), typeof(double), typeof(int), typeof(DateTime),
        };

    /// <summary>
    /// Options for every read and write, so the two cannot drift. Compression lives here because v6
    /// moved it off the writer; Snappy, as before. Calling this also guarantees the static constructor
    /// above has run - see its remarks.
    /// </summary>
    internal static ParquetOptions Options() =>
        new() { CompressionMethod = CompressionMethod.Snappy };

    /// <summary>
    /// Write one column, dispatching on the array's runtime element type.
    ///
    /// <para>Nullable value arrays go through the <c>T?</c> overload rather than being flattened here,
    /// because parquet stores the null-ness and a caller that wrote NaN for null would change what the
    /// file means.</para>
    /// </summary>
    internal static async Task WriteColumnAsync(
        ParquetRowGroupWriter rowGroup, DataField field, Array values)
    {
        switch (values)
        {
            case string[] s: await rowGroup.WriteAsync(field, s); return;
            case double[] d: await rowGroup.WriteAsync(field, (ReadOnlyMemory<double>)d); return;
            case long[] l: await rowGroup.WriteAsync(field, (ReadOnlyMemory<long>)l); return;
            case bool[] b: await rowGroup.WriteAsync(field, (ReadOnlyMemory<bool>)b); return;
            case double?[] dn: await rowGroup.WriteAsync(field, (ReadOnlyMemory<double?>)dn); return;
            case long?[] ln: await rowGroup.WriteAsync(field, (ReadOnlyMemory<long?>)ln); return;
            case bool?[] bn: await rowGroup.WriteAsync(field, (ReadOnlyMemory<bool?>)bn); return;
            case int[] i: await rowGroup.WriteAsync(field, (ReadOnlyMemory<int>)i); return;
            case int?[] inn: await rowGroup.WriteAsync(field, (ReadOnlyMemory<int?>)inn); return;
            case DateTime[] t: await rowGroup.WriteAsync(field, (ReadOnlyMemory<DateTime>)t); return;
            case DateTime?[] tn: await rowGroup.WriteAsync(field, (ReadOnlyMemory<DateTime?>)tn); return;
            default:
                throw new NotSupportedException(
                    $"Column '{field.Name}' is {values.GetType().Name}, which ParquetColumnIo does not "
                    + $"write. Supported element types: {string.Join(", ", Supported)} (plus their "
                    + "nullable forms and int). Add the case rather than coercing at the call site.");
        }
    }

    /// <summary>
    /// Read one column of one row group, returning the values as an <see cref="Array"/> in the field's own
    /// CLR type - the shape <c>DataColumn.Data</c> had, so callers coerce exactly as they did before.
    /// </summary>
    internal static async Task<Array> ReadColumnAsync(
        ParquetRowGroupReader rowGroup, DataField field)
    {
        // Taken from the reader rather than a parameter: v6 wants a destination buffer, and a caller
        // passing a count that disagrees with the row group would silently truncate or over-read.
        var rowCount = (int)rowGroup.RowCount;
        var type = field.ClrType;

        // A STRING column reports ClrType as ReadOnlyMemory<char> in Parquet.Net 6, not string - so
        // dispatching on typeof(string) alone silently falls through to the "unsupported" throw, which is
        // how this first surfaced (161 tests, "CLR type ReadOnlyMemory`1"). The non-generic
        // ReadAsync(field, Memory<string?>) overload is still the right way to read one: it handles the
        // definition levels internally, where the generic overload demands an explicit buffer for a
        // nullable field. Verified end to end - plain sequences, modified sequences with parentheses and
        // mass deltas, non-ASCII, empty strings, and nulls kept DISTINCT from empty.
        //
        // typeof(string) is kept alongside it so this keeps working if a later version reports the
        // friendlier type. ParquetOptions.PreferUntypedString looks like the knob for this and is NOT:
        // setting it false leaves ClrType exactly as it was. Measured, not assumed.
        if (type == typeof(string) || type == typeof(ReadOnlyMemory<char>))
        {
            var buffer = new string?[rowCount];
            await rowGroup.ReadAsync(field, buffer.AsMemory());
            return buffer;
        }
        if (type == typeof(double)) return await ReadValuesAsync<double>(rowGroup, field, rowCount);
        if (type == typeof(long)) return await ReadValuesAsync<long>(rowGroup, field, rowCount);
        if (type == typeof(int)) return await ReadValuesAsync<int>(rowGroup, field, rowCount);
        if (type == typeof(bool)) return await ReadValuesAsync<bool>(rowGroup, field, rowCount);
        if (type == typeof(float)) return await ReadValuesAsync<float>(rowGroup, field, rowCount);
        // 'Acquired Time' in a Skyline report. The full set present across the 117 committed fixture
        // parquets is double / string / int64 / bool / int32 / timestamp, so this completes it - taken
        // from the files rather than added one exception at a time.
        if (type == typeof(DateTime)) return await ReadValuesAsync<DateTime>(rowGroup, field, rowCount);

        throw new NotSupportedException(
            $"Column '{field.Name}' has CLR type {type.Name}, which ParquetColumnIo does not read. "
            + "Add the case rather than guessing at the call site.");
    }

    /// <summary>
    /// A nullable field is read into <c>T?[]</c> and a non-nullable one into <c>T[]</c>, because the
    /// overloads are distinct and reading a nullable column into the non-nullable buffer loses the
    /// null-ness that callers turn into NaN.
    /// </summary>
    private static async Task<Array> ReadValuesAsync<T>(
        ParquetRowGroupReader rowGroup, DataField field, int rowCount)
        where T : struct
    {
        if (field.IsNullable)
        {
            var nullable = new T?[rowCount];
            await rowGroup.ReadAsync(field, nullable.AsMemory());
            return nullable;
        }
        var plain = new T[rowCount];
        await rowGroup.ReadAsync(field, plain.AsMemory());
        return plain;
    }
}
