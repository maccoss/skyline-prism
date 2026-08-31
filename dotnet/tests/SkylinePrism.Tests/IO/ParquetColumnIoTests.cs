using System;
using System.Collections.Generic;
using System.IO;
using SkylinePrism.Core.IO;
using Xunit;

namespace SkylinePrism.Tests.IO;

/// <summary>
/// The parquet IO shim, and specifically the trap that Parquet.Net 6 sets for it.
///
/// <para><b>Why a string test earns its place.</b> In v6 a string column reports its
/// <c>DataField.ClrType</c> as <c>ReadOnlyMemory&lt;char&gt;</c>, not <c>string</c>. A reader that
/// dispatches on <c>typeof(string)</c> therefore misses every string column in PRISM - peptide sequences,
/// protein groups, sample names - and the migration first surfaced exactly that way, taking out 161 tests
/// at once. It failed loudly only because the shim throws on an unrecognised type; a reader that fell back
/// to something permissive would have written undecoded UTF-8 into the outputs and passed.</para>
///
/// <para>So this asserts on VALUES through a real write-then-read cycle, not on the type mapping - the
/// mapping is Parquet.Net's business and may change again. The cases are the ones PRISM's data actually
/// contains, including the distinction PRISM cares about most: a null is not an empty string.</para>
/// </summary>
public class ParquetColumnIoTests
{
    private static string TempPath()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_pqio_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return Path.Combine(dir, "columns.parquet");
    }

    /// <summary>
    /// Strings survive a round trip intact - the check that would have caught the v6 default the moment
    /// it was introduced. Modified-sequence notation is included in both flavours PRISM sees, because
    /// those are the values a byte-vs-string mix-up would corrupt most visibly.
    /// </summary>
    [Fact]
    public void StringColumnsRoundTripThroughTheShim()
    {
        var path = TempPath();
        var peptides = new[]
        {
            "PEPTIDEK",
            "C(unimod:4)HEMK",      // Skyline export notation
            "M[+15.99491]AGIC",     // BLIB notation
            "naive-with-accent-naïve",
            "",                     // empty, which must stay distinct from null
        };
        var values = new[] { 1.5, 2.5, 3.5, 4.5, 5.5 };

        using (var writer = StreamingWideWriter.Create(
            path,
            new List<(string, Type)> { ("peptide", typeof(string)) },
            new List<string> { "sampleA" }))
        {
            writer.WriteRowGroup(new List<Array> { peptides }, new List<double[]> { values });
        }

        using var reader = ParquetColumnReader.Open(path);
        var readPeptides = reader.ReadStrings("peptide");
        var readValues = reader.ReadDoubles("sampleA");

        Assert.Equal(peptides, readPeptides);
        Assert.Equal(values, readValues);
    }

    /// <summary>
    /// The other half of the same concern: numbers must not be quietly reshaped either. NaN is the value
    /// PRISM uses for "no data" (Skyline exports are dense, so it is the rare case that matters), and it
    /// has to survive as NaN rather than becoming zero.
    /// </summary>
    [Fact]
    public void DoubleColumnsRoundTripIncludingNaN()
    {
        var path = TempPath();
        var labels = new[] { "a", "b", "c", "d" };
        var values = new[] { 0.0, double.NaN, -1234.5678901234, 1e300 };

        using (var writer = StreamingWideWriter.Create(
            path,
            new List<(string, Type)> { ("label", typeof(string)) },
            new List<string> { "s1" }))
        {
            writer.WriteRowGroup(new List<Array> { labels }, new List<double[]> { values });
        }

        using var reader = ParquetColumnReader.Open(path);
        var read = reader.ReadDoubles("s1");

        Assert.Equal(4, read.Length);
        Assert.Equal(0.0, read[0]);
        Assert.True(double.IsNaN(read[1]), "NaN must survive as NaN, not become zero");
        Assert.Equal(-1234.5678901234, read[2], 12);
        Assert.Equal(1e300, read[3]);
    }

    /// <summary>
    /// Several row groups, because the readers concatenate across them and an off-by-one in the
    /// destination buffer - which v6 makes the caller supply - would show up here and nowhere else.
    /// </summary>
    [Fact]
    public void ValuesConcatenateAcrossRowGroups()
    {
        var path = TempPath();
        using (var writer = StreamingWideWriter.Create(
            path,
            new List<(string, Type)> { ("label", typeof(string)) },
            new List<string> { "s1" }))
        {
            writer.WriteRowGroup(new List<Array> { new[] { "r1", "r2" } }, new List<double[]> { new[] { 1.0, 2.0 } });
            writer.WriteRowGroup(new List<Array> { new[] { "r3" } }, new List<double[]> { new[] { 3.0 } });
            writer.WriteRowGroup(new List<Array> { new[] { "r4", "r5", "r6" } },
                new List<double[]> { new[] { 4.0, 5.0, 6.0 } });
        }

        using var reader = ParquetColumnReader.Open(path);

        Assert.Equal(6, reader.RowCount);
        Assert.Equal(new[] { "r1", "r2", "r3", "r4", "r5", "r6" }, reader.ReadStrings("label"));
        Assert.Equal(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, reader.ReadDoubles("s1"));
    }

    /// <summary>
    /// long and bool are the other element types the writers' schema builder emits, so they belong in the
    /// same round trip rather than being assumed to work because doubles do.
    /// </summary>
    [Fact]
    public void LongAndBoolMetaColumnsRoundTrip()
    {
        var path = TempPath();
        var longs = new[] { 0L, -1L, long.MaxValue };
        var bools = new[] { true, false, true };

        using (var writer = StreamingWideWriter.Create(
            path,
            new List<(string, Type)> { ("count", typeof(long)), ("flag", typeof(bool)) },
            new List<string> { "s1" }))
        {
            writer.WriteRowGroup(
                new List<Array> { longs, bools },
                new List<double[]> { new[] { 1.0, 2.0, 3.0 } });
        }

        var table = ParquetTable.Load(path);
        Assert.Equal(longs, table.GetLong("count"));
        Assert.Equal(bools, table.GetBool("flag"));
    }
}
