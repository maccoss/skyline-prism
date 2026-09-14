using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.RawData;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// The cycles file accumulates under its real name, one replicate at a time.
/// </summary>
/// <remarks>
/// <para>It used to be rewritten in full after every replicate, into a staging file that was renamed
/// at the end. That is O(n^2) in bytes - measured on a real 48-replicate cache, 883 MB written to
/// persist 36 MB, about 92 GB projected at 500 - and, more to the point, it meant the real file did
/// not exist until the run finished. A run that died at replicate 48 of 48 left everything it had
/// measured under a name nothing was looking for.</para>
///
/// <para>Appending fixes both: each row is written once, and the file is valid and complete under
/// its own name after every replicate.</para>
/// </remarks>
public class IonCyclesAppendTests : IDisposable
{
    private readonly List<string> _dirs = new();

    public void Dispose()
    {
        foreach (var dir in _dirs)
        {
            try
            {
                Directory.Delete(dir, recursive: true);
            }
            catch (IOException)
            {
            }
        }
    }

    [Fact]
    public void TheFileIsCompleteAndCorrectAfterEveryReplicate()
    {
        var dir = NewDir();
        var path = Path.Combine(dir, IonAccountingStore.CyclesFile);
        IonAccountingStore.BeginCycles(dir);

        for (var r = 1; r <= 6; r++)
        {
            IonAccountingStore.AppendCycles(dir, Cycles($"rep{r}", r * 10), "key");

            // Readable, and holding everything measured so far - not just the last replicate, and
            // not a torn file. This is the property the staging file existed to provide.
            var read = IonAccountingStore.ReadCycles(dir);
            Assert.Equal(Enumerable.Range(1, r).Sum(i => i * 10), read.Count);
            Assert.Equal(r, IonAccountingStore.SamplesWithCycles(dir).Count);

            // One row group per replicate is the whole point: a rewrite would leave exactly one.
            using var reader = ParquetColumnReader.Open(path);
            Assert.Equal(r, reader.RowGroupCount);
        }

        // And no staging file was ever involved.
        Assert.False(File.Exists(path + ".new"));
    }

    /// <summary>
    /// What a run leaves when it stops partway is a valid file, under the real name.
    /// </summary>
    [Fact]
    public void ARunThatStopsLeavesWhatItHadUnderTheRealName()
    {
        var dir = NewDir();
        IonAccountingStore.BeginCycles(dir);
        IonAccountingStore.AppendCycles(dir, Cycles("a", 5), "key");
        IonAccountingStore.AppendCycles(dir, Cycles("b", 5), "key");
        // ... and here the process dies. Nothing finalizes, nothing is renamed.

        Assert.True(File.Exists(Path.Combine(dir, IonAccountingStore.CyclesFile)));
        Assert.Equal(new[] { "a", "b" }, IonAccountingStore.SamplesWithCycles(dir).OrderBy(s => s));
        Assert.Equal(5, IonAccountingStore.ReadCycles(dir, "a").Count);
    }

    /// <summary>
    /// A measurement REPLACES the previous one rather than growing it.
    /// </summary>
    /// <remarks>
    /// Appending to whatever was there would silently carry a previous run's replicates into this
    /// one's file - under this run's settings key, which is the one thing the key exists to prevent.
    /// </remarks>
    [Fact]
    public void BeginningAMeasurementReplacesTheLastOne()
    {
        var dir = NewDir();
        IonAccountingStore.BeginCycles(dir);
        IonAccountingStore.AppendCycles(dir, Cycles("old", 9), "key-1");
        Assert.Equal(9, IonAccountingStore.ReadCycles(dir).Count);

        IonAccountingStore.BeginCycles(dir);
        IonAccountingStore.AppendCycles(dir, Cycles("new", 4), "key-2");

        Assert.Equal(4, IonAccountingStore.ReadCycles(dir).Count);
        Assert.Equal(new[] { "new" }, IonAccountingStore.SamplesWithCycles(dir));
    }

    /// <summary>
    /// Appended and written-whole produce the same table - the schemas cannot be allowed to drift.
    /// </summary>
    /// <remarks>
    /// Parquet will accept a row group whose schema differs from the file's and fail later, at read
    /// time, somewhere else. Both paths go through one column definition; this is what says so.
    /// </remarks>
    [Fact]
    public void AppendingGivesTheSameTableAsWritingItWhole()
    {
        var appended = NewDir();
        var whole = NewDir();
        var all = new List<IonCycleRow>();

        IonAccountingStore.BeginCycles(appended);
        foreach (var sample in new[] { "a", "b", "c" })
        {
            var rows = Cycles(sample, 7);
            all.AddRange(rows);
            IonAccountingStore.AppendCycles(appended, rows, "key");
        }

        IonAccountingStore.Write(whole, Result(all));

        var fromAppend = IonAccountingStore.ReadCycles(appended).OrderBy(c => c.Sample)
            .ThenBy(c => c.Cycle).ToArray();
        var fromWhole = IonAccountingStore.ReadCycles(whole).OrderBy(c => c.Sample)
            .ThenBy(c => c.Cycle).ToArray();

        Assert.Equal(fromWhole.Length, fromAppend.Length);
        Assert.Equal(fromWhole, fromAppend);

        // Same columns, in the same order.
        using var a = ParquetColumnReader.Open(Path.Combine(appended, IonAccountingStore.CyclesFile));
        using var w = ParquetColumnReader.Open(Path.Combine(whole, IonAccountingStore.CyclesFile));
        Assert.Equal(w.ColumnNames, a.ColumnNames);
    }

    [Fact]
    public void AppendingNothingIsNotAFile()
    {
        var dir = NewDir();
        IonAccountingStore.BeginCycles(dir);
        IonAccountingStore.AppendCycles(dir, Array.Empty<IonCycleRow>(), "key");

        Assert.False(File.Exists(Path.Combine(dir, IonAccountingStore.CyclesFile)));
        Assert.Empty(IonAccountingStore.SamplesWithCycles(dir));
    }

    private string NewDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism-append-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        _dirs.Add(dir);
        return dir;
    }

    private static List<IonCycleRow> Cycles(string sample, int count) =>
        Enumerable.Range(0, count)
            .Select(i => new IonCycleRow(
                sample, i, i * 0.5, (i + 1) * 0.5, 1, 167,
                1e5 + i, 2e6 + i, 1e4 + i, 5e4 + i, 9e4 + i,
                1e5, 2e6, 1e4, 5e4, 9e4))
            .ToList();

    private static IonAccountingResult Result(IReadOnlyList<IonCycleRow> cycles)
    {
        var rows = cycles.Select(c => c.Sample).Distinct(StringComparer.Ordinal)
            .Select(s => new IonAccountingRow(
                s, "experimental", s + ".raw", Ms2ReadStatus.Ok, "test", 1, 167,
                1000, 1000, 100, 100, 0, false, 0, 30, 0, 0, 0, 7,
                Array.Empty<double>(), Array.Empty<double>()))
            .ToArray();
        return new IonAccountingResult(
            "key", "10 ppm", "10 ppm", "scheme", Array.Empty<string>(), 1, false, rows, cycles);
    }
}
