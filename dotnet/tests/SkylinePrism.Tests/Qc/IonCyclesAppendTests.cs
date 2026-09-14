using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
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
    /// A file that exists but holds no footer is started over, not appended to.
    /// </summary>
    /// <remarks>
    /// An append killed between creating the file and flushing its footer leaves zero bytes behind.
    /// Deciding appendability on existence alone asks parquet to append to a file with nothing to
    /// append to, which throws - and would then throw for every remaining replicate of the run,
    /// because nothing repairs it. Restarting an empty file loses nothing.
    /// </remarks>
    [Fact]
    public void AnEmptyFileIsStartedOverRatherThanAppendedTo()
    {
        var dir = NewDir();
        var path = Path.Combine(dir, IonAccountingStore.CyclesFile);
        File.WriteAllBytes(path, Array.Empty<byte>());

        IonAccountingStore.AppendCycles(dir, Cycles("a", 3), "key");

        Assert.Equal(3, IonAccountingStore.ReadCycles(dir).Count);
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
    /// A process killed mid-append loses the replicate in flight, not the whole measurement.
    /// </summary>
    /// <remarks>
    /// <para>Parquet keeps its metadata at the END of the file, so an append overwrites the existing
    /// footer with the new row group and writes a fresh one after it. A process killed in between -
    /// and this repository documents ion accounting dying that way twice, to a native fault no catch
    /// block sees - leaves a file with no footer at all. That reads as NOTHING, which is the case
    /// this test pins: not "everything except the replicate in flight", zero.</para>
    ///
    /// <para>So the bytes each append is about to overwrite are kept beside the file first, and
    /// everything saved before the interrupted append comes back.</para>
    /// </remarks>
    [Fact]
    public void AnInterruptedAppendCostsOnlyTheReplicateInFlight()
    {
        var dir = NewDir();
        var path = Path.Combine(dir, IonAccountingStore.CyclesFile);

        for (var r = 1; r <= 3; r++)
            IonAccountingStore.AppendCycles(dir, Cycles($"rep{r}", 100), "key", replace: r == 1);

        // A fourth append runs, which is what writes the backup describing the first three. Killing
        // the process inside it leaves the file overwritten from that footer's offset onward.
        IonAccountingStore.AppendCycles(dir, Cycles("rep4", 100), "key");
        var backup = ParquetWideWriter.FooterBackupOf(path);
        var offset = BitConverter.ToInt64(File.ReadAllBytes(backup), 0);
        using (var fs = new FileStream(path, FileMode.Open, FileAccess.Write))
        {
            fs.SetLength(offset + 64);
            fs.Seek(offset, SeekOrigin.Begin);
            fs.Write(new byte[64], 0, 64);
        }

        // The WHOLE file is gone, not just the last replicate - 300 saved rows and parquet cannot
        // open it at all. This is the measured fact the backup exists for.
        Assert.ThrowsAny<Exception>(() => ParquetColumnReader.Open(path).Dispose());

        Assert.True(IonAccountingStore.RepairCycles(dir));
        Assert.Equal(300, IonAccountingStore.ReadCycles(dir).Count);
        Assert.Equal(
            new[] { "rep1", "rep2", "rep3" },
            IonAccountingStore.SamplesWithCycles(dir).OrderBy(s => s, StringComparer.Ordinal));
    }

    /// <summary>
    /// And nobody has to ask: reading the directory repairs it, because the reader is where the
    /// damage is noticed.
    /// </summary>
    [Fact]
    public void ReadingATornFileRepairsItWithoutBeingAsked()
    {
        var dir = NewDir();
        var path = Path.Combine(dir, IonAccountingStore.CyclesFile);

        IonAccountingStore.AppendCycles(dir, Cycles("a", 50), "key", replace: true);
        IonAccountingStore.AppendCycles(dir, Cycles("b", 50), "key");
        var offset = BitConverter.ToInt64(
            File.ReadAllBytes(ParquetWideWriter.FooterBackupOf(path)), 0);
        using (var fs = new FileStream(path, FileMode.Open, FileAccess.Write))
            fs.SetLength(offset);

        Assert.Equal(50, IonAccountingStore.ReadCycles(dir).Count);
    }

    /// <summary>
    /// A read that lands in the middle of an append waits it out instead of reporting no data.
    /// </summary>
    /// <remarks>
    /// For the width of one append the file has no footer - momentarily headless rather than
    /// damaged, which is why the repair refuses to touch it while a writer is live. Measured with a
    /// writer appending back to back, 4.6% of opens failed and a single 50 ms retry recovered every
    /// one. The backup is deleted here so the repair cannot stand in for the retry being tested.
    /// </remarks>
    [Fact]
    public async Task AReadWaitsOutAnAppendRatherThanReportingNothing()
    {
        var dir = NewDir();
        var path = Path.Combine(dir, IonAccountingStore.CyclesFile);
        IonAccountingStore.AppendCycles(dir, Cycles("a", 40), "key", replace: true);
        IonAccountingStore.AppendCycles(dir, Cycles("b", 40), "key");

        var whole = File.ReadAllBytes(path);
        var offset = whole.Length - 8 - BitConverter.ToInt32(whole, whole.Length - 8);
        File.Delete(ParquetWideWriter.FooterBackupOf(path));

        using (var fs = new FileStream(path, FileMode.Open, FileAccess.Write, FileShare.ReadWrite))
            fs.SetLength(offset);

        // 30 ms against the reader's 5 x 75 ms budget, so a loaded CI runner has 270 ms of slack
        // before this turns into a spurious failure on someone else's pull request. Long enough that
        // the read below is overwhelmingly likely to have started and found the file headless, which
        // is the state being tested.
        var finish = Task.Run(() =>
        {
            Thread.Sleep(30);
            using var fs = new FileStream(
                path, FileMode.Open, FileAccess.Write, FileShare.ReadWrite);
            fs.SetLength(0);
            fs.Write(whole, 0, whole.Length);
        });

        var rows = IonAccountingStore.ReadCycles(dir);
        await finish;

        Assert.Equal(80, rows.Count);
    }

    /// <summary>
    /// "Could not read it" and "there is nothing in it" are different answers.
    /// </summary>
    /// <remarks>
    /// They used to be the same empty list, and the caller that decides what to reuse turned it into
    /// "no replicate has traces" - re-measuring the whole cohort, every instrument file again, with
    /// nothing said. The file being damaged is not a fact about the data.
    /// </remarks>
    [Fact]
    public void AnUnreadableFileIsNotReportedAsAnEmptyOne()
    {
        var dir = NewDir();

        Assert.Empty(IonAccountingStore.SamplesWithCycles(dir, out var absent));
        Assert.False(absent);

        // There, and not parquet at all - with no backup, so the repair cannot rescue it either.
        File.WriteAllBytes(
            Path.Combine(dir, IonAccountingStore.CyclesFile), new byte[32]);

        Assert.Empty(IonAccountingStore.SamplesWithCycles(dir, out var damaged));
        Assert.True(damaged);
    }

    /// <summary>
    /// A backup that is not shaped like a footer is not written over the file it claims to fix.
    /// </summary>
    /// <remarks>
    /// The repair truncates to the offset the backup records, so acting on a torn or foreign backup
    /// would discard bytes in exchange for a footer describing nothing - turning one broken state
    /// into a different one. The blob is checked for its own trailing PAR1 and a self-consistent
    /// length first.
    /// </remarks>
    [Fact]
    public void ADamagedBackupIsNotActedOn()
    {
        var dir = NewDir();
        var path = Path.Combine(dir, IonAccountingStore.CyclesFile);

        IonAccountingStore.AppendCycles(dir, Cycles("a", 40), "key", replace: true);
        IonAccountingStore.AppendCycles(dir, Cycles("b", 40), "key");

        var backup = ParquetWideWriter.FooterBackupOf(path);
        var offset = BitConverter.ToInt64(File.ReadAllBytes(backup), 0);
        using (var fs = new FileStream(path, FileMode.Open, FileAccess.Write))
            fs.SetLength(offset);
        var torn = new FileInfo(path).Length;

        // Truncated mid-write: the offset still reads, the footer behind it does not.
        var blob = File.ReadAllBytes(backup);
        File.WriteAllBytes(backup, blob[..(blob.Length / 2)]);

        Assert.False(IonAccountingStore.RepairCycles(dir));
        Assert.Equal(torn, new FileInfo(path).Length);
    }

    /// <summary>
    /// A live measurement is never rewound by a reader that looked at the wrong moment.
    /// </summary>
    /// <remarks>
    /// Mid-append and interrupted-append are the same thing from outside - no footer either way -
    /// so the only thing separating them is whether a measurement is running. The guard lives in
    /// RepairCycles so every caller gets it rather than the ones that remembered.
    /// </remarks>
    [Fact]
    public void RepairRefusesWhileAMeasurementIsRunning()
    {
        var dir = NewDir();
        var path = Path.Combine(dir, IonAccountingStore.CyclesFile);

        IonAccountingStore.AppendCycles(dir, Cycles("a", 40), "key", replace: true);
        IonAccountingStore.AppendCycles(dir, Cycles("b", 40), "key");
        var offset = BitConverter.ToInt64(
            File.ReadAllBytes(ParquetWideWriter.FooterBackupOf(path)), 0);
        using (var fs = new FileStream(path, FileMode.Open, FileAccess.Write))
            fs.SetLength(offset);

        using (IonAccountingStore.MarkMeasuring(dir))
            Assert.False(IonAccountingStore.RepairCycles(dir));

        // ...and once it is over, the same file is repaired.
        Assert.True(IonAccountingStore.RepairCycles(dir));
        Assert.Equal(40, IonAccountingStore.ReadCycles(dir).Count);
    }

    /// <summary>
    /// A repair is reported, because a replicate was lost and a crash was survived.
    /// </summary>
    /// <remarks>
    /// Recovering silently leaves the next reader unable to explain why the replicate count moved,
    /// which is the same failure as reporting an unreadable file as an empty one: the machine copes
    /// and the person is not told.
    /// </remarks>
    [Fact]
    public void ARepairIsSaidOutLoud()
    {
        var dir = NewDir();
        var path = Path.Combine(dir, IonAccountingStore.CyclesFile);

        IonAccountingStore.AppendCycles(dir, Cycles("a", 20), "key", replace: true);
        IonAccountingStore.AppendCycles(dir, Cycles("b", 20), "key");
        var offset = BitConverter.ToInt64(
            File.ReadAllBytes(ParquetWideWriter.FooterBackupOf(path)), 0);
        using (var fs = new FileStream(path, FileMode.Open, FileAccess.Write))
            fs.SetLength(offset);

        var log = new List<string>();
        Assert.Equal(new[] { "a" }, IonAccountingStore.SamplesWithCycles(dir, log.Add));

        Assert.Contains(log, line => line.Contains("repaired", StringComparison.Ordinal));
    }

    /// <summary>
    /// A failure reading the staging file does not send the repair at a different file.
    /// </summary>
    /// <remarks>
    /// Repair only ever rebuilds the real cycles file. Triggering it from a failed read of
    /// <c>.new</c> would examine a file this read is not looking at, and a success there would send
    /// the caller back to re-read the corrupt one.
    /// </remarks>
    [Fact]
    public void ACorruptStagingFileDoesNotTriggerARepairOfTheRealOne()
    {
        var dir = NewDir();
        var path = Path.Combine(dir, IonAccountingStore.CyclesFile);

        IonAccountingStore.AppendCycles(dir, Cycles("real", 12), "key", replace: true);
        var intact = new FileInfo(path).Length;

        // Newer than the real file, and not parquet at all.
        var staging = path + ".new";
        File.WriteAllBytes(staging, new byte[64]);
        File.SetLastWriteTimeUtc(staging, DateTime.UtcNow.AddMinutes(5));

        // Whatever it makes of the pair, the real file is not rebuilt on the staging file's account.
        _ = IonAccountingStore.ReadCycles(dir);
        Assert.Equal(intact, new FileInfo(path).Length);
    }

    /// <summary>A file that is intact is never rewound, however stale the backup beside it.</summary>
    [Fact]
    public void RepairLeavesAReadableFileAlone()
    {
        var dir = NewDir();
        IonAccountingStore.AppendCycles(dir, Cycles("a", 7), "key", replace: true);
        IonAccountingStore.AppendCycles(dir, Cycles("b", 7), "key");

        Assert.False(IonAccountingStore.RepairCycles(dir));
        Assert.Equal(14, IonAccountingStore.ReadCycles(dir).Count);
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

        // The replacement is part of the OPEN, not a delete before it: a delete can be refused and
        // then succeed a moment later, inside the append's own retry window, and the new rows would
        // land on top of the old measurement with a second settings key and nothing said.
        Assert.True(
            File.Exists(Path.Combine(dir, IonAccountingStore.CyclesFile)),
            "BeginCycles must not delete the file the first append replaces");
        IonAccountingStore.AppendCycles(dir, Cycles("new", 4), "key-2", replace: true);

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
