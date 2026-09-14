using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.RawData;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// How the cycles cache behaves when something already has the file open.
/// </summary>
/// <remarks>
/// <para>All of this is here because of one run. Forty-eight instrument files were read over several
/// hours, and none of the cycles were saved: the write was refused because
/// <c>ion_cycles.parquet</c> was "locked by another process". Nothing else was running. PRISM was
/// holding the file itself - a reader opened for a plot, sharing Read and therefore excluding every
/// writer - and it was replacing that same file after every one of the forty-eight replicates, so it
/// had forty-eight chances to collide with itself.</para>
///
/// <para>Both halves are pinned here, because either one alone leaves the failure reachable.</para>
/// </remarks>
public class IonCyclesCacheTests : IDisposable
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
                // A test that leaves a handle open is a test failure, not a cleanup failure.
            }
        }
    }

    [Fact]
    public void AnOpenReaderDoesNotStopTheCacheFromBeingRewritten()
    {
        var dir = NewDir();
        IonAccountingStore.Write(dir, Result("A", cycles: 3));

        // Exactly what the GUI does to populate the replicate picker, held open across the write -
        // which is the situation a run is in whenever anyone looks at the ion tab while it works.
        using (var held = ParquetColumnReader.Open(
                   Path.Combine(dir, IonAccountingStore.CyclesFile)))
        {
            Assert.Equal(3, held.RowCount);
            IonAccountingStore.Write(dir, Result("A", cycles: 5));
        }

        Assert.Equal(5, IonAccountingStore.ReadCycles(dir, "A").Count);
    }

    [Fact]
    public void AnOpenReaderDoesNotStopTheSummaryFromBeingRewritten()
    {
        var dir = NewDir();
        IonAccountingStore.Write(dir, Result("A", cycles: 2));

        using (var held = ParquetColumnReader.Open(
                   Path.Combine(dir, IonAccountingStore.FileName)))
        {
            Assert.Equal(1, held.RowCount);
            IonAccountingStore.Write(dir, Result("A", cycles: 2, ms2Acquired: 999.0));
        }

        var read = IonAccountingStore.Read(dir);
        Assert.NotNull(read);
        Assert.Equal(999.0, read!.Rows.Single().Ms2Acquired);
    }

    [Fact]
    public void AProgressSaveLeavesTheRealCyclesFileAlone()
    {
        var dir = NewDir();
        var path = Path.Combine(dir, IonAccountingStore.CyclesFile);

        IonAccountingStore.Write(dir, Result("A", cycles: 4), log: null, finalize: false);

        // The whole point: mid-run, the name nothing else watches is the only one touched. The
        // previous version created and replaced the real file after every replicate.
        Assert.False(File.Exists(path));
        Assert.True(File.Exists(path + ".new"));
    }

    [Fact]
    public void ARunInterruptedAfterProgressSavesStillYieldsItsCycles()
    {
        var dir = NewDir();
        IonAccountingStore.Write(dir, Result("A", cycles: 4), log: null, finalize: false);

        // No finalize ever happens - the run died. The next read recovers the staged file, so the
        // replicates measured before the interruption are all still there.
        Assert.Equal(new[] { "A" }, IonAccountingStore.SamplesWithCycles(dir));
        Assert.Equal(4, IonAccountingStore.ReadCycles(dir, "A").Count);
    }

    [Fact]
    public void TheFinalWriteReplacesWhateverTheLastRunLeft()
    {
        var dir = NewDir();
        IonAccountingStore.Write(dir, Result("A", cycles: 7));
        Assert.False(File.Exists(Path.Combine(dir, IonAccountingStore.CyclesFile + ".new")));

        IonAccountingStore.Write(dir, Result("A", cycles: 2));

        Assert.Equal(2, IonAccountingStore.ReadCycles(dir, "A").Count);
        Assert.False(File.Exists(Path.Combine(dir, IonAccountingStore.CyclesFile + ".new")));
    }

    [Fact]
    public void TheFinalWriteLandsEvenWithAReaderHoldingTheTarget()
    {
        var dir = NewDir();
        IonAccountingStore.Write(dir, Result("A", cycles: 3));

        // A rename-over is refused on Windows while any handle is open on the target, even a fully
        // permissive one, so this is the case the copy fallback exists for. The measured cycles
        // must reach the real name anyway.
        using (ParquetColumnReader.Open(Path.Combine(dir, IonAccountingStore.CyclesFile)))
        {
            IonAccountingStore.Write(dir, Result("A", cycles: 6));
        }

        Assert.Equal(6, IonAccountingStore.ReadCycles(dir, "A").Count);
        Assert.False(File.Exists(Path.Combine(dir, IonAccountingStore.CyclesFile + ".new")));
    }

    [Fact]
    public void AStagedFileNewerThanTheRealOneIsTheOneThatSurvives()
    {
        var dir = NewDir();
        var path = Path.Combine(dir, IonAccountingStore.CyclesFile);

        // A run interrupted at four replicates, recovered, then a second run interrupted at nine.
        // The real file holds the first attempt and the staging file holds the second, so taking
        // the real one because it exists would throw away the longer measurement of the two.
        IonAccountingStore.Write(dir, Result("A", cycles: 4));
        IonAccountingStore.Write(dir, Result("A", cycles: 9), log: null, finalize: false);
        File.SetLastWriteTimeUtc(path, DateTime.UtcNow.AddHours(-2));
        File.SetLastWriteTimeUtc(path + ".new", DateTime.UtcNow);

        IonAccountingStore.RecoverStagedCycles(dir);

        Assert.Equal(9, IonAccountingStore.ReadCycles(dir, "A").Count);
        Assert.False(File.Exists(path + ".new"));
    }

    [Fact]
    public void AStaleStagedFileDoesNotDisplaceANewerRealOne()
    {
        var dir = NewDir();
        var path = Path.Combine(dir, IonAccountingStore.CyclesFile);

        IonAccountingStore.Write(dir, Result("A", cycles: 9), log: null, finalize: false);
        IonAccountingStore.Write(dir, Result("A", cycles: 4));
        File.Copy(path, path + ".new");
        File.SetLastWriteTimeUtc(path + ".new", DateTime.UtcNow.AddHours(-2));

        IonAccountingStore.RecoverStagedCycles(dir);

        Assert.Equal(4, IonAccountingStore.ReadCycles(dir, "A").Count);
    }

    [Fact]
    public void ALiveRunsStagingFileIsLeftAloneByAReadFromElsewhere()
    {
        var dir = NewDir();
        var path = Path.Combine(dir, IonAccountingStore.CyclesFile);
        IonAccountingStore.Write(dir, Result("A", cycles: 6), log: null, finalize: false);

        // The GUI drawing the pane while the run works. The staging file is the run's own progress
        // store and is rewritten after every replicate: copying a half-written one would put a torn
        // parquet under the real name, and deleting it would take the run's progress away.
        using (IonAccountingStore.MarkMeasuring(dir))
        {
            IonAccountingStore.RecoverStagedCycles(dir);
            Assert.False(File.Exists(path));
            Assert.True(File.Exists(path + ".new"));
        }

        // Once the run is over - or has died, taking the mark with it - recovery is free to act.
        IonAccountingStore.RecoverStagedCycles(dir);
        Assert.Equal(6, IonAccountingStore.ReadCycles(dir, "A").Count);
    }

    [Fact]
    public void TheFinalWriteWaitsOutAHolderThatLetsGo()
    {
        var dir = NewDir();
        var path = Path.Combine(dir, IonAccountingStore.CyclesFile);
        IonAccountingStore.Write(dir, Result("A", cycles: 2));

        // Exclusive, which is what an antivirus scan or an SMB lease on a freshly created file
        // looks like: every way of replacing the target is refused until it lets go. These clear
        // in a second or two, so the write waits rather than throwing away a measurement.
        using var holding = new ManualResetEventSlim();
        var holder = new Thread(() =>
        {
            using var exclusive = new FileStream(
                path, FileMode.Open, FileAccess.Read, FileShare.None);
            holding.Set();
            Thread.Sleep(1000);
        }) { IsBackground = true };
        holder.Start();

        Assert.True(holding.Wait(TimeSpan.FromSeconds(10)));
        IonAccountingStore.Write(dir, Result("A", cycles: 8));
        holder.Join(TimeSpan.FromSeconds(10));

        Assert.Equal(8, IonAccountingStore.ReadCycles(dir, "A").Count);
        Assert.False(File.Exists(path + ".new"));
    }

    [Fact]
    public void CyclesMeasuredUnderOtherSettingsAreNotOfferedForReuse()
    {
        var dir = NewDir();
        IonAccountingStore.Write(dir, Result("A", cycles: 5));

        // The two files are written separately and the summary goes FIRST, so a failure between
        // them leaves a new summary beside an older set of traces. Replicate names are identical
        // from run to run, so the name alone cannot tell a stale trace from this run's - the key in
        // the file can.
        Assert.Equal(new[] { "A" }, IonAccountingStore.SamplesWithCycles(dir, null, "key"));
        Assert.Empty(IonAccountingStore.SamplesWithCycles(dir, null, "a-different-key"));

        // With nothing to check against, whatever is there is taken - which is also how a file
        // written before the key column behaves.
        Assert.Equal(new[] { "A" }, IonAccountingStore.SamplesWithCycles(dir));
    }

    /// <summary>
    /// A name PRISM cannot claim must not cost a measurement.
    /// </summary>
    /// <remarks>
    /// Recovery is still tried first and is still the normal outcome. But when the real name is
    /// held by something PRISM cannot argue with - a scanner, a NAS, another machine - the
    /// alternative to reading the staged file where it lies is refusing to draw anything at all,
    /// forever, over a rename. A 48-replicate measurement finished with every cycle on disk and the
    /// pane said a measurement was waiting and someone should rename a file by hand.
    /// </remarks>
    [Fact]
    public void AStagedMeasurementIsReadWhereItLiesWhenTheRealNameIsHeld()
    {
        var dir = NewDir();
        var path = Path.Combine(dir, IonAccountingStore.CyclesFile);
        IonAccountingStore.Write(dir, Result("A", cycles: 2));
        IonAccountingStore.Write(dir, Result("A", cycles: 9), log: null, finalize: false);
        File.SetLastWriteTimeUtc(path, DateTime.UtcNow.AddHours(-2));
        File.SetLastWriteTimeUtc(path + ".new", DateTime.UtcNow);

        // Exclusive, so recovery cannot put the staged file under the real name however it tries.
        using var held = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.None);

        // The NEWER of the two, which is the staged one - not the stale file that happens to own
        // the name.
        Assert.Equal(9, IonAccountingStore.ReadCycles(dir, "A").Count);
        Assert.Equal(new[] { "A" }, IonAccountingStore.SamplesWithCycles(dir));
    }

    [Fact]
    public void AMeasurementThatCannotClaimTheNameIsStillASuccess()
    {
        var dir = NewDir();
        var path = Path.Combine(dir, IonAccountingStore.CyclesFile);
        IonAccountingStore.Write(dir, Result("A", cycles: 2));

        var lines = new List<string>();

        // Read-only rather than held open, on both counts deliberately. It is refused identically
        // on every platform - FileShare is emulated on POSIX and a create is not always stopped by
        // it - and it fails FAST: ParquetWideWriter retries an IOException fifteen times at 300 ms
        // but lets UnauthorizedAccessException straight through, so the test does not sit out a
        // retry budget it is not testing.
        using (ShortWriteBudget())
        using (ReadOnlyFile(path))
        {
            // Forty-eight instrument files and an hour. Throwing over a file NAME reported all of
            // it as "Ion accounting failed" when every cycle was on disk and readable.
            IonAccountingStore.Write(dir, Result("A", cycles: 6), lines.Add);
        }

        Assert.True(File.Exists(path + ".new"), "the measurement should have been staged");

        // The write was refused at the OPEN, so it never touched the target - and the two cycles
        // the previous run left in it are still there. Deleting a good file because this run could
        // not replace it would be worse than the problem being solved.
        Assert.True(File.Exists(path), "a target this write never opened must not be deleted");

        Assert.Equal(6, IonAccountingStore.ReadCycles(dir, "A").Count);
        Assert.DoesNotContain(lines, l => l.Contains("WARNING", StringComparison.Ordinal));
        Assert.Contains(lines, l => l.Contains("nothing to do by hand", StringComparison.Ordinal));
    }

    /// <summary>
    /// "Best" and "worst" of WHAT. The two quantities rank the cohort differently, so a panel picked
    /// by one and captioned as the other names the wrong replicate - and nothing on the page could
    /// reveal it, because both numbers are real.
    /// </summary>
    [Fact]
    public void RepresentativesRankOnTheQuantityAskedFor()
    {
        // Ions say b is worst; signal says a is. Both are true: the ion count weights each scan by
        // its injection time and the summed TIC does not.
        var a = Row("a", ms2Acquired: 1000, ms2Assigned: 100, ms2Signal: 1000, ms2SignalAssigned: 50);
        var b = Row("b", ms2Acquired: 1000, ms2Assigned: 50, ms2SignalAssigned: 300, ms2Signal: 1000);
        var result = new IonAccountingResult(
            "k", "10 ppm", "10 ppm", "scheme", Array.Empty<string>(), 1, false,
            new[] { a, b }, Array.Empty<IonCycleRow>());

        // Best first, so the head of the list is the HIGHEST fraction in that quantity.
        Assert.Equal("a", result.Representatives(signal: false)[0].Sample);
        Assert.Equal("b", result.Representatives(signal: false)[^1].Sample);
        Assert.Equal("b", result.Representatives(signal: true)[0].Sample);
        Assert.Equal("a", result.Representatives(signal: true)[^1].Sample);
    }

    /// <summary>
    /// The empty state names the file and a remedy, and never asks anyone to rename anything - the
    /// staged file is read in place, so a staged file that exists is a file that was read.
    /// </summary>
    [Fact]
    public void TheEmptyStateSaysWhatToDoAboutIt()
    {
        var dir = NewDir();
        var absent = IonAccountingStore.DescribeMissingCycles(dir);
        Assert.Contains(IonAccountingStore.CyclesFile, absent, StringComparison.Ordinal);
        Assert.Contains("Re-run", absent, StringComparison.Ordinal);
        Assert.DoesNotContain("rename", absent, StringComparison.OrdinalIgnoreCase);

        // Present but unreadable is a different sentence, and must not read as "never measured".
        // Written as bytes that are NOT parquet: IonAccountingStore.Write would produce a valid
        // file, which exercises only the file-exists branch and would pass whatever the unreadable
        // path did.
        File.WriteAllBytes(
            Path.Combine(dir, IonAccountingStore.CyclesFile), new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 });
        Assert.Empty(IonAccountingStore.ReadCycles(dir, "A"));

        var present = IonAccountingStore.DescribeMissingCycles(dir);
        Assert.Contains("could not be read", present, StringComparison.Ordinal);
        Assert.DoesNotContain("rename", present, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// A target the write truncated but never finished must not shadow the intact staging file.
    /// </summary>
    /// <remarks>
    /// FileMode.Create truncates on OPEN, so a write that fails part way leaves a file that parses
    /// as nothing and carries a FRESH timestamp - newer than the staging file, which is what the
    /// readers prefer. Left alone it loses a whole cohort to tidy up after a failure, which is the
    /// exact shape of the bug this file exists for.
    /// </remarks>
    [Fact]
    public void ATruncatedRealFileDoesNotShadowTheStagedMeasurement()
    {
        var dir = NewDir();
        var path = Path.Combine(dir, IonAccountingStore.CyclesFile);

        // The staging file holds a complete measurement; the real one was left part-written after
        // it, so it is both newer AND unreadable.
        IonAccountingStore.Write(dir, Result("A", cycles: 7), log: null, finalize: false);
        File.WriteAllBytes(path, new byte[] { 0x50, 0x41, 0x52, 0x31, 0, 0, 0, 0 });
        File.SetLastWriteTimeUtc(path + ".new", DateTime.UtcNow.AddHours(-1));
        File.SetLastWriteTimeUtc(path, DateTime.UtcNow);

        Assert.Equal(7, IonAccountingStore.ReadCycles(dir, "A").Count);
        Assert.Equal(new[] { "A" }, IonAccountingStore.SamplesWithCycles(dir));
    }

    /// <summary>
    /// A refused write leaves the CURRENT measurement staged, not whatever was staged before.
    /// </summary>
    /// <remarks>
    /// A progress save that fails leaves a short staging file behind and the run carries on, so
    /// writing one only when it is absent would promise a complete measurement about a file missing
    /// replicates. The message says "nothing is lost"; it has to be true.
    /// </remarks>
    [Fact]
    public void ARefusedWriteStagesThisMeasurementOverAnOlderOne()
    {
        var dir = NewDir();
        var path = Path.Combine(dir, IonAccountingStore.CyclesFile);

        // A short progress file from earlier in the same run, and a real file held exclusively so
        // the end-of-run write cannot land.
        IonAccountingStore.Write(dir, Result("A", cycles: 2), log: null, finalize: false);
        IonAccountingStore.Write(dir, Result("A", cycles: 2));

        var lines = new List<string>();
        using (ShortWriteBudget())
        using (ReadOnlyFile(path))
        {
            IonAccountingStore.Write(dir, Result("A", cycles: 9), lines.Add);
        }

        // Nine, not the two the earlier progress save left - whichever file it comes from.
        Assert.Equal(9, IonAccountingStore.ReadCycles(dir, "A").Count);
    }

    /// <summary>
    /// Make a file unwritable for the life of the scope, and writable again afterwards - or the
    /// directory cleanup in Dispose cannot remove it.
    /// </summary>
    private static IDisposable ReadOnlyFile(string path)
    {
        File.SetAttributes(path, File.GetAttributes(path) | FileAttributes.ReadOnly);
        return new Restore(() =>
        {
            // The file may be gone: POSIX lets a read-only file be unlinked from a writable
            // directory, so a run that decided to remove it succeeds there and fails on Windows.
            if (File.Exists(path))
                File.SetAttributes(path, File.GetAttributes(path) & ~FileAttributes.ReadOnly);
        });
    }

    /// <summary>The product waits about 30 s; a test waiting that long is a test nobody runs.</summary>
    private static IDisposable ShortWriteBudget()
    {
        var attempts = IonAccountingStore.WriteAttempts;
        var delay = IonAccountingStore.WriteDelayMs;
        IonAccountingStore.WriteAttempts = 2;
        IonAccountingStore.WriteDelayMs = 1;
        return new Restore(() =>
        {
            IonAccountingStore.WriteAttempts = attempts;
            IonAccountingStore.WriteDelayMs = delay;
        });
    }

    private sealed class Restore : IDisposable
    {
        private readonly Action _undo;

        internal Restore(Action undo) => _undo = undo;

        public void Dispose() => _undo();
    }

    /// <summary>A row with both quantities, for the ranking test.</summary>
    private static IonAccountingRow Row(
        string sample, double ms2Acquired, double ms2Assigned,
        double ms2Signal, double ms2SignalAssigned) =>
        new(sample, "experimental", sample + ".raw", Ms2ReadStatus.Ok, "test", 10, 100,
            Ms1Acquired: 1000, Ms2Acquired: ms2Acquired,
            Ms1Assigned: 100, Ms2Assigned: ms2Assigned,
            Ms2Explained: 0, HasExplained: false,
            0, 30, 0, 0, 0, 1, Array.Empty<double>(), Array.Empty<double>(),
            AcquiredUtc: null, Ms1Signal: 1000, Ms2Signal: ms2Signal,
            Ms1SignalAssigned: 100, Ms2SignalAssigned: ms2SignalAssigned,
            Ms2SignalExplained: 0, HasSignal: true);

    private string NewDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism-cycles-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        _dirs.Add(dir);
        return dir;
    }

    private static IonAccountingResult Result(
        string sample, int cycles, double ms2Acquired = 100.0)
    {
        var row = new IonAccountingRow(
            sample, "experimental", sample + ".raw", Ms2ReadStatus.Ok, "fake",
            10, 100, 50.0, ms2Acquired, 5.0, 20.0, 25.0, true,
            0.0, 30.0, 0, 0, 0, cycles,
            Array.Empty<double>(), Array.Empty<double>());

        var traces = Enumerable.Range(0, cycles)
            .Select(i => new IonCycleRow(
                sample, i, i * 0.5, (i + 1) * 0.5, 1, 10,
                1.0, 2.0, 0.5, 1.0, 1.5, 3.0, 4.0, 1.0, 2.0, 2.5))
            .ToList();

        return new IonAccountingResult(
            "key", "10 ppm", "10 ppm", "scheme", Array.Empty<string>(), 1, false,
            new[] { row }, traces);
    }
}
