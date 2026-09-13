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

        var attempts = IonAccountingStore.PlacementAttempts;
        var delay = IonAccountingStore.PlacementDelayMs;
        IonAccountingStore.PlacementAttempts = 2;
        IonAccountingStore.PlacementDelayMs = 1;
        var lines = new List<string>();
        bool refused;
        try
        {
            using (new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.None))
            {
                // Forty-eight instrument files and several hours. Throwing over a file NAME
                // reported all of it as "Ion accounting failed" when every cycle was on disk and
                // readable.
                IonAccountingStore.Write(dir, Result("A", cycles: 6), lines.Add);

                // Windows refuses every way of replacing a file something holds exclusively.
                // POSIX does not - a rename over an open file is ordinary there - so the placement
                // genuinely succeeds on Linux and macOS. Both outcomes are a success; asserting
                // the Windows one everywhere is what broke this on the other two.
                refused = File.Exists(path + ".new");
            }
        }
        finally
        {
            IonAccountingStore.PlacementAttempts = attempts;
            IonAccountingStore.PlacementDelayMs = delay;
        }

        // The measurement survives either way, which is the whole point.
        Assert.Equal(6, IonAccountingStore.ReadCycles(dir, "A").Count);
        Assert.DoesNotContain(lines, l => l.Contains("WARNING", StringComparison.Ordinal));
        if (refused)
        {
            Assert.Contains(
                lines, l => l.Contains("nothing to do by hand", StringComparison.Ordinal));
        }
    }

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
