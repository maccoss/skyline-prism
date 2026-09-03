using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using SkylinePrism.Core.RawData;
using Xunit;
using Xunit.Abstractions;

namespace SkylinePrism.Tests.RawData;

/// <summary>
/// Bounded-concurrency cohort reading. These run against fake readers rather than instrument files,
/// because what needs pinning is the CONTRACT - every file read exactly once, concurrency actually
/// bounded, one bad file costing only itself - and none of that needs a 3 GB file.
///
/// <para>The measured speedup on real storage is ~1.3x and lives in the reader benchmarks; it is
/// deliberately not asserted here, because a timing assertion on a shared CI machine fails for
/// reasons that have nothing to do with the code.</para>
/// </summary>
[Collection(nameof(Ms2SignalReadManyTests))]
[CollectionDefinition(nameof(Ms2SignalReadManyTests), DisableParallelization = true)]
public class Ms2SignalReadManyTests : IDisposable
{
    private readonly ITestOutputHelper _out;

    public Ms2SignalReadManyTests(ITestOutputHelper output)
    {
        _out = output;
        Ms2SignalReaders.Clear();
    }

    public void Dispose() => Ms2SignalReaders.Clear();

    /// <summary>A reader that records how many reads were in flight at once.</summary>
    private sealed class ConcurrencyProbe : IMs2SignalReader
    {
        private int _inFlight;

        public int Peak;
        public ConcurrentBag<string> Seen = new();

        public string Describe() => "probe";

        public bool CanRead(string dataPath) => true;

        public Ms2SignalRecord Read(
            string dataPath, Action<string>? log = null, CancellationToken ct = default)
        {
            var now = Interlocked.Increment(ref _inFlight);
            InterlockedMax(ref Peak, now);
            try
            {
                // Long enough that lanes genuinely overlap; short enough not to slow the suite.
                Thread.Sleep(40);
                Seen.Add(dataPath);
                return new Ms2SignalRecord(
                    dataPath, Ms2ReadStatus.Ok, Describe(), Ms2SignalSource.ReportedTic,
                    1, 10, 1000, 0, 1, Ms2CycleModel.Ms1Bounded,
                    Array.Empty<Ms2Cycle>(), Array.Empty<SkylinePrism.Core.Qc.IsolationWindow>());
            }
            finally
            {
                Interlocked.Decrement(ref _inFlight);
            }
        }

        private static void InterlockedMax(ref int target, int value)
        {
            int seen;
            while (value > (seen = Volatile.Read(ref target)))
                if (Interlocked.CompareExchange(ref target, value, seen) == seen)
                    return;
        }
    }

    /// <summary>Every file is read exactly once, and every record comes back.</summary>
    [Fact]
    public void ReadsEveryFileExactlyOnce()
    {
        var probe = new ConcurrencyProbe();
        Ms2SignalReaders.Register(probe);

        var paths = Enumerable.Range(0, 25).Select(i => $"run{i:00}.raw").ToList();
        var records = new ConcurrentBag<Ms2SignalRecord>();

        Ms2SignalReaders.ReadMany(paths, records.Add, lanes: 8);

        Assert.Equal(paths.Count, records.Count);
        Assert.Equal(
            paths.OrderBy(p => p, StringComparer.Ordinal),
            records.Select(r => r.DataPath).OrderBy(p => p, StringComparer.Ordinal));
        // Exactly once - a duplicated read would double-count a replicate's acquired signal.
        Assert.Equal(paths.Count, probe.Seen.Distinct(StringComparer.Ordinal).Count());
    }

    /// <summary>
    /// The bound is honoured. Not a performance assertion - an unbounded read of a 192-replicate
    /// cohort would open 192 files at once against someone else's file server.
    /// </summary>
    [Fact]
    public void ConcurrencyIsBounded()
    {
        var probe = new ConcurrencyProbe();
        Ms2SignalReaders.Register(probe);

        Ms2SignalReaders.ReadMany(
            Enumerable.Range(0, 24).Select(i => $"run{i:00}.raw").ToList(),
            _ => { }, lanes: 4);

        _out.WriteLine($"peak concurrent reads: {probe.Peak} (bound 4)");
        Assert.True(probe.Peak <= 4, $"peak concurrency {probe.Peak} exceeded the bound of 4");
        // ...and it really did run concurrently, or the bound proves nothing.
        Assert.True(probe.Peak > 1, "nothing ran concurrently, so the bound was not exercised");
    }

    /// <summary>Lanes are clamped: 0 or negative must not mean unbounded or throw.</summary>
    [Fact]
    public void LaneCountIsClamped()
    {
        var probe = new ConcurrencyProbe();
        Ms2SignalReaders.Register(probe);

        var records = new ConcurrentBag<Ms2SignalRecord>();
        Ms2SignalReaders.ReadMany(new[] { "a.raw", "b.raw" }, records.Add, lanes: 0);

        Assert.Equal(2, records.Count);
        Assert.Equal(1, probe.Peak);   // clamped to serial, not to unbounded
    }

    /// <summary>
    /// One unreadable file costs only itself. This is the whole reason the reader contract forbids
    /// throwing: a cohort read must not be abandoned 40 files in.
    /// </summary>
    [Fact]
    public void OneBadFileDoesNotAbandonTheRest()
    {
        Ms2SignalReaders.Register(new FailingReader("b.raw"));

        var records = new List<Ms2SignalRecord>();
        Ms2SignalReaders.ReadMany(
            new[] { "a.raw", "b.raw", "c.raw" },
            r => { lock (records) records.Add(r); },
            lanes: 3);

        Assert.Equal(3, records.Count);
        Assert.Single(records, r => r.Status != Ms2ReadStatus.Ok);
        Assert.Equal("b.raw", records.Single(r => r.Status != Ms2ReadStatus.Ok).DataPath);
    }

    private sealed class FailingReader : IMs2SignalReader
    {
        private readonly string _bad;

        public FailingReader(string bad) => _bad = bad;

        public string Describe() => "failing";

        public bool CanRead(string dataPath) => true;

        public Ms2SignalRecord Read(
            string dataPath, Action<string>? log = null, CancellationToken ct = default) =>
            string.Equals(dataPath, _bad, StringComparison.Ordinal)
                ? Ms2SignalRecord.Unavailable(dataPath, Ms2ReadStatus.Failed, Describe(), "boom")
                : new Ms2SignalRecord(
                    dataPath, Ms2ReadStatus.Ok, Describe(), Ms2SignalSource.ReportedTic,
                    1, 10, 1000, 0, 1, Ms2CycleModel.Ms1Bounded,
                    Array.Empty<Ms2Cycle>(), Array.Empty<SkylinePrism.Core.Qc.IsolationWindow>());
    }

    /// <summary>An empty cohort is a no-op, and null arguments are rejected rather than ignored.</summary>
    [Fact]
    public void EdgeCases()
    {
        Ms2SignalReaders.ReadMany(Array.Empty<string>(), _ => Assert.Fail("nothing to read"));

        Assert.Throws<ArgumentNullException>(
            () => Ms2SignalReaders.ReadMany(null!, _ => { }));
        Assert.Throws<ArgumentNullException>(
            () => Ms2SignalReaders.ReadMany(new[] { "a.raw" }, null!));
    }
}
