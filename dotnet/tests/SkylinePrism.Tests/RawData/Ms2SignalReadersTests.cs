using System;
using System.Collections.Generic;
using System.Threading;
using SkylinePrism.Core.RawData;
using Xunit;

namespace SkylinePrism.Tests.RawData;

/// <summary>
/// The optional raw-file seam. What matters here is the DEGRADED path, because that is the shipped one
/// for the cross-platform CLI and for any build without ProteoWizard: the accounting must still produce
/// its assigned and per-list unions, with the acquired total reported as unknown rather than guessed.
///
/// <para>These tests mutate a process-wide registry, so they run in one collection and clear it after
/// each - a reader leaking between tests would make an unrelated one pass for the wrong reason.</para>
/// </summary>
[Collection(nameof(Ms2SignalReadersTests))]
[CollectionDefinition(nameof(Ms2SignalReadersTests), DisableParallelization = true)]
public class Ms2SignalReadersTests : IDisposable
{
    public Ms2SignalReadersTests() => Ms2SignalReaders.Clear();

    public void Dispose() => Ms2SignalReaders.Clear();

    private sealed class FakeReader : IMs2SignalReader
    {
        private readonly Func<string, bool> _claims;
        private readonly Func<string, Ms2SignalRecord>? _read;

        public FakeReader(
            string name, Func<string, bool>? claims = null, Func<string, Ms2SignalRecord>? read = null)
        {
            Name = name;
            _claims = claims ?? (_ => true);
            _read = read;
        }

        public string Name { get; }

        public string Describe() => Name;

        public bool CanRead(string dataPath) => _claims(dataPath);

        public Ms2SignalRecord Read(string dataPath, Action<string>? log = null, CancellationToken ct = default)
            => _read?.Invoke(dataPath) ?? Ok(dataPath, Name);
    }

    private static Ms2SignalRecord Ok(string path, string reader) => new(
        path, Ms2ReadStatus.Ok, reader, Ms2SignalSource.ReportedTic,
        Ms1Count: 100, Ms2Count: 900, TotalMs2Signal: 1.5e9,
        RtStartMin: 0.5, RtStopMin: 60.0, Ms2CycleModel.Ms1Bounded,
        new List<Ms2Cycle> { new(0, 0.5, 0.9, 9, 1.5e7) },
        Array.Empty<SkylinePrism.Core.Qc.IsolationWindow>());

    /// <summary>
    /// The shipped state of the cross-platform CLI: no reader at all. The caller still gets a record,
    /// so the accounting reports the acquired total as unknown instead of failing or inventing one.
    /// </summary>
    [Fact]
    public void WithNoReaderRegisteredTheReadIsUnavailableRatherThanAFailure()
    {
        Assert.Empty(Ms2SignalReaders.All);
        Assert.Null(Ms2SignalReaders.For("run.raw"));

        var record = Ms2SignalReaders.Read("run.raw");

        Assert.Equal(Ms2ReadStatus.NoReader, record.Status);
        Assert.False(record.IsUsable);
        Assert.Equal(0, record.TotalMs2Signal);
        Assert.Equal(Ms2CycleModel.None, record.CycleModel);
        Assert.Empty(record.Cycles);
        Assert.NotNull(record.Message);
    }

    /// <summary>A registered reader that claims the file gets used, and its record comes back whole.</summary>
    [Fact]
    public void AReaderThatClaimsTheFileIsUsed()
    {
        Ms2SignalReaders.Register(new FakeReader("fake 1.0"));

        var record = Ms2SignalReaders.Read("run.raw");

        Assert.Equal(Ms2ReadStatus.Ok, record.Status);
        Assert.True(record.IsUsable);
        Assert.Equal("fake 1.0", record.Reader);
        Assert.Equal(1.5e9, record.TotalMs2Signal);
        Assert.Equal(Ms2SignalSource.ReportedTic, record.SignalSource);
    }

    /// <summary>
    /// Later registrations are tried first, so a vendor-specific reader can take precedence over a
    /// general one that would also claim the file.
    /// </summary>
    [Fact]
    public void TheMostRecentlyRegisteredReaderWins()
    {
        Ms2SignalReaders.Register(new FakeReader("general"));
        Ms2SignalReaders.Register(new FakeReader("specific"));

        Assert.Equal("specific", Ms2SignalReaders.Read("run.raw").Reader);
        Assert.Equal(2, Ms2SignalReaders.All.Count);
    }

    /// <summary>A reader that does not recognize the format is passed over, not treated as a failure.</summary>
    [Fact]
    public void ReadersThatDoNotClaimTheFileArePassedOver()
    {
        Ms2SignalReaders.Register(new FakeReader("mzml", claims: p => p.EndsWith(".mzML", StringComparison.OrdinalIgnoreCase)));
        Ms2SignalReaders.Register(new FakeReader("thermo", claims: p => p.EndsWith(".raw", StringComparison.OrdinalIgnoreCase)));

        Assert.Equal("thermo", Ms2SignalReaders.Read("run.raw").Reader);
        Assert.Equal("mzml", Ms2SignalReaders.Read("run.mzML").Reader);
        Assert.Equal(Ms2ReadStatus.NoReader, Ms2SignalReaders.Read("run.wiff").Status);
    }

    /// <summary>
    /// A reader that throws while PROBING is a broken reader, not a reason to abandon the file - the
    /// next candidate still gets its turn. One bad reader must not take a 192-replicate cohort with it.
    /// </summary>
    [Fact]
    public void AReaderThatThrowsWhileProbingDoesNotBlockTheOthers()
    {
        Ms2SignalReaders.Register(new FakeReader("good"));
        Ms2SignalReaders.Register(new FakeReader(
            "broken", claims: _ => throw new InvalidOperationException("native load failed")));

        var record = Ms2SignalReaders.Read("run.raw");

        Assert.Equal("good", record.Reader);
        Assert.Equal(Ms2ReadStatus.Ok, record.Status);
    }

    /// <summary>
    /// The contract is that a failed read is a RECORD, not an exception. Pinned because the caller
    /// loops over a whole cohort and one unreadable file must cost only that file.
    /// </summary>
    [Fact]
    public void AFailedReadIsReportedAsARecord()
    {
        var record = Ms2SignalRecord.Unavailable(
            "gone.raw", Ms2ReadStatus.NotFound, "fake 1.0", "no such file");

        Assert.False(record.IsUsable);
        Assert.Equal(Ms2ReadStatus.NotFound, record.Status);
        Assert.Equal("no such file", record.Message);
        Assert.Equal(Ms2SignalSource.Unknown, record.SignalSource);
        Assert.Empty(record.IsolationWindows);
        Assert.True(double.IsNaN(record.RtStartMin));
    }

    /// <summary>An OK record with no signal is still not usable - a zero denominator is not an answer.</summary>
    [Fact]
    public void AnEmptyRunIsNotUsableEvenWhenTheReadSucceeded()
    {
        var record = Ok("run.raw", "fake") with { TotalMs2Signal = 0 };
        Assert.Equal(Ms2ReadStatus.Ok, record.Status);
        Assert.False(record.IsUsable);
    }

    [Fact]
    public void RegisteringNullIsRejected()
        => Assert.Throws<ArgumentNullException>(() => Ms2SignalReaders.Register(null!));
}
