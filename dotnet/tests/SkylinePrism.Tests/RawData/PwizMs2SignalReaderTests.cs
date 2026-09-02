using System;
using System.IO;
using System.Linq;
using SkylinePrism.Core.RawData;
using SkylinePrism.Pwiz;
using Xunit;
using Xunit.Abstractions;

namespace SkylinePrism.Tests.RawData;

/// <summary>
/// The pwiz reader against a real instrument file, opt-in and read-only.
///
/// <para>Set <c>PRISM_MS2_RAW_FILE</c> to a data file. Unset in CI, and skipped in a build with no
/// pwiz-sharp checkout - which is the shipped state of the cross-platform CLI, so both skips are
/// normal rather than a gap.</para>
///
/// <para>Everything here goes through <see cref="Ms2SignalReaders"/> rather than naming the concrete
/// reader. That is not stylistic: a build without pwiz-sharp does not compile that type at all, so a
/// direct reference here would stop the whole test project building in exactly the configuration the
/// cross-platform CLI ships.</para>
/// </summary>
public class PwizMs2SignalReaderTests
{
    private const string FileVar = "PRISM_MS2_RAW_FILE";

    private readonly ITestOutputHelper _out;

    public PwizMs2SignalReaderTests(ITestOutputHelper output) => _out = output;

    /// <summary>Registration works the same in both builds; only the outcome differs.</summary>
    [Fact]
    public void RegistrationIsIdempotentAndMatchesTheBuild()
    {
        PwizReaderRegistration.Register();
        PwizReaderRegistration.Register();

        if (PwizReaderRegistration.IsAvailable)
        {
            // Exactly one, however many times Register was called.
            Assert.Single(Ms2SignalReaders.All, r => r.Describe().Contains("pwiz"));
            Assert.NotNull(Ms2SignalReaders.For("run.raw"));
        }
        else
        {
            // A build with no pwiz reports the acquired total as unknown, which is the correct
            // answer rather than a failure.
            _out.WriteLine("built without pwiz-sharp: no reader registered.");
        }
    }

    /// <summary>A format nothing here can open is not claimed, so the caller falls through cleanly.</summary>
    [Fact]
    public void UnknownExtensionsAreNotClaimed()
    {
        if (!PwizReaderRegistration.IsAvailable)
            return;

        PwizReaderRegistration.Register();
        var reader = Ms2SignalReaders.For("run.raw");
        Assert.NotNull(reader);
        Assert.True(reader!.CanRead("run.raw"));
        Assert.True(reader.CanRead("run.mzML"));
        Assert.True(reader.CanRead(@"C:\data\run.d"));
        Assert.False(reader.CanRead("report.csv"));
        Assert.False(reader.CanRead("peptides.parquet"));
        Assert.False(reader.CanRead(""));
    }

    /// <summary>A missing file is a record, never an exception - one bad file must cost only itself.</summary>
    [Fact]
    public void AMissingFileIsReportedNotThrown()
    {
        if (!PwizReaderRegistration.IsAvailable)
            return;

        PwizReaderRegistration.Register();
        var record = Ms2SignalReaders.Read(
            Path.Combine(Path.GetTempPath(), "no-such-" + Guid.NewGuid() + ".raw"));

        Assert.Equal(Ms2ReadStatus.NotFound, record.Status);
        Assert.False(record.IsUsable);
    }

    /// <summary>
    /// The measurement that matters: the acquired MS2 total ion current, which is the denominator no
    /// Skyline export can supply.
    /// </summary>
    [Fact]
    public void ReadsAcquiredMs2SignalFromARealFile()
    {
        var path = Environment.GetEnvironmentVariable(FileVar);
        if (string.IsNullOrWhiteSpace(path))
        {
            _out.WriteLine($"skipped: set {FileVar} to an instrument data file.");
            return;
        }
        if (!PwizReaderRegistration.IsAvailable)
        {
            _out.WriteLine("skipped: built without pwiz-sharp.");
            return;
        }
        Assert.True(File.Exists(path) || Directory.Exists(path), $"no such file: {path}");

        PwizReaderRegistration.Register();
        var started = DateTime.UtcNow;
        var record = Ms2SignalReaders.Read(path, _out.WriteLine);
        var elapsed = DateTime.UtcNow - started;

        _out.WriteLine($"file          : {path}");
        _out.WriteLine($"size          : {new FileInfo(path).Length / 1024.0 / 1024:N0} MB");
        _out.WriteLine($"status        : {record.Status} ({record.Reader}, {record.SignalSource})");
        _out.WriteLine($"MS1 / MS2     : {record.Ms1Count:N0} / {record.Ms2Count:N0}");
        _out.WriteLine($"MS2 TIC       : {record.TotalMs2Signal:E4}");
        _out.WriteLine($"RT range      : {record.RtStartMin:0.00} - {record.RtStopMin:0.00} min");
        _out.WriteLine($"cycles        : {record.Cycles.Count:N0} ({record.CycleModel})");
        _out.WriteLine($"windows       : {record.IsolationWindows.Count:N0}");
        if (record.IsolationWindows.Count > 0)
        {
            var w = record.IsolationWindows;
            _out.WriteLine($"  {w[0].Start:0.000}-{w[0].End:0.000} .. {w[^1].Start:0.000}-{w[^1].End:0.000}"
                + $", median width {w.Select(x => x.Width).OrderBy(x => x).ElementAt(w.Count / 2):0.000} Th");
        }
        _out.WriteLine($"elapsed       : {elapsed.TotalSeconds:0.0} s");

        Assert.Equal(Ms2ReadStatus.Ok, record.Status);
        Assert.True(record.IsUsable);
        Assert.True(record.Ms2Count > 0);
        Assert.True(record.TotalMs2Signal > 0);

        // The per-cycle totals must add up to the run total, or the trace in Plot B would disagree
        // with the bar in Plot A.
        var fromCycles = record.Cycles.Sum(c => c.Ms2Signal);
        Assert.Equal(record.TotalMs2Signal, fromCycles, 3);
        Assert.Equal(record.Ms2Count, record.Cycles.Sum(c => c.Ms2Count));
    }
}
