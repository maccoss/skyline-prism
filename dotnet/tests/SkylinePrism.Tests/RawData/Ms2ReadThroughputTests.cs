using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using SkylinePrism.Core.RawData;
using SkylinePrism.Pwiz;
using Xunit;
using Xunit.Abstractions;

namespace SkylinePrism.Tests.RawData;

/// <summary>
/// Does reading a cohort's instrument files benefit from being parallelized, and why?
///
/// <para>Opt-in: <c>PRISM_MS2_RAW_DIR</c>, with <c>PRISM_MS2_BENCH_FILES</c> (default 2) and
/// <c>PRISM_MS2_BENCH_PARALLEL</c> (default 2). Reads real files off real storage, so it takes
/// minutes per file and is skipped everywhere by default.</para>
/// </summary>
/// <remarks>
/// <para><b>The mechanism first.</b> CPU time against wall time says what the read is waiting on, and
/// that decides the answer before any A/B: a read that saturates a core is compute-bound in the vendor
/// SDK and will scale with cores, while one using a fraction of a core is waiting on storage, where
/// concurrent readers on a network share are as likely to contend as to help.</para>
/// <para><b>Parallel arm runs FIRST, deliberately.</b> Both arms read the same files, so any OS or
/// share caching benefits whichever runs second - which is the sequential arm. That biases the
/// comparison AGAINST parallelism, so a parallel win cannot be a caching artifact. The reverse order
/// would have been the flattering one.</para>
/// </remarks>
public class Ms2ReadThroughputTests
{
    private const string RawVar = "PRISM_MS2_RAW_DIR";

    private readonly ITestOutputHelper _out;

    public Ms2ReadThroughputTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void IsParallelReadingWorthIt()
    {
        var rawDir = Environment.GetEnvironmentVariable(RawVar);
        if (string.IsNullOrWhiteSpace(rawDir))
        {
            _out.WriteLine($"skipped: set {RawVar} to a directory of instrument files.");
            return;
        }
        if (!PwizReaderRegistration.IsAvailable)
        {
            _out.WriteLine("skipped: built without pwiz-sharp.");
            return;
        }
        PwizReaderRegistration.Register();

        var count = Env("PRISM_MS2_BENCH_FILES", 2);
        var parallel = Env("PRISM_MS2_BENCH_PARALLEL", 2);

        var files = Directory.GetFiles(rawDir, "*.raw")
            .Concat(Directory.GetFiles(rawDir, "*.mzML"))
            .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
            .Take(count)
            .ToList();
        Assert.True(files.Count > 1, $"need at least 2 files in {rawDir}");

        var bytes = files.Sum(f => new FileInfo(f).Length);
        _out.WriteLine($"{files.Count} files, {bytes / 1073741824.0:0.00} GB total, "
            + $"{Environment.ProcessorCount} logical cores");
        _out.WriteLine($"parallelism under test: {parallel}");
        _out.WriteLine("");

        // Parallel first - see the remarks on why that order is the conservative one.
        var par = Measure($"parallel x{parallel}", files, bytes,
            () => Parallel.ForEach(
                files, new ParallelOptions { MaxDegreeOfParallelism = parallel },
                f => Consume(Ms2SignalReaders.Read(f))));

        var seq = Measure("sequential", files, bytes,
            () => { foreach (var f in files) Consume(Ms2SignalReaders.Read(f)); });

        _out.WriteLine("");
        _out.WriteLine($"speedup: {seq.Wall.TotalSeconds / par.Wall.TotalSeconds:0.00}x "
            + $"({seq.Wall.TotalSeconds:0} s -> {par.Wall.TotalSeconds:0} s)");
        _out.WriteLine($"cores busy, sequential : {seq.Cores:0.00}");
        _out.WriteLine($"cores busy, parallel   : {par.Cores:0.00}");
        _out.WriteLine("");
        _out.WriteLine(seq.Cores > 0.8
            ? "A single read saturates a core, so the cost is compute in the vendor SDK and "
              + "parallelism scales with cores rather than fighting the storage."
            : $"A single read uses only {seq.Cores:0.00} of a core, so it is waiting on storage. "
              + "The speedup above is what the share will actually give - trust it over the core count.");

        // Both arms must have done the same work, or the comparison is meaningless. Bytes are the
        // check that survives a differing file set.
        Assert.Equal(seq.Spectra, par.Spectra);
        Assert.Equal(seq.Signal, par.Signal, 3);
        _out.WriteLine($"same work both arms: {seq.Spectra:N0} MS2 spectra, MS2 TIC {seq.Signal:E4}");
    }

    private sealed record Arm(TimeSpan Wall, double Cores, long Spectra, double Signal);

    private long _spectra;
    private double _signal;

    private void Consume(Ms2SignalRecord record)
    {
        lock (this)
        {
            _spectra += record.Ms2Count;
            _signal += record.TotalMs2Signal;
        }
    }

    private Arm Measure(string label, IReadOnlyList<string> files, long bytes, Action run)
    {
        _spectra = 0;
        _signal = 0;

        var process = Process.GetCurrentProcess();
        var cpuBefore = process.TotalProcessorTime;
        var clock = Stopwatch.StartNew();
        run();
        clock.Stop();
        var cpu = Process.GetCurrentProcess().TotalProcessorTime - cpuBefore;

        var cores = cpu.TotalSeconds / clock.Elapsed.TotalSeconds;
        _out.WriteLine($"{label,-16} {clock.Elapsed.TotalSeconds,7:0.0} s  "
            + $"{bytes / 1048576.0 / clock.Elapsed.TotalSeconds,6:0.0} MB/s  "
            + $"cpu {cpu.TotalSeconds,7:0.0} s ({cores:0.00} cores busy)");

        return new Arm(clock.Elapsed, cores, _spectra, _signal);
    }

    private static int Env(string name, int fallback) =>
        int.TryParse(Environment.GetEnvironmentVariable(name), out var v) && v > 0 ? v : fallback;
}
