using System;
using System.Diagnostics;
using System.Linq;
using SkylinePrism.Core.RawData;
using SkylinePrism.Pwiz;
using ThermoFisher.CommonCore.Data;
using ThermoFisher.CommonCore.Data.Business;
using ThermoFisher.CommonCore.Data.FilterEnums;
using ThermoFisher.CommonCore.RawFileReader;
using Xunit;
using Xunit.Abstractions;

namespace SkylinePrism.Tests.RawData;

/// <summary>
/// A SPIKE, not a feature: can the acquired MS2 total be had substantially faster by asking the
/// Thermo SDK for a scan-filtered TIC directly, instead of going through pwiz?
///
/// <para>Why it might be: pwiz's <c>Read</c> eagerly constructs <c>SpectrumList_Thermo</c>, whose
/// <c>CreateIndex</c> loops over every scan in the file calling <c>GetFilterForScanNumber</c> -
/// 166,252 SDK calls on a FLARE file - and there is no ReaderConfig switch to skip it. That index
/// build is ~1.5 s of a ~1.6 s read. Worse, it already computes each scan's MSOrder, which pwiz's
/// TIC chromatogram then recomputes in a second pass over the same scans.</para>
///
/// <para>This measures the ceiling before any production code is written for it. If the win is real
/// AND the total matches the pwiz path exactly, it is worth a Thermo-specific fast path; if not,
/// this test is the record of why not.</para>
/// </summary>
public class ThermoDirectSpikeTests
{
    private const string FileVar = "PRISM_MS2_RAW_FILE";

    private readonly ITestOutputHelper _out;

    public ThermoDirectSpikeTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void CanTheThermoSdkGiveTheMs2TotalFaster()
    {
        var path = Environment.GetEnvironmentVariable(FileVar);
        if (string.IsNullOrWhiteSpace(path) || !PwizReaderRegistration.IsAvailable)
        {
            _out.WriteLine($"skipped: needs {FileVar} and a pwiz build.");
            return;
        }

        // Reference: the current implementation, through pwiz.
        PwizReaderRegistration.Register();
        var viaPwizClock = Stopwatch.StartNew();
        var viaPwiz = Ms2SignalReaders.Read(path);
        viaPwizClock.Stop();

        // Candidate: the SDK's own filtered TIC. No spectrum list, so no index build.
        var directClock = Stopwatch.StartNew();
        double directTotal = 0;
        int directPoints = 0, ms1Points = 0;
        double firstTime = double.NaN, lastTime = double.NaN;

        var raw = RawFileReaderAdapter.FileFactory(path);
        try
        {
            raw.SelectInstrument(Device.MS, 1);

            // Two traces in ONE call: MS2 carries the signal, MS1 marks the cycle boundaries that
            // Plot B needs. The filter string is Thermo's own abbreviated form.
            var ms2 = new ChromatogramTraceSettings(TraceType.TIC) { Filter = "ms2" };
            var ms1 = new ChromatogramTraceSettings(TraceType.TIC) { Filter = "ms" };
            var data = raw.GetChromatogramDataEx(
                new ThermoFisher.CommonCore.Data.Interfaces.IChromatogramSettingsEx[] { ms2, ms1 },
                -1, -1, new MassOptions());

            if (data?.IntensitiesArray?.Length > 0 && data.IntensitiesArray[0] is { } intensities
                && data.PositionsArray?[0] is { } times)
            {
                directPoints = intensities.Length;
                foreach (var v in intensities)
                    directTotal += v;
                if (times.Length > 0)
                {
                    firstTime = times[0];
                    lastTime = times[times.Length - 1];
                }
            }
            if (data?.PositionsArray?.Length > 1 && data.PositionsArray[1] is { } ms1Times)
                ms1Points = ms1Times.Length;
        }
        finally
        {
            raw.Dispose();
        }
        directClock.Stop();

        _out.WriteLine($"via pwiz     : {viaPwiz.TotalMs2Signal:R}");
        _out.WriteLine($"               {viaPwiz.Ms2Count:N0} MS2, {viaPwiz.Ms1Count:N0} MS1, "
            + $"{viaPwizClock.Elapsed.TotalSeconds:0.00} s");
        _out.WriteLine($"thermo direct: {directTotal:R}");
        _out.WriteLine($"               {directPoints:N0} MS2 points, {ms1Points:N0} MS1 points, "
            + $"{directClock.Elapsed.TotalSeconds:0.00} s  (RT {firstTime:0.00}-{lastTime:0.00} min)");
        _out.WriteLine("");
        _out.WriteLine($"speedup      : "
            + $"{viaPwizClock.Elapsed.TotalSeconds / Math.Max(1e-6, directClock.Elapsed.TotalSeconds):0.0}x");
        var relative = viaPwiz.TotalMs2Signal > 0
            ? Math.Abs(directTotal - viaPwiz.TotalMs2Signal) / viaPwiz.TotalMs2Signal
            : double.NaN;
        _out.WriteLine($"relative difference in the total: {relative:E2}");

        // A faster wrong answer is worthless, so the numbers decide this, not the clock. A spike is
        // allowed to report a mismatch rather than assert - that is the finding either way.
        if (directPoints == 0)
        {
            _out.WriteLine(
                "VERDICT: the filtered TIC came back empty - the abbreviated filter string is not "
                + "matching. Not viable as written.");
            return;
        }
        _out.WriteLine(relative < 1e-12
            ? "VERDICT: same total, and the counts should match too - a Thermo fast path is viable."
            : "VERDICT: totals DISAGREE. The filtered trace is not the same population as the "
              + "per-scan MS2 TIC; not viable without understanding why.");
    }
    /// <summary>
    /// Does the direct route parallelise? This is the question the earlier benchmark answered NO to,
    /// and it deserves re-asking: that benchmark measured the per-spectrum walk, which saturated a
    /// core, so concurrency only added contention. The direct read does almost no compute - it is
    /// dominated by opening the file over SMB - and latency overlaps where compute does not.
    /// </summary>
    [Fact]
    public void DoesTheDirectRouteParallelise()
    {
        var rawDir = Environment.GetEnvironmentVariable("PRISM_MS2_RAW_DIR");
        if (string.IsNullOrWhiteSpace(rawDir))
        {
            _out.WriteLine("skipped: set PRISM_MS2_RAW_DIR.");
            return;
        }

        var files = System.IO.Directory.GetFiles(rawDir, "*.raw")
            .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
            .Take(int.TryParse(Environment.GetEnvironmentVariable("PRISM_MS2_BENCH_FILES"), out var n)
                ? n : 8)
            .ToList();
        Assert.True(files.Count > 1);

        // Parallel FIRST, so any caching favours the sequential arm and cannot manufacture a win.
        var parallelTotals = new System.Collections.Concurrent.ConcurrentBag<double>();
        var parClock = Stopwatch.StartNew();
        System.Threading.Tasks.Parallel.ForEach(
            files,
            new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = files.Count },
            f => parallelTotals.Add(DirectMs2Total(f)));
        parClock.Stop();

        var seqTotals = new List<double>();
        var seqClock = Stopwatch.StartNew();
        foreach (var f in files)
            seqTotals.Add(DirectMs2Total(f));
        seqClock.Stop();

        _out.WriteLine($"{files.Count} files, direct route");
        _out.WriteLine($"  parallel x{files.Count,-3} {parClock.Elapsed.TotalSeconds,6:0.00} s  "
            + $"({parClock.Elapsed.TotalSeconds / files.Count:0.00} s/file)");
        _out.WriteLine($"  sequential    {seqClock.Elapsed.TotalSeconds,6:0.00} s  "
            + $"({seqClock.Elapsed.TotalSeconds / files.Count:0.00} s/file)");
        _out.WriteLine($"  speedup: {seqClock.Elapsed.TotalSeconds / parClock.Elapsed.TotalSeconds:0.00}x");
        _out.WriteLine("");
        _out.WriteLine($"  projected for 93 replicates: "
            + $"sequential {seqClock.Elapsed.TotalSeconds / files.Count * 93:0} s, "
            + $"parallel {parClock.Elapsed.TotalSeconds / files.Count * 93:0} s");

        // Same work both arms, or the timing means nothing.
        Assert.Equal(seqTotals.Count, parallelTotals.Count);
        Assert.Equal(seqTotals.Sum(), parallelTotals.Sum(), 3);
    }

    /// <summary>The MS2 total for one file, by the direct route. No pwiz, no spectrum index.</summary>
    private static double DirectMs2Total(string path)
    {
        var raw = RawFileReaderAdapter.FileFactory(path);
        try
        {
            raw.SelectInstrument(Device.MS, 1);
            var ms2 = new ChromatogramTraceSettings(TraceType.TIC) { Filter = "ms2" };
            var data = raw.GetChromatogramDataEx(
                new ThermoFisher.CommonCore.Data.Interfaces.IChromatogramSettingsEx[] { ms2 },
                -1, -1, new MassOptions());

            var total = 0.0;
            if (data?.IntensitiesArray?.Length > 0 && data.IntensitiesArray[0] is { } intensities)
                foreach (var v in intensities)
                    total += v;
            return total;
        }
        finally
        {
            raw.Dispose();
        }
    }
}
