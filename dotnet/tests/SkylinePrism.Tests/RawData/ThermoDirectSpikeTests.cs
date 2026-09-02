using System;
using System.Collections.Generic;
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
    /// <summary>
    /// Parallelism, re-tested properly. The earlier arms shared their files and ran parallel FIRST,
    /// which I chose so caching would favour the sequential arm - reasoning that a parallel win
    /// would then be trustworthy. That reasoning was wrong in one important way: this read touches
    /// only a few MB per file (index plus chromatogram), not the 3.3 GB, and a few MB per file for
    /// eight files sits comfortably in the page cache. So the sequential arm was very likely reading
    /// from RAM what the parallel arm had just fetched over SMB - a large systematic bias against
    /// parallelism, not a conservative one.
    ///
    /// <para>Fixed two ways. First it measures the caching directly, by reading one set twice. Then
    /// it compares arms on DISJOINT file sets so neither inherits the other's cache, and runs the
    /// orderings both ways round so any residual order effect cancels rather than accumulating.</para>
    /// </summary>
    [Fact]
    public void DoesTheDirectRouteParalleliseOnDisjointFiles()
    {
        var rawDir = Environment.GetEnvironmentVariable("PRISM_MS2_RAW_DIR");
        if (string.IsNullOrWhiteSpace(rawDir))
        {
            _out.WriteLine("skipped: set PRISM_MS2_RAW_DIR.");
            return;
        }

        var perArm = int.TryParse(Environment.GetEnvironmentVariable("PRISM_MS2_ARM_FILES"), out var a)
            ? a : 10;
        var lanes = int.TryParse(Environment.GetEnvironmentVariable("PRISM_MS2_LANES"), out var l)
            ? l : 8;

        var all = System.IO.Directory.GetFiles(rawDir, "*.raw")
            .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
            .ToList();
        Assert.True(all.Count >= perArm * 4,
            $"need {perArm * 4} files for four disjoint arms, found {all.Count}");

        _out.WriteLine($"{perArm} files per arm, {lanes} lanes, "
            + $"{Environment.ProcessorCount} logical cores");
        _out.WriteLine("");

        // 1. How much does the page cache actually give? Same set, twice, sequentially.
        var setC = all.Skip(perArm * 4).Take(perArm).ToList();
        if (setC.Count == perArm)
        {
            var cold = Time(() => Sequential(setC));
            var warm = Time(() => Sequential(setC));
            _out.WriteLine($"caching check (same {perArm} files, sequential twice):");
            _out.WriteLine($"  first pass  {cold:0.00} s ({cold / perArm:0.00} s/file)");
            _out.WriteLine($"  second pass {warm:0.00} s ({warm / perArm:0.00} s/file)");
            _out.WriteLine($"  the cache is worth {cold / Math.Max(1e-6, warm):0.00}x");
            _out.WriteLine("");
        }

        // 2. Disjoint arms, both orderings. Sequential first in the first pair, parallel first in
        //    the second, so an order effect shows up as disagreement between the pairs.
        var setA = all.Take(perArm).ToList();
        var setB = all.Skip(perArm).Take(perArm).ToList();
        var setD = all.Skip(perArm * 2).Take(perArm).ToList();
        var setE = all.Skip(perArm * 3).Take(perArm).ToList();

        var seq1 = Time(() => Sequential(setA));
        var par1 = Time(() => InParallel(setB, lanes));
        var par2 = Time(() => InParallel(setD, lanes));
        var seq2 = Time(() => Sequential(setE));

        var seq = (seq1 + seq2) / 2;
        var par = (par1 + par2) / 2;

        _out.WriteLine($"sequential : {seq1:0.00} s and {seq2:0.00} s  -> mean {seq:0.00} s "
            + $"({seq / perArm:0.00} s/file)");
        _out.WriteLine($"parallel   : {par1:0.00} s and {par2:0.00} s  -> mean {par:0.00} s "
            + $"({par / perArm:0.00} s/file)");
        _out.WriteLine($"speedup    : {seq / par:0.00}x");
        _out.WriteLine("");
        _out.WriteLine($"projected for 93 replicates: sequential {seq / perArm * 93:0} s, "
            + $"parallel {par / perArm * 93:0} s");
        _out.WriteLine("");
        _out.WriteLine(par < seq * 0.9
            ? "VERDICT: parallelism helps on disjoint files. The earlier arms were measuring the "
              + "page cache, not the concurrency."
            : "VERDICT: parallelism still does not help, even with the cache bias removed and the "
              + "orderings balanced.");
    }

    private static double Time(Action work)
    {
        var clock = Stopwatch.StartNew();
        work();
        clock.Stop();
        return clock.Elapsed.TotalSeconds;
    }

    private static double Sequential(IReadOnlyList<string> files)
    {
        var total = 0.0;
        foreach (var f in files)
            total += DirectMs2Total(f);
        return total;
    }

    private static double InParallel(IReadOnlyList<string> files, int lanes)
    {
        var bag = new System.Collections.Concurrent.ConcurrentBag<double>();
        System.Threading.Tasks.Parallel.ForEach(
            files,
            new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = lanes },
            f => bag.Add(DirectMs2Total(f)));
        return bag.Sum();
    }
    /// <summary>
    /// How many concurrent reads is best on this share? Each lane count gets its OWN disjoint set of
    /// files, so no arm reads what another has already pulled into the page cache - which is worth
    /// 2.13x here and is what invalidated the first parallel measurements.
    /// </summary>
    [Fact]
    public void HowManyLanesIsBest()
    {
        var rawDir = Environment.GetEnvironmentVariable("PRISM_MS2_RAW_DIR");
        if (string.IsNullOrWhiteSpace(rawDir))
        {
            _out.WriteLine("skipped: set PRISM_MS2_RAW_DIR.");
            return;
        }

        var perArm = int.TryParse(Environment.GetEnvironmentVariable("PRISM_MS2_ARM_FILES"), out var a)
            ? a : 8;
        var lanesToTry = new[] { 1, 2, 4, 8, 16 };

        var all = System.IO.Directory.GetFiles(rawDir, "*.raw")
            .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
            .ToList();
        var needed = perArm * lanesToTry.Length;
        Assert.True(all.Count >= needed,
            $"need {needed} files for {lanesToTry.Length} disjoint arms of {perArm}, found {all.Count}");

        _out.WriteLine($"{perArm} cold files per arm, disjoint sets, "
            + $"{Environment.ProcessorCount} logical cores");
        _out.WriteLine("");
        _out.WriteLine("lanes   time    s/file   speedup   projected 93");

        var baseline = double.NaN;
        var best = (Lanes: 1, PerFile: double.MaxValue);
        for (var i = 0; i < lanesToTry.Length; i++)
        {
            var lanes = lanesToTry[i];
            var set = all.Skip(perArm * i).Take(perArm).ToList();
            var seconds = Time(() => InParallel(set, lanes));
            var perFile = seconds / perArm;
            if (lanes == 1)
                baseline = perFile;
            if (perFile < best.PerFile)
                best = (lanes, perFile);

            _out.WriteLine($"{lanes,5}  {seconds,6:0.00} s  {perFile,6:0.00}   "
                + $"{baseline / perFile,6:0.00}x   {perFile * 93,8:0} s");
        }

        _out.WriteLine("");
        _out.WriteLine($"best: {best.Lanes} lanes at {best.PerFile:0.00} s/file "
            + $"-> {best.PerFile * 93:0} s for the cohort, {baseline / best.PerFile:0.00}x over serial");
    }
}
