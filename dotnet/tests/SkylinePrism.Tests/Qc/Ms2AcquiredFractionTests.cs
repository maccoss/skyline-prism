using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.RawData;
using SkylinePrism.Pwiz;
using Xunit;
using Xunit.Abstractions;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// The whole question, end to end: what fraction of the MS2 the instrument acquired does an analysis
/// actually assign to a peptide?
///
/// <para>Opt-in and read-only. <c>PRISM_MS2_OUTPUT_DIR</c> is a completed PRISM output directory and
/// <c>PRISM_MS2_RAW_DIR</c> the directory holding that cohort's data files; <c>PRISM_MS2_MAX_FILES</c>
/// caps how many are read (default 3), because a file takes minutes and a cohort is hundreds of
/// gigabytes. Skipped in CI and in a build with no pwiz-sharp.</para>
/// </summary>
public class Ms2AcquiredFractionTests
{
    private const string OutputVar = "PRISM_MS2_OUTPUT_DIR";
    private const string RawVar = "PRISM_MS2_RAW_DIR";
    private const string MaxVar = "PRISM_MS2_MAX_FILES";

    private readonly ITestOutputHelper _out;

    public Ms2AcquiredFractionTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void AssignedAsAFractionOfAcquired()
    {
        var dir = Environment.GetEnvironmentVariable(OutputVar);
        var rawDir = Environment.GetEnvironmentVariable(RawVar);
        if (string.IsNullOrWhiteSpace(dir) || string.IsNullOrWhiteSpace(rawDir))
        {
            _out.WriteLine($"skipped: set {OutputVar} and {RawVar}.");
            return;
        }
        if (!PwizReaderRegistration.IsAvailable)
        {
            _out.WriteLine("skipped: built without pwiz-sharp.");
            return;
        }
        PwizReaderRegistration.Register();

        var max = int.TryParse(Environment.GetEnvironmentVariable(MaxVar), out var m) ? m : 3;

        // The accounting, from the cached results if the run has them - recomputing 93 replicates to
        // read 3 raw files would be the slow part for no reason.
        var accounting = Ms2SignalAccounting.ReadCached(dir);
        if (accounting is null)
        {
            var scheme = IsolationSchemeCatalog
                .Load(Path.Combine(dir, IsolationSchemeCatalog.FileName))!
                .UsableSchemes.Single();
            accounting = Ms2SignalAccounting.Compute(
                dir, scheme, ProductMassTolerance.ParseSetting("10 ppm")!,
                Array.Empty<ProteinList>(), null, _out.WriteLine);
        }
        Assert.NotNull(accounting);

        var files = Directory.GetFiles(rawDir, "*.raw")
            .Concat(Directory.GetFiles(rawDir, "*.mzML"))
            .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
            .ToList();
        _out.WriteLine($"{accounting!.Rows.Count} replicates, {files.Count} data files in {rawDir}");
        _out.WriteLine($"tolerance {accounting.Tolerance}, scheme \"{accounting.IsolationScheme}\"");
        _out.WriteLine("");

        var matched = 0;
        var fractions = new List<double>();
        foreach (var row in accounting.Rows)
        {
            if (matched >= max)
                break;

            var path = ResolveDataFile(row.Sample, files);
            if (path is null)
                continue;
            matched++;

            var record = Ms2SignalReaders.Read(path);
            if (!record.IsUsable)
            {
                _out.WriteLine($"{Replicate(row.Sample)}: could not read {Path.GetFileName(path)} "
                    + $"({record.Status}: {record.Message})");
                continue;
            }

            var fraction = row.AssignedArea / record.TotalMs2Signal;
            fractions.Add(fraction);

            _out.WriteLine($"{Replicate(row.Sample)}  ({row.SampleType})");
            _out.WriteLine($"  acquired MS2 TIC : {record.TotalMs2Signal:E4}  "
                + $"({record.Ms2Count:N0} spectra, {record.Cycles.Count:N0} cycles, {record.Reader})");
            _out.WriteLine($"  assigned to a peptide : {row.AssignedArea:E4}");
            _out.WriteLine($"  naive sum (over-counts): {row.SummedArea:E4}");
            _out.WriteLine($"  ASSIGNED / ACQUIRED   : {fraction:P2}");
            _out.WriteLine($"  summed / acquired     : {row.SummedArea / record.TotalMs2Signal:P2} "
                + "(what a sum would have claimed)");

            // The scheme the reader found in the file must be the one the accounting used, or the two
            // halves of the fraction are not describing the same acquisition.
            _out.WriteLine($"  isolation windows in the file: {record.IsolationWindows.Count}");
            _out.WriteLine("");

            Assert.True(fraction > 0 && fraction < 1.0,
                $"{row.Sample}: assigned/acquired = {fraction}, which is not a fraction");
        }

        Assert.True(matched > 0, "no replicate could be matched to a data file");
        _out.WriteLine($"median assigned/acquired over {fractions.Count} replicate(s): "
            + $"{Median(fractions):P2}");
    }

    /// <summary>The replicate part of a PRISM sample id (<c>replicate__@__batch</c>).</summary>
    private static string Replicate(string sampleId)
    {
        var i = sampleId.IndexOf("__@__", StringComparison.Ordinal);
        return i < 0 ? sampleId : sampleId[..i];
    }

    /// <summary>
    /// The data file for a replicate. Matched on the file STEM containing the replicate name rather
    /// than equalling it: acquisition software routinely prefixes the run (this cohort's replicate
    /// FLARE-001-1-B1-013 is 2026-extended-FLARE-001-1-B1-013.raw), and Skyline stores the replicate
    /// name, not the file name. The longest match wins, so a replicate that is a prefix of another
    /// does not steal its file.
    /// </summary>
    private static string? ResolveDataFile(string sampleId, IReadOnlyList<string> files)
    {
        var replicate = Replicate(sampleId);
        string? best = null;
        var bestLength = -1;

        foreach (var file in files)
        {
            var stem = Path.GetFileNameWithoutExtension(file);
            if (string.Equals(stem, replicate, StringComparison.OrdinalIgnoreCase))
                return file;
            if (stem.EndsWith(replicate, StringComparison.OrdinalIgnoreCase) && stem.Length > bestLength)
            {
                best = file;
                bestLength = stem.Length;
            }
        }
        return best;
    }

    private static double Median(IReadOnlyList<double> values)
    {
        if (values.Count == 0)
            return double.NaN;
        var sorted = values.OrderBy(v => v).ToArray();
        var mid = sorted.Length / 2;
        return sorted.Length % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    }
}
