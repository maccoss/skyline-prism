using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.RawData;
using SkylinePrism.Core.Visualization;
using SkylinePrism.Pwiz;
using Xunit;
using Xunit.Abstractions;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// Plot B end to end on a real replicate: acquired MS2 from the instrument file, assigned signal from
/// the run, and a protein panel over the top. Opt-in and read-only.
///
/// <para><c>PRISM_MS2_OUTPUT_DIR</c> is a completed PRISM output directory, <c>PRISM_MS2_RAW_DIR</c>
/// the matching data files. Set <c>PRISM_MS2_PLOT_OUT</c> to write the PNG somewhere lookable - a
/// profile whose numbers are right can still be unreadable, and no assertion catches that.</para>
/// </summary>
public class Ms2SignalProfileRealTests
{
    private readonly ITestOutputHelper _out;

    public Ms2SignalProfileRealTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void ProfilesARealReplicate()
    {
        var dir = Environment.GetEnvironmentVariable("PRISM_MS2_OUTPUT_DIR");
        var rawDir = Environment.GetEnvironmentVariable("PRISM_MS2_RAW_DIR");
        if (string.IsNullOrWhiteSpace(dir) || string.IsNullOrWhiteSpace(rawDir)
            || !PwizReaderRegistration.IsAvailable)
        {
            _out.WriteLine("skipped: needs PRISM_MS2_OUTPUT_DIR, PRISM_MS2_RAW_DIR and a pwiz build.");
            return;
        }
        PwizReaderRegistration.Register();

        var scheme = IsolationSchemeCatalog
            .Load(Path.Combine(dir, IsolationSchemeCatalog.FileName))!.UsableSchemes.Single();
        var tolerance = ProductMassTolerance.ParseSetting("10 ppm")!;

        var dataset = MergedDataset.Open(Path.Combine(dir, "merged_data"));
        var cols = Ms2SignalRegions.Resolve(
            ParquetTable.ReadColumnNames(dataset.RepresentativeFile()).ToList())!;

        // A panel built from the cohort's own protein groups, so it matches something real.
        var lists = OnePanel(dir);
        var classified = Ms2SignalPeptides.Classify(dir, lists);

        // The first replicate, and its data file.
        var sample = File.ReadAllLines(Path.Combine(dir, "sample_metadata.csv"))[1].Split(',')[0];
        var replicate = sample.Split("__@__")[0];
        var raw = Directory.GetFiles(rawDir, "*.raw")
            .FirstOrDefault(f => Path.GetFileNameWithoutExtension(f)
                .EndsWith(replicate, StringComparison.OrdinalIgnoreCase));
        Assert.NotNull(raw);

        var record = Ms2SignalReaders.Read(raw!);
        Assert.True(record.IsUsable);

        var loaded = Ms2SignalRegions.Load(dataset, cols, sample, scheme, classified.Classes);
        var merged = new List<Ms2SignalUnion.MergedRegion>();
        var union = Ms2SignalUnion.Compute(loaded.Regions, tolerance, lists.Count, merged.Add);

        var profile = Ms2SignalProfile.Build(
            sample, merged, record.Cycles,
            classified.ListNames, lists.Select(l => l.ColorHex).ToList(), binWidthMin: 0.25);

        _out.WriteLine($"replicate      : {replicate}");
        _out.WriteLine($"acquired cycles: {record.Cycles.Count:N0}, "
            + $"RT {record.RtStartMin:0.0}-{record.RtStopMin:0.0} min");
        _out.WriteLine($"merged regions : {merged.Count:N0}");
        _out.WriteLine($"bins           : {profile.BinCount} of {profile.BinWidthMin} min");
        _out.WriteLine($"acquired total : {profile.Acquired.Sum():E4}  "
            + $"(reader said {record.TotalMs2Signal:E4})");
        _out.WriteLine($"assigned total : {profile.Assigned.Sum():E4}  "
            + $"(union said {union.AssignedArea:E4})");
        _out.WriteLine($"panel          : {classified.ListNames.FirstOrDefault()} "
            + $"-> {profile.PerList.FirstOrDefault()?.Sum():E4}");

        // The traces must conserve the totals they came from, or Plot B contradicts Plot A.
        // RELATIVE tolerance: the trace sums bins and the reader sums cycles, so these are the same
        // ~166,000 doubles added in different orders at a magnitude of 1e11. Decimal places would
        // demand 15 significant digits - the same trap as in PwizMs2SignalReaderTests, which I had
        // already fixed there and then walked into again here.
        Assert.True(
            Math.Abs(record.TotalMs2Signal - profile.Acquired.Sum())
                <= 1e-12 * record.TotalMs2Signal,
            $"acquired trace sums to {profile.Acquired.Sum():R}, "
                + $"reader said {record.TotalMs2Signal:R}");
        Assert.True(
            Math.Abs(profile.Assigned.Sum() - union.AssignedArea) <= 1e-9 * union.AssignedArea,
            $"trace sums to {profile.Assigned.Sum():R}, union said {union.AssignedArea:R}");

        // Assigned can never exceed acquired in a bin. If it does, the two halves are not describing
        // the same acquisition - the wrong file, or the wrong isolation scheme.
        var overspill = Enumerable.Range(0, profile.BinCount)
            .Where(i => profile.Acquired[i] > 0 && profile.Assigned[i] > profile.Acquired[i])
            .ToList();
        _out.WriteLine($"bins where assigned exceeds acquired: {overspill.Count}");

        var fractions = profile.AssignedFraction.Where(double.IsFinite).ToList();
        _out.WriteLine($"assigned/acquired per bin: median {Median(fractions):P2}, "
            + $"min {fractions.Min():P2}, max {fractions.Max():P2}");

        if (Environment.GetEnvironmentVariable("PRISM_MS2_PLOT_OUT") is { Length: > 0 } dump)
        {
            Directory.CreateDirectory(dump);
            File.WriteAllBytes(
                Path.Combine(dump, "ms2_rt_profile_real.png"),
                PlotRenderer.Ms2RtProfilePng(
                    profile, $"MS2 Signal over Retention Time - {replicate}"));
        }

        Assert.True(overspill.Count < profile.BinCount * 0.05,
            $"{overspill.Count} of {profile.BinCount} bins assign more than was acquired");
    }

    private static IReadOnlyList<ProteinList> OnePanel(string dir)
    {
        using var reader = ParquetColumnReader.Open(
            Path.Combine(dir, "corrected_peptides.parquet"));
        var members = reader.HasColumn("leading_protein")
            ? reader.ReadStrings("leading_protein")
                .Where(v => !string.IsNullOrWhiteSpace(v))
                .SelectMany(v => ProteinListMatcher.SplitGroups(v))
                .Distinct(StringComparer.Ordinal)
                .OrderBy(v => v, StringComparer.Ordinal)
                .Take(200)
                .ToList()
            : new List<string>();
        return new[] { new ProteinList { Name = "Panel A", ColorHex = "#2ca02c", Members = members } };
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
