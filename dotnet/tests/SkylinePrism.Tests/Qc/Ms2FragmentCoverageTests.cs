using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Library;
using SkylinePrism.Core.Qc;
using Xunit;
using Xunit.Abstractions;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// How much of a detected peptide's fragment signal do the document's transitions actually cover?
///
/// <para>This is the difference between two readings of the assigned fraction. "Signal in the
/// transitions the document carries" is what the accounting measures directly. "Signal attributable
/// to peptides the analysis detected" is the question people usually mean, and it is larger, because
/// a document holding six fragment ions of a peptide that produces twenty-odd cannot account for the
/// other fourteen.</para>
///
/// <para>The spectral library knows the answer: it holds the peptide's whole fragment spectrum. For
/// each measured transition, find its intensity in the library spectrum; the measured transitions'
/// share of the library's total intensity is the coverage factor. Opt-in via
/// <c>PRISM_MS2_OUTPUT_DIR</c> and <c>PRISM_MS2_BLIB</c>, read-only, skipped in CI.</para>
/// </summary>
public class Ms2FragmentCoverageTests
{
    private readonly ITestOutputHelper _out;

    public Ms2FragmentCoverageTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void WhatShareOfALibrarySpectrumDoTheTransitionsCover()
    {
        var dir = Environment.GetEnvironmentVariable("PRISM_MS2_OUTPUT_DIR");
        var blib = Environment.GetEnvironmentVariable("PRISM_MS2_BLIB");
        if (string.IsNullOrWhiteSpace(dir) || string.IsNullOrWhiteSpace(blib))
        {
            _out.WriteLine("skipped: set PRISM_MS2_OUTPUT_DIR and PRISM_MS2_BLIB.");
            return;
        }
        Assert.True(File.Exists(blib), $"no library at {blib}");

        var library = SpectralLibrary.LoadBlib(blib);
        _out.WriteLine($"library: {library.Count:N0} spectra");

        var dataset = MergedDataset.Open(Path.Combine(dir, "merged_data"));
        var names = ParquetTable.ReadColumnNames(dataset.RepresentativeFile()).ToList();
        var cols = Ms2SignalRegions.Resolve(names)!;
        var sample = File.ReadAllLines(Path.Combine(dir, "sample_metadata.csv"))[1].Split(',')[0];

        // One replicate's fragment transitions, grouped by precursor.
        using var conn = new DuckDB.NET.Data.DuckDBConnection("Data Source=:memory:");
        conn.Open();
        DuckDbTuning.Apply(
            conn, DuckDbMerge.AutoMemoryBudgetMb(), DuckDbMerge.ResolveTempDirectory(dataset.Root));

        using var cmd = DuckDbTuning.StreamingCommand(conn, $@"
            SELECT ""{cols.Peptide}"" AS pep,
                   TRY_CAST(""PrecursorCharge"" AS INTEGER) AS pz,
                   TRY_CAST(""{cols.ProductMz}"" AS DOUBLE) AS fmz,
                   TRY_CAST(""{cols.Abundance}"" AS DOUBLE) AS area
            FROM {MergedParquetReader.Scan(dataset.ScanTarget)}
            WHERE ""{cols.Sample}"" = '{sample.Replace("'", "''")}'
              AND NOT {MergedParquetReader.IsPrecursorSql(cols.Transition)}");

        var byPrecursor = new Dictionary<(string, int), List<(double Mz, double Area)>>();
        using (var reader = cmd.ExecuteReader())
        {
            while (reader.Read())
            {
                if (reader.IsDBNull(0) || reader.IsDBNull(1) || reader.IsDBNull(2))
                    continue;
                var key = (reader.GetString(0), reader.GetInt32(1));
                if (!byPrecursor.TryGetValue(key, out var list))
                    byPrecursor[key] = list = new List<(double, double)>();
                list.Add((reader.GetDouble(2), reader.IsDBNull(3) ? 0 : reader.GetDouble(3)));
            }
        }
        _out.WriteLine($"precursors in the replicate: {byPrecursor.Count:N0}");
        _out.WriteLine("");

        var coverage = new List<double>();
        var matched = 0;
        var unmatched = 0;
        double measuredAreaTotal = 0, scaledAreaTotal = 0;

        foreach (var ((peptide, charge), transitions) in byPrecursor)
        {
            var spectrum = library.GetSpectrum(peptide, charge);
            if (spectrum is null || spectrum.FragmentsByMz.Count == 0)
            {
                unmatched++;
                continue;
            }
            matched++;

            var libraryTotal = spectrum.FragmentsByMz.Values.Sum();
            if (libraryTotal <= 0)
                continue;

            // The library intensity of the fragments the document actually measures, counting each
            // library peak ONCE. Summing MatchByMz over the transitions double-counts: its tolerance
            // is 0.02 Da, so two transitions can both match the same library peak and push the
            // "covered" share above 1 - which then made area/share SHRINK the total, an impossible
            // result that is what exposed the bug.
            var creditedPeaks = new HashSet<double>();
            foreach (var (mz, _) in transitions)
            {
                double? best = null;
                var bestGap = double.MaxValue;
                foreach (var (libMz, _) in spectrum.FragmentsByMz)
                {
                    var gap = Math.Abs(libMz - mz);
                    if (gap <= 0.02 && gap < bestGap)
                    {
                        bestGap = gap;
                        best = libMz;
                    }
                }
                if (best.HasValue)
                    creditedPeaks.Add(best.Value);
            }
            var covered = creditedPeaks.Sum(k => spectrum.FragmentsByMz[k]);

            var share = covered / libraryTotal;
            if (share <= 0)
                continue;
            // Capped: a share above 1 would now mean the library total is wrong, not the matching.
            share = Math.Min(1.0, share);
            coverage.Add(share);

            var area = transitions.Sum(t => double.IsFinite(t.Area) ? t.Area : 0);
            measuredAreaTotal += area;
            scaledAreaTotal += area / share;
        }

        _out.WriteLine($"precursors matched to a library spectrum : {matched:N0}");
        _out.WriteLine($"unmatched                                : {unmatched:N0}");
        _out.WriteLine($"library fragments per spectrum (median)  : "
            + $"{MedianInt(byPrecursor.Keys.Take(2000).Select(k => library.GetSpectrum(k.Item1, k.Item2)?.FragmentsByMz.Count ?? 0)):N0}");
        _out.WriteLine("");

        if (coverage.Count == 0)
        {
            _out.WriteLine("No precursor matched a library spectrum - cannot estimate coverage.");
            return;
        }

        var sorted = coverage.OrderBy(v => v).ToArray();
        _out.WriteLine("share of the library spectrum's intensity that the measured transitions cover:");
        _out.WriteLine($"  median : {Percentile(sorted, 0.50):P1}");
        _out.WriteLine($"  p25    : {Percentile(sorted, 0.25):P1}");
        _out.WriteLine($"  p75    : {Percentile(sorted, 0.75):P1}");
        _out.WriteLine($"  min    : {sorted[0]:P1}");
        _out.WriteLine($"  max    : {sorted[^1]:P1}");
        _out.WriteLine("");
        _out.WriteLine($"summed measured area          : {measuredAreaTotal:E4}");
        _out.WriteLine($"summed area scaled by coverage: {scaledAreaTotal:E4}  "
            + $"({scaledAreaTotal / measuredAreaTotal:0.0}x)");
        _out.WriteLine("");
        var factor = scaledAreaTotal / measuredAreaTotal;
        _out.WriteLine($"AREA-WEIGHTED coverage factor: {factor:0.000}x");
        _out.WriteLine(
            "The area-weighted factor is the one that matters, and it is smaller than 1/median "
            + "would suggest: Skyline picks a peptide's most intense fragments, and the intense "
            + "precursors that dominate the area sum are the ones whose six transitions already "
            + "carry nearly all of their library intensity.");
        _out.WriteLine(
            $"So ~10.1% of acquired MS2 over the document's transitions corresponds to about "
            + $"{10.1 * factor:0.0}% once each peptide's unmeasured fragments are credited to it.");
    }

    private static double Percentile(double[] sorted, double q)
    {
        if (sorted.Length == 0)
            return double.NaN;
        var i = (int)Math.Round(q * (sorted.Length - 1));
        return sorted[Math.Clamp(i, 0, sorted.Length - 1)];
    }

    private static double MedianInt(IEnumerable<int> values)
    {
        var sorted = values.Where(v => v > 0).OrderBy(v => v).ToArray();
        return sorted.Length == 0 ? 0 : sorted[sorted.Length / 2];
    }
}
