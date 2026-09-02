using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Qc;
using Xunit;
using Xunit.Abstractions;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// Would an MS1 mode work - acquired MS1 TIC as the denominator, precursor rows as the numerator?
///
/// <para>Attractive for one reason above all: Skyline's own <c>TicArea</c> IS an MS1 total, so an
/// MS1-mode accounting can be checked against a number Skyline computed independently. The MS2 side
/// has no such reference.</para>
///
/// <para>The obstacle measured first: summing precursor areas for one replicate gave 2.40e12 against
/// a TicArea of 7.38e10 - THIRTY-TWO TIMES the total. That has to be understood before an MS1 mode
/// can be trusted. The hypothesis this tests is that it is the union problem in its most severe
/// form: MS1 has no isolation windows, so every precursor in the run competes for the same signal
/// space, where in MS2 a 3 Th window separates them into 167 compartments.</para>
///
/// <para>Opt-in via <c>PRISM_MS2_OUTPUT_DIR</c>, read-only, skipped in CI.</para>
/// </summary>
public class Ms1SignalFeasibilityTests
{
    private readonly ITestOutputHelper _out;

    public Ms1SignalFeasibilityTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void DoesTheUnionReconcileMs1WithTicArea()
    {
        var dir = Environment.GetEnvironmentVariable("PRISM_MS2_OUTPUT_DIR");
        if (string.IsNullOrWhiteSpace(dir))
        {
            _out.WriteLine("skipped: set PRISM_MS2_OUTPUT_DIR.");
            return;
        }

        var dataset = MergedDataset.Open(Path.Combine(dir, "merged_data"));
        var names = ParquetTable.ReadColumnNames(dataset.RepresentativeFile()).ToList();
        var cols = Ms2SignalRegions.Resolve(names)!;
        var sample = File.ReadAllLines(Path.Combine(dir, "sample_metadata.csv"))[1].Split(',')[0];

        using var conn = new DuckDB.NET.Data.DuckDBConnection("Data Source=:memory:");
        conn.Open();
        DuckDbTuning.Apply(
            conn, DuckDbMerge.AutoMemoryBudgetMb(), DuckDbMerge.ResolveTempDirectory(dataset.Root));

        // Precursor (MS1) rows only. The precursor m/z takes the place of the product m/z: on MS1 the
        // signal is extracted at the PRECURSOR mass, and the isotope is part of which m/z that is.
        using var cmd = DuckDbTuning.StreamingCommand(conn, $@"
            SELECT TRY_CAST(""{cols.PrecursorMz}"" AS DOUBLE) AS pmz,
                   TRY_CAST(""{cols.StartTime}"" AS DOUBLE) AS rt0,
                   TRY_CAST(""{cols.EndTime}"" AS DOUBLE) AS rt1,
                   TRY_CAST(""{cols.Abundance}"" AS DOUBLE) AS area,
                   ""{cols.Peptide}"" AS pep,
                   ""{cols.Transition}"" AS ion,
                   MAX(TRY_CAST(""TicArea"" AS DOUBLE)) OVER () AS tic
            FROM {MergedParquetReader.Scan(dataset.ScanTarget)}
            WHERE ""{cols.Sample}"" = '{sample.Replace("'", "''")}'
              AND {MergedParquetReader.IsPrecursorSql(cols.Transition)}");

        var regions = new List<Ms2SignalUnion.Region>();
        var peptideIds = new Dictionary<string, int>(StringComparer.Ordinal);
        var isotopes = new Dictionary<string, int>(StringComparer.Ordinal);
        var tic = double.NaN;
        var naive = 0.0;

        using (var reader = cmd.ExecuteReader())
        {
            while (reader.Read())
            {
                var pmz = reader.IsDBNull(0) ? double.NaN : reader.GetDouble(0);
                var rt0 = reader.IsDBNull(1) ? double.NaN : reader.GetDouble(1);
                var rt1 = reader.IsDBNull(2) ? double.NaN : reader.GetDouble(2);
                var area = reader.IsDBNull(3) ? double.NaN : reader.GetDouble(3);
                var pep = reader.IsDBNull(4) ? "" : reader.GetString(4);
                var ion = reader.IsDBNull(5) ? "" : reader.GetString(5);
                if (double.IsNaN(tic) && !reader.IsDBNull(6))
                    tic = reader.GetDouble(6);

                isotopes[ion] = isotopes.GetValueOrDefault(ion) + 1;
                if (double.IsFinite(area))
                    naive += area;

                if (!peptideIds.TryGetValue(pep, out var id))
                    peptideIds[pep] = id = peptideIds.Count;

                // ONE window: MS1 has no isolation compartments, which is the whole point.
                regions.Add(new Ms2SignalUnion.Region(
                    0, pmz, rt0, rt1, area, true, 0, id));
            }
        }

        _out.WriteLine($"replicate            : {sample}");
        _out.WriteLine($"precursor rows       : {regions.Count:N0} "
            + $"({string.Join(", ", isotopes.OrderByDescending(k => k.Value).Take(4).Select(k => $"{k.Key} {k.Value:N0}"))})");
        _out.WriteLine($"naive sum            : {naive:E4}");
        _out.WriteLine($"Skyline TicArea (MS1): {tic:E4}");
        _out.WriteLine($"  naive / TicArea    : {naive / tic:0.0}x");
        _out.WriteLine("");

        foreach (var setting in new[] { "10 ppm", "20 ppm" })
        {
            var tolerance = ProductMassTolerance.ParseSetting(setting)!;
            var union = Ms2SignalUnion.Compute(regions, tolerance, 0);
            _out.WriteLine($"union at {setting,-8}: {union.AssignedArea:E4}  "
                + $"({union.AssignedArea / tic:0.00}x TicArea)");
            _out.WriteLine($"  merged groups      : {union.MergedGroups:N0} of {union.Regions:N0} "
                + $"regions, largest {union.LargestGroup}");
            _out.WriteLine($"  removed            : {1 - union.AssignedArea / union.SummedArea:P1} "
                + $"({union.DuplicateArea / Math.Max(1e-9, union.DuplicateArea + union.SharedArea):P0} "
                + "of it duplicate rows)");
        }

        _out.WriteLine("");
        _out.WriteLine(
            "READ THIS AS FEASIBILITY, NOT A RESULT. If the union lands at or below 1.0x TicArea the "
            + "MS1 mode reconciles with a number Skyline computed independently, which is a stronger "
            + "check than anything available on the MS2 side. If it stays far above, the gap is not "
            + "the union and needs finding first - a unit mismatch between transition Area and "
            + "TicArea would be the next thing to rule out.");
    }
}
