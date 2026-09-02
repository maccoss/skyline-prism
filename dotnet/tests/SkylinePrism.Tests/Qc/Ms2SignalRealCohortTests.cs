using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.Visualization;
using Xunit;
using Xunit.Abstractions;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// The accounting against a REAL PRISM output directory, opt-in and read-only.
///
/// <para>The committed cohort fixture is 327 peptides, which gives a median of 2 precursors per
/// isolation window where a real run has tens - so it cannot show how much co-isolation sharing there
/// is, only the bookkeeping kind. This is how the numbers get checked against a run that can.</para>
///
/// <para>Set <c>PRISM_MS2_OUTPUT_DIR</c> to a completed output directory (one with
/// <c>merged_data/</c>, <c>peptides_rollup.parquet</c> and <c>isolation_schemes.xml</c>). Unset in CI,
/// where it skips. Nothing is written to that directory - results are computed and reported, never
/// persisted, so pointing this at a colleague's analysis cannot disturb it.</para>
/// </summary>
public class Ms2SignalRealCohortTests
{
    private const string DirVar = "PRISM_MS2_OUTPUT_DIR";
    private const string ToleranceVar = "PRISM_MS2_TOLERANCE";

    private readonly ITestOutputHelper _out;

    public Ms2SignalRealCohortTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void AccountingOnARealCohort()
    {
        var dir = Environment.GetEnvironmentVariable(DirVar);
        if (string.IsNullOrWhiteSpace(dir))
        {
            _out.WriteLine($"skipped: set {DirVar} to a completed PRISM output directory.");
            return;
        }
        Assert.True(Directory.Exists(dir), $"{DirVar} is not a directory: {dir}");

        var setting = Environment.GetEnvironmentVariable(ToleranceVar) is { Length: > 0 } t ? t : "10 ppm";
        var tolerance = ProductMassTolerance.ParseSetting(setting);
        Assert.NotNull(tolerance);

        var scheme = OnlyScheme(dir);
        _out.WriteLine($"directory : {dir}");
        _out.WriteLine($"scheme    : {scheme.Name} ({scheme.Windows.Count} windows, "
            + $"{scheme.Windows.Min(w => w.Start):0.0}-{scheme.Windows.Max(w => w.End):0.0} m/z, "
            + $"median width {Median(scheme.Windows.Select(w => w.Width)):0.00} Th)");
        _out.WriteLine($"tolerance : {tolerance!.Describe()}");

        var sampleTypes = ReadSampleTypes(Path.Combine(dir, "sample_metadata.csv"));
        var started = DateTime.UtcNow;
        var result = Ms2SignalAccounting.Compute(
            dir, scheme, tolerance, Array.Empty<ProteinList>(), sampleTypes, _out.WriteLine);
        var elapsed = DateTime.UtcNow - started;

        Assert.NotNull(result);
        var rows = result!.Rows;
        _out.WriteLine($"elapsed   : {elapsed.TotalSeconds:0.0} s for {rows.Count} replicates");
        _out.WriteLine("");

        // Per replicate: does the union remove a plausible amount, and does the split between the two
        // kinds of over-counting look like a real run rather than like the thin fixture?
        var fractions = rows.Select(r => r.DoubleCountedFraction).Where(double.IsFinite).ToList();
        var totalDuplicate = rows.Sum(r => (long)r.DuplicateRows);
        var totalShared = rows.Sum(r => (long)r.SharedAcrossPeptides);
        var totalRegions = rows.Sum(r => (long)r.Regions);

        _out.WriteLine($"fragment peaks      : {totalRegions:N0} across the cohort "
            + $"({rows.Average(r => r.Regions):N0} per replicate)");
        _out.WriteLine($"double counted      : median {Median(fractions):P2} of the naive sum "
            + $"(min {fractions.Min():P2}, max {fractions.Max():P2})");
        // Rows and AREA split very differently, which is the whole reason both are reported.
        var duplicateArea = rows.Sum(r => r.DuplicateArea);
        var sharedArea = rows.Sum(r => r.SharedArea);
        var removedArea = duplicateArea + sharedArea;
        var summedArea = rows.Sum(r => r.SummedArea);

        _out.WriteLine($"  duplicate rows    : {totalDuplicate:N0} rows "
            + $"({(double)totalDuplicate / Math.Max(1, totalDuplicate + totalShared):P1} of merges), "
            + $"{duplicateArea / Math.Max(1e-9, removedArea):P1} of the area removed "
            + $"= {duplicateArea / summedArea:P2} of the sum");
        _out.WriteLine($"  co-isolation      : {totalShared:N0} rows "
            + $"({(double)totalShared / Math.Max(1, totalDuplicate + totalShared):P1} of merges), "
            + $"{sharedArea / Math.Max(1e-9, removedArea):P1} of the area removed "
            + $"= {sharedArea / summedArea:P2} of the sum");
        _out.WriteLine($"  merged-away rows  : "
            + $"{(double)(totalDuplicate + totalShared) / totalRegions:P2} of all peaks");
        _out.WriteLine($"largest merge group : {rows.Max(r => r.LargestGroup)} transitions");
        _out.WriteLine($"outside the scheme  : {rows.Sum(r => (long)r.OutsideScheme):N0} "
            + $"({(double)rows.Sum(r => (long)r.OutsideScheme) / Math.Max(1, totalRegions):P2})");
        _out.WriteLine($"unknown peptides    : {rows.Sum(r => (long)r.UnknownPeptides):N0} "
            + "(rows whose peptide is not in the peptide matrix)");
        _out.WriteLine($"skipped (non-finite): {rows.Sum(r => (long)r.Skipped):N0}");
        _out.WriteLine("");

        // Signal by sample type: reference and QC injections are different material at different
        // loads, so they SHOULD differ from the experimental samples. A type whose signal matches the
        // others exactly would mean the metadata never reached the accounting.
        foreach (var group in rows.GroupBy(r => r.SampleType).OrderBy(g => g.Key, StringComparer.Ordinal))
        {
            var assigned = group.Select(r => r.AssignedArea).ToList();
            _out.WriteLine($"{group.Key,-14} n={group.Count(),3}  "
                + $"median assigned {Median(assigned):E3}  "
                + $"CV {Cv(assigned):0.0}%");
        }
        _out.WriteLine("");

        // The five quietest replicates: this is the reading the plot exists to give.
        foreach (var row in rows.OrderBy(r => r.AssignedArea).Take(5))
        {
            _out.WriteLine($"quietest: {row.Sample,-45} {row.AssignedArea:E3} "
                + $"({row.SampleType}, {row.Regions:N0} peaks)");
        }
        var loudest = rows.OrderByDescending(r => r.AssignedArea).First();
        _out.WriteLine($"loudest : {loudest.Sample,-45} {loudest.AssignedArea:E3} "
            + $"({loudest.SampleType}, {loudest.Regions:N0} peaks)");

        if (Environment.GetEnvironmentVariable("PRISM_MS2_PLOT_OUT") is { Length: > 0 } dump)
        {
            Directory.CreateDirectory(dump);
            File.WriteAllBytes(
                Path.Combine(dump, "ms2_real_cohort.png"),
                PlotRenderer.Ms2AccountingPng(result, "MS2 Signal Assigned to Peptides"));
        }

        // Properties that must hold on any run, checked here on data the fixture cannot represent.
        foreach (var row in rows)
        {
            Assert.True(row.AssignedArea <= row.SummedArea + 1e-6,
                $"{row.Sample}: union exceeded the sum");
            Assert.True(row.AssignedArea > 0, $"{row.Sample}: no assigned signal at all");
        }
    }

    /// <summary>
    /// WHY the duplicate-row correction is so much larger in area than in row count. The accounting
    /// says 4.8% of peaks merge away but 20% of the area does, which only makes sense if the duplicated
    /// rows are far more intense than average - and that is a claim worth checking against the table
    /// rather than assuming. Prints the multiplicity of each (peptide, product m/z, integration bounds)
    /// key in one replicate, and the mean area at each multiplicity.
    /// </summary>
    [Fact]
    public void WhyDuplicateRowsDominateTheRemovedArea()
    {
        var dir = Environment.GetEnvironmentVariable(DirVar);
        if (string.IsNullOrWhiteSpace(dir))
        {
            _out.WriteLine($"skipped: set {DirVar} to a completed PRISM output directory.");
            return;
        }

        var dataset = SkylinePrism.Core.IO.MergedDataset.Open(Path.Combine(dir, "merged_data"));
        var names = SkylinePrism.Core.IO.ParquetTable
            .ReadColumnNames(dataset.RepresentativeFile()).ToList();
        _out.WriteLine("columns: " + string.Join(", ", names));

        var cols = Ms2SignalRegions.Resolve(names);
        Assert.NotNull(cols);

        var sample = FirstSample(dir);
        _out.WriteLine($"replicate: {sample}");
        _out.WriteLine("");

        using var conn = new DuckDB.NET.Data.DuckDBConnection("Data Source=:memory:");
        conn.Open();
        SkylinePrism.Core.IO.DuckDbTuning.Apply(
            conn, SkylinePrism.Core.IO.DuckDbMerge.AutoMemoryBudgetMb(),
            SkylinePrism.Core.IO.DuckDbMerge.ResolveTempDirectory(dataset.Root));

        // Multiplicity of an identical measurement, and how intense those measurements are.
        var sql = $@"
            WITH frag AS (
                SELECT ""{cols!.Peptide}"" AS pep,
                       ""{cols.ProductMz}"" AS fmz,
                       ""{cols.StartTime}"" AS rt0,
                       ""{cols.EndTime}"" AS rt1,
                       ""Protein"" AS prot,
                       ""PrecursorCharge"" AS pz,
                       TRY_CAST(""{cols.PrecursorMz}"" AS DOUBLE) AS pmz,
                       TRY_CAST(""{cols.Abundance}"" AS DOUBLE) AS area
                FROM {SkylinePrism.Core.IO.MergedParquetReader.Scan(dataset.ScanTarget)}
                WHERE ""{cols.Sample}"" = '{sample.Replace("'", "''")}'
                  AND NOT {SkylinePrism.Core.IO.MergedParquetReader.IsPrecursorSql(cols.Transition)}
            ), keyed AS (
                SELECT pep, fmz, rt0, rt1,
                       COUNT(*) AS n,
                       AVG(area) AS mean_area,
                       COUNT(DISTINCT prot) AS proteins,
                       COUNT(DISTINCT pz) AS charges,
                       MAX(pmz) - MIN(pmz) AS pmz_spread
                FROM frag GROUP BY pep, fmz, rt0, rt1
            )
            SELECT n,
                   CAST(COUNT(*) AS BIGINT) AS keys,
                   CAST(SUM(n) AS BIGINT) AS rows_,
                   AVG(mean_area) AS mean_area,
                   AVG(proteins) AS mean_proteins,
                   AVG(charges) AS mean_charges,
                   -- Precursors more than one isolation window apart CANNOT be merged by the union:
                   -- they were fragmented in different spectra. This is the check that the merging is
                   -- confined to measurements that really are the same detector signal.
                   CAST(SUM(CASE WHEN pmz_spread > 3.0 THEN n ELSE 0 END) AS BIGINT) AS rows_apart
            FROM keyed GROUP BY n ORDER BY n";

        using var cmd = SkylinePrism.Core.IO.DuckDbTuning.StreamingCommand(conn, sql);
        using var reader = cmd.ExecuteReader();

        _out.WriteLine(
            "copies  distinct keys        rows   mean area   proteins  charges   rows >1 window apart");
        double overallRows = 0, weighted = 0;
        var rowsAtOne = 0.0;
        var meanAtOne = 0.0;
        while (reader.Read())
        {
            var n = Convert.ToInt64(reader.GetValue(0).ToString());
            var keys = Convert.ToInt64(reader.GetValue(1).ToString());
            var rowCount = Convert.ToInt64(reader.GetValue(2).ToString());
            var mean = Convert.ToDouble(reader.GetValue(3));
            var proteins = Convert.ToDouble(reader.GetValue(4));
            var charges = Convert.ToDouble(reader.GetValue(5));
            var apart = Convert.ToInt64(reader.GetValue(6).ToString());
            _out.WriteLine(
                $"{n,6}  {keys,13:N0} {rowCount,11:N0}   {mean:E3}   {proteins,8:0.00} {charges,8:0.00}"
                + $"   {apart,10:N0} ({(double)apart / rowCount:P0})");
            overallRows += rowCount;
            weighted += mean * rowCount;
            if (n == 1)
            {
                rowsAtOne = rowCount;
                meanAtOne = mean;
            }
        }

        // The load-bearing claim: the rows the union actually merges (same peptide, same window) are
        // far more intense than average. Measured directly rather than inferred from the table above.
        using var cmd2 = SkylinePrism.Core.IO.DuckDbTuning.StreamingCommand(conn, $@"
            WITH frag AS (
                SELECT ""{cols.Peptide}"" AS pep,
                       ""{cols.ProductMz}"" AS fmz,
                       ""{cols.StartTime}"" AS rt0,
                       ""{cols.EndTime}"" AS rt1,
                       TRY_CAST(""{cols.PrecursorMz}"" AS DOUBLE) AS pmz,
                       TRY_CAST(""{cols.Abundance}"" AS DOUBLE) AS area
                FROM {SkylinePrism.Core.IO.MergedParquetReader.Scan(dataset.ScanTarget)}
                WHERE ""{cols.Sample}"" = '{sample.Replace("'", "''")}'
                  AND NOT {SkylinePrism.Core.IO.MergedParquetReader.IsPrecursorSql(cols.Transition)}
            ), keyed AS (
                SELECT COUNT(*) AS n, AVG(area) AS mean_area,
                       MAX(pmz) - MIN(pmz) AS spread
                FROM frag GROUP BY pep, fmz, rt0, rt1
            )
            SELECT
                CAST(SUM(CASE WHEN n > 1 AND spread <= 3.0 THEN n - 1 ELSE 0 END) AS BIGINT) AS mergeable,
                AVG(CASE WHEN n > 1 AND spread <= 3.0 THEN mean_area END) AS mergeable_area,
                AVG(CASE WHEN n = 1 THEN mean_area END) AS unique_area
            FROM keyed");
        using (var r2 = cmd2.ExecuteReader())
        {
            if (r2.Read())
            {
                var mergeable = Convert.ToInt64(r2.GetValue(0).ToString());
                var mergeableArea = Convert.ToDouble(r2.GetValue(1));
                var uniqueArea = Convert.ToDouble(r2.GetValue(2));
                _out.WriteLine("");
                _out.WriteLine($"same-window duplicates    : {mergeable:N0} rows the union merges away");
                _out.WriteLine($"  their mean area         : {mergeableArea:E3}");
                _out.WriteLine($"  unique-row mean area    : {uniqueArea:E3}");
                _out.WriteLine($"  ratio                   : {mergeableArea / uniqueArea:0.0}x");
            }
        }

        _out.WriteLine("");
        _out.WriteLine($"mean area, all rows       : {weighted / overallRows:E3}");
        _out.WriteLine($"mean area, unique rows    : {meanAtOne:E3}");
        _out.WriteLine($"unique rows               : {rowsAtOne / overallRows:P2} of the replicate");
        _out.WriteLine(
            "A copies>1 population that is much more intense than the unique one is the explanation: "
            + "the peptides Skyline exports several times are the abundant, widely-shared ones.");
    }

    private static string FirstSample(string dir)
    {
        var path = Path.Combine(dir, "sample_metadata.csv");
        var lines = File.ReadAllLines(path);
        var header = lines[0].Split(',');
        var idCol = Array.FindIndex(header, h => h.Trim() == "sample_id");
        return lines[1].Split(',')[idCol];
    }

    /// <summary>The cohort's own isolation scheme, which must be unambiguous for this to mean anything.</summary>
    private static IsolationScheme OnlyScheme(string dir)
    {
        var catalog = IsolationSchemeCatalog.Load(
            Path.Combine(dir, IsolationSchemeCatalog.FileName));
        Assert.NotNull(catalog);
        var usable = catalog!.UsableSchemes;
        Assert.True(usable.Count == 1,
            $"expected exactly one usable isolation scheme, found {usable.Count}");
        return usable[0];
    }

    private static Dictionary<string, string> ReadSampleTypes(string path)
    {
        var types = new Dictionary<string, string>(StringComparer.Ordinal);
        if (!File.Exists(path))
            return types;
        var lines = File.ReadAllLines(path);
        if (lines.Length < 2)
            return types;

        var header = lines[0].Split(',');
        var idCol = Array.FindIndex(header, h => h.Trim() == "sample_id");
        var typeCol = Array.FindIndex(header, h => h.Trim() == "sample_type");
        if (idCol < 0 || typeCol < 0)
            return types;

        foreach (var line in lines.Skip(1))
        {
            var cells = line.Split(',');
            if (cells.Length > Math.Max(idCol, typeCol))
                types[cells[idCol]] = cells[typeCol];
        }
        return types;
    }

    private static double Median(IEnumerable<double> values)
    {
        var sorted = values.Where(double.IsFinite).OrderBy(v => v).ToArray();
        if (sorted.Length == 0)
            return double.NaN;
        var mid = sorted.Length / 2;
        return sorted.Length % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    }

    /// <summary>CV on LINEAR values, as every CV in PRISM is.</summary>
    private static double Cv(IReadOnlyList<double> values)
    {
        if (values.Count < 2)
            return double.NaN;
        var mean = values.Average();
        if (mean <= 0)
            return double.NaN;
        var variance = values.Sum(v => (v - mean) * (v - mean)) / (values.Count - 1);
        return Math.Sqrt(variance) / mean * 100.0;
    }
}
