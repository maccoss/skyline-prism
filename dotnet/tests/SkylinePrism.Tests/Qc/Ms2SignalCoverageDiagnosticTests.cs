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
/// What the assigned fraction is actually measuring, quantified rather than argued.
///
/// <para>Two questions this answers on real data. How many fragment ions per precursor does the
/// document carry - because the assigned signal is the signal in THOSE transitions, not all the
/// signal a detected peptide produced, and a document holding six of a peptide's fragments cannot
/// account for the rest. And what does the MS1 side look like, since the merged table already
/// carries precursor rows and Skyline's own TicArea is an MS1 total, which makes an MS1-mode
/// accounting checkable against a number Skyline computed independently.</para>
///
/// <para>Opt-in via <c>PRISM_MS2_OUTPUT_DIR</c>, read-only, skipped in CI.</para>
/// </summary>
public class Ms2SignalCoverageDiagnosticTests
{
    private const string DirVar = "PRISM_MS2_OUTPUT_DIR";

    private readonly ITestOutputHelper _out;

    public Ms2SignalCoverageDiagnosticTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void HowMuchOfAPeptideDoWeActuallyCount()
    {
        var dir = Environment.GetEnvironmentVariable(DirVar);
        if (string.IsNullOrWhiteSpace(dir))
        {
            _out.WriteLine($"skipped: set {DirVar}.");
            return;
        }

        var dataset = MergedDataset.Open(Path.Combine(dir, "merged_data"));
        var names = ParquetTable.ReadColumnNames(dataset.RepresentativeFile()).ToList();
        var cols = Ms2SignalRegions.Resolve(names)!;
        var sample = File.ReadAllLines(Path.Combine(dir, "sample_metadata.csv"))[1].Split(',')[0];
        _out.WriteLine($"replicate: {sample}");
        _out.WriteLine("");

        using var conn = new DuckDB.NET.Data.DuckDBConnection("Data Source=:memory:");
        conn.Open();
        DuckDbTuning.Apply(
            conn, DuckDbMerge.AutoMemoryBudgetMb(), DuckDbMerge.ResolveTempDirectory(dataset.Root));

        var scan = MergedParquetReader.Scan(dataset.ScanTarget);
        var isPrec = MergedParquetReader.IsPrecursorSql(cols.Transition);
        var where = $"\"{cols.Sample}\" = '{sample.Replace("'", "''")}'";

        // 1. Fragments per precursor. This is the ceiling on what "assigned" can ever mean: the
        //    document's transition list, not the peptide's full fragment ladder.
        Query(conn, $@"
            SELECT n_fragments, CAST(COUNT(*) AS BIGINT) AS precursors
            FROM (
                SELECT ""{cols.Peptide}"" AS pep, ""{cols.PrecursorMz}"" AS pmz,
                       CAST(COUNT(*) AS BIGINT) AS n_fragments
                FROM {scan} WHERE {where} AND NOT {isPrec}
                GROUP BY pep, pmz
            ) GROUP BY n_fragments ORDER BY n_fragments",
            "fragments per precursor:", "  %6s fragments : %10s precursors");

        // 2. MS1 rows: are they there, and do they carry areas? An MS1 mode needs both.
        Query(conn, $@"
            SELECT ""{cols.Transition}"" AS ion,
                   CAST(COUNT(*) AS BIGINT) AS rows_,
                   CAST(SUM(CASE WHEN TRY_CAST(""{cols.Abundance}"" AS DOUBLE) > 0 THEN 1 ELSE 0 END)
                        AS BIGINT) AS with_area
            FROM {scan} WHERE {where} AND {isPrec}
            GROUP BY ion ORDER BY rows_ DESC",
            "precursor (MS1) rows by ion:", "  %-22s %10s rows, %10s with area");

        // 3. The MS1 numbers an MS1-mode accounting would work from, against Skyline's own TicArea -
        //    which is an MS1 total, so this is a cross-check that does not exist on the MS2 side.
        using var cmd = DuckDbTuning.StreamingCommand(conn, $@"
            SELECT
                SUM(CASE WHEN {isPrec} THEN TRY_CAST(""{cols.Abundance}"" AS DOUBLE) ELSE 0 END),
                SUM(CASE WHEN NOT {isPrec} THEN TRY_CAST(""{cols.Abundance}"" AS DOUBLE) ELSE 0 END),
                MAX(TRY_CAST(""TicArea"" AS DOUBLE))
            FROM {scan} WHERE {where}");
        using (var r = cmd.ExecuteReader())
        {
            Assert.True(r.Read());
            var ms1 = Convert.ToDouble(r.GetValue(0));
            var ms2 = Convert.ToDouble(r.GetValue(1));
            var tic = r.IsDBNull(2) ? double.NaN : Convert.ToDouble(r.GetValue(2));

            _out.WriteLine("");
            _out.WriteLine($"summed MS1 (precursor) area : {ms1:E4}");
            _out.WriteLine($"summed MS2 (fragment) area  : {ms2:E4}");
            _out.WriteLine($"Skyline TicArea (MS1)       : {tic:E4}");
            if (double.IsFinite(tic) && tic > 0)
            {
                _out.WriteLine($"  summed MS1 / TicArea      : {ms1 / tic:P2}");
                _out.WriteLine(
                    "  NB both are naive sums here, so this is an upper bound on what an MS1-mode "
                    + "assigned fraction would be - the union correction only reduces it.");
            }
        }
    }

    private void Query(
        DuckDB.NET.Data.DuckDBConnection conn, string sql, string heading, string format)
    {
        _out.WriteLine(heading);
        using var cmd = DuckDbTuning.StreamingCommand(conn, sql);
        using var reader = cmd.ExecuteReader();
        var rows = 0;
        while (reader.Read() && rows++ < 40)
        {
            var cells = Enumerable.Range(0, reader.FieldCount)
                .Select(i => reader.IsDBNull(i) ? "-" : Format(reader.GetValue(i)))
                .Cast<object>()
                .ToArray();
            _out.WriteLine(Sprintf(format, cells));
        }
        if (rows == 0)
            _out.WriteLine("  (none)");
        _out.WriteLine("");
    }

    private static string Format(object value) =>
        value is double d ? d.ToString("N0")
            : long.TryParse(value.ToString(), out var l) ? l.ToString("N0")
            : value.ToString() ?? "";

    /// <summary>Minimal printf for the fixed formats above; C# has no varargs %s.</summary>
    private static string Sprintf(string format, object[] args)
    {
        var text = format;
        foreach (var arg in args)
        {
            var i = text.IndexOf('%');
            if (i < 0)
                break;
            var end = i + 1;
            while (end < text.Length && (char.IsDigit(text[end]) || text[end] == '-'))
                end++;
            if (end >= text.Length)
                break;
            var spec = text[i..(end + 1)];
            var widthText = spec[1..^1];
            var width = int.TryParse(widthText.TrimStart('-'), out var w) ? w : 0;
            var value = arg.ToString() ?? "";
            value = widthText.StartsWith("-", StringComparison.Ordinal)
                ? value.PadRight(width)
                : value.PadLeft(width);
            text = text[..i] + value + text[(end + 1)..];
        }
        return text;
    }
}
