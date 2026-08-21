using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using DuckDB.NET.Data;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.IO;

/// <summary>
/// Estimate per-sample batch labels from acquisition-time gaps, porting cli.py:
/// estimate_batches_from_parquet (the "gap" / "fixed" paths). Samples are ordered by acquisition
/// time; a large gap (IQR outlier vs the typical inter-sample gap) starts a new batch. This is the
/// fallback when neither a metadata Batch column nor the Source Document distinguishes batches.
/// </summary>
public static class BatchEstimator
{
    /// <summary>
    /// Returns sample -> "batch_N". Empty when acquisition times are unavailable/insufficient or no
    /// significant gaps are found (method "gap"/"auto"). Method "fixed" divides into n_batches by
    /// acquisition order.
    /// </summary>
    public static Dictionary<string, string> Estimate(
        MergedDataset dataset, string sampleCol, string acqCol,
        string method = "auto", int? nBatches = null, double gapIqrMultiplier = 1.5,
        Action<string>? log = null)
    {
        var rows = ReadSampleTimes(dataset, sampleCol, acqCol);
        return AssignBatches(rows, method, nBatches, gapIqrMultiplier, log);
    }

    /// <summary>The pure assignment core (testable): sort by time, then fixed/gap batching.</summary>
    internal static Dictionary<string, string> AssignBatches(
        List<(string Sample, DateTime Time)> rows, string method, int? nBatches, double gapIqrMultiplier,
        Action<string>? log = null)
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        if (rows.Count < 2)
            return map;

        rows = rows.OrderBy(r => r.Time).ToList();

        if (method == "fixed" && nBatches is > 1)
        {
            var n = rows.Count;
            var size = n / nBatches.Value;
            var remainder = n % nBatches.Value;
            var idx = 0;
            for (var bIdx = 0; bIdx < nBatches.Value; bIdx++)
            {
                var count = size + (bIdx < remainder ? 1 : 0);
                for (var k = 0; k < count && idx < n; k++, idx++)
                    map[rows[idx].Sample] = $"batch_{bIdx + 1}";
            }
            return map;
        }

        // Gap detection (auto / gap): inter-sample gaps in minutes.
        var gaps = new double[rows.Count - 1];
        for (var i = 1; i < rows.Count; i++)
            gaps[i - 1] = (rows[i].Time - rows[i - 1].Time).TotalMinutes;

        var q1 = Stats.PercentileLinear(gaps, 25);
        var q3 = Stats.PercentileLinear(gaps, 75);
        var iqr = q3 - q1;
        var medianGap = Stats.PercentileLinear(gaps, 50);

        // Two candidate thresholds; the larger wins. The Tukey rule (q3 + k*IQR) is the intended test,
        // but it degenerates to q3 on evenly spaced runs where IQR ~ 0, so a floor of 1.1x the median gap
        // was added to keep it above the typical spacing. NOTE: on a CONTINUOUS run that floor is what
        // binds, and it sits only 10% above the typical spacing - so a single slightly longer gap (a
        // wash, a blank, a queue pause) starts a new batch. See the warning logged below.
        var tukey = q3 + gapIqrMultiplier * iqr;
        var floor = medianGap * 1.1;
        var threshold = Math.Max(tukey, floor);

        var batchNum = 1;
        var breaks = new List<double>();
        map[rows[0].Sample] = $"batch_{batchNum}";
        for (var i = 1; i < rows.Count; i++)
        {
            if (gaps[i - 1] > threshold)
            {
                batchNum++;
                breaks.Add(gaps[i - 1]);
            }
            map[rows[i].Sample] = $"batch_{batchNum}";
        }

        if (log is not null && method != "fixed")
        {
            log($"  Gap threshold: {threshold:F1} min (median gap {medianGap:F1} min, "
                + $"IQR rule {tukey:F1} min, floor 1.1x median {floor:F1} min; "
                + $"{(floor >= tukey ? "the FLOOR is binding" : "the IQR rule is binding")}).");
            if (breaks.Count > 0)
                log($"  Gaps that started a new batch: "
                    + string.Join(", ", breaks.Select(b => $"{b:F1} min")) + ".");

            // The floor binding means the run is evenly spaced - i.e. it looks like ONE continuous
            // sequence - yet it still split. That is the case most likely to be wrong.
            if (floor >= tukey && breaks.Count > 0)
            {
                log("  WARNING: the batches came from a threshold only 10% above the typical spacing, "
                    + "which is what happens on a continuously acquired run. If these samples were not "
                    + "actually run in separate batches, set batch_estimation.method: none (or supply a "
                    + "real batch annotation) - otherwise ComBat will 'correct' between batches that do "
                    + "not exist.");
            }
        }

        if (batchNum <= 1 && method == "auto" && nBatches is > 1)
            return AssignBatches(rows, "fixed", nBatches, gapIqrMultiplier, log);

        return batchNum > 1 ? map : new Dictionary<string, string>(StringComparer.Ordinal);
    }

    private static List<(string Sample, DateTime Time)> ReadSampleTimes(
        MergedDataset dataset, string sampleCol, string acqCol)
    {
        var rows = new List<(string, DateTime)>();
        using var conn = new DuckDBConnection("Data Source=:memory:");
        conn.Open();
        DuckDbTuning.Apply(
            conn, DuckDbMerge.AutoMemoryBudgetMb(), DuckDbMerge.ResolveTempDirectory(dataset.Root));
        using var cmd = DuckDbTuning.StreamingCommand(conn,
            $"SELECT DISTINCT \"{sampleCol}\" AS s, \"{acqCol}\" AS t "
            + $"FROM {MergedParquetReader.Scan(dataset.ScanTarget)} "
            + "WHERE t IS NOT NULL");
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            if (reader.IsDBNull(0) || reader.IsDBNull(1))
                continue;
            var sample = reader.GetString(0);
            var raw = reader.GetValue(1);
            if (TryParseTime(raw, out var time))
                rows.Add((sample, time));
        }
        return rows;
    }

    private static bool TryParseTime(object raw, out DateTime time)
    {
        if (raw is DateTime dt)
        {
            time = dt;
            return true;
        }
        var s = Convert.ToString(raw, CultureInfo.InvariantCulture);
        return DateTime.TryParse(s, CultureInfo.InvariantCulture, DateTimeStyles.None, out time)
            || DateTime.TryParse(s, CultureInfo.CurrentCulture, DateTimeStyles.None, out time);
    }
}
