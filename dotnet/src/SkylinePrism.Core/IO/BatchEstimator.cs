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
        string mergedParquet, string sampleCol, string acqCol,
        string method = "auto", int? nBatches = null, double gapIqrMultiplier = 1.5)
    {
        var rows = ReadSampleTimes(mergedParquet, sampleCol, acqCol);
        return AssignBatches(rows, method, nBatches, gapIqrMultiplier);
    }

    /// <summary>The pure assignment core (testable): sort by time, then fixed/gap batching.</summary>
    internal static Dictionary<string, string> AssignBatches(
        List<(string Sample, DateTime Time)> rows, string method, int? nBatches, double gapIqrMultiplier)
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
        var threshold = Math.Max(q3 + gapIqrMultiplier * iqr, medianGap * 1.1);

        var batchNum = 1;
        map[rows[0].Sample] = $"batch_{batchNum}";
        for (var i = 1; i < rows.Count; i++)
        {
            if (gaps[i - 1] > threshold)
                batchNum++;
            map[rows[i].Sample] = $"batch_{batchNum}";
        }

        if (batchNum <= 1 && method == "auto" && nBatches is > 1)
            return AssignBatches(rows, "fixed", nBatches, gapIqrMultiplier);

        return batchNum > 1 ? map : new Dictionary<string, string>(StringComparer.Ordinal);
    }

    private static List<(string Sample, DateTime Time)> ReadSampleTimes(
        string parquet, string sampleCol, string acqCol)
    {
        var rows = new List<(string, DateTime)>();
        using var conn = new DuckDBConnection("Data Source=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText =
            $"SELECT DISTINCT \"{sampleCol}\" AS s, \"{acqCol}\" AS t FROM read_parquet('{parquet.Replace("'", "''")}') "
            + "WHERE t IS NOT NULL";
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
