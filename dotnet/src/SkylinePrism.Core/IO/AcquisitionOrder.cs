using System;
using System.Collections.Generic;
using System.Linq;

namespace SkylinePrism.Core.IO;

/// <summary>
/// The order a cohort's runs were acquired in, from Skyline's <c>Acquired Time</c>. Written to
/// <c>sample_metadata.csv</c> as <c>acquired_time</c> and <c>run_order</c>, and the x axis of the QC
/// report's peak-quality section - one rule for both, so the run a plot calls "run 11" is the row of
/// the metadata that says 11.
/// </summary>
/// <remarks>
/// A replicate is placed at its EARLIEST time. Skyline lets one replicate hold several injections,
/// each with its own acquired time, and the replicate entered the queue with the first of them.
/// </remarks>
public static class AcquisitionOrder
{
    /// <summary>
    /// How <c>acquired_time</c> is written: sortable as text, and read as a date by Excel, pandas and
    /// R without a format string. No time zone, because Skyline records none - the instrument PC's
    /// local clock is all the file says.
    /// </summary>
    public const string TimeFormat = "yyyy-MM-dd HH:mm:ss";

    /// <summary>Each sample's earliest acquired time; a sample with none is absent.</summary>
    public static Dictionary<string, DateTime> ReadTimes(
        MergedDataset dataset, string sampleCol, string acquiredTimeCol)
    {
        var earliest = new Dictionary<string, DateTime>(StringComparer.Ordinal);
        foreach (var (sample, time) in BatchEstimator.ReadSampleTimes(dataset, sampleCol, acquiredTimeCol))
        {
            if (!earliest.TryGetValue(sample, out var seen) || time < seen)
                earliest[sample] = time;
        }
        return earliest;
    }

    /// <summary>
    /// 1-based position of each timed sample in acquisition order. Two samples stamped with the same
    /// second are ordered by sample id, so the numbering never depends on the order rows were read.
    /// Samples without a time get no number rather than being placed at either end.
    /// </summary>
    public static Dictionary<string, int> Rank(IReadOnlyDictionary<string, DateTime> times)
    {
        var rank = new Dictionary<string, int>(StringComparer.Ordinal);
        var ordered = times
            .OrderBy(kv => kv.Value)
            .ThenBy(kv => kv.Key, StringComparer.Ordinal);
        foreach (var (sample, _) in ordered)
            rank[sample] = rank.Count + 1;
        return rank;
    }
}
