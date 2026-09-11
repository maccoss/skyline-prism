using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.RawData;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// The acquired MS2 total per replicate: the denominator that turns the signal accounting from "how
/// much signal was assigned" into "what fraction of what the instrument acquired was assigned".
/// </summary>
/// <remarks>
/// <para>Kept in its own file rather than folded into
/// <see cref="Ms2SignalAccounting.AccountingFile"/>, because the two are populated at different
/// times and at wildly different cost. The accounting reads <c>merged_data/</c> and takes about as
/// long as Stage 2; this reads the instrument files, which on a 192-replicate cohort is ~1.1 TB over
/// a network share. Writing them together would mean populating one invalidated the other, and the
/// expensive one would be the one thrown away.</para>
///
/// <para>So the fraction is a JOIN at render time, not a stored column - see
/// <see cref="Ms2SignalAccounting.Result.WithAcquired"/>. A directory may have the accounting and no
/// acquired totals (the normal state, and the plot simply omits the bar), or acquired totals that
/// were read before the last recompute, which still join correctly because the key is the sample id
/// and the instrument file has not changed.</para>
/// </remarks>
public static class Ms2AcquiredSignal
{
    /// <summary>One row per replicate that was read, successfully or not.</summary>
    public const string FileName = "ms2_signal.parquet";

    /// <summary>
    /// Acquired signal per acquisition cycle, long format: one row per cycle per replicate.
    /// </summary>
    /// <remarks>
    /// Written on the same pass that computes the totals, for every replicate, even though the only
    /// thing that reads it shows one replicate at a time. Capturing a cycle is free while the file is
    /// already open and decoded; going back for it means re-reading the cohort, which on the one this
    /// was written against is ~1.1 TB over a share. The cost of keeping it is about 3,000 rows per
    /// replicate - a few MB zstd across 192 of them.
    /// </remarks>
    public const string CyclesFile = "ms2_cycles.parquet";

    /// <param name="TotalMs2Signal">Acquired MS2 total ion current over the run. LINEAR, NaN when the
    /// read did not succeed.</param>
    public sealed record Entry(
        string Sample,
        string DataPath,
        string Status,
        string Reader,
        string SignalSource,
        long Ms1Count,
        long Ms2Count,
        double TotalMs2Signal,
        double RtStartMin,
        double RtStopMin,
        string CycleModel)
    {
        public bool IsUsable =>
            string.Equals(Status, nameof(Ms2ReadStatus.Ok), StringComparison.Ordinal)
            && TotalMs2Signal > 0;
    }

    /// <summary>What a populate run did, for the log and for the caller's summary line.</summary>
    public sealed record PopulateResult(
        IReadOnlyList<Entry> Entries, int Matched, int Unmatched, int Usable)
    {
        public bool AnyUsable => Usable > 0;
    }

    /// <summary>
    /// Read the instrument files for a cohort's replicates and persist their acquired MS2 totals.
    /// </summary>
    /// <param name="samples">Sample ids to resolve, normally the accounting's own row order.</param>
    /// <param name="rawDirectory">Directory holding the cohort's data files.</param>
    /// <param name="maxFiles">Stop after this many successful matches; 0 for all. A cohort is hundreds
    /// of gigabytes and a spot check of three files answers most questions.</param>
    /// <remarks>
    /// Never throws for a file it cannot read: <see cref="IMs2SignalReader"/> is contracted to return
    /// a non-OK record instead, and one bad file in 192 must not cost the other 191. What it does not
    /// do is invent a total - an unreadable file is written as a row with its status, so the report
    /// can say which replicates have no denominator rather than quietly plotting a shorter bar.
    /// </remarks>
    public static PopulateResult Populate(
        string outputDir,
        string rawDirectory,
        IEnumerable<string> samples,
        Action<string>? log = null,
        int lanes = Ms2SignalReaders.DefaultLanes,
        int maxFiles = 0,
        CancellationToken ct = default)
    {
        var sampleList = samples?.ToList() ?? new List<string>();
        var files = ReplicateDataFiles.Enumerate(rawDirectory);
        if (files.Count == 0)
        {
            log?.Invoke($"  No instrument data files under {rawDirectory}, so acquired MS2 signal "
                + "cannot be read. The accounting will plot assigned signal without a denominator.");
            return new PopulateResult(Array.Empty<Entry>(), 0, sampleList.Count, 0);
        }

        var (matched, unmatched) = ReplicateDataFiles.ResolveAll(sampleList, files);
        log?.Invoke($"  Matched {matched.Count:N0} of {sampleList.Count:N0} replicate(s) to a data "
            + $"file in {rawDirectory} ({files.Count:N0} file(s) there).");
        if (unmatched.Count > 0)
        {
            // Named, not just counted: the usual cause is that the raw directory holds a different
            // subset than the document, and the only way to see that is to read the names.
            log?.Invoke("  No data file for: "
                + string.Join(", ", unmatched.Take(5).Select(ReplicateDataFiles.ReplicateOf))
                + (unmatched.Count > 5 ? $", and {unmatched.Count - 5:N0} more." : "."));
        }

        var ordered = sampleList.Where(matched.ContainsKey).ToList();
        if (maxFiles > 0 && ordered.Count > maxFiles)
        {
            log?.Invoke($"  Reading the first {maxFiles:N0} of them (max-files).");
            ordered = ordered.Take(maxFiles).ToList();
        }

        // Path -> sample, so a record coming back out of ReadMany (in completion order) can be put
        // back against the replicate it belongs to.
        var sampleByPath = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var sample in ordered)
            sampleByPath[matched[sample]] = sample;

        var entries = new List<Entry>();
        var cyclesBySample = new Dictionary<string, IReadOnlyList<Ms2Cycle>>(StringComparer.Ordinal);
        var gate = new object();
        Ms2SignalReaders.ReadMany(
            ordered.Select(s => matched[s]),
            record =>
            {
                var sample = sampleByPath.TryGetValue(record.DataPath, out var s) ? s : record.DataPath;
                var entry = new Entry(
                    sample,
                    record.DataPath,
                    record.Status.ToString(),
                    record.Reader,
                    record.SignalSource.ToString(),
                    record.Ms1Count,
                    record.Ms2Count,
                    record.Status == Ms2ReadStatus.Ok ? record.TotalMs2Signal : double.NaN,
                    record.RtStartMin,
                    record.RtStopMin,
                    record.CycleModel.ToString());
                // ReadMany calls this from worker threads.
                lock (gate)
                {
                    entries.Add(entry);
                    if (record.Cycles.Count > 0)
                        cyclesBySample[sample] = record.Cycles;
                }
                if (record.Status != Ms2ReadStatus.Ok)
                    log?.Invoke($"    {Path.GetFileName(record.DataPath)}: {record.Status} "
                        + $"({record.Message})");
            },
            lanes, log, ct);

        // Back into the caller's replicate order - ReadMany returns in completion order, and a file
        // ordering that shuffles between runs makes two ms2_signal.parquet files hard to diff.
        var order = ordered.Select((s, i) => (s, i)).ToDictionary(t => t.s, t => t.i, StringComparer.Ordinal);
        entries = entries.OrderBy(e => order.TryGetValue(e.Sample, out var i) ? i : int.MaxValue).ToList();

        Write(outputDir, entries);
        WriteCycles(outputDir, entries.Select(e => e.Sample), cyclesBySample);
        var usable = entries.Count(e => e.IsUsable);
        log?.Invoke($"  Acquired MS2 signal read for {usable:N0} of {entries.Count:N0} file(s); "
            + $"wrote {FileName}.");
        return new PopulateResult(entries, matched.Count, unmatched.Count, usable);
    }

    /// <summary>Persist the entries, replacing whatever was there.</summary>
    public static void Write(string outputDir, IReadOnlyList<Entry> entries)
    {
        Directory.CreateDirectory(outputDir);
        var meta = new List<ParquetWideWriter.MetaColumn>
        {
            ParquetWideWriter.Strings("sample", entries.Select(e => e.Sample).ToArray()),
            ParquetWideWriter.Strings("data_path", entries.Select(e => e.DataPath).ToArray()),
            ParquetWideWriter.Strings("status", entries.Select(e => e.Status).ToArray()),
            ParquetWideWriter.Strings("reader", entries.Select(e => e.Reader).ToArray()),
            ParquetWideWriter.Strings("signal_source", entries.Select(e => e.SignalSource).ToArray()),
            ParquetWideWriter.Longs("ms1_count", entries.Select(e => e.Ms1Count).ToArray()),
            ParquetWideWriter.Longs("ms2_count", entries.Select(e => e.Ms2Count).ToArray()),
            ParquetWideWriter.Doubles(
                "total_ms2_signal", entries.Select(e => e.TotalMs2Signal).ToArray()),
            ParquetWideWriter.Doubles("rt_start_min", entries.Select(e => e.RtStartMin).ToArray()),
            ParquetWideWriter.Doubles("rt_stop_min", entries.Select(e => e.RtStopMin).ToArray()),
            ParquetWideWriter.Strings("cycle_model", entries.Select(e => e.CycleModel).ToArray()),
        };
        ParquetWideWriter.Write(
            Path.Combine(outputDir, FileName), meta,
            Array.Empty<string>(), Array.Empty<double[]>(), entries.Count);
    }

    /// <summary>
    /// Persist the per-cycle traces, in the caller's replicate order, as one long table.
    /// </summary>
    public static void WriteCycles(
        string outputDir,
        IEnumerable<string> sampleOrder,
        IReadOnlyDictionary<string, IReadOnlyList<Ms2Cycle>> cyclesBySample)
    {
        Directory.CreateDirectory(outputDir);
        var samples = new List<string>();
        var index = new List<long>();
        var rtStart = new List<double>();
        var rtStop = new List<double>();
        var counts = new List<long>();
        var signal = new List<double>();

        foreach (var sample in sampleOrder)
        {
            if (!cyclesBySample.TryGetValue(sample, out var cycles))
                continue;
            foreach (var cycle in cycles)
            {
                samples.Add(sample);
                index.Add(cycle.Index);
                rtStart.Add(cycle.RtStartMin);
                rtStop.Add(cycle.RtStopMin);
                counts.Add(cycle.Ms2Count);
                signal.Add(cycle.Ms2Signal);
            }
        }

        var meta = new List<ParquetWideWriter.MetaColumn>
        {
            ParquetWideWriter.Strings("sample", samples.ToArray()),
            ParquetWideWriter.Longs("cycle", index.ToArray()),
            ParquetWideWriter.Doubles("rt_start_min", rtStart.ToArray()),
            ParquetWideWriter.Doubles("rt_stop_min", rtStop.ToArray()),
            ParquetWideWriter.Longs("ms2_count", counts.ToArray()),
            ParquetWideWriter.Doubles("ms2_signal", signal.ToArray()),
        };
        ParquetWideWriter.Write(
            Path.Combine(outputDir, CyclesFile), meta,
            Array.Empty<string>(), Array.Empty<double[]>(), samples.Count);
    }

    /// <summary>
    /// The cycle trace for one replicate, in acquisition order, or empty when none was recorded.
    /// </summary>
    public static IReadOnlyList<Ms2Cycle> ReadCycles(string outputDir, string sample)
    {
        var path = Path.Combine(outputDir, CyclesFile);
        if (!File.Exists(path))
            return Array.Empty<Ms2Cycle>();
        try
        {
            using var reader = ParquetColumnReader.Open(path);
            if (reader.RowCount == 0 || !reader.HasColumn("sample"))
                return Array.Empty<Ms2Cycle>();

            var samples = reader.ReadStrings("sample");
            var index = Longs(reader, "cycle", samples.Length);
            var rtStart = Doubles(reader, "rt_start_min", samples.Length);
            var rtStop = Doubles(reader, "rt_stop_min", samples.Length);
            var counts = Longs(reader, "ms2_count", samples.Length);
            var signal = Doubles(reader, "ms2_signal", samples.Length);

            var cycles = new List<Ms2Cycle>();
            for (var i = 0; i < samples.Length; i++)
                if (string.Equals(samples[i], sample, StringComparison.Ordinal))
                    cycles.Add(new Ms2Cycle(
                        (int)index[i], rtStart[i], rtStop[i], (int)counts[i], signal[i]));
            return cycles;
        }
        catch (Exception)
        {
            // Same contract as Read: a trace that cannot be read costs the acquired band on one
            // plot, never the report.
            return Array.Empty<Ms2Cycle>();
        }
    }

    /// <summary>Replicates that have a cycle trace, for a picker that should only offer those.</summary>
    public static IReadOnlyList<string> SamplesWithCycles(string outputDir)
    {
        var path = Path.Combine(outputDir, CyclesFile);
        if (!File.Exists(path))
            return Array.Empty<string>();
        try
        {
            using var reader = ParquetColumnReader.Open(path);
            if (reader.RowCount == 0 || !reader.HasColumn("sample"))
                return Array.Empty<string>();
            return reader.ReadStrings("sample").Distinct(StringComparer.Ordinal).ToList();
        }
        catch (Exception)
        {
            return Array.Empty<string>();
        }
    }

    /// <summary>Every persisted entry, or an empty list when the file is absent or unreadable.</summary>
    public static IReadOnlyList<Entry> Read(string outputDir)
    {
        var path = Path.Combine(outputDir, FileName);
        if (!File.Exists(path))
            return Array.Empty<Entry>();
        try
        {
            using var reader = ParquetColumnReader.Open(path);
            if (reader.RowCount == 0 || !reader.HasColumn("sample"))
                return Array.Empty<Entry>();

            var samples = reader.ReadStrings("sample");
            var paths = Strings(reader, "data_path", samples.Length);
            var status = Strings(reader, "status", samples.Length);
            var readers = Strings(reader, "reader", samples.Length);
            var sources = Strings(reader, "signal_source", samples.Length);
            var ms1 = Longs(reader, "ms1_count", samples.Length);
            var ms2 = Longs(reader, "ms2_count", samples.Length);
            var total = reader.HasColumn("total_ms2_signal")
                ? reader.ReadDoubles("total_ms2_signal")
                : new double[samples.Length];
            var rtStart = Doubles(reader, "rt_start_min", samples.Length);
            var rtStop = Doubles(reader, "rt_stop_min", samples.Length);
            var cycles = Strings(reader, "cycle_model", samples.Length);

            var entries = new List<Entry>(samples.Length);
            for (var i = 0; i < samples.Length; i++)
                entries.Add(new Entry(
                    samples[i], paths[i], status[i], readers[i], sources[i],
                    ms1[i], ms2[i], total[i], rtStart[i], rtStop[i], cycles[i]));
            return entries;
        }
        catch (Exception)
        {
            // A denominator that cannot be read is a plot without an acquired bar, never a failed
            // report. The accounting's own numbers do not depend on this file at all.
            return Array.Empty<Entry>();
        }
    }

    /// <summary>Sample id -> acquired MS2 total, for the usable reads only.</summary>
    public static IReadOnlyDictionary<string, double> ReadTotals(string outputDir)
    {
        var totals = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (var entry in Read(outputDir))
            if (entry.IsUsable)
                totals[entry.Sample] = entry.TotalMs2Signal;
        return totals;
    }

    private static string[] Strings(ParquetColumnReader reader, string name, int n) =>
        reader.HasColumn(name) ? reader.ReadStrings(name) : Enumerable.Repeat("", n).ToArray();

    /// <summary>
    /// A whole-number column. Read as doubles because that is what ParquetColumnReader exposes -
    /// the same route <see cref="Ms2SignalAccounting"/> takes for its own count columns - and the
    /// counts here are spectrum counts, far inside a double's exact integer range.
    /// </summary>
    private static long[] Longs(ParquetColumnReader reader, string name, int n)
    {
        if (!reader.HasColumn(name))
            return new long[n];
        var raw = reader.ReadDoubles(name);
        var values = new long[raw.Length];
        for (var i = 0; i < raw.Length; i++)
            values[i] = double.IsFinite(raw[i]) ? (long)raw[i] : 0L;
        return values;
    }

    private static double[] Doubles(ParquetColumnReader reader, string name, int n) =>
        reader.HasColumn(name) ? reader.ReadDoubles(name) : Enumerable.Repeat(double.NaN, n).ToArray();
}
