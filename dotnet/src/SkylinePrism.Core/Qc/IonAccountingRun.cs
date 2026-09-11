using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.RawData;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// Computes ion accounting for a cohort: how many ions reached the detector in each replicate, and
/// what fraction of them a peptide sequence explains, at MS1 and at MS2.
///
/// <para><b>The shape of the work.</b> One streaming pass over <c>merged_data/</c> supplies the
/// geometry - where Skyline extracted, when, and for which peptides - and one pass over each
/// instrument file supplies every magnitude. The two are interleaved: a replicate's claims are built
/// as its rows arrive, its file is walked immediately, and the claims are then dropped. That is what
/// keeps peak memory at one replicate's claims (about 30 MB) rather than the cohort's.</para>
///
/// <para><b>Why this replaces exporting Skyline's ion counts.</b> The alternative was a report
/// carrying <c>LC Peak Transition Ion Count</c>, which Skyline computes 29x slower per row - about
/// four hours instead of ten minutes on a 46M-row document - and which still could not be summed
/// correctly, because per-transition totals count shared signal once per transition. Computing from
/// the spectra costs single-digit seconds per file and gets the union right by construction.</para>
/// </summary>
public static class IonAccountingRun
{
    /// <summary>
    /// Concurrent instrument-file reads.
    ///
    /// <para>Eight because the read is what costs: measured on one 4.44 GB Thermo file, 206.7 s
    /// total of which 204.5 s is inside the reader and 1.0 s is masking 465,307 claims against
    /// 168,920 spectra. Per-spectrum decoding runs at about 817 spectra/s cold, and that figure is
    /// the same from local disk as over the share to within the network's 2.6x - so the limit is
    /// the decoding, not the storage, and lanes buy close to their count until the cores run out.
    /// Sixteen logical processors was the machine this was measured on.</para>
    /// </summary>
    public const int DefaultLanes = 8;

    /// <summary>
    /// Compute for every replicate in <paramref name="outputDir"/>, reusing the cache unless
    /// <paramref name="force"/> or the settings have changed. Null when the inputs to do it are
    /// missing, with the reason logged.
    /// </summary>
    /// <param name="rawDir">
    /// Directory of instrument files. Replicates are paired to files one-to-one by name; a replicate
    /// with no file of its own gets no numbers rather than a guessed denominator.
    /// </param>
    /// <param name="precursorTolerance">
    /// The document's precursor extraction window. Null computes the MS2 half only - an MS1 fraction
    /// with a guessed tolerance would be a different number with nothing on the plot to say so.
    /// </param>
    public static IonAccountingResult? Compute(
        string outputDir, string rawDir, IsolationScheme scheme,
        ProductMassTolerance productTolerance, ProductMassTolerance? precursorTolerance,
        IReadOnlyList<ProteinList> lists,
        IReadOnlyDictionary<string, string>? sampleTypes = null,
        Action<string>? log = null, int memoryBudgetMb = 0, bool force = false,
        int maxReplicates = 0, int lanes = 0, CancellationToken ct = default)
    {
        if (productTolerance is null)
            throw new ArgumentNullException(nameof(productTolerance));

        if (!IonAccountingReaders.Available)
        {
            log?.Invoke(
                "  Ion accounting skipped: this build has no instrument-file reader, so the number "
                + "of ions acquired cannot be measured.");
            return null;
        }

        var mergedRoot = Path.Combine(outputDir, "merged_data");
        if (!MergedDataset.Exists(mergedRoot))
        {
            log?.Invoke("  Ion accounting skipped: no merged_data/ in the output directory.");
            return null;
        }

        var dataset = MergedDataset.Open(mergedRoot);
        var representative = dataset.RepresentativeFile();
        var cols = Ms2SignalRegions.Resolve(ParquetTable.ReadColumnNames(representative).ToList());
        if (cols is null)
        {
            log?.Invoke(
                "  Ion accounting skipped: the merged table lacks Product Mz, Precursor Mz, Start "
                + "Time or End Time. Re-export the report with those columns to enable it.");
            return null;
        }

        var classified = Ms2SignalPeptides.Classify(outputDir, lists);
        if (classified.AssignedPeptides == 0)
        {
            log?.Invoke(
                "  Ion accounting skipped: peptides_rollup.parquet named no peptides, so nothing "
                + "claims any signal.");
            return null;
        }
        if (lists.Count > 0 && !classified.HasGroupColumns)
        {
            log?.Invoke(
                "  Ion accounting: corrected_peptides.parquet has no protein-group columns, so no "
                + "protein list could be matched. Only the assigned totals are meaningful.");
        }

        // Pair replicates to files before anything expensive: the file set is part of the cache key,
        // and a cohort with no files at all has no work to do.
        var samples = SamplesOf(outputDir);
        var files = ReplicateDataFiles.Enumerate(rawDir);
        var resolution = ReplicateDataFiles.ResolveAll(samples, files);
        ReportPairing(rawDir, samples.Count, files.Count, resolution, log);
        if (resolution.Matched.Count == 0)
            return null;

        var productText = productTolerance.Describe();
        var precursorText = precursorTolerance?.Describe() ?? "not read";
        var schemeText = scheme.Describe();
        var sources = resolution.Matched.Values
            .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
            .Append(representative)
            .ToList();
        var settingsKey = IonAccountingStore.SettingsKeyFor(
            productText, precursorText, schemeText, classified.ListNames, sources);

        if (!force)
        {
            var cached = IonAccountingStore.Read(outputDir);
            if (cached is not null && cached.MatchesSettings(settingsKey))
            {
                log?.Invoke(
                    $"  Ion accounting: reusing the cache for {cached.Rows.Count:N0} replicate(s) "
                    + $"({IonAccountingStore.SummarizeSettings(productText, precursorText, schemeText, lists.Count)}).");
                return cached;
            }
            if (cached is not null)
            {
                log?.Invoke(
                    "  Ion accounting: the cache was computed for different settings or different "
                    + "files, so it is being recomputed.");
            }
        }

        log?.Invoke(
            $"  Ion accounting over {resolution.Matched.Count:N0} replicate(s): "
            + IonAccountingStore.SummarizeSettings(
                productText, precursorText, schemeText, lists.Count) + ".");
        if (precursorTolerance is null)
        {
            log?.Invoke(
                "    The document did not give a precursor extraction window, so only the MS2 half "
                + "is computed.");
        }

        var rows = new List<IonAccountingRow>();
        var cycles = new List<IonCycleRow>();
        var clock = Stopwatch.StartNew();
        var read = 0;
        var effectiveLanes = Math.Max(1, lanes > 0 ? lanes : DefaultLanes);
        log?.Invoke(
            $"    Reading {effectiveLanes} file(s) at a time; the merged_data pass stays on one "
            + "thread.");

        // The producer is this thread and it must stay the only one touching DuckDB. The workers
        // only read instrument files, which is 99.5% of the cost.
        using var gate = new SemaphoreSlim(effectiveLanes);
        var pending = new List<Task>();
        var sync = new object();

        ClaimedRegionLoader.ForEachSample(
            dataset, cols, scheme, productTolerance, precursorTolerance, classified.Classes,
            (sample, loaded) =>
            {
                ct.ThrowIfCancellationRequested();
                if (!resolution.Matched.TryGetValue(sample, out var path))
                    return;
                if (maxReplicates > 0 && read >= maxReplicates)
                    return;
                read++;

                // Blocks BEFORE the index is built, so at most `lanes` replicates' claims exist at
                // once. Waiting after would let the producer run ahead of the readers and hold the
                // whole cohort's geometry.
                gate.Wait(ct);

                var index = new ClaimedSignalIndex(loaded.Regions, classified.ListNames.Count);
                var request = new IonAccountingRequest(index, scheme, classified.ListNames);

                pending.Add(Task.Run(
                    () =>
                    {
                        try
                        {
                            // Each file's lines are collected and emitted together. Interleaving
                            // eight files' multi-line reports would make the log unreadable, and
                            // this log is how a surprising fraction gets explained.
                            var lines = new List<string> { $"  {sample}: {loaded.Describe()}" };
                            var record = IonAccountingReaders.Read(
                                path, request, line => lines.Add(line), ct);

                            lock (sync)
                            {
                                foreach (var line in lines)
                                    log?.Invoke(line);

                                rows.Add(ToRow(
                                    sample, sampleTypes, path, record, loaded,
                                    classified.ListNames.Count));
                                foreach (var cycle in record.Cycles)
                                {
                                    cycles.Add(new IonCycleRow(
                                        sample, cycle.Index, cycle.RtStartMin, cycle.RtStopMin,
                                        cycle.Ms1Count, cycle.Ms2Count,
                                        cycle.Ms1Acquired, cycle.Ms2Acquired,
                                        cycle.Ms1Assigned, cycle.Ms2Assigned));
                                }

                                // Written after EVERY replicate, not once at the end. This is the
                                // longest operation in the product, and the first real run of it
                                // was interrupted after an hour and left nothing behind at all. A
                                // partial cache carries a settings key that stops matching once
                                // more replicates are added, so the next run recomputes rather
                                // than trusting a short file.
                                SaveProgress(
                                    outputDir, settingsKey, productText, precursorText, schemeText,
                                    classified, rows, cycles);
                            }
                        }
                        finally
                        {
                            gate.Release();
                        }
                    },
                    ct));
            },
            memoryBudgetMb);

        Task.WaitAll(pending.ToArray(), ct);

        // Replicates with no file of their own still get a row, so the plot can show a gap rather
        // than silently omitting an injection.
        foreach (var sample in samples)
        {
            if (resolution.Matched.ContainsKey(sample))
                continue;
            var status = resolution.Ambiguous.Contains(sample, StringComparer.Ordinal)
                ? Ms2ReadStatus.NotFound
                : Ms2ReadStatus.NotFound;
            rows.Add(new IonAccountingRow(
                sample, SampleTypeOf(sampleTypes, sample), "", status, "none",
                0, 0, 0, 0, 0, 0, double.NaN, double.NaN, 0, 0, 0, 0,
                new double[classified.ListNames.Count], new double[classified.ListNames.Count]));
        }

        rows.Sort((a, b) => string.Compare(a.Sample, b.Sample, StringComparison.Ordinal));

        var result = new IonAccountingResult(
            settingsKey, productText, precursorText, schemeText, classified.ListNames,
            classified.AssignedPeptides, classified.HasGroupColumns, rows, cycles);

        IonAccountingStore.Write(outputDir, result);
        ReportTotals(result, clock, log);
        return result;
    }

    /// <summary>
    /// Write what has been read so far, so an interrupted run is not a wasted one.
    /// </summary>
    private static void SaveProgress(
        string outputDir, string settingsKey, string productText, string precursorText,
        string schemeText, Ms2SignalPeptides.Classified classified,
        IReadOnlyList<IonAccountingRow> rows, IReadOnlyList<IonCycleRow> cycles)
    {
        try
        {
            IonAccountingStore.Write(outputDir, new IonAccountingResult(
                settingsKey, productText, precursorText, schemeText, classified.ListNames,
                classified.AssignedPeptides, classified.HasGroupColumns, rows, cycles));
        }
        catch (IOException)
        {
            // A cache that cannot be written is not a reason to abandon the reads already done.
        }
    }

    private static IonAccountingRow ToRow(
        string sample, IReadOnlyDictionary<string, string>? sampleTypes, string path,
        IonAccountingRecord record, ClaimedRegionLoader.Loaded loaded, int listCount) =>
        new(sample,
            SampleTypeOf(sampleTypes, sample),
            Path.GetFileName(path.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)),
            record.Status,
            record.Reader,
            record.Ms1Count,
            record.Ms2Count,
            record.Ms1Acquired,
            record.Ms2Acquired,
            record.Ms1Assigned,
            record.Ms2Assigned,
            record.RtStartMin,
            record.RtStopMin,
            loaded.Regions.Count,
            record.ScansOutsideScheme,
            record.SpectraMissingInjectionTime,
            record.Cycles.Count,
            Padded(record.Ms1ByList, listCount),
            Padded(record.Ms2ByList, listCount));

    private static double[] Padded(IReadOnlyList<double> values, int count)
    {
        var padded = new double[count];
        for (var i = 0; i < count && i < values.Count; i++)
            padded[i] = values[i];
        return padded;
    }

    private static string SampleTypeOf(
        IReadOnlyDictionary<string, string>? sampleTypes, string sample) =>
        sampleTypes is not null && sampleTypes.TryGetValue(sample, out var type) ? type : "";

    /// <summary>
    /// Every sample id in the merged table. Taken from <c>sample_metadata.csv</c> when it is there,
    /// because that is one small file rather than a scan, and from the table itself otherwise.
    /// </summary>
    private static List<string> SamplesOf(string outputDir)
    {
        var metadata = Path.Combine(outputDir, "sample_metadata.csv");
        if (File.Exists(metadata))
        {
            try
            {
                var samples = new List<string>();
                var lines = File.ReadAllLines(metadata);
                if (lines.Length > 1)
                {
                    var header = lines[0].Split(',');
                    // sample_id FIRST. The file carries both, and they are different things:
                    // sample_id is the merged table's own "<replicate>__@__<batch>" key, which is
                    // what ForEachSample yields, while sample is the bare replicate name. Keying on
                    // the bare name matches nothing and the run silently produces no rows.
                    var column = Array.FindIndex(
                        header, h => h.Trim().Equals("sample_id", StringComparison.OrdinalIgnoreCase));
                    if (column < 0)
                    {
                        column = Array.FindIndex(
                            header, h => h.Trim().Equals("sample", StringComparison.OrdinalIgnoreCase));
                    }
                    if (column >= 0)
                    {
                        foreach (var line in lines.Skip(1))
                        {
                            var parts = line.Split(',');
                            if (column < parts.Length && parts[column].Length > 0)
                                samples.Add(parts[column].Trim());
                        }
                    }
                }
                if (samples.Count > 0)
                    return samples.Distinct(StringComparer.Ordinal).ToList();
            }
            catch (IOException)
            {
                // Fall through to the table.
            }
        }
        // Fall back to the peptide matrix's own columns: it is wide, one column per replicate, so
        // the sample set is its header. Cheaper than a DISTINCT over merged_data/, which is a full
        // scan of the cohort - and these are the replicates the analysis actually reports.
        var rollup = Path.Combine(outputDir, "peptides_rollup.parquet");
        if (!File.Exists(rollup))
            return new List<string>();
        try
        {
            using var reader = ParquetColumnReader.Open(rollup);
            return reader.ColumnNames
                .Where(c => !NonSampleColumns.Contains(c, StringComparer.OrdinalIgnoreCase))
                .ToList();
        }
        catch (Exception)
        {
            return new List<string>();
        }
    }

    /// <summary>
    /// Columns of a wide peptide parquet that are not replicates. Kept beside
    /// <see cref="Ms2SignalPeptides"/>'s own list deliberately: both read the same file, and a column
    /// added to one and not the other would surface as a phantom replicate with no data file.
    /// </summary>
    private static readonly string[] NonSampleColumns =
    {
        "peptide", "peptide_modified_sequence", "modified_sequence", "n_transitions", "mean_rt",
        "protein", "leading_protein", "leading_gene_name", "leading_name",
    };

    private static void ReportPairing(
        string rawDir, int sampleCount, int fileCount,
        ReplicateDataFiles.Resolution resolution, Action<string>? log)
    {
        if (log is null)
            return;

        log($"  Ion accounting: {fileCount:N0} data file(s) in {rawDir}, paired to "
            + $"{resolution.Matched.Count:N0} of {sampleCount:N0} replicate(s).");
        if (resolution.Unmatched.Count > 0)
        {
            log($"    {resolution.Unmatched.Count:N0} replicate(s) matched no data file, e.g. "
                + string.Join(", ", resolution.Unmatched.Take(3))
                + ". They get no acquired total rather than a guessed one.");
        }
        if (resolution.Ambiguous.Count > 0)
        {
            log($"    {resolution.Ambiguous.Count:N0} replicate(s) matched a file another replicate "
                + "also matched, so none of them was assigned it: "
                + string.Join(", ", resolution.Ambiguous.Take(4))
                + ". Reference and QC injections are usually named identically in every plate's "
                + "document, so each plate needs its own file.");
        }
        if (resolution.Matched.Count == 0)
            log("  Ion accounting skipped: no replicate could be paired to a data file.");
    }

    private static void ReportTotals(
        IonAccountingResult result, Stopwatch clock, Action<string>? log)
    {
        if (log is null)
            return;

        var usable = result.Usable;
        log($"  Ion accounting finished in {clock.Elapsed.TotalMinutes:F1} min: "
            + $"{usable.Count:N0} replicate(s) with numbers, {result.Cycles.Count:N0} cycles cached.");
        if (usable.Count == 0)
            return;

        var ms2 = usable.Where(r => double.IsFinite(r.Ms2Fraction)).Select(r => r.Ms2Fraction).ToArray();
        var ms1 = usable.Where(r => double.IsFinite(r.Ms1Fraction)).Select(r => r.Ms1Fraction).ToArray();
        if (ms2.Length > 0)
        {
            log($"    MS2 assigned fraction: {IonAccountingStore.Percent(ms2.Min())} to "
                + $"{IonAccountingStore.Percent(ms2.Max())} (median "
                + $"{IonAccountingStore.Percent(Median(ms2))}).");
        }
        if (ms1.Length > 0)
        {
            log($"    MS1 assigned fraction: {IonAccountingStore.Percent(ms1.Min())} to "
                + $"{IonAccountingStore.Percent(ms1.Max())} (median "
                + $"{IonAccountingStore.Percent(Median(ms1))}).");
        }

        var exceeded = usable.Count(r => r.Exceeded);
        if (exceeded > 0)
        {
            log($"    WARNING: {exceeded:N0} replicate(s) assigned more signal than was acquired, "
                + "which is impossible. Their fractions will not be plotted.");
        }
    }

    private static double Median(double[] values)
    {
        var sorted = (double[])values.Clone();
        Array.Sort(sorted);
        var mid = sorted.Length / 2;
        return sorted.Length % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    }
}
