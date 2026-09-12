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
    /// Concurrent instrument-file reads. Reading is what costs, so this is the one setting that
    /// changes how long a cohort takes - and the choice it presents is speed against MEMORY, not
    /// speed against nothing.
    /// </summary>
    /// <remarks>
    /// <para>Measured on 4.44 GB Thermo files of about 168,920 spectra each, over an SMB share, on a
    /// 16-processor machine:</para>
    /// <para>Measured on FLARE Extended over a network share, same files, <c>--force</c> each time,
    /// nothing else running. A FLARE file is about 166,000 spectra.</para>
    /// <list type="table">
    /// <listheader><term>lanes</term><description>wall clock / per-file rate / AGGREGATE / peak working set</description></listheader>
    /// <item><term>2</term><description>9.1 min for 4 files, 591-708 spectra/s each, <b>~1,220 spectra/s</b></description></item>
    /// <item><term>4</term><description>6.9 min for 4 files, 430-438 each, <b>~1,600 spectra/s</b>, 15.4 GB</description></item>
    /// <item><term>8</term><description>11.1 min for 8 files, 262-269 each, <b>~1,990 spectra/s</b>, 32.1 GB</description></item>
    /// </list>
    /// <para>Aggregate throughput keeps climbing - eight lanes is 64% faster than two - while the
    /// per-file rate falls from ~650 to ~265 spectra/s, which is the share saturating. So the choice
    /// is not speed against nothing; it is speed against MEMORY.</para>
    /// <para><b>Four, because the default runs on machines nobody measured.</b> Eight reached
    /// 32.1 GB of 64 and was still climbing when the run ended - which on a 32 GB laptop is not
    /// slower, it is a run that cannot finish. The failure is not graceful either: on 2026-09-11 two
    /// ion-accounting processes died with a native access violation inside DuckDB's allocator, 13
    /// and 16 minutes into a 6.5 GB Skyline document being opened beside them (see the DuckDB
    /// caution in CLAUDE.md), because a pool sized when memory was free cannot be honoured when it
    /// is not.</para>
    /// <para>So the default is chosen for the machine we cannot see, not the fastest one we
    /// measured: four takes 31 of the available 64 percentage points at half the memory, and
    /// <c>--probe-lanes</c> exists for anyone whose storage or RAM says otherwise. On the share this
    /// was built against the probe recommends EIGHT, and shipping four anyway is the deliberate
    /// choice - speed that needs half a machine to itself is not a default.</para>
    /// <para><b>An earlier measurement said four was WORSE than two</b> (205 vs 185 s per file, at
    /// 27.8 GB). That was before <c>ForEachSample</c> took a <c>wanted</c> predicate: the loader was
    /// building a claim set for every sample in the cohort and the caller was discarding all but
    /// <c>--max</c> of them, so the high-lane arms were memory-bound rather than share-bound. The
    /// conditions the old number described no longer exist, which is why it was re-measured rather
    /// than trusted.</para>
    /// <para>Still unmeasured: one share, one file size. The lever of LOCALITY is also untested
    /// against lane count - a local copy read at 1,077 spectra/s on one lane against ~650 over the
    /// share, and a sequential copy of the same file runs at 212 MB/s, twenty-odd times faster than
    /// the walk reads it. Staging a file locally before reading it is still the thing worth trying
    /// next. Measure rather than assuming - and do not read a stall out of a quiet log, because a
    /// file only reports when it is done.</para>
    /// </remarks>
    public const int DefaultLanes = 4;

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

        var mergedRoot = MergedDataset.Locate(outputDir);
        if (mergedRoot is null)
        {
            log?.Invoke(
                $"  Ion accounting skipped: no {MergedDataset.DirectoryName}/ or "
                + $"{MergedDataset.LegacyFileName} in the output directory.");
            return null;
        }

        var dataset = MergedDataset.Open(mergedRoot);
        var representative = dataset.RepresentativeFile();
        var cols = SignalColumns.Resolve(ParquetTable.ReadColumnNames(representative).ToList());
        if (cols is null)
        {
            log?.Invoke(
                "  Ion accounting skipped: the merged table lacks Product Mz, Precursor Mz, Start "
                + "Time or End Time. Re-export the report with those columns to enable it.");
            return null;
        }

        var classified = AssignedPeptides.Classify(outputDir, lists);
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

        // sample_metadata.csv is written by every run and is right here, so a caller that passed no
        // types still gets bars colored by type rather than one flat color.
        sampleTypes ??= IonAccountingStore.SampleTypes(outputDir);
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

        // What the cache already holds for THESE settings. A keyed cache is not automatically
        // complete: the key covers the files that could be measured, not the ones that were, so a
        // --max run writes a whole-cohort key over a handful of rows. Reusing by COVERAGE rather
        // than trusting the key outright fixes that, and makes adding a plate to a cohort measure
        // only the new plate.
        var reusable = new Dictionary<string, IonAccountingRow>(StringComparer.Ordinal);
        var reusedCycles = new List<IonCycleRow>();
        if (!force)
        {
            var cached = IonAccountingStore.Read(outputDir);
            if (cached is not null && cached.MatchesSettings(settingsKey))
            {
                foreach (var row in cached.Rows)
                {
                    if (row.IsUsable)
                        reusable[row.Sample] = row;
                }

                var missing = resolution.Matched.Keys
                    .Count(s => !reusable.ContainsKey(s));
                if (missing == 0)
                {
                    log?.Invoke(
                        $"  Ion accounting: reusing the cache for {cached.Rows.Count:N0} replicate(s) "
                        + $"({IonAccountingStore.SummarizeSettings(productText, precursorText, schemeText, lists.Count)}).");
                    return cached;
                }

                // Read the reusable replicates' traces NOW: the first incremental save rewrites
                // ion_cycles.parquet from what is in memory, so anything not loaded is lost.
                //
                // One read of the whole table, filtered here, rather than one read per sample: the
                // file holds the cohort, so per-sample reads are quadratic in the replicate count -
                // fine for the six this was first written against, and 192 reads of a 190,000-row
                // table for a real one.
                reusedCycles.AddRange(
                    IonAccountingStore.ReadCycles(outputDir)
                        .Where(c => reusable.ContainsKey(c.Sample)));

                log?.Invoke(
                    $"  Ion accounting: the cache covers {reusable.Count:N0} replicate(s) of these "
                    + $"settings; measuring the {missing:N0} it does not.");
            }
            else if (cached is not null)
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

        var rows = new List<IonAccountingRow>(reusable.Values);
        var cycles = new List<IonCycleRow>(reusedCycles);
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

        // Decided UP FRONT rather than inside the callback, because the callback runs after the
        // claims have already been built. The order is the query's ORDER BY, so taking the first
        // maxReplicates of the samples that still need measuring picks exactly the ones the loop
        // below would have accepted.
        var stillNeeded = resolution.Matched.Keys
            .Where(s => !reusable.ContainsKey(s))
            .OrderBy(s => s, StringComparer.Ordinal)
            .ToList();
        if (maxReplicates > 0 && stillNeeded.Count > maxReplicates)
            stillNeeded = stillNeeded.Take(maxReplicates).ToList();
        var wanted = new HashSet<string>(stillNeeded, StringComparer.Ordinal);

        ClaimedRegionLoader.ForEachSample(
            dataset, cols, scheme, productTolerance, precursorTolerance, classified.Classes,
            (sample, loaded) =>
            {
                ct.ThrowIfCancellationRequested();
                if (!resolution.Matched.TryGetValue(sample, out var path))
                    return;
                if (reusable.ContainsKey(sample))
                    return;   // already measured for these settings

                // No second bound on maxReplicates here. `wanted` already holds exactly the samples
                // this run should measure, chosen by ordinal sort; re-bounding on arrival order
                // would disagree with it wherever DuckDB's collation differs from ordinal, and the
                // symptom would be measuring fewer replicates than asked with nothing said.
                read++;

                // Blocks BEFORE the index is built, so at most `lanes` replicates' claims exist at
                // once. Waiting after would let the producer run ahead of the readers and hold the
                // whole cohort's geometry.
                gate.Wait(ct);

                var index = new ClaimedSignalIndex(loaded.Regions, classified.ListNames.Count);

                // The second claim set: everything these peptides can account for, not just what the
                // document quantifies on. Built with NO list count - the per-list breakdown answers
                // "what share of the assigned signal does this panel hold", which is a question
                // about the quantified set, and carrying masks here would double the memory of the
                // larger of the two indexes for a number nothing plots.
                var explained = loaded.HasExplained
                    ? new ClaimedSignalIndex(loaded.ExplainedRegions, listCount: 0)
                    : null;
                var request = new IonAccountingRequest(
                    index, scheme, classified.ListNames, explained);

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
                                        cycle.Ms1Assigned, cycle.Ms2Assigned,
                                        cycle.Ms2Explained));
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
            memoryBudgetMb,
            wanted: wanted.Contains);

        Task.WaitAll(pending.ToArray(), ct);

        // Replicates with no file of their own still get a row, so the plot can show a gap rather
        // than silently omitting an injection.
        var measured = new HashSet<string>(rows.Select(r => r.Sample), StringComparer.Ordinal);
        foreach (var sample in samples)
        {
            if (measured.Contains(sample))
                continue;
            // NotFound whichever it was - no file of its own, or a file another replicate also
            // claimed. The distinction is in the pairing report above rather than in a status,
            // because there is no reading of this row that should differ: either way the replicate
            // has no measured denominator and the plot must show a gap rather than a zero.
            rows.Add(new IonAccountingRow(
                sample, SampleTypeOf(sampleTypes, sample), "", Ms2ReadStatus.NotFound, "none",
                0, 0, 0, 0, 0, 0, 0, false, double.NaN, double.NaN, 0, 0, 0, 0,
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
        string schemeText, AssignedPeptides.Classified classified,
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
            record.Ms2Explained,
            record.HasExplained,
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
    /// <see cref="AssignedPeptides"/>'s own list deliberately: both read the same file, and a column
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
