using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.RawData;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// One replicate's ion accounting, as it is cached and as the plots read it.
/// </summary>
/// <param name="DataFile">The instrument file this came from, or empty when none was resolved.</param>
/// <param name="Status">The read's outcome, kept so a failed read is distinguishable from an empty one.</param>
public sealed record IonAccountingRow(
    string Sample,
    string SampleType,
    string DataFile,
    Ms2ReadStatus Status,
    string Reader,
    int Ms1Count,
    int Ms2Count,
    double Ms1Acquired,
    double Ms2Acquired,
    double Ms1Assigned,
    double Ms2Assigned,
    double Ms2Explained,
    bool HasExplained,
    double RtStartMin,
    double RtStopMin,
    int Claims,
    int ScansOutsideScheme,
    int SpectraMissingInjectionTime,
    int CycleCount,
    IReadOnlyList<double> Ms1ByList,
    IReadOnlyList<double> Ms2ByList)
{
    public double Ms1Fraction => Ms1Acquired > 0 ? Ms1Assigned / Ms1Acquired : double.NaN;
    public double Ms2Fraction => Ms2Acquired > 0 ? Ms2Assigned / Ms2Acquired : double.NaN;

    /// <summary>
    /// The share of acquired MS2 ions the peptides can ACCOUNT FOR - their theoretical b/y ions and
    /// surviving precursor - against <see cref="Ms2Fraction"/>'s share they are quantified on.
    /// NaN when this cache carries no explained total, which is not the same as zero.
    /// </summary>
    public double Ms2ExplainedFraction =>
        HasExplained && Ms2Acquired > 0 ? Ms2Explained / Ms2Acquired : double.NaN;

    /// <inheritdoc cref="IonAccountingRecord.Exceeded"/>
    public bool Exceeded => Ms1Assigned > Ms1Acquired || Ms2Assigned > Ms2Acquired;

    /// <summary>
    /// The explained total exceeded what was acquired, or fell below the quantified total. Both are
    /// impossible and both mean a defect in claim building.
    ///
    /// <para>Deliberately SEPARATE from <see cref="Exceeded"/>. A fault confined to the theoretical
    /// claim set must not blank the quantified fraction as well - the caption that withholds a
    /// figure also names a cause, and naming the isolation scheme for a quantified total that is
    /// perfectly sound sends a reader after the wrong thing.</para>
    /// </summary>
    public bool ExplainedImpossible =>
        HasExplained && (Ms2Explained > Ms2Acquired || Ms2Explained < Ms2Assigned);

    /// <inheritdoc cref="IonAccountingRecord.MeanMs1IonsPerScan"/>
    public double MeanMs1IonsPerScan => Ms1Count > 0 ? Ms1Acquired / Ms1Count : double.NaN;

    /// <inheritdoc cref="IonAccountingRecord.MeanMs2IonsPerScan"/>
    public double MeanMs2IonsPerScan => Ms2Count > 0 ? Ms2Acquired / Ms2Count : double.NaN;

    /// <inheritdoc cref="IonAccountingRecord.IonScaleImplausible"/>
    public bool IonScaleImplausible =>
        Implausible(MeanMs1IonsPerScan) || Implausible(MeanMs2IonsPerScan);

    private static bool Implausible(double meanPerScan) =>
        IonAccountingRecord.IsIonScaleImplausible(meanPerScan);

    /// <summary>Whether this row carries numbers worth plotting.</summary>
    public bool IsUsable => Status == Ms2ReadStatus.Ok && (Ms1Acquired > 0 || Ms2Acquired > 0);
}

/// <summary>
/// A whole cohort's ion accounting plus the settings that produced it.
/// </summary>
/// <param name="Cycles">
/// Every replicate's per-cycle traces - populated by a RUN, and deliberately empty when this comes
/// back from <see cref="IonAccountingStore.Read"/>. A cohort's cycles run to hundreds of thousands
/// of rows and the time plots show one replicate at a time, so a reader asks for the one it needs
/// through <see cref="IonAccountingStore.ReadCycles"/> rather than paying for all of them.
/// </param>
public sealed record IonAccountingResult(
    string SettingsKey,
    string ProductTolerance,
    string PrecursorTolerance,
    string IsolationScheme,
    IReadOnlyList<string> ListNames,
    int AssignedPeptides,
    bool ListsMatchable,
    IReadOnlyList<IonAccountingRow> Rows,
    IReadOnlyList<IonCycleRow> Cycles)
{
    /// <summary>Replicates with numbers, which is what every plot iterates.</summary>
    public IReadOnlyList<IonAccountingRow> Usable => Rows.Where(r => r.IsUsable).ToArray();

    /// <summary>
    /// Whether these cached numbers answer the question the settings now ask. The file is keyed on
    /// its name alone, so without this a re-run replots the previous run's numbers under the new
    /// run's caption - and nothing fails loudly, because both the plot and the caption read as
    /// correct.
    /// </summary>
    public bool MatchesSettings(string settingsKey) =>
        string.Equals(SettingsKey, settingsKey, StringComparison.Ordinal);

    /// <summary>
    /// The median, best and worst replicate by MS2 assigned fraction, or fewer when there are fewer.
    ///
    /// <para>Three, not one: a single representative hides whether the cohort is uniform, and best
    /// against worst is the comparison that tells a user whether one injection misbehaved or the
    /// whole run did. Rows whose fraction is not physical are excluded - see
    /// <see cref="IonAccountingRow.Exceeded"/>.</para>
    /// </summary>
    public IReadOnlyList<IonAccountingRow> Representatives()
    {
        var ranked = Rows
            .Where(r => r.IsUsable && !r.Exceeded && double.IsFinite(r.Ms2Fraction))
            .OrderBy(r => r.Ms2Fraction)
            .ToArray();
        if (ranked.Length == 0)
            return Array.Empty<IonAccountingRow>();
        if (ranked.Length <= 3)
            return ranked;

        // Distinct by sample, so a three-replicate cohort does not list one row three times.
        var picks = new List<IonAccountingRow> { ranked[^1], ranked[ranked.Length / 2], ranked[0] };
        return picks
            .GroupBy(r => r.Sample, StringComparer.Ordinal)
            .Select(g => g.First())
            .ToArray();
    }
}

/// <summary>One cycle of one replicate, the long-format row the time plots read.</summary>
public readonly record struct IonCycleRow(
    string Sample,
    int Cycle,
    double RtStartMin,
    double RtStopMin,
    int Ms1Count,
    int Ms2Count,
    double Ms1Acquired,
    double Ms2Acquired,
    double Ms1Assigned,
    double Ms2Assigned,
    double Ms2Explained = 0);

/// <summary>
/// Reads and writes the ion-accounting cache.
///
/// <para><b>Why a cache at all.</b> Computing this reads every instrument file in the cohort - a
/// terabyte on a network share for the cohort this was built for. The bar plot and both time plots
/// then become pure file reads, so switching replicate or list in the GUI is instant and a re-run of
/// the QC report costs nothing.</para>
///
/// <para><b>Keyed on what was requested, and the key is IN the file.</b> Anything that changes the
/// numbers belongs in <see cref="SettingsKeyFor"/>: both extraction tolerances, the isolation scheme,
/// the selected lists, and a fingerprint of the instrument files and of <c>merged_data/</c>. Nothing
/// fails loudly if a setting is left out - the plots look right and the caption comes from the cache
/// - which is exactly why this is the one place to add it.</para>
/// </summary>
public static class IonAccountingStore
{
    public const string FileName = "ion_accounting.parquet";
    public const string CyclesFile = "ion_cycles.parquet";
    public const string ListsFile = "ion_accounting_lists.parquet";

    /// <summary>
    /// The cache validity key. <paramref name="sources"/> is every input whose CONTENT would change
    /// the answer: the instrument files, and <c>merged_data/</c> for the claim geometry.
    /// </summary>
    public static string SettingsKeyFor(
        string productTolerance, string precursorTolerance, string isolationScheme,
        IEnumerable<string> listNames, IReadOnlyList<string> sources) =>
        string.Join(
            "|",
            // v2: v1 multiplied the intensity by the injection time in MILLISECONDS, so
            // every total it cached is 1000x too large. The fractions were right, but the
            // columns are named "ions" - so the key is bumped to make every directory
            // recompute rather than replot the old magnitudes under the new caption.
            //
            // v3: the explained total arrived. A v2 file is not WRONG - every number in it is
            // still right - but it carries no explained column, so reusing it would draw a
            // section whose second series is silently absent on some replicates and present on
            // others, depending on when each was measured. Recomputing is the honest answer.
            "ions-v3",
            productTolerance,
            precursorTolerance,
            isolationScheme,
            string.Join(",", listNames),
            SourceFingerprint.Compute(sources));

    /// <summary>One phrase naming a set of settings, for the log line on a cache miss.</summary>
    public static string SummarizeSettings(
        string productTolerance, string precursorTolerance, string isolationScheme, int listCount) =>
        $"product {productTolerance}, precursor {precursorTolerance}, "
        + $"isolation scheme \"{isolationScheme}\", {listCount} protein list(s)";

    public static void Write(string outputDir, IonAccountingResult result)
    {
        var rows = result.Rows;
        var n = rows.Count;

        var meta = new List<ParquetWideWriter.MetaColumn>
        {
            ParquetWideWriter.Strings("sample", rows.Select(r => r.Sample).ToArray()),
            ParquetWideWriter.Strings("sample_type", rows.Select(r => r.SampleType).ToArray()),
            ParquetWideWriter.Strings("data_file", rows.Select(r => r.DataFile).ToArray()),
            ParquetWideWriter.Strings("status", rows.Select(r => r.Status.ToString()).ToArray()),
            ParquetWideWriter.Strings("reader", rows.Select(r => r.Reader).ToArray()),
            ParquetWideWriter.Longs("ms1_count", rows.Select(r => (long)r.Ms1Count).ToArray()),
            ParquetWideWriter.Longs("ms2_count", rows.Select(r => (long)r.Ms2Count).ToArray()),
            // LINEAR counts of ions - intensity (a rate) times injection time in SECONDS. Never log.
            ParquetWideWriter.Doubles("ms1_acquired", rows.Select(r => r.Ms1Acquired).ToArray()),
            ParquetWideWriter.Doubles("ms2_acquired", rows.Select(r => r.Ms2Acquired).ToArray()),
            ParquetWideWriter.Doubles("ms1_assigned", rows.Select(r => r.Ms1Assigned).ToArray()),
            ParquetWideWriter.Doubles("ms2_assigned", rows.Select(r => r.Ms2Assigned).ToArray()),
            ParquetWideWriter.Doubles("ms2_explained", rows.Select(r => r.Ms2Explained).ToArray()),
            // A flag, not an inference from a zero: an export with no charge column measures no
            // explained total at all, and a reader that read that back as 0.0 would plot a peptide
            // set that accounts for nothing rather than one that was never asked.
            ParquetWideWriter.Bools("has_explained", rows.Select(r => r.HasExplained).ToArray()),
            ParquetWideWriter.Doubles("rt_start_min", rows.Select(r => r.RtStartMin).ToArray()),
            ParquetWideWriter.Doubles("rt_stop_min", rows.Select(r => r.RtStopMin).ToArray()),
            ParquetWideWriter.Longs("claims", rows.Select(r => (long)r.Claims).ToArray()),
            ParquetWideWriter.Longs(
                "scans_outside_scheme", rows.Select(r => (long)r.ScansOutsideScheme).ToArray()),
            ParquetWideWriter.Longs(
                "missing_injection_time",
                rows.Select(r => (long)r.SpectraMissingInjectionTime).ToArray()),
            ParquetWideWriter.Longs("cycle_count", rows.Select(r => (long)r.CycleCount).ToArray()),
            // Repeated per row: parquet dictionary-encodes them to nothing and it makes the file
            // self-describing to anything that opens it.
            ParquetWideWriter.Strings("product_tolerance", Repeat(result.ProductTolerance, n)),
            ParquetWideWriter.Strings("precursor_tolerance", Repeat(result.PrecursorTolerance, n)),
            ParquetWideWriter.Strings("isolation_scheme", Repeat(result.IsolationScheme, n)),
            ParquetWideWriter.Longs("assigned_peptides", Repeat((long)result.AssignedPeptides, n)),
            ParquetWideWriter.Strings("settings_key", Repeat(result.SettingsKey, n)),
            ParquetWideWriter.Longs("lists_matchable", Repeat(result.ListsMatchable ? 1L : 0L, n)),
        };

        ParquetWideWriter.Write(
            Path.Combine(outputDir, FileName), meta,
            Array.Empty<string>(), Array.Empty<double[]>(), n);

        WriteCycles(outputDir, result.Cycles);
        WriteLists(outputDir, result);
    }

    private static void WriteCycles(string outputDir, IReadOnlyList<IonCycleRow> cycles)
    {
        var path = Path.Combine(outputDir, CyclesFile);
        if (cycles.Count == 0)
        {
            // A re-run that read no file must not leave the previous run's traces behind.
            if (File.Exists(path))
                File.Delete(path);
            return;
        }

        var meta = new List<ParquetWideWriter.MetaColumn>
        {
            ParquetWideWriter.Strings("sample", cycles.Select(c => c.Sample).ToArray()),
            ParquetWideWriter.Longs("cycle", cycles.Select(c => (long)c.Cycle).ToArray()),
            ParquetWideWriter.Doubles("rt_start_min", cycles.Select(c => c.RtStartMin).ToArray()),
            ParquetWideWriter.Doubles("rt_stop_min", cycles.Select(c => c.RtStopMin).ToArray()),
            ParquetWideWriter.Longs("ms1_count", cycles.Select(c => (long)c.Ms1Count).ToArray()),
            ParquetWideWriter.Longs("ms2_count", cycles.Select(c => (long)c.Ms2Count).ToArray()),
            ParquetWideWriter.Doubles("ms1_acquired", cycles.Select(c => c.Ms1Acquired).ToArray()),
            ParquetWideWriter.Doubles("ms2_acquired", cycles.Select(c => c.Ms2Acquired).ToArray()),
            ParquetWideWriter.Doubles("ms1_assigned", cycles.Select(c => c.Ms1Assigned).ToArray()),
            ParquetWideWriter.Doubles("ms2_assigned", cycles.Select(c => c.Ms2Assigned).ToArray()),
            ParquetWideWriter.Doubles("ms2_explained", cycles.Select(c => c.Ms2Explained).ToArray()),
        };
        ParquetWideWriter.Write(
            path, meta, Array.Empty<string>(), Array.Empty<double[]>(), cycles.Count);
    }

    private static void WriteLists(string outputDir, IonAccountingResult result)
    {
        var path = Path.Combine(outputDir, ListsFile);
        if (result.ListNames.Count == 0)
        {
            if (File.Exists(path))
                File.Delete(path);
            return;
        }

        var samples = new List<string>();
        var names = new List<string>();
        var ms1 = new List<double>();
        var ms2 = new List<double>();
        foreach (var row in result.Rows)
        {
            for (var l = 0; l < result.ListNames.Count; l++)
            {
                samples.Add(row.Sample);
                names.Add(result.ListNames[l]);
                ms1.Add(l < row.Ms1ByList.Count ? row.Ms1ByList[l] : double.NaN);
                ms2.Add(l < row.Ms2ByList.Count ? row.Ms2ByList[l] : double.NaN);
            }
        }

        var meta = new List<ParquetWideWriter.MetaColumn>
        {
            ParquetWideWriter.Strings("sample", samples.ToArray()),
            ParquetWideWriter.Strings("list", names.ToArray()),
            ParquetWideWriter.Doubles("ms1_assigned", ms1.ToArray()),
            ParquetWideWriter.Doubles("ms2_assigned", ms2.ToArray()),
        };
        ParquetWideWriter.Write(
            path, meta, Array.Empty<string>(), Array.Empty<double[]>(), samples.Count);
    }

    /// <summary>
    /// Read the cache, or null when it is absent or unreadable. Never throws: a corrupt cache is a
    /// reason to recompute, not to fail a report.
    /// </summary>
    public static IonAccountingResult? Read(string outputDir)
    {
        var path = Path.Combine(outputDir, FileName);
        if (!File.Exists(path))
            return null;

        try
        {
            using var reader = ParquetColumnReader.Open(path);
            var samples = reader.ReadStrings("sample");
            if (samples.Length == 0)
                return null;

            var types = Strings(reader, "sample_type", samples.Length);
            var files = Strings(reader, "data_file", samples.Length);
            var statuses = Strings(reader, "status", samples.Length);
            var readers = Strings(reader, "reader", samples.Length);
            var ms1Count = Nums(reader, "ms1_count", samples.Length);
            var ms2Count = Nums(reader, "ms2_count", samples.Length);
            var ms1Acq = Nums(reader, "ms1_acquired", samples.Length);
            var ms2Acq = Nums(reader, "ms2_acquired", samples.Length);
            var ms1Asg = Nums(reader, "ms1_assigned", samples.Length);
            var ms2Asg = Nums(reader, "ms2_assigned", samples.Length);

            // Optional: a cache written before the explained total existed has neither column. The
            // settings key would refuse to REUSE such a file anyway, but it is still DISPLAYED, so
            // reading it must not throw.
            var ms2Exp = reader.HasColumn("ms2_explained")
                ? Nums(reader, "ms2_explained", samples.Length)
                : new double[samples.Length];
            var hasExp = reader.HasColumn("has_explained")
                ? reader.ReadDoubles("has_explained").Select(v => v != 0).ToArray()
                : new bool[samples.Length];
            var rt0 = Nums(reader, "rt_start_min", samples.Length);
            var rt1 = Nums(reader, "rt_stop_min", samples.Length);
            var claims = Nums(reader, "claims", samples.Length);
            var outside = Nums(reader, "scans_outside_scheme", samples.Length);
            var noInj = Nums(reader, "missing_injection_time", samples.Length);
            var cycleCount = Nums(reader, "cycle_count", samples.Length);

            var lists = ReadLists(outputDir, out var listNames);

            // Sample type is NOT in the settings key - it cannot change a number - so a cache
            // measured before the types were read would keep blank ones forever, and the only way
            // to color a bar would be to re-read every instrument file. Filled in here instead,
            // from the metadata every run writes. A type already in the file wins.
            var metadataTypes = SampleTypes(outputDir);

            var rows = new List<IonAccountingRow>(samples.Length);
            for (var i = 0; i < samples.Length; i++)
            {
                var perList = lists.GetValueOrDefault(samples[i]);
                var type = string.IsNullOrWhiteSpace(types[i])
                    ? metadataTypes.GetValueOrDefault(samples[i], "")
                    : types[i];
                rows.Add(new IonAccountingRow(
                    samples[i], type, files[i],
                    Enum.TryParse<Ms2ReadStatus>(statuses[i], out var status)
                        ? status
                        : Ms2ReadStatus.Failed,
                    readers[i],
                    (int)ms1Count[i], (int)ms2Count[i],
                    ms1Acq[i], ms2Acq[i], ms1Asg[i], ms2Asg[i],
                    ms2Exp[i], hasExp[i],
                    rt0[i], rt1[i],
                    (int)claims[i], (int)outside[i], (int)noInj[i], (int)cycleCount[i],
                    perList?.Ms1 ?? Array.Empty<double>(),
                    perList?.Ms2 ?? Array.Empty<double>()));
            }

            return new IonAccountingResult(
                First(reader, "settings_key"),
                First(reader, "product_tolerance"),
                First(reader, "precursor_tolerance"),
                First(reader, "isolation_scheme"),
                listNames,
                (int)FirstNum(reader, "assigned_peptides"),
                FirstNum(reader, "lists_matchable") != 0,
                rows,
                Array.Empty<IonCycleRow>());
        }
        catch (Exception)
        {
            return null;
        }
    }

    /// <summary>
    /// One replicate's cycle traces, read on demand. Separate from <see cref="Read"/> because the
    /// summary is small and this is not: a cohort's cycles run to hundreds of thousands of rows, and
    /// the time plots only ever show one replicate at a time.
    /// </summary>
    public static IReadOnlyList<IonCycleRow> ReadCycles(string outputDir, string? sample = null)
    {
        var path = Path.Combine(outputDir, CyclesFile);
        if (!File.Exists(path))
            return Array.Empty<IonCycleRow>();

        try
        {
            using var reader = ParquetColumnReader.Open(path);
            var samples = reader.ReadStrings("sample");
            var cycle = reader.ReadDoubles("cycle");
            var rt0 = reader.ReadDoubles("rt_start_min");
            var rt1 = reader.ReadDoubles("rt_stop_min");
            var ms1c = reader.ReadDoubles("ms1_count");
            var ms2c = reader.ReadDoubles("ms2_count");
            var ms1a = reader.ReadDoubles("ms1_acquired");
            var ms2a = reader.ReadDoubles("ms2_acquired");
            var ms1s = reader.ReadDoubles("ms1_assigned");
            var ms2s = reader.ReadDoubles("ms2_assigned");
            var ms2e = reader.HasColumn("ms2_explained")
                ? reader.ReadDoubles("ms2_explained")
                : new double[samples.Length];

            var rows = new List<IonCycleRow>();
            for (var i = 0; i < samples.Length; i++)
            {
                if (sample is not null && !string.Equals(samples[i], sample, StringComparison.Ordinal))
                    continue;
                rows.Add(new IonCycleRow(
                    samples[i], (int)cycle[i], rt0[i], rt1[i], (int)ms1c[i], (int)ms2c[i],
                    ms1a[i], ms2a[i], ms1s[i], ms2s[i], ms2e[i]));
            }
            return rows;
        }
        catch (Exception)
        {
            return Array.Empty<IonCycleRow>();
        }
    }

    /// <summary>Which replicates have cycle traces cached, for a GUI replicate picker.</summary>
    public static IReadOnlyList<string> SamplesWithCycles(string outputDir)
    {
        var path = Path.Combine(outputDir, CyclesFile);
        if (!File.Exists(path))
            return Array.Empty<string>();
        try
        {
            using var reader = ParquetColumnReader.Open(path);
            return reader.ReadStrings("sample").Distinct(StringComparer.Ordinal).ToArray();
        }
        catch (Exception)
        {
            return Array.Empty<string>();
        }
    }

    /// <summary>
    /// Sample type per sample id, from <c>sample_metadata.csv</c>, or empty when it has none.
    /// </summary>
    /// <remarks>
    /// Keyed on <c>sample_id</c>, not <c>sample</c>: the file carries both, and they are different
    /// things - <c>sample_id</c> is the merged table's <c>replicate__@__batch</c> key, which is what
    /// every other table here is keyed on, while <c>sample</c> is the bare replicate name.
    /// </remarks>
    internal static IReadOnlyDictionary<string, string> SampleTypes(string outputDir)
    {
        var types = new Dictionary<string, string>(StringComparer.Ordinal);
        var path = Path.Combine(outputDir, "sample_metadata.csv");
        if (!File.Exists(path))
            return types;

        try
        {
            var lines = File.ReadAllLines(path);
            if (lines.Length < 2)
                return types;

            var header = lines[0].Split(',');
            var idColumn = Array.FindIndex(
                header, h => h.Trim().Equals("sample_id", StringComparison.OrdinalIgnoreCase));
            var typeColumn = Array.FindIndex(
                header, h => h.Trim().Equals("sample_type", StringComparison.OrdinalIgnoreCase));
            if (idColumn < 0 || typeColumn < 0)
                return types;

            foreach (var line in lines.Skip(1))
            {
                var parts = line.Split(',');
                if (idColumn < parts.Length && typeColumn < parts.Length)
                    types[parts[idColumn].Trim()] = parts[typeColumn].Trim();
            }
        }
        catch (IOException)
        {
            // A label is not worth failing a report over.
        }
        return types;
    }

    private sealed record PerList(double[] Ms1, double[] Ms2);

    private static Dictionary<string, PerList> ReadLists(
        string outputDir, out IReadOnlyList<string> listNames)
    {
        listNames = Array.Empty<string>();
        var path = Path.Combine(outputDir, ListsFile);
        if (!File.Exists(path))
            return new Dictionary<string, PerList>(StringComparer.Ordinal);

        try
        {
            using var reader = ParquetColumnReader.Open(path);
            var samples = reader.ReadStrings("sample");
            var names = reader.ReadStrings("list");
            var ms1 = reader.ReadDoubles("ms1_assigned");
            var ms2 = reader.ReadDoubles("ms2_assigned");

            // The name order is the BIT order, taken from first appearance rather than sorted: the
            // masks were built against it and a sorted order would re-label every list's total.
            var order = new List<string>();
            var index = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var name in names)
            {
                if (index.ContainsKey(name))
                    continue;
                index[name] = order.Count;
                order.Add(name);
            }

            var byCycleSample = new Dictionary<string, PerList>(StringComparer.Ordinal);
            for (var i = 0; i < samples.Length; i++)
            {
                if (!byCycleSample.TryGetValue(samples[i], out var entry))
                {
                    entry = new PerList(new double[order.Count], new double[order.Count]);
                    byCycleSample[samples[i]] = entry;
                }
                var l = index[names[i]];
                entry.Ms1[l] = ms1[i];
                entry.Ms2[l] = ms2[i];
            }

            listNames = order;
            return byCycleSample;
        }
        catch (Exception)
        {
            return new Dictionary<string, PerList>(StringComparer.Ordinal);
        }
    }

    private static string[] Strings(ParquetColumnReader reader, string name, int count) =>
        reader.ColumnNames.Contains(name, StringComparer.Ordinal)
            ? reader.ReadStrings(name)
            : Enumerable.Repeat("", count).ToArray();

    private static double[] Nums(ParquetColumnReader reader, string name, int count) =>
        reader.ColumnNames.Contains(name, StringComparer.Ordinal)
            ? reader.ReadDoubles(name)
            : new double[count];

    private static string First(ParquetColumnReader reader, string name)
    {
        var values = reader.ColumnNames.Contains(name, StringComparer.Ordinal)
            ? reader.ReadStrings(name)
            : Array.Empty<string>();
        return values.Length > 0 ? values[0] : "";
    }

    private static double FirstNum(ParquetColumnReader reader, string name)
    {
        var values = reader.ColumnNames.Contains(name, StringComparer.Ordinal)
            ? reader.ReadDoubles(name)
            : Array.Empty<double>();
        return values.Length > 0 ? values[0] : 0;
    }

    private static T[] Repeat<T>(T value, int count) => Enumerable.Repeat(value, count).ToArray();

    /// <summary>How a fraction reads in a caption, or "n/a" when there is none.</summary>
    public static string Percent(double fraction) =>
        double.IsFinite(fraction)
            ? fraction.ToString("P1", CultureInfo.InvariantCulture)
            : "n/a";
}
