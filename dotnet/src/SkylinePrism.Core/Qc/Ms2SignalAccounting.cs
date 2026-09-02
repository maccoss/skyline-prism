using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using SkylinePrism.Core.IO;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// MS2 signal accounting for a whole cohort: per replicate, how much of the signal Skyline integrated
/// the run assigns to a peptide, and how much of that belongs to peptides in selected protein lists.
///
/// <para><b>Counted once.</b> Every total is a union measure (<see cref="Ms2SignalUnion"/>), not a sum:
/// a DIA isolation window co-isolates tens of peptides and two fragments whose extraction windows
/// overlap are the same detector counts. The naive sum is kept alongside so a report can say how much
/// double counting was removed, but it is never plotted as "assigned".</para>
///
/// <para><b>Raw values.</b> Magnitudes come from <c>merged_data/</c> - linear Skyline peak areas,
/// untouched by normalization, ComBat or marker adjustment. The pipeline outputs supply peptide
/// identity only (<see cref="Ms2SignalPeptides"/>).</para>
///
/// <para><b>What is NOT here.</b> The acquired MS2 total - the denominator that would turn these into a
/// fraction of what the instrument measured - needs the instrument files and is not available from any
/// Skyline export. Until a reader supplies it (<see cref="RawData.IMs2SignalReader"/>), the largest bar
/// is "signal Skyline integrated for this document's targets" and must be labelled as such. Calling it
/// "total MS2" would make an unknown coverage read as complete coverage, which is the single largest
/// misreading risk in this feature.</para>
/// </summary>
public static class Ms2SignalAccounting
{
    /// <summary>Per-replicate results, one row per replicate.</summary>
    public const string AccountingFile = "ms2_signal_accounting.parquet";

    /// <summary>Per-replicate, per-list results in long format. Absent when no list was selected.</summary>
    public const string ListsFile = "ms2_signal_lists.parquet";

    /// <param name="AssignedArea">Union measure over the transitions of peptides that reached the
    /// peptide matrix. LINEAR.</param>
    /// <param name="SummedArea">The naive sum over the same transitions, for the double-counting
    /// figure. Never plotted as assigned signal.</param>
    /// <param name="DuplicateArea">Of the area the union removed, how much came from one peptide
    /// exported several times; <paramref name="SharedArea"/> is the rest, from genuinely co-isolated
    /// peptides. They sum to <c>SummedArea - AssignedArea</c>. Both are reported because the ROW counts
    /// mislead: on a real plasma cohort 5.6% of rows merge away but 21% of the area does, and almost
    /// all of that area is the duplicate kind.</param>
    /// <param name="ListArea">Union measure per selected list, aligned with
    /// <see cref="Result.ListNames"/>. Each nests inside <see cref="AssignedArea"/> and lists may
    /// overlap each other.</param>
    public sealed record Row(
        string Sample,
        string SampleType,
        double AssignedArea,
        double SummedArea,
        IReadOnlyList<double> ListArea,
        int Regions,
        int MergedGroups,
        int LargestGroup,
        int DuplicateRows,
        int SharedAcrossPeptides,
        int OutsideScheme,
        int UnknownPeptides,
        int Skipped,
        double DuplicateArea,
        double SharedArea)
    {
        /// <summary>How much of the naive sum was double counting, as a fraction. NaN with no signal.</summary>
        public double DoubleCountedFraction =>
            SummedArea > 0 ? 1 - AssignedArea / SummedArea : double.NaN;
    }

    /// <param name="ListNames">Selected lists, aligned with every row's <see cref="Row.ListArea"/>.</param>
    /// <param name="ListColors">Each list's own colour, carried rather than looked up by name so a
    /// user-defined list keeps its colour and the plot never has to guess one.</param>
    /// <param name="Tolerance">The extraction window used, as
    /// <see cref="ProductMassTolerance.Describe"/> puts it - so a plot can name it.</param>
    /// <param name="ListsMatchable">False when the peptide output carries no protein-group columns, in
    /// which case every list bar is zero for want of data rather than for want of members.</param>
    /// <param name="Measure">Which quantity was totalled. Carried because it is not recoverable from
    /// the numbers, and the two are not interchangeable - a plot has to say which it is showing.</param>
    public sealed record Result(
        IReadOnlyList<Row> Rows,
        IReadOnlyList<string> ListNames,
        IReadOnlyList<string> ListColors,
        IReadOnlyList<int> PerListPeptides,
        int AssignedPeptides,
        string Tolerance,
        string IsolationScheme,
        bool ListsMatchable,
        Ms2SignalMeasure Measure = Ms2SignalMeasure.Signal)
    {
        public bool IsEmpty => Rows.Count == 0;
    }

    /// <summary>
    /// Compute the accounting for every replicate in <paramref name="outputDir"/>, or null when
    /// <c>merged_data/</c> is absent (a directory whose intermediates were cleaned up) or the export
    /// lacked a column the accounting needs.
    ///
    /// <para><b>Cost.</b> One streaming pass over the whole merged table plus an ORDER BY the sample id,
    /// which DuckDB spills to disk. That is comparable to Stage 2, so it is not something to run on
    /// every report - <see cref="ReadCached"/> exists for that.</para>
    /// </summary>
    public static Result? Compute(
        string outputDir, IsolationScheme scheme, ProductMassTolerance tolerance,
        IReadOnlyList<ProteinList> lists, IReadOnlyDictionary<string, string>? sampleTypes = null,
        Action<string>? log = null, int memoryBudgetMb = 0,
        Ms2SignalMeasure measure = Ms2SignalMeasure.Signal)
    {
        var mergedRoot = Path.Combine(outputDir, "merged_data");
        if (!MergedDataset.Exists(mergedRoot))
        {
            log?.Invoke("  MS2 signal accounting skipped: no merged_data/ in the output directory.");
            return null;
        }

        var dataset = MergedDataset.Open(mergedRoot);
        var columnNames = ParquetTable.ReadColumnNames(dataset.RepresentativeFile()).ToList();
        var cols = Ms2SignalRegions.Resolve(columnNames);
        if (cols is null)
        {
            // Product m/z and the integration bounds are what place a fragment in signal space; without
            // them there is no accounting to do, and saying so beats a plot of nothing.
            log?.Invoke(
                "  MS2 signal accounting skipped: the merged table lacks Product Mz, Start Time or "
                + "End Time. Re-export the report with those columns to enable it.");
            return null;
        }

        var classified = Ms2SignalPeptides.Classify(outputDir, lists);
        if (lists.Count > 0 && !classified.HasGroupColumns)
        {
            log?.Invoke(
                "  MS2 signal accounting: corrected_peptides.parquet has no protein-group columns, so "
                + "no protein list could be matched. Only the assigned total is meaningful.");
        }

        // Ions asked for but absent is worth saying plainly rather than silently answering a
        // different question with the same-looking number.
        if (measure == Ms2SignalMeasure.Ions && !cols.HasIonCounts)
        {
            log?.Invoke(
                "  MS2 signal accounting: measure 'ions' was requested but this export has no "
                + "LC Peak ion-count column, so peak areas are being summed instead. Re-export with "
                + "the ion-count columns, which need a Skyline that computes them and a document "
                + "whose spectrum metadata carries injection times.");
            measure = Ms2SignalMeasure.Signal;
        }
        else if (measure == Ms2SignalMeasure.Ions)
        {
            log?.Invoke(
                "  MS2 signal accounting: summing Skyline ion counts (intensity x injection time per "
                + "spectrum). Neither side is background subtracted and both are counts, so no unit "
                + "or background correction applies.");
        }

        // Which signal was summed changes what the fraction MEANS, so it is stated every run.
        if (measure == Ms2SignalMeasure.Signal)
        log?.Invoke(cols.Background is null
            ? "  MS2 signal accounting: this export has no Background column, so the assigned signal "
              + "is background-SUBTRACTED while an acquired total ion current is not. The assigned "
              + "fraction is therefore an under-estimate. Re-export to include Background."
            : "  MS2 signal accounting: summing Area + Background, so the assigned signal is gross "
              + "and comparable with an acquired total ion current.");
        log?.Invoke(
            $"  MS2 signal accounting: {scheme.Windows.Count} isolation windows, "
            + $"extraction {tolerance.Describe()}, {classified.AssignedPeptides:N0} assigned peptides"
            + (lists.Count > 0 ? $", {lists.Count} protein list(s)." : "."));

        var rows = new List<Row>();
        Ms2SignalRegions.LoadBySample(
            dataset, cols, scheme, classified.Classes,
            (sample, loaded) =>
            {
                var union = Ms2SignalUnion.Compute(loaded.Regions, tolerance, lists.Count);
                rows.Add(new Row(
                    sample,
                    sampleTypes?.GetValueOrDefault(sample, "unknown") ?? "unknown",
                    union.AssignedArea,
                    union.SummedArea,
                    union.ListArea,
                    union.Regions,
                    union.MergedGroups,
                    union.LargestGroup,
                    union.DuplicateRows,
                    union.SharedAcrossPeptides,
                    loaded.OutsideScheme,
                    loaded.UnknownPeptides,
                    union.Skipped,
                    union.DuplicateArea,
                    union.SharedArea));
            },
            memoryBudgetMb,
            measure);

        if (rows.Count == 0)
        {
            log?.Invoke("  MS2 signal accounting: the merged table has no fragment rows.");
            return null;
        }

        var result = new Result(
            rows, classified.ListNames, lists.Select(l => l.ColorHex).ToList(),
            classified.PerListPeptides, classified.AssignedPeptides,
            tolerance.Describe(), scheme.Name, classified.HasGroupColumns || lists.Count == 0,
            measure);

        var removed = Median(rows.Select(r => r.DoubleCountedFraction));
        log?.Invoke(
            $"  MS2 signal accounting: {rows.Count:N0} replicate(s); counting shared signal once "
            + $"removed a median {Percent(removed)} of the naive sum.");
        return result;
    }

    /// <summary>Persist the results so a later <c>prism qc -d</c> can plot them without recomputing.</summary>
    public static void Write(string outputDir, Result result)
    {
        var n = result.Rows.Count;
        var meta = new List<ParquetWideWriter.MetaColumn>
        {
            ParquetWideWriter.Strings("sample", result.Rows.Select(r => r.Sample).ToArray()),
            ParquetWideWriter.Strings("sample_type", result.Rows.Select(r => r.SampleType).ToArray()),
            ParquetWideWriter.Doubles("assigned_area", result.Rows.Select(r => r.AssignedArea).ToArray()),
            ParquetWideWriter.Doubles("summed_area", result.Rows.Select(r => r.SummedArea).ToArray()),
            ParquetWideWriter.Longs("regions", result.Rows.Select(r => (long)r.Regions).ToArray()),
            ParquetWideWriter.Longs("merged_groups", result.Rows.Select(r => (long)r.MergedGroups).ToArray()),
            ParquetWideWriter.Longs("largest_group", result.Rows.Select(r => (long)r.LargestGroup).ToArray()),
            ParquetWideWriter.Longs("duplicate_rows", result.Rows.Select(r => (long)r.DuplicateRows).ToArray()),
            ParquetWideWriter.Longs(
                "shared_across_peptides", result.Rows.Select(r => (long)r.SharedAcrossPeptides).ToArray()),
            ParquetWideWriter.Longs("outside_scheme", result.Rows.Select(r => (long)r.OutsideScheme).ToArray()),
            ParquetWideWriter.Longs(
                "unknown_peptides", result.Rows.Select(r => (long)r.UnknownPeptides).ToArray()),
            ParquetWideWriter.Longs("skipped", result.Rows.Select(r => (long)r.Skipped).ToArray()),
            ParquetWideWriter.Doubles(
                "duplicate_area", result.Rows.Select(r => r.DuplicateArea).ToArray()),
            ParquetWideWriter.Doubles("shared_area", result.Rows.Select(r => r.SharedArea).ToArray()),
            // Repeated per row rather than kept in a sidecar: parquet dictionary-encodes them to nothing,
            // and it makes the file self-describing to anything that opens it.
            ParquetWideWriter.Strings("tolerance", Repeat(result.Tolerance, n)),
            ParquetWideWriter.Strings("isolation_scheme", Repeat(result.IsolationScheme, n)),
            ParquetWideWriter.Longs("assigned_peptides", Repeat((long)result.AssignedPeptides, n)),
        };

        ParquetWideWriter.Write(
            Path.Combine(outputDir, AccountingFile), meta,
            Array.Empty<string>(), Array.Empty<double[]>(), n);

        var listsPath = Path.Combine(outputDir, ListsFile);
        if (result.ListNames.Count == 0)
        {
            // A re-run with the lists removed must not leave the previous run's bars behind.
            if (File.Exists(listsPath))
                File.Delete(listsPath);
            return;
        }

        // Long format: one row per (replicate, list). A column per list would make the schema depend on
        // the user's list names, which a reader then cannot tell apart from the fixed columns.
        var samples = new List<string>();
        var names = new List<string>();
        var colors = new List<string>();
        var areas = new List<double>();
        var peptides = new List<long>();
        foreach (var row in result.Rows)
        {
            for (var l = 0; l < result.ListNames.Count; l++)
            {
                samples.Add(row.Sample);
                names.Add(result.ListNames[l]);
                colors.Add(l < result.ListColors.Count ? result.ListColors[l] : "");
                areas.Add(l < row.ListArea.Count ? row.ListArea[l] : double.NaN);
                peptides.Add(l < result.PerListPeptides.Count ? result.PerListPeptides[l] : 0);
            }
        }

        ParquetWideWriter.Write(
            listsPath,
            new List<ParquetWideWriter.MetaColumn>
            {
                ParquetWideWriter.Strings("sample", samples.ToArray()),
                ParquetWideWriter.Strings("list_name", names.ToArray()),
                ParquetWideWriter.Strings("color", colors.ToArray()),
                ParquetWideWriter.Doubles("area", areas.ToArray()),
                ParquetWideWriter.Longs("peptides", peptides.ToArray()),
            },
            Array.Empty<string>(), Array.Empty<double[]>(), samples.Count);
    }

    /// <summary>
    /// Read back what <see cref="Write"/> persisted, or null when it is not there. This is the path
    /// <c>prism qc -d</c> takes: the report stays a pure file reader, so it works on a directory whose
    /// <c>merged_data/</c> has been cleaned up.
    /// </summary>
    public static Result? ReadCached(string outputDir)
    {
        var path = Path.Combine(outputDir, AccountingFile);
        if (!File.Exists(path))
            return null;

        using var reader = ParquetColumnReader.Open(path);
        if (reader.RowCount == 0 || !reader.HasColumn("sample"))
            return null;

        var samples = reader.ReadStrings("sample");
        var types = reader.HasColumn("sample_type") ? reader.ReadStrings("sample_type") : null;
        var assigned = reader.ReadDoubles("assigned_area");
        var summed = reader.ReadDoubles("summed_area");

        var listAreas = ReadCachedLists(
            outputDir, out var listNames, out var listColors, out var listPeptides);

        // Read each count column ONCE. Reading per row turns a 192-replicate file into 192 decodes of
        // every column, which is quadratic for no reason.
        var regions = Counts(reader, "regions");
        var mergedGroups = Counts(reader, "merged_groups");
        var largestGroup = Counts(reader, "largest_group");
        var duplicateRows = Counts(reader, "duplicate_rows");
        var sharedAcross = Counts(reader, "shared_across_peptides");
        var outsideScheme = Counts(reader, "outside_scheme");
        var unknownPeptides = Counts(reader, "unknown_peptides");
        var skipped = Counts(reader, "skipped");
        var duplicateArea = Counts(reader, "duplicate_area");
        var sharedArea = Counts(reader, "shared_area");
        var assignedPeptides = Counts(reader, "assigned_peptides");

        var rows = new List<Row>(reader.RowCount);
        for (var i = 0; i < reader.RowCount; i++)
        {
            var sample = samples[i] ?? "";
            rows.Add(new Row(
                sample,
                types is not null && i < types.Length ? types[i] ?? "unknown" : "unknown",
                assigned[i],
                summed[i],
                listAreas.GetValueOrDefault(sample) ?? EmptyAreas(listNames.Count),
                At(regions, i),
                At(mergedGroups, i),
                At(largestGroup, i),
                At(duplicateRows, i),
                At(sharedAcross, i),
                At(outsideScheme, i),
                At(unknownPeptides, i),
                At(skipped, i),
                Area(duplicateArea, i),
                Area(sharedArea, i)));
        }

        return new Result(
            rows, listNames, listColors, listPeptides,
            At(assignedPeptides, 0),
            First(reader, "tolerance") ?? "unknown",
            First(reader, "isolation_scheme") ?? "unknown",
            ListsMatchable: true);
    }

    private static Dictionary<string, double[]> ReadCachedLists(
        string outputDir, out IReadOnlyList<string> listNames, out IReadOnlyList<string> listColors,
        out IReadOnlyList<int> listPeptides)
    {
        listNames = Array.Empty<string>();
        listColors = Array.Empty<string>();
        listPeptides = Array.Empty<int>();
        var byArea = new Dictionary<string, double[]>(StringComparer.Ordinal);

        var path = Path.Combine(outputDir, ListsFile);
        if (!File.Exists(path))
            return byArea;

        using var reader = ParquetColumnReader.Open(path);
        if (reader.RowCount == 0 || !reader.HasColumn("sample") || !reader.HasColumn("list_name"))
            return byArea;

        var samples = reader.ReadStrings("sample");
        var names = reader.ReadStrings("list_name");
        var hexes = reader.HasColumn("color") ? reader.ReadStrings("color") : null;
        var areas = reader.ReadDoubles("area");
        var peptides = reader.HasColumn("peptides") ? reader.ReadDoubles("peptides") : null;

        // The write order is (replicate-major, list order within it), so first appearances recover the
        // list order without needing it stored separately.
        var order = new List<string>();
        var index = new Dictionary<string, int>(StringComparer.Ordinal);
        var counts = new List<int>();
        var hexOrder = new List<string>();
        for (var i = 0; i < names.Length; i++)
        {
            var name = names[i] ?? "";
            if (index.ContainsKey(name))
                continue;
            index[name] = order.Count;
            order.Add(name);
            counts.Add(peptides is null ? 0 : (int)peptides[i]);
            hexOrder.Add(hexes is not null && i < hexes.Length ? hexes[i] ?? "" : "");
        }

        foreach (var (sample, i) in samples.Select((s, i) => (s ?? "", i)))
        {
            if (!byArea.TryGetValue(sample, out var slot))
                byArea[sample] = slot = EmptyAreas(order.Count);
            var l = index[names[i] ?? ""];
            if (l < slot.Length)
                slot[l] = areas[i];
        }

        listNames = order;
        listColors = hexOrder;
        listPeptides = counts;
        return byArea;
    }

    private static double[] EmptyAreas(int count) => count == 0 ? Array.Empty<double>() : new double[count];

    /// <summary>A whole-number column, or null when this file predates it.</summary>
    private static double[]? Counts(ParquetColumnReader reader, string column) =>
        reader.HasColumn(column) ? reader.ReadDoubles(column) : null;

    /// <summary>An area column, or 0 for a file written before it existed.</summary>
    private static double Area(double[]? column, int row) =>
        column is not null && row < column.Length && double.IsFinite(column[row]) ? column[row] : 0;

    private static int At(double[]? column, int row) =>
        column is not null && row < column.Length && double.IsFinite(column[row])
            ? (int)column[row]
            : 0;

    private static string? First(ParquetColumnReader reader, string column) =>
        reader.HasColumn(column) && !reader.IsNumericColumn(column)
            ? reader.ReadStrings(column).FirstOrDefault()
            : null;

    private static T[] Repeat<T>(T value, int count)
    {
        var array = new T[count];
        for (var i = 0; i < count; i++)
            array[i] = value;
        return array;
    }

    private static double Median(IEnumerable<double> values)
    {
        var sorted = values.Where(double.IsFinite).OrderBy(v => v).ToArray();
        if (sorted.Length == 0)
            return double.NaN;
        var mid = sorted.Length / 2;
        return sorted.Length % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    }

    /// <summary>Culture-invariant percent for a log line, since these reach files people diff.</summary>
    internal static string Percent(double fraction) =>
        double.IsFinite(fraction)
            ? (fraction * 100).ToString("0.0", CultureInfo.InvariantCulture) + "%"
            : "n/a";
}
