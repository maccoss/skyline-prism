using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SkylinePrism.Core.IO;

/// <summary>
/// A per-sample annotation table read from a CSV: sample id to (column to value), plus the column
/// names in file order.
///
/// <para>This exists so the GUI's grouping dropdowns are not limited to the Skyline Replicates
/// report. That report is written only when PRISM exported it from a live or closed Skyline
/// document, so an output directory produced by the CLI has none - and the QC pane could then
/// colour a plot by nothing but sample type, while the run's own <c>sample_metadata.csv</c> was
/// sitting beside the matrices carrying at least the batch. Reading that file costs nothing and
/// makes <c>batch</c> a grouping the QC plots can use, which for a batch-correction tool is the
/// grouping most worth looking at.</para>
///
/// <para>Deliberately dumb: everything is a string, nothing is type-inferred, and no column is
/// special. The caller decides which column is the key and which columns are worth offering.</para>
/// </summary>
public sealed class SampleAnnotationTable
{
    /// <summary>Nothing read: no columns, and every lookup is "".</summary>
    public static readonly SampleAnnotationTable Empty =
        new(new Dictionary<string, Dictionary<string, string>>(StringComparer.Ordinal),
            Array.Empty<string>());

    private readonly Dictionary<string, Dictionary<string, string>> _bySample;

    private SampleAnnotationTable(
        Dictionary<string, Dictionary<string, string>> bySample, IReadOnlyList<string> columns)
    {
        _bySample = bySample;
        Columns = columns;
    }

    /// <summary>The annotation columns, in file order, excluding the key column.</summary>
    public IReadOnlyList<string> Columns { get; }

    /// <summary>Whether no row was read.</summary>
    public bool IsEmpty => _bySample.Count == 0;

    /// <summary>Sample ids present, in file order.</summary>
    public IReadOnlyCollection<string> Samples => _bySample.Keys;

    /// <summary>
    /// The value of <paramref name="column"/> for <paramref name="sampleId"/>, or "" when either is
    /// absent. Never throws - a grouping dropdown must degrade to "(none)" rather than take the
    /// pane down.
    /// </summary>
    public string ValueOf(string sampleId, string column)
        => _bySample.TryGetValue(sampleId, out var row) ? row.GetValueOrDefault(column, "") : "";

    /// <summary>
    /// Read <paramref name="csvPath"/>, keying rows on <paramref name="keyColumn"/> and dropping
    /// <paramref name="excludeColumns"/> from <see cref="Columns"/> (their values are still read,
    /// so a caller that knows a column by another name can still ask for it).
    ///
    /// <para>Returns <see cref="Empty"/> rather than throwing when the file is missing, unreadable,
    /// empty, or has no such key column. A pane opens on whatever the directory holds; a malformed
    /// annotation file must cost the grouping dropdown, not the plot.</para>
    /// </summary>
    public static SampleAnnotationTable Read(
        string csvPath, string keyColumn, params string[] excludeColumns)
    {
        string[] lines;
        try
        {
            if (!File.Exists(csvPath))
                return Empty;
            lines = SharedFile.ReadAllLines(csvPath);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            return Empty;
        }

        if (lines.Length < 2)
            return Empty;

        var header = SplitCsvLine(lines[0]);
        var keyIdx = Array.FindIndex(header, h => Matches(h, keyColumn));
        if (keyIdx < 0)
            return Empty;

        var excluded = new HashSet<string>(excludeColumns, StringComparer.OrdinalIgnoreCase);
        var columns = header
            .Where((h, i) => i != keyIdx && !excluded.Contains(h) && h.Length > 0)
            .ToList();

        var bySample = new Dictionary<string, Dictionary<string, string>>(StringComparer.Ordinal);
        for (var r = 1; r < lines.Length; r++)
        {
            if (lines[r].Length == 0)
                continue;
            var cells = SplitCsvLine(lines[r]);
            if (keyIdx >= cells.Length)
                continue;
            var key = cells[keyIdx];
            if (key.Length == 0)
                continue;

            var row = new Dictionary<string, string>(StringComparer.Ordinal);
            for (var c = 0; c < header.Length && c < cells.Length; c++)
                if (c != keyIdx)
                    row[header[c]] = cells[c];

            // Last row wins on a duplicate key, matching how every other CSV reader here behaves.
            bySample[key] = row;
        }

        return bySample.Count == 0 ? Empty : new SampleAnnotationTable(bySample, columns);
    }

    /// <summary>
    /// Build a table directly from values already in memory - the case that matters is an external
    /// clinical CSV that has already been joined to the samples, where re-reading the file would
    /// mean redoing the key-column detection and could disagree with the join the user is looking
    /// at.
    /// </summary>
    public static SampleAnnotationTable FromValues(
        IReadOnlyList<string> sampleIds,
        IReadOnlyList<string> columns,
        Func<string, string, string?> valueOf)
    {
        var bySample = new Dictionary<string, Dictionary<string, string>>(StringComparer.Ordinal);
        foreach (var id in sampleIds)
        {
            var row = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var c in columns)
                row[c] = valueOf(id, c) ?? "";
            bySample[id] = row;
        }

        return bySample.Count == 0
            ? Empty
            : new SampleAnnotationTable(bySample, columns.ToList());
    }

    /// <summary>
    /// This table with <paramref name="other"/>'s rows and columns folded in. <b>This table wins</b>
    /// on a column name present in both, so a richer source layered under a more authoritative one
    /// adds to it rather than overwriting it.
    /// </summary>
    public SampleAnnotationTable MergedWith(SampleAnnotationTable other)
    {
        if (other.IsEmpty)
            return this;
        if (IsEmpty)
            return other;

        var mine = new HashSet<string>(Columns, StringComparer.Ordinal);
        var columns = Columns.Concat(other.Columns.Where(c => !mine.Contains(c))).ToList();

        var merged = new Dictionary<string, Dictionary<string, string>>(StringComparer.Ordinal);
        foreach (var id in _bySample.Keys.Concat(other._bySample.Keys).Distinct(StringComparer.Ordinal))
        {
            var row = new Dictionary<string, string>(StringComparer.Ordinal);
            if (other._bySample.TryGetValue(id, out var theirs))
                foreach (var kv in theirs)
                    row[kv.Key] = kv.Value;
            if (_bySample.TryGetValue(id, out var ours))
                foreach (var kv in ours)
                    row[kv.Key] = kv.Value; // ours last: this table wins
            merged[id] = row;
        }

        return new SampleAnnotationTable(merged, columns);
    }

    /// <summary>Case- and separator-insensitive header match, the convention used throughout PRISM.</summary>
    private static bool Matches(string header, string wanted)
        => Normalize(header) == Normalize(wanted);

    private static string Normalize(string s)
        => s.Replace(" ", "").Replace("_", "").ToLowerInvariant();

    /// <summary>
    /// Minimal RFC 4180 split: commas separate, double quotes group, and a doubled quote inside a
    /// quoted field is one quote. Enough for the files PRISM writes and for a hand-made clinical
    /// table, and it will not mangle a quoted value containing a comma - which a naive Split(',')
    /// silently would.
    /// </summary>
    private static string[] SplitCsvLine(string line)
    {
        var cells = new List<string>();
        var sb = new System.Text.StringBuilder();
        var inQuotes = false;
        for (var i = 0; i < line.Length; i++)
        {
            var ch = line[i];
            if (inQuotes)
            {
                if (ch == '"')
                {
                    if (i + 1 < line.Length && line[i + 1] == '"')
                    {
                        sb.Append('"');
                        i++;
                    }
                    else
                    {
                        inQuotes = false;
                    }
                }
                else
                {
                    sb.Append(ch);
                }
            }
            else if (ch == '"')
            {
                inQuotes = true;
            }
            else if (ch == ',')
            {
                cells.Add(sb.ToString().Trim());
                sb.Clear();
            }
            else
            {
                sb.Append(ch);
            }
        }

        cells.Add(sb.ToString().Trim());
        return cells.ToArray();
    }
}
