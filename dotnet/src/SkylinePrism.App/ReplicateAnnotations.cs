using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Skyline;

namespace SkylinePrism.App;

/// <summary>
/// One output directory's replicate annotations, read from its exported Replicates reports
/// ("label.metadata.csv" per document, plus the legacy single-document "Metadata.csv"): sample ID to
/// (column to value), and the union of the annotation columns.
/// </summary>
/// <remarks>
/// <para>Immutable once read, and read as a whole, so a pane takes a snapshot for the directory it is
/// showing and installs it only once that directory is still the one on screen. The QC and Ion
/// accounting panes each hold their own: when they were one window-wide dictionary, opening the Ion pane
/// on directory B replaced the annotations the QC pane's still-cached matrices for A were grouped by,
/// a read finishing after the output box had moved on left a stale set installed, and two panes reading
/// at once mutated the same dictionaries from two threads.</para>
///
/// <para>Each file's rows are stored under BOTH the document-qualified sample ID
/// ("replicate__@__label", which is the merged Sample ID) and the bare replicate name, so a QC injection
/// named the same in several documents keeps its own document's annotations. The first document to claim
/// a bare name keeps it; files are read in name order so that is deterministic.</para>
/// </remarks>
internal sealed class ReplicateAnnotations
{
    /// <summary>No reports read: every lookup is "", and there are no columns.</summary>
    public static readonly ReplicateAnnotations Empty =
        new(new Dictionary<string, Dictionary<string, string>>(StringComparer.Ordinal), new List<string>());

    private const string BatchSeparator = "__@__";

    private readonly Dictionary<string, Dictionary<string, string>> _bySample;

    private ReplicateAnnotations(Dictionary<string, Dictionary<string, string>> bySample, List<string> columns)
    {
        _bySample = bySample;
        Columns = columns;
    }

    /// <summary>The annotation columns present in any of the reports, in first-seen order.</summary>
    public IReadOnlyList<string> Columns { get; }

    /// <summary>Whether no report contributed a row.</summary>
    public bool IsEmpty => _bySample.Count == 0;

    /// <summary>
    /// The value of <paramref name="column"/> for <paramref name="sampleId"/>, or "" when it has none.
    /// The document-qualified entry is authoritative when it exists: a column its own document did not
    /// export reads as "", not as another document's value for a same-named replicate. The bare
    /// replicate name is consulted only for a sample with no qualified entry at all - a single-document
    /// run, or a legacy Metadata.csv export.
    /// </summary>
    public string ValueOf(string sampleId, string column)
    {
        if (_bySample.TryGetValue(sampleId, out var qualified))
            return qualified.GetValueOrDefault(column, "");
        if (_bySample.TryGetValue(ReplicateOf(sampleId), out var bare))
            return bare.GetValueOrDefault(column, "");
        return "";
    }

    /// <summary>Sample IDs are "replicate__@__batch"; the Replicates report is keyed by replicate.</summary>
    public static string ReplicateOf(string sampleId)
    {
        var i = sampleId.IndexOf(BatchSeparator, StringComparison.Ordinal);
        return i >= 0 ? sampleId[..i] : sampleId;
    }

    /// <summary>
    /// Read every Replicates report under <paramref name="reportsDir"/> - normally the run's
    /// skyline-reports folder. <see cref="Empty"/> when the folder or the reports are missing.
    /// </summary>
    public static ReplicateAnnotations Read(string reportsDir)
    {
        if (!Directory.Exists(reportsDir))
            return Empty;

        var files = Directory.EnumerateFiles(reportsDir, "*.metadata.csv")
            .Concat(Directory.EnumerateFiles(reportsDir, "Metadata.csv"))
            // A sidecar is named after its destination, so an unfinished
            // ".prism-partial-<hex>.<label>.metadata.csv" matches the glob above. Left by a stopped run,
            // it would be read as a document's replicate metadata under the bogus label
            // ".prism-partial-<hex>.<label>" - and, because a leading dot sorts first, its rows would
            // seed every bare replicate-name key before the real files are read.
            .Where(f => !HeadlessSkylineExporter.IsSidecar(f))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
            .ToList();

        var bySample = new Dictionary<string, Dictionary<string, string>>(StringComparer.Ordinal);
        var columns = new List<string>();
        foreach (var file in files)
        {
            var name = Path.GetFileName(file);
            // "<label>.metadata.csv" -> "<label>"; the legacy "Metadata.csv" has no document label.
            var label = name.EndsWith(".metadata.csv", StringComparison.OrdinalIgnoreCase)
                ? name[..^".metadata.csv".Length]
                : null;
            ReadReport(file, label, bySample, columns);
        }
        return bySample.Count == 0 && columns.Count == 0 ? Empty : new ReplicateAnnotations(bySample, columns);
    }

    // Parse one exported Replicates report (dynamic annotation columns) into replicate -> column -> value.
    private static void ReadReport(
        string path, string? documentLabel,
        Dictionary<string, Dictionary<string, string>> bySample, List<string> columns)
    {
        if (!File.Exists(path))
            return;
        var lines = File.ReadAllLines(path);
        if (lines.Length < 2)
            return;
        var header = SplitCsvLine(lines[0]);
        var repIdx = -1;
        foreach (var cand in new[] { "Replicate", "Replicate Name", "ReplicateName", "ReplicateLocator" })
        {
            repIdx = Array.FindIndex(header, h => h.Trim().Equals(cand, StringComparison.OrdinalIgnoreCase));
            if (repIdx >= 0)
                break;
        }
        if (repIdx < 0)
            return;

        var cols = new List<(string Name, int Idx)>();
        for (var i = 0; i < header.Length; i++)
            if (i != repIdx && !string.IsNullOrWhiteSpace(header[i]))
                cols.Add((header[i].Trim(), i));
        // Union across documents: a column present in any Replicates report can be grouped by.
        foreach (var name in cols.Select(c => c.Name))
            if (!columns.Contains(name, StringComparer.Ordinal))
                columns.Add(name);

        for (var r = 1; r < lines.Length; r++)
        {
            if (string.IsNullOrWhiteSpace(lines[r]))
                continue;
            var f = SplitCsvLine(lines[r]);
            if (f.Length <= repIdx)
                continue;
            var rep = f[repIdx].Trim();
            if (rep.Length == 0)
                continue;
            var map = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var (name, idx) in cols)
                map[name] = idx < f.Length ? f[idx].Trim() : "";
            // Document-qualified key first (this is the merged Sample ID), then the bare replicate name as
            // a fallback for single-document runs and legacy Metadata.csv exports.
            if (!string.IsNullOrEmpty(documentLabel))
                bySample[rep + BatchSeparator + documentLabel] = map;
            if (!bySample.ContainsKey(rep) || string.IsNullOrEmpty(documentLabel))
                bySample[rep] = map;
        }
    }

    /// <summary>One CSV line into fields, honoring quotes and doubled quotes inside them.</summary>
    internal static string[] SplitCsvLine(string line)
    {
        var fields = new List<string>();
        var sb = new System.Text.StringBuilder();
        var inQuotes = false;
        for (var i = 0; i < line.Length; i++)
        {
            var c = line[i];
            if (inQuotes)
            {
                if (c == '"')
                {
                    if (i + 1 < line.Length && line[i + 1] == '"') { sb.Append('"'); i++; }
                    else inQuotes = false;
                }
                else sb.Append(c);
            }
            else if (c == '"') inQuotes = true;
            else if (c == ',') { fields.Add(sb.ToString()); sb.Clear(); }
            else sb.Append(c);
        }
        fields.Add(sb.ToString());
        return fields.ToArray();
    }
}
