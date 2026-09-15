using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.RawData;
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
/// <para>A labeled report's rows are stored under the document-qualified sample ID only
/// ("replicate__@__label", which is the merged Sample ID), so a QC injection named the same in several
/// documents keeps its own document's annotations, and a document that shipped no report borrows
/// nothing from one that did. Only the legacy label-less "Metadata.csv" stores bare replicate names,
/// and those answer only for a sample with no qualified entry.</para>
///
/// <para>Never throws for a file it cannot read. The panes must open on whatever else the directory
/// holds, so an unreadable report is logged and skipped and its replicates fall back to their sample
/// type; the pipeline's own reader of these files is <see cref="ReplicateMetadata"/>, which this
/// deliberately does not share, because the panes want a per-directory value and no type mapping.</para>
/// </remarks>
internal sealed class ReplicateAnnotations
{
    /// <summary>No reports read: every lookup is "", and there are no columns.</summary>
    public static readonly ReplicateAnnotations Empty =
        new(new Dictionary<string, Dictionary<string, string>>(StringComparer.Ordinal), new List<string>());

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
    /// replicate name - which only a legacy label-less Metadata.csv provides - answers only for a
    /// sample with no qualified entry at all.
    /// </summary>
    public string ValueOf(string sampleId, string column)
    {
        if (_bySample.TryGetValue(sampleId, out var qualified))
            return qualified.GetValueOrDefault(column, "");
        if (_bySample.TryGetValue(ReplicateDataFiles.ReplicateOf(sampleId), out var bare))
            return bare.GetValueOrDefault(column, "");
        return "";
    }

    /// <summary>
    /// Read every Replicates report under <paramref name="reportsDir"/> - normally the run's
    /// skyline-reports folder. <see cref="Empty"/> when the folder or the reports are missing. A file
    /// that cannot be read is reported through <paramref name="log"/> and skipped, never thrown.
    /// </summary>
    public static ReplicateAnnotations Read(string reportsDir, Action<string>? log = null)
    {
        List<string> files;
        try
        {
            if (!Directory.Exists(reportsDir))
                return Empty;

            files = Directory.EnumerateFiles(reportsDir, "*.metadata.csv")
                .Concat(Directory.EnumerateFiles(reportsDir, "Metadata.csv"))
                // A sidecar is named after its destination, so an unfinished
                // ".prism-partial-<hex>.<label>.metadata.csv" matches the glob above. Left by a stopped
                // run, it would be read as a document's replicate metadata under the bogus label
                // ".prism-partial-<hex>.<label>".
                .Where(f => !HeadlessSkylineExporter.IsSidecar(f))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
                .ToList();
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            log?.Invoke($"Replicates reports not read from {reportsDir}: {ex.Message}");
            return Empty;
        }

        var bySample = new Dictionary<string, Dictionary<string, string>>(StringComparer.Ordinal);
        var columns = new List<string>();
        foreach (var file in files)
        {
            var name = Path.GetFileName(file);
            // "<label>.metadata.csv" -> "<label>"; the legacy "Metadata.csv" has no document label.
            var label = name.EndsWith(".metadata.csv", StringComparison.OrdinalIgnoreCase)
                ? name[..^".metadata.csv".Length]
                : null;
            try
            {
                ReadReport(file, label, bySample, columns);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                // The rest of the directory still counts; this document's replicates group by their
                // sample type until the file can be read.
                log?.Invoke($"Replicates report {name} not read: {ex.Message}");
            }
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
        var header = CsvLine.Split(lines[0]);
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
            var f = CsvLine.Split(lines[r]);
            if (f.Length <= repIdx)
                continue;
            var rep = f[repIdx].Trim();
            if (rep.Length == 0)
                continue;
            var map = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var (name, idx) in cols)
                map[name] = idx < f.Length ? f[idx].Trim() : "";
            // A labeled report keys its rows by the merged Sample ID and nothing else, so no other
            // document can borrow them; only the legacy label-less file keys by bare replicate name.
            if (!string.IsNullOrEmpty(documentLabel))
                bySample[rep + ReplicateDataFiles.SampleIdSeparator + documentLabel] = map;
            else
                bySample[rep] = map;
        }
    }
}
