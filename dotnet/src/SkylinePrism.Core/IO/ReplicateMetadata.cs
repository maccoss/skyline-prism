using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SkylinePrism.Core.IO;

/// <summary>
/// Per-replicate sample annotations parsed from Skyline's built-in Replicates report
/// (invariant CSV). Maps the replicate name -> PRISM sample type (Skyline "Standard"
/// -> reference, "Quality Control" -> qc, else experimental) and -> batch (from a Batch
/// annotation column when present). Column detection is tolerant of naming variants.
/// </summary>
public sealed class ReplicateMetadata
{
    /// <summary>
    /// The separator DuckDbMerge uses to build "&lt;replicate&gt;__@__&lt;batch&gt;" sample IDs. When several
    /// documents are merged, the batch part is the source-document label, so the sample ID doubles as the
    /// document-qualified replicate key.
    /// </summary>
    public const string SampleIdSeparator = "__@__";

    public Dictionary<string, string> TypeByReplicate { get; } = new(StringComparer.Ordinal);
    public Dictionary<string, string> BatchByReplicate { get; } = new(StringComparer.Ordinal);

    /// <summary>
    /// Same annotations keyed by the DOCUMENT-QUALIFIED replicate ("&lt;replicate&gt;__@__&lt;document&gt;"),
    /// populated when <see cref="TryLoad(IReadOnlyList{string}, Action{string}, string, string, IReadOnlyList{string})"/>
    /// is given per-file document labels. Reference/QC injections routinely carry the SAME replicate name in
    /// every plate's document, so the unqualified maps above would silently let the last file win - collapsing
    /// two documents into one batch and overwriting sample types. These qualified entries take precedence.
    /// </summary>
    public Dictionary<string, string> TypeBySampleId { get; } = new(StringComparer.Ordinal);
    public Dictionary<string, string> BatchBySampleId { get; } = new(StringComparer.Ordinal);

    public bool HasTypes => TypeByReplicate.Count > 0 || TypeBySampleId.Count > 0;
    public bool HasBatches => BatchByReplicate.Count > 0 || BatchBySampleId.Count > 0;

    /// <summary>Document-qualified key for a replicate, matching the merged "Sample ID" column.</summary>
    public static string QualifiedKey(string replicate, string documentLabel)
        => replicate + SampleIdSeparator + documentLabel;

    /// <summary>Sample type for a merged sample ID: the document-qualified entry wins over the bare replicate.</summary>
    public string? TypeFor(string sampleId, string replicate)
        => TypeBySampleId.GetValueOrDefault(sampleId) ?? TypeByReplicate.GetValueOrDefault(replicate);

    /// <summary>Batch for a merged sample ID: the document-qualified entry wins over the bare replicate.</summary>
    public string? BatchFor(string sampleId, string replicate)
        => BatchBySampleId.GetValueOrDefault(sampleId) ?? BatchByReplicate.GetValueOrDefault(replicate);

    /// <summary>Whether any batch annotation applies to this sample (qualified or bare).</summary>
    public bool HasBatchFor(string sampleId, string replicate)
        => BatchBySampleId.ContainsKey(sampleId) || BatchByReplicate.ContainsKey(replicate);

    private static readonly string[] ReplicateCols = { "Replicate", "Replicate Name", "ReplicateName", "ReplicateLocator" };
    private static readonly string[] SampleTypeCols = { "Sample Type", "SampleType" };
    private static readonly string[] BatchCols = { "Batch", "Batch Name", "BatchName", "Batch Name Annotation" };

    /// <summary>
    /// Parse the exported Replicates report. The report's columns are variable (they depend on
    /// the document's annotation fields), so all headers are logged, Sample Type is auto-detected
    /// (standard built-in), and the batch/type columns may be named explicitly via
    /// <paramref name="batchColumn"/> / <paramref name="sampleTypeColumn"/> (case-insensitive).
    /// </summary>
    /// <summary>
    /// Load and merge several metadata files. Later files win on a duplicate bare replicate name, but when
    /// <paramref name="documentLabels"/> supplies a per-file source-document label (the same label
    /// <see cref="DuckDbMerge"/> uses to build the "Sample ID"), each file's rows are ALSO recorded under the
    /// document-qualified key, and those take precedence at lookup - so a replicate name reused across
    /// documents (every plate's "Ref_01") keeps its own document's sample type and batch.
    /// </summary>
    public static ReplicateMetadata? TryLoad(
        IReadOnlyList<string>? paths, Action<string>? log = null,
        string? sampleTypeColumn = null, string? batchColumn = null,
        IReadOnlyList<string>? documentLabels = null)
    {
        if (paths is null || paths.Count == 0)
            return null;
        if (documentLabels is not null && documentLabels.Count != paths.Count)
            throw new ArgumentException(
                "documentLabels must have the same length as paths.", nameof(documentLabels));

        ReplicateMetadata? merged = null;
        for (var i = 0; i < paths.Count; i++)
        {
            var md = TryLoad(paths[i], log, sampleTypeColumn, batchColumn);
            if (md is null)
                continue;

            // Qualify this file's rows with its source document, so same-named replicates stay distinct.
            var label = documentLabels?[i];
            if (!string.IsNullOrWhiteSpace(label))
            {
                foreach (var kv in md.TypeByReplicate)
                    md.TypeBySampleId[QualifiedKey(kv.Key, label!)] = kv.Value;
                foreach (var kv in md.BatchByReplicate)
                    md.BatchBySampleId[QualifiedKey(kv.Key, label!)] = kv.Value;
            }

            if (merged is null)
            {
                merged = md;
                continue;
            }
            foreach (var kv in md.TypeByReplicate)
                merged.TypeByReplicate[kv.Key] = kv.Value;
            foreach (var kv in md.BatchByReplicate)
                merged.BatchByReplicate[kv.Key] = kv.Value;
            foreach (var kv in md.TypeBySampleId)
                merged.TypeBySampleId[kv.Key] = kv.Value;
            foreach (var kv in md.BatchBySampleId)
                merged.BatchBySampleId[kv.Key] = kv.Value;
        }
        return merged;
    }

    public static ReplicateMetadata? TryLoad(
        string? path, Action<string>? log = null,
        string? sampleTypeColumn = null, string? batchColumn = null)
    {
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
            return null;

        var lines = File.ReadAllLines(path);
        if (lines.Length < 2)
            return null;

        var header = SplitCsv(lines[0]);
        log?.Invoke("Replicates report columns: " + string.Join(" | ", header.Select(h => h.Trim())));

        var repIdx = FindColumn(header, ReplicateCols);
        if (repIdx < 0)
        {
            log?.Invoke("Replicates metadata: no replicate-name column found; auto-generating sample metadata.");
            return null;
        }
        var typeIdx = !string.IsNullOrWhiteSpace(sampleTypeColumn)
            ? FindColumn(header, new[] { sampleTypeColumn! })
            : FindColumn(header, SampleTypeCols);
        // For an explicit batch column, also match the "annotation_<Name>" form and the
        // annotation display name (Skyline may export either).
        var batchIdx = !string.IsNullOrWhiteSpace(batchColumn)
            ? FindColumn(header, new[] { batchColumn!, "annotation_" + batchColumn })
            : FindColumn(header, BatchCols);

        if (!string.IsNullOrWhiteSpace(batchColumn) && batchIdx < 0)
            log?.Invoke($"Replicates metadata: requested batch column '{batchColumn}' not found in the report.");

        var md = new ReplicateMetadata();
        for (var i = 1; i < lines.Length; i++)
        {
            if (string.IsNullOrWhiteSpace(lines[i]))
                continue;
            var f = SplitCsv(lines[i]);
            if (f.Length <= repIdx)
                continue;
            var rep = f[repIdx].Trim();
            if (rep.Length == 0)
                continue;

            if (typeIdx >= 0 && f.Length > typeIdx)
                md.TypeByReplicate[rep] = MapSampleType(f[typeIdx]);
            if (batchIdx >= 0 && f.Length > batchIdx)
            {
                var b = f[batchIdx].Trim();
                if (b.Length > 0 && !b.Equals("#N/A", StringComparison.OrdinalIgnoreCase))
                    md.BatchByReplicate[rep] = b;
            }
        }

        log?.Invoke(
            $"Replicates metadata: replicate='{header[repIdx]}'"
            + (typeIdx >= 0 ? $", type='{header[typeIdx]}'" : ", type=(none)")
            + (batchIdx >= 0 ? $", batch='{header[batchIdx]}'" : ", batch=(none)")
            + $"; {md.TypeByReplicate.Count} typed, {md.BatchByReplicate.Count} batched.");
        return md;
    }

    /// <summary>Skyline sample type -> PRISM sample type.</summary>
    public static string MapSampleType(string? skylineType)
    {
        var t = (skylineType ?? "").Trim();
        if (t.Length == 0)
            return "experimental";
        if (t.Equals("Standard", StringComparison.OrdinalIgnoreCase))
            return "reference";
        if (t.Equals("Quality Control", StringComparison.OrdinalIgnoreCase)
            || t.Equals("QC", StringComparison.OrdinalIgnoreCase))
            return "qc";
        // Solvent / Blank / Double Blank are analytical blanks, excluded from ref/qc/experimental
        // groupings (matches Python's SKYLINE_SAMPLE_TYPE_MAP). Only Unknown/unannotated is experimental.
        if (t.Equals("Solvent", StringComparison.OrdinalIgnoreCase)
            || t.Equals("Blank", StringComparison.OrdinalIgnoreCase)
            || t.Equals("Double Blank", StringComparison.OrdinalIgnoreCase))
            return "blank";
        return "experimental"; // Unknown and any unannotated/custom value
    }

    private static int FindColumn(string[] header, string[] candidates)
    {
        foreach (var cand in candidates)
        {
            for (var i = 0; i < header.Length; i++)
                if (header[i].Trim().Equals(cand, StringComparison.OrdinalIgnoreCase))
                    return i;
        }
        return -1;
    }

    // Minimal RFC4180 splitter (handles double-quoted fields with embedded commas).
    private static string[] SplitCsv(string line)
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
