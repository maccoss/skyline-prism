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
    public Dictionary<string, string> TypeByReplicate { get; } = new(StringComparer.Ordinal);
    public Dictionary<string, string> BatchByReplicate { get; } = new(StringComparer.Ordinal);

    public bool HasTypes => TypeByReplicate.Count > 0;
    public bool HasBatches => BatchByReplicate.Count > 0;

    private static readonly string[] ReplicateCols = { "Replicate", "Replicate Name", "ReplicateName", "ReplicateLocator" };
    private static readonly string[] SampleTypeCols = { "Sample Type", "SampleType" };
    private static readonly string[] BatchCols = { "Batch", "Batch Name", "BatchName", "Batch Name Annotation" };

    /// <summary>
    /// Parse the exported Replicates report. The report's columns are variable (they depend on
    /// the document's annotation fields), so all headers are logged, Sample Type is auto-detected
    /// (standard built-in), and the batch/type columns may be named explicitly via
    /// <paramref name="batchColumn"/> / <paramref name="sampleTypeColumn"/> (case-insensitive).
    /// </summary>
    /// <summary>Load and merge several metadata files (later files win on a duplicate replicate).</summary>
    public static ReplicateMetadata? TryLoad(
        IReadOnlyList<string>? paths, Action<string>? log = null,
        string? sampleTypeColumn = null, string? batchColumn = null)
    {
        if (paths is null || paths.Count == 0)
            return null;
        ReplicateMetadata? merged = null;
        foreach (var p in paths)
        {
            var md = TryLoad(p, log, sampleTypeColumn, batchColumn);
            if (md is null)
                continue;
            if (merged is null)
            {
                merged = md;
                continue;
            }
            foreach (var kv in md.TypeByReplicate)
                merged.TypeByReplicate[kv.Key] = kv.Value;
            foreach (var kv in md.BatchByReplicate)
                merged.BatchByReplicate[kv.Key] = kv.Value;
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
