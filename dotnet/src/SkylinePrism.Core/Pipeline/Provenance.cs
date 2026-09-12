using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.Qc;

namespace SkylinePrism.Core.Pipeline;

/// <summary>
/// Reads and writes parameters.json - the provenance record that captures the full PrismConfig plus
/// run statistics so a run can be reproduced exactly (CLI --from-provenance, or the tool's "Open
/// provenance"). Named parameters.json (not metadata.json) to avoid confusion with the scientific
/// sample/experiment metadata. The complete config is embedded under "processing_parameters"
/// (snake_case keys matching the YAML), a superset of the Python schema.
/// </summary>
public static class Provenance
{
    /// <summary>The provenance file's name, in the output directory.</summary>
    public const string FileName = "parameters.json";

    public sealed record Stats(int NSamples, int NPeptides, int NProteins, int NProteinGroups);

    /// <summary>
    /// The header facts of a completed run, as recorded in its parameters.json: which PRISM built the
    /// outputs, when, and from what. Read back by the QC report so the report states the provenance of
    /// the numbers it shows rather than the provenance of whatever binary rendered it.
    /// </summary>
    /// <param name="IsolationSchemes">
    /// One summary per isolation scheme the run recorded, empty when none was (a run predating
    /// <see cref="RecordIsolationSchemes"/>, or one whose data files were never reachable).
    /// </param>
    public sealed record RunInfo(
        string PipelineVersion, string ProcessingDate, string Host, IReadOnlyList<string> SourceFiles,
        IReadOnlyList<string>? IsolationSchemes = null);

    private static JsonSerializerOptions Options() => new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        PropertyNameCaseInsensitive = true,
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.Never,
    };

    public static void Write(
        string path, PrismConfig config, IReadOnlyList<string> sourceFiles, Stats stats,
        string processingDateUtc, IReadOnlyList<FastaArchive.Entry>? archivedFasta = null)
    {
        var doc = new Dictionary<string, object?>
        {
            ["pipeline_version"] = PrismVersion.Current,
            ["processing_date"] = processingDateUtc,
            // The machine that produced the outputs. Recorded here rather than read off the current
            // host when the QC report renders, because `prism qc` can regenerate a report anywhere.
            ["host"] = Environment.MachineName,
            ["source_files"] = sourceFiles,
            // Where each FASTA came from and where the run kept a copy. Recorded beside the config
            // rather than inside it so the config still names the ORIGINAL path: the stage cache stamps
            // that path, and rewriting it to the copy would invalidate every downstream stage on a
            // re-run that changed nothing.
            ["fasta_files"] = archivedFasta is { Count: > 0 } ? archivedFasta : null,
            ["processing_parameters"] = config,
            ["statistics"] = new Dictionary<string, int>
            {
                ["n_samples"] = stats.NSamples,
                ["n_peptides"] = stats.NPeptides,
                ["n_proteins"] = stats.NProteins,
                ["n_protein_groups"] = stats.NProteinGroups,
            },
        };
        File.WriteAllText(path, JsonSerializer.Serialize(doc, Options()));
    }

    /// <summary>
    /// Reconstruct the PrismConfig from a run's parameters.json. Reads the embedded full config; if
    /// the file only carries the Python-subset sections, those still deserialize onto defaults.
    /// </summary>
    public static PrismConfig LoadConfig(string metadataJsonPath) =>
        LoadConfig(metadataJsonPath, out _);

    /// <summary>
    /// As <see cref="LoadConfig(string)"/>, also reporting any FASTA that was resolved to the copy the
    /// run archived because the original path no longer exists. The caller should say so: substituting
    /// a database silently is exactly the failure this archive exists to prevent.
    /// </summary>
    public static PrismConfig LoadConfig(string metadataJsonPath, out IReadOnlyList<string> redirectedFasta)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(metadataJsonPath));
        if (!doc.RootElement.TryGetProperty("processing_parameters", out var pp))
            throw new InvalidOperationException(
                $"'{metadataJsonPath}' has no 'processing_parameters' (not a PRISM provenance file).");
        var config = JsonSerializer.Deserialize<PrismConfig>(pp.GetRawText(), Options()) ?? new PrismConfig();

        redirectedFasta = Array.Empty<string>();
        if (doc.RootElement.TryGetProperty("fasta_files", out var fasta)
            && fasta.ValueKind == JsonValueKind.Array)
        {
            var entries = JsonSerializer.Deserialize<List<FastaArchive.Entry>>(fasta.GetRawText(), Options());
            if (entries is { Count: > 0 })
            {
                var dir = Path.GetDirectoryName(Path.GetFullPath(metadataJsonPath)) ?? ".";
                redirectedFasta = FastaArchive.Restore(config, dir, entries);
            }
        }
        return config;
    }

    /// <summary>
    /// The run header from a parameters.json - version, date, inputs. Returns null when the file is
    /// absent or unreadable: a QC report regenerated beside a run that predates provenance, or one
    /// generated by hand, must still render.
    /// </summary>
    public static RunInfo? ReadRunInfo(string metadataJsonPath)
    {
        try
        {
            if (!File.Exists(metadataJsonPath))
                return null;
            using var doc = JsonDocument.Parse(File.ReadAllText(metadataJsonPath));
            var root = doc.RootElement;
            var version = root.TryGetProperty("pipeline_version", out var v) ? v.GetString() : null;
            var date = root.TryGetProperty("processing_date", out var d) ? d.GetString() : null;
            var host = root.TryGetProperty("host", out var h) ? h.GetString() : null;
            return new RunInfo(
                version ?? "unknown", date ?? "unknown", host ?? "unknown", ReadSourceFiles(root),
                IsolationSchemeSummaries(metadataJsonPath));
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Record the isolation windows a run learned about in its <c>parameters.json</c>, under
    /// <c>isolation_schemes</c>. Returns true only when the file was actually changed - there is no
    /// provenance file, nothing to record, or it already says exactly this.
    /// </summary>
    /// <remarks>
    /// <para><b>Why here as well as <c>isolation_schemes.xml</c>.</b> The windows exist in exactly two
    /// places otherwise: the instrument files and that XML, both beside the run. The instrument files
    /// are routinely moved off a share or deleted once an analysis is done, and at that point nothing
    /// can say what the data was acquired with - the Spectrum density map falls back to a guessed grid
    /// and nothing on it says the grid is a guess. <c>parameters.json</c> is the file that travels with
    /// a result, so the scheme is written into it too, in full: the window edges are what makes it
    /// reconstructable rather than merely described.</para>
    ///
    /// <para><b>Additive and non-fatal.</b> The existing document is re-read and one property is
    /// set, so nothing else in it moves - <c>processing_parameters</c> in particular, which
    /// <see cref="LoadConfig(string)"/> reads back for <c>--from-provenance</c>. A run whose
    /// provenance cannot be rewritten still has the XML; losing this is losing a convenience.</para>
    /// </remarks>
    public static bool RecordIsolationSchemes(string outputDir, IsolationSchemeCatalog catalog)
    {
        if (catalog is null)
            return false;
        var path = Path.Combine(outputDir, FileName);
        if (!File.Exists(path))
            return false;

        var entries = Entries(catalog);
        if (entries.Count == 0)
            return false;

        try
        {
            if (JsonNode.Parse(File.ReadAllText(path)) is not JsonObject root)
                return false;

            var array = new JsonArray();
            foreach (var entry in entries)
                array.Add(entry);

            // Re-recording what the file already says is not a change, and saying so lets a caller
            // report the write honestly instead of announcing one on every re-run. The measurement
            // timestamp round-trips through isolation_schemes.xml unaltered, so a re-read of the same
            // acquisition compares equal and a genuinely new reading does not.
            if (string.Equals(
                    root["isolation_schemes"]?.ToJsonString(), array.ToJsonString(), StringComparison.Ordinal))
            {
                return false;
            }

            root["isolation_schemes"] = array;
            File.WriteAllText(path, root.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
            return true;
        }
        catch (Exception ex) when (ex is IOException or JsonException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>
    /// One entry per scheme worth recording: the measured ones first, then any document that declared
    /// real windows of its own. A document that declares none ("Results only", the normal DIA setting)
    /// contributes nothing - there is no geometry in it to preserve.
    /// </summary>
    private static List<JsonObject> Entries(IsolationSchemeCatalog catalog)
    {
        var entries = new List<JsonObject>();
        foreach (var measured in catalog.Measured)
        {
            var entry = new JsonObject
            {
                ["source"] = "measured",
                ["data_file"] = measured.DataFile,
                ["recorded"] = measured.MeasuredUtc,
            };
            Describe(entry, measured.Scheme);
            entries.Add(entry);
        }
        foreach (var (batch, scheme) in catalog.ByBatch.OrderBy(
                     kv => kv.Key, StringComparer.OrdinalIgnoreCase))
        {
            if (!scheme.HasWindows || catalog.IsMeasured(scheme))
                continue;
            var entry = new JsonObject { ["source"] = "document", ["batch"] = batch };
            if (catalog.AcquisitionFor(batch) is { } method)
                entry["acquisition"] = method;
            Describe(entry, scheme);
            entries.Add(entry);
        }
        return entries;
    }

    /// <summary>The scheme itself: the one-line summary a human reads, and the edges a program needs.</summary>
    private static void Describe(JsonObject entry, IsolationScheme scheme)
    {
        entry["name"] = scheme.Name;
        entry["summary"] = scheme.Describe();
        entry["window_count"] = scheme.Windows.Count;
        entry["mz_start"] = Round(scheme.MzLow);
        entry["mz_end"] = Round(scheme.MzHigh);
        entry["scheduled"] = scheme.IsScheduled;

        var windows = new JsonArray();
        foreach (var window in scheme.Windows)
        {
            var node = new JsonObject { ["start"] = Round(window.Start), ["end"] = Round(window.End) };
            if (window.Margin != 0)
                node["margin"] = Round(window.Margin);
            if (window.IsScheduled)
            {
                node["rt_start"] = Round(window.RtStart);
                node["rt_stop"] = Round(window.RtStop);
            }
            windows.Add(node);
        }
        entry["windows"] = windows;
    }

    /// <summary>
    /// Window edges to 1e-6 Th. Instrument-reported edges carry float noise well below any real
    /// window, and writing it out unrounded turns a 167-window scheme into a wall of 17 digits.
    /// </summary>
    private static double Round(double value) =>
        double.IsFinite(value) ? Math.Round(value, 6, MidpointRounding.AwayFromZero) : value;

    /// <summary>
    /// The isolation schemes recorded in a provenance file, as their one-line summaries. Empty when
    /// the file is absent, unreadable, or predates this being recorded.
    /// </summary>
    public static IReadOnlyList<string> IsolationSchemeSummaries(string metadataJsonPath)
    {
        try
        {
            if (!File.Exists(metadataJsonPath))
                return Array.Empty<string>();
            using var doc = JsonDocument.Parse(File.ReadAllText(metadataJsonPath));
            if (!doc.RootElement.TryGetProperty("isolation_schemes", out var schemes)
                || schemes.ValueKind != JsonValueKind.Array)
            {
                return Array.Empty<string>();
            }

            var list = new List<string>();
            foreach (var entry in schemes.EnumerateArray())
            {
                var summary = entry.TryGetProperty("summary", out var s) ? s.GetString() : null;
                if (string.IsNullOrWhiteSpace(summary))
                    continue;
                var source = entry.TryGetProperty("source", out var src) ? src.GetString() : null;
                list.Add(string.Equals(source, "measured", StringComparison.Ordinal)
                    ? summary + " (measured from the data)"
                    : summary);
            }
            return list;
        }
        catch (Exception ex) when (ex is IOException or JsonException or UnauthorizedAccessException)
        {
            return Array.Empty<string>();
        }
    }

    /// <summary>The input source files recorded in a provenance file (empty if absent).</summary>
    public static IReadOnlyList<string> SourceFiles(string metadataJsonPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(metadataJsonPath));
        return ReadSourceFiles(doc.RootElement);
    }

    private static List<string> ReadSourceFiles(JsonElement root)
    {
        var list = new List<string>();
        if (root.TryGetProperty("source_files", out var sf) && sf.ValueKind == JsonValueKind.Array)
            foreach (var e in sf.EnumerateArray())
                if (e.GetString() is { } s)
                    list.Add(s);
        return list;
    }
}
