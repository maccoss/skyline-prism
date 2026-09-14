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
        var json = File.ReadAllText(metadataJsonPath);
        var config = ConfigFromJson(json, metadataJsonPath);
        using var doc = JsonDocument.Parse(json);

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
    /// The config a run recorded, exactly as it recorded it.
    /// </summary>
    /// <remarks>
    /// <see cref="LoadConfig(string)"/> is for RE-RUNNING that config, so it redirects a FASTA whose
    /// original path has gone to the copy the run archived. That is right for running and wrong for
    /// COMPARING: the redirect rewrites <c>parsimony.fasta_path</c> to a path inside the output
    /// directory, so a result whose database has since moved would compare as "different settings"
    /// against the very config that produced it - and the comparison would fail exactly when the
    /// archive had done its job.
    /// </remarks>
    public static PrismConfig ConfigFromJson(string json, string describePath)
    {
        using var doc = JsonDocument.Parse(json);
        if (!doc.RootElement.TryGetProperty("processing_parameters", out var pp))
            throw new InvalidOperationException(
                $"'{describePath}' has no 'processing_parameters' (not a PRISM provenance file).");
        return JsonSerializer.Deserialize<PrismConfig>(pp.GetRawText(), Options()) ?? new PrismConfig();
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
    /// Record the extraction windows a run used, under <c>extraction</c>. Returns true only when the
    /// file was actually changed.
    /// </summary>
    /// <remarks>
    /// <para><b>Why this is here at all.</b> The tolerances live in the Skyline document, and a result
    /// outlives the document as surely as it outlives the instrument files. Without them an archived
    /// output directory cannot say what its ion accounting was extracted with - and the extraction
    /// window decides how much fragment sharing is found between co-isolated peptides, so every
    /// assigned figure moves with it. Same argument as <see cref="RecordIsolationSchemes"/>, same
    /// file, for the same reason.</para>
    ///
    /// <para><b>The quadruple, not a string.</b> An earlier version of this recorded
    /// <c>ProductMassTolerance.ToSetting()</c>, which is documented to return <b>null</b> for tof,
    /// orbitrap and ft_icr, and for QIT with selective extraction - so on a resolving-power
    /// instrument it silently recorded nothing at all, and the feature read as working because the
    /// common centroided case did. The other string form, <c>Describe()</c>, is a caption and does
    /// not parse back. A tolerance is four values - analyzer, resolution, the m/z that resolving
    /// power is calibrated at, and whether selective extraction halves the window - and all four are
    /// written, with the caption beside them for a human. The edges of an isolation scheme are
    /// recorded rather than its summary for exactly the same reason.</para>
    ///
    /// <para><b>Additive and non-fatal</b>, like the schemes above: the document is re-read and one
    /// property set, so <c>processing_parameters</c> does not move.</para>
    /// </remarks>
    public static bool RecordExtraction(
        string outputDir, ProductMassTolerance? product, ProductMassTolerance? precursor,
        string? source = null)
    {
        if (product is null && precursor is null)
            return false;
        var path = Path.Combine(outputDir, FileName);
        if (!File.Exists(path))
            return false;

        try
        {
            if (JsonNode.Parse(File.ReadAllText(path)) is not JsonObject root)
                return false;

            var entry = new JsonObject();
            if (source is not null)
                entry["source"] = source;
            if (product is not null)
                entry["product"] = Describe(product);
            if (precursor is not null)
                entry["precursor"] = Describe(precursor);

            // Re-recording what the file already says is not a change. No timestamp is written, so
            // the comparison is over the settings themselves and a re-run of the same document is
            // correctly silent.
            if (string.Equals(
                    root["extraction"]?.ToJsonString(), entry.ToJsonString(), StringComparison.Ordinal))
            {
                return false;
            }

            root["extraction"] = entry;
            File.WriteAllText(path, root.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
            return true;
        }
        catch (Exception ex) when (ex is IOException or JsonException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>
    /// The extraction settings a previous run recorded, or (null, null) when it recorded none.
    /// </summary>
    /// <remarks>
    /// The point of writing them: a directory whose document has moved on can still say what its
    /// numbers were extracted with, and can be re-measured the same way without one.
    /// </remarks>
    public static (ProductMassTolerance? Product, ProductMassTolerance? Precursor) ReadExtraction(
        string outputDir)
    {
        var path = Path.Combine(outputDir, FileName);
        if (!File.Exists(path))
            return (null, null);
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            if (!doc.RootElement.TryGetProperty("extraction", out var extraction))
                return (null, null);
            return (Tolerance(extraction, "product"), Tolerance(extraction, "precursor"));
        }
        catch (Exception ex) when (ex is IOException or JsonException or UnauthorizedAccessException)
        {
            return (null, null);
        }
    }

    /// <summary>All four values a tolerance is, plus the caption a human reads.</summary>
    private static JsonObject Describe(ProductMassTolerance tolerance) =>
        new()
        {
            ["analyzer"] = tolerance.Analyzer,
            ["resolution"] = tolerance.Resolution,
            ["resolution_mz"] = tolerance.ResolutionMz,
            ["selective_extraction"] = tolerance.SelectiveExtraction,
            ["summary"] = tolerance.Describe(),
        };

    /// <inheritdoc cref="Describe"/>
    private static ProductMassTolerance? Tolerance(JsonElement extraction, string name)
    {
        if (!extraction.TryGetProperty(name, out var e) || e.ValueKind != JsonValueKind.Object)
            return null;
        if (!e.TryGetProperty("analyzer", out var analyzer)
            || analyzer.ValueKind != JsonValueKind.String
            || !e.TryGetProperty("resolution", out var resolution)
            || !resolution.TryGetDouble(out var res))
        {
            return null;
        }

        double? resMz = null;
        if (e.TryGetProperty("resolution_mz", out var mz) && mz.ValueKind == JsonValueKind.Number
            && mz.TryGetDouble(out var mzValue))
        {
            resMz = mzValue;
        }
        var selective = e.TryGetProperty("selective_extraction", out var s)
            && s.ValueKind == JsonValueKind.True;

        return new ProductMassTolerance(analyzer.GetString()!, res, resMz, selective);
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
