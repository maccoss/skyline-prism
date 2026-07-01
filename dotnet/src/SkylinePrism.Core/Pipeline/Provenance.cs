using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text.Json;
using SkylinePrism.Core.Config;

namespace SkylinePrism.Core.Pipeline;

/// <summary>
/// Reads and writes metadata.json - the provenance record that captures the full PrismConfig plus
/// run statistics so a run can be reproduced exactly (CLI --from-provenance, or the tool's "Open
/// provenance"). The complete config is embedded under "processing_parameters" (snake_case keys
/// matching the YAML), a superset of the Python schema.
/// </summary>
public static class Provenance
{
    public sealed record Stats(int NSamples, int NPeptides, int NProteins, int NProteinGroups);

    private static JsonSerializerOptions Options() => new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        PropertyNameCaseInsensitive = true,
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.Never,
    };

    public static void Write(
        string path, PrismConfig config, IReadOnlyList<string> sourceFiles, Stats stats, string processingDateUtc)
    {
        var version = Assembly.GetExecutingAssembly().GetName().Version?.ToString() ?? "0.0.0";
        var doc = new Dictionary<string, object?>
        {
            ["pipeline_version"] = version,
            ["processing_date"] = processingDateUtc,
            ["source_files"] = sourceFiles,
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
    /// Reconstruct the PrismConfig from a run's metadata.json. Reads the embedded full config; if
    /// the file only carries the Python-subset sections, those still deserialize onto defaults.
    /// </summary>
    public static PrismConfig LoadConfig(string metadataJsonPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(metadataJsonPath));
        if (!doc.RootElement.TryGetProperty("processing_parameters", out var pp))
            throw new InvalidOperationException(
                $"'{metadataJsonPath}' has no 'processing_parameters' (not a PRISM provenance file).");
        return JsonSerializer.Deserialize<PrismConfig>(pp.GetRawText(), Options()) ?? new PrismConfig();
    }

    /// <summary>The input source files recorded in a provenance file (empty if absent).</summary>
    public static IReadOnlyList<string> SourceFiles(string metadataJsonPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(metadataJsonPath));
        var list = new List<string>();
        if (doc.RootElement.TryGetProperty("source_files", out var sf) && sf.ValueKind == JsonValueKind.Array)
            foreach (var e in sf.EnumerateArray())
                if (e.GetString() is { } s)
                    list.Add(s);
        return list;
    }
}
