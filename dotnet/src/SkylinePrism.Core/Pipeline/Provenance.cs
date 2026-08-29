using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text.Json;
using SkylinePrism.Core.Config;

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
    public sealed record Stats(int NSamples, int NPeptides, int NProteins, int NProteinGroups);

    /// <summary>
    /// The header facts of a completed run, as recorded in its parameters.json: which PRISM built the
    /// outputs, when, and from what. Read back by the QC report so the report states the provenance of
    /// the numbers it shows rather than the provenance of whatever binary rendered it.
    /// </summary>
    public sealed record RunInfo(
        string PipelineVersion, string ProcessingDate, string Host, IReadOnlyList<string> SourceFiles);

    /// <summary>Version of the running PRISM assembly (the 4-part X.Y.Z.0 that `prism --version` prints).</summary>
    public static string AssemblyVersion =>
        Assembly.GetExecutingAssembly().GetName().Version?.ToString() ?? "0.0.0";

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
            ["pipeline_version"] = AssemblyVersion,
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
                version ?? "unknown", date ?? "unknown", host ?? "unknown", ReadSourceFiles(root));
        }
        catch
        {
            return null;
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
