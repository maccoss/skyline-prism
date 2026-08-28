using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using SkylinePrism.Core.Config;

namespace SkylinePrism.Core.Pipeline;

/// <summary>
/// Per-stage reuse for a re-run into the same output directory: a stage whose inputs and settings are
/// unchanged keeps its output instead of recomputing it.
///
/// <para>Changing one setting used to re-run everything. On a real cohort, changing
/// <c>protein_rollup.method</c> re-exported ~15 GB from Skyline and redid the transition rollup and
/// normalization, none of which that setting can affect - hours to change one parameter. Only the
/// Stage 1 merge was cached (<see cref="SourceFingerprint"/>, whose input-file fingerprint this
/// reuses).</para>
///
/// <para>A stage's fingerprint covers its upstream stages' fingerprints (so invalidation chains), the
/// external files it reads by path, the config keys it declares in <see cref="StageDependencies"/>,
/// and the assembly version - so any release re-runs everything rather than trusting that a code
/// change did not move a number. The entry is written only after the stage's outputs are closed, and
/// reuse additionally requires every declared output to still exist and be non-empty.</para>
/// </summary>
public sealed class StageCache
{
    public const string FileName = "stage_cache.json";

    /// <summary>Bump to invalidate every cache written by an earlier shape of this file.</summary>
    private const int SchemaVersion = 1;

    public sealed record Entry(string Fingerprint, List<string> Outputs);

    private sealed record Document(int Version, Dictionary<string, Entry> Stages);

    private readonly string _path;
    private readonly string _outputDir;
    private readonly Dictionary<string, Entry> _entries;
    private readonly bool _disabled;

    private StageCache(string path, string outputDir, Dictionary<string, Entry> entries, bool disabled)
    {
        _path = path;
        _outputDir = outputDir;
        _entries = entries;
        _disabled = disabled;
    }

    /// <summary>
    /// Read the sidecar for an output directory. <paramref name="forceReprocess"/> yields a cache that
    /// never reuses anything but still RECORDS, so the next run benefits.
    /// </summary>
    public static StageCache Load(string outputDir, bool forceReprocess = false)
    {
        var path = Path.Combine(outputDir, FileName);
        var entries = new Dictionary<string, Entry>(StringComparer.Ordinal);
        try
        {
            if (File.Exists(path))
            {
                var doc = JsonSerializer.Deserialize<Document>(File.ReadAllText(path));
                if (doc is not null && doc.Version == SchemaVersion && doc.Stages is not null)
                    entries = new Dictionary<string, Entry>(doc.Stages, StringComparer.Ordinal);
            }
        }
        catch
        {
            // A corrupt sidecar means "recompute", never "fail the run".
        }
        return new StageCache(path, outputDir, entries, forceReprocess);
    }

    /// <summary>
    /// The fingerprint of a stage: its declared config keys, the external files it reads, its upstream
    /// stages, and the PRISM version. Returned so the caller can pass it to <see cref="Record"/> and
    /// on to downstream stages.
    /// </summary>
    public static string Fingerprint(
        string stageId, PrismConfig config, IEnumerable<string>? upstream = null,
        IEnumerable<string>? extraInputs = null)
    {
        var sb = new StringBuilder();
        sb.Append("stage=").Append(stageId).Append('\n');
        // Any release invalidates every stage. Blunt, and deliberately so: a change to a rollup's
        // arithmetic leaves no trace in the config, and silently reusing across it would be the one
        // failure this cache must never produce.
        sb.Append("prism=").Append(
            Assembly.GetExecutingAssembly().GetName().Version?.ToString() ?? "0").Append('\n');
        foreach (var up in (upstream ?? Array.Empty<string>()).OrderBy(x => x, StringComparer.Ordinal))
            sb.Append("upstream=").Append(up).Append('\n');
        foreach (var input in (extraInputs ?? Array.Empty<string>()).OrderBy(x => x, StringComparer.Ordinal))
            sb.Append("input=").Append(input).Append('\n');
        foreach (var file in StageDependencies.ExternalFiles(stageId, config)
                     .OrderBy(x => x, StringComparer.OrdinalIgnoreCase))
            sb.Append("file=").Append(FileStamp(file)).Append('\n');
        sb.Append(StageDependencies.Values(stageId, config));

        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(sb.ToString()))).ToLowerInvariant();
    }

    /// <summary>Path, size and last-write-time - the same stamp <see cref="SourceFingerprint"/> uses.</summary>
    private static string FileStamp(string path)
    {
        try
        {
            var info = new FileInfo(path);
            return info.Exists
                ? $"{info.FullName}|{info.Length}|{info.LastWriteTimeUtc.Ticks}"
                : $"{path}|missing";
        }
        catch
        {
            return $"{path}|unreadable";
        }
    }

    /// <summary>
    /// Whether this stage can be skipped: a recorded entry with the same fingerprint, and every output
    /// it claimed still present and non-empty. A file deleted by hand correctly forces a re-run.
    /// </summary>
    public bool CanReuse(string stageId, string fingerprint)
    {
        if (_disabled)
            return false;
        if (!_entries.TryGetValue(stageId, out var entry) || entry.Fingerprint != fingerprint)
            return false;
        foreach (var output in entry.Outputs)
        {
            var full = Path.IsPathRooted(output) ? output : Path.Combine(_outputDir, output);
            if (Directory.Exists(full))
                continue;
            var info = new FileInfo(full);
            if (!info.Exists || info.Length == 0)
                return false;
        }
        return true;
    }

    /// <summary>
    /// Record a completed stage. Call AFTER its outputs are written and closed - an entry written
    /// first would survive a crash mid-write and vouch for a truncated file.
    /// </summary>
    public void Record(string stageId, string fingerprint, params string?[] outputs)
    {
        var relative = outputs
            .Where(o => !string.IsNullOrWhiteSpace(o))
            .Select(o => Relative(o!))
            .ToList();
        _entries[stageId] = new Entry(fingerprint, relative);
        Save();
    }

    /// <summary>Drop a stage's entry - used before recomputing, so a crash cannot leave it vouched for.</summary>
    public void Invalidate(string stageId)
    {
        if (_entries.Remove(stageId))
            Save();
    }

    private string Relative(string path)
    {
        try
        {
            var full = Path.GetFullPath(path);
            var root = Path.GetFullPath(_outputDir);
            return full.StartsWith(root, StringComparison.OrdinalIgnoreCase)
                ? Path.GetRelativePath(root, full)
                : full;
        }
        catch
        {
            return path;
        }
    }

    private void Save()
    {
        try
        {
            var doc = new Document(SchemaVersion, _entries);
            File.WriteAllText(_path, JsonSerializer.Serialize(doc, new JsonSerializerOptions
            {
                WriteIndented = true,
            }));
        }
        catch
        {
            // Losing the cache costs time on the next run; failing the run costs the run.
        }
    }
}
