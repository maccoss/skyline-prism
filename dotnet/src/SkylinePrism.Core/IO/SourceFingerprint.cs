using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace SkylinePrism.Core.IO;

/// <summary>
/// Fingerprint of a merge's input files (path + size + last-write-time) plus a small cache sidecar,
/// so a re-run to the same output directory can reuse merged_data.parquet instead of re-merging -
/// unless the inputs changed or --force-reprocess is set. Mirrors the Python source-fingerprint cache.
/// </summary>
public static class SourceFingerprint
{
    public sealed record CacheEntry(string Fingerprint, long TotalRows, string SortColumn);

    public static string Compute(IReadOnlyList<string> inputs)
    {
        var sb = new StringBuilder();
        foreach (var p in inputs.OrderBy(x => x, StringComparer.Ordinal))
        {
            var fi = new FileInfo(p);
            sb.Append(Path.GetFullPath(p)).Append('|')
              .Append(fi.Exists ? fi.Length : -1).Append('|')
              .Append(fi.Exists ? fi.LastWriteTimeUtc.Ticks : 0).Append('\n');
        }
        var hash = SHA256.HashData(Encoding.UTF8.GetBytes(sb.ToString()));
        return Convert.ToHexString(hash);
    }

    public static CacheEntry? TryRead(string path)
    {
        try
        {
            return File.Exists(path)
                ? JsonSerializer.Deserialize<CacheEntry>(File.ReadAllText(path))
                : null;
        }
        catch
        {
            return null;
        }
    }

    public static void Write(string path, CacheEntry entry)
    {
        try
        {
            File.WriteAllText(path, JsonSerializer.Serialize(entry));
        }
        catch (IOException)
        {
            // A missing cache just means we re-merge next time; never fail the run over it.
        }
    }
}
