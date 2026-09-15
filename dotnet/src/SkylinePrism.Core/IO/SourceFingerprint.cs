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
    /// <param name="SortColumn">
    /// The PEPTIDE column, despite the name. The merge stopped sorting in dotnet-v26.12.0 and now
    /// hash-partitions on this column instead; the field keeps its old name so sidecars written by
    /// earlier releases still deserialize rather than forcing a needless re-merge of a cohort that has
    /// not changed. Rename it only together with a cache-version bump.
    /// </param>
    public sealed record CacheEntry(string Fingerprint, long TotalRows, string SortColumn);

    /// <param name="relativeTo">
    /// The output directory, when the caller has one. An input that lives under it is stamped by its
    /// path RELATIVE to it, which is what lets two machines agree about the same file.
    /// </param>
    /// <remarks>
    /// <para>The tool exports every report into <c>&lt;output&gt;/skyline-reports/</c>, so its merge
    /// inputs are always under the output directory - and a cohort on a share is <c>Z:\...</c> from one
    /// machine and <c>Y:\...</c> from another. Stamped by full path, that difference alone re-merged a
    /// cohort and recomputed every stage under it, for results that would have come out identical. Size
    /// and last-write time are properties of the file itself and already read the same from either
    /// machine; only the path had to change. It is the same rule <c>StageCache.Relative</c> uses for the
    /// outputs it records.</para>
    /// <para>An input OUTSIDE the output directory keeps its full path: there is nothing to make it
    /// relative to, and two unrelated files of the same size and time would otherwise look alike.</para>
    /// </remarks>
    public static string Compute(IReadOnlyList<string> inputs, string? relativeTo = null)
    {
        var sb = new StringBuilder();
        foreach (var p in inputs.OrderBy(x => x, StringComparer.Ordinal))
        {
            var fi = new FileInfo(p);
            sb.Append(Identify(p, relativeTo)).Append('|')
              .Append(fi.Exists ? fi.Length : -1).Append('|')
              .Append(fi.Exists ? fi.LastWriteTimeUtc.Ticks : 0).Append('\n');
        }
        var hash = SHA256.HashData(Encoding.UTF8.GetBytes(sb.ToString()));
        return Convert.ToHexString(hash);
    }

    /// <summary>
    /// How one input is named in the stamp: relative to <paramref name="relativeTo"/> when it lies
    /// under it, else its full path.
    /// </summary>
    private static string Identify(string path, string? relativeTo)
    {
        string full;
        try
        {
            full = Path.GetFullPath(path);
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException
                                       or PathTooLongException or System.Security.SecurityException)
        {
            return path;
        }
        if (string.IsNullOrWhiteSpace(relativeTo))
            return full;
        try
        {
            var relative = Path.GetRelativePath(Path.GetFullPath(relativeTo), full);
            // Kept only when it really is UNDER the directory. A prefix test is not a boundary test:
            // "/data/outside/report.csv" starts with "/data/out", and stamping it as
            // "../outside/report.csv" both contradicts the rule above and lets two unrelated files of
            // the same size and time collide when the directories are mounted differently.
            // GetRelativePath returns a rooted path when there is no common root at all.
            return Path.IsPathRooted(relative) || relative == ".."
                   || relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal)
                   || relative.StartsWith(".." + Path.AltDirectorySeparatorChar, StringComparison.Ordinal)
                ? full
                : relative;
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException
                                       or PathTooLongException or System.Security.SecurityException)
        {
            return full;
        }
    }

    /// <summary>
    /// The stamp to key this run on: the machine-independent one, unless what is already recorded is
    /// the pre-26.24 absolute form and still describes these inputs.
    /// </summary>
    /// <remarks>
    /// The merge stamp is not only the merge's own cache key - it is the upstream ingredient of every
    /// stage fingerprint below it. Switching an existing directory to the new form would therefore
    /// invalidate the whole chain and recompute the transition rollup, which is the most expensive
    /// stage there is, for a cohort that has not changed. Keeping the recorded form while it still
    /// holds costs nothing and leaves that chain intact; the directory moves to the new form the next
    /// time its inputs really do change, which is the moment everything below is recomputed anyway.
    /// </remarks>
    public static string Preferred(
        string? recorded, IReadOnlyList<string> inputs, string? relativeTo, string suffix = "")
    {
        var legacy = Compute(inputs) + suffix;
        return recorded is not null && string.Equals(recorded, legacy, StringComparison.Ordinal)
            ? legacy
            : Compute(inputs, relativeTo) + suffix;
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
