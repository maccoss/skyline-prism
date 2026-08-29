using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Config;

namespace SkylinePrism.Core.Pipeline;

/// <summary>
/// Keeps a copy of the FASTA a run used beside its outputs, so the run stays reproducible after the
/// database moves.
///
/// <para>Provenance records <c>parsimony.fasta_path</c> as an absolute path on the machine that ran it.
/// That is enough to describe the run and not enough to repeat it: search databases get reorganized,
/// renamed and version-bumped, and an output directory handed to a collaborator points at a path that
/// exists on nobody else's disk. A re-run then silently falls back to the Skyline accession column for
/// parsimony, or to observed peptide counts for iBAQ - the same numbers never come back, and nothing
/// says why.</para>
///
/// <para>The original path is still what the config carries, so the stage cache keeps seeing the same
/// file and a re-run is not invalidated by the copy existing. The archive is consulted only when the
/// original has gone - see <see cref="Provenance.LoadConfig"/>.</para>
/// </summary>
public static class FastaArchive
{
    /// <summary>Subdirectory of the output directory holding the copies.</summary>
    public const string DirectoryName = "fasta";

    /// <summary>
    /// One archived database: which config key named it, where it came from, and the copy's path
    /// relative to the output directory.
    /// </summary>
    public sealed record Entry(string ConfigKey, string OriginalPath, string ArchivedPath);

    /// <summary>
    /// Copy every FASTA the config names into <paramref name="outputDir"/>, returning what was archived.
    /// Silent about a file it cannot read: a copy is a convenience, and failing the run at Stage 5 over
    /// it would throw away everything already computed.
    /// </summary>
    public static IReadOnlyList<Entry> Archive(
        PrismConfig config, string outputDir, Action<string>? report = null)
    {
        var wanted = new (string Key, string? Path)[]
        {
            ("parsimony.fasta_path", config.Parsimony.FastaPath),
            ("protein_rollup.ibaq.fasta_path", config.ProteinRollup.Ibaq.FastaPath),
        };

        var entries = new List<Entry>();
        var taken = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var (key, path) in wanted)
        {
            if (string.IsNullOrWhiteSpace(path))
                continue;
            try
            {
                var source = new FileInfo(path!);
                if (!source.Exists)
                    continue;

                // Already inside the output directory - a re-run reading last time's archive, or a
                // database the user keeps beside the results. Copying it onto itself helps nobody.
                if (IsInside(source.FullName, outputDir))
                {
                    entries.Add(new Entry(key, source.FullName, Relative(source.FullName, outputDir)));
                    continue;
                }

                // The same file named by both keys is archived once and referenced twice.
                if (taken.TryGetValue(source.FullName, out var already))
                {
                    entries.Add(new Entry(key, source.FullName, already));
                    continue;
                }

                var target = UniqueTarget(outputDir, source);
                Directory.CreateDirectory(Path.GetDirectoryName(target)!);
                source.CopyTo(target, overwrite: true);

                var relative = Relative(target, outputDir);
                taken[source.FullName] = relative;
                entries.Add(new Entry(key, source.FullName, relative));
                report?.Invoke(
                    $"  Archived {source.Name} ({source.Length / 1024.0 / 1024.0:0.#} MB) to {relative} - "
                    + "the run stays reproducible if the database moves.");
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                          or ArgumentException or NotSupportedException)
            {
                report?.Invoke($"  Could not archive the FASTA for {key}: {ex.Message}");
            }
        }
        return entries;
    }

    /// <summary>
    /// Point <paramref name="config"/> at the archived copies for any database whose original path has
    /// gone. Returns the keys that were redirected, so the caller can say so rather than substituting a
    /// file silently.
    /// </summary>
    public static IReadOnlyList<string> Restore(
        PrismConfig config, string provenanceDir, IReadOnlyList<Entry> entries)
    {
        var redirected = new List<string>();
        foreach (var entry in entries)
        {
            // The original wins whenever it is still there: the stage cache stamps a path, size and
            // write time, so preferring the copy would invalidate every downstream stage for nothing.
            if (!string.IsNullOrWhiteSpace(entry.OriginalPath) && File.Exists(entry.OriginalPath))
                continue;

            var archived = Path.GetFullPath(Path.Combine(provenanceDir, entry.ArchivedPath));
            if (!File.Exists(archived))
                continue;

            switch (entry.ConfigKey)
            {
                case "parsimony.fasta_path":
                    config.Parsimony.FastaPath = archived;
                    break;
                case "protein_rollup.ibaq.fasta_path":
                    config.ProteinRollup.Ibaq.FastaPath = archived;
                    break;
                default:
                    continue;
            }
            redirected.Add(entry.ConfigKey);
        }
        return redirected;
    }

    private static string UniqueTarget(string outputDir, FileInfo source)
    {
        var dir = Path.Combine(outputDir, DirectoryName);
        var stem = Path.GetFileNameWithoutExtension(source.Name);
        var ext = Path.GetExtension(source.Name);

        var candidate = Path.Combine(dir, source.Name);
        // A different database that happens to share a file name must not overwrite the first one.
        for (var i = 2; File.Exists(candidate) && new FileInfo(candidate).Length != source.Length; i++)
            candidate = Path.Combine(dir, $"{stem}_{i}{ext}");
        return candidate;
    }

    private static bool IsInside(string path, string directory)
    {
        var root = Path.GetFullPath(directory).TrimEnd(Path.DirectorySeparatorChar)
            + Path.DirectorySeparatorChar;
        return Path.GetFullPath(path).StartsWith(root, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>Forward slashes, so the recorded path reads the same on either platform.</summary>
    private static string Relative(string path, string outputDir) =>
        Path.GetRelativePath(outputDir, path).Replace('\\', '/');
}
