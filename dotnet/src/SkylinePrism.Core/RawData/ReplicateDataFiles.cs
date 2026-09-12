using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SkylinePrism.Core.RawData;

/// <summary>
/// Matches a cohort's replicates to the instrument data files they were acquired into.
/// </summary>
/// <remarks>
/// <para>Needed because the two halves of the MS2 signal fraction come from different places: the
/// numerator from <c>merged_data/</c>, keyed by PRISM sample id, and the denominator from a data
/// file, named by whatever the acquisition software called it. Nothing records the correspondence -
/// Skyline stores the replicate NAME, and a document's recorded data paths point at the acquisition
/// machine, which is where <c>SkylineIsolationImporter.ResolveDataFile</c> stops being useful once
/// the files have been copied somewhere else.</para>
///
/// <para>So the match is on the file stem, and it is deliberately not equality: acquisition software
/// routinely prefixes the run. On the cohort this was written against, replicate
/// <c>FLARE-001-1-B1-013</c> was acquired into <c>2026-extended-FLARE-001-1-B1-013.raw</c>.</para>
/// </remarks>
public static class ReplicateDataFiles
{
    /// <summary>Separator PRISM joins a replicate and its batch with to make a sample id.</summary>
    public const string SampleIdSeparator = "__@__";

    /// <summary>
    /// Extensions worth offering to a reader.
    ///
    /// <para>This must stay a SUPERSET of what the readers' own <c>CanRead</c> accepts. A file whose
    /// extension is missing here is never enumerated, so it is never offered, and the replicate
    /// reports a missing denominator while the reader would have opened it happily. Matched
    /// case-insensitively (see <see cref="Enumerate"/>).</para>
    /// </summary>
    public static readonly string[] DataFileExtensions =
    {
        ".raw", ".mzML", ".mzXML", ".mzML.gz", ".mz5", ".d", ".wiff", ".wiff2", ".lcd", ".yep",
        ".baf",
    };

    /// <summary>The replicate half of a PRISM sample id (<c>replicate__@__batch</c>).</summary>
    public static string ReplicateOf(string sampleId)
    {
        if (string.IsNullOrEmpty(sampleId))
            return sampleId ?? string.Empty;
        var i = sampleId.IndexOf(SampleIdSeparator, StringComparison.Ordinal);
        return i < 0 ? sampleId : sampleId[..i];
    }

    /// <summary>
    /// The data file for a replicate, or null when none of <paramref name="files"/> matches.
    /// </summary>
    /// <remarks>
    /// An exact stem match wins outright. Otherwise the longest stem ENDING in the replicate name
    /// wins, which is what handles the acquisition prefix - and the "longest" is load-bearing rather
    /// than arbitrary: where one replicate name is a suffix of another (<c>..._B1_1</c> inside
    /// <c>..._B1_11</c>), the shorter would otherwise steal the longer one's file.
    ///
    /// <para>Matching a SUFFIX and not a substring is the other half of that. A substring match would
    /// let a replicate collide with any file that merely contains its name somewhere in the middle,
    /// and prefixes are the pattern acquisition software actually produces.</para>
    /// </remarks>
    public static string? Resolve(string sampleId, IReadOnlyList<string> files)
    {
        if (files is null || files.Count == 0)
            return null;
        var replicate = ReplicateOf(sampleId);
        if (string.IsNullOrEmpty(replicate))
            return null;

        string? best = null;
        var bestLength = -1;
        foreach (var file in files)
        {
            var stem = Path.GetFileNameWithoutExtension(file);
            if (string.Equals(stem, replicate, StringComparison.OrdinalIgnoreCase))
                return file;
            if (stem.EndsWith(replicate, StringComparison.OrdinalIgnoreCase) && stem.Length > bestLength)
            {
                best = file;
                bestLength = stem.Length;
            }
        }
        return best;
    }

    /// <summary>
    /// Every data file under <paramref name="directory"/>, in a stable order. Not recursive: a raw
    /// directory is normally flat, and recursing into one holding Bruker <c>.d</c> folders would
    /// enumerate their contents rather than the acquisitions.
    /// </summary>
    public static List<string> Enumerate(string directory)
    {
        if (string.IsNullOrWhiteSpace(directory) || !Directory.Exists(directory))
            return new List<string>();

        // Enumerated once and filtered in code rather than with one glob per extension: a glob
        // pattern is matched case-INSENSITIVELY on Windows and case-SENSITIVELY on Linux, so
        // "*.raw" found a .RAW file on one platform and not the other. Directories are included as
        // well as files, because a Bruker .d and an Agilent .d are acquisitions that happen to be
        // folders and a reader opens them by directory path.
        var found = new List<string>();
        foreach (var entry in Directory.EnumerateFileSystemEntries(directory))
        {
            var name = entry.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            if (DataFileExtensions.Any(e => name.EndsWith(e, StringComparison.OrdinalIgnoreCase)))
                found.Add(entry);
        }
        return found
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    /// <param name="Matched">Sample id to data file, one file per sample and one sample per file.</param>
    /// <param name="Unmatched">Samples no file matched - a missing denominator the plot must name.</param>
    /// <param name="Ambiguous">
    /// Samples that matched a file another sample also matched. None of them is assigned it.
    /// </param>
    public sealed record Resolution(
        Dictionary<string, string> Matched, List<string> Unmatched, List<string> Ambiguous);

    /// <summary>
    /// Resolve every sample against a directory of data files, one-to-one, reporting what did not
    /// match rather than dropping it silently.
    ///
    /// <para><b>One file cannot serve two samples.</b> Reference and QC injections are normally named
    /// identically in every plate's document, so <c>QC_1__@__plateA</c> and <c>QC_1__@__plateB</c>
    /// both match <c>QC_1.raw</c> - but those are two injections, run on two plates, and whichever
    /// file is present is at most one of them. Assigning it to both gives two replicates the same
    /// denominator; assigning it to whichever came first makes the answer depend on dictionary order.
    /// Both are reported as <see cref="Resolution.Ambiguous"/> instead, so the plot omits their
    /// acquired bars and says why. This is the collision CLAUDE.md warns about for metadata, in the
    /// one place it reaches instrument files.</para>
    /// </summary>
    public static Resolution ResolveAll(
        IEnumerable<string> sampleIds, IReadOnlyList<string> files)
    {
        // First pass: what each sample would take, and how many samples want each file.
        var wanted = new Dictionary<string, string>(StringComparer.Ordinal);
        var claimants = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase);
        var unmatched = new List<string>();
        foreach (var sampleId in sampleIds)
        {
            var path = Resolve(sampleId, files);
            if (path is null)
            {
                unmatched.Add(sampleId);
                continue;
            }
            wanted[sampleId] = path;
            if (!claimants.TryGetValue(path, out var list))
                claimants[path] = list = new List<string>();
            list.Add(sampleId);
        }

        var matched = new Dictionary<string, string>(StringComparer.Ordinal);
        var ambiguous = new List<string>();
        foreach (var (sampleId, path) in wanted)
        {
            if (claimants[path].Count == 1)
                matched[sampleId] = path;
            else
                ambiguous.Add(sampleId);
        }
        ambiguous.Sort(StringComparer.Ordinal);
        return new Resolution(matched, unmatched, ambiguous);
    }
}
