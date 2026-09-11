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

    /// <summary>Extensions worth offering to a reader, in the order a directory is searched.</summary>
    public static readonly string[] DataFileExtensions =
    {
        ".raw", ".mzML", ".mzXML", ".d", ".wiff", ".wiff2",
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

        var found = new List<string>();
        foreach (var ext in DataFileExtensions)
        {
            // Directories as well as files: a Bruker .d and an Agilent .d are acquisitions that happen
            // to be folders, and a reader opens them by directory path.
            found.AddRange(Directory.EnumerateFileSystemEntries(directory, "*" + ext));
        }
        return found
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    /// <summary>
    /// Resolve every sample against a directory of data files, reporting the ones that did not match
    /// rather than dropping them silently - an unmatched replicate is a missing denominator, and the
    /// plot has to be able to say which.
    /// </summary>
    public static (Dictionary<string, string> Matched, List<string> Unmatched) ResolveAll(
        IEnumerable<string> sampleIds, IReadOnlyList<string> files)
    {
        var matched = new Dictionary<string, string>(StringComparer.Ordinal);
        var unmatched = new List<string>();
        foreach (var sampleId in sampleIds)
        {
            var path = Resolve(sampleId, files);
            if (path is null)
                unmatched.Add(sampleId);
            else
                matched[sampleId] = path;
        }
        return (matched, unmatched);
    }
}
