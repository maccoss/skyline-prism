using System.Collections.Generic;
using System.IO;

namespace SkylinePrism.Core.IO;

/// <summary>
/// Read a file without refusing to share it - the way every text input a user might also have open
/// (sample_metadata.csv, a clinical CSV, protein_groups.csv, a Skyline report) must be read.
/// </summary>
/// <remarks>
/// <para><see cref="File.ReadAllLines(string)"/> opens with <see cref="FileShare.Read"/>, which
/// demands that nobody else holds the file for writing. Excel does exactly that for every CSV it has
/// open, so the read fails with "being used by another process". In the Differential pane that
/// failure emptied every picker on a level switch - the run was fine, the user had only looked at
/// sample_metadata.csv in Excel. Readers that swallow the exception were worse: they returned no
/// metadata at all, and the QC pane silently lost its grouping columns.</para>
/// <para>Sharing <c>ReadWrite | Delete</c> also keeps PRISM from taking its own files hostage while a
/// run rewrites them (see <see cref="ParquetColumnIo"/>). The cost is that a read racing a write can
/// see a partial file, which every caller here already treats as unreadable input.</para>
/// </remarks>
public static class SharedFile
{
    /// <summary>Open <paramref name="path"/> for reading, sharing read, write and delete.</summary>
    public static FileStream OpenRead(string path) =>
        new(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);

    /// <summary>A text reader over <see cref="OpenRead"/>, detecting the encoding from a BOM as <see cref="File.OpenText"/> does.</summary>
    public static StreamReader OpenText(string path) => new(OpenRead(path), detectEncodingFromByteOrderMarks: true);

    /// <summary><see cref="File.ReadAllLines(string)"/>, without refusing to share the file.</summary>
    public static string[] ReadAllLines(string path)
    {
        using var reader = OpenText(path);
        var lines = new List<string>();
        while (reader.ReadLine() is { } line)
            lines.Add(line);
        return lines.ToArray();
    }

    /// <summary>
    /// Whether another program has <paramref name="path"/> open in a way that stops it being written -
    /// what Excel does to every CSV it shows. False for a file that does not exist, or that cannot be
    /// written for some other reason (read-only, no permission): that is not something closing a
    /// window would fix, so it is left to fail with its own message.
    /// </summary>
    /// <remarks>
    /// Probes by opening for WRITE without writing anything, which leaves the content and the
    /// modification time alone. Its own handle shares everything, so the probe cannot itself block a
    /// reader. Windows only in effect: elsewhere sharing is advisory, the probe always succeeds, and
    /// nothing is reported.
    /// </remarks>
    public static bool IsHeldOpen(string path)
    {
        try
        {
            using var probe = new FileStream(path, FileMode.Open, FileAccess.Write,
                FileShare.ReadWrite | FileShare.Delete);
            return false;
        }
        catch (FileNotFoundException)
        {
            return false;
        }
        catch (DirectoryNotFoundException)
        {
            return false;
        }
        catch (IOException ex) when (IsSharingViolation(ex))
        {
            return true;
        }
        catch (IOException)
        {
            return false;
        }
        catch (System.UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>
    /// Which of <paramref name="fileNames"/> in <paramref name="directory"/> another program has open,
    /// in the order given. Pass the files a run will write
    /// (<see cref="Pipeline.PrismPipeline.UserOpenedOutputs"/>), never a directory listing: a file the
    /// run only reads can be open in Excel without stopping anything.
    /// </summary>
    /// <remarks>
    /// Asked BEFORE a run, because the first write to one of them comes at Stage 5 - after the whole
    /// pipeline - so finding out from the failure costs the user the run. A name that does not exist
    /// yet is not held.
    /// </remarks>
    public static IReadOnlyList<string> HeldOpen(string directory, IEnumerable<string> fileNames)
    {
        var held = new List<string>();
        if (!Directory.Exists(directory))
            return held;
        foreach (var name in fileNames)
            if (IsHeldOpen(Path.Combine(directory, name)))
                held.Add(name);
        return held;
    }

    /// <summary>
    /// Whether <paramref name="ex"/>, or anything it wraps, is Windows refusing a file because another
    /// process has it open (ERROR_SHARING_VIOLATION or ERROR_LOCK_VIOLATION).
    /// </summary>
    public static bool IsSharingViolation(System.Exception? ex)
    {
        for (var e = ex; e is not null; e = e.InnerException)
        {
            if (e is IOException && (e.HResult & 0xFFFF) is 32 or 33)
                return true;
            if (e is System.AggregateException agg)
                foreach (var inner in agg.InnerExceptions)
                    if (IsSharingViolation(inner))
                        return true;
        }

        return false;
    }
}
