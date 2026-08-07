#nullable enable

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Threading;
using SkylinePrism.Core.Qc;

namespace SkylinePrism.Skyline;

/// <summary>
/// Gets a DIA acquisition's REAL isolation windows by having Skyline read them out of a raw data file -
/// the only place they exist for a "Results only" document, and a file format PRISM cannot open itself.
/// </summary>
/// <remarks>
/// <para>This is the command-line form of Transition Settings &gt; Full-Scan &gt; Isolation scheme &gt;
/// Add &gt; <i>Import from a data file</i>. The flag accepts a data-file path in place of a scheme name:
/// <code>--full-scan-isolation-scheme=path\to\run.raw</code></para>
///
/// <para><b>Nothing here touches the user's document.</b> The import runs against a THROWAWAY document
/// created with <c>--new</c> in a temp directory, whose only purpose is to be saved and parsed for its
/// <c>&lt;isolation_scheme&gt;</c> element; it is deleted afterwards. Pointing the flag at the user's own
/// document would rewrite its Full-Scan settings and dirty it, which PRISM must never do.</para>
///
/// <para>Unlike most SkylineCmd usage in PRISM, these flags go in ONE invocation: they are mutually
/// dependent (a new document, then the acquisition method, then the scheme, then save), and each
/// invocation is a separate process with no shared document state.</para>
///
/// <para>Measured: 167 windows read from a 5.2 GB Thermo .raw on a network share in ~10 s - Skyline reads
/// scan headers, not the whole file.</para>
/// </remarks>
public static class SkylineIsolationImporter
{
    /// <summary>
    /// Override for how long to wait on Skyline here, in seconds.
    /// </summary>
    public const string TimeoutEnvVar = "PRISM_ISOLATION_TIMEOUT_SEC";

    private static readonly TimeSpan DefaultTimeout = TimeSpan.FromMinutes(5);

    /// <summary>
    /// How long to let Skyline read the data file before giving up on it.
    /// <para>
    /// This step is an ENRICHMENT - without it the density map falls back to uniform bins and every
    /// other part of the run is unaffected - so it must never be able to hold a run up indefinitely.
    /// It normally takes ~10 s (167 windows from a 5.2 GB Thermo .raw on a share), but a file that
    /// has moved behind a slow or half-mounted link can leave Skyline reading with nothing to print,
    /// and there is no output to time out on. Five minutes is roughly 30x the expected time.
    /// </para>
    /// </summary>
    internal static TimeSpan Timeout
    {
        get
        {
            var raw = Environment.GetEnvironmentVariable(TimeoutEnvVar);
            if (double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out var seconds)
                && seconds > 0)
                return TimeSpan.FromSeconds(seconds);
            return DefaultTimeout;
        }
    }

    /// <summary>
    /// Import the isolation scheme from <paramref name="dataFilePath"/> (.raw file or .d folder, anything
    /// Skyline can open). Returns null when the file is unreachable, Skyline reports an error, or the
    /// result has no windows - this is a best-effort enrichment, never a reason to fail a run.
    /// </summary>
    public static IsolationScheme? ImportFromDataFile(
        string? dataFilePath, ISkylineCommandRunner runner, Action<string> log,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(dataFilePath))
            return null;
        // A data file may be a file (.raw, .mzML) or a directory (.d) - accept either.
        if (!File.Exists(dataFilePath) && !Directory.Exists(dataFilePath))
        {
            log($"Cannot read isolation windows: the data file is not reachable ({dataFilePath}).");
            return null;
        }

        var tempDir = Path.Combine(Path.GetTempPath(), "prism-isolation-" + Guid.NewGuid().ToString("N"));
        var tempDoc = Path.Combine(tempDir, "isolation-probe.sky");
        try
        {
            Directory.CreateDirectory(tempDir);
            log($"Reading isolation windows from {Path.GetFileName(dataFilePath)} via Skyline...");
            runner.Run(
                new[]
                {
                    "--new=" + tempDoc,
                    "--overwrite",
                    "--full-scan-acquisition-method=DIA",
                    "--full-scan-isolation-scheme=" + dataFilePath,
                    "--save",
                },
                // Skyline logs one line per imported window; at 167 windows that would bury the run log,
                // so only the failures are surfaced.
                message =>
                {
                    if (!message.Contains("Prespecified isolation windows", StringComparison.Ordinal))
                        log(message);
                },
                cancellationToken,
                Timeout);

            var scheme = IsolationScheme.Parse(SkyDocumentInfo.ReadIsolationSchemeXml(tempDoc));
            if (scheme is null || !scheme.HasWindows)
            {
                log($"Skyline read no isolation windows from {Path.GetFileName(dataFilePath)} "
                    + "(not a DIA acquisition?).");
                return null;
            }

            // Name it after the data file so the UI shows where the windows came from. Skyline already
            // names the imported scheme after the file, but not every version is guaranteed to.
            var named = new IsolationScheme(Path.GetFileNameWithoutExtension(dataFilePath), scheme.Windows);
            log($"Isolation windows read from the data file: {named.Describe()}.");
            return named;
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (TimeoutException ex)
        {
            // Deliberately not fatal: the map falls back to uniform bins and the rest of the run is
            // unaffected. Say what was lost and how to get it, rather than only that time ran out.
            log($"{ex.Message} The density map will use uniform bins instead of this run's real "
                + "isolation windows; everything else is unaffected. This usually means the data file "
                + $"is slow to reach ({dataFilePath}) - raise {TimeoutEnvVar} (seconds) to wait longer.");
            return null;
        }
        catch (Exception ex)
        {
            log($"Could not read isolation windows from {Path.GetFileName(dataFilePath)}: {ex.Message}");
            return null;
        }
        finally
        {
            try
            {
                if (Directory.Exists(tempDir))
                    Directory.Delete(tempDir, recursive: true);
            }
            catch (IOException)
            {
                // A leftover temp probe document is harmless.
            }
        }
    }

    /// <summary>
    /// Pick a data file to read the windows from, given the paths a document recorded. Recorded paths go
    /// stale (data gets archived or moved with the document), so a path that no longer resolves is retried
    /// beside the document itself before giving up - that alone rescued a real cohort whose .sky pointed
    /// at a mapped drive the files had since left.
    /// </summary>
    public static string? ResolveDataFile(IEnumerable<string> recordedPaths, string? documentPath)
    {
        var documentDir = string.IsNullOrWhiteSpace(documentPath)
            ? null
            : Path.GetDirectoryName(Path.GetFullPath(documentPath));

        string? firstCandidate = null;
        foreach (var recorded in recordedPaths)
        {
            if (string.IsNullOrWhiteSpace(recorded))
                continue;
            // Skyline can qualify a path with a sample index/name ("file.wiff|sample|1"); the file is the
            // part before the first '|'.
            var path = recorded.Split('|')[0];
            firstCandidate ??= path;
            if (File.Exists(path) || Directory.Exists(path))
                return path;

            if (documentDir is null)
                continue;
            var beside = Path.Combine(documentDir, Path.GetFileName(path));
            if (File.Exists(beside) || Directory.Exists(beside))
                return beside;
        }
        return null; // nothing reachable; the caller falls back to asking the user
    }
}
