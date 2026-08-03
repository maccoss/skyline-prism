#nullable enable

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SkylinePrism.Skyline;

/// <summary>
/// Finds the <c>SkylineCmd.exe</c> used to export reports from a document that is NOT open in a running
/// Skyline (see <see cref="HeadlessSkylineExporter"/>).
///
/// <para><b>The gotcha (field guide, section 12):</b> Skyline installs via ClickOnce into
/// <c>%LOCALAPPDATA%\Apps\2.0\**</c>, which holds two kinds of sibling folders. Only the <i>application</i>
/// folder - the one containing <c>Skyline.exe</c> / <c>Skyline-daily.exe</c> - has a working
/// <c>SkylineCmd.exe</c>; the copy in the sibling <c>…exe_…</c> folders fails at startup with
/// <i>"Unable to find Skyline.exe"</i>. So a candidate only counts when a Skyline executable sits
/// beside it.</para>
///
/// Search order: explicit path, then <c>PRISM_SKYLINECMD</c>, then the newest qualifying ClickOnce
/// application folder, then conventional Program Files installs, then PATH.
/// </summary>
public static class SkylineCmdLocator
{
    /// <summary>Environment variable that overrides discovery (full path to SkylineCmd.exe).</summary>
    public const string OverrideEnvVar = "PRISM_SKYLINECMD";

    private static readonly string[] SkylineExeNames = { "Skyline.exe", "Skyline-daily.exe" };

    /// <summary>A discovered SkylineCmd.exe, with the Skyline build that sits next to it.</summary>
    public sealed record Candidate(string CmdPath, string SkylineExePath, DateTime LastWriteUtc)
    {
        /// <summary>"Skyline-daily" / "Skyline" - which channel this build is.</summary>
        public string Channel => Path.GetFileNameWithoutExtension(SkylineExePath);

        public override string ToString() => $"{Channel} ({LastWriteUtc.ToLocalTime():yyyy-MM-dd}) - {CmdPath}";
    }

    /// <summary>
    /// Resolve the SkylineCmd.exe to use, or null when none is installed. <paramref name="explicitPath"/>
    /// (a user setting) wins when it exists.
    /// </summary>
    public static string? Find(string? explicitPath = null, Action<string>? log = null)
    {
        if (!string.IsNullOrWhiteSpace(explicitPath))
        {
            if (File.Exists(explicitPath))
                return Path.GetFullPath(explicitPath!);
            log?.Invoke($"Configured SkylineCmd path does not exist: {explicitPath}");
        }

        var env = Environment.GetEnvironmentVariable(OverrideEnvVar);
        if (!string.IsNullOrWhiteSpace(env) && File.Exists(env))
            return Path.GetFullPath(env!);

        var best = FindAll(log).FirstOrDefault();
        if (best is not null)
        {
            log?.Invoke($"Using SkylineCmd: {best}");
            return best.CmdPath;
        }

        // Last resort: on PATH (a manual/portable install).
        var onPath = SearchPath("SkylineCmd.exe");
        if (onPath is not null)
            log?.Invoke($"Using SkylineCmd from PATH: {onPath}");
        return onPath;
    }

    /// <summary>
    /// Every usable SkylineCmd.exe on this machine, newest first. Exposed so the UI can let the user pick
    /// when both Skyline and Skyline-daily are installed.
    /// </summary>
    public static IReadOnlyList<Candidate> FindAll(Action<string>? log = null)
    {
        var found = new Dictionary<string, Candidate>(StringComparer.OrdinalIgnoreCase);

        foreach (var dir in CandidateDirectories())
        {
            try
            {
                var cmd = Path.Combine(dir, "SkylineCmd.exe");
                if (!File.Exists(cmd))
                    continue;
                // The decisive check: a real Skyline executable must sit beside it.
                var exe = SkylineExeNames
                    .Select(n => Path.Combine(dir, n))
                    .FirstOrDefault(File.Exists);
                if (exe is null)
                    continue;
                if (!found.ContainsKey(cmd))
                    found[cmd] = new Candidate(cmd, exe, File.GetLastWriteTimeUtc(cmd));
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                // Unreadable ClickOnce folder; skip it.
            }
        }

        var result = found.Values.OrderByDescending(c => c.LastWriteUtc).ToList();
        if (result.Count == 0)
            log?.Invoke("No SkylineCmd.exe found (Skyline does not appear to be installed for this user).");
        return result;
    }

    // Directories worth probing: ClickOnce application folders + conventional installs.
    private static IEnumerable<string> CandidateDirectories()
    {
        var local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        var clickOnce = Path.Combine(local, "Apps", "2.0");
        if (Directory.Exists(clickOnce))
        {
            // The application folders are nested ~3 levels down (<hash>/<hash>/skyl..tion_<...>). Enumerating
            // the whole tree is slow and throws on the sibling data folders, so walk with a depth bound and
            // swallow per-directory errors.
            foreach (var dir in EnumerateDirectoriesSafe(clickOnce, maxDepth: 4))
                yield return dir;
        }

        foreach (var root in new[]
                 {
                     Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                     Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86),
                 })
        {
            if (string.IsNullOrEmpty(root) || !Directory.Exists(root))
                continue;
            foreach (var name in new[] { "Skyline", "Skyline-daily", "SkylineRunner" })
            {
                var dir = Path.Combine(root, name);
                if (Directory.Exists(dir))
                    yield return dir;
            }
        }
    }

    private static IEnumerable<string> EnumerateDirectoriesSafe(string root, int maxDepth)
    {
        var queue = new Queue<(string Dir, int Depth)>();
        queue.Enqueue((root, 0));
        while (queue.Count > 0)
        {
            var (dir, depth) = queue.Dequeue();
            yield return dir;
            if (depth >= maxDepth)
                continue;
            string[] children;
            try
            {
                children = Directory.GetDirectories(dir);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                continue;
            }
            foreach (var child in children)
                queue.Enqueue((child, depth + 1));
        }
    }

    private static string? SearchPath(string exeName)
    {
        var path = Environment.GetEnvironmentVariable("PATH");
        if (string.IsNullOrEmpty(path))
            return null;
        foreach (var dir in path!.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            try
            {
                var candidate = Path.Combine(dir.Trim(), exeName);
                if (File.Exists(candidate))
                    return candidate;
            }
            catch (ArgumentException)
            {
                // Malformed PATH entry; skip.
            }
        }
        return null;
    }
}
