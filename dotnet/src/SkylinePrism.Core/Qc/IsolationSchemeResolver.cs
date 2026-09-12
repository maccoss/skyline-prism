using System;
using System.IO;
using System.Linq;
using SkylinePrism.Core.RawData;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// The isolation scheme ion accounting will account against: from the output directory's cache when
/// one is there, otherwise read from a data file and cached.
/// </summary>
/// <remarks>
/// <para><b>Shared by the CLI and the tool</b>, which is the point. This logic used to live in
/// <c>Program.cs</c> against <c>Console</c>, so the GUI could not reach it - and the alternative to
/// moving it was a second copy that would drift. The one behaviour worth not duplicating is the
/// fallback: a DIA analysis document normally stores <c>&lt;isolation_scheme name="Results only"
/// /&gt;</c> with NO windows, so an empty cache is the ordinary case rather than an error, and the
/// windows come from a data file instead.</para>
/// </remarks>
public static class IsolationSchemeResolver
{
    /// <summary>
    /// Resolve a scheme, importing from <paramref name="rawDir"/> if the cache has none.
    /// </summary>
    /// <param name="named">
    /// Pick this scheme by name. Required when the cache holds several: fragments in different
    /// isolation windows never share signal, so guessing which scheme applies would change the
    /// answer silently.
    /// </param>
    /// <returns>Null when none could be resolved; the reason is logged.</returns>
    public static IsolationScheme? Resolve(
        string outputDir, string? rawDir, Action<string>? log = null, string? named = null)
    {
        var cached = FromCache(outputDir, named, log);
        if (cached is not null)
            return cached;

        return string.IsNullOrWhiteSpace(rawDir) ? null : FromData(outputDir, rawDir!, log);
    }

    private static IsolationScheme? FromCache(string outputDir, string? named, Action<string>? log)
    {
        var path = Path.Combine(outputDir, IsolationSchemeCatalog.FileName);
        var usable = IsolationSchemeCatalog.Load(path)?.UsableSchemes;
        if (usable is null || usable.Count == 0)
            return null;   // normal for a DIA document; the caller imports from data instead

        if (!string.IsNullOrWhiteSpace(named))
        {
            var match = usable.FirstOrDefault(
                s => string.Equals(s.Name, named, StringComparison.OrdinalIgnoreCase));
            if (match is null)
            {
                log?.Invoke(
                    $"No isolation scheme named '{named}'. Available: "
                    + string.Join(", ", usable.Select(s => s.Name)));
            }
            return match;
        }

        if (usable.Count > 1)
        {
            log?.Invoke(
                "More than one isolation scheme is cached, so one must be named: "
                + string.Join(", ", usable.Select(s => s.Name)));
            return null;
        }

        log?.Invoke($"Isolation scheme: {usable[0].Describe()}");
        return usable[0];
    }

    /// <summary>
    /// Read the windows from the first data file, and cache them.
    /// </summary>
    /// <remarks>
    /// Headers only - the windows are a property of the ACQUISITION METHOD, so one file describes
    /// every replicate of the cohort and there is no reason to decode a peak to find them.
    /// </remarks>
    private static IsolationScheme? FromData(string outputDir, string rawDir, Action<string>? log)
    {
        var files = ReplicateDataFiles.Enumerate(rawDir);
        if (files.Count == 0)
        {
            log?.Invoke($"No instrument data files in {rawDir}.");
            return null;
        }

        var first = files[0];
        log?.Invoke(
            "No isolation scheme with windows was cached, which is normal for a DIA analysis "
            + $"document. Reading the windows from {Path.GetFileName(first)}.");

        var record = Ms2SignalReaders.Read(first, log);
        if (record.IsolationWindows.Count == 0)
        {
            log?.Invoke(
                "That file reported no repeating isolation windows, so there is no scheme to account "
                + "against. A DDA acquisition has one window per spectrum and is not supported here.");
            return null;
        }

        var name = $"Imported from {Path.GetFileNameWithoutExtension(first)}";
        var scheme = new IsolationScheme(name, record.IsolationWindows);
        log?.Invoke($"Isolation scheme: {scheme.Describe()}");

        try
        {
            var path = Path.Combine(outputDir, IsolationSchemeCatalog.FileName);
            var catalog = IsolationSchemeCatalog.Load(path) ?? new IsolationSchemeCatalog();
            catalog.AddDocumentScheme(name, scheme);
            catalog.Save(path);
            log?.Invoke($"Cached it in {IsolationSchemeCatalog.FileName}.");
        }
        catch (IOException ex)
        {
            // Not fatal: the scheme is in hand, and re-reading one file next time costs seconds.
            log?.Invoke($"Could not cache the scheme: {ex.Message}");
        }

        return scheme;
    }
}
