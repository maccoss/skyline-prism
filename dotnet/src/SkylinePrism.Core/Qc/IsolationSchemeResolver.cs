using System;
using System.IO;
using System.Linq;
using System.Threading;
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
        string outputDir, string? rawDir, Action<string>? log = null, string? named = null,
        CancellationToken ct = default)
    {
        var cached = FromCache(outputDir, named, log);
        if (cached is not null)
            return cached;
        if (string.IsNullOrWhiteSpace(rawDir))
            return null;

        log?.Invoke(
            "No isolation scheme with windows was cached, which is normal for a DIA analysis document.");
        return FromData(outputDir, rawDir!, log, ct);
    }

    private static IsolationScheme? FromCache(string outputDir, string? named, Action<string>? log)
    {
        var path = Path.Combine(outputDir, IsolationSchemeCatalog.FileName);
        var catalog = IsolationSchemeCatalog.Load(path);
        var usable = catalog?.UsableSchemes;
        if (catalog is null || usable is null || usable.Count == 0)
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
            // A cohort can end up with both a scheme Skyline declared and one PRISM read out of the
            // data - the same acquisition under two names, which deduplication on the layout cannot
            // collapse. The measured one is the acquisition's own answer, so it wins rather than the
            // pair being reported as an ambiguity the user has to resolve by hand.
            var measured = usable.Where(catalog.IsMeasured).ToList();
            if (measured.Count == 1)
            {
                log?.Invoke(
                    $"Isolation scheme: {measured[0].Describe()} (measured from the data; "
                    + $"{usable.Count - 1} other cached scheme(s) not used).");
                return measured[0];
            }

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
    public static IsolationScheme? FromData(
        string outputDir, string rawDir, Action<string>? log, CancellationToken ct = default)
    {
        var files = ReplicateDataFiles.Enumerate(rawDir);
        if (files.Count == 0)
        {
            log?.Invoke($"No instrument data files in {rawDir}.");
            return null;
        }

        var first = files[0];
        log?.Invoke($"Reading the acquisition's isolation windows from {Path.GetFileName(first)}.");

        // The windows only, never a full signal read: they are scan headers in the first two cycles,
        // so this costs the file open and little else. Asking Read() for them measured the whole run
        // to use one field of the answer.
        var windows = IsolationWindowProbe.Read(first, log, ct);
        if (windows.Count == 0)
        {
            log?.Invoke(
                "That file reported no repeating isolation windows, so there is no scheme to account "
                + "against. A DDA acquisition has one window per spectrum and is not supported here.");
            return null;
        }

        var name = $"Imported from {Path.GetFileNameWithoutExtension(first)}";
        var scheme = new IsolationScheme(name, windows);
        log?.Invoke($"Isolation scheme: {scheme.Describe()}");
        Record(outputDir, scheme, first, log);
        return scheme;
    }

    /// <summary>
    /// Write a measured scheme down twice, in the two places a later reader will look.
    /// </summary>
    /// <remarks>
    /// The data files are the only other copy of this, and they are the first thing to move off a
    /// share. <c>isolation_schemes.xml</c> is what the density picker reloads, so it holds the window
    /// edges; <c>parameters.json</c> is what gets archived with a result, so it holds the same thing
    /// in the file people keep. Both are written beside the outputs and neither is fatal to lose.
    /// </remarks>
    private static void Record(
        string outputDir, IsolationScheme scheme, string dataFile, Action<string>? log)
    {
        try
        {
            var path = Path.Combine(outputDir, IsolationSchemeCatalog.FileName);
            var catalog = IsolationSchemeCatalog.Load(path) ?? new IsolationSchemeCatalog();
            // The full path, because this is provenance: a relative or mixed-separator form
            // records where the reader happened to be standing rather than where the file is.
            catalog.AddMeasuredScheme(scheme, Path.GetFullPath(dataFile));
            catalog.Save(path);
            var alsoJson = Pipeline.Provenance.RecordIsolationSchemes(outputDir, catalog);
            log?.Invoke(
                $"Cached it in {IsolationSchemeCatalog.FileName}"
                + (alsoJson ? $" and recorded it in {Pipeline.Provenance.FileName}." : ".")
                + " The windows survive there if the data files do not.");
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            // Not fatal: the scheme is in hand, and re-reading one file next time costs seconds.
            log?.Invoke($"Could not record the scheme: {ex.Message}");
        }
    }
}
