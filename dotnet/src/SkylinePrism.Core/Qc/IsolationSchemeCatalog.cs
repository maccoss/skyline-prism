using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Xml.Linq;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// What a run learned about isolation windows, saved next to the run's outputs so the Spectrum density
/// tab can bin on real windows later - when the tool is reopened on an old output directory, with no
/// Skyline running.
/// </summary>
/// <remarks>
/// Two different things are recorded, because they answer different questions:
/// <list type="bullet">
/// <item><b>Per-batch document scheme</b> - what each input document's Full-Scan settings say. Usually
/// "Results only" (no windows), which is exactly why the second list is needed.</item>
/// <item><b>Scheme library</b> - every isolation scheme saved in the user's Skyline. When the document
/// defines no windows, these are the real layouts the acquisition could have used, and the user picks
/// the one that was.</item>
/// </list>
/// </remarks>
public sealed class IsolationSchemeCatalog
{
    public const string FileName = "isolation_schemes.xml";

    /// <summary>Batch label -> the scheme named by that input's document (may have no windows).</summary>
    public Dictionary<string, IsolationScheme> ByBatch { get; } = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Batch label -> the document's Full-Scan acquisition method (DIA / PRM / DDA / ...). Recorded
    /// because it decides what the density map MEANS: only DIA co-fragments many precursors per spectrum,
    /// so only there is a cell's count "how crowded was that spectrum".
    /// </summary>
    public Dictionary<string, string> AcquisitionByBatch { get; } = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>The acquisition method recorded for a batch, or null if unknown.</summary>
    public string? AcquisitionFor(string batchLabel) =>
        AcquisitionByBatch.TryGetValue(batchLabel, out var m) ? m : null;

    /// <summary>False only when we know the batch was NOT DIA; unknown counts as "do not warn".</summary>
    public bool IsNonDia(string batchLabel) =>
        AcquisitionFor(batchLabel) is { } m && !string.Equals(m, "DIA", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Extra schemes the USER supplied - currently only a Thermo inclusion list loaded for a scheduled
    /// acquisition, whose windows exist nowhere else.
    /// <para>
    /// Skyline's saved isolation-scheme list is deliberately NOT collected here. Those are generic
    /// templates (SWATH (25 m/z), SWATH (VW 64), ...) unrelated to how any given data was acquired, and
    /// offering them invites picking one: binning a 3.0014 Th forbidden-zone acquisition on a 25 Th
    /// SWATH grid gives a map that looks plausible and is wrong. The acquisition's own windows come
    /// from its data file; where they cannot be read, labelled uniform bins are the honest fallback.
    /// </para>
    /// </summary>
    public List<IsolationScheme> Library { get; } = new();

    /// <summary>Library schemes that actually define windows, deduplicated by name, alphabetical.</summary>
    public IReadOnlyList<IsolationScheme> UsableSchemes => Library
        .Where(s => s.HasWindows)
        .GroupBy(s => s.Name, StringComparer.OrdinalIgnoreCase)
        .Select(g => g.First())
        .OrderBy(s => s.Name, StringComparer.OrdinalIgnoreCase)
        .ToList();

    /// <summary>
    /// The scheme to use for a batch without asking: the document's own, but only when it defines
    /// windows. "Results only" deliberately returns null - the caller must then ask the user.
    /// </summary>
    public IsolationScheme? DocumentSchemeFor(string batchLabel) =>
        ByBatch.TryGetValue(batchLabel, out var s) && s.HasWindows ? s : null;

    /// <summary>The document's scheme NAME for a batch even when it has no windows (for the UI to explain).</summary>
    public string? DocumentSchemeNameFor(string batchLabel) =>
        ByBatch.TryGetValue(batchLabel, out var s) ? s.Name : null;

    public void AddDocumentScheme(string batchLabel, IsolationScheme scheme)
    {
        ByBatch[batchLabel] = scheme;
        if (scheme.HasWindows)
            Library.Add(scheme);
    }

    public void SetAcquisition(string batchLabel, string? acquisitionMethod)
    {
        if (!string.IsNullOrWhiteSpace(acquisitionMethod))
            AcquisitionByBatch[batchLabel] = acquisitionMethod!;
    }

    public void AddLibraryScheme(IsolationScheme scheme) => Library.Add(scheme);

    public bool IsEmpty => ByBatch.Count == 0 && Library.Count == 0 && AcquisitionByBatch.Count == 0;

    public void Save(string path)
    {
        var root = new XElement("prism_isolation_schemes");
        var batches = ByBatch.Keys.Concat(AcquisitionByBatch.Keys)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(b => b, StringComparer.OrdinalIgnoreCase);
        foreach (var batch in batches)
        {
            var element = new XElement("document", new XAttribute("batch", batch));
            if (ByBatch.TryGetValue(batch, out var scheme))
                element.Add(new XAttribute("scheme", scheme.Name));
            if (AcquisitionByBatch.TryGetValue(batch, out var method))
                element.Add(new XAttribute("acquisition", method));
            root.Add(element);
        }
        foreach (var scheme in Library
                     .GroupBy(s => s.Name, StringComparer.OrdinalIgnoreCase)
                     .Select(g => g.First())
                     .OrderBy(s => s.Name, StringComparer.OrdinalIgnoreCase))
        {
            root.Add(new XElement("IsolationScheme",
                new XAttribute("name", scheme.Name),
                scheme.Windows.Select(w =>
                {
                    var element = new XElement("isolation_window",
                        new XAttribute("start", Inv(w.Start)),
                        new XAttribute("end", Inv(w.End)),
                        new XAttribute("margin", Inv(w.Margin)));
                    // Scheduled (PRM/MTM) windows carry their firing interval. Skyline's own schema has no
                    // such attributes - they are PRISM's extension and are simply absent for DIA.
                    if (w.IsScheduled)
                    {
                        element.Add(new XAttribute("rt_start", Inv(w.RtStart)));
                        element.Add(new XAttribute("rt_stop", Inv(w.RtStop)));
                    }
                    return element;
                })));
        }
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        new XDocument(root).Save(path);
    }

    /// <summary>Load a saved catalog, or null when the file is absent or unreadable.</summary>
    public static IsolationSchemeCatalog? Load(string path)
    {
        if (!File.Exists(path))
            return null;
        try
        {
            var root = XDocument.Load(path).Root;
            if (root is null)
                return null;

            var catalog = new IsolationSchemeCatalog();
            foreach (var element in root.Elements()
                         .Where(e => string.Equals(e.Name.LocalName, "IsolationScheme", StringComparison.OrdinalIgnoreCase)))
            {
                var scheme = IsolationScheme.Parse(element.ToString());
                if (scheme is not null)
                    catalog.Library.Add(scheme);
            }
            foreach (var element in root.Elements()
                         .Where(e => string.Equals(e.Name.LocalName, "document", StringComparison.OrdinalIgnoreCase)))
            {
                var batch = element.Attribute("batch")?.Value;
                if (string.IsNullOrWhiteSpace(batch))
                    continue;
                catalog.SetAcquisition(batch!, element.Attribute("acquisition")?.Value);

                var name = element.Attribute("scheme")?.Value;
                if (string.IsNullOrWhiteSpace(name))
                    continue;
                // Re-attach the windows from the library when the document's scheme is one of them.
                var known = catalog.Library.FirstOrDefault(
                    s => string.Equals(s.Name, name, StringComparison.OrdinalIgnoreCase));
                catalog.ByBatch[batch!] = known ?? new IsolationScheme(name!, Array.Empty<IsolationWindow>());
            }
            return catalog;
        }
        catch (Exception ex) when (ex is IOException or System.Xml.XmlException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    private static string Inv(double v) =>
        v.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
}
