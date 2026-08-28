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
/// <item><b>Scheme library</b> - every scheme with real windows this run saw, gathered from the input
/// documents (see <see cref="Library"/>, which explains why Skyline's own saved scheme list is
/// deliberately NOT collected). When a document defines no windows, these are what the picker can
/// offer, since a cohort's plates are usually the same acquisition.</item>
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
    /// Every scheme with real windows this run learned about, so reopening the output directory can offer
    /// them again with no Skyline running. Populated from the input documents (see
    /// <see cref="AddDocumentScheme"/>) and refilled from the saved file by <see cref="Load"/>.
    /// <para>
    /// Skyline's saved isolation-scheme list is deliberately NOT collected here. Those are generic
    /// templates (SWATH (25 m/z), SWATH (VW 64), ...) unrelated to how any given data was acquired, and
    /// offering them invites picking one: binning a 3.0014 Th forbidden-zone acquisition on a 25 Th
    /// SWATH grid gives a map that looks plausible and is wrong. The acquisition's own windows come
    /// from its data file; where they cannot be read, labeled uniform bins are the honest fallback.
    /// </para>
    /// </summary>
/// <para>
    /// Read-only on purpose: every entry has to arrive through <see cref="AddDocumentScheme"/> (or
    /// <see cref="Load"/>), which is what keeps a library entry paired with the batch it came from. A
    /// publicly mutable list let a caller add a scheme with no such pairing, which is the state the
    /// removal of the inclusion-list loader was meant to make impossible.
    /// </para>
    /// </summary>
    public IReadOnlyList<IsolationScheme> Library => _library;

    private readonly List<IsolationScheme> _library = new();

    /// <summary>
    /// Library schemes that actually define windows, deduplicated, alphabetical - what the picker offers.
    /// <para>
    /// Deduplicated by name AND window layout, not by name alone. Two plates can legitimately name their
    /// schemes the same thing and mean different windows (a re-acquisition with a wider range, or the
    /// scheme edited between plates); collapsing those on the name would drop a layout the user might
    /// need to pick. <see cref="IsolationScheme.LayoutKey"/> is the tie-breaker.
    /// </para>
    /// </summary>
    public IReadOnlyList<IsolationScheme> UsableSchemes => _library
        .Where(s => s.HasWindows)
        .GroupBy(s => (s.Name.ToLowerInvariant(), s.LayoutKey))
        .Select(g => g.First())
        .OrderBy(s => s.Name, StringComparer.OrdinalIgnoreCase)
        .ThenBy(s => s.Windows.Count)
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
            _library.Add(scheme);
    }

    public void SetAcquisition(string batchLabel, string? acquisitionMethod)
    {
        if (!string.IsNullOrWhiteSpace(acquisitionMethod))
            AcquisitionByBatch[batchLabel] = acquisitionMethod!;
    }

    public bool IsEmpty => ByBatch.Count == 0 && _library.Count == 0 && AcquisitionByBatch.Count == 0;

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
            {
                element.Add(new XAttribute("scheme", scheme.Name));
                // The batch's OWN windows, inline. Referencing the library by name was wrong whenever two
                // documents named their schemes the same thing and meant different windows: the library
                // keeps one entry per name, so the other batch came back binned on a grid it was never
                // acquired with - a map that looks plausible and is not this batch's. The name attribute
                // stays for files written before this, and for a window-less scheme it is all there is.
                if (scheme.HasWindows)
                    element.Add(SchemeElement(scheme));
            }
            if (AcquisitionByBatch.TryGetValue(batch, out var method))
                element.Add(new XAttribute("acquisition", method));
            root.Add(element);
        }
        // The library, for the picker: deduplicated the same way UsableSchemes is, so a same-named
        // scheme with different windows is kept rather than silently dropped on the way to disk.
        foreach (var scheme in UsableSchemes)
            root.Add(SchemeElement(scheme));
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        new XDocument(root).Save(path);
    }

    /// <summary>One scheme as an &lt;IsolationScheme&gt; element, windows and firing intervals included.</summary>
    private static XElement SchemeElement(IsolationScheme scheme) => new(
        "IsolationScheme",
        new XAttribute("name", scheme.Name),
        scheme.Windows.Select(w =>
        {
            var element = new XElement("isolation_window",
                new XAttribute("start", Inv(w.Start)),
                new XAttribute("end", Inv(w.End)),
                new XAttribute("margin", Inv(w.Margin)));
            // A scheduled window carries the interval it fires in. Skyline's own schema has no such
            // attributes - they are PRISM's extension and are simply absent for plain DIA, where every
            // window is on for the whole gradient.
            if (w.IsScheduled)
            {
                element.Add(new XAttribute("rt_start", Inv(w.RtStart)));
                element.Add(new XAttribute("rt_stop", Inv(w.RtStop)));
            }
            return element;
        }));

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
                    catalog._library.Add(scheme);
            }
            foreach (var element in root.Elements()
                         .Where(e => string.Equals(e.Name.LocalName, "document", StringComparison.OrdinalIgnoreCase)))
            {
                var batch = element.Attribute("batch")?.Value;
                if (string.IsNullOrWhiteSpace(batch))
                    continue;
                catalog.SetAcquisition(batch!, element.Attribute("acquisition")?.Value);

                // This batch's own windows, written inline since the name alone cannot identify them.
                var inline = element.Elements()
                    .FirstOrDefault(e => string.Equals(
                        e.Name.LocalName, "IsolationScheme", StringComparison.OrdinalIgnoreCase));
                if (inline is not null && IsolationScheme.Parse(inline.ToString()) is { } own)
                {
                    catalog.ByBatch[batch!] = own;
                    if (own.HasWindows && !catalog._library.Any(s => s.LayoutKey == own.LayoutKey
                            && string.Equals(s.Name, own.Name, StringComparison.OrdinalIgnoreCase)))
                        catalog._library.Add(own);
                    continue;
                }

                var name = element.Attribute("scheme")?.Value;
                if (string.IsNullOrWhiteSpace(name))
                    continue;
                // No inline windows: a file written before they were, or a window-less scheme. Fall back
                // to the library by name, which is right whenever the names do not collide.
                var known = catalog._library.FirstOrDefault(
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
