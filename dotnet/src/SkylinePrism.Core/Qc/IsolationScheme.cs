using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Xml.Linq;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// One isolation window: the precursor m/z range co-fragmented into a single MS/MS spectrum.
/// <see cref="Margin"/> is the edge Skyline excludes from chromatogram extraction; it is carried for
/// display but NOT used for membership, because everything between <see cref="Start"/> and
/// <see cref="End"/> is physically fragmented together whether or not Skyline quantified it there.
///
/// <para><b>Scheduled windows.</b> A DIA window is on for the whole gradient - its cycle repeats from
/// start to end - so <see cref="RtStart"/>/<see cref="RtStop"/> are NaN, meaning "always on". A targeted
/// method (PRM, or MTM where several co-eluting precursors share one window) fires each window only
/// during its scheduled interval, and those bounds carry it. This is Cadenza's <c>Slot</c>: an m/z range
/// crossed with an RT range. Without it, a scheduled window would be drawn across the whole gradient and
/// "never acquired here" would be indistinguishable from "acquired, nothing found".</para>
/// </summary>
public readonly record struct IsolationWindow(
    double Start, double End, double Margin = 0,
    double RtStart = double.NaN, double RtStop = double.NaN)
{
    public double Width => End - Start;
    public double Center => (Start + End) / 2;

    /// <summary>True when this window fires only during a scheduled retention-time interval.</summary>
    public bool IsScheduled => !double.IsNaN(RtStart) && !double.IsNaN(RtStop);

    /// <summary>Half-open so a precursor sitting exactly on a boundary lands in one window, not two.</summary>
    public bool Contains(double mz) => mz >= Start && mz < End;

    /// <summary>Was this window being acquired at <paramref name="rt"/>? Always true when unscheduled.</summary>
    public bool IsOnAt(double rt) => !IsScheduled || (rt >= RtStart && rt <= RtStop);

    /// <summary>
    /// Does this window cover <paramref name="mz"/> at any point in [<paramref name="rtStart"/>,
    /// <paramref name="rtStop"/>]? A precursor eluting outside a scheduled window's firing interval was
    /// never fragmented by it, however well its m/z matches.
    /// </summary>
    public bool Covers(double mz, double rtStart, double rtStop) =>
        Contains(mz) && (!IsScheduled || (rtStart <= RtStop && rtStop >= RtStart));
}

/// <summary>
/// A named Skyline isolation scheme - the actual window layout of a DIA acquisition.
/// </summary>
/// <remarks>
/// Parsed from the two XML spellings Skyline produces for the same thing:
/// <code>
/// GetSettingsListItem("IsolationSchemeList", name):
///   &lt;IsolationScheme name="SWATH (25 m/z)"&gt;
///     &lt;isolation_window start="400" end="424" margin="0.5" ce_range="5" /&gt; ...
///
/// inside a saved .sky (transition_settings/transition_full_scan):
///   &lt;isolation_scheme name="SWATH (25 m/z)" precursor_filter="..."&gt;
///     &lt;isolation_window start="400" end="424" margin="0.5" /&gt; ...
/// </code>
/// A document that takes its windows from the data files instead stores only
/// <c>&lt;isolation_scheme name="Results only" /&gt;</c> - no windows - which is the normal setting for a
/// DIA analysis document. That case parses to a scheme with an empty window list: named, but unusable
/// as a grid, and the caller has to get the layout elsewhere.
/// </remarks>
public sealed record IsolationScheme(string Name, IReadOnlyList<IsolationWindow> Windows)
{
    /// <summary>Skyline's name for "the windows are whatever the data files say".</summary>
    public const string ResultsOnlyName = "Results only";

    public bool HasWindows => Windows.Count > 0;
    public double MzLow => Windows.Min(w => w.Start);
    public double MzHigh => Windows.Max(w => w.End);

    /// <summary>True when any window fires only during a scheduled interval (a targeted method).</summary>
    public bool IsScheduled => Windows.Any(w => w.IsScheduled);

    /// <summary>
    /// Indices of every window containing <paramref name="mz"/>. Normally one; a staggered/overlapping
    /// scheme genuinely fragments a precursor in more than one window, and it belongs in each of them.
    /// Empty when the precursor falls outside the scheme (which is the signal that the scheme is wrong
    /// for this data - see <see cref="Coverage"/>).
    /// </summary>
    public IEnumerable<int> IndicesContaining(double mz)
    {
        for (var i = 0; i < Windows.Count; i++)
            if (Windows[i].Contains(mz))
                yield return i;
    }

    /// <summary>
    /// Indices of every window that fragmented a precursor of <paramref name="mz"/> eluting over
    /// [<paramref name="rtStart"/>, <paramref name="rtStop"/>]. Same as
    /// <see cref="IndicesContaining"/> for an always-on DIA scheme; for a scheduled method it also
    /// requires the window to have been firing while the peak eluted, which is what stops a PRM target
    /// from being credited to a same-m/z slot scheduled at a different time.
    /// </summary>
    public IEnumerable<int> IndicesCovering(double mz, double rtStart, double rtStop)
    {
        for (var i = 0; i < Windows.Count; i++)
            if (Windows[i].Covers(mz, rtStart, rtStop))
                yield return i;
    }

    /// <summary>
    /// Fraction of the given precursor m/z values that fall inside some window (1.0 = every one). The
    /// check that the chosen scheme actually matches the acquisition: pick the wrong one and precursors
    /// land outside its windows.
    /// </summary>
    public double Coverage(IReadOnlyList<double> precursorMz)
    {
        if (precursorMz.Count == 0 || !HasWindows)
            return 0;
        var inside = precursorMz.Count(mz => Windows.Any(w => w.Contains(mz)));
        return (double)inside / precursorMz.Count;
    }

    /// <summary>
    /// The built-in starting scheme's name, so the UI can mark it as the default.
    /// <para>
    /// The range in the name is NOMINAL, as it is in Skyline's own "SWATH (25 m/z)": the windows
    /// actually run 400.43 to 901.66, because 167 windows of ~3.0014 Th starting at a forbidden-zone
    /// edge do not land on a round number. Rounding the windows to make the label exact would defeat
    /// the point of the scheme. The picker shows the real extents next to the name via
    /// <see cref="Describe"/>, so nothing has to be inferred from the label.
    /// </para>
    /// </summary>
    public const string AstralDefaultName = "Astral 3 Th, 400-900 m/z";

    /// <summary>
    /// The scheme to start from when an acquisition's own windows cannot be read: a modern Astral-style
    /// narrow-window DIA cycle.
    ///
    /// <para><b>Why this and not Skyline's saved schemes.</b> Those are generic templates - SWATH
    /// (25 m/z), SWATH (VW 64) - designed for instruments and experiments that narrow-window DIA has
    /// moved on from. Binning a 3 Th acquisition on a 25 Th grid produces a map that looks plausible
    /// and is wrong, so offering them was worse than offering nothing.</para>
    ///
    /// <para><b>Where the numbers come from.</b> A real forbidden-zone acquisition, whose 167 windows
    /// were read from the raw file. The edges are not round: they start at 400.4319 and step by
    /// ~3.0014 Th, because the boundaries are placed in the m/z regions where peptide isotope clusters
    /// do not fall. Generating the windows from (start, step, width) rather than storing all 167
    /// reproduces that acquisition to within <b>0.084 mDa</b> - a thousandth of a window - while making
    /// it obvious this is a template rather than a measurement of any one run.</para>
    ///
    /// <para>It is still only a starting point. When the real windows are available they win, and the
    /// tab says which it is using.</para>
    /// </summary>
    public static IsolationScheme AstralDefault()
    {
        const double start = 400.4319;
        const double step = 3.001365;
        const double width = 3.001364;
        const int count = 167; // 400.4319 -> 901.66 m/z

        var windows = new List<IsolationWindow>(count);
        for (var i = 0; i < count; i++)
        {
            var lo = start + i * step;
            windows.Add(new IsolationWindow(lo, lo + width));
        }
        return new IsolationScheme(AstralDefaultName, windows);
    }

    /// <summary>Short human-readable summary for the UI: "35 windows, 400-1240 m/z, 24 Th".</summary>
    public string Describe()
    {
        if (!HasWindows)
            return $"{Name} (no windows defined)";
        // Windows imported from a data file carry the instrument's own rounding, so nominally identical
        // widths differ in the 4th decimal (3.0013 vs 3.0014). Compare at 0.001 Th, or an evenly spaced
        // scheme would describe itself as "variable width".
        var widths = Windows.Select(w => Math.Round(w.Width, 3)).Distinct().ToList();
        var width = widths.Count == 1
            ? $"{widths[0].ToString("0.###", CultureInfo.InvariantCulture)} Th"
            : "variable width";
        return $"{Windows.Count} windows, "
            + $"{MzLow.ToString("0.#", CultureInfo.InvariantCulture)}-"
            + $"{MzHigh.ToString("0.#", CultureInfo.InvariantCulture)} m/z, {width}"
            + (IsScheduled ? ", RT-scheduled" : "");
    }

    /// <summary>
    /// Parse either XML spelling. Returns null when the text is not an isolation scheme at all; returns a
    /// scheme with no windows for "Results only" (so the caller can tell "not a scheme" from "a scheme
    /// that defines no windows"). Numbers are read invariant - Skyline writes XML invariant regardless of
    /// the machine's locale.
    /// </summary>
    public static IsolationScheme? Parse(string? xml)
    {
        if (string.IsNullOrWhiteSpace(xml))
            return null;

        XElement root;
        try
        {
            root = XElement.Parse(xml);
        }
        catch (System.Xml.XmlException)
        {
            return null;
        }

        var element = IsSchemeElement(root) ? root : root.Descendants().FirstOrDefault(IsSchemeElement);
        if (element is null)
            return null;

        var name = Attr(element, "name") ?? "(unnamed)";
        var windows = element.Descendants()
            .Where(e => string.Equals(e.Name.LocalName, "isolation_window", StringComparison.OrdinalIgnoreCase))
            .Select(ParseWindow)
            .Where(w => w.HasValue)
            .Select(w => w!.Value)
            .OrderBy(w => w.Start)
            .ToList();

        return new IsolationScheme(name, windows);

        static bool IsSchemeElement(XElement e) =>
            string.Equals(e.Name.LocalName, "isolation_scheme", StringComparison.OrdinalIgnoreCase)
            || string.Equals(e.Name.LocalName, "IsolationScheme", StringComparison.OrdinalIgnoreCase);
    }

    private static IsolationWindow? ParseWindow(XElement e)
    {
        var start = Num(Attr(e, "start"));
        var end = Num(Attr(e, "end"));
        if (start is null || end is null || end <= start)
            return null;
        // rt_start / rt_stop are PRISM's own extension for scheduled (PRM/MTM) windows; Skyline's DIA
        // schemes never carry them, and their absence means "always on".
        var rtStart = Num(Attr(e, "rt_start"));
        var rtStop = Num(Attr(e, "rt_stop"));
        var scheduled = rtStart is not null && rtStop is not null && rtStop > rtStart;
        return new IsolationWindow(
            start.Value, end.Value, Num(Attr(e, "margin")) ?? 0,
            scheduled ? rtStart!.Value : double.NaN,
            scheduled ? rtStop!.Value : double.NaN);
    }

    // Attribute names are matched case-insensitively for the same reason the element name is: the
    // settings-list and .sky spellings of the same scheme do not agree on casing.
    private static string? Attr(XElement e, string name) => e.Attributes()
        .FirstOrDefault(a => string.Equals(a.Name.LocalName, name, StringComparison.OrdinalIgnoreCase))
        ?.Value;

    private static double? Num(string? text) =>
        double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var v) ? v : null;
}
