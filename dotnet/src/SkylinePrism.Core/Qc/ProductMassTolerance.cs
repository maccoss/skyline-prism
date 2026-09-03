using System;
using System.Globalization;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// The m/z range Skyline summed a product ion over, for one target m/z.
///
/// <para>Distinct from <see cref="IsolationWindow"/>, which is a PRECURSOR isolation window - the
/// slice of precursors the instrument fragmented together. This is the much narrower range around a
/// single FRAGMENT m/z that Skyline integrated to produce that transition's area. MS2 signal
/// accounting needs both: two transitions read the same detector counts only when they share an
/// isolation window AND their extraction windows overlap.</para>
/// </summary>
public readonly record struct ExtractionWindow(double Start, double End)
{
    public double Width => End - Start;
    public double Center => (Start + End) / 2;

    /// <summary>
    /// Do these two ranges share any m/z, and therefore any detector counts? Strict, so windows that
    /// merely touch are separate - matching the retention-time rule in <see cref="Ms2SignalUnion"/>,
    /// where a peak ending exactly where the next begins shares no scan.
    /// </summary>
    public bool Overlaps(ExtractionWindow other) => Start < other.End && other.Start < End;
}

/// <summary>
/// Skyline's product-ion extraction settings, evaluated to the actual window it extracted over.
///
/// <para>MS2 signal accounting needs this because summing transition areas double-counts: in a DIA
/// isolation window with tens of co-isolated peptides, two fragments whose extraction windows overlap
/// are the same counts, credited twice. Deciding "same signal" needs the range Skyline actually used,
/// not an approximation of it.</para>
///
/// <para><b>The arithmetic is Skyline's, transcribed rather than derived</b>, from
/// <c>pwiz_tools/Skyline/Model/DocSettings/TransitionSettings.cs</c> - <c>GetDenominator</c> (line
/// 2638) and <c>GetFilterWindow</c> (line 2621) - and applied the way Skyline applies it, in
/// <c>Model/Results/SpectrumFilterPair.cs</c> (lines 318-319 and 336-338):</para>
/// <code>
/// double startFilter = targetMz - filterWindow / 2;
/// double endFilter   = startFilter + filterWindow;
/// </code>
/// <para>So <c>GetFilterWindow</c> returns a width that is halved at the point of use, and the stated
/// <c>product_res</c> is the <b>+/- tolerance</b>: <c>centroided res="10"</c> extracts +/-10 ppm and
/// <c>qit res="0.7"</c> extracts +/-0.7 m/z. Only <see cref="WindowAt"/> is exposed, so there is one
/// unambiguous geometry rather than a width whose convention has to be remembered.</para>
/// </summary>
/// <param name="Analyzer">Skyline's <c>product_mass_analyzer</c>: centroided, qit, tof, orbitrap, ft_icr.</param>
/// <param name="Resolution">Skyline's <c>product_res</c> - ppm for centroided, m/z for QIT, resolving power otherwise.</param>
/// <param name="ResolutionMz">Skyline's <c>product_res_mz</c>, the m/z the resolving power is calibrated at. Required for orbitrap and ft_icr, absent otherwise.</param>
/// <param name="SelectiveExtraction">Skyline's <c>selective_extraction</c>: halves the window for every
/// analyzer except <c>centroided</c> - it divides the resolving-power denominators and, on Skyline's
/// default arm, the QIT tolerance too. Skyline offers it for any profile-mode analyzer, QIT included.</param>
public sealed record ProductMassTolerance(
    string Analyzer, double Resolution, double? ResolutionMz = null, bool SelectiveExtraction = false)
{
    /// <summary>Skyline's RES_PER_FILTER / RES_PER_FILTER_SELECTIVE (TransitionSettings.cs:2325).</summary>
    private double ResPerFilter => SelectiveExtraction ? 1.0 : 2.0;

    /// <summary>Skyline's GetDenominator (TransitionSettings.cs:2638).</summary>
    private double Denominator => Analyzer switch
    {
        "tof" => Resolution / ResPerFilter,
        "orbitrap" => Math.Sqrt(ResolutionMz ?? 0) * Resolution / ResPerFilter,
        "ft_icr" => (ResolutionMz ?? 0) * Resolution / ResPerFilter,
        "centroided" => Resolution * 2 / 1e6,
        _ => Resolution * ResPerFilter,     // qit, and Skyline's own default arm
    };

    /// <summary>
    /// Skyline's GetFilterWindow (TransitionSettings.cs:2621). Private on purpose: it is a width that
    /// only means anything once halved and centred on the target, which is <see cref="WindowAt"/>'s
    /// job. Exposing it invites reading it as the +/- tolerance, which is twice too wide.
    /// </summary>
    private double FilterWidthAt(double mz)
    {
        var d = Denominator;
        if (d <= 0 || !double.IsFinite(d) || !double.IsFinite(mz))
            return double.NaN;
        return Analyzer switch
        {
            "orbitrap" => mz * Math.Sqrt(mz) / d,
            "tof" => mz / d,
            "ft_icr" => mz * mz / d,
            "centroided" => mz * d,
            _ => d,
        };
    }

    /// <summary>
    /// The m/z range Skyline extracted a product ion at <paramref name="mz"/> over. Two product ions
    /// in one isolation window are the same detector signal when their windows
    /// <see cref="ExtractionWindow.Overlaps">overlap</see>.
    /// </summary>
    public ExtractionWindow WindowAt(double mz)
    {
        var width = FilterWidthAt(mz);
        return new ExtractionWindow(mz - width / 2, mz + width / 2);
    }

    /// <summary>
    /// Whether this analyzer needs <c>product_res_mz</c>. Skyline requires it for orbitrap and ft_icr
    /// and rejects it for the others, so a document missing it for those two cannot be interpreted.
    /// </summary>
    public bool IsUsable =>
        Analyzer.Length > 0
        && Resolution > 0
        && (Analyzer is not ("orbitrap" or "ft_icr") || ResolutionMz is > 0);

    /// <summary>
    /// Build from the raw <c>&lt;transition_full_scan&gt;</c> attribute strings, or null when the
    /// document does not carry enough to interpret them. Invariant culture: these are XML attributes,
    /// not display text, and a comma decimal separator would silently mis-parse.
    /// </summary>
    public static ProductMassTolerance? Parse(
        string? analyzer, string? resolution, string? resolutionMz = null, string? selectiveExtraction = null)
    {
        if (string.IsNullOrWhiteSpace(analyzer) || !TryNum(resolution, out var res))
            return null;
        var tolerance = new ProductMassTolerance(
            analyzer.Trim().ToLowerInvariant(),
            res,
            TryNum(resolutionMz, out var mz) ? mz : null,
            string.Equals(selectiveExtraction?.Trim(), "true", StringComparison.OrdinalIgnoreCase));
        return tolerance.IsUsable ? tolerance : null;

        static bool TryNum(string? s, out double value) =>
            double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out value);
    }

    /// <summary>
    /// Parse the human-readable form a config file uses - <c>"10 ppm"</c> or <c>"0.4 m/z"</c> - rather
    /// than Skyline's analyzer/resolution attribute pair, which means nothing outside a .sky.
    ///
    /// <para>Both spell a +/- tolerance, matching what the document's <c>product_res</c> means: "10 ppm"
    /// extracts +/-10 ppm and "0.4 m/z" extracts +/-0.4 m/z. Returns null for anything else, so a
    /// mistyped setting is reported rather than silently treated as a default.</para>
    /// </summary>
    public static ProductMassTolerance? ParseSetting(string? setting)
    {
        if (string.IsNullOrWhiteSpace(setting))
            return null;

        var text = setting.Trim().ToLowerInvariant();
        // Accept the spellings people actually write: "m/z", "mz", "th", "da", "dalton".
        var isPpm = text.EndsWith("ppm", StringComparison.Ordinal);
        var isMz = text.EndsWith("m/z", StringComparison.Ordinal)
            || text.EndsWith("mz", StringComparison.Ordinal)
            || text.EndsWith("th", StringComparison.Ordinal)
            || text.EndsWith("da", StringComparison.Ordinal)
            || text.EndsWith("dalton", StringComparison.Ordinal);
        if (!isPpm && !isMz)
            return null;

        // The leading number, taken explicitly rather than by trimming letters off the end - the unit
        // spellings share letters with nothing in a number, but a character-set trim is a rule nobody
        // can check by reading it.
        var end = 0;
        while (end < text.Length && (char.IsDigit(text[end]) || text[end] is '.' or '+' or '-' or 'e'))
            end++;

        var number = text[..end];
        if (!double.TryParse(number, NumberStyles.Float, CultureInfo.InvariantCulture, out var value)
            || value <= 0)
        {
            return null;
        }

        // "centroided" is Skyline's ppm analyzer; "qit" is its fixed-width one. Going through the same
        // analyzer names keeps one implementation of the window arithmetic.
        var tolerance = new ProductMassTolerance(isPpm ? "centroided" : "qit", value);
        return tolerance.IsUsable ? tolerance : null;
    }

    /// <summary>
    /// The config spelling of this tolerance - <c>"10 ppm"</c> or <c>"0.7 m/z"</c> - chosen so that
    /// <see cref="ParseSetting"/> gives back an equal tolerance; this is how the Skyline tool hands a
    /// document's own extraction setting to <c>qc_report.ms2_signal.extraction_tolerance</c>.
    ///
    /// <para>Null whenever the setting cannot express this window, so a caller keeps the configured
    /// value and says so rather than writing a number that means something else:</para>
    /// <list type="bullet">
    /// <item><description>the resolving-power analyzers (tof, orbitrap, ft_icr), whose window is not
    /// one +/- number - a nearest-ppm conversion would be right at a single m/z only;</description></item>
    /// <item><description>a QIT window with <see cref="SelectiveExtraction"/>, which HALVES it
    /// (<c>ResPerFilter</c> reaches the qit arm, matching Skyline's own default arm). The setting has
    /// no way to say "selective", so <c>"0.7 m/z"</c> would parse back to +/-0.7 where the document
    /// extracted +/-0.35 - twice too wide, over-merging fragment sharing while the plot caption named
    /// the document's own number.</description></item>
    /// </list>
    /// </summary>
    public string? ToSetting() => Analyzer switch
    {
        // "R" round-trips the double exactly, so what is written parses back to the same tolerance.
        // Selective extraction does not affect the centroided arm's window (no ResPerFilter there).
        "centroided" => Resolution.ToString("R", CultureInfo.InvariantCulture) + " ppm",
        "qit" when !SelectiveExtraction => Resolution.ToString("R", CultureInfo.InvariantCulture) + " m/z",
        _ => null,
    };

    /// <summary>
    /// How the tolerance reads in a log line or a plot caption. Stated as the +/- tolerance where that
    /// is a single number, because that is what the document's own setting means.
    /// </summary>
    public string Describe() => Analyzer switch
    {
        "centroided" => $"+/-{Resolution:0.###} ppm (centroided)",
        "qit" => $"+/-{Resolution:0.###} m/z (QIT)",
        "tof" => $"resolving power {Resolution:0} (TOF)",
        "orbitrap" => $"resolving power {Resolution:0} at m/z {ResolutionMz:0} (Orbitrap)",
        "ft_icr" => $"resolving power {Resolution:0} at m/z {ResolutionMz:0} (FT-ICR)",
        _ => $"{Resolution:0.###} ({Analyzer})",
    };
}
