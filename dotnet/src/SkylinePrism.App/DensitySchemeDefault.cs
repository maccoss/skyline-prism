namespace SkylinePrism.App;

/// <summary>
/// Which isolation scheme the Spectrum density picker should start on, as a precedence rule that can
/// be tested without a window.
/// </summary>
/// <remarks>
/// <para>Extracted from <c>MainWindow</c> for the same reason <see cref="BatchCorrectionDefault"/> and
/// <c>VizNavigation</c> were: it is a decision with cases, it lives in a file at zero coverage, and
/// getting the ORDER wrong is not cosmetic. The map is read as "how crowded was that spectrum", which
/// is only true when the rows are the windows the data was acquired with; a wrong grid produces a map
/// that looks exactly as plausible as a right one.</para>
///
/// <para>The order, most authoritative first:</para>
/// <list type="number">
/// <item>The batch document's own scheme. Declared windows are the acquisition as Skyline recorded
/// it, and the picker is locked on them - there is nothing to choose.</item>
/// <item>Whatever the user was already on. A default that reasserted itself would silently re-bin a
/// map someone was reading, which is the complaint that makes people stop trusting the window.</item>
/// <item>A scheme MEASURED from the instrument files. This is what the instrument actually ran.</item>
/// <item>The preferred built-in. A plausible modern DIA cycle, and a guess.</item>
/// <item>Uniform bins - approximate, and labeled as such.</item>
/// </list>
/// </remarks>
internal static class DensitySchemeDefault
{
    /// <summary>
    /// The index to preselect. Every parameter is an index into the picker, or -1 when that option is
    /// not on offer.
    /// </summary>
    /// <param name="documentIndex">The batch document's own scheme, with windows.</param>
    /// <param name="keptIndex">The entry the user was already on, if it is still in the list.</param>
    /// <param name="measuredIndex">A scheme read out of the instrument data files.</param>
    /// <param name="builtInIndex">The preferred built-in layout.</param>
    /// <param name="fallbackIndex">Last resort: the uniform-bin entry, which always exists.</param>
    public static int Choose(
        int documentIndex, int keptIndex, int measuredIndex, int builtInIndex, int fallbackIndex)
    {
        if (documentIndex >= 0)
            return documentIndex;
        if (keptIndex >= 0)
            return keptIndex;
        if (measuredIndex >= 0)
            return measuredIndex;
        return builtInIndex >= 0 ? builtInIndex : fallbackIndex;
    }
}
