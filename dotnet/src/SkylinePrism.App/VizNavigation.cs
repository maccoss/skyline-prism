namespace SkylinePrism.App;

/// <summary>One of the panes on the Visualization tab. The values are the nav rail's row order.</summary>
/// <summary>
/// The Analysis rail's panes, in the order the ListBox lists them - the index IS the enum value, the
/// way <see cref="VizPane"/> works, so the two rails behave identically.
/// </summary>
internal enum AnalysisPane
{
    Inputs = 0,
    Settings = 1,
    Log = 2,
}

internal enum VizPane
{
    Qc = 0,
    Density = 1,
    DynamicRange = 2,
    IonAccounting = 3,
}

/// <summary>
/// Which visualization pane is on screen, and what that means for work that must only run while a
/// pane is visible.
/// </summary>
/// <remarks>
/// Its own class for the same reason as <see cref="QcPlotChrome"/>: pure state with no window in it,
/// so it can be tested without a dispatcher. The window is ~2,800 lines of code-behind at no
/// coverage, and the rule below is one worth pinning - a pane's visibility is now two conditions
/// rather than one, and getting it wrong fails silently in both directions. Too eager and PRISM
/// polls Skyline while the user is on the Settings tab; too lazy and a plot never loads.
/// </remarks>
internal static class VizNavigation
{
    /// <summary>
    /// The pane the user can actually see: null whenever the Visualization tab is not the selected
    /// one, whatever the nav rail says.
    /// </summary>
    /// <remarks>
    /// The nav rail keeps its selection while the user is over on Analysis, so the rail alone cannot
    /// answer this - which is exactly the mistake that would leave the dynamic-range tab polling
    /// Skyline's selection from behind another tab.
    /// </remarks>
    public static VizPane? Current(bool visualizationTabSelected, int navIndex)
    {
        if (!visualizationTabSelected)
            return null;
        return navIndex switch
        {
            (int)VizPane.Qc => VizPane.Qc,
            (int)VizPane.Density => VizPane.Density,
            (int)VizPane.DynamicRange => VizPane.DynamicRange,
            (int)VizPane.IonAccounting => VizPane.IonAccounting,
            _ => null,   // nothing selected yet (-1), or a row added to the XAML and not to this enum
        };
    }

    /// <summary>
    /// Whether to poll Skyline for its current selection. Only the dynamic-range plot follows it, and
    /// only while it is on screen - the poll is a live RPC to another process on a timer, so leaving
    /// it running behind a hidden pane costs Skyline work for a plot nobody is looking at.
    /// </summary>
    public static bool ShouldFollowSkylineSelection(VizPane? pane) => pane == VizPane.DynamicRange;
}
