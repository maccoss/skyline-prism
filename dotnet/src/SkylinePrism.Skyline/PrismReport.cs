#nullable enable

namespace SkylinePrism.Skyline;

/// <summary>
/// The two PRISM transition report definitions bundled under <c>Reports/</c>, and the rule that relates
/// them: <c>PRISM-Ions</c> is <c>PRISM</c> plus exactly one column, Skyline's per-transition LC Peak ion
/// count (<see cref="TransitionIonCountColumn"/>). Both exporters pick a definition here from the
/// caller's <c>includeIonCounts</c>, so the file name, the view name and the report Skyline is asked to
/// export can never disagree.
///
/// <para>Why two files rather than one with an optional column: the ion count is expensive for Skyline
/// to compute - per spectrum, for every transition - and a report is installed into the user's
/// Skyline settings under its view name. Two names mean the fast report a user already relies on is
/// never silently replaced by the slow one, and an export's file stem says nothing about which was used,
/// so the name is the only record. <c>PrismReportDefinitionTests</c> holds the two files to the
/// superset relation.</para>
/// </summary>
public static class PrismReport
{
    /// <summary>The standard transition report's view name - what Skyline exports it as.</summary>
    public const string Name = "PRISM";

    /// <summary>The ion-count variant's view name.</summary>
    public const string IonsName = "PRISM-Ions";

    /// <summary>The bundled definition of <see cref="Name"/>.</summary>
    public const string FileName = "Skyline-PRISM.skyr";

    /// <summary>The bundled definition of <see cref="IonsName"/>.</summary>
    public const string IonsFileName = "Skyline-PRISM-Ions.skyr";

    /// <summary>
    /// The one column that distinguishes the two: Skyline's LC Peak Transition Ion Count, the sum over
    /// the spectra inside the peak boundaries of transition intensity times injection time.
    /// </summary>
    public const string TransitionIonCountColumn =
        "Results!*.Value.TransitionIonMetrics.LcPeakTransitionIonCount";

    /// <summary>
    /// What asking for ion counts costs, in one sentence, for a log line or a tooltip. Measured on the
    /// 6.5 GB FLARE document (46M transition rows): 7.9M rows in 42 minutes against the standard
    /// report's 5.4M rows/minute. Lives here because both exporters and the GUI say it, and five
    /// copies of a measured number drift.
    /// </summary>
    public const string IonCountCostNote =
        "Skyline computes LC Peak Transition Ion Count per spectrum for every transition, so this "
        + "export takes roughly 30x longer than the standard report (about 4 hours instead of "
        + "9.5 minutes on a 46M-row document).";

    /// <summary>The view name to install and export for the requested variant.</summary>
    public static string NameFor(bool includeIonCounts) => includeIonCounts ? IonsName : Name;

    /// <summary>The bundled .skyr file name for the requested variant.</summary>
    public static string FileFor(bool includeIonCounts) => includeIonCounts ? IonsFileName : FileName;
}
