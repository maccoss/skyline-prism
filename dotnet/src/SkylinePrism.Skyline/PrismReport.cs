#nullable enable

namespace SkylinePrism.Skyline;

/// <summary>
/// The PRISM transition report definition bundled under <c>Reports/</c>: one view name and one file,
/// so the file name, the view name and the report Skyline is asked to export can never disagree.
/// </summary>
/// <remarks>
/// There used to be a second definition, <c>PRISM-Ions</c> - the same columns plus Skyline's
/// per-transition LC Peak ion count - for <c>qc_report.ms2_signal.measure: ions</c>. It is gone with
/// that setting. It was never cheap: Skyline computes that column per spectrum for every transition,
/// measured at about 30x the standard report's cost, roughly 4 hours instead of 9.5 minutes on a
/// 46M-row document. Ion counts are now measured from the instrument files by
/// <c>prism ion-accounting</c>, which takes single-digit minutes per file and gets the union right -
/// per-transition counts cannot be summed correctly, because co-isolated peptides sharing a fragment
/// extract the same detector counts.
/// </remarks>
public static class PrismReport
{
    /// <summary>The view name to install and export.</summary>
    public const string Name = "PRISM";

    /// <summary>The bundled definition of <see cref="Name"/>.</summary>
    public const string FileName = "Skyline-PRISM.skyr";
}
