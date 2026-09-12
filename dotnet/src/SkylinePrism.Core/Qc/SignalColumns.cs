using System;
using System.Collections.Generic;
using SkylinePrism.Core.IO;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// The columns needed to place a transition in signal space, resolved to whatever the export
/// spelled them.
/// </summary>
/// <remarks>
/// <para><b>Geometry only - no magnitudes.</b> There is deliberately no abundance, background or
/// ion-count column here. Every number the ion accounting reports is measured from the instrument
/// files, so <c>merged_data/</c> is read for WHERE Skyline extracted, WHEN, and for WHOM, and
/// nothing else. An export with no <c>Area</c> column at all still supports the accounting.</para>
///
/// <para>Beyond <see cref="SkylineColumns"/> because that type does not bind
/// <c>StartTime</c>/<c>EndTime</c>, which only the density view had needed until this.</para>
/// </remarks>
/// <param name="PrecursorCharge">
/// Null when the export has no charge column. Only the THEORETICAL claim set needs it - to enumerate
/// a peptide's b/y ions you must know how many charges they can carry - so an export without it
/// still supports the quantified accounting in full, and simply reports no explained total.
/// </param>
public sealed record SignalColumns(
    string Sample,
    string Peptide,
    string Transition,
    string PrecursorMz,
    string ProductMz,
    string StartTime,
    string EndTime,
    string? PrecursorCharge = null)
{
    /// <summary>
    /// How many protein lists can be accounted for at once - one bit each in
    /// <see cref="PeptideClass.ListMask"/>.
    /// </summary>
    public const int MaxLists = 32;

    /// <summary>
    /// Resolve against a merged table's actual column names, or null when it lacks one. A report
    /// exported without <c>Product Mz</c> cannot place a fragment in m/z and simply has no
    /// accounting, which is a thing to report rather than to guess around.
    /// </summary>
    public static SignalColumns? Resolve(ICollection<string> available)
    {
        var sample = SkylineColumns.FindColumn(available, "Sample ID", "Replicate Name");
        var peptide = SkylineColumns.FindColumn(
            available, "Peptide Modified Sequence Unimod Ids", "Peptide Modified Sequence", "Peptide");
        var transition = SkylineColumns.FindColumn(available, "Fragment Ion");
        var precursorMz = SkylineColumns.FindColumn(available, "Precursor Mz");
        var productMz = SkylineColumns.FindColumn(available, "Product Mz");
        var start = SkylineColumns.FindColumn(available, "Start Time");
        var end = SkylineColumns.FindColumn(available, "End Time");

        // Optional, and deliberately not part of the null check below: it gates only the explained
        // total, so an older export missing it loses that one number rather than the whole section.
        var precursorCharge = SkylineColumns.FindColumn(available, "Precursor Charge");

        return sample is null || peptide is null || transition is null
            || precursorMz is null || productMz is null || start is null || end is null
            ? null
            : new SignalColumns(
                sample, peptide, transition, precursorMz, productMz, start, end, precursorCharge);
    }
}

/// <summary>How a peptide was classified by the run, supplied by the caller as pure identity.</summary>
/// <param name="Assigned">The peptide reached the peptide matrix (the row set of peptides_rollup).</param>
/// <param name="ListMask">Bit per selected protein list claiming it.</param>
public readonly record struct PeptideClass(bool Assigned, uint ListMask);
