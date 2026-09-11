using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using SkylinePrism.Core.Qc;

namespace SkylinePrism.Core.RawData;

/// <summary>
/// What to measure while walking one data file: which regions of signal space the run's peptides
/// claim, and how to place an MS2 scan in the same isolation windows those claims were placed in.
/// </summary>
/// <remarks>
/// The window mapping lives HERE rather than in the reader because the two sides have to agree. A
/// claim is placed by its precursor's m/z (<see cref="ClaimedRegionLoader"/>); a scan is placed by
/// its own isolation center. Resolving both through <see cref="WindowIndexFor"/> is what makes those
/// the same index - a reader that rounded or ordered windows differently would silently compare a
/// scan against another window's claims and report a near-zero assigned fraction.
/// </remarks>
/// <param name="Claims">The regions peptides claim, already merged for sharing.</param>
/// <param name="Scheme">The isolation windows, in the order the claims were indexed against.</param>
/// <param name="ListNames">Protein lists with a bit in <see cref="ClaimedRegion.ListMask"/>, in bit order.</param>
public sealed record IonAccountingRequest(
    ClaimedSignalIndex Claims,
    IsolationScheme Scheme,
    IReadOnlyList<string> ListNames)
{
    /// <summary>How many per-list totals the reader should accumulate.</summary>
    public int ListCount => ListNames?.Count ?? 0;

    /// <summary>
    /// The scheme window an MS2 scan belongs to, by the same rule that placed the claims: the
    /// narrowest window containing the m/z and firing at that time. Returns
    /// <see cref="ClaimedSignalIndex.AnyWindow"/>'s sibling -1 when no window covers it, which the
    /// reader must treat as "no claim can match this scan" rather than as window 0.
    /// </summary>
    public int WindowIndexFor(double centerMz, double rtMinutes)
    {
        if (!double.IsFinite(centerMz) || Scheme is null)
            return -1;

        var best = -1;
        var bestWidth = double.PositiveInfinity;
        for (var i = 0; i < Scheme.Windows.Count; i++)
        {
            var w = Scheme.Windows[i];
            if (!w.Contains(centerMz))
                continue;
            if (double.IsFinite(rtMinutes) && !w.IsOnAt(rtMinutes))
                continue;
            if (w.Width < bestWidth)
            {
                bestWidth = w.Width;
                best = i;
            }
        }
        return best;
    }
}

/// <summary>
/// One acquisition cycle's ion accounting, both MS levels, in retention-time order. This is what the
/// time-versus-ions plots read, and what makes them a file read rather than a recomputation.
/// </summary>
/// <param name="Ms1Acquired">Ion-proportional total over the cycle's MS1 scans. See
/// <see cref="IonAccountingRecord"/> for what the unit is.</param>
/// <param name="Ms1Assigned">The part of it inside a region some peptide claimed.</param>
public readonly record struct IonCycle(
    int Index,
    double RtStartMin,
    double RtStopMin,
    int Ms1Count,
    int Ms2Count,
    double Ms1Acquired,
    double Ms2Acquired,
    double Ms1Assigned,
    double Ms2Assigned);

/// <summary>
/// What one instrument data file contributes to ion accounting: how many ions reached the detector,
/// and what fraction of them a peptide sequence explains, at each MS level.
///
/// <para><b>The unit: ions.</b> The reported intensity is a RATE - ions per second - so every total
/// here is a scan's intensity multiplied by its ion injection time IN SECONDS, summed. Both factors
/// matter and each was got wrong once:</para>
/// <list type="bullet">
/// <item><description>Dropping the injection time entirely leaves a rate, and a rate summed over
/// scans is not a count. Dividing an intensity-time integral by it is dimensionally meaningless -
/// the defect this feature replaces.</description></item>
/// <item><description>Using MILLISECONDS makes every total 1000x too large. The fraction is
/// untouched, because both sides carry the same weighting and the ratio cancels, so nothing about
/// the fraction can reveal it. What reveals it is per-scan plausibility: 3.7e8 ions in one MS1 scan
/// is impossible against any AGC target, and 3.7e5 is right.</description></item>
/// </list>
/// <para>The conversion assumes the vendor reports intensity as a rate, which is what makes the
/// product a count; Thermo does, and it is the same assumption behind Skyline's own
/// <c>LC Peak Transition Ion Count</c>. Where it does not hold the totals are ion-PROPORTIONAL
/// rather than absolute - and the fraction, which is what the plots show, is unaffected either
/// way.</para>
///
/// <para><b>Before comparing these totals with Skyline's own ion counts, check which unit Skyline
/// multiplies by.</b> It has not been verified here. If Skyline uses milliseconds - the same slip
/// this code made - its counts sit 1000x above a true ion count, and the two would disagree by that
/// factor with neither side being wrong about anything except the unit. The RATIO of assigned to
/// acquired is the safe thing to compare, because it is insensitive to the choice.</para>
///
/// <para><b>Why the reader computes both halves.</b> The assigned total is the union of the regions
/// peptides claim, evaluated against each spectrum. It cannot be assembled from Skyline's
/// per-transition areas or ion counts: those sum shared signal once per transition, which
/// double-counts in a DIA window and can push assigned past acquired.</para>
/// </summary>
/// <param name="Ms1ByList">Per-list MS1 assigned totals, in <see cref="IonAccountingRequest.ListNames"/> order.</param>
/// <param name="SpectraMissingInjectionTime">
/// Scans with no ion-injection-time cvParam. They cannot be converted to ions at all, so they are
/// EXCLUDED from both totals rather than given an invented weight - weighting one at a full second
/// would count it as roughly 141 typical scans here and distort the totals far more than leaving it
/// out. The fraction stays valid because both sides lose the same scans. Non-zero is worth saying
/// out loud; on every file measured it is zero.
/// </param>
public sealed record IonAccountingRecord(
    string DataPath,
    Ms2ReadStatus Status,
    string Reader,
    int Ms1Count,
    int Ms2Count,
    double Ms1Acquired,
    double Ms2Acquired,
    double Ms1Assigned,
    double Ms2Assigned,
    IReadOnlyList<double> Ms1ByList,
    IReadOnlyList<double> Ms2ByList,
    double RtStartMin,
    double RtStopMin,
    int SpectraMissingInjectionTime,
    int ScansOutsideScheme,
    IReadOnlyList<IonCycle> Cycles,
    string? Message = null)
{
    public bool IsUsable => Status == Ms2ReadStatus.Ok && (Ms1Acquired > 0 || Ms2Acquired > 0);

    /// <summary>Assigned over acquired at MS1, or NaN when nothing was acquired.</summary>
    public double Ms1Fraction => Ms1Acquired > 0 ? Ms1Assigned / Ms1Acquired : double.NaN;

    /// <inheritdoc cref="Ms1Fraction"/>
    public double Ms2Fraction => Ms2Acquired > 0 ? Ms2Assigned / Ms2Acquired : double.NaN;

    /// <summary>
    /// True when more signal was assigned than acquired, which is impossible and therefore a defect
    /// - a units mismatch, a window index that does not line up, or claims merged too loosely.
    ///
    /// <para>Callers must refuse to display a fraction when this is set, rather than clamping it.
    /// Clamping to 100% turns a visible bug into a plausible reading - and the earlier version of
    /// this feature shipped a fraction computed from mismatched units for exactly that reason: it
    /// looked like a coverage percentage, so nothing about it invited checking.</para>
    /// </summary>
    public bool Exceeded => Ms1Assigned > Ms1Acquired || Ms2Assigned > Ms2Acquired;


    /// <summary>
    /// Mean ions in one MS1 scan, or NaN when none were read. The one figure here that can be
    /// checked against something already known: the instrument's AGC target.
    /// </summary>
    public double MeanMs1IonsPerScan => Ms1Count > 0 ? Ms1Acquired / Ms1Count : double.NaN;

    /// <inheritdoc cref="MeanMs1IonsPerScan"/>
    public double MeanMs2IonsPerScan => Ms2Count > 0 ? Ms2Acquired / Ms2Count : double.NaN;

    /// <summary>
    /// True when the mean ions per scan is outside anything an instrument can hold, which means the
    /// totals are in the wrong UNIT rather than merely surprising.
    /// </summary>
    /// <remarks>
    /// <para>This exists because a units error on the injection time is invisible to every check
    /// built on the fraction: it scales the numerator and the denominator identically and cancels
    /// exactly. Fractions in range, stable across replicates, cycles summing to totals - all of them
    /// pass while every absolute number is wrong by a constant. So the check has to be made on an
    /// absolute quantity, and the per-scan ion count is the only one with an external reference.</para>
    /// <para>The bounds are deliberately wide, because the point is to catch a factor of 1000 and not
    /// to opine on anyone's method. Real AGC targets span roughly 1e3 to 1e7; anything at or above
    /// <see cref="MaxPlausibleIonsPerScan"/> is beyond any current instrument, and a mean below one
    /// ion per scan is not a measurement. Multiplying by milliseconds instead of seconds put a real
    /// Astral file's MS1 mean at 3.7e8, which this catches; corrected it is 3.7e5, which it does
    /// not.</para>
    /// </remarks>
    public bool IonScaleImplausible =>
        Implausible(MeanMs1IonsPerScan) || Implausible(MeanMs2IonsPerScan);

    /// <summary>Beyond any current instrument's AGC target by a wide margin. See <see cref="IonScaleImplausible"/>.</summary>
    public const double MaxPlausibleIonsPerScan = 1e8;

    /// <summary>A mean below this is not a measurement. See <see cref="IonScaleImplausible"/>.</summary>
    public const double MinPlausibleIonsPerScan = 1.0;

    /// <summary>
    /// Whether a mean ions-per-scan is outside what an instrument can hold. One implementation,
    /// called by both this record and the cached row - two copies of a threshold rule drift, and the
    /// drift would be silent because neither copy fires on healthy data.
    /// </summary>
    public static bool IsIonScaleImplausible(double meanPerScan) =>
        double.IsFinite(meanPerScan)
        && meanPerScan > 0
        && (meanPerScan >= MaxPlausibleIonsPerScan || meanPerScan < MinPlausibleIonsPerScan);

    private static bool Implausible(double meanPerScan) => IsIonScaleImplausible(meanPerScan);

    /// <summary>A record standing for a read that did not happen, so callers never see a null.</summary>
    public static IonAccountingRecord Unavailable(
        string dataPath, Ms2ReadStatus status, string reader, string? message = null) =>
        new(dataPath, status, reader, 0, 0, 0, 0, 0, 0,
            Array.Empty<double>(), Array.Empty<double>(), double.NaN, double.NaN, 0, 0,
            Array.Empty<IonCycle>(), message);
}

/// <summary>
/// A reader that can measure acquired and assigned ions in the SAME pass over a data file.
///
/// <para>Separate from <see cref="IMs2SignalReader"/> because the two do different amounts of work:
/// acquired-only can be read from scan headers, or even from the file's own TIC chromatogram, while
/// assigned needs every spectrum's peak arrays decoded and masked against the claims. Keeping them
/// apart means the cheap question stays cheap, and a build whose reader predates this interface
/// still answers it.</para>
/// </summary>
public interface IIonAccountingReader : IMs2SignalReader
{
    /// <summary>
    /// Walk one file once, accumulating acquired and assigned ions per cycle at both MS levels.
    /// <b>Never throws</b>, on the same contract as <see cref="IMs2SignalReader.Read"/>: one bad file
    /// in a cohort must not abandon the rest.
    /// </summary>
    IonAccountingRecord ReadAccounting(
        string dataPath, IonAccountingRequest request, Action<string>? log = null,
        CancellationToken ct = default);
}

/// <summary>Finding an <see cref="IIonAccountingReader"/> among the registered readers.</summary>
public static class IonAccountingReaders
{
    /// <summary>Every registered reader that can do ion accounting, most recently registered first.</summary>
    public static IReadOnlyList<IIonAccountingReader> All =>
        Ms2SignalReaders.All.OfType<IIonAccountingReader>().ToArray();

    /// <summary>The first accounting reader that claims <paramref name="dataPath"/>, or null.</summary>
    public static IIonAccountingReader? For(string dataPath)
    {
        foreach (var reader in All)
        {
            // A reader that throws while probing is a broken reader, not a reason to fail the run.
            try
            {
                if (reader.CanRead(dataPath))
                    return reader;
            }
            catch (Exception)
            {
                // fall through to the next candidate
            }
        }
        return null;
    }

    /// <summary>Whether this build can compute ion accounting at all.</summary>
    public static bool Available => All.Count > 0;

    /// <summary>
    /// Read with whichever accounting reader claims the file, or a
    /// <see cref="Ms2ReadStatus.NoReader"/> record when this build has none.
    /// </summary>
    public static IonAccountingRecord Read(
        string dataPath, IonAccountingRequest request, Action<string>? log = null,
        CancellationToken ct = default)
    {
        var reader = For(dataPath);
        return reader is null
            ? IonAccountingRecord.Unavailable(
                dataPath, Ms2ReadStatus.NoReader, "none",
                "This build has no instrument-file reader, so ion accounting is unavailable.")
            : reader.ReadAccounting(dataPath, request, log, ct);
    }
}
