using System;
using System.Collections.Generic;
using System.Threading;

namespace SkylinePrism.Core.RawData;

/// <summary>
/// How a run's MS2 signal was divided into acquisition cycles. Recorded per file rather than assumed,
/// because the plot's x axis means something different under each and a reader that silently fell back
/// would otherwise look like a reader that succeeded.
/// </summary>
public enum Ms2CycleModel
{
    /// <summary>Unknown - no cycles were derived (a failed read, or a file with no MS2).</summary>
    None = 0,

    /// <summary>A cycle opens at each MS1 survey scan and closes at the next. The normal DIA shape.</summary>
    Ms1Bounded,

    /// <summary>MS2-only acquisition: a cycle closes when the isolation window target m/z wraps.</summary>
    IsolationWrap,

    /// <summary>Neither pattern was found, so cycles are fixed retention-time bins.</summary>
    FixedRtBins,
}

/// <summary>
/// Where a scan's total ion current came from. The reported value and a summed one are not
/// interchangeable - summing a centroided peak list misses what the instrument counted before
/// centroiding - so a cohort that mixes them has to say so rather than average over the difference.
/// </summary>
public enum Ms2SignalSource
{
    Unknown = 0,

    /// <summary>The scan's own <c>MS:1000285</c> total ion current. Preferred: it is a header field, so
    /// no peak array is decoded, it is the instrument's own pre-centroiding total, and msconvert
    /// propagates it - so a vendor file and its converted mzML give the same number.</summary>
    ReportedTic,

    /// <summary>Summed peak intensities, used only where the cvParam is absent.</summary>
    SummedPeaks,
}

/// <summary>Why a read produced no usable signal, or <see cref="Ok"/> when it did.</summary>
public enum Ms2ReadStatus
{
    Ok = 0,

    /// <summary>The path did not resolve to a file that exists.</summary>
    NotFound,

    /// <summary>No registered reader recognized the format.</summary>
    Unsupported,

    /// <summary>No raw reader is available in this build at all.</summary>
    NoReader,

    /// <summary>The reader was present and the file existed, but reading it failed.</summary>
    Failed,

    /// <summary>The caller cancelled before the read finished.</summary>
    Cancelled,
}

/// <summary>One acquisition cycle's MS2 total, in retention-time order.</summary>
/// <param name="Index">Zero-based position in the run.</param>
/// <param name="Ms2Signal">Summed MS2 total ion current over the cycle's scans. LINEAR, never log.</param>
public readonly record struct Ms2Cycle(
    int Index, double RtStartMin, double RtStopMin, int Ms2Count, double Ms2Signal);

/// <summary>
/// What one instrument data file contributes to MS2 signal accounting: how much MS2 signal it acquired
/// in total, and how that was distributed over the gradient.
///
/// <para>This is the DENOMINATOR the accounting needs and no Skyline export can supply. Skyline's
/// <c>TicArea</c> is one value per replicate and is MS1 by construction - on the committed cohort the
/// precursor areas of 327 of 75,202 peptides alone come to 45.6% of it - so it must never be
/// substituted for this. When no reader is available the honest result is a record with a non-OK
/// <see cref="Status"/> and the plot omitting its acquired bar, not a guessed total.</para>
/// </summary>
/// <param name="TotalMs2Signal">Summed MS2 total ion current over the whole run. LINEAR.</param>
/// <param name="Reader">Which reader produced this, for the report and for a mixed-cohort warning.</param>
/// <param name="IsolationWindows">Windows observed in the file, if the reader collected them. A DIA
/// analysis document normally stores none (<c>isolation_scheme name="Results only"</c>), so the raw
/// file is where they live; capturing them on this pass avoids a Skyline round-trip.</param>
public sealed record Ms2SignalRecord(
    string DataPath,
    Ms2ReadStatus Status,
    string Reader,
    Ms2SignalSource SignalSource,
    int Ms1Count,
    int Ms2Count,
    double TotalMs2Signal,
    double RtStartMin,
    double RtStopMin,
    Ms2CycleModel CycleModel,
    IReadOnlyList<Ms2Cycle> Cycles,
    IReadOnlyList<Qc.IsolationWindow> IsolationWindows,
    string? Message = null)
{
    public bool IsUsable => Status == Ms2ReadStatus.Ok && TotalMs2Signal > 0;

    /// <summary>A record standing for a read that did not happen, so callers never see a null.</summary>
    public static Ms2SignalRecord Unavailable(
        string dataPath, Ms2ReadStatus status, string reader, string? message = null) =>
        new(dataPath, status, reader, Ms2SignalSource.Unknown, 0, 0, 0, double.NaN, double.NaN,
            Ms2CycleModel.None, Array.Empty<Ms2Cycle>(), Array.Empty<Qc.IsolationWindow>(), message);
}

/// <summary>
/// Reads acquired MS2 signal out of an instrument data file.
///
/// <para>Core defines the seam and implements none of it: the only implementation depends on
/// ProteoWizard, which is a large native dependency that not every build carries. A build without one
/// still produces every other part of the accounting - the assigned and per-list unions come from
/// <c>merged_data/</c> and need no raw file at all.</para>
/// </summary>
public interface IMs2SignalReader
{
    /// <summary>
    /// Names the reader and how it measured, e.g. <c>pwiz-sharp 3.0.24</c> or
    /// <c>pwiz-sharp 3.0.24 +summed</c> where the reported TIC was absent. Reaches the QC report, so a
    /// cohort read by more than one reader is visible rather than silently mixed.
    /// </summary>
    string Describe();

    /// <summary>Cheap check, by extension or header - not a full open.</summary>
    bool CanRead(string dataPath);

    /// <summary>
    /// Read one file. <b>Never throws</b>: a missing file, an unreadable one or a cancelled read all
    /// come back as a record with a non-OK <see cref="Ms2SignalRecord.Status"/> and a message. One bad
    /// file in a 192-replicate cohort must not abandon the other 191.
    /// </summary>
    Ms2SignalRecord Read(string dataPath, Action<string>? log = null, CancellationToken ct = default);
}

/// <summary>
/// Where an optional raw-file reader registers itself. Empty in a build without one, which is the
/// normal state for the cross-platform CLI.
/// </summary>
public static class Ms2SignalReaders
{
    private static readonly List<IMs2SignalReader> Registered = new();

    /// <summary>Register a reader. Idempotent per instance; later registrations are tried first, so a
    /// more specific reader can take precedence over a general one.</summary>
    public static void Register(IMs2SignalReader reader)
    {
        if (reader is null)
            throw new ArgumentNullException(nameof(reader));
        lock (Registered)
        {
            if (!Registered.Contains(reader))
                Registered.Insert(0, reader);
        }
    }

    /// <summary>Every registered reader, most recently registered first.</summary>
    public static IReadOnlyList<IMs2SignalReader> All
    {
        get { lock (Registered) return Registered.ToArray(); }
    }

    /// <summary>The first reader that claims <paramref name="dataPath"/>, or null when none does.</summary>
    public static IMs2SignalReader? For(string dataPath)
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

    /// <summary>
    /// Read with whichever reader claims the file, or an <see cref="Ms2ReadStatus.NoReader"/> record
    /// when this build has none. The caller gets a record either way.
    /// </summary>
    public static Ms2SignalRecord Read(
        string dataPath, Action<string>? log = null, CancellationToken ct = default)
    {
        var reader = For(dataPath);
        return reader is null
            ? Ms2SignalRecord.Unavailable(
                dataPath, Ms2ReadStatus.NoReader, "none",
                "This build has no instrument-file reader, so acquired MS2 signal is unknown.")
            : reader.Read(dataPath, log, ct);
    }

    /// <summary>Drops every registration. For tests, which must not leak readers into each other.</summary>
    internal static void Clear()
    {
        lock (Registered) Registered.Clear();
    }
}
