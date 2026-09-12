using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using SkylinePrism.Core.Qc;

namespace SkylinePrism.Core.RawData;

/// <summary>
/// A reader that can answer "what isolation windows was this acquired with?" without measuring
/// anything else.
///
/// <para>Separate from <see cref="IMs2SignalReader"/> for the same reason
/// <see cref="IIonAccountingReader"/> is: the two questions cost different amounts. The windows are a
/// property of the ACQUISITION METHOD, so they are visible in the first couple of cycles of scan
/// headers - on a 3.3 GB Thermo file over SMB that is ~0.05 s once the file is open, against seconds
/// for a full signal read and minutes for a masked ion-accounting walk. Asking the expensive question
/// to get the cheap answer is what made reading the scheme too slow to do while a window was
/// opening.</para>
/// </summary>
public interface IIsolationWindowReader : IMs2SignalReader
{
    /// <summary>
    /// The repeating isolation windows, read from the first acquisition cycles. Empty - never null,
    /// and <b>never throwing</b>, on the same contract as <see cref="IMs2SignalReader.Read"/> - when
    /// the file cannot be opened, has no MS2, or has no repeating cycle (a DDA run isolates a
    /// different precursor every spectrum, so it has no scheme to report).
    /// </summary>
    IReadOnlyList<IsolationWindow> ReadIsolationWindows(
        string dataPath, Action<string>? log = null, CancellationToken ct = default);
}

/// <summary>
/// Reading isolation windows out of an instrument data file with whatever reader this build carries.
/// </summary>
public static class IsolationWindowProbe
{
    /// <summary>Whether any registered reader can answer the cheap question.</summary>
    public static bool Available => Ms2SignalReaders.All.OfType<IIsolationWindowReader>().Any();

    /// <summary>
    /// The windows in one data file. Empty when this build has no reader, no reader claims the file,
    /// or the file declares none.
    /// </summary>
    /// <remarks>
    /// Falls back to a full <see cref="IMs2SignalReader.Read"/> when the reader that claims the file
    /// predates <see cref="IIsolationWindowReader"/> - that record carries the windows too, it just
    /// pays for a signal measurement nobody asked for. Correctness first, speed where the reader
    /// offers it.
    /// </remarks>
    public static IReadOnlyList<IsolationWindow> Read(
        string dataPath, Action<string>? log = null, CancellationToken ct = default)
    {
        var reader = Ms2SignalReaders.For(dataPath);
        if (reader is null)
            return Array.Empty<IsolationWindow>();

        if (reader is IIsolationWindowReader cheap)
            return cheap.ReadIsolationWindows(dataPath, log, ct);

        log?.Invoke(
            $"  {reader.Describe()} cannot read isolation windows on their own, so the whole file is "
            + "being measured to get them.");
        return reader.Read(dataPath, log, ct).IsolationWindows;
    }
}
