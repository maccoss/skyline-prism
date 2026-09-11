using System;
using System.Collections.Generic;
using System.Linq;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// One region of signal space a peptide claims: an m/z extraction window over a retention-time span,
/// within one isolation window (or, at MS1, within the survey scan).
/// </summary>
/// <param name="MsLevel">1 for a precursor isotope, 2 for a fragment.</param>
/// <param name="WindowIndex">
/// Which isolation window the claim sits in. Always <see cref="AnyWindow"/> at MS1: a survey scan
/// covers the whole range, so every MS1 claim competes with every other.
/// </param>
/// <param name="MzLow">Inclusive low edge of the extraction window Skyline used.</param>
/// <param name="MzHigh">Inclusive high edge.</param>
/// <param name="RtStart">Peak start, as Skyline integrated it.</param>
/// <param name="RtStop">Peak end.</param>
/// <param name="ListMask">Bit per selected protein list whose peptides claim this region.</param>
public readonly record struct ClaimedRegion(
    int MsLevel,
    int WindowIndex,
    double MzLow,
    double MzHigh,
    double RtStart,
    double RtStop,
    uint ListMask);

/// <summary>
/// Answers, for one spectrum, how much of its signal falls inside the regions peptides claim.
/// </summary>
/// <remarks>
/// <para><b>Why this exists rather than a sum of Skyline's peak areas.</b> Summing per-transition
/// areas counts shared signal more than once: in a DIA window two peptides whose fragments fall
/// within the extraction tolerance of each other extract the SAME detector counts, and crediting
/// both can push the assigned total past what was acquired. Working from the spectrum instead
/// removes the problem rather than correcting it - the claims are merged into disjoint m/z ranges
/// first, so a peak inside two peptides' windows is counted once because there is only one detector
/// reading of it. Nothing has to detect or subtract the overlap.</para>
///
/// <para>It also removes two corrections that the area route needed and could not always make. No
/// background is subtracted from a spectrum, so none has to be added back (Skyline's <c>Area</c> is
/// background-subtracted and the <c>Background</c> column is not always exported). And multiplying
/// the summed intensity by the scan's ion injection time gives a count of ions in the same units as
/// the scan's own total, so assigned over acquired is a genuine dimensionless fraction.</para>
///
/// <para><b>Pure by design.</b> It takes m/z and intensity arrays, so it can be tested against
/// synthetic spectra with answers worked out by hand - which matters more here than anywhere else in
/// the accounting, because a mistake in the interval arithmetic produces a plausible wrong number
/// rather than a crash.</para>
///
/// <para><b>Sweeps rather than searches.</b> A replicate claims on the order of half a million
/// regions, far too many to test against every scan. Scans within one isolation window arrive in
/// retention-time order, so each window keeps a cursor: claims open as their peak begins and close
/// as it ends, and only the few hundred active at that moment are merged. The merged ranges are
/// rebuilt only when the active set actually changes.</para>
/// </remarks>
public sealed class ClaimedSignalIndex
{
    /// <summary>Window index for a claim not confined to an isolation window, i.e. every MS1 claim.</summary>
    public const int AnyWindow = -1;

    private sealed class Lane
    {
        public ClaimedRegion[] ByStart = Array.Empty<ClaimedRegion>();
        public int Cursor;                               // next claim in ByStart not yet opened
        public readonly List<ClaimedRegion> Active = new();
        public double MergedAtRt = double.NaN;           // RT the merged ranges were built for
        public double LastRt = double.NegativeInfinity;  // to notice scans arriving out of order
        public bool Dirty = true;
        public double[] Low = Array.Empty<double>();     // merged, disjoint, ascending
        public double[] High = Array.Empty<double>();
        public uint[] Mask = Array.Empty<uint>();        // union of list bits over each merged range
    }

    private readonly Dictionary<(int Level, int Window), Lane> _lanes = new();
    private readonly int _listCount;

    /// <param name="listCount">How many protein lists have a bit in <see cref="ClaimedRegion.ListMask"/>.</param>
    public ClaimedSignalIndex(IEnumerable<ClaimedRegion> regions, int listCount = 0)
    {
        _listCount = Math.Max(0, listCount);
        foreach (var group in (regions ?? Array.Empty<ClaimedRegion>())
                     .Where(r => r.MzHigh >= r.MzLow && r.RtStop >= r.RtStart)
                     .GroupBy(r => (r.MsLevel, r.WindowIndex)))
        {
            _lanes[group.Key] = new Lane
            {
                // Sorted by peak start, which is the order the sweep opens them in.
                ByStart = group.OrderBy(r => r.RtStart).ToArray(),
            };
        }
    }

    /// <summary>Total claims held, for a log line that says the geometry actually loaded.</summary>
    public int RegionCount => _lanes.Values.Sum(l => l.ByStart.Length);

    /// <summary>
    /// Scans that arrived EARLIER than the previous scan of their own lane.
    ///
    /// <para>The sweep is forward-only - a claim that has closed is not reopened - so this must be
    /// zero for the assigned total to be right. It is zero on every real file measured, because
    /// spectra come in acquisition order and each isolation window fires once per cycle. But if a
    /// reader ever returned them otherwise the only symptom would be an assigned total that is
    /// quietly too low, with nothing to say so. Counting it turns that into something a caller can
    /// report.</para>
    /// </summary>
    public int BackwardScans { get; private set; }

    /// <summary>
    /// Sum the intensity of <paramref name="mz"/>/<paramref name="intensity"/> that falls inside the
    /// regions claimed at this retention time, and the part of it each protein list claims.
    /// </summary>
    /// <param name="msLevel">1 or 2.</param>
    /// <param name="windowIndex">
    /// The scan's isolation window, or <see cref="AnyWindow"/> for a survey scan. An MS2 scan whose
    /// window matched no claim contributes nothing, which is correct: no peptide was extracted there.
    /// </param>
    /// <param name="perList">
    /// Filled with the per-list sums; must be at least as long as the list count. Added to, not
    /// overwritten, so a caller can accumulate across a cycle.
    /// </param>
    /// <returns>Summed intensity inside the claimed regions. NOT yet multiplied by injection time.</returns>
    public double Claimed(
        int msLevel, int windowIndex, double rtMinutes,
        ReadOnlySpan<double> mz, ReadOnlySpan<double> intensity, Span<double> perList)
    {
        if (mz.Length == 0 || mz.Length != intensity.Length)
            return 0;
        if (!_lanes.TryGetValue((msLevel, windowIndex), out var lane))
            return 0;
        if (!double.IsFinite(rtMinutes))
            return 0;

        Advance(lane, rtMinutes);
        if (lane.Low.Length == 0)
            return 0;

        // Both sides ascending, so one walk rather than a search per peak. A spectrum's m/z array is
        // required to be sorted; the merged ranges are disjoint and sorted by construction.
        var total = 0.0;
        var range = 0;
        for (var i = 0; i < mz.Length; i++)
        {
            var m = mz[i];
            while (range < lane.High.Length && lane.High[range] < m)
                range++;
            if (range >= lane.Low.Length)
                break;
            if (m < lane.Low[range])
                continue;   // between ranges

            var value = intensity[i];
            if (!double.IsFinite(value) || value <= 0)
                continue;
            total += value;

            var mask = lane.Mask[range];
            for (var l = 0; l < _listCount && l < perList.Length; l++)
                if ((mask & (1u << l)) != 0)
                    perList[l] += value;
        }
        return total;
    }

    /// <summary>
    /// Open claims whose peak has started and close those whose peak has ended, then rebuild the
    /// merged ranges if the active set moved.
    /// </summary>
    private void Advance(Lane lane, double rt)
    {
        if (rt < lane.LastRt)
            BackwardScans++;
        lane.LastRt = rt;

        while (lane.Cursor < lane.ByStart.Length && lane.ByStart[lane.Cursor].RtStart <= rt)
        {
            lane.Active.Add(lane.ByStart[lane.Cursor++]);
            lane.Dirty = true;
        }

        // Removal is a compaction rather than a heap: the active set is small and this runs once per
        // scan, where a priority queue would cost more in allocation than it saves in comparisons.
        var kept = 0;
        for (var i = 0; i < lane.Active.Count; i++)
        {
            if (lane.Active[i].RtStop >= rt)
                lane.Active[kept++] = lane.Active[i];
            else
                lane.Dirty = true;
        }
        if (kept != lane.Active.Count)
            lane.Active.RemoveRange(kept, lane.Active.Count - kept);

        if (!lane.Dirty && lane.MergedAtRt == rt)
            return;
        Merge(lane);
        lane.MergedAtRt = rt;
        lane.Dirty = false;
    }

    /// <summary>
    /// Collapse the active claims into disjoint ascending m/z ranges. This is where shared signal
    /// stops being counted twice: two peptides claiming overlapping windows become ONE range, and
    /// the range carries the union of their list bits so each list is still credited with it.
    /// </summary>
    private static void Merge(Lane lane)
    {
        if (lane.Active.Count == 0)
        {
            lane.Low = Array.Empty<double>();
            lane.High = Array.Empty<double>();
            lane.Mask = Array.Empty<uint>();
            return;
        }

        var ordered = lane.Active.OrderBy(r => r.MzLow).ToArray();
        var low = new List<double>(ordered.Length);
        var high = new List<double>(ordered.Length);
        var mask = new List<uint>(ordered.Length);

        var curLow = ordered[0].MzLow;
        var curHigh = ordered[0].MzHigh;
        var curMask = ordered[0].ListMask;
        for (var i = 1; i < ordered.Length; i++)
        {
            var r = ordered[i];
            if (r.MzLow <= curHigh)
            {
                // Touching counts as overlapping: a peak exactly on the shared edge is one reading.
                if (r.MzHigh > curHigh)
                    curHigh = r.MzHigh;
                curMask |= r.ListMask;
                continue;
            }
            low.Add(curLow);
            high.Add(curHigh);
            mask.Add(curMask);
            curLow = r.MzLow;
            curHigh = r.MzHigh;
            curMask = r.ListMask;
        }
        low.Add(curLow);
        high.Add(curHigh);
        mask.Add(curMask);

        lane.Low = low.ToArray();
        lane.High = high.ToArray();
        lane.Mask = mask.ToArray();
    }
}
