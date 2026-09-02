using System;
using System.Collections.Generic;
using System.Linq;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// How much MS2 signal a set of transitions accounts for, counting shared signal ONCE.
///
/// <para><b>Why this is not a sum.</b> A DIA isolation window co-isolates tens of peptides. Two
/// peptides whose fragment ions fall within Skyline's extraction tolerance of each other extract the
/// same detector counts, and adding their areas credits that signal twice — which can push "assigned"
/// past what the instrument actually acquired. So the quantity is the measure of a UNION over regions
/// of MS2 signal space, not a sum over targets.</para>
///
/// <para>A transition occupies <b>(isolation window, product m/z ± extraction tolerance, [RT start,
/// RT stop])</b>. Two are the same signal only when all three coincide. Retention time is part of the
/// test deliberately: two peptides sharing a fragment mass in one window but eluting apart extract the
/// same chromatogram yet integrate <i>different parts</i> of it, which is genuinely different signal
/// and must count twice.</para>
///
/// <para><b>The totals are nested, not a partition.</b> A region claimed by a list peptide counts for
/// that list AND for the assigned total, and two lists may both claim it. That is the question being
/// asked — "what portion of the signal does this panel account for" — rather than "who owns this
/// peptide". Nothing is double-counted <i>within</i> a total.</para>
///
/// <para>Pure: no I/O, no DuckDB, no parquet. Everything it needs arrives in <see cref="Region"/>.</para>
/// </summary>
public static class Ms2SignalUnion
{
    /// <summary>Lists are carried as a bit per list, so one pass computes every total.</summary>
    public const int MaxLists = 32;

    /// <summary>
    /// One integrated transition peak, reduced to what decides whether it shares signal with another.
    /// <paramref name="Area"/> is RAW Skyline peak area, LINEAR — never a normalized or corrected value.
    /// </summary>
    /// <param name="WindowIndex">Isolation window the precursor fell in; regions in different windows never share.</param>
    /// <param name="PeptideId">Which peptide this came from. Only used to tell the two kinds of
    /// collision apart — see <see cref="Result.SharedAcrossPeptides"/>.</param>
    /// <param name="ListMask">Bit per selected protein list claiming this transition's peptide.</param>
    public readonly record struct Region(
        int WindowIndex, double ProductMz, double RtStart, double RtStop, double Area,
        bool Assigned, uint ListMask, int PeptideId = 0);

    /// <param name="AssignedArea">Union measure over regions whose peptide reached the peptide matrix.</param>
    /// <param name="ListArea">Union measure per list, aligned with the caller's list order.</param>
    /// <param name="SummedArea">The naive sum over the same regions — kept so the plot and the log can
    /// show how much double counting the union removed. Never plotted as "assigned".</param>
    /// <param name="LargestGroup">Most transitions merged into one region. A large value means heavy
    /// fragment sharing, and is worth surfacing rather than hiding.</param>
    /// <param name="DuplicateRows">Merged-away rows belonging to the SAME peptide. Skyline exports a
    /// shared peptide once per protein assignment, so one measurement can appear several times with
    /// every field identical. Real over-counting that a sum would commit — but it is bookkeeping, not
    /// co-isolation, and it dominates in practice.</param>
    /// <param name="SharedAcrossPeptides">Merged-away rows belonging to a DIFFERENT peptide than the one
    /// already in the group. This is fragment sharing between co-isolated peptides — the effect the
    /// accounting exists to handle, and the one worth reporting as such.</param>
    /// <remarks>
    /// The two counts split differently on a slice than on a whole run, so read the committed fixture's
    /// figure with care. On the fixture (327 peptides, median 2 precursors per isolation window) the
    /// union removes 86% of the sum and ALL of it is duplicate rows — 1,750 of 1,750, with zero
    /// co-isolation, because a slice that thin barely co-isolates. On the full cohort one replicate has
    /// 462,843 fragment peaks and the union removes 17.7%, split 23,245 duplicate rows to 5,623 genuine
    /// shares. So the fixture can prove the bookkeeping correction works but cannot exercise the
    /// co-isolation one; that is what <c>PRISM_MS2_COHORT</c> in the tests is for.
    /// </remarks>
    public sealed record Result(
        double AssignedArea, IReadOnlyList<double> ListArea, double SummedArea,
        int Regions, int MergedGroups, int LargestGroup, int Skipped,
        int DuplicateRows, int SharedAcrossPeptides);

    /// <summary>
    /// Collapse <paramref name="regions"/> to their union measure.
    ///
    /// <para><b>m/z grouping.</b> Two targets read the same detector counts when the m/z ranges
    /// Skyline extracted them over overlap - <see cref="ProductMassTolerance.WindowAt"/>, which is
    /// Skyline's own window rather than an approximation of it. Grouping is single-linkage over
    /// m/z-sorted transitions, so a dense run of near-identical masses chains into one group;
    /// <see cref="Result.LargestGroup"/> is reported so that chaining is visible rather than silent.</para>
    ///
    /// <para><b>Region magnitude.</b> Within a merged group, <c>max(Area)</c>. Every member extracted
    /// the same m/z from the same window, so their chromatograms are identical and the areas differ
    /// only by integration bounds; the union interval's true area is at least the largest of them, so
    /// this is a conservative under-estimate rather than a guess.</para>
    ///
    /// <para>A region is credited to a list when ANY member is in that list, at the group's full
    /// magnitude — see the type remarks on why the totals nest rather than partition.</para>
    /// </summary>
    /// <param name="listCount">Number of selected lists; bits above this are ignored.</param>
    public static Result Compute(
        IReadOnlyList<Region> regions, ProductMassTolerance tolerance, int listCount)
    {
        if (listCount is < 0 or > MaxLists)
            throw new ArgumentOutOfRangeException(nameof(listCount), listCount,
                $"At most {MaxLists} protein lists can be accounted for at once.");

        var listArea = new double[listCount];
        if (regions.Count == 0)
            return new Result(0, listArea, 0, 0, 0, 0, 0, 0, 0);

        // Non-finite geometry cannot be placed in signal space. Counted, never silently dropped:
        // Skyline writes #N/A for an unintegrated peak, and a run full of them should be visible.
        var usable = new List<Region>(regions.Count);
        var summed = 0.0;
        var skipped = 0;
        foreach (var r in regions)
        {
            if (!double.IsFinite(r.ProductMz) || !double.IsFinite(r.Area)
                || !double.IsFinite(r.RtStart) || !double.IsFinite(r.RtStop) || r.RtStop < r.RtStart)
            {
                skipped++;
                continue;
            }
            usable.Add(r);
            if (r.Assigned)
                summed += r.Area;
        }
        if (usable.Count == 0)
            return new Result(0, listArea, 0, 0, 0, 0, skipped, 0, 0);

        // Sorted so grouping is a sweep, and so the result does not depend on the order rows came back
        // from DuckDB - which is arbitrary, since preserve_insertion_order is off.
        usable.Sort(static (a, b) =>
        {
            var w = a.WindowIndex.CompareTo(b.WindowIndex);
            if (w != 0) return w;
            var m = a.ProductMz.CompareTo(b.ProductMz);
            if (m != 0) return m;
            var s = a.RtStart.CompareTo(b.RtStart);
            return s != 0 ? s : a.RtStop.CompareTo(b.RtStop);
        });

        var acc = new Accumulator { ListArea = listArea };

        var i = 0;
        while (i < usable.Count)
        {
            // One m/z cluster: same isolation window, each transition's extraction window overlapping
            // the previous one's. Both windows are evaluated rather than one width reused - they widen
            // with m/z for every analyzer except QIT, so a single width is only right at one mass.
            var j = i + 1;
            var previous = tolerance.WindowAt(usable[i].ProductMz);
            while (j < usable.Count && usable[j].WindowIndex == usable[i].WindowIndex)
            {
                var current = tolerance.WindowAt(usable[j].ProductMz);
                if (!previous.Overlaps(current))
                    break;
                previous = current;
                j++;
            }

            MergeRtWithin(usable, i, j, listCount, acc);
            i = j;
        }

        return new Result(
            acc.Assigned, listArea, summed, usable.Count, acc.MergedGroups, acc.LargestGroup, skipped,
            acc.DuplicateRows, acc.SharedAcrossPeptides);
    }

    /// <summary>
    /// Within one m/z cluster, merge transitions whose RT spans overlap and credit each merged span
    /// once. The cluster is re-sorted by RT start because the outer sweep ordered it by m/z.
    /// </summary>
    private static void MergeRtWithin(
        List<Region> all, int from, int to, int listCount, Accumulator acc)
    {
        var cluster = new List<Region>(to - from);
        for (var k = from; k < to; k++)
            cluster.Add(all[k]);
        cluster.Sort(static (a, b) => a.RtStart.CompareTo(b.RtStart));

        var groupPeptides = new HashSet<int>();
        var openStop = double.NegativeInfinity;
        var groupArea = 0.0;
        var groupAssigned = false;
        uint groupMask = 0;
        var groupSize = 0;

        void Close()
        {
            if (groupSize == 0)
                return;
            acc.MergedGroups++;
            acc.LargestGroup = Math.Max(acc.LargestGroup, groupSize);
            if (groupAssigned)
            {
                acc.Assigned += groupArea;
                for (var l = 0; l < listCount; l++)
                    if ((groupMask & (1u << l)) != 0)
                        acc.ListArea[l] += groupArea;
            }
            groupArea = 0;
            groupAssigned = false;
            groupMask = 0;
            groupSize = 0;
            groupPeptides.Clear();
        }

        foreach (var r in cluster)
        {
            // Touching spans do not overlap: a peak ending exactly where the next begins shares no
            // scan. Strict inequality keeps adjacent-but-distinct elutions counted separately.
            if (groupSize > 0 && r.RtStart >= openStop)
                Close();

            if (groupSize > 0)
            {
                // Every row after the first in a group is one a sum would have counted again. Which
                // KIND it is decides what the number means.
                if (groupPeptides.Contains(r.PeptideId))
                    acc.DuplicateRows++;
                else
                    acc.SharedAcrossPeptides++;
            }
            groupPeptides.Add(r.PeptideId);
            groupArea = Math.Max(groupArea, r.Area);
            groupAssigned |= r.Assigned;
            groupMask |= r.ListMask;
            openStop = groupSize == 0 ? r.RtStop : Math.Max(openStop, r.RtStop);
            groupSize++;
        }
        Close();
    }

    /// <summary>
    /// Running totals across clusters. A class rather than <c>ref</c> parameters because the per-group
    /// close-out is a local function, and a local function cannot capture a <c>ref</c> parameter.
    /// </summary>
    private sealed class Accumulator
    {
        public double Assigned;
        public double[] ListArea = Array.Empty<double>();
        public int MergedGroups;
        public int LargestGroup;
        public int DuplicateRows;
        public int SharedAcrossPeptides;
    }
}
