using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.RawData;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// MS2 signal against retention time for ONE replicate: what the instrument acquired, how much of it
/// the run assigns to a peptide, and how much of that belongs to each selected protein list.
///
/// <para>This is the RT view of what <see cref="Ms2SignalAccounting"/> reports as one number per
/// replicate. The same union rule applies - shared signal is counted once - because the traces are
/// built from <see cref="Ms2SignalUnion.MergedRegion"/>, the regions the union actually counted,
/// rather than from the raw transitions.</para>
/// </summary>
/// <remarks>
/// <para><b>Everything is binned to the same grid, by the same rule, so the traces are
/// comparable.</b> Acquired signal arrives per acquisition cycle and assigned signal arrives as
/// integrals over elution spans. Both are "this much signal, collected over this interval", so
/// both are distributed across the bins their interval covers in proportion to the overlap. Using
/// different rules for the two - cycles binned whole, peaks spread - aliased the acquired trace
/// into a sawtooth whenever the bin width was not a multiple of the cycle time.</para>
///
/// <para><b>An integrated peak is spread uniformly over its span.</b> A merged region carries an area
/// and a [start, stop], not a chromatographic shape - Skyline's peak boundaries, not its trace. So the
/// area is distributed evenly across the bins the span covers. That is an approximation and the wrong
/// one in detail: a real peak is concentrated near its apex, so this flattens each peak slightly.
/// Assigning the whole area to the apex bin was the alternative and is worse, because it turns a
/// smooth elution into a spike whose height depends on the bin width. Totals are unaffected either
/// way: the trace sums to the same assigned total whatever the distribution.</para>
/// </remarks>
public sealed class Ms2SignalProfile
{
    private Ms2SignalProfile(
        string sample, double binWidthMin, double[] binStart,
        double[] acquired, double[] assigned, double[][] perList,
        IReadOnlyList<string> listNames, IReadOnlyList<string> listColors, bool hasAcquired)
    {
        Sample = sample;
        BinWidthMin = binWidthMin;
        BinStartMin = binStart;
        Acquired = acquired;
        Assigned = assigned;
        PerList = perList;
        ListNames = listNames;
        ListColors = listColors;
        HasAcquired = hasAcquired;
    }

    public string Sample { get; }

    public double BinWidthMin { get; }

    /// <summary>Left edge of each bin, in minutes.</summary>
    public IReadOnlyList<double> BinStartMin { get; }

    /// <summary>
    /// Acquired MS2 signal per bin. All zero when <see cref="HasAcquired"/> is false - no instrument
    /// file was read - and the plot must then omit the trace rather than draw a floor of zeros.
    /// </summary>
    public IReadOnlyList<double> Acquired { get; }

    /// <summary>Signal assigned to a peptide per bin, shared signal counted once.</summary>
    public IReadOnlyList<double> Assigned { get; }

    /// <summary>One trace per selected list, aligned with <see cref="ListNames"/>.</summary>
    public IReadOnlyList<IReadOnlyList<double>> PerList { get; }

    public IReadOnlyList<string> ListNames { get; }

    public IReadOnlyList<string> ListColors { get; }

    /// <summary>Whether an acquired trace is available at all.</summary>
    public bool HasAcquired { get; }

    public int BinCount => BinStartMin.Count;

    public bool IsEmpty => BinCount == 0;

    /// <summary>
    /// Assigned as a fraction of acquired, per bin, or NaN where nothing was acquired. The reading the
    /// plot exists to give: where in the gradient the analysis is accounting for the signal and where
    /// it is not.
    /// </summary>
    public IReadOnlyList<double> AssignedFraction
    {
        get
        {
            var fraction = new double[BinCount];
            for (var i = 0; i < BinCount; i++)
                fraction[i] = Acquired[i] > 0 ? Assigned[i] / Acquired[i] : double.NaN;
            return fraction;
        }
    }

    /// <summary>
    /// Build the traces.
    /// </summary>
    /// <param name="merged">Regions the union counted once, from
    /// <see cref="Ms2SignalUnion.Compute"/>'s observer.</param>
    /// <param name="cycles">Acquired signal per acquisition cycle, or null when no instrument file was
    /// read - in which case the acquired trace is absent rather than zero.</param>
    /// <param name="binWidthMin">Bin width. The default is a few acquisition cycles wide on a typical
    /// DIA method, which smooths the cycle-to-cycle sawtooth without hiding a real feature.</param>
    public static Ms2SignalProfile Build(
        string sample,
        IReadOnlyList<Ms2SignalUnion.MergedRegion> merged,
        IReadOnlyList<Ms2Cycle>? cycles,
        IReadOnlyList<string> listNames,
        IReadOnlyList<string> listColors,
        double binWidthMin = 0.25)
    {
        if (binWidthMin <= 0)
            throw new ArgumentOutOfRangeException(
                nameof(binWidthMin), binWidthMin, "Bin width must be positive.");

        var names = listNames?.ToList() ?? new List<string>();
        var colors = listColors?.ToList() ?? new List<string>();

        // The grid spans everything either source knows about, so neither trace is silently clipped
        // by the other's range.
        var (from, to) = Extent(merged, cycles);
        if (!double.IsFinite(from) || !double.IsFinite(to) || to < from)
        {
            return new Ms2SignalProfile(
                sample ?? "", binWidthMin, Array.Empty<double>(), Array.Empty<double>(),
                Array.Empty<double>(), Array.Empty<double[]>(), names, colors, false);
        }

        var count = Math.Max(1, (int)Math.Ceiling((to - from) / binWidthMin));
        var binStart = new double[count];
        for (var i = 0; i < count; i++)
            binStart[i] = from + i * binWidthMin;

        var acquired = new double[count];
        var assigned = new double[count];
        var perList = new double[names.Count][];
        for (var l = 0; l < names.Count; l++)
            perList[l] = new double[count];

        var hasAcquired = false;
        if (cycles is not null)
        {
            foreach (var cycle in cycles)
            {
                if (!double.IsFinite(cycle.Ms2Signal))
                    continue;
                // Spread over the cycle's OWN span, exactly as a peak is spread over its elution.
                // Assigning each cycle whole to the bin holding its midpoint was the first attempt
                // and it ALIASES: bins then hold alternating numbers of cycles whenever the bin width
                // is not a multiple of the cycle time, which on a real DIA method (~0.025 min cycles
                // in 0.25 min bins, so 10 or 11 per bin) is a visible ~10% ripple that looks like
                // instrument behaviour and is not. Spreading claims no sub-cycle structure - only
                // that this much signal was collected during this interval, which is what a cycle IS.
                Spread(
                    cycle.RtStartMin, cycle.RtStopMin, cycle.Ms2Signal,
                    from, binWidthMin, count, acquired);
                hasAcquired = true;
            }
        }

        foreach (var region in merged)
        {
            if (!region.Assigned || !double.IsFinite(region.Area) || region.Area <= 0)
                continue;
            Spread(region, from, binWidthMin, count, assigned);
            for (var l = 0; l < names.Count; l++)
                if ((region.ListMask & (1u << l)) != 0)
                    Spread(region, from, binWidthMin, count, perList[l]);
        }

        return new Ms2SignalProfile(
            sample ?? "", binWidthMin, binStart, acquired, assigned, perList, names, colors,
            hasAcquired);
    }

    /// <summary>
    /// Distribute a region's area across the bins its elution span covers, in proportion to how much
    /// of the span falls in each. A region shorter than a bin lands entirely in one.
    /// </summary>
    private static void Spread(
        Ms2SignalUnion.MergedRegion region, double from, double width, int count, double[] into) =>
        Spread(region.RtStart, region.RtStop, region.Area, from, width, count, into);

    /// <summary>
    /// Distribute <paramref name="amount"/> across the bins that [<paramref name="start"/>,
    /// <paramref name="stop"/>] covers, in proportion to the overlap. Shared by peaks and by
    /// acquisition cycles, which is the point: both are "this much signal, over this interval", and
    /// binning them by different rules is what produced an aliased acquired trace.
    /// </summary>
    private static void Spread(
        double start, double stop, double amount, double from, double width, int count, double[] into)
    {
        if (!double.IsFinite(start) || !double.IsFinite(stop) || !double.IsFinite(amount))
            return;

        var span = stop - start;
        if (span <= 0)
        {
            // A zero-width interval still carries signal; put it where it happened.
            var bin = BinOf(start, from, width, count);
            if (bin >= 0)
                into[bin] += amount;
            return;
        }

        var first = BinOf(start, from, width, count, clamp: true);
        var last = BinOf(stop, from, width, count, clamp: true);
        if (first < 0 || last < 0)
            return;

        for (var bin = first; bin <= last; bin++)
        {
            var binFrom = from + bin * width;
            var overlap = Math.Min(stop, binFrom + width) - Math.Max(start, binFrom);
            if (overlap > 0)
                into[bin] += amount * (overlap / span);
        }
    }

    private static int BinOf(double rt, double from, double width, int count, bool clamp = false)
    {
        if (!double.IsFinite(rt))
            return -1;
        var bin = (int)Math.Floor((rt - from) / width);
        if (bin < 0)
            return clamp ? 0 : -1;
        if (bin >= count)
            return clamp ? count - 1 : -1;
        return bin;
    }

    private static (double From, double To) Extent(
        IReadOnlyList<Ms2SignalUnion.MergedRegion> merged, IReadOnlyList<Ms2Cycle>? cycles)
    {
        var from = double.PositiveInfinity;
        var to = double.NegativeInfinity;

        foreach (var region in merged)
        {
            if (!region.Assigned)
                continue;
            if (double.IsFinite(region.RtStart))
                from = Math.Min(from, region.RtStart);
            if (double.IsFinite(region.RtStop))
                to = Math.Max(to, region.RtStop);
        }
        if (cycles is not null)
        {
            foreach (var cycle in cycles)
            {
                if (double.IsFinite(cycle.RtStartMin))
                    from = Math.Min(from, cycle.RtStartMin);
                if (double.IsFinite(cycle.RtStopMin))
                    to = Math.Max(to, cycle.RtStopMin);
            }
        }

        return double.IsFinite(from) && double.IsFinite(to) ? (from, to) : (double.NaN, double.NaN);
    }
}
