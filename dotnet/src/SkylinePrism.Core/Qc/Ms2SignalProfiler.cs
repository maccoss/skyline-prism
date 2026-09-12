using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.IO;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// Builds the MS2 signal profile for ONE replicate: acquired, assigned and per-list signal against
/// retention time.
/// </summary>
/// <remarks>
/// <para>Separate from <see cref="Ms2SignalAccounting"/> because the two want opposite things from
/// the same scan. The accounting reduces a replicate to a handful of totals and keeps those for the
/// whole cohort; the profile needs every merged REGION of one replicate, with its retention-time
/// span intact. Regions are far too many to persist - the point of the union is that it reduces
/// them - so a profile is rebuilt on demand for whichever replicate is being looked at.</para>
///
/// <para>That is one scan of one replicate's slice of <c>merged_data/</c>, which is seconds rather
/// than the minutes the cohort-wide accounting takes. Still off the UI thread, and still worth
/// caching per replicate in a caller that lets the user flip between them.</para>
/// </remarks>
public static class Ms2SignalProfiler
{
    /// <summary>
    /// The profile for one replicate, or null when the merged data or the required columns are
    /// missing - the same conditions under which the accounting itself declines.
    /// </summary>
    /// <param name="sample">PRISM sample id, as it appears in the accounting's rows.</param>
    /// <param name="binWidthMin">Retention-time bin width. Re-binning is cheap once loaded, so a
    /// caller offering a control should rebuild rather than cache one width.</param>
    /// <remarks>
    /// The acquired trace comes from <see cref="Ms2AcquiredSignal.CyclesFile"/> when a raw read has
    /// been done for this directory, and is simply absent otherwise - which
    /// <see cref="Ms2SignalProfile"/> renders as "no acquired trace" rather than as a flat zero.
    /// </remarks>
    public static Ms2SignalProfile? ForReplicate(
        string outputDir,
        string sample,
        IsolationScheme scheme,
        ProductMassTolerance tolerance,
        IReadOnlyList<ProteinList> lists,
        double binWidthMin = Ms2SignalProfile.DefaultBinWidthMin,
        Action<string>? log = null,
        int memoryBudgetMb = 0,
        Ms2SignalMeasure measure = Ms2SignalMeasure.Signal)
    {
        var mergedRoot = MergedDataset.Locate(outputDir) ?? string.Empty;
        if (!MergedDataset.Exists(mergedRoot))
        {
            log?.Invoke("  No merged_data/ in the output directory, so no MS2 profile can be built.");
            return null;
        }

        var dataset = MergedDataset.Open(mergedRoot);
        var cols = Ms2SignalRegions.Resolve(
            ParquetTable.ReadColumnNames(dataset.RepresentativeFile()).ToList());
        if (cols is null)
        {
            log?.Invoke("  The merged table lacks Product Mz, Start Time or End Time, so no MS2 "
                + "profile can be built.");
            return null;
        }

        var safeLists = lists ?? Array.Empty<ProteinList>();
        var classified = Ms2SignalPeptides.Classify(outputDir, safeLists);

        var loaded = Ms2SignalRegions.Load(
            dataset, cols, sample, scheme, classified.Classes, memoryBudgetMb, measure);
        if (loaded.Regions.Count == 0)
        {
            log?.Invoke($"  No MS2 regions for {sample}; nothing to profile.");
            return null;
        }

        // The union's observer is the only way to see the regions it kept - the Result it returns is
        // already reduced to totals, which is exactly what a profile cannot use.
        var merged = new List<Ms2SignalUnion.MergedRegion>();
        Ms2SignalUnion.Compute(loaded.Regions, tolerance, safeLists.Count, merged.Add);

        // Absent unless a raw read has been done for this directory. Empty means no acquired trace,
        // which is a different statement from an acquired trace of zero.
        var cycles = Ms2AcquiredSignal.ReadCycles(outputDir, sample);

        return Ms2SignalProfile.Build(
            sample,
            merged,
            cycles.Count > 0 ? cycles : null,
            classified.ListNames,
            safeLists.Select(l => l.ColorHex).ToList(),
            binWidthMin);
    }

    /// <summary>
    /// Which replicates to profile in a report that cannot show all of them: the best, the median
    /// and the worst.
    /// </summary>
    /// <remarks>
    /// <para>Three panels rather than one, because the median alone cannot show the spread and a
    /// single bad replicate is the thing a reader most wants to find. Ranked on the acquired
    /// FRACTION when a denominator exists, which is the quantity that actually means "how well did
    /// this replicate do" - and on assigned signal when it does not, which is weaker (a replicate
    /// can rank low simply for having been injected lighter) and so is labelled differently by the
    /// caller.</para>
    ///
    /// <para>Fewer than three replicates gives fewer than three panels, without repeating one under
    /// two headings.</para>
    /// </remarks>
    public static IReadOnlyList<(string Sample, string Role)> ChooseRepresentatives(
        Ms2SignalAccounting.Result result)
    {
        if (result is null || result.Rows.Count == 0)
            return Array.Empty<(string, string)>();

        var byFraction = result.HasAcquired;
        var ranked = result.Rows
            .Select(r => (Row: r, Key: byFraction ? r.AcquiredFraction : r.AssignedArea))
            .Where(t => double.IsFinite(t.Key))
            .OrderBy(t => t.Key)
            .ToList();

        if (ranked.Count == 0)
            return Array.Empty<(string, string)>();
        if (ranked.Count == 1)
            return new[] { (ranked[0].Row.Sample, "only replicate") };
        if (ranked.Count == 2)
            return new[] { (ranked[1].Row.Sample, "best"), (ranked[0].Row.Sample, "worst") };

        return new[]
        {
            (ranked[^1].Row.Sample, "best"),
            (ranked[ranked.Count / 2].Row.Sample, "median"),
            (ranked[0].Row.Sample, "worst"),
        };
    }
}
