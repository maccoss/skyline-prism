using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using SkylinePrism.Core.RawData;

namespace SkylinePrism.Core.Qc;

/// <param name="Lanes">Files read at once.</param>
/// <param name="SpectraPerSecond">Aggregate over all lanes - the figure that decides the setting.</param>
/// <param name="PerFileSpectraPerSecond">What one reader achieved, which FALLS as lanes rise.</param>
/// <param name="PeakWorkingSetGb">Process working set at the end of the arm.</param>
public readonly record struct LaneArm(
    int Lanes, double SpectraPerSecond, double PerFileSpectraPerSecond, double PeakWorkingSetGb);

/// <summary>
/// How many instrument files this storage is worth reading at once.
///
/// <para><b>Why this is not a constant.</b> The answer is a property of the storage, not of PRISM.
/// Measured on one SMB share: aggregate throughput rose from ~1,220 spectra/s at two lanes to
/// ~1,990 at eight, while the per-file rate fell from ~650 to ~265 - the share saturating, with more
/// readers dividing a fixed pipe more finely. A local NVMe saturates somewhere else entirely, and a
/// busier share somewhere else again. Shipping any single number bakes one lab's storage into
/// everybody's binary, which is how <c>DefaultLanes</c> came to say 2 and then 4 on no better
/// evidence than which share was measured last.</para>
///
/// <para><b>Why it is cheap.</b> Throughput is visible in the first few thousand spectra; the
/// remaining hundred and sixty thousand only confirm it. Reading a bounded slice turns an hour of
/// arms into a couple of minutes, which is the difference between a probe someone runs and one they
/// read about.</para>
///
/// <para><b>What it cannot tell you.</b> It measures this directory, now. A share under load from
/// someone else's run answers for that load. It also says nothing about MEMORY beyond the arms it
/// ran - and memory, not speed, is the reason to stop short of the fastest arm: eight lanes reached
/// 32 GB of 64 on the measured share, which is the regime where a DuckDB read dies with a native
/// access violation if anything else wants memory (see the caution in CLAUDE.md).</para>
/// </summary>
public static class LaneProbe
{
    /// <summary>Spectra per file per arm. Enough to pass the ramp, small enough to stay cheap.</summary>
    public const int DefaultSliceSpectra = 4000;

    /// <summary>
    /// The arms, in order. Each reads its OWN files, so the full set needs 1+2+4+8 = 15 of them;
    /// arms that cannot be filled with unread files are skipped and said so.
    /// </summary>
    public static readonly IReadOnlyList<int> DefaultArms = new[] { 1, 2, 4, 8 };

    /// <summary>
    /// Time a bounded read of <paramref name="files"/> at each lane count.
    /// </summary>
    /// <remarks>
    /// <para><b>An arm is skipped when there are not enough files to fill it.</b> Running 8 lanes
    /// over 4 files measures 4-way concurrency and labels it 8, which is exactly the mistake that
    /// made an earlier run of this comparison meaningless - the 4-lane and 8-lane arms came back
    /// within noise of each other because both had only four files to read.</para>
    ///
    /// <para><b>Every arm reads DIFFERENT files, and that is the whole design.</b> Re-reading one
    /// set measures the page cache from the second arm onward: on a share whose real rate is about
    /// 650 spectra/s, a warm re-read of the same files clocked 5,097 on a single lane and inverted
    /// the recommendation. Disjoint files cost 1+2+4+8 = 15 of them and every read is cold.</para>
    ///
    /// <para>The arms therefore measure different files, which assumes the directory is reasonably
    /// homogeneous - true of a cohort acquired by one method, and the case this is for. A directory
    /// of wildly mixed acquisitions will give a noisy answer, and the per-file column is where that
    /// shows up.</para>
    /// </remarks>
    public static IReadOnlyList<LaneArm> Run(
        IReadOnlyList<string> files, IsolationScheme scheme, IReadOnlyList<int>? arms = null,
        int sliceSpectra = DefaultSliceSpectra, Action<string>? log = null,
        CancellationToken ct = default)
    {
        if (files is null || files.Count == 0)
            return Array.Empty<LaneArm>();

        var reader = IonAccountingReaders.For(files[0]);
        if (reader is null)
        {
            log?.Invoke("  No instrument-file reader in this build, so there is nothing to probe.");
            return Array.Empty<LaneArm>();
        }

        // An empty claim set: the probe measures READING, and masking is about 1% of a file's cost,
        // so including claims would add the cost of building them to every arm and measure nothing
        // extra.
        var empty = new ClaimedSignalIndex(Array.Empty<ClaimedRegion>());
        var results = new List<LaneArm>();

        var taken = 0;
        foreach (var lanes in (arms ?? DefaultArms).Where(a => a > 0).Distinct().OrderBy(a => a))
        {
            if (taken + lanes > files.Count)
            {
                log?.Invoke(
                    $"  {lanes} lanes: skipped, {files.Count:N0} file(s) available and this arm needs "
                    + $"{lanes:N0} unread ones - an arm with fewer files than lanes measures a "
                    + "smaller concurrency and mislabels it, and re-reading measures the page cache.");
                continue;
            }

            // Disjoint from every earlier arm, so nothing here has been read yet.
            var subset = files.Skip(taken).Take(lanes).ToArray();
            taken += lanes;
            var spectra = 0L;
            var clock = Stopwatch.StartNew();

            Parallel.ForEach(
                subset,
                new ParallelOptions { MaxDegreeOfParallelism = lanes, CancellationToken = ct },
                path =>
                {
                    var request = new IonAccountingRequest(
                        empty, scheme, Array.Empty<string>(), Explained: null,
                        MaxSpectra: sliceSpectra, SliceFromMiddle: true);
                    var record = reader.ReadAccounting(path, request, log: null, ct);
                    Interlocked.Add(ref spectra, record.Ms1Count + record.Ms2Count);
                });

            clock.Stop();
            var seconds = Math.Max(0.001, clock.Elapsed.TotalSeconds);
            var aggregate = spectra / seconds;
            var arm = new LaneArm(
                lanes, aggregate, aggregate / lanes, PeakWorkingSetGb());
            results.Add(arm);

            log?.Invoke(
                $"  {lanes,2} lane(s): {aggregate:N0} spectra/s aggregate, "
                + $"{arm.PerFileSpectraPerSecond:N0} per file, {arm.PeakWorkingSetGb:F1} GB");
        }

        return results;
    }

    /// <summary>
    /// The arm to recommend: the fastest one that is meaningfully faster than the arm below it.
    /// </summary>
    /// <remarks>
    /// A 10% floor, because past the knee the gain is small and the cost is not - each extra lane
    /// holds another file's decode buffers, and running out of memory here does not degrade, it
    /// faults. Where two arms are within 10% of each other the SMALLER one is recommended.
    /// </remarks>
    public static int Recommend(IReadOnlyList<LaneArm> arms, double minimumGain = 0.10)
    {
        if (arms is null || arms.Count == 0)
            return 1;

        var best = arms[0];
        foreach (var arm in arms.Skip(1))
        {
            if (best.SpectraPerSecond <= 0)
            {
                best = arm;
                continue;
            }
            var gain = (arm.SpectraPerSecond - best.SpectraPerSecond) / best.SpectraPerSecond;
            if (gain >= minimumGain)
                best = arm;
        }
        return best.Lanes;
    }

    /// <summary>
    /// Peak working set since the process started - CUMULATIVE, so an arm's figure includes every
    /// earlier arm's peak.
    /// </summary>
    /// <remarks>
    /// Sampling the current working set after an arm finished read the state AFTER a collection and
    /// reported 0.1 GB for the eight-lane arm, which is lower than the one-lane arm and obviously
    /// not what it cost. Since memory grows with lanes and the arms run in increasing order, the
    /// cumulative peak is attributable to the arm that reports it.
    /// </remarks>
    private static double PeakWorkingSetGb()
    {
        using var self = Process.GetCurrentProcess();
        self.Refresh();
        return self.PeakWorkingSet64 / (1024.0 * 1024.0 * 1024.0);
    }
}
