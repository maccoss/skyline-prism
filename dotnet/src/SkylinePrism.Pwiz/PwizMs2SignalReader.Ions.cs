using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using Pwiz.Data.Common.Cv;
using Pwiz.Data.Common.Params;
using Pwiz.Data.MsData;
using Pwiz.Data.MsData.Readers;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.RawData;

namespace SkylinePrism.Pwiz;

/// <summary>
/// Ion accounting: acquired and assigned ions, both MS levels, in ONE pass over the file.
/// </summary>
/// <remarks>
/// <para><b>Why one pass matters here and not for the TIC total.</b> The acquired total alone can be
/// read from scan headers, or from the file's own TIC chromatogram. Assigned cannot: it needs each
/// spectrum's peak arrays masked against the regions peptides claim. A cohort is a terabyte on a
/// network share, so the file is opened once and both halves and every cycle come off the same
/// walk.</para>
///
/// <para><b>Cost, measured over the share rather than out of the page cache:</b> 600-820 spectra/s,
/// which is 200-290 s for a typical 4-5 GB Thermo file of 160,000-170,000 spectra. An early figure of
/// 20,000 spectra/s and "roughly 8 s per file" was a warm-cache artifact - the probe re-read a file
/// the previous run had just pulled across, so it measured local RAM. Quote a throughput here only
/// from a cold file on the share.</para>
///
/// <para><b>Both sides are summed from the SAME array.</b> The denominator is the sum of the
/// spectrum's own peak intensities, not the reported total-ion-current cvParam. The cvParam is the
/// instrument's pre-centroiding total and includes signal that is not in the centroided peak list at
/// all, so dividing a centroided numerator by it would understate the fraction by however much
/// centroiding removed. The reported total is still accumulated, separately, so the gap between the
/// two is visible rather than assumed.</para>
///
/// <para><b>The unit is intensity times injection time IN SECONDS.</b> The reported intensity is a
/// RATE - ions per second - so the injection time has to be in seconds for the product to be a count
/// of ions. The cvParam is specified in milliseconds, so this is a factor of 1000, and getting it
/// wrong does not disturb the fraction at all: both sides carry the same weighting and the ratio
/// cancels. It disturbs only the absolute totals, which is what makes it easy to ship - the check
/// that catches it is per-scan plausibility against the AGC target, not the fraction. At one point
/// this code multiplied by milliseconds and reported 3.7e5 ions per MS1 scan as 3.7e8.</para>
///
/// <para>Dropping the factor ENTIRELY is a different and larger error, and the one this feature
/// replaces: a rate summed over scans is not a count, so dividing an intensity-time integral by it
/// is dimensionally meaningless.</para>
/// </remarks>
public sealed partial class PwizMs2SignalReader
{
    /// <inheritdoc />
    public IonAccountingRecord ReadAccounting(
        string dataPath, IonAccountingRequest request, Action<string>? log = null,
        CancellationToken ct = default)
    {
        if (request is null)
            throw new ArgumentNullException(nameof(request));

        if (!File.Exists(dataPath) && !Directory.Exists(dataPath))
        {
            return IonAccountingRecord.Unavailable(
                dataPath, Ms2ReadStatus.NotFound, Describe(), "No such file.");
        }

        try
        {
            using var msd = new MSData();
            ReaderList.Default.Read(
                dataPath, msd, new ReaderConfig { CombineIonMobilitySpectra = true });
            return AccountingFromSpectrumWalk(msd, dataPath, request, log, ct);
        }
        catch (OperationCanceledException)
        {
            return IonAccountingRecord.Unavailable(
                dataPath, Ms2ReadStatus.Cancelled, Describe(), "Cancelled.");
        }
        catch (Exception ex)
        {
            // Never throws, by contract: one unreadable file in a cohort must cost only that file.
            log?.Invoke($"  Could not read {Path.GetFileName(dataPath)}: {ex.Message}");
            return IonAccountingRecord.Unavailable(
                dataPath, Ms2ReadStatus.Failed, Describe(), ex.Message);
        }
    }

    private IonAccountingRecord AccountingFromSpectrumWalk(
        MSData msd, string dataPath, IonAccountingRequest request, Action<string>? log,
        CancellationToken ct)
    {
        var spectra = msd.Run.SpectrumList
            ?? throw new InvalidDataException($"No spectra in {Path.GetFileName(dataPath)}.");

        var claims = request.Claims;
        var listCount = request.ListCount;
        var ms1ByList = new double[listCount];
        var ms2ByList = new double[listCount];
        // One scan's per-list claims, reused. ClaimedSignalIndex fills this with RAW INTENSITY -
        // the same thing it returns as its scalar, which the caller then multiplies by the
        // injection time - so the per-list arrays have to be weighted here too. Accumulating them
        // straight out of Claimed left them a sum of rates under columns named ms1_assigned /
        // ms2_assigned and documented as ion counts: a panel's "share of the assigned total" then
        // carried units of 1/time, which is the same class of error as the accounting this feature
        // replaced. Weighted per scan and not at the end, because the injection time varies.
        var scanByList = listCount > 0 ? new double[listCount] : Array.Empty<double>();

        var explained = request.Explained;
        var cycles = new List<IonCycle>();
        int ms1 = 0, ms2 = 0, noInjection = 0, outsideScheme = 0, unsorted = 0;
        double ms1Acquired = 0, ms2Acquired = 0, ms1Assigned = 0, ms2Assigned = 0;
        // The same sums WITHOUT the injection-time weighting - what the instrument reports as TIC.
        // Accumulated over exactly the same scans as the ion totals, which is why they are gathered
        // here rather than in a pass of their own: a scan with no injection time is excluded from
        // both, so the two views always describe the same set of spectra and their fractions are
        // answering the same question of the same data.
        double ms1Signal = 0, ms2Signal = 0, ms1SignalAssigned = 0, ms2SignalAssigned = 0;
        double ms2SignalExplained = 0;
        double ms2Explained = 0;
        double reportedMs1 = 0, reportedMs2 = 0;
        double rtFirst = double.NaN, rtLast = double.NaN;

        // A cycle opens at each MS1 and closes at the next. Unlike the acquired-only walk there is no
        // isolation-wrap fallback: ion accounting needs MS1 totals, so a file with no MS1 has no MS1
        // half to bound anything with, and one cycle covering the run is the honest answer.
        var cycleStart = double.NaN;
        var cycleStop = double.NaN;
        int cycleMs1 = 0, cycleMs2 = 0;
        double cMs1Acq = 0, cMs2Acq = 0, cMs1Asg = 0, cMs2Asg = 0, cMs2Exp = 0;
        double cMs1Sig = 0, cMs2Sig = 0, cMs1SigAsg = 0, cMs2SigAsg = 0, cMs2SigExp = 0;

        void CloseCycle()
        {
            if (cycleMs1 == 0 && cycleMs2 == 0)
                return;
            cycles.Add(new IonCycle(
                cycles.Count, cycleStart, cycleStop, cycleMs1, cycleMs2,
                cMs1Acq, cMs2Acq, cMs1Asg, cMs2Asg, cMs2Exp,
                cMs1Sig, cMs2Sig, cMs1SigAsg, cMs2SigAsg, cMs2SigExp));
            cycleMs1 = cycleMs2 = 0;
            cMs1Acq = cMs2Acq = cMs1Asg = cMs2Asg = cMs2Exp = 0;
            cMs1Sig = cMs2Sig = cMs1SigAsg = cMs2SigAsg = cMs2SigExp = 0;
            cycleStart = double.NaN;
            cycleStop = double.NaN;
        }

        // Reused across spectra, grown as needed: only touched for a spectrum whose m/z array is
        // not ascending. The mzML specification requires ascending m/z, and vendor readers do not
        // always deliver it - zero spectra across all 39 Thermo files of one cohort, and a few
        // hundred in EVERY file of another acquired on a different instrument. The masking sweep is
        // forward-only, so an unsorted array would silently under-count; this path is load-bearing,
        // not theoretical, and the count is reported per file.
        double[] sortedMz = Array.Empty<double>();
        double[] sortedIntensity = Array.Empty<double>();

        // Split so the two costs can be told apart. They are both per spectrum, and the first full
        // walk was 27x slower than a sequential copy of the same file - which is a statement about
        // one of these two and not the other.
        var readTicks = 0L;
        var maskTicks = 0L;
        var walk = Stopwatch.StartNew();

        // The probe's bound. Counted in spectra CONSIDERED, not spectra of a usable level, so the
        // slice is a fixed amount of read work regardless of the file's MS1/MS2 mix - which is what
        // makes two arms of a throughput comparison comparable.
        var slice = request.MaxSpectra > 0
            ? Math.Min(request.MaxSpectra, spectra.Count)
            : spectra.Count;
        var from = request.SliceFromMiddle ? Math.Max(0, (spectra.Count - slice) / 2) : 0;
        var limit = Math.Min(spectra.Count, from + slice);

        for (var i = from; i < limit; i++)
        {
            if ((i & 0x3FF) == 0)
                ct.ThrowIfCancellationRequested();

            var readStart = Stopwatch.GetTimestamp();
            var spectrum = spectra.GetSpectrum(i, getBinaryData: true);
            readTicks += Stopwatch.GetTimestamp() - readStart;
            var level = spectrum.Params.CvParamValueOrDefault(CVID.MS_ms_level, 0);
            if (level is not (1 or 2))
                continue;

            var scan = spectrum.ScanList.Scans.Count > 0 ? spectrum.ScanList.Scans[0] : null;
            var rt = Minutes(scan?.CvParam(CVID.MS_scan_start_time));
            if (double.IsFinite(rt))
            {
                if (!double.IsFinite(rtFirst))
                    rtFirst = rt;
                rtLast = rt;
            }

            // SECONDS, honoring the unit the file declares - the same rule as the retention time
            // below, and for the same reason. Measured on a real Astral file: 7.012 to 50.013
            // milliseconds, i.e. 0.007 to 0.050 s.
            //
            // A scan with no injection time cannot be converted to ions at all, so it is EXCLUDED
            // from both totals rather than given an invented weight. Weighting it 1 would make that
            // one scan count as if it had injected for a full second - about 141x a typical scan
            // here - which would distort the totals far more than omitting it. The fraction stays
            // valid either way; the record reports how many were left out.
            var injection = Seconds(scan?.CvParam(CVID.MS_ion_injection_time));
            if (!double.IsFinite(injection) || injection <= 0)
            {
                noInjection++;
                continue;
            }

            var mzArray = spectrum.GetMZArray();
            var intensityArray = spectrum.GetIntensityArray();
            var mz = mzArray is null ? default : CollectionsMarshal.AsSpan(mzArray.Data);
            var intensity = intensityArray is null
                ? default
                : CollectionsMarshal.AsSpan(intensityArray.Data);

            if (mz.Length != intensity.Length)
            {
                // Mismatched arrays are a broken spectrum, not a reason to abandon the file.
                mz = default;
                intensity = default;
            }
            else if (!IsAscending(mz))
            {
                unsorted++;
                var peaks = mz.Length;
                Sort(mz, intensity, ref sortedMz, ref sortedIntensity);
                mz = sortedMz.AsSpan(0, peaks);
                intensity = sortedIntensity.AsSpan(0, peaks);
            }

            var summed = Sum(intensity);
            var reported = spectrum.Params.CvParamValueOrDefault(
                CVID.MS_total_ion_current, double.NaN);

            var windowIndex = ClaimedSignalIndex.AnyWindow;
            if (level == 2)
            {
                var target = double.NaN;
                if (spectrum.Precursors.Count > 0)
                {
                    target = spectrum.Precursors[0].IsolationWindow
                        .CvParamValueOrDefault(CVID.MS_isolation_window_target_m_z, double.NaN);
                }
                windowIndex = request.WindowIndexFor(target, rt);
                if (windowIndex < 0)
                {
                    // The scan still acquired ions - they are counted - but no claim can match it.
                    outsideScheme++;
                }
            }

            // The masked sum. A window index of -1 finds no lane and returns 0 without searching.
            var maskStart = Stopwatch.GetTimestamp();
            if (listCount > 0)
                Array.Clear(scanByList);
            var claimed = claims.Claimed(
                level, windowIndex, rt, mz, intensity, scanByList);

            // The second mask, against everything the peptides can account for. MS2 only: the
            // explained index carries no MS1 lane, because at MS1 the theoretical claim IS the
            // precursor isotope envelope Skyline already extracts and the two totals cannot differ.
            // Per-list sums are deliberately not accumulated here - the lists answer "what share of
            // the assigned signal does this panel hold", which is a question about the quantified
            // set, and a second set of per-list arrays would double the memory for a number no plot
            // shows.
            var explainedClaimed = 0.0;
            if (explained is not null && level == 2)
            {
                explainedClaimed = explained.Claimed(
                    level, windowIndex, rt, mz, intensity, Span<double>.Empty);
            }
            maskTicks += Stopwatch.GetTimestamp() - maskStart;

            var acquiredIons = summed * injection;
            var assignedIons = claimed * injection;
            if (listCount > 0)
            {
                var byList = level == 1 ? ms1ByList : ms2ByList;
                for (var l = 0; l < listCount; l++)
                    byList[l] += scanByList[l] * injection;
            }
            var reportedIons = double.IsFinite(reported) && reported > 0 ? reported * injection : 0;

            if (level == 1)
            {
                CloseCycle();
                ms1++;
                ms1Acquired += acquiredIons;
                ms1Assigned += assignedIons;
                reportedMs1 += reportedIons;
                cycleMs1++;
                cMs1Acq += acquiredIons;
                cMs1Asg += assignedIons;
                ms1Signal += summed;
                ms1SignalAssigned += claimed;
                cMs1Sig += summed;
                cMs1SigAsg += claimed;
            }
            else
            {
                ms2++;
                ms2Acquired += acquiredIons;
                ms2Assigned += assignedIons;
                reportedMs2 += reportedIons;
                cycleMs2++;
                cMs2Acq += acquiredIons;
                cMs2Asg += assignedIons;
                ms2Signal += summed;
                ms2SignalAssigned += claimed;
                cMs2Sig += summed;
                cMs2SigAsg += claimed;

                var explainedIons = explainedClaimed * injection;
                ms2Explained += explainedIons;
                cMs2Exp += explainedIons;
                ms2SignalExplained += explainedClaimed;
                cMs2SigExp += explainedClaimed;
            }

            if (!double.IsFinite(cycleStart))
                cycleStart = rt;
            cycleStop = rt;
        }
        CloseCycle();

        var record = new IonAccountingRecord(
            dataPath,
            ms1 + ms2 > 0 ? Ms2ReadStatus.Ok : Ms2ReadStatus.Failed,
            Describe(), ms1, ms2,
            ms1Acquired, ms2Acquired, ms1Assigned, ms2Assigned,
            ms2Explained, explained is not null,
            ms1ByList, ms2ByList, rtFirst, rtLast, noInjection, outsideScheme, cycles,
            ms1 + ms2 > 0 ? null : "The file has no MS1 or MS2 spectra.",
            RunStart(msd),
            ms1Signal, ms2Signal, ms1SignalAssigned, ms2SignalAssigned, ms2SignalExplained,
            // Measured, not merely zero. Every walk that reaches here accumulated it, so the flag
            // is what tells a plot apart from a cache written before the columns existed.
            HasSignal: true);

        Report(record, claims, reportedMs1, reportedMs2, unsorted, log);
        log?.Invoke(
            $"    timing: {walk.Elapsed.TotalSeconds:F1} s total - "
            + $"{readTicks / (double)Stopwatch.Frequency:F1} s reading spectra, "
            + $"{maskTicks / (double)Stopwatch.Frequency:F1} s masking them "
            + $"({(ms1 + ms2) / Math.Max(0.001, walk.Elapsed.TotalSeconds):N0} spectra/s).");
        return record;
    }

    /// <summary>
    /// What the walk found, in the run log. Written to be readable as a diagnosis rather than as
    /// telemetry: the two things that make a fraction wrong - scans that matched no isolation window,
    /// and a fraction over 1 - are named outright.
    /// </summary>
    private static void Report(
        IonAccountingRecord record, ClaimedSignalIndex claims,
        double reportedMs1, double reportedMs2, int unsorted, Action<string>? log)
    {
        if (log is null)
            return;

        var name = Path.GetFileName(record.DataPath);
        log($"  {name}: {record.Ms1Count:N0} MS1 and {record.Ms2Count:N0} MS2 spectra, "
            + $"{claims.RegionCount:N0} claimed regions, {record.Cycles.Count:N0} cycles.");
        log($"    MS1 ions {record.Ms1Acquired:E3} acquired, {record.Ms1Assigned:E3} assigned "
            + $"({Percent(record.Ms1Fraction)}); "
            + $"MS2 ions {record.Ms2Acquired:E3} acquired, {record.Ms2Assigned:E3} assigned "
            + $"({Percent(record.Ms2Fraction)}).");

        // The second numerator, on its own line rather than folded into the one above: the point of
        // measuring it is the COMPARISON, and a reader who cannot see both figures side by side has
        // to go to the parquet to make it.
        if (record.HasExplained)
        {
            var ratio = record.Ms2Assigned > 0 ? record.Ms2Explained / record.Ms2Assigned : double.NaN;
            log($"    MS2 explained by any b/y or precursor ion: {record.Ms2Explained:E3} "
                + $"({Percent(record.Ms2ExplainedFraction)} of acquired"
                + (double.IsFinite(ratio) ? $", {ratio:0.0}x the quantified total" : "")
                + ").");
            if (record.ExplainedBelowAssigned)
            {
                log("    WARNING: that is BELOW the quantified total, which is impossible - the "
                    + "explained set contains the quantified one by construction. This is a defect "
                    + "in claim building, not a property of the data.");
            }
        }

        // Per SCAN, which is the only figure here a reader can sanity-check against something they
        // already know: it should land near the instrument's AGC target. The absolute totals cannot
        // be checked that way and neither can the fraction - a units error cancels out of the ratio
        // exactly - so this line is the one that would have caught multiplying by milliseconds
        // instead of seconds, which put these at 3.7e8 and 7.3e6.
        var perMs1 = record.Ms1Count > 0 ? record.Ms1Acquired / record.Ms1Count : double.NaN;
        var perMs2 = record.Ms2Count > 0 ? record.Ms2Acquired / record.Ms2Count : double.NaN;
        log($"    mean ions per scan: MS1 {perMs1:E2}, MS2 {perMs2:E2} "
            + "(compare with the AGC target - these are the figures a units error shows up in, "
            + "because it cancels out of the fraction).");
        if (record.IonScaleImplausible)
        {
            log("    WARNING: that is outside anything an instrument can hold, so the totals are "
                + "in the wrong UNIT rather than merely surprising - ions are intensity times the "
                + "injection time in SECONDS. The fraction is unaffected either way, which is "
                + "exactly why it cannot be trusted to reveal this.");
        }

        // The reported cvParam total, for the gap centroiding leaves. Not the denominator - see the
        // class remarks - but worth knowing, because a large gap means the peak lists are sparse
        // relative to what the detector saw.
        if (reportedMs1 > 0 || reportedMs2 > 0)
        {
            log($"    Reported total-ion-current, same weighting: MS1 {reportedMs1:E3}, "
                + $"MS2 {reportedMs2:E3} (the denominator above sums the centroided peaks instead).");
        }

        if (record.SpectraMissingInjectionTime > 0)
        {
            log($"    {record.SpectraMissingInjectionTime:N0} scans reported no ion injection "
                + "time, so they could not be converted to ions and are excluded from both totals. "
                + "The fraction stays valid; the totals cover the remaining scans.");
        }
        if (record.ScansOutsideScheme > 0)
        {
            log($"    {record.ScansOutsideScheme:N0} MS2 scans fell in no isolation window of the "
                + "scheme, so nothing could be assigned in them. A large count means the scheme does "
                + "not match this acquisition.");
        }
        if (unsorted > 0)
        {
            log($"    {unsorted:N0} spectra had a non-ascending m/z array and were sorted before "
                + "masking.");
        }
        if (claims.BackwardScans > 0)
        {
            // The sweep is forward-only, so this makes the assigned total too low - and would
            // otherwise be invisible, because a smaller number looks exactly like less signal.
            log($"    WARNING: {claims.BackwardScans:N0} scans arrived earlier than the previous "
                + "scan of their own isolation window. The claim sweep only moves forward, so the "
                + "assigned total is UNDERSTATED by whatever those scans held.");
        }
        if (record.Exceeded)
        {
            log("    WARNING: more signal was assigned than acquired, which is impossible. The "
                + "fraction will not be plotted. Check that the isolation scheme matches the "
                + "acquisition and that the extraction tolerances are the document's own.");
        }
    }

    private static string Percent(double fraction) =>
        double.IsFinite(fraction) ? fraction.ToString("P1") : "n/a";

    /// <summary>
    /// A time cvParam in SECONDS, honoring the unit it declares.
    /// </summary>
    /// <remarks>
    /// Used for the ion injection time, whose product with the intensity is only a count of ions
    /// when the time is in seconds - the intensity being a rate. An absent unit is treated as
    /// milliseconds, which is what the controlled vocabulary specifies for MS:1000927 and what every
    /// file measured here declares.
    /// </remarks>
    private static double Seconds(CVParam? param)
    {
        if (param is null)
            return double.NaN;

        double value = param;
        return param.Units switch
        {
            CVID.UO_second => value,
            CVID.UO_microsecond => value / 1_000_000.0,
            CVID.UO_nanosecond => value / 1_000_000_000.0,
            CVID.UO_minute => value * 60.0,
            _ => value / 1000.0,   // UO_millisecond, and the cvParam's specified default
        };
    }

    private static double Sum(ReadOnlySpan<double> values)
    {
        var total = 0.0;
        foreach (var value in values)
        {
            if (double.IsFinite(value) && value > 0)
                total += value;
        }
        return total;
    }

    private static bool IsAscending(ReadOnlySpan<double> mz)
    {
        for (var i = 1; i < mz.Length; i++)
        {
            if (mz[i] < mz[i - 1])
                return false;
        }
        return true;
    }

    /// <summary>
    /// Sort one spectrum's peaks by m/z into the scratch buffers, which grow but are never shrunk.
    /// </summary>
    private static void Sort(
        ReadOnlySpan<double> mz, ReadOnlySpan<double> intensity,
        ref double[] sortedMz, ref double[] sortedIntensity)
    {
        if (sortedMz.Length < mz.Length)
        {
            sortedMz = new double[mz.Length];
            sortedIntensity = new double[mz.Length];
        }
        mz.CopyTo(sortedMz);
        intensity.CopyTo(sortedIntensity);
        Array.Sort(sortedMz, sortedIntensity, 0, mz.Length);
    }
}
