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
/// network share, so the file is opened once and both halves and every cycle come off the same walk.
/// Measured on a 3.18 GB Thermo file: 165,003 spectra, about 20,000 spectra/s with peaks decoded,
/// roughly 8 s per file.</para>
///
/// <para><b>Both sides are summed from the SAME array.</b> The denominator is the sum of the
/// spectrum's own peak intensities, not the reported total-ion-current cvParam. The cvParam is the
/// instrument's pre-centroiding total and includes signal that is not in the centroided peak list at
/// all, so dividing a centroided numerator by it would understate the fraction by however much
/// centroiding removed. The reported total is still accumulated, separately, so the gap between the
/// two is visible rather than assumed.</para>
///
/// <para><b>The unit is intensity times injection time.</b> A scan's intensity is a rate; multiplying
/// by the ion injection time gives the ion-proportional count Skyline itself reports. Skipping that
/// factor is the defect this replaces: measured on a real Astral file the two differ by 7.0x, the
/// mean injection time, and the error looks like a plausible fraction rather than like a bug.</para>
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

        var cycles = new List<IonCycle>();
        int ms1 = 0, ms2 = 0, noInjection = 0, outsideScheme = 0, unsorted = 0;
        double ms1Acquired = 0, ms2Acquired = 0, ms1Assigned = 0, ms2Assigned = 0;
        double reportedMs1 = 0, reportedMs2 = 0;
        double rtFirst = double.NaN, rtLast = double.NaN;

        // A cycle opens at each MS1 and closes at the next. Unlike the acquired-only walk there is no
        // isolation-wrap fallback: ion accounting needs MS1 totals, so a file with no MS1 has no MS1
        // half to bound anything with, and one cycle covering the run is the honest answer.
        var cycleStart = double.NaN;
        var cycleStop = double.NaN;
        int cycleMs1 = 0, cycleMs2 = 0;
        double cMs1Acq = 0, cMs2Acq = 0, cMs1Asg = 0, cMs2Asg = 0;

        void CloseCycle()
        {
            if (cycleMs1 == 0 && cycleMs2 == 0)
                return;
            cycles.Add(new IonCycle(
                cycles.Count, cycleStart, cycleStop, cycleMs1, cycleMs2,
                cMs1Acq, cMs2Acq, cMs1Asg, cMs2Asg));
            cycleMs1 = cycleMs2 = 0;
            cMs1Acq = cMs2Acq = cMs1Asg = cMs2Asg = 0;
            cycleStart = double.NaN;
            cycleStop = double.NaN;
        }

        // Reused across spectra, grown as needed: only touched for a spectrum whose m/z array is not
        // ascending, which the mzML specification forbids and no measured file has done.
        double[] sortedMz = Array.Empty<double>();
        double[] sortedIntensity = Array.Empty<double>();

        // Split so the two costs can be told apart. They are both per spectrum, and the first full
        // walk was 27x slower than a sequential copy of the same file - which is a statement about
        // one of these two and not the other.
        var readTicks = 0L;
        var maskTicks = 0L;
        var walk = Stopwatch.StartNew();

        for (var i = 0; i < spectra.Count; i++)
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

            // Milliseconds, per the cvParam's own unit, and verified on a real file: 7.012 to 50.013
            // over 1,500 spectra. A scan with none is weighted 1, on BOTH sides, so its fraction
            // stays right while the absolute totals mix two weightings - which the record reports.
            var injection = scan?.CvParamValueOrDefault(CVID.MS_ion_injection_time, double.NaN)
                ?? double.NaN;
            if (!double.IsFinite(injection) || injection <= 0)
            {
                injection = 1.0;
                noInjection++;
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
            var claimed = claims.Claimed(
                level, windowIndex, rt, mz, intensity,
                level == 1 ? ms1ByList : ms2ByList);
            maskTicks += Stopwatch.GetTimestamp() - maskStart;

            var acquiredIons = summed * injection;
            var assignedIons = claimed * injection;
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
            ms1ByList, ms2ByList, rtFirst, rtLast, noInjection, outsideScheme, cycles,
            ms1 + ms2 > 0 ? null : "The file has no MS1 or MS2 spectra.");

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
            log($"    {record.SpectraMissingInjectionTime:N0} scans reported no ion injection time, "
                + "so their intensity is counted unweighted on both sides. Fractions stay valid; the "
                + "absolute totals mix two weightings.");
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
        if (record.Exceeded)
        {
            log("    WARNING: more signal was assigned than acquired, which is impossible. The "
                + "fraction will not be plotted. Check that the isolation scheme matches the "
                + "acquisition and that the extraction tolerances are the document's own.");
        }
    }

    private static string Percent(double fraction) =>
        double.IsFinite(fraction) ? fraction.ToString("P1") : "n/a";

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
