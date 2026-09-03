using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using Pwiz.Data.Common.Cv;
using Pwiz.Data.Common.Params;
using Pwiz.Data.MsData;
using Pwiz.Data.MsData.Readers;
using Pwiz.Data.MsData.Spectra;
using SkylinePrism.Core.RawData;
using PrismIsolationWindow = SkylinePrism.Core.Qc.IsolationWindow;

namespace SkylinePrism.Pwiz;

/// <summary>
/// Reads the acquired MS2 total ion current out of an instrument data file, through pwiz-sharp.
///
/// <para>This is the denominator the MS2 signal accounting cannot get anywhere else: Skyline's
/// <c>TicArea</c> is MS1 only. Its TIC chromatogram filter sets <c>Ms1ProductFilters</c> and leaves
/// <c>Ms2ProductFilters</c> empty, and <c>SpectrumFilterPair.FilterQ3SpectrumList</c> returns null
/// immediately for it - commented "All-ions extraction for MS1 scans only". On a real 93-replicate
/// cohort the assigned MS2 signal came to 0.51x TicArea, a number that looks like a coverage fraction
/// and is not one.</para>
/// </summary>
/// <remarks>
/// <para><b>Headers only.</b> Spectra are read at <see cref="DetailLevel.FullMetadata"/>, so the peak
/// arrays are never decoded: the total ion current is a header cvParam. That is what makes scanning a
/// multi-gigabyte run to total it affordable.</para>
/// <para><b>The reported TIC, not a sum.</b> <c>MS:1000285</c> is the instrument's own total, taken
/// before centroiding, and msconvert propagates it - so a vendor file and its converted mzML give the
/// same number. Summing a centroided peak list would not. Where the cvParam is absent the reader falls
/// back to summing and says so in <see cref="Describe"/>, so a cohort that mixes the two is visible
/// rather than silently averaged over.</para>
/// </remarks>
public sealed class PwizMs2SignalReader : IMs2SignalReader
{
    /// <summary>Extensions pwiz can open here. A directory-shaped format (.d) counts as a path.</summary>
    private static readonly string[] Extensions =
    {
        ".raw", ".mzml", ".mzxml", ".mzml.gz", ".mz5", ".d", ".wiff", ".wiff2", ".lcd", ".yep", ".baf",
    };

    /// <summary>
    /// Beyond this many distinct isolation windows the acquisition is not a repeating DIA cycle -
    /// DDA gives one window per spectrum - and reporting them as a scheme would be misleading.
    /// </summary>
    private const int MaxSchemeWindows = 2_000;


    /// <summary>
    /// Whether to collect isolation windows. They are a property of the ACQUISITION METHOD, so every
    /// replicate of a cohort has the same set, and collecting them once per file is waste.
    ///
    /// <para>Init-only, not settable: a shared reader must not be reconfigurable underneath a read
    /// that is already running on another thread. A caller wanting both behaviours constructs two
    /// readers, which cost nothing.</para>
    /// </summary>
    public bool CollectWindows { get; init; } = true;

    /// <summary>
    /// Registers the vendor readers before any instance exists, so constructing one is enough and no
    /// caller has to remember.
    /// </summary>
    static PwizMs2SignalReader() => VendorReaders.EnsureRegistered();

    /// <inheritdoc />
    /// <remarks>
    /// A CONSTANT, deliberately. This used to fold in how many spectra had fallen back to summed
    /// intensities, which meant the reader carried mutable per-file state - fine while one file was
    /// read at a time, a data race the moment a cohort is read in parallel. That detail is per read,
    /// so it lives on <see cref="Ms2SignalRecord.Reader"/> instead, which is per read too.
    /// </remarks>
    public string Describe() => "pwiz-sharp";

    /// <summary>
    /// How a read describes itself, given what it found. The "+summed" suffix carries the
    /// PROPORTION, not just the fact: a handful of spectra without the total-ion-current cvParam is
    /// noise, whereas a file with none means the totals are summed centroided peaks rather than the
    /// instrument's own pre-centroiding value, and those are not interchangeable.
    /// </summary>
    private static string DescribeRead(int ticSpectra, int summedFallbacks)
    {
        if (summedFallbacks == 0)
            return "pwiz-sharp";
        var total = ticSpectra + summedFallbacks;
        return total == summedFallbacks
            ? "pwiz-sharp +summed"
            : $"pwiz-sharp +summed({(double)summedFallbacks / total:P1})";
    }

    /// <inheritdoc />
    public bool CanRead(string dataPath)
    {
        if (string.IsNullOrWhiteSpace(dataPath))
            return false;
        var name = dataPath.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        return Extensions.Any(e => name.EndsWith(e, StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>
    /// Force the general spectrum-walk path, skipping the TIC-chromatogram optimisation. For the
    /// cross-check that keeps the two honest - the fast path is only trustworthy because it agrees
    /// with the reference, so that agreement has to be testable rather than asserted in a comment.
    /// </summary>
    internal Ms2SignalRecord ReadViaSpectrumWalk(
        string dataPath, Action<string>? log = null, CancellationToken ct = default)
    {
        try
        {
            using var msd = new MSData();
            ReaderList.Default.Read(
                dataPath, msd, new ReaderConfig { CombineIonMobilitySpectra = true });
            return FromSpectrumWalk(msd, dataPath, log, ct);
        }
        catch (Exception ex)
        {
            return Ms2SignalRecord.Unavailable(
                dataPath, Ms2ReadStatus.Failed, Describe(), ex.Message);
        }
    }

    /// <inheritdoc />
    public Ms2SignalRecord Read(
        string dataPath, Action<string>? log = null, CancellationToken ct = default)
    {
        if (!File.Exists(dataPath) && !Directory.Exists(dataPath))
        {
            return Ms2SignalRecord.Unavailable(
                dataPath, Ms2ReadStatus.NotFound, Describe(), "No such file.");
        }

        try
        {
            return ReadCore(dataPath, log, ct);
        }
        catch (OperationCanceledException)
        {
            return Ms2SignalRecord.Unavailable(
                dataPath, Ms2ReadStatus.Cancelled, Describe(), "Cancelled.");
        }
        catch (Exception ex)
        {
            // Never throws, by contract: one unreadable file in a 192-replicate cohort must cost only
            // that file. A licence-gated vendor SDK surfaces here too - the reader recognizes the
            // format and then refuses to open it.
            log?.Invoke($"  Could not read {Path.GetFileName(dataPath)}: {ex.Message}");
            return Ms2SignalRecord.Unavailable(
                dataPath, Ms2ReadStatus.Failed, Describe(), ex.Message);
        }
    }

    private Ms2SignalRecord ReadCore(string dataPath, Action<string>? log, CancellationToken ct)
    {
        using var msd = new MSData();
        // Combine the ion mobility dimension: pwiz otherwise presents an uncombined TIMS frame as
        // hundreds of spectra sharing one retention time and one isolation window, which would turn
        // one acquisition cycle into hundreds.
        //
        // This OPEN is the dominant cost of a read, not the data: on a 3.3 GB Thermo file over SMB
        // it is ~2.2 s of a 2.7 s total, against 0.41 s for the chromatogram and 0.05 s for the
        // isolation windows. It is the vendor SDK building its scan index, so it is I/O latency
        // rather than compute - which is why reading several files at once helps here even though
        // it barely helped the per-spectrum walk it replaced.
        var openClock = System.Diagnostics.Stopwatch.StartNew();
        ReaderList.Default.Read(dataPath, msd, new ReaderConfig { CombineIonMobilitySpectra = true });
        openClock.Stop();
        log?.Invoke($"    timing: open {openClock.Elapsed.TotalSeconds:0.00} s");

        // The run TIC chromatogram, where the format offers one, answers this in a couple of calls
        // instead of a walk over every spectrum. See FromTicChromatogram.
        if (FromTicChromatogram(msd, dataPath, log, ct) is { } fast)
            return fast;

        return FromSpectrumWalk(msd, dataPath, log, ct);
    }

    /// <summary>
    /// The fast path: total the run TIC chromatogram over its MS2 points.
    ///
    /// <para>A vendor TIC chromatogram carries time and intensity for EVERY scan, and pwiz adds a
    /// third array holding the MS order per point. Summing intensity where that order is 2 gives the
    /// acquired MS2 total directly, and the MS1 points mark the cycle boundaries, so the per-cycle
    /// trace falls out of the same arrays. That replaces a walk over 165,000 spectra, each of which
    /// costs several vendor-SDK calls and a Spectrum allocation.</para>
    ///
    /// <para>Returns null when the format has no TIC chromatogram or no MS-order array, and the
    /// caller falls back to the spectrum walk. The two paths are asserted to agree.</para>
    /// </summary>
    private Ms2SignalRecord? FromTicChromatogram(
        MSData msd, string dataPath, Action<string>? log, CancellationToken ct)
    {
        var chromatograms = msd.Run.ChromatogramList;
        if (chromatograms is null || chromatograms.Count == 0)
            return null;

        var chromClock = System.Diagnostics.Stopwatch.StartNew();
        Chromatogram? tic = null;
        for (var i = 0; i < chromatograms.Count && tic is null; i++)
        {
            if (!string.Equals(
                    chromatograms.ChromatogramIdentity(i).Id, "TIC", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }
            ct.ThrowIfCancellationRequested();
            tic = chromatograms.GetChromatogram(i, getBinaryData: true);
        }

        var times = tic?.BinaryDataArrays
            .FirstOrDefault(a => a.HasCVParam(CVID.MS_time_array))?.Data;
        var intensities = tic?.BinaryDataArrays
            .FirstOrDefault(a => a.HasCVParam(CVID.MS_intensity_array))?.Data;
        // Without the MS-order array every point is of unknown level, and an MS2-only total cannot
        // be taken from this. Falling back beats totalling MS1 and MS2 together, which is exactly
        // the mistake Skyline TicArea invites.
        var levels = tic?.IntegerDataArrays.FirstOrDefault()?.Data;

        if (times is null || intensities is null || levels is null
            || times.Count == 0 || intensities.Count != times.Count || levels.Count != times.Count)
        {
            return null;
        }
        chromClock.Stop();

        var cycles = new List<Ms2Cycle>();
        int ms1 = 0, ms2 = 0;
        double totalMs2 = 0;
        var cycleStart = double.NaN;
        var cycleStop = double.NaN;
        var cycleMs2 = 0;
        var cycleSignal = 0.0;

        void CloseCycle()
        {
            if (cycleMs2 == 0)
                return;
            cycles.Add(new Ms2Cycle(cycles.Count, cycleStart, cycleStop, cycleMs2, cycleSignal));
            cycleMs2 = 0;
            cycleSignal = 0;
            cycleStart = double.NaN;
            cycleStop = double.NaN;
        }

        for (var i = 0; i < times.Count; i++)
        {
            // Thermo writes the RAW MSOrder, which is negative for scan kinds that are not a plain
            // MSn (Ng = -3, Nl = -2, Par = -1). Only a literal 1 and 2 are MS1 and MS2 here.
            var level = levels[i];
            if (level == 1)
            {
                ms1++;
                CloseCycle();
                continue;
            }
            if (level != 2)
                continue;

            ms2++;
            totalMs2 += intensities[i];
            cycleSignal += intensities[i];
            cycleMs2++;
            if (!double.IsFinite(cycleStart))
                cycleStart = times[i];
            cycleStop = times[i];
        }
        CloseCycle();

        if (ms2 == 0)
            return null;    // nothing this path can claim; let the walk report why

        // Isolation windows are NOT in the chromatogram. For DIA they repeat every cycle, so the
        // first couple of cycles carry the whole scheme: a few hundred spectrum reads rather than
        // every one.
        var windowClock = System.Diagnostics.Stopwatch.StartNew();
        var windows = CollectWindows
            ? SampleIsolationWindows(msd, ms2 / Math.Max(1, cycles.Count), ct)
            : Array.Empty<PrismIsolationWindow>();
        windowClock.Stop();

        log?.Invoke(
            $"  {Path.GetFileName(dataPath)}: {ms2:N0} MS2 and {ms1:N0} MS1 scans, "
            + $"MS2 TIC {totalMs2:E3}, {cycles.Count:N0} cycles, {windows.Count} isolation windows "
            + "(from the run TIC chromatogram).");
        log?.Invoke(
            $"    timing: chromatogram {chromClock.Elapsed.TotalSeconds:0.00} s, "
            + $"isolation windows {windowClock.Elapsed.TotalSeconds:0.00} s");

        return new Ms2SignalRecord(
            dataPath, Ms2ReadStatus.Ok, "pwiz-sharp tic", Ms2SignalSource.ReportedTic,
            ms1, ms2, totalMs2, times[0], times[times.Count - 1],
            cycles.Count > 0 ? Ms2CycleModel.Ms1Bounded : Ms2CycleModel.FixedRtBins,
            cycles, windows);
    }

    /// <summary>
    /// Isolation windows from the first couple of acquisition cycles. A DIA cycle sweeps the whole
    /// m/z range and starts over, so two of them contain every window and reading more would re-read
    /// the same set. Empty rather than guessed when the spectra are not available.
    /// </summary>
    private static IReadOnlyList<PrismIsolationWindow> SampleIsolationWindows(
        MSData msd, int perCycle, CancellationToken ct)
    {
        var windows = new Dictionary<(long, long), PrismIsolationWindow>();
        var spectra = msd.Run.SpectrumList;
        if (spectra is null || spectra.Count == 0)
            return Array.Empty<PrismIsolationWindow>();

        // Two cycles worth, with a floor so a short or irregular run still samples something.
        var limit = Math.Min(spectra.Count, Math.Max(64, perCycle * 2 + 8));
        for (var i = 0; i < limit; i++)
        {
            ct.ThrowIfCancellationRequested();
            var spectrum = spectra.GetSpectrum(i, DetailLevel.FullMetadata);
            if (spectrum.Params.CvParamValueOrDefault(CVID.MS_ms_level, 0) != 2
                || spectrum.Precursors.Count == 0)
            {
                continue;
            }
            AddWindow(windows, spectrum.Precursors[0].IsolationWindow);
        }
        return windows.Values.OrderBy(w => w.Start).ToList();
    }

    /// <summary>One isolation window, keyed on rounded edges so float noise cannot multiply it.</summary>
    private static void AddWindow(
        Dictionary<(long, long), PrismIsolationWindow> windows, IsolationWindow window)
    {
        var target = window.CvParamValueOrDefault(CVID.MS_isolation_window_target_m_z, double.NaN);
        if (!double.IsFinite(target))
            return;
        var lower = window.CvParamValueOrDefault(CVID.MS_isolation_window_lower_offset, 0.0);
        var upper = window.CvParamValueOrDefault(CVID.MS_isolation_window_upper_offset, 0.0);
        var start = target - lower;
        var end = target + upper;
        // 1e-4 Th is far below any real window and far above the noise.
        var key = ((long)Math.Round(start * 1e4), (long)Math.Round(end * 1e4));
        windows.TryAdd(key, new PrismIsolationWindow(start, end));
    }

    /// <summary>
    /// The general path: every spectrum header. Slower - several vendor-SDK calls and an allocation
    /// per spectrum - but it needs nothing beyond a spectrum list, so it covers formats with no TIC
    /// chromatogram and is the reference the fast path is checked against.
    /// </summary>
    private Ms2SignalRecord FromSpectrumWalk(
        MSData msd, string dataPath, Action<string>? log, CancellationToken ct)
    {
        var spectra = msd.Run.SpectrumList
            ?? throw new InvalidDataException($"No spectra in {Path.GetFileName(dataPath)}.");

        var cycles = new List<Ms2Cycle>();
        var windows = new Dictionary<(long, long), PrismIsolationWindow>();
        var tooManyWindows = false;

        int ms1 = 0, ms2 = 0;
        int ticSpectra = 0, summedFallbacks = 0;
        double totalMs2 = 0;
        double rtFirst = double.NaN, rtLast = double.NaN;

        // A cycle opens at each MS1 and closes at the next. For MS2-only acquisitions there is no MS1
        // to bound it, so the fallback is a wrap in the isolation target m/z - a DIA cycle sweeps the
        // range and starts over.
        var sawMs1 = false;
        var cycleStart = double.NaN;
        var cycleStop = double.NaN;
        var cycleMs2 = 0;
        var cycleSignal = 0.0;
        var previousTarget = double.NaN;
        var model = Ms2CycleModel.None;

        void CloseCycle()
        {
            if (cycleMs2 == 0)
                return;
            cycles.Add(new Ms2Cycle(cycles.Count, cycleStart, cycleStop, cycleMs2, cycleSignal));
            cycleMs2 = 0;
            cycleSignal = 0;
            cycleStart = double.NaN;
            cycleStop = double.NaN;
        }

        for (var i = 0; i < spectra.Count; i++)
        {
            if ((i & 0x3FF) == 0)
                ct.ThrowIfCancellationRequested();

            // Metadata only: the total ion current is a header field, so no peak array is decoded.
            var spectrum = spectra.GetSpectrum(i, DetailLevel.FullMetadata);
            var level = spectrum.Params.CvParamValueOrDefault(CVID.MS_ms_level, 0);

            var scan = spectrum.ScanList.Scans.Count > 0 ? spectrum.ScanList.Scans[0] : null;
            var rt = Minutes(scan?.CvParam(CVID.MS_scan_start_time));
            if (double.IsFinite(rt))
            {
                if (!double.IsFinite(rtFirst))
                    rtFirst = rt;
                rtLast = rt;
            }

            if (level == 1)
            {
                sawMs1 = true;
                model = Ms2CycleModel.Ms1Bounded;
                CloseCycle();
                ms1++;
                continue;
            }
            if (level != 2)
                continue;

            ms2++;

            var target = double.NaN;
            if (spectrum.Precursors.Count > 0)
            {
                var window = spectrum.Precursors[0].IsolationWindow;
                target = window.CvParamValueOrDefault(CVID.MS_isolation_window_target_m_z, double.NaN);
                var lower = window.CvParamValueOrDefault(CVID.MS_isolation_window_lower_offset, 0.0);
                var upper = window.CvParamValueOrDefault(CVID.MS_isolation_window_upper_offset, 0.0);
                if (double.IsFinite(target) && !tooManyWindows)
                {
                    // Keyed on the rounded edges so floating noise does not multiply one window into
                    // several. 1e-4 Th is far below any real window and far above the noise.
                    var start = target - lower;
                    var end = target + upper;
                    var key = ((long)Math.Round(start * 1e4), (long)Math.Round(end * 1e4));
                    if (!windows.ContainsKey(key))
                    {
                        if (windows.Count >= MaxSchemeWindows)
                            tooManyWindows = true;
                        else
                            windows[key] = new PrismIsolationWindow(start, end);
                    }
                }
            }

            // MS2-only acquisition: a cycle ends when the isolation sweep wraps back down.
            if (!sawMs1 && double.IsFinite(target) && double.IsFinite(previousTarget)
                && target < previousTarget)
            {
                model = Ms2CycleModel.IsolationWrap;
                CloseCycle();
            }
            if (double.IsFinite(target))
                previousTarget = target;

            var tic = spectrum.Params.CvParamValueOrDefault(CVID.MS_total_ion_current, double.NaN);
            if (!double.IsFinite(tic) || tic <= 0)
            {
                // The slow path: this decodes the peak arrays, which is why a file with no reported
                // TIC takes substantially longer than one that has it.
                tic = SummedIntensity(spectra, i);
                summedFallbacks++;
            }
            else
            {
                ticSpectra++;
            }

            totalMs2 += tic;
            cycleSignal += tic;
            cycleMs2++;
            if (!double.IsFinite(cycleStart))
                cycleStart = rt;
            cycleStop = rt;
        }
        CloseCycle();

        if (model == Ms2CycleModel.None && cycles.Count > 0)
            model = Ms2CycleModel.FixedRtBins;

        log?.Invoke(
            $"  {Path.GetFileName(dataPath)}: {ms2:N0} MS2 and {ms1:N0} MS1 spectra, "
            + $"MS2 TIC {totalMs2:E3}, {cycles.Count:N0} cycles ({model}), "
            + $"reader {DescribeRead(ticSpectra, summedFallbacks)}.");
        if (summedFallbacks > 0)
        {
            log?.Invoke(
                $"    {summedFallbacks:N0} of {ticSpectra + summedFallbacks:N0} MS2 spectra had no "
                + "total-ion-current cvParam, so their totals are summed peak intensities. That is a "
                + "post-centroiding sum, not the instrument's own total.");
        }

        if (tooManyWindows)
        {
            // Not a repeating DIA cycle - DDA gives one window per spectrum. Reporting these as a
            // scheme would be worse than reporting none.
            windows.Clear();
        }

        return new Ms2SignalRecord(
            dataPath,
            ms2 > 0 ? Ms2ReadStatus.Ok : Ms2ReadStatus.Failed,
            DescribeRead(ticSpectra, summedFallbacks),
            // The source names where MOST of the signal came from; the reader string carries the
            // proportion.
            summedFallbacks > ticSpectra ? Ms2SignalSource.SummedPeaks : Ms2SignalSource.ReportedTic,
            ms1, ms2, totalMs2, rtFirst, rtLast, model, cycles,
            windows.Values.OrderBy(w => w.Start).ToList(),
            ms2 > 0 ? null : "The file has no MS2 spectra.");
    }

    /// <summary>
    /// Summed peak intensities, for a spectrum with no total-ion-current cvParam. Decodes the arrays,
    /// so it is the slow path and deliberately only reached when the header does not carry the value.
    /// </summary>
    private static double SummedIntensity(ISpectrumList spectra, int index)
    {
        var spectrum = spectra.GetSpectrum(index, getBinaryData: true);
        var intensity = spectrum.GetIntensityArray();
        if (intensity is null)
            return 0;

        var sum = 0.0;
        foreach (var value in intensity.Data)
            sum += value;
        return sum;
    }

    /// <summary>
    /// A time cvParam in minutes, honouring the unit it declares.
    /// </summary>
    /// <remarks>
    /// Vendors differ: Thermo records scan start time in minutes, Bruker in seconds. Reading the value
    /// and assuming minutes made a 64-minute diaPASEF run look like 64 hours in MARS, which is where
    /// this rule comes from. An absent or unrecognized unit is treated as minutes, which is mzML's
    /// default.
    /// </remarks>
    private static double Minutes(CVParam? param)
    {
        if (param is null)
            return double.NaN;

        double value = param;
        return param.Units switch
        {
            CVID.UO_second => value / 60.0,
            CVID.UO_millisecond => value / 60_000.0,
            _ => value,
        };
    }
}
