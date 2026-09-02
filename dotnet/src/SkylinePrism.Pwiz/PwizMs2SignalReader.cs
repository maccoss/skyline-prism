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

    private bool _summedFallbackUsed;

    /// <summary>
    /// Registers the vendor readers before any instance exists, so constructing one is enough and no
    /// caller has to remember.
    /// </summary>
    static PwizMs2SignalReader() => VendorReaders.EnsureRegistered();

    /// <inheritdoc />
    public string Describe() => _summedFallbackUsed ? "pwiz-sharp +summed" : "pwiz-sharp";

    /// <inheritdoc />
    public bool CanRead(string dataPath)
    {
        if (string.IsNullOrWhiteSpace(dataPath))
            return false;
        var name = dataPath.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        return Extensions.Any(e => name.EndsWith(e, StringComparison.OrdinalIgnoreCase));
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
        ReaderList.Default.Read(dataPath, msd, new ReaderConfig { CombineIonMobilitySpectra = true });

        var spectra = msd.Run.SpectrumList
            ?? throw new InvalidDataException($"No spectra in {Path.GetFileName(dataPath)}.");

        var cycles = new List<Ms2Cycle>();
        var windows = new Dictionary<(long, long), PrismIsolationWindow>();
        var tooManyWindows = false;

        int ms1 = 0, ms2 = 0;
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
                tic = SummedIntensity(spectra, i);
                _summedFallbackUsed = true;
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
            + $"MS2 TIC {totalMs2:E3}, {cycles.Count:N0} cycles ({model}).");

        if (tooManyWindows)
        {
            // Not a repeating DIA cycle - DDA gives one window per spectrum. Reporting these as a
            // scheme would be worse than reporting none.
            windows.Clear();
        }

        return new Ms2SignalRecord(
            dataPath,
            ms2 > 0 ? Ms2ReadStatus.Ok : Ms2ReadStatus.Failed,
            Describe(),
            _summedFallbackUsed ? Ms2SignalSource.SummedPeaks : Ms2SignalSource.ReportedTic,
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
