using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.RawData;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// One replicate's ion accounting, as it is cached and as the plots read it.
/// </summary>
/// <param name="DataFile">The instrument file this came from, or empty when none was resolved.</param>
/// <param name="Status">The read's outcome, kept so a failed read is distinguishable from an empty one.</param>
public sealed record IonAccountingRow(
    string Sample,
    string SampleType,
    string DataFile,
    Ms2ReadStatus Status,
    string Reader,
    int Ms1Count,
    int Ms2Count,
    double Ms1Acquired,
    double Ms2Acquired,
    double Ms1Assigned,
    double Ms2Assigned,
    double Ms2Explained,
    bool HasExplained,
    double RtStartMin,
    double RtStopMin,
    int Claims,
    int ScansOutsideScheme,
    int SpectraMissingInjectionTime,
    int CycleCount,
    IReadOnlyList<double> Ms1ByList,
    IReadOnlyList<double> Ms2ByList,
    /// <summary>
    /// When the instrument began acquiring this replicate, from the data file's own start
    /// timestamp. Null for a cache written before this was recorded, and for a file that does not
    /// declare one - which is why ordering by it is offered only when every row has one.
    /// </summary>
    DateTime? AcquiredUtc = null,
    double Ms1Signal = 0,
    double Ms2Signal = 0,
    double Ms1SignalAssigned = 0,
    double Ms2SignalAssigned = 0,
    double Ms2SignalExplained = 0,
    /// <summary>
    /// Whether the summed TIC was measured at all. A flag rather than an inference from a zero: a
    /// cache written before this column carries no signal, and reading that back as 0.0 would draw a
    /// replicate that acquired nothing.
    /// </summary>
    bool HasSignal = false)
{
    /// <summary>
    /// The assigned share of the summed TIC, which is NOT the assigned share of the ions - the ion
    /// count weights each scan by its injection time and the TIC does not. NaN when this cache
    /// carries no signal, which is not the same as zero.
    /// </summary>
    public double Ms1SignalFraction =>
        HasSignal && Ms1Signal > 0 ? Ms1SignalAssigned / Ms1Signal : double.NaN;

    /// <inheritdoc cref="Ms1SignalFraction"/>
    public double Ms2SignalFraction =>
        HasSignal && Ms2Signal > 0 ? Ms2SignalAssigned / Ms2Signal : double.NaN;

    /// <inheritdoc cref="Ms1SignalFraction"/>
    public double Ms2SignalExplainedFraction =>
        HasSignal && HasExplained && Ms2Signal > 0 ? Ms2SignalExplained / Ms2Signal : double.NaN;

    public double Ms1Fraction => Ms1Acquired > 0 ? Ms1Assigned / Ms1Acquired : double.NaN;
    public double Ms2Fraction => Ms2Acquired > 0 ? Ms2Assigned / Ms2Acquired : double.NaN;

    /// <summary>
    /// The share of acquired MS2 ions the peptides can ACCOUNT FOR - their theoretical b/y ions and
    /// surviving precursor - against <see cref="Ms2Fraction"/>'s share they are quantified on.
    /// NaN when this cache carries no explained total, which is not the same as zero.
    /// </summary>
    public double Ms2ExplainedFraction =>
        HasExplained && Ms2Acquired > 0 ? Ms2Explained / Ms2Acquired : double.NaN;

    /// <summary>
    /// More ions were assigned than acquired, which is impossible and therefore a defect. Callers
    /// must refuse to show a fraction rather than clamping it.
    /// </summary>
    /// <remarks>
    /// Narrower than <see cref="IonAccountingRecord.Exceeded"/>, which folds the explained total in;
    /// here that is <see cref="ExplainedImpossible"/>'s job, so a fault confined to the theoretical
    /// claim set does not blank a quantified fraction that is perfectly sound.
    /// </remarks>
    public bool Exceeded => Ms1Assigned > Ms1Acquired || Ms2Assigned > Ms2Acquired;

    /// <summary>
    /// The same impossibility for the summed TIC. <b>Not implied by <see cref="Exceeded"/>:</b> the
    /// ion totals weight each scan by its injection time and the signal totals do not, so a fault
    /// confined to short-injection scans can leave the ion fraction under 1 while the signal
    /// fraction is over it. Whichever quantity is being drawn has to be the one that is checked.
    /// </summary>
    public bool SignalExceeded =>
        HasSignal && (Ms1SignalAssigned > Ms1Signal || Ms2SignalAssigned > Ms2Signal);

    /// <inheritdoc cref="SignalExceeded"/>
    public bool SignalExplainedImpossible =>
        HasSignal && HasExplained
        && (Ms2SignalExplained > Ms2Signal || Ms2SignalExplained < Ms2SignalAssigned);

    /// <summary>
    /// The assigned fraction for the quantity actually being shown.
    /// </summary>
    /// <remarks>
    /// Here rather than at each caller. There are four of these numbers and three places that want
    /// one - the plot, the cohort status line and the per-replicate status line - and picking the
    /// wrong one is invisible: the plot draws TIC percentages while the line beside it reports
    /// ion-weighted ones, both plausible, differing by whatever the injection times were doing. That
    /// shipped.
    /// </remarks>
    public double Ms1FractionIn(bool signal) => signal ? Ms1SignalFraction : Ms1Fraction;

    /// <inheritdoc cref="Ms1FractionIn"/>
    public double Ms2FractionIn(bool signal) => signal ? Ms2SignalFraction : Ms2Fraction;

    /// <inheritdoc cref="Ms1FractionIn"/>
    public double Ms2ExplainedFractionIn(bool signal) =>
        signal ? Ms2SignalExplainedFraction : Ms2ExplainedFraction;

    /// <summary>The impossibility check for the quantity actually being shown.</summary>
    public bool ExceededIn(bool signal) => signal ? SignalExceeded : Exceeded;

    /// <inheritdoc cref="ExceededIn"/>
    public bool ExplainedImpossibleIn(bool signal) =>
        signal ? SignalExplainedImpossible : ExplainedImpossible;

    /// <summary>
    /// The explained total exceeded what was acquired, or fell below the quantified total. Both are
    /// impossible and both mean a defect in claim building.
    ///
    /// <para>Deliberately SEPARATE from <see cref="Exceeded"/>. A fault confined to the theoretical
    /// claim set must not blank the quantified fraction as well - the caption that withholds a
    /// figure also names a cause, and naming the isolation scheme for a quantified total that is
    /// perfectly sound sends a reader after the wrong thing.</para>
    /// </summary>
    public bool ExplainedImpossible =>
        HasExplained && (Ms2Explained > Ms2Acquired || Ms2Explained < Ms2Assigned);

    /// <inheritdoc cref="IonAccountingRecord.MeanMs1IonsPerScan"/>
    public double MeanMs1IonsPerScan => Ms1Count > 0 ? Ms1Acquired / Ms1Count : double.NaN;

    /// <inheritdoc cref="IonAccountingRecord.MeanMs2IonsPerScan"/>
    public double MeanMs2IonsPerScan => Ms2Count > 0 ? Ms2Acquired / Ms2Count : double.NaN;

    /// <inheritdoc cref="IonAccountingRecord.IonScaleImplausible"/>
    public bool IonScaleImplausible =>
        Implausible(MeanMs1IonsPerScan) || Implausible(MeanMs2IonsPerScan);

    private static bool Implausible(double meanPerScan) =>
        IonAccountingRecord.IsIonScaleImplausible(meanPerScan);

    /// <summary>Whether this row carries numbers worth plotting.</summary>
    public bool IsUsable => Status == Ms2ReadStatus.Ok && (Ms1Acquired > 0 || Ms2Acquired > 0);
}

/// <summary>
/// A whole cohort's ion accounting plus the settings that produced it.
/// </summary>
/// <param name="Cycles">
/// Every replicate's per-cycle traces - populated by a RUN, and deliberately empty when this comes
/// back from <see cref="IonAccountingStore.Read"/>. A cohort's cycles run to hundreds of thousands
/// of rows and the time plots show one replicate at a time, so a reader asks for the one it needs
/// through <see cref="IonAccountingStore.ReadCycles"/> rather than paying for all of them.
/// </param>
public sealed record IonAccountingResult(
    string SettingsKey,
    string ProductTolerance,
    string PrecursorTolerance,
    string IsolationScheme,
    IReadOnlyList<string> ListNames,
    int AssignedPeptides,
    bool ListsMatchable,
    IReadOnlyList<IonAccountingRow> Rows,
    IReadOnlyList<IonCycleRow> Cycles)
{
    /// <summary>Replicates with numbers, which is what every plot iterates.</summary>
    public IReadOnlyList<IonAccountingRow> Usable => Rows.Where(r => r.IsUsable).ToArray();

    /// <summary>
    /// Whether these cached numbers answer the question the settings now ask. The file is keyed on
    /// its name alone, so without this a re-run replots the previous run's numbers under the new
    /// run's caption - and nothing fails loudly, because both the plot and the caption read as
    /// correct.
    /// </summary>
    public bool MatchesSettings(string settingsKey) =>
        string.Equals(SettingsKey, settingsKey, StringComparison.Ordinal);

    /// <summary>
    /// The median, best and worst replicate by MS2 assigned fraction, or fewer when there are fewer.
    ///
    /// <para>Three, not one: a single representative hides whether the cohort is uniform, and best
    /// against worst is the comparison that tells a user whether one injection misbehaved or the
    /// whole run did. Rows whose fraction is not physical are excluded - see
    /// <see cref="IonAccountingRow.Exceeded"/>.</para>
    /// </summary>
    /// <param name="signal">
    /// Rank on the summed TIC rather than on the ion count. The two are different quantities with
    /// different fractions, so "worst" is a different replicate in each - and a panel captioned as
    /// one while chosen by the other is wrong in a way nothing on the page could reveal.
    /// </param>
    public IReadOnlyList<IonAccountingRow> Representatives(bool signal = false)
    {
        var ranked = Rows
            .Where(r => r.IsUsable && !r.ExceededIn(signal)
                && double.IsFinite(r.Ms2FractionIn(signal)))
            .OrderBy(r => r.Ms2FractionIn(signal))
            .ToArray();
        if (ranked.Length == 0)
            return Array.Empty<IonAccountingRow>();
        if (ranked.Length <= 3)
            return ranked;

        // Distinct by sample, so a three-replicate cohort does not list one row three times.
        var picks = new List<IonAccountingRow> { ranked[^1], ranked[ranked.Length / 2], ranked[0] };
        return picks
            .GroupBy(r => r.Sample, StringComparer.Ordinal)
            .Select(g => g.First())
            .ToArray();
    }
}

/// <summary>One cycle of one replicate, the long-format row the time plots read.</summary>
public readonly record struct IonCycleRow(
    string Sample,
    int Cycle,
    double RtStartMin,
    double RtStopMin,
    int Ms1Count,
    int Ms2Count,
    double Ms1Acquired,
    double Ms2Acquired,
    double Ms1Assigned,
    double Ms2Assigned,
    double Ms2Explained = 0,
    double Ms1Signal = 0,
    double Ms2Signal = 0,
    double Ms1SignalAssigned = 0,
    double Ms2SignalAssigned = 0,
    double Ms2SignalExplained = 0);

/// <summary>
/// Reads and writes the ion-accounting cache.
///
/// <para><b>Why a cache at all.</b> Computing this reads every instrument file in the cohort - a
/// terabyte on a network share for the cohort this was built for. The bar plot and both time plots
/// then become pure file reads, so switching replicate or list in the GUI is instant and a re-run of
/// the QC report costs nothing.</para>
///
/// <para><b>Keyed on what was requested, and the key is IN the file.</b> Anything that changes the
/// numbers belongs in <see cref="SettingsKeyFor"/>: both extraction tolerances, the isolation scheme,
/// the selected lists, and a fingerprint of the instrument files and of <c>merged_data/</c>. Nothing
/// fails loudly if a setting is left out - the plots look right and the caption comes from the cache
/// - which is exactly why this is the one place to add it.</para>
/// </summary>
public static class IonAccountingStore
{
    public const string FileName = "ion_accounting.parquet";
    public const string CyclesFile = "ion_cycles.parquet";
    public const string ListsFile = "ion_accounting_lists.parquet";

    /// <summary>
    /// The cache validity key. <paramref name="sources"/> is every input whose CONTENT would change
    /// the answer: the instrument files, <c>merged_data/</c> for the claim geometry, and the rollup
    /// and corrected peptide matrices for which peptides claim at all.
    /// </summary>
    /// <param name="listKeys">
    /// One entry per selected protein list, identifying it by its CONTENT and not only its name -
    /// see <see cref="ProteinListIdentity"/>. A list edited in place keeps its name and gives
    /// different per-list totals, so keying on the name alone reuses stale ones.
    /// </param>
    public static string SettingsKeyFor(
        string productTolerance, string precursorTolerance, string isolationScheme,
        IEnumerable<string> listKeys, IReadOnlyList<string> sources) =>
        string.Join(
            "|",
            // v2: v1 multiplied the intensity by the injection time in MILLISECONDS, so
            // every total it cached is 1000x too large. The fractions were right, but the
            // columns are named "ions" - so the key is bumped to make every directory
            // recompute rather than replot the old magnitudes under the new caption.
            //
            // v4: the summed TIC arrived, acquired and assigned, per replicate and per cycle.
            // Same reasoning as v3 - a v3 file is not WRONG, but it carries no signal columns, so
            // reusing it would offer a Signal view that is silently empty on the replicates measured
            // before and populated on the ones measured after. The quantity picker would then be
            // showing two different cohorts depending on when each file happened to be read.
            //
            // v3: the explained total arrived. A v2 file is not WRONG - every number in it is
            // still right - but it carries no explained column, so reusing it would draw a
            // section whose second series is silently absent on some replicates and present on
            // others, depending on when each was measured. Recomputing is the honest answer.
            "ions-v4",
            productTolerance,
            precursorTolerance,
            isolationScheme,
            string.Join(",", listKeys),
            SourceFingerprint.Compute(sources));

    /// <summary>
    /// A protein list's identity for the cache key: its name, its size, and a digest of its members.
    /// </summary>
    /// <remarks>
    /// Sorted and case-folded so the same set written in a different order is the same list, and
    /// digested rather than joined so a 2,000-protein panel does not put 2,000 accessions into a
    /// string that is stored in every row of the cache.
    /// </remarks>
    public static string ProteinListIdentity(string name, IEnumerable<string>? members)
    {
        var ordered = (members ?? Array.Empty<string>())
            .Where(m => !string.IsNullOrWhiteSpace(m))
            .Select(m => m.Trim().ToUpperInvariant())
            .OrderBy(m => m, StringComparer.Ordinal)
            .ToList();
        var digest = System.Security.Cryptography.SHA256.HashData(
            System.Text.Encoding.UTF8.GetBytes(string.Join("\n", ordered)));
        return $"{name}#{ordered.Count}#{Convert.ToHexString(digest, 0, 8)}";
    }

    /// <summary>One phrase naming a set of settings, for the log line on a cache miss.</summary>
    public static string SummarizeSettings(
        string productTolerance, string precursorTolerance, string isolationScheme, int listCount) =>
        $"product {productTolerance}, precursor {precursorTolerance}, "
        + $"isolation scheme \"{isolationScheme}\", {listCount} protein list(s)";

    /// <param name="log">
    /// Told exactly what reached disk. A measurement is hours of instrument reads and the two files
    /// are written separately, so "it failed" is not enough: the summary can survive while the
    /// cycles do not, and the caller has to know which views it has lost.
    /// </param>
    /// <param name="finalize">
    /// False for a progress save mid-run, which leaves <c>ion_cycles.parquet</c> alone and keeps the
    /// cycles in the staging file beside it. See <see cref="WriteCycles"/> for why that matters.
    /// </param>
    public static void Write(
        string outputDir, IonAccountingResult result, Action<string>? log = null,
        bool finalize = true)
    {
        var rows = result.Rows;
        var n = rows.Count;

        var meta = new List<ParquetWideWriter.MetaColumn>
        {
            ParquetWideWriter.Strings("sample", rows.Select(r => r.Sample).ToArray()),
            ParquetWideWriter.Strings("sample_type", rows.Select(r => r.SampleType).ToArray()),
            ParquetWideWriter.Strings("data_file", rows.Select(r => r.DataFile).ToArray()),
            ParquetWideWriter.Strings("status", rows.Select(r => r.Status.ToString()).ToArray()),
            ParquetWideWriter.Strings("reader", rows.Select(r => r.Reader).ToArray()),
            ParquetWideWriter.Longs("ms1_count", rows.Select(r => (long)r.Ms1Count).ToArray()),
            ParquetWideWriter.Longs("ms2_count", rows.Select(r => (long)r.Ms2Count).ToArray()),
            // LINEAR counts of ions - intensity (a rate) times injection time in SECONDS. Never log.
            ParquetWideWriter.Doubles("ms1_acquired", rows.Select(r => r.Ms1Acquired).ToArray()),
            ParquetWideWriter.Doubles("ms2_acquired", rows.Select(r => r.Ms2Acquired).ToArray()),
            ParquetWideWriter.Doubles("ms1_assigned", rows.Select(r => r.Ms1Assigned).ToArray()),
            ParquetWideWriter.Doubles("ms2_assigned", rows.Select(r => r.Ms2Assigned).ToArray()),
            ParquetWideWriter.Doubles("ms2_explained", rows.Select(r => r.Ms2Explained).ToArray()),
            // The unweighted sums - what the instrument calls TIC. A DIFFERENT QUANTITY from the
            // ion columns above, not another unit for them: these are sums of rates, those are
            // counts. Never add or compare one against the other.
            ParquetWideWriter.Doubles("ms1_signal", rows.Select(r => r.Ms1Signal).ToArray()),
            ParquetWideWriter.Doubles("ms2_signal", rows.Select(r => r.Ms2Signal).ToArray()),
            ParquetWideWriter.Doubles(
                "ms1_signal_assigned", rows.Select(r => r.Ms1SignalAssigned).ToArray()),
            ParquetWideWriter.Doubles(
                "ms2_signal_assigned", rows.Select(r => r.Ms2SignalAssigned).ToArray()),
            ParquetWideWriter.Doubles(
                "ms2_signal_explained", rows.Select(r => r.Ms2SignalExplained).ToArray()),
            ParquetWideWriter.Bools("has_signal", rows.Select(r => r.HasSignal).ToArray()),
            // A flag, not an inference from a zero: an export with no charge column measures no
            // explained total at all, and a reader that read that back as 0.0 would plot a peptide
            // set that accounts for nothing rather than one that was never asked.
            ParquetWideWriter.Bools("has_explained", rows.Select(r => r.HasExplained).ToArray()),
            ParquetWideWriter.Doubles("rt_start_min", rows.Select(r => r.RtStartMin).ToArray()),
            ParquetWideWriter.Doubles("rt_stop_min", rows.Select(r => r.RtStopMin).ToArray()),
            ParquetWideWriter.Longs("claims", rows.Select(r => (long)r.Claims).ToArray()),
            ParquetWideWriter.Longs(
                "scans_outside_scheme", rows.Select(r => (long)r.ScansOutsideScheme).ToArray()),
            ParquetWideWriter.Longs(
                "missing_injection_time",
                rows.Select(r => (long)r.SpectraMissingInjectionTime).ToArray()),
            ParquetWideWriter.Longs("cycle_count", rows.Select(r => (long)r.CycleCount).ToArray()),
            // When the instrument started this replicate, round-trip UTC. A string rather than a
            // number because "not recorded" has to survive: a file that declares no start time, and
            // a cache written before this column existed, are both legitimately unknown and must not
            // read back as any particular instant.
            ParquetWideWriter.Strings(
                "acquired_utc",
                rows.Select(r => r.AcquiredUtc?.ToUniversalTime()
                    .ToString("O", System.Globalization.CultureInfo.InvariantCulture) ?? "").ToArray()),
            // Repeated per row: parquet dictionary-encodes them to nothing and it makes the file
            // self-describing to anything that opens it.
            ParquetWideWriter.Strings("product_tolerance", Repeat(result.ProductTolerance, n)),
            ParquetWideWriter.Strings("precursor_tolerance", Repeat(result.PrecursorTolerance, n)),
            ParquetWideWriter.Strings("isolation_scheme", Repeat(result.IsolationScheme, n)),
            ParquetWideWriter.Longs("assigned_peptides", Repeat((long)result.AssignedPeptides, n)),
            ParquetWideWriter.Strings("settings_key", Repeat(result.SettingsKey, n)),
            ParquetWideWriter.Longs("lists_matchable", Repeat(result.ListsMatchable ? 1L : 0L, n)),
        };

        ParquetWideWriter.Write(
            Path.Combine(outputDir, FileName), meta,
            Array.Empty<string>(), Array.Empty<double[]>(), n);

        WriteCycles(
            outputDir, result.Cycles, result.SettingsKey, result.Rows.Count, log, finalize);
        WriteLists(outputDir, result);
    }

    /// <param name="rowCount">
    /// How many replicates the result carries. Deleting is gated on this: an empty cycle list from
    /// a result that measured NOTHING is a run that read no file, and the previous run's traces
    /// should not be left behind - but an empty list from a result that does carry replicates means
    /// the cycles were lost on the way, and deleting then destroys a whole cohort's gradient data
    /// to tidy up after a failure. One of those happened.
    /// </param>
    /// <param name="finalize">
    /// Whether this write is the end of a run. <b>The real file is created ONCE, when it is.</b>
    /// Progress saves go to the staging file only - not as an optimization, but because a save
    /// happens after every replicate, so a 48-replicate run replaced <c>ion_cycles.parquet</c>
    /// forty-eight times and lost a race against a reader of its own each time. Nothing is risked
    /// by waiting: the staging file holds every cycle measured so far, and
    /// <see cref="RecoverStagedCycles"/> puts it under the real name on the next read, so an
    /// interrupted run still leaves exactly what it had.
    /// </param>
    private static void WriteCycles(
        string outputDir, IReadOnlyList<IonCycleRow> cycles, string settingsKey, int rowCount,
        Action<string>? log, bool finalize)
    {
        var path = Path.Combine(outputDir, CyclesFile);
        if (cycles.Count == 0)
        {
            if (rowCount == 0 && File.Exists(path))
                File.Delete(path);
            return;
        }

        // Written beside the real name and renamed into place. The previous version wrote straight
        // over it, so a file another process had open - which happens on a share, and did - failed
        // after fifteen retries and threw away every cycle of a 48-replicate measurement that had
        // just taken hours. Renaming cannot half-succeed, and when even the rename is refused the
        // .new file is left where a later run or a hand rename recovers the whole measurement.
        var staging = path + ".new";

        var meta = new List<ParquetWideWriter.MetaColumn>
        {
            ParquetWideWriter.Strings("sample", cycles.Select(c => c.Sample).ToArray()),
            ParquetWideWriter.Longs("cycle", cycles.Select(c => (long)c.Cycle).ToArray()),
            ParquetWideWriter.Doubles("rt_start_min", cycles.Select(c => c.RtStartMin).ToArray()),
            ParquetWideWriter.Doubles("rt_stop_min", cycles.Select(c => c.RtStopMin).ToArray()),
            ParquetWideWriter.Longs("ms1_count", cycles.Select(c => (long)c.Ms1Count).ToArray()),
            ParquetWideWriter.Longs("ms2_count", cycles.Select(c => (long)c.Ms2Count).ToArray()),
            ParquetWideWriter.Doubles("ms1_acquired", cycles.Select(c => c.Ms1Acquired).ToArray()),
            ParquetWideWriter.Doubles("ms2_acquired", cycles.Select(c => c.Ms2Acquired).ToArray()),
            ParquetWideWriter.Doubles("ms1_assigned", cycles.Select(c => c.Ms1Assigned).ToArray()),
            ParquetWideWriter.Doubles("ms2_assigned", cycles.Select(c => c.Ms2Assigned).ToArray()),
            ParquetWideWriter.Doubles("ms1_signal", cycles.Select(c => c.Ms1Signal).ToArray()),
            ParquetWideWriter.Doubles("ms2_signal", cycles.Select(c => c.Ms2Signal).ToArray()),
            ParquetWideWriter.Doubles(
                "ms1_signal_assigned", cycles.Select(c => c.Ms1SignalAssigned).ToArray()),
            ParquetWideWriter.Doubles(
                "ms2_signal_assigned", cycles.Select(c => c.Ms2SignalAssigned).ToArray()),
            ParquetWideWriter.Doubles(
                "ms2_signal_explained", cycles.Select(c => c.Ms2SignalExplained).ToArray()),
            ParquetWideWriter.Doubles("ms2_explained", cycles.Select(c => c.Ms2Explained).ToArray()),
            // The same key the summary carries, so the two files can be checked against each other.
            // They are written separately and the summary is written FIRST, so a failure between
            // them leaves a new summary beside an older set of traces - and without this the only
            // thing tying a trace to a measurement was the replicate name, which is identical
            // across runs. Repeated per row and dictionary-encoded to nothing.
            ParquetWideWriter.Strings("settings_key", Repeat(settingsKey, cycles.Count)),
        };
        ParquetWideWriter.Write(
            staging, meta, Array.Empty<string>(), Array.Empty<double[]>(), cycles.Count);
        if (!finalize)
            return;

        PlaceStagedCycles(staging, path, log);
    }

    /// <summary>
    /// Put the staged cycles under the real name - the one time a run does it.
    /// </summary>
    /// <remarks>
    /// Renaming is tried first because it cannot half-succeed. It is also the strictest operation
    /// there is: Windows refuses a rename-over while ANY handle is open on the target, even one
    /// shared for write and delete, so it fails in exactly the case this is meant to survive. An
    /// overwriting copy is not refused by a well-behaved reader, which is why it is the fallback
    /// rather than the failure. Both directions were measured, not assumed.
    /// </remarks>
    /// <summary>
    /// How long to wait for a file a scanner has just opened, as attempts x milliseconds. Settable
    /// only so the give-up path can be exercised in a second rather than half a minute; nothing
    /// outside tests changes it.
    /// </summary>
    internal static int PlacementAttempts = 30;

    /// <inheritdoc cref="PlacementAttempts"/>
    internal static int PlacementDelayMs = 1000;

    private static void PlaceStagedCycles(string staging, string path, Action<string>? log)
    {
        var maxAttempts = Math.Max(1, PlacementAttempts);
        var delayMs = Math.Max(0, PlacementDelayMs);
        Exception? last = null;
        for (var attempt = 1; attempt <= maxAttempts; attempt++)
        {
            if (TryPlace(staging, path, ref last))
            {
                RemoveStaging(staging, path, log);
                return;
            }
            if (attempt < maxAttempts)
                Thread.Sleep(delayMs);
        }

        // NOT an exception. The measurement succeeded - every replicate was read and every cycle is
        // on disk under a name the readers know to look for. Throwing here reported "Ion accounting
        // failed" after forty-eight files and several hours, for a file NAME that could not be
        // claimed. See CyclesPathFor: the staging file is read where it lies.
        log?.Invoke(
            $"  NOTE: this measurement could not be put under {CyclesFile} after {maxAttempts} "
            + $"attempts over {maxAttempts * delayMs / 1000.0:0.#} s - {last?.Message}");
        log?.Invoke($"  {Held(staging, path)}");
        log?.Invoke(
            $"  It is complete and is in {Path.GetFileName(staging)} beside it, which is where "
            + "PRISM reads it from. Nothing is lost and there is nothing to do by hand.");
    }

    /// <summary>
    /// Which of the two files is actually unavailable, said in as many words.
    /// </summary>
    /// <remarks>
    /// <b>The exception cannot be trusted for this.</b> <see cref="File.Move(string, string, bool)"/>
    /// names the DESTINATION in its message whatever went wrong, so a locked staging file was
    /// reported as "ion_cycles.parquet is being used by another process" - about a file that did not
    /// exist. Three rounds of investigation went to the wrong file on the strength of that sentence.
    ///
    /// <para>The two cases even have distinct text, which is what makes the misattribution so easy
    /// to miss: a held SOURCE gives "The process cannot access the file ... because it is being used
    /// by another process", a held DESTINATION gives "Access to the path is denied", and both quote
    /// the destination path. Measured both ways.</para>
    ///
    /// <para>The usual culprit is a virus scanner or a NAS indexer opening the file PRISM has just
    /// closed - the staging file is tens of megabytes and lands on a network share, so the scan is
    /// not instant. It opens without sharing, which blocks the copy as well as the rename.</para>
    /// </remarks>
    private static string Held(string staging, string path)
    {
        var who = FileHolders.Describe(staging) ?? FileHolders.Describe(path);
        var stagingHeld = IsUnavailable(staging);
        var targetHeld = File.Exists(path) && IsUnavailable(path);

        var which = (stagingHeld, targetHeld) switch
        {
            (true, true) => $"Both {Path.GetFileName(staging)} and {CyclesFile} are open elsewhere.",
            (true, false) => $"{Path.GetFileName(staging)} is open elsewhere - typically a scanner "
                + "reading back the file PRISM has just written.",
            (false, true) => $"{CyclesFile} is open elsewhere.",
            _ => "Neither file is held now, so whatever had one has let go since - a scan of a "
                + "freshly written file is the usual reason, and it ends when the scan does.",
        };
        return who is null
            ? which + " Nothing on this machine has either file open, so the holder is on another "
                + "machine or is the file server itself."
            : which + $" Open on this machine by: {who}.";
    }

    /// <summary>
    /// Whether a file is unavailable to the operations that PLACE it - not merely to a reader.
    /// </summary>
    /// <remarks>
    /// The share mode is the whole point, and getting it wrong made this useless. A reader asking
    /// for FileShare.ReadWrite|Delete is admitted by almost any holder, so probing that way reported
    /// "neither file is locked" about a file that a rename could not touch - which sent the
    /// investigation somewhere else twice. Asking for exclusive access is the question actually
    /// being asked: can anything replace this file right now?
    /// </remarks>
    private static bool IsUnavailable(string file)
    {
        try
        {
            using var probe = new FileStream(
                file, FileMode.Open, FileAccess.Read, FileShare.None);
            return false;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return true;
        }
    }

    /// <summary>One attempt: rename if it can, copy if it cannot.</summary>
    private static bool TryPlace(string staging, string path, ref Exception? last)
    {
        try
        {
            File.Move(staging, path, overwrite: true);
            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            last = ex;
        }

        try
        {
            File.Copy(staging, path, overwrite: true);
            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            last = ex;
            return false;
        }
    }

    /// <summary>
    /// Drop the staging file once its content is under the real name. A copy leaves it behind; a
    /// rename does not, so this is a no-op in the ordinary case. Failing to remove it is never worth
    /// failing a write that succeeded - it is a duplicate, and the next run replaces it.
    /// </summary>
    private static void RemoveStaging(string staging, string path, Action<string>? log)
    {
        if (!File.Exists(staging))
            return;
        try
        {
            File.Delete(staging);
        }
        catch (Exception cleanup) when (cleanup is IOException or UnauthorizedAccessException)
        {
            log?.Invoke(
                $"  {Path.GetFileName(staging)} could not be removed after {CyclesFile} was "
                + "written from it; it is a duplicate and the next run replaces it.");
        }
    }

    private static void WriteLists(string outputDir, IonAccountingResult result)
    {
        var path = Path.Combine(outputDir, ListsFile);
        if (result.ListNames.Count == 0)
        {
            if (File.Exists(path))
                File.Delete(path);
            return;
        }

        var samples = new List<string>();
        var names = new List<string>();
        var ms1 = new List<double>();
        var ms2 = new List<double>();
        foreach (var row in result.Rows)
        {
            for (var l = 0; l < result.ListNames.Count; l++)
            {
                samples.Add(row.Sample);
                names.Add(result.ListNames[l]);
                ms1.Add(l < row.Ms1ByList.Count ? row.Ms1ByList[l] : double.NaN);
                ms2.Add(l < row.Ms2ByList.Count ? row.Ms2ByList[l] : double.NaN);
            }
        }

        var meta = new List<ParquetWideWriter.MetaColumn>
        {
            ParquetWideWriter.Strings("sample", samples.ToArray()),
            ParquetWideWriter.Strings("list", names.ToArray()),
            ParquetWideWriter.Doubles("ms1_assigned", ms1.ToArray()),
            ParquetWideWriter.Doubles("ms2_assigned", ms2.ToArray()),
        };
        ParquetWideWriter.Write(
            path, meta, Array.Empty<string>(), Array.Empty<double[]>(), samples.Count);
    }

    /// <summary>
    /// Read the cache, or null when it is absent or unreadable. Never throws: a corrupt cache is a
    /// reason to recompute, not to fail a report.
    /// </summary>
    public static IonAccountingResult? Read(string outputDir, Action<string>? log = null)
    {
        var path = Path.Combine(outputDir, FileName);
        if (!File.Exists(path))
            return null;

        // The first thing anything does with this directory, so a measurement whose cycles write
        // was blocked is put right before anyone notices it was.
        RecoverStagedCycles(outputDir, log);

        try
        {
            using var reader = ParquetColumnReader.Open(path);
            var samples = reader.ReadStrings("sample");
            if (samples.Length == 0)
                return null;

            var types = Strings(reader, "sample_type", samples.Length);
            var files = Strings(reader, "data_file", samples.Length);
            var statuses = Strings(reader, "status", samples.Length);
            var readers = Strings(reader, "reader", samples.Length);
            var ms1Count = Nums(reader, "ms1_count", samples.Length);
            var ms2Count = Nums(reader, "ms2_count", samples.Length);
            var ms1Acq = Nums(reader, "ms1_acquired", samples.Length);
            var ms2Acq = Nums(reader, "ms2_acquired", samples.Length);
            var ms1Asg = Nums(reader, "ms1_assigned", samples.Length);
            var ms2Asg = Nums(reader, "ms2_assigned", samples.Length);

            // Optional: a cache written before the explained total existed has neither column. The
            // settings key would refuse to REUSE such a file anyway, but it is still DISPLAYED, so
            // reading it must not throw.
            var ms2Exp = reader.HasColumn("ms2_explained")
                ? Nums(reader, "ms2_explained", samples.Length)
                : new double[samples.Length];
            var hasExp = reader.HasColumn("has_explained")
                ? reader.ReadDoubles("has_explained").Select(v => v != 0).ToArray()
                : new bool[samples.Length];
            var rt0 = Nums(reader, "rt_start_min", samples.Length);
            var rt1 = Nums(reader, "rt_stop_min", samples.Length);
            var claims = Nums(reader, "claims", samples.Length);
            var outside = Nums(reader, "scans_outside_scheme", samples.Length);
            var noInj = Nums(reader, "missing_injection_time", samples.Length);
            var cycleCount = Nums(reader, "cycle_count", samples.Length);
            var acquired = Strings(reader, "acquired_utc", samples.Length);
            var ms1Sig = Nums(reader, "ms1_signal", samples.Length);
            var ms2Sig = Nums(reader, "ms2_signal", samples.Length);
            var ms1SigAsg = Nums(reader, "ms1_signal_assigned", samples.Length);
            var ms2SigAsg = Nums(reader, "ms2_signal_assigned", samples.Length);
            var ms2SigExp = Nums(reader, "ms2_signal_explained", samples.Length);
            var hasSig = reader.HasColumn("has_signal")
                ? reader.ReadDoubles("has_signal").Select(v => v != 0).ToArray()
                : new bool[samples.Length];

            var lists = ReadLists(outputDir, out var listNames);

            // Sample type is NOT in the settings key - it cannot change a number - so a cache
            // measured before the types were read would keep blank ones forever, and the only way
            // to color a bar would be to re-read every instrument file. Filled in here instead,
            // from the metadata every run writes. A type already in the file wins.
            var metadataTypes = SampleTypes(outputDir);

            var rows = new List<IonAccountingRow>(samples.Length);
            for (var i = 0; i < samples.Length; i++)
            {
                var perList = lists.GetValueOrDefault(samples[i]);
                var type = string.IsNullOrWhiteSpace(types[i])
                    ? metadataTypes.GetValueOrDefault(samples[i], "")
                    : types[i];
                rows.Add(new IonAccountingRow(
                    samples[i], type, files[i],
                    Enum.TryParse<Ms2ReadStatus>(statuses[i], out var status)
                        ? status
                        : Ms2ReadStatus.Failed,
                    readers[i],
                    (int)ms1Count[i], (int)ms2Count[i],
                    ms1Acq[i], ms2Acq[i], ms1Asg[i], ms2Asg[i],
                    ms2Exp[i], hasExp[i],
                    rt0[i], rt1[i],
                    (int)claims[i], (int)outside[i], (int)noInj[i], (int)cycleCount[i],
                    perList?.Ms1 ?? Array.Empty<double>(),
                    perList?.Ms2 ?? Array.Empty<double>(),
                    ParseUtc(acquired[i]),
                    ms1Sig[i], ms2Sig[i], ms1SigAsg[i], ms2SigAsg[i], ms2SigExp[i],
                    i < hasSig.Length && hasSig[i]));
            }

            return new IonAccountingResult(
                First(reader, "settings_key"),
                First(reader, "product_tolerance"),
                First(reader, "precursor_tolerance"),
                First(reader, "isolation_scheme"),
                listNames,
                (int)FirstNum(reader, "assigned_peptides"),
                FirstNum(reader, "lists_matchable") != 0,
                rows,
                Array.Empty<IonCycleRow>());
        }
        catch (Exception)
        {
            return null;
        }
    }

    /// <summary>
    /// One replicate's cycle traces, read on demand. Separate from <see cref="Read"/> because the
    /// summary is small and this is not: a cohort's cycles run to hundreds of thousands of rows, and
    /// the time plots only ever show one replicate at a time.
    /// </summary>
    public static IReadOnlyList<IonCycleRow> ReadCycles(string outputDir, string? sample = null)
    {
        var path = CyclesPathFor(outputDir);
        if (path is null)
            return Array.Empty<IonCycleRow>();

        try
        {
            using var reader = ParquetColumnReader.Open(path);
            var samples = reader.ReadStrings("sample");
            var cycle = reader.ReadDoubles("cycle");
            var rt0 = reader.ReadDoubles("rt_start_min");
            var rt1 = reader.ReadDoubles("rt_stop_min");
            var ms1c = reader.ReadDoubles("ms1_count");
            var ms2c = reader.ReadDoubles("ms2_count");
            var ms1a = reader.ReadDoubles("ms1_acquired");
            var ms2a = reader.ReadDoubles("ms2_acquired");
            var ms1s = reader.ReadDoubles("ms1_assigned");
            var ms2s = reader.ReadDoubles("ms2_assigned");
            var ms2e = reader.HasColumn("ms2_explained")
                ? reader.ReadDoubles("ms2_explained")
                : new double[samples.Length];
            var ms1sig = Nums(reader, "ms1_signal", samples.Length);
            var ms2sig = Nums(reader, "ms2_signal", samples.Length);
            var ms1sigA = Nums(reader, "ms1_signal_assigned", samples.Length);
            var ms2sigA = Nums(reader, "ms2_signal_assigned", samples.Length);
            var ms2sigE = Nums(reader, "ms2_signal_explained", samples.Length);

            var rows = new List<IonCycleRow>();
            for (var i = 0; i < samples.Length; i++)
            {
                if (sample is not null && !string.Equals(samples[i], sample, StringComparison.Ordinal))
                    continue;
                rows.Add(new IonCycleRow(
                    samples[i], (int)cycle[i], rt0[i], rt1[i], (int)ms1c[i], (int)ms2c[i],
                    ms1a[i], ms2a[i], ms1s[i], ms2s[i], ms2e[i],
                    ms1sig[i], ms2sig[i], ms1sigA[i], ms2sigA[i], ms2sigE[i]));
            }
            return rows;
        }
        catch (Exception)
        {
            return Array.Empty<IonCycleRow>();
        }
    }

    /// <summary>Which replicates have cycle traces cached, for a GUI replicate picker.</summary>
    /// <param name="log">
    /// Told why the answer is empty. Returning nothing looks identical whether the file is absent,
    /// locked by another process, or corrupt - and the caller turns all three into "no replicate has
    /// cached cycles to profile", which reads as a property of the data.
    /// </param>
    /// <param name="expectKey">
    /// The settings key the caller is about to reuse against, or null to take whatever is there.
    /// A trace measured under different settings is not a trace of this measurement, and the
    /// replicate names are identical across runs, so the name alone cannot tell them apart.
    /// </param>
    public static IReadOnlyList<string> SamplesWithCycles(
        string outputDir, Action<string>? log = null, string? expectKey = null)
    {
        var path = CyclesPathFor(outputDir, log);
        if (path is null)
        {
            log?.Invoke(
                $"  No {CyclesFile} in {outputDir} - the across-the-gradient views need it.");
            return Array.Empty<string>();
        }
        try
        {
            using var reader = ParquetColumnReader.Open(path);
            if (expectKey is not null && reader.HasColumn("settings_key"))
            {
                var keys = reader.ReadStrings("settings_key");
                if (keys.Length > 0 && !string.Equals(keys[0], expectKey, StringComparison.Ordinal))
                {
                    log?.Invoke(
                        $"  {CyclesFile} was measured under different settings than {FileName}, so "
                        + "none of its traces are reused - they will be measured again.");
                    return Array.Empty<string>();
                }
            }
            // A file written before the key column existed cannot be checked, and is taken as
            // before rather than thrown away: it was written by a run whose summary matched.
            return reader.ReadStrings("sample").Distinct(StringComparer.Ordinal).ToArray();
        }
        catch (Exception ex)
        {
            log?.Invoke($"  Could not read {CyclesFile}: {ex.Message}");
            return Array.Empty<string>();
        }
    }

    /// <summary>
    /// Sample type per sample id, from <c>sample_metadata.csv</c>, or empty when it has none.
    /// </summary>
    /// <remarks>
    /// Keyed on <c>sample_id</c>, not <c>sample</c>: the file carries both, and they are different
    /// things - <c>sample_id</c> is the merged table's <c>replicate__@__batch</c> key, which is what
    /// every other table here is keyed on, while <c>sample</c> is the bare replicate name.
    /// </remarks>
    internal static IReadOnlyDictionary<string, string> SampleTypes(string outputDir)
    {
        var types = new Dictionary<string, string>(StringComparer.Ordinal);
        var path = Path.Combine(outputDir, "sample_metadata.csv");
        if (!File.Exists(path))
            return types;

        try
        {
            var lines = File.ReadAllLines(path);
            if (lines.Length < 2)
                return types;

            var header = lines[0].Split(',');
            var idColumn = Array.FindIndex(
                header, h => h.Trim().Equals("sample_id", StringComparison.OrdinalIgnoreCase));
            var typeColumn = Array.FindIndex(
                header, h => h.Trim().Equals("sample_type", StringComparison.OrdinalIgnoreCase));
            if (idColumn < 0 || typeColumn < 0)
                return types;

            foreach (var line in lines.Skip(1))
            {
                var parts = line.Split(',');
                if (idColumn < parts.Length && typeColumn < parts.Length)
                    types[parts[idColumn].Trim()] = parts[typeColumn].Trim();
            }
        }
        catch (IOException)
        {
            // A label is not worth failing a report over.
        }
        return types;
    }

    private sealed record PerList(double[] Ms1, double[] Ms2);

    private static Dictionary<string, PerList> ReadLists(
        string outputDir, out IReadOnlyList<string> listNames)
    {
        listNames = Array.Empty<string>();
        var path = Path.Combine(outputDir, ListsFile);
        if (!File.Exists(path))
            return new Dictionary<string, PerList>(StringComparer.Ordinal);

        try
        {
            using var reader = ParquetColumnReader.Open(path);
            var samples = reader.ReadStrings("sample");
            var names = reader.ReadStrings("list");
            var ms1 = reader.ReadDoubles("ms1_assigned");
            var ms2 = reader.ReadDoubles("ms2_assigned");

            // The name order is the BIT order, taken from first appearance rather than sorted: the
            // masks were built against it and a sorted order would re-label every list's total.
            var order = new List<string>();
            var index = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var name in names)
            {
                if (index.ContainsKey(name))
                    continue;
                index[name] = order.Count;
                order.Add(name);
            }

            var byCycleSample = new Dictionary<string, PerList>(StringComparer.Ordinal);
            for (var i = 0; i < samples.Length; i++)
            {
                if (!byCycleSample.TryGetValue(samples[i], out var entry))
                {
                    entry = new PerList(new double[order.Count], new double[order.Count]);
                    byCycleSample[samples[i]] = entry;
                }
                var l = index[names[i]];
                entry.Ms1[l] = ms1[i];
                entry.Ms2[l] = ms2[i];
            }

            listNames = order;
            return byCycleSample;
        }
        catch (Exception)
        {
            return new Dictionary<string, PerList>(StringComparer.Ordinal);
        }
    }

    /// <summary>
    /// Put a staged cycles file under the real name, if a run left one behind.
    /// </summary>
    /// <remarks>
    /// <para>A run that never reached its end - interrupted, crashed, or blocked by a locked file -
    /// leaves everything it measured in <c>ion_cycles.parquet.new</c>. Recovering it here, from
    /// every entry point that opens the directory rather than only the one that lists replicates,
    /// is what makes that automatic instead of something a user has to be told about.</para>
    ///
    /// <para><b>Newer wins, not "only when the real one is missing".</b> The staging file is always
    /// written after the file it stages, so when it is the newer of the two it holds strictly more
    /// than the real one does. Recovering only into an empty slot loses a whole second attempt:
    /// interrupt a run at ten replicates and the next read recovers those ten; interrupt the next
    /// run at thirty and the real file already exists, so the thirty would be passed over in favor
    /// of the ten. These runs take hours and being interrupted is the case this exists for.</para>
    /// </remarks>
    public static void RecoverStagedCycles(string outputDir, Action<string>? log = null)
    {
        // A measurement in this process is actively writing that staging file after every
        // replicate. Copying a half-written one would put a torn parquet under the real name, and
        // deleting it would take away the run's own progress store - so leave a live run alone.
        // The run finalizes it itself, and a run that dies leaves the mark behind with it.
        if (IsMeasuring(outputDir))
            return;

        var path = Path.Combine(outputDir, CyclesFile);
        var staging = path + ".new";
        if (!File.Exists(staging))
            return;
        if (File.Exists(path)
            && File.GetLastWriteTimeUtc(staging) <= File.GetLastWriteTimeUtc(path))
        {
            return;
        }

        try
        {
            File.Copy(staging, path, overwrite: true);
            File.Delete(staging);
            log?.Invoke(
                $"  Recovered {CyclesFile} from a run that did not finish writing it. Nothing "
                + "measured was lost.");
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            log?.Invoke(
                $"  {CyclesFile} is held by another process, so {Path.GetFileName(staging)} could "
                + $"not be put under it - reading the measurement from {Path.GetFileName(staging)} "
                + $"instead: {ex.Message}");
        }
    }

    /// <summary>
    /// Output directories with a measurement running in THIS process.
    /// </summary>
    /// <remarks>
    /// In-process only, and deliberately: the GUI reads the same directory the run is writing, and
    /// those two are always the same process. A second PRISM on another machine writing the same
    /// output directory is a different problem, and not one a lock file would solve either.
    /// </remarks>
    private static readonly ConcurrentDictionary<string, byte> Measuring =
        new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Mark an output directory as being measured until the returned scope is disposed, so a read
    /// from elsewhere in the process does not disturb the staging file the run is writing.
    /// </summary>
    public static IDisposable MarkMeasuring(string outputDir) => new MeasuringScope(outputDir);

    private static bool IsMeasuring(string outputDir) =>
        Measuring.ContainsKey(FullPathOrSelf(outputDir));

    private static string FullPathOrSelf(string outputDir)
    {
        try
        {
            return Path.GetFullPath(outputDir);
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException or IOException)
        {
            return outputDir;
        }
    }

    private sealed class MeasuringScope : IDisposable
    {
        private readonly string _key;

        internal MeasuringScope(string outputDir)
        {
            _key = FullPathOrSelf(outputDir);
            Measuring[_key] = 0;
        }

        public void Dispose() => Measuring.TryRemove(_key, out _);
    }

    /// <summary>
    /// The file the cycles are actually in: the real one, or the staging file beside it when the
    /// real name could not be claimed. Null when there are none.
    /// </summary>
    /// <remarks>
    /// <para><b>A file name PRISM cannot claim must not cost a measurement.</b> Recovery is still
    /// tried first and is still the normal outcome - one rename, and the staging file is gone. But
    /// when the real name is held by something PRISM cannot argue with, the alternative to reading
    /// the staging file is refusing to draw anything at all, forever, over a rename. That is what
    /// happened: a 48-replicate measurement finished, every cycle on disk, and the pane said a
    /// measurement was waiting and someone should rename a file by hand.</para>
    ///
    /// <para>The stale file is NOT read in preference to it. The staging file is written after the
    /// file it stages, so the newer of the two is the one with more in it - the same rule
    /// <see cref="RecoverStagedCycles"/> uses to decide whether to act at all.</para>
    ///
    /// <para>During a measurement the staging file is ignored: the run rewrites it after every
    /// replicate, so a read could catch it half-written. The real file, stale or absent, is the
    /// honest answer until the run finishes.</para>
    /// </remarks>
    internal static string? CyclesPathFor(string outputDir, Action<string>? log = null)
    {
        var path = Path.Combine(outputDir, CyclesFile);
        var staging = path + ".new";

        if (IsMeasuring(outputDir) || !File.Exists(staging))
            return File.Exists(path) ? path : null;

        RecoverStagedCycles(outputDir, log);
        if (!File.Exists(staging))
            return File.Exists(path) ? path : null;

        // Recovery was refused. Read the staged measurement in place when it is the newer of the
        // two, which it is whenever it holds anything the real file does not.
        if (!File.Exists(path)
            || File.GetLastWriteTimeUtc(staging) > File.GetLastWriteTimeUtc(path))
        {
            return staging;
        }
        return path;
    }

    /// <summary>
    /// Why the across-the-gradient views have nothing, in a sentence a reader can act on.
    /// </summary>
    public static string DescribeMissingCycles(string outputDir)
    {
        // Asked only when there is nothing to draw, so the staging file has already been tried and
        // did not work either. It no longer asks anyone to rename anything: CyclesPathFor reads it
        // in place, so a staged file that exists is a file that was read.
        var path = CyclesPathFor(outputDir);
        if (path is not null)
        {
            return $"{Path.GetFileName(path)} is there but could not be read - see the log. It may "
                + "be mid-write, or truncated by a run that was interrupted; re-running ion "
                + "accounting rewrites it.";
        }
        return $"This directory has no {CyclesFile}, which is what the across-the-gradient views "
            + "read. Re-run ion accounting to create it.";
    }

    /// <summary>
    /// A recorded acquisition time, or null when the cache does not carry one. Unparseable is null
    /// too: an instant guessed from a malformed stamp would put a replicate somewhere specific in
    /// run order with nothing to say it was a guess.
    /// </summary>
    private static DateTime? ParseUtc(string? text) =>
        !string.IsNullOrWhiteSpace(text)
        && DateTime.TryParse(
            text, System.Globalization.CultureInfo.InvariantCulture,
            System.Globalization.DateTimeStyles.RoundtripKind | System.Globalization.DateTimeStyles.AdjustToUniversal,
            out var value)
            ? value
            : null;

    private static string[] Strings(ParquetColumnReader reader, string name, int count) =>
        reader.ColumnNames.Contains(name, StringComparer.Ordinal)
            ? reader.ReadStrings(name)
            : Enumerable.Repeat("", count).ToArray();

    private static double[] Nums(ParquetColumnReader reader, string name, int count) =>
        reader.ColumnNames.Contains(name, StringComparer.Ordinal)
            ? reader.ReadDoubles(name)
            : new double[count];

    private static string First(ParquetColumnReader reader, string name)
    {
        var values = reader.ColumnNames.Contains(name, StringComparer.Ordinal)
            ? reader.ReadStrings(name)
            : Array.Empty<string>();
        return values.Length > 0 ? values[0] : "";
    }

    private static double FirstNum(ParquetColumnReader reader, string name)
    {
        var values = reader.ColumnNames.Contains(name, StringComparer.Ordinal)
            ? reader.ReadDoubles(name)
            : Array.Empty<double>();
        return values.Length > 0 ? values[0] : 0;
    }

    private static T[] Repeat<T>(T value, int count) => Enumerable.Repeat(value, count).ToArray();

    /// <summary>How a fraction reads in a caption, or "n/a" when there is none.</summary>
    public static string Percent(double fraction) =>
        double.IsFinite(fraction)
            ? fraction.ToString("P1", CultureInfo.InvariantCulture)
            : "n/a";
}
