using System;
using System.Collections.Generic;
using System.Globalization;
using DuckDB.NET.Data;
using SkylinePrism.Core.IO;

using System.Linq;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// What the MS2 signal accounting totals.
/// </summary>
public enum Ms2SignalMeasure
{
    /// <summary>
    /// Integrated peak areas, gross (<c>Area + Background</c>), against an acquired total ion
    /// current. Works on any export, but the two sides are not the same quantity: a peak area is an
    /// intensity-time integral and a summed TIC is an intensity, so the ratio needs the acquisition's
    /// cycle duration applied before it means anything.
    /// </summary>
    Signal,

    /// <summary>
    /// Skyline's own ion counts - intensity times injection time, per spectrum, summed across the
    /// peak. Both sides become counts of ions, neither is background subtracted, and no unit or
    /// background correction is needed. Requires an export carrying the LC Peak ion-count columns.
    /// </summary>
    Ions,
}

/// <summary>
/// Reads one replicate's integrated fragment peaks out of <c>merged_data/</c> and places each in MS2
/// signal space, ready for <see cref="Ms2SignalUnion"/>.
///
/// <para><b>Raw values only.</b> Areas come from <c>merged_data/</c>'s abundance column — LINEAR raw
/// Skyline peak areas, untouched by normalization, ComBat or marker adjustment. Nothing here reads a
/// value out of a pipeline output; the caller supplies peptide IDENTITY (which peptides survived, and
/// which lists claim them) and identity alone.</para>
///
/// <para>Modelled on <see cref="PrecursorDensity.Load"/>, which already reads <c>merged_data/</c> at QC
/// time: one bounded, single-threaded, streaming connection per call. The result set scales with the
/// data — roughly 450k rows for a real replicate — so streaming mode is not optional here.</para>
/// </summary>
public static class Ms2SignalRegions
{
    /// <summary>
    /// Columns this view needs, resolved to whatever the export spelled them. Beyond
    /// <see cref="SkylineColumns"/> because that type does not bind <c>StartTime</c>/<c>EndTime</c>,
    /// which only the density view had needed until now.
    /// </summary>
    /// <param name="Background">Skyline's <c>Background</c> column, or null when the export predates
    /// it. See <see cref="GrossSignalSql"/> for why the accounting wants it.</param>
    /// <param name="TransitionIons">Skyline's <c>LC Peak Transition Ion Count</c> - the per-transition
    /// ion count, summed over the spectra inside the peak boundaries. Null when the export predates
    /// Skyline's ion counting, which is what makes <see cref="Ms2SignalMeasure.Ions"/> unavailable.</param>
    public sealed record Columns(
        string Sample, string Peptide, string Transition, string Abundance,
        string PrecursorMz, string ProductMz, string StartTime, string EndTime,
        string? Background = null, string? TransitionIons = null)
    {
        /// <summary>Whether this export can support <see cref="Ms2SignalMeasure.Ions"/>.</summary>
        public bool HasIonCounts => TransitionIons is not null;
    }

    /// <summary>
    /// The signal expression the accounting sums: <c>Area + Background</c> where the export carries
    /// the background, and <c>Area</c> alone where it does not.
    ///
    /// <para><b>Why gross and not net.</b> Skyline's <c>Area</c> is BACKGROUND-SUBTRACTED - its own
    /// test asserts that integrating without background yields <c>Area + BackgroundArea</c> - while
    /// the acquired total ion current this is compared against includes background, because a TIC is
    /// every ion the detector counted. Dividing a net numerator by a gross denominator understates
    /// the assigned fraction by however much background the peaks sit on, which is not a small or
    /// predictable amount in DIA.</para>
    ///
    /// <para><b>Only for this accounting.</b> Quantification keeps the net area, which is the right
    /// quantity for comparing a peptide between samples - adding background back would put detector
    /// baseline into every abundance. The gross figure exists solely to be comparable with a gross
    /// denominator.</para>
    /// </summary>
    /// <summary>
    /// Skyline's per-transition ion count column - the one <see cref="Ms2SignalMeasure.Ions"/> reads -
    /// by whatever spelling an export or the merge gave it, or null when the export has none.
    /// <c>FindColumn</c> matches ignoring case, spaces and underscores, so "LC Peak Transition Ion
    /// Count" is found however it was written. The Apex variant is deliberately NOT accepted: it is
    /// one spectrum, not a total. Public because the Skyline tool asks the same question of a
    /// pre-exported report before offering <c>ions</c> as a choice.
    /// </summary>
    public static string? FindIonCountColumn(IEnumerable<string> available)
    {
        var names = available as ICollection<string> ?? available.ToList();
        return SkylineColumns.FindColumn(
            names, "LC Peak Transition Ion Count", "LC Peak Analyte Ion Count Fragment");
    }

    internal static string GrossSignalSql(Columns cols, Ms2SignalMeasure measure = Ms2SignalMeasure.Signal)
    {
        // Ions are already gross and already a count, so there is nothing to add or convert - which
        // is the whole reason to prefer them.
        if (measure == Ms2SignalMeasure.Ions && cols.TransitionIons is not null)
            return $@"TRY_CAST(""{cols.TransitionIons}"" AS DOUBLE)";

        return cols.Background is null
            ? $@"TRY_CAST(""{cols.Abundance}"" AS DOUBLE)"
            : $@"COALESCE(TRY_CAST(""{cols.Abundance}"" AS DOUBLE), 0)
                 + COALESCE(TRY_CAST(""{cols.Background}"" AS DOUBLE), 0)";
    }

    /// <summary>
    /// Resolve against a merged table's actual column names, or null when it lacks one. A report
    /// exported without <c>Product Mz</c> cannot place fragments in m/z and simply has no accounting.
    /// </summary>
    public static Columns? Resolve(ICollection<string> available)
    {
        var sample = SkylineColumns.FindColumn(available, "Sample ID", "Replicate Name");
        var peptide = SkylineColumns.FindColumn(
            available, "Peptide Modified Sequence Unimod Ids", "Peptide Modified Sequence", "Peptide");
        var transition = SkylineColumns.FindColumn(available, "Fragment Ion");
        var area = SkylineColumns.FindColumn(available, "Area");
        var precursorMz = SkylineColumns.FindColumn(available, "Precursor Mz");
        var productMz = SkylineColumns.FindColumn(available, "Product Mz");
        var start = SkylineColumns.FindColumn(available, "Start Time");
        var end = SkylineColumns.FindColumn(available, "End Time");

        // Optional: an export made before the Background column was added still works, on net area,
        // and the caller says so rather than quietly reporting a lower fraction.
        var background = SkylineColumns.FindColumn(available, "Background");

        var transitionIons = FindIonCountColumn(available);

        return sample is null || peptide is null || transition is null || area is null
            || precursorMz is null || productMz is null || start is null || end is null
            ? null
            : new Columns(
                sample, peptide, transition, area, precursorMz, productMz, start, end, background,
                transitionIons);
    }

    /// <summary>How a peptide was classified by the run, supplied by the caller as pure identity.</summary>
    /// <param name="Assigned">Peptide reached the peptide matrix (the row set of peptides_rollup).</param>
    /// <param name="ListMask">Bit per selected protein list claiming it.</param>
    public readonly record struct PeptideClass(bool Assigned, uint ListMask);

    /// <param name="Regions">One per integrated fragment peak, placed in signal space.</param>
    /// <param name="OutsideScheme">Fragments whose precursor fell in no isolation window at that time.
    /// Reported rather than dropped silently: a large count means the scheme is wrong for this run,
    /// which would otherwise look like a small assigned fraction.</param>
    public sealed record Loaded(
        IReadOnlyList<Ms2SignalUnion.Region> Regions, int OutsideScheme, int UnknownPeptides);

    /// <summary>
    /// Load one replicate's fragment peaks. Precursor (MS1) rows are excluded using the same predicate
    /// Stage 2 uses, so the accounting covers exactly the rows the rollup considered.
    /// </summary>
    /// <param name="classes">Peptide identity. A peptide absent from it is neither assigned nor listed.</param>
    /// <param name="scheme">Isolation windows; fragments outside every window are counted, not used.</param>
    public static Loaded Load(
        MergedDataset dataset, Columns cols, string sample, IsolationScheme scheme,
        IReadOnlyDictionary<string, PeptideClass> classes, int memoryBudgetMb = 0,
        Ms2SignalMeasure measure = Ms2SignalMeasure.Signal)
    {
        using var conn = new DuckDBConnection("Data Source=:memory:");
        conn.Open();
        DuckDbTuning.Apply(
            conn,
            memoryBudgetMb > 0 ? memoryBudgetMb : DuckDbMerge.AutoMemoryBudgetMb(),
            DuckDbMerge.ResolveTempDirectory(dataset.Root));

        // Not aggregated: every transition is its own region, because two peptides sharing a fragment
        // mass are exactly what this is measuring and a GROUP BY would erase them.
        using var cmd = DuckDbTuning.StreamingCommand(conn, $@"
            SELECT
                ""{cols.Peptide}"" AS pep,
                TRY_CAST(""{cols.PrecursorMz}"" AS DOUBLE) AS pmz,
                TRY_CAST(""{cols.ProductMz}"" AS DOUBLE) AS fmz,
                TRY_CAST(""{cols.StartTime}"" AS DOUBLE) AS rt0,
                TRY_CAST(""{cols.EndTime}"" AS DOUBLE) AS rt1,
                {GrossSignalSql(cols, measure)} AS area
            FROM {MergedParquetReader.Scan(dataset.ScanTarget)}
            WHERE ""{cols.Sample}"" = '{Esc(sample)}'
              AND NOT {MergedParquetReader.IsPrecursorSql(cols.Transition)}");

        using var reader = cmd.ExecuteReader();
        // Peptide identity becomes a small int inside the accumulator, so the union can tell a duplicate
        // row of ONE peptide (Skyline exports a shared peptide once per protein assignment) from genuine
        // sharing between two.
        var block = new Accumulator();
        while (reader.Read())
            block.Add(reader, ordinalOffset: 0, scheme, classes);

        return block.Take();
    }

    /// <summary>
    /// Load every replicate in one pass, handing each sample's regions to <paramref name="onSample"/>
    /// as its block completes.
    ///
    /// <para><b>Why one pass and not one per replicate.</b> <c>merged_data/</c> is hive-partitioned on
    /// the PEPTIDE column, so a filter on the sample prunes nothing - a per-replicate query reads the
    /// whole cohort. On a 192-replicate cohort that is 192 full scans of ~47 GB. This runs one scan and
    /// pays for an ORDER BY instead, which DuckDB spills to <c>temp_directory</c>, so peak memory is one
    /// replicate's regions (~22 MB) rather than the cohort's.</para>
    ///
    /// <para>The ordering is what makes the streaming safe: a sample's rows arrive contiguously, so the
    /// accumulator can be flushed and reused. Without it every sample would have to be held at once -
    /// ~89M regions on that cohort.</para>
    /// </summary>
    /// <param name="onSample">Called once per sample, in ascending sample-id order.</param>
    public static void LoadBySample(
        MergedDataset dataset, Columns cols, IsolationScheme scheme,
        IReadOnlyDictionary<string, PeptideClass> classes,
        Action<string, Loaded> onSample, int memoryBudgetMb = 0,
        Ms2SignalMeasure measure = Ms2SignalMeasure.Signal)
    {
        using var conn = new DuckDBConnection("Data Source=:memory:");
        conn.Open();
        DuckDbTuning.Apply(
            conn,
            memoryBudgetMb > 0 ? memoryBudgetMb : DuckDbMerge.AutoMemoryBudgetMb(),
            DuckDbMerge.ResolveTempDirectory(dataset.Root));

        using var cmd = DuckDbTuning.StreamingCommand(conn, $@"
            SELECT
                ""{cols.Sample}"" AS samp,
                ""{cols.Peptide}"" AS pep,
                TRY_CAST(""{cols.PrecursorMz}"" AS DOUBLE) AS pmz,
                TRY_CAST(""{cols.ProductMz}"" AS DOUBLE) AS fmz,
                TRY_CAST(""{cols.StartTime}"" AS DOUBLE) AS rt0,
                TRY_CAST(""{cols.EndTime}"" AS DOUBLE) AS rt1,
                {GrossSignalSql(cols, measure)} AS area
            FROM {MergedParquetReader.Scan(dataset.ScanTarget)}
            WHERE NOT {MergedParquetReader.IsPrecursorSql(cols.Transition)}
            ORDER BY samp");

        using var reader = cmd.ExecuteReader();
        var block = new Accumulator();
        string? current = null;

        while (reader.Read())
        {
            var sample = reader.IsDBNull(0) ? "" : reader.GetString(0);
            if (current is not null && !string.Equals(sample, current, StringComparison.Ordinal))
            {
                onSample(current, block.Take());
                block.Reset();
            }
            current = sample;
            block.Add(reader, ordinalOffset: 1, scheme, classes);
        }

        if (current is not null)
            onSample(current, block.Take());
    }

    /// <summary>
    /// One sample's rows being turned into regions. Shared by both entry points so the single-replicate
    /// and whole-cohort paths cannot classify a row differently.
    /// </summary>
    private sealed class Accumulator
    {
        private readonly Dictionary<string, int> _peptideIds = new(StringComparer.Ordinal);
        private List<Ms2SignalUnion.Region> _regions = new();
        private int _outside;
        private int _unknown;

        public void Add(
            DuckDBDataReader reader, int ordinalOffset, IsolationScheme scheme,
            IReadOnlyDictionary<string, PeptideClass> classes)
        {
            var pep = reader.IsDBNull(ordinalOffset) ? "" : reader.GetString(ordinalOffset);
            var pmz = Num(reader, ordinalOffset + 1);
            var fmz = Num(reader, ordinalOffset + 2);
            var rt0 = Num(reader, ordinalOffset + 3);
            var rt1 = Num(reader, ordinalOffset + 4);
            var area = Num(reader, ordinalOffset + 5);

            // A precursor that no window covers at that time is not MS2 signal this scheme explains.
            var window = WindowIndexFor(scheme, pmz, rt0, rt1);
            if (window < 0)
            {
                _outside++;
                return;
            }

            if (!classes.TryGetValue(pep, out var cls))
            {
                cls = default;      // neither assigned nor listed
                _unknown++;
            }

            if (!_peptideIds.TryGetValue(pep, out var peptideId))
                _peptideIds[pep] = peptideId = _peptideIds.Count;

            _regions.Add(new Ms2SignalUnion.Region(
                window, fmz, rt0, rt1, area, cls.Assigned, cls.ListMask, peptideId));
        }

        /// <summary>The block so far, handing off the list rather than copying it.</summary>
        public Loaded Take() => new(_regions, _outside, _unknown);

        public void Reset()
        {
            _regions = new List<Ms2SignalUnion.Region>();
            _peptideIds.Clear();
            _outside = 0;
            _unknown = 0;
        }
    }

    /// <summary>
    /// The isolation window a precursor was fragmented in. Overlapping schemes (staggered DIA) can
    /// cover one m/z with several windows; the narrowest is taken, matching how
    /// <see cref="PrecursorDensityMap"/> resolves the same ambiguity so the two views agree about
    /// which spectrum a precursor belongs to.
    /// </summary>
    private static int WindowIndexFor(IsolationScheme scheme, double mz, double rtStart, double rtStop)
    {
        if (!double.IsFinite(mz))
            return -1;

        var best = -1;
        var bestWidth = double.PositiveInfinity;
        for (var i = 0; i < scheme.Windows.Count; i++)
        {
            var w = scheme.Windows[i];
            if (!w.Contains(mz))
                continue;
            // A scheduled window only explains signal while it was firing.
            if (double.IsFinite(rtStart) && double.IsFinite(rtStop)
                && !w.IsOnAt(rtStart) && !w.IsOnAt(rtStop))
                continue;
            if (w.Width < bestWidth)
            {
                bestWidth = w.Width;
                best = i;
            }
        }
        return best;
    }

    private static double Num(DuckDBDataReader reader, int ordinal) =>
        reader.IsDBNull(ordinal) ? double.NaN : reader.GetDouble(ordinal);

    /// <summary>Single quotes doubled, the SQL-literal rule the other readers here use.</summary>
    private static string Esc(string s) => s.Replace("'", "''", StringComparison.Ordinal);
}
