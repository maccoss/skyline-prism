using System;
using System.Collections.Generic;
using System.Linq;
using DuckDB.NET.Data;
using SkylinePrism.Core.IO;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// Turns <c>merged_data/</c> into the regions of signal space a run's peptides claim, for both MS
/// levels, ready for <see cref="ClaimedSignalIndex"/>.
///
/// <para><b>Geometry and identity only - no magnitudes.</b> Nothing here reads <c>Area</c>,
/// <c>Background</c> or an ion-count column. Every number the accounting reports comes from the
/// instrument file instead, so this pass only answers "where did Skyline extract, when, and for
/// whom". That is what removes four problems at once: the units mismatch (an area is an
/// intensity-time integral, a scan total is not), double counting where two peptides share a
/// fragment mass, the background Skyline subtracts from <c>Area</c> and does not always export, and
/// the 29x cost of exporting Skyline's own per-transition ion counts.</para>
///
/// <para><b>One pass, both levels.</b> Precursor and fragment rows are read together and separated
/// in memory by the same predicate Stage 2 uses, rather than queried twice: <c>merged_data/</c> is
/// hive-partitioned on the PEPTIDE column, so neither a level filter nor a sample filter prunes
/// anything and a second query would re-read the whole cohort.</para>
/// </summary>
public static class ClaimedRegionLoader
{
    /// <param name="Regions">Deduplicated claims, ready to construct a <see cref="ClaimedSignalIndex"/>.</param>
    /// <param name="Ms1Rows">Precursor-isotope rows that became claims.</param>
    /// <param name="Ms2Rows">Fragment rows that became claims.</param>
    /// <param name="DuplicateRows">
    /// Rows whose geometry another row already claimed. Skyline exports a shared peptide once per
    /// protein assignment, so this is normally a large number and not a problem; it is reported
    /// because it is the difference between the row count and the claim count.
    /// </param>
    /// <param name="OutsideScheme">Fragments whose precursor fell in no isolation window at that
    /// time. Reported rather than dropped silently: a large count means the scheme is wrong for this
    /// run, which would otherwise look like a small assigned fraction.</param>
    /// <param name="Unassigned">Rows belonging to a peptide that did not reach the peptide matrix.
    /// Not claims - signal a target was extracted at but that the analysis does not report.</param>
    /// <param name="UnknownPeptides">Rows whose peptide the caller's identity map did not mention.</param>
    /// <param name="NoGeometry">Rows with no usable m/z or retention-time span.</param>
    /// <param name="ExplainedRegions">
    /// MS2 claims for everything the peptide can put into the spectrum, not just what Skyline
    /// quantifies on: its theoretical b/y ions and its surviving precursor isotopes, UNIONED with the
    /// quantified claims above.
    ///
    /// <para>The union is what makes explained >= quantified true by construction rather than by
    /// hope. Skyline sometimes quantifies on an ion this enumeration does not produce - a 3+
    /// fragment, a neutral loss - and without the union such a transition would count toward the
    /// quantified total and not the explained one, which reads as a defect on the plot.</para>
    ///
    /// <para>MS2 only. At MS1 the two sets would be identical (the theoretical MS1 claim IS the
    /// precursor isotope envelope, which is what Skyline already extracts), so building both would
    /// double the memory for a number that cannot differ.</para>
    /// </param>
    /// <param name="ExplainedMeasured">
    /// Whether a theoretical claim set was BUILT - which is not the same as whether it found
    /// anything. A replicate whose MS2 rows all fell outside the isolation scheme was measured and
    /// explained nothing; an export with no <c>Precursor Charge</c> column was never asked. Deriving
    /// this from the region count would collapse the two, and telling them apart is the distinction
    /// the whole explained series rests on.
    /// </param>
    /// <param name="Precursors">Distinct (peptide, charge) precursors seen, assigned ones only.</param>
    /// <param name="Unreconciled">
    /// Precursors whose sequence did not reproduce Skyline's own <c>Precursor Mz</c>, so no
    /// theoretical claim was made for them. **Reported loudly**: this is the count that reveals a
    /// modification PRISM cannot resolve - notably a heavy isotope label, which the peptide-level
    /// sequence does not carry - and its symptom is otherwise just a quietly smaller explained total.
    /// </param>
    public sealed record Loaded(
        IReadOnlyList<ClaimedRegion> Regions,
        int Ms1Rows,
        int Ms2Rows,
        int DuplicateRows,
        int OutsideScheme,
        int Unassigned,
        int UnknownPeptides,
        int NoGeometry,
        IReadOnlyList<ClaimedRegion> ExplainedRegions,
        int Precursors,
        int Unreconciled,
        bool ExplainedMeasured)
    {
        /// <inheritdoc cref="ExplainedMeasured"/>
        public bool HasExplained => ExplainedMeasured;

        /// <summary>A one-line summary for the run log, so a surprising fraction can be explained.</summary>
        public string Describe()
        {
            var line =
                $"{Regions.Count:N0} claims ({Ms1Rows:N0} MS1 + {Ms2Rows:N0} MS2 rows, "
                + $"{DuplicateRows:N0} duplicate); skipped {Unassigned:N0} unassigned, "
                + $"{OutsideScheme:N0} outside the scheme, {NoGeometry:N0} without geometry"
                + (UnknownPeptides > 0 ? $"; {UnknownPeptides:N0} peptides not in the identity map" : "");
            if (ExplainedMeasured)
            {
                line += $"; {ExplainedRegions.Count:N0} explained claims over {Precursors:N0} precursor(s)";
                if (Unreconciled > 0)
                {
                    var pct = Precursors > 0 ? 100.0 * Unreconciled / Precursors : 0;
                    line += $", {Unreconciled:N0} ({pct:0.0}%) NOT RECONCILED against Skyline's "
                        + "precursor m/z and excluded";
                }
            }
            return line;
        }
    }

    /// <summary>
    /// Load one replicate's claims.
    ///
    /// <para>Use this for a single replicate the user picked. For a whole cohort use
    /// <see cref="ForEachSample"/>: the sample filter prunes no partitions, so N replicates cost N
    /// full scans here and one there.</para>
    /// </summary>
    /// <param name="precursorTolerance">
    /// The m/z range Skyline extracted each precursor isotope over. Null drops the MS1 half rather
    /// than guessing a default, because a guessed tolerance changes how much sharing is found with
    /// nothing on the plot to show it.
    /// </param>
    public static Loaded ForReplicate(
        MergedDataset dataset, SignalColumns cols, string sample, IsolationScheme scheme,
        ProductMassTolerance? productTolerance, ProductMassTolerance? precursorTolerance,
        IReadOnlyDictionary<string, PeptideClass> classes, int memoryBudgetMb = 0)
    {
        using var conn = Connect(dataset, memoryBudgetMb);
        using var cmd = DuckDbTuning.StreamingCommand(conn, Sql(cols, dataset.ScanTarget, sample));
        using var reader = cmd.ExecuteReader();

        var block = new Accumulator(
            scheme, productTolerance, precursorTolerance, classes, cols.PrecursorCharge is not null);
        while (reader.Read())
            block.Add(reader, ordinalOffset: 0);
        return block.Take();
    }

    /// <summary>
    /// Load every replicate in one pass, handing each sample's claims to <paramref name="onSample"/>
    /// as its block completes - in ascending sample-id order, which is what the <c>ORDER BY</c> buys:
    /// a sample's rows arrive contiguously, so one replicate's claims are held at a time rather than
    /// the cohort's.
    /// </summary>
    /// <param name="wanted">
    /// Which samples the caller will actually use. Rows of any other sample are read past without
    /// being accumulated, and no <see cref="Loaded"/> is built for them.
    ///
    /// <para><b>This bounds the WORK, not just the output.</b> A replicate's explained claim set runs
    /// to millions of regions - a Large Object Heap array of a hundred megabytes and more - so
    /// building one per sample and letting the caller discard it made <c>--max 3</c> on a
    /// 93-replicate document do ninety of those for nothing, alongside DuckDB's own native buffer
    /// pool. The rows still have to be read, because sample boundaries are only known by reading
    /// them; nothing is allocated for them.</para>
    /// </param>
    public static void ForEachSample(
        MergedDataset dataset, SignalColumns cols, IsolationScheme scheme,
        ProductMassTolerance? productTolerance, ProductMassTolerance? precursorTolerance,
        IReadOnlyDictionary<string, PeptideClass> classes,
        Action<string, Loaded> onSample, int memoryBudgetMb = 0,
        Func<string, bool>? wanted = null)
    {
        if (onSample is null)
            throw new ArgumentNullException(nameof(onSample));

        using var conn = Connect(dataset, memoryBudgetMb);
        using var cmd = DuckDbTuning.StreamingCommand(conn, Sql(cols, dataset.ScanTarget, sample: null));
        using var reader = cmd.ExecuteReader();

        var block = new Accumulator(
            scheme, productTolerance, precursorTolerance, classes, cols.PrecursorCharge is not null);
        string? current = null;
        var keep = false;

        while (reader.Read())
        {
            var sample = reader.IsDBNull(0) ? "" : reader.GetString(0);
            if (!string.Equals(sample, current, StringComparison.Ordinal))
            {
                // A sample boundary, and the first row is one: current is null, which equals no
                // sample id. Deciding here means the predicate is consulted in exactly one place.
                if (current is not null && keep)
                {
                    onSample(current, block.Take());
                    block.Reset();
                }
                current = sample;
                keep = wanted is null || wanted(sample);
            }
            if (keep)
                block.Add(reader, ordinalOffset: 1);
        }

        if (current is not null && keep)
            onSample(current, block.Take());
    }

    private static DuckDBConnection Connect(MergedDataset dataset, int memoryBudgetMb)
    {
        var conn = new DuckDBConnection("Data Source=:memory:");
        conn.Open();
        DuckDbTuning.Apply(
            conn,
            memoryBudgetMb > 0 ? memoryBudgetMb : DuckDbMerge.AutoMemoryBudgetMb(),
            DuckDbMerge.ResolveTempDirectory(dataset.Root));
        return conn;
    }

    /// <summary>
    /// The projection. The precursor flag is selected as a COLUMN rather than filtered on, which is
    /// what makes this one pass over both levels.
    /// </summary>
    internal static string Sql(SignalColumns cols, string scanTarget, string? sample)
    {
        var samp = sample is null
            ? $@"""{cols.Sample}"" AS samp,"
            : "";
        var where = sample is null
            ? ""
            : $@"WHERE ""{cols.Sample}"" = '{Esc(sample)}'";
        var order = sample is null ? "ORDER BY samp" : "";

        // Selected as a literal NULL when the export has no charge column, so the ordinals below do
        // not move with the schema. An older export then simply builds no theoretical claims.
        var charge = cols.PrecursorCharge is null
            ? "CAST(NULL AS INTEGER)"
            : $@"TRY_CAST(""{cols.PrecursorCharge}"" AS INTEGER)";

        return $@"
            SELECT
                {samp}
                ""{cols.Peptide}"" AS pep,
                {MergedParquetReader.IsPrecursorSql(cols.Transition)} AS isprec,
                TRY_CAST(""{cols.PrecursorMz}"" AS DOUBLE) AS pmz,
                TRY_CAST(""{cols.ProductMz}"" AS DOUBLE) AS mz,
                TRY_CAST(""{cols.StartTime}"" AS DOUBLE) AS rt0,
                TRY_CAST(""{cols.EndTime}"" AS DOUBLE) AS rt1,
                {charge} AS pz
            FROM {MergedParquetReader.Scan(scanTarget)}
            {where}
            {order}";
    }

    /// <summary>
    /// One sample's rows becoming claims. Shared by both entry points so the single-replicate and
    /// whole-cohort paths cannot classify a row differently.
    /// </summary>
    private sealed class Accumulator
    {
        private readonly IsolationScheme _scheme;
        private readonly ProductMassTolerance? _productTolerance;
        private readonly ProductMassTolerance? _precursorTolerance;
        private readonly IReadOnlyDictionary<string, PeptideClass> _classes;

        // A set, not a list: identical geometry claimed by identical lists is one claim, and Skyline
        // exports a shared peptide once per protein assignment, so the repeats are routine. Merging
        // would collapse them anyway - this just does it before they cost memory.
        private HashSet<ClaimedRegion> _claims = new();
        private int _ms1, _ms2, _duplicates, _outside, _unassigned, _unknown, _noGeometry;

        // A LIST, not a set, unlike the quantified claims: theoretical ions are enumerated once per
        // precursor (see _seenPrecursors) and the quantified MS2 claims added here have already been
        // deduplicated by _claims, so the repeats a set would collapse have been prevented instead.
        // At roughly ten times the quantified claim count, a second hash set is memory worth saving,
        // and ClaimedSignalIndex merges overlapping ranges anyway.
        private List<ClaimedRegion> _explained = new();

        /// <summary>
        /// Theoretical ions per precursor, for pre-sizing <see cref="_explained"/>. A b/y enumeration
        /// at 1+ and 2+ over a tryptic peptide plus three precursor isotopes lands near here; the
        /// list still grows if it is wrong, but not through twenty doublings of a multi-hundred-
        /// megabyte array.
        /// </summary>
        private const int TypicalIonsPerPrecursor = 64;
        private HashSet<(string Peptide, int Charge)> _seenPrecursors = new();
        private int _precursors, _unreconciled;

        private readonly bool _wantExplained;

        public Accumulator(
            IsolationScheme scheme, ProductMassTolerance? productTolerance,
            ProductMassTolerance? precursorTolerance,
            IReadOnlyDictionary<string, PeptideClass> classes,
            bool wantExplained)
        {
            _scheme = scheme;
            _productTolerance = productTolerance;
            _precursorTolerance = precursorTolerance;
            _classes = classes;
            _wantExplained = wantExplained && productTolerance is not null;
        }

        public void Add(DuckDBDataReader reader, int ordinalOffset)
        {
            var pep = reader.IsDBNull(ordinalOffset) ? "" : reader.GetString(ordinalOffset);
            var isPrecursor = !reader.IsDBNull(ordinalOffset + 1) && reader.GetBoolean(ordinalOffset + 1);
            var pmz = Num(reader, ordinalOffset + 2);
            var mz = Num(reader, ordinalOffset + 3);
            var rt0 = Num(reader, ordinalOffset + 4);
            var rt1 = Num(reader, ordinalOffset + 5);
            var charge = reader.IsDBNull(ordinalOffset + 6)
                ? 0
                : Convert.ToInt32(reader.GetValue(ordinalOffset + 6));

            if (isPrecursor)
                _ms1++;
            else
                _ms2++;

            // Identity first: a peptide the analysis does not report claims nothing, so there is no
            // point placing it in signal space.
            if (!_classes.TryGetValue(pep, out var cls))
            {
                _unknown++;
                _unassigned++;
                return;
            }
            if (!cls.Assigned)
            {
                _unassigned++;
                return;
            }

            // The tolerance for this level. Precursor rows carry the ISOTOPE's own m/z in the product
            // column - each isotope is its own row - so both levels center on the same column.
            var tolerance = isPrecursor ? _precursorTolerance : _productTolerance;
            if (tolerance is null)
            {
                _noGeometry++;
                return;
            }

            var window = tolerance.WindowAt(mz);
            if (!double.IsFinite(window.Start) || !double.IsFinite(window.End)
                || !double.IsFinite(rt0) || !double.IsFinite(rt1) || rt1 < rt0)
            {
                _noGeometry++;
                return;
            }

            // MS1 is one lane: a survey scan measures the whole range at once, so every precursor
            // claim competes with every other. At MS2 a claim belongs to the isolation window its
            // precursor was fragmented in - two fragments of the same mass in different windows were
            // never co-isolated and are different signal.
            var windowIndex = ClaimedSignalIndex.AnyWindow;
            if (!isPrecursor)
            {
                windowIndex = WindowIndexFor(_scheme, pmz, rt0, rt1);
                if (windowIndex < 0)
                {
                    _outside++;
                    return;
                }
            }

            var claim = new ClaimedRegion(
                isPrecursor ? 1 : 2, windowIndex,
                window.Start, window.End, rt0, rt1, cls.ListMask);
            if (!_claims.Add(claim))
                _duplicates++;
            else if (!isPrecursor && _wantExplained)
                _explained.Add(claim);   // the union half: what Skyline quantifies on is explained too

            if (!isPrecursor && _wantExplained)
                AddTheoretical(pep, charge, pmz, rt0, rt1, windowIndex, cls.ListMask);
        }

        /// <summary>
        /// Everything this precursor can put into its own MS2 spectrum, claimed once per precursor
        /// rather than once per row.
        /// </summary>
        /// <remarks>
        /// <para><b>Reconciled first.</b> The masses come from PRISM's residue and modification
        /// tables, so a modification they cannot resolve would otherwise place a peptide's worth of
        /// windows on m/z belonging to nothing and count another peptide's signal as this one's.
        /// Checking the computed precursor m/z against the one Skyline exported for the same row is
        /// free and external, and turns that into a precursor that is skipped and COUNTED.</para>
        ///
        /// <para>The isolation window and retention span are the precursor's own, already resolved
        /// for the fragment row that brought us here - every transition of one precursor shares the
        /// peak boundaries Skyline integrated, so the first row to arrive carries them all.</para>
        /// </remarks>
        private void AddTheoretical(
            string peptide, int charge, double precursorMz, double rt0, double rt1,
            int windowIndex, uint listMask)
        {
            if (charge <= 0 || !double.IsFinite(precursorMz))
                return;
            if (!_seenPrecursors.Add((peptide, charge)))
                return;

            _precursors++;
            if (!PeptideFragments.Reconciles(peptide, charge, precursorMz))
            {
                _unreconciled++;
                return;
            }

            if (_explained.Capacity < _explained.Count + TypicalIonsPerPrecursor)
            {
                _explained.Capacity = Math.Max(
                    _explained.Count + TypicalIonsPerPrecursor,
                    Math.Max(1024, _precursors * TypicalIonsPerPrecursor));
            }

            foreach (var ion in PeptideFragments.Enumerate(peptide, charge))
            {
                var w = _productTolerance!.WindowAt(ion.Mz);
                if (!double.IsFinite(w.Start) || !double.IsFinite(w.End))
                    continue;
                _explained.Add(new ClaimedRegion(2, windowIndex, w.Start, w.End, rt0, rt1, listMask));
            }
        }

        public Loaded Take() => new(
            _claims.ToArray(), _ms1, _ms2, _duplicates, _outside, _unassigned, _unknown, _noGeometry,
            _explained.ToArray(), _precursors, _unreconciled, _wantExplained);

        public void Reset()
        {
            _claims = new HashSet<ClaimedRegion>();
            _explained = new List<ClaimedRegion>();
            _seenPrecursors = new HashSet<(string, int)>();
            _ms1 = _ms2 = _duplicates = _outside = _unassigned = _unknown = _noGeometry = 0;
            _precursors = _unreconciled = 0;
        }
    }

    /// <summary>
    /// The isolation window a precursor was fragmented in. Overlapping schemes (staggered DIA) can
    /// cover one m/z with several windows; the narrowest is taken, matching how
    /// <see cref="PrecursorDensityMap"/> resolves the same ambiguity, so the two views agree about
    /// which spectrum a precursor belongs to.
    /// </summary>
    internal static int WindowIndexFor(
        IsolationScheme scheme, double mz, double rtStart, double rtStop)
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
