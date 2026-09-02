using System;
using System.Collections.Generic;
using System.Globalization;
using DuckDB.NET.Data;
using SkylinePrism.Core.IO;

namespace SkylinePrism.Core.Qc;

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
    public sealed record Columns(
        string Sample, string Peptide, string Transition, string Abundance,
        string PrecursorMz, string ProductMz, string StartTime, string EndTime);

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

        return sample is null || peptide is null || transition is null || area is null
            || precursorMz is null || productMz is null || start is null || end is null
            ? null
            : new Columns(sample, peptide, transition, area, precursorMz, productMz, start, end);
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
        IReadOnlyDictionary<string, PeptideClass> classes, int memoryBudgetMb = 0)
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
                TRY_CAST(""{cols.Abundance}"" AS DOUBLE) AS area
            FROM {MergedParquetReader.Scan(dataset.ScanTarget)}
            WHERE ""{cols.Sample}"" = '{Esc(sample)}'
              AND NOT {MergedParquetReader.IsPrecursorSql(cols.Transition)}");

        using var reader = cmd.ExecuteReader();
        var regions = new List<Ms2SignalUnion.Region>();
        // Peptide identity as a small int, so the union can tell a duplicate row of ONE peptide (Skyline
        // exports a shared peptide once per protein assignment) from genuine sharing between two.
        var peptideIds = new Dictionary<string, int>(StringComparer.Ordinal);
        var outside = 0;
        var unknown = 0;

        while (reader.Read())
        {
            var pep = reader.IsDBNull(0) ? "" : reader.GetString(0);
            var pmz = Num(reader, 1);
            var fmz = Num(reader, 2);
            var rt0 = Num(reader, 3);
            var rt1 = Num(reader, 4);
            var area = Num(reader, 5);

            // A precursor that no window covers at that time is not MS2 signal this scheme explains.
            var window = WindowIndexFor(scheme, pmz, rt0, rt1);
            if (window < 0)
            {
                outside++;
                continue;
            }

            if (!classes.TryGetValue(pep, out var cls))
            {
                cls = default;      // neither assigned nor listed
                unknown++;
            }

            if (!peptideIds.TryGetValue(pep, out var peptideId))
                peptideIds[pep] = peptideId = peptideIds.Count;

            regions.Add(new Ms2SignalUnion.Region(
                window, fmz, rt0, rt1, area, cls.Assigned, cls.ListMask, peptideId));
        }

        return new Loaded(regions, outside, unknown);
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
