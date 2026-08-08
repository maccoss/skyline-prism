using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using DuckDB.NET.Data;
using SkylinePrism.Core.IO;

namespace SkylinePrism.Core.Qc;

/// <summary>One detected precursor in one replicate: its m/z and the RT span of its integrated peak.</summary>
public readonly record struct DetectedPrecursor(double Mz, double RtStart, double RtStop);

/// <summary>
/// Precursor counts on an (m/z band x RT bin) grid. When the bands are the acquisition's real isolation
/// windows, one cell IS one DIA spectrum: cell (i, j) counts the precursors whose m/z falls in window i
/// and whose peak is eluting during RT bin j - i.e. how many peptide precursors that spectrum had to
/// resolve. Rows are explicit [Low, High) bands rather than a uniform bin size, because real schemes are
/// not obliged to be uniform, gapless or non-overlapping.
/// </summary>
public sealed record PrecursorDensityMap(
    IReadOnlyList<IsolationWindow> Rows,
    double RtLow, double RtBinMin,
    int[,] Counts,
    string RowSource,
    int PrecursorsOutsideRows = 0)
{
    public int MzBins => Counts.GetLength(0);
    public int RtBins => Counts.GetLength(1);
    public double RtHigh => RtLow + RtBins * RtBinMin;
    public bool IsEmpty => Counts.Length == 0 || Rows.Count == 0;
    public double MzLow => Rows.Count == 0 ? 0 : Rows.Min(w => w.Start);
    public double MzHigh => Rows.Count == 0 ? 0 : Rows.Max(w => w.End);

    /// <summary>Busiest cell (0 when empty) - the top of the color scale.</summary>
    public int MaxCount
    {
        get
        {
            var max = 0;
            foreach (var c in Counts)
                if (c > max)
                    max = c;
            return max;
        }
    }

    /// <summary>
    /// Rasterize onto a uniform grid for drawing: heatmap plottables have equal-height cells, but real
    /// isolation windows do not. Each display row takes the count of the data row covering its center
    /// (the largest, where windows overlap), so a variable-width or staggered scheme still renders with
    /// its true m/z extents.
    /// <para>NaN means "no spectrum here", and draws as a gap rather than a zero: either no window covers
    /// that m/z at all, or - for a scheduled PRM/MTM window - the window was not firing at that time. A
    /// zero therefore always means "acquired, nothing detected", which is the reading the plot is for.</para>
    /// </summary>
    public double[,] ToDisplayGrid(int displayRows = 600)
    {
        if (IsEmpty)
            return new double[0, 0];

        double low = MzLow, high = MzHigh;
        var rows = Math.Max(1, displayRows);
        var height = (high - low) / rows;
        var grid = new double[rows, RtBins];
        var candidates = new List<int>();
        for (var r = 0; r < rows; r++)
        {
            // Display row 0 is drawn at the TOP, so walk m/z downward.
            var center = high - (r + 0.5) * height;
            candidates.Clear();
            for (var i = 0; i < Rows.Count; i++)
                if (Rows[i].Contains(center))
                    candidates.Add(i);

            for (var c = 0; c < RtBins; c++)
            {
                // The source window is chosen PER CELL, not per row: which window covers a given m/z can
                // change with time. Dynamic DIA shifts its cycle's windows along m/z as the gradient runs,
                // and scheduled slots overlap in m/z at different times - picking one window per row would
                // show a single segment and blank every other one.
                var time = RtLow + (c + 0.5) * RtBinMin;
                var source = -1;
                foreach (var i in candidates)
                {
                    if (!Rows[i].IsOnAt(time))
                        continue;
                    // Narrowest window wins, so overlapping windows show the finer structure.
                    if (source < 0 || Rows[i].Width < Rows[source].Width)
                        source = i;
                }
                grid[r, c] = source < 0 ? double.NaN : Counts[source, c];
            }
        }
        return grid;
    }

    /// <summary>
    /// How many SPECTRA had each precursor load: <c>result[n]</c> is the number of spectra that had to
    /// resolve exactly <c>n</c> precursors, for n = 0..<see cref="MaxCount"/>.
    /// <para>
    /// The heatmap shows where the load is; this shows how it is distributed. A long tail says a few
    /// spectra are carrying many co-isolated precursors, which is what limits identification - and it is
    /// invisible on a map whose color scale is set by that same tail.
    /// </para>
    /// <para>
    /// Only cells that were actually ACQUIRED are counted, using the same rule as
    /// <see cref="ToDisplayGrid"/>: a scheduled window that was not firing at that time is not a
    /// spectrum with zero precursors, it is not a spectrum at all. Counting those would pile a huge
    /// spike onto bin 0 that is purely an artifact of the schedule.
    /// </para>
    /// </summary>
    public int[] PrecursorsPerSpectrumHistogram()
    {
        if (IsEmpty)
            return Array.Empty<int>();

        var histogram = new int[MaxCount + 1];
        for (var i = 0; i < MzBins; i++)
            for (var j = 0; j < RtBins; j++)
                if (WasAcquired(i, j))
                    histogram[Counts[i, j]]++;
        return histogram;
    }

    /// <summary>
    /// Per RT bin: the mean, minimum and maximum precursor load across the spectra acquired at that
    /// time, plus the bin's center time.
    /// <para>
    /// This is the load over the gradient - where the instrument is working hardest. The spread between
    /// min and max at one time says whether the load is even across the m/z range or concentrated in a
    /// few windows, which the mean alone hides.
    /// </para>
    /// <para>
    /// A time with no acquired spectrum at all yields NaN for all three rather than zero, so a gap in
    /// the schedule reads as a gap instead of as an idle instrument.
    /// </para>
    /// </summary>
    public IReadOnlyList<(double TimeMin, double Mean, double Min, double Max)> LoadOverTime()
    {
        var series = new List<(double, double, double, double)>(RtBins);
        if (IsEmpty)
            return series;

        for (var j = 0; j < RtBins; j++)
        {
            var time = RtLow + (j + 0.5) * RtBinMin;
            long sum = 0;
            var n = 0;
            var min = int.MaxValue;
            var max = 0;
            for (var i = 0; i < MzBins; i++)
            {
                if (!WasAcquired(i, j))
                    continue;
                var c = Counts[i, j];
                sum += c;
                n++;
                if (c < min) min = c;
                if (c > max) max = c;
            }
            series.Add(n == 0
                ? (time, double.NaN, double.NaN, double.NaN)
                : (time, (double)sum / n, min, max));
        }
        return series;
    }

    /// <summary>
    /// Whether row <paramref name="i"/> was firing during RT bin <paramref name="j"/>. Always true for
    /// ordinary DIA (a window is on for the whole gradient); false outside a scheduled window's interval.
    /// </summary>
    private bool WasAcquired(int i, int j) =>
        Rows[i].IsOnAt(RtLow + (j + 0.5) * RtBinMin);

    /// <summary>The row containing <paramref name="mz"/>, or -1 (used by the tool's hover readout).</summary>
    public int RowAt(double mz)
    {
        for (var i = 0; i < Rows.Count; i++)
            if (Rows[i].Contains(mz))
                return i;
        return -1;
    }
}

/// <summary>
/// "How many peptide precursors were detected in each DIA spectrum" - the m/z x RT density map behind
/// the tool's Spectrum density tab, computed from the merged transition-level report.
/// </summary>
/// <remarks>
/// Same construction as Cadenza's <c>CoverageCurves.BuildHeatmap</c> (a precursor contributes one count
/// to every RT bin its peak spans, at its m/z row), but sourced from the PRISM report rather than a
/// DIA-NN report: <c>Precursor Mz</c> gives the row, <c>Start Time</c>/<c>End Time</c> give the span, and
/// <c>Detection Q Value</c> decides what counts as detected.
/// </remarks>
public static class PrecursorDensity
{
    /// <summary>Default bin for the APPROXIMATE uniform fallback only (Cadenza's value).</summary>
    public const double DefaultMzBinTh = 2.0;

    /// <summary>Cadenza's default RT bin, in minutes.</summary>
    public const double DefaultRtBinMin = 0.1;

    /// <summary>Widen the requested bins if needed to keep the grid (and the render) bounded.</summary>
    private const int MaxBinsPerAxis = 4000;

    /// <summary>
    /// The merged-parquet columns this view needs, resolved to their actual spelling (the CSV export
    /// spells them with spaces, the parquet export without), or null when the report predates them.
    /// </summary>
    public sealed record Columns(
        string Sample, string Peptide, string PrecursorCharge, string PrecursorMz,
        string StartTime, string EndTime, string? DetectionQValue);

    /// <summary>
    /// Resolve the required columns against a merged parquet's schema. Returns null when any of them is
    /// missing, which is how the caller decides the view is unavailable for this report.
    /// </summary>
    public static Columns? Resolve(ICollection<string> available)
    {
        // "Sample ID" is the merge-synthesized, batch-disambiguated column and must win over the bare
        // replicate name, so identically named QC injections in different documents stay distinct runs
        // (same rule as SkylineColumns.Detect).
        var sample = SkylineColumns.FindColumn(available, "Sample ID", "Replicate Name");
        var peptide = SkylineColumns.FindColumn(
            available, "Peptide Modified Sequence Unimod Ids", "Peptide Modified Sequence",
            "Modified Sequence", "Peptide");
        var charge = SkylineColumns.FindColumn(available, "Precursor Charge");
        var mz = SkylineColumns.FindColumn(available, "Precursor Mz");
        var start = SkylineColumns.FindColumn(available, "Start Time", "Min Start Time");
        var end = SkylineColumns.FindColumn(available, "End Time", "Max End Time");
        if (sample is null || peptide is null || charge is null || mz is null || start is null || end is null)
            return null;

        return new Columns(
            sample, peptide, charge, mz, start, end,
            SkylineColumns.FindColumn(available, "Detection Q Value"));
    }

    /// <summary>Load one replicate's precursors, then bin them. See <see cref="Load"/> / <see cref="Bin"/>.</summary>
    public static PrecursorDensityMap Build(
        string mergedParquetPath, Columns cols, string sample,
        double mzBinTh = DefaultMzBinTh, double rtBinMin = DefaultRtBinMin, double? qValueCutoff = null)
        => Bin(Load(mergedParquetPath, cols, sample, qValueCutoff), mzBinTh, rtBinMin);

    /// <summary>Load one replicate's precursors, then bin them on a real isolation scheme.</summary>
    public static PrecursorDensityMap Build(
        string mergedParquetPath, Columns cols, string sample, IsolationScheme scheme,
        double rtBinMin = DefaultRtBinMin, double? qValueCutoff = null)
        => Bin(Load(mergedParquetPath, cols, sample, qValueCutoff), scheme, rtBinMin);

    /// <summary>
    /// One row per detected precursor in <paramref name="sample"/>. The report is transition-level, so
    /// the precursor is repeated for every fragment; the rows are collapsed by (peptide, charge, m/z) and
    /// all of a precursor's fragments share the boundaries Skyline integrated. m/z is part of the key so
    /// two label types of the same peptide (which the report does not otherwise distinguish) stay the
    /// separate precursors they are. Rows whose m/z or boundaries are missing or non-numeric (Skyline
    /// writes "#N/A" for an unintegrated peak) are dropped - those are precursors that were targeted but
    /// not detected in this run.
    /// </summary>
    public static List<DetectedPrecursor> Load(
        string mergedParquetPath, Columns cols, string sample, double? qValueCutoff = null)
    {
        var qFilter = qValueCutoff is { } q && cols.DetectionQValue is not null
            ? $" AND TRY_CAST(\"{cols.DetectionQValue}\" AS DOUBLE) <= {Num(q)}"
            : "";

        using var conn = new DuckDBConnection("Data Source=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = $@"
            SELECT mz, rt0, rt1 FROM (
                SELECT
                    MIN(TRY_CAST(""{cols.PrecursorMz}"" AS DOUBLE)) AS mz,
                    MIN(TRY_CAST(""{cols.StartTime}"" AS DOUBLE)) AS rt0,
                    MAX(TRY_CAST(""{cols.EndTime}"" AS DOUBLE)) AS rt1
                FROM read_parquet('{Esc(mergedParquetPath)}')
                WHERE ""{cols.Sample}"" = '{Esc(sample)}'{qFilter}
                GROUP BY ""{cols.Peptide}"", ""{cols.PrecursorCharge}"", ""{cols.PrecursorMz}""
            )
            WHERE mz IS NOT NULL AND rt0 IS NOT NULL AND rt1 IS NOT NULL
              AND isfinite(mz) AND isfinite(rt0) AND isfinite(rt1) AND rt1 >= rt0";

        using var reader = cmd.ExecuteReader();
        var result = new List<DetectedPrecursor>();
        while (reader.Read())
            result.Add(new DetectedPrecursor(reader.GetDouble(0), reader.GetDouble(1), reader.GetDouble(2)));
        return result;
    }

    /// <summary>
    /// Bin precursors on the acquisition's REAL isolation windows: each precursor adds a count to every
    /// window containing its m/z (more than one only for a staggered/overlapping scheme, where it really
    /// was fragmented twice), for every RT bin its peak spans. This is the honest version of the map -
    /// a cell is a spectrum, at the m/z boundaries the instrument actually used.
    /// </summary>
    /// <remarks>
    /// Precursors outside every window are counted in
    /// <see cref="PrecursorDensityMap.PrecursorsOutsideRows"/> rather than forced into the nearest row:
    /// a large count there means the scheme does not match the data (usually the wrong scheme picked for
    /// a "Results only" document), and silently clamping would hide exactly that.
    /// </remarks>
    public static PrecursorDensityMap Bin(
        IReadOnlyList<DetectedPrecursor> precursors, IsolationScheme scheme,
        double rtBinMin = DefaultRtBinMin)
    {
        if (!(rtBinMin > 0))
            throw new ArgumentOutOfRangeException(nameof(rtBinMin), rtBinMin, "RT bin must be greater than 0.");
        if (!scheme.HasWindows)
            throw new ArgumentException($"Isolation scheme '{scheme.Name}' defines no windows.", nameof(scheme));
        if (precursors.Count == 0)
            return new PrecursorDensityMap(scheme.Windows, 0, rtBinMin, new int[0, 0], scheme.Name);

        var (rtLo, nRt, rtBin) = RtGrid(precursors, rtBinMin, scheme);
        var counts = new int[scheme.Windows.Count, nRt];
        var outside = 0;
        foreach (var p in precursors)
        {
            var from = Math.Max(0, (int)((p.RtStart - rtLo) / rtBin));
            var to = Math.Min(nRt, (int)((p.RtStop - rtLo) / rtBin) + 1);
            var matched = false;
            // Covers(), not Contains(): for a scheduled (PRM/MTM) window the peak must also fall inside
            // the interval the window was firing, or a target would be credited to a same-m/z slot
            // scheduled at a completely different time.
            foreach (var row in scheme.IndicesCovering(p.Mz, p.RtStart, p.RtStop))
            {
                matched = true;
                var window = scheme.Windows[row];
                for (var j = from; j < to; j++)
                    if (window.IsOnAt(rtLo + (j + 0.5) * rtBin))
                        counts[row, j]++;
            }
            if (!matched)
                outside++;
        }
        return new PrecursorDensityMap(scheme.Windows, rtLo, rtBin, counts, scheme.Name, outside);
    }

    /// <summary>
    /// Bin precursors on a uniform m/z grid. This is the FALLBACK for when the acquisition's real windows
    /// are unknown: the cell edges are arbitrary, so a cell only approximates a spectrum. Prefer the
    /// <see cref="IsolationScheme"/> overload, and label the plot honestly when using this one.
    /// </summary>
    public static PrecursorDensityMap Bin(
        IReadOnlyList<DetectedPrecursor> precursors,
        double mzBinTh = DefaultMzBinTh, double rtBinMin = DefaultRtBinMin)
    {
        if (!(mzBinTh > 0))
            throw new ArgumentOutOfRangeException(nameof(mzBinTh), mzBinTh, "m/z bin must be greater than 0.");
        if (!(rtBinMin > 0))
            throw new ArgumentOutOfRangeException(nameof(rtBinMin), rtBinMin, "RT bin must be greater than 0.");
        if (precursors.Count == 0)
            return new PrecursorDensityMap(
                Array.Empty<IsolationWindow>(), 0, rtBinMin, new int[0, 0], UniformSource(mzBinTh));

        double mzLo = double.PositiveInfinity, mzHi = double.NegativeInfinity;
        foreach (var p in precursors)
        {
            if (p.Mz < mzLo) mzLo = p.Mz;
            if (p.Mz > mzHi) mzHi = p.Mz;
        }
        mzLo = Math.Floor(mzLo);
        mzHi = Math.Ceiling(mzHi);

        // A bin far below the data's own resolution would allocate a huge grid and render as noise, so
        // widen it until the axis fits. Reported back in the map, so the plot never claims a bin it
        // did not use.
        mzBinTh = Math.Max(mzBinTh, (mzHi - mzLo) / MaxBinsPerAxis);
        var nMz = Math.Max(1, (int)Math.Ceiling((mzHi - mzLo) / mzBinTh));
        var rows = new IsolationWindow[nMz];
        for (var i = 0; i < nMz; i++)
            rows[i] = new IsolationWindow(mzLo + i * mzBinTh, mzLo + (i + 1) * mzBinTh);

        var (rtLo, nRt, rtBin) = RtGrid(precursors, rtBinMin);
        var counts = new int[nMz, nRt];
        foreach (var p in precursors)
        {
            var row = Math.Clamp((int)((p.Mz - mzLo) / mzBinTh), 0, nMz - 1);
            var from = Math.Max(0, (int)((p.RtStart - rtLo) / rtBin));
            var to = Math.Min(nRt, (int)((p.RtStop - rtLo) / rtBin) + 1);
            for (var j = from; j < to; j++)
                counts[row, j]++;
        }
        return new PrecursorDensityMap(rows, rtLo, rtBin, counts, UniformSource(mzBinTh));
    }

    /// <summary>Label that marks a map as approximate, so it can never be mistaken for real windows.</summary>
    public static string UniformSource(double mzBinTh) =>
        $"uniform {mzBinTh.ToString("0.###", CultureInfo.InvariantCulture)} Th bins (approximate)";

    private static (double RtLow, int Bins, double BinSize) RtGrid(
        IReadOnlyList<DetectedPrecursor> precursors, double rtBinMin, IsolationScheme? scheme = null)
    {
        double rtLo = double.PositiveInfinity, rtHi = double.NegativeInfinity;
        foreach (var p in precursors)
        {
            if (p.RtStart < rtLo) rtLo = p.RtStart;
            if (p.RtStop > rtHi) rtHi = p.RtStop;
        }
        // A scheduled method's own schedule also defines the time axis: slots that fired but detected
        // nothing must still appear, or the map silently drops the parts of the method that failed - which
        // is exactly what someone looks at this plot to find.
        if (scheme is not null)
        {
            foreach (var w in scheme.Windows)
            {
                if (!w.IsScheduled)
                    continue;
                if (w.RtStart < rtLo) rtLo = w.RtStart;
                if (w.RtStop > rtHi) rtHi = w.RtStop;
            }
        }
        var bin = Math.Max(rtBinMin, (rtHi - rtLo) / MaxBinsPerAxis);
        return (rtLo, Math.Max(1, (int)Math.Ceiling((rtHi - rtLo) / bin)), bin);
    }

    private static string Num(double v) => v.ToString("R", CultureInfo.InvariantCulture);

    private static string Esc(string s) => s.Replace("'", "''");
}
