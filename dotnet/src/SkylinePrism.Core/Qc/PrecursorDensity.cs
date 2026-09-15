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
/// Co-eluting precursor counts on an (m/z band x RT column) grid. Cell (i, j) is the GREATEST number of
/// precursors whose m/z falls in window i and which were eluting AT ANY ONE INSTANT inside column j -
/// how many peptide precursors a single spectrum had to resolve.
/// <para>
/// At an instant, not over the column: a cell is not a tally of everything that passed through that
/// stretch of time. Two peptides, one finishing before the other starts, are one and one, never two -
/// no spectrum ever saw both. That is what keeps the number from growing with the column width, and it
/// is why cells cannot be summed or averaged across columns as though they were counts. Peaks are CLOSED
/// intervals: two that meet at exactly one instant were both in a spectrum acquired then, and count as
/// two in the column holding that instant.
/// </para>
/// <para>
/// When the bands are the acquisition's real isolation windows and the column is about one acquisition
/// cycle, one cell IS one DIA spectrum. Rows are explicit [Low, High) bands rather than a uniform bin
/// size, because real schemes are not obliged to be uniform, gapless or non-overlapping.
/// </para>
/// </summary>
public sealed record PrecursorDensityMap(
    IReadOnlyList<IsolationWindow> Rows,
    double RtLow, double RtBinMin,
    int[,] Counts,
    string RowSource,
    int PrecursorsOutsideRows = 0,
    bool RowsAreWindows = true,
    double RtBinRequested = 0)
{
    // RowsAreWindows: whether a cell IS a spectrum. True when Rows are the acquisition's real isolation
    // windows; false on the approximate uniform-bin fallback, where a row is a bin no single spectrum
    // covered. Carried explicitly rather than inferred from RowSource - a display string is not a
    // contract, and what a cell counts is the one thing a reader must not be told wrongly.

    /// <summary>
    /// Whether the RT bin had to be widened past what the caller asked for.
    /// </summary>
    /// <remarks>
    /// <para>The same class of fact as <see cref="RowsAreWindows"/>, on the other axis: what a cell
    /// counts is the one thing a reader must not be told wrongly.</para>
    ///
    /// <para><b>It no longer means the counts run high.</b> It did when a cell held the union of
    /// everything that eluted during its column - widening then pooled cycles and inflated the
    /// number. A cell now holds the greatest number of precursors co-eluting at any ONE INSTANT
    /// inside its column, so the maximum over the map is the same however the columns are cut. That
    /// bin independence is the whole point of the sweep, and it retired the inflation this flag was
    /// added to warn about.</para>
    ///
    /// <para>What widening still changes is WHICH spectrum each cell speaks for. At about one
    /// acquisition cycle a column is one spectrum, and the cell is what that spectrum resolved.
    /// Wider, every cell reports the worst spectrum inside it rather than a typical one - so more
    /// cells carry a high number and <see cref="PrecursorsPerSpectrumHistogram"/> shifts right, while
    /// the extreme stays honest. The peak is reliable either way; the distribution is not.</para>
    /// </remarks>
    public bool RtBinWidened => RtBinRequested > 0 && RtBinMin > RtBinRequested * 1.001;

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
    /// that m/z at all, or - for a scheduled (dynamic DIA) window - it was not firing at that time. A
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
                var source = -1;
                foreach (var i in candidates)
                {
                    if (!WasAcquired(i, c))
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
    /// The distribution of the map's cells: <c>result[n]</c> is the number of acquired cells whose load
    /// is exactly <c>n</c>, for n = 0..<see cref="MaxCount"/>. A cell is one spectrum only while a column
    /// is about one acquisition cycle wide - <see cref="RtBinWidened"/> says what a wider one makes it.
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
    /// Averaged across WINDOWS at one time; each window contributes its cell at that column, which is one
    /// spectrum's load at about one cycle and its worst spectrum's when wider (<see cref="RtBinWidened"/>).
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
    /// Whether row <paramref name="i"/> was firing during RT column <paramref name="j"/> - the ONE rule
    /// for "this cell is a spectrum", shared with the fill through
    /// <see cref="PrecursorDensity.ColumnAcquired"/>, so a count can never land in a cell that
    /// <see cref="ToDisplayGrid"/> draws as a gap and the summaries skip. Always true for ordinary DIA
    /// (a window is on for the whole gradient); false outside a scheduled window's interval.
    /// </summary>
    private bool WasAcquired(int i, int j) => PrecursorDensity.ColumnAcquired(Rows[i], RtLow, RtBinMin, j);

    /// <summary>The row containing <paramref name="mz"/>, or -1 (used by the tool's hover readout).</summary>
    public int RowAt(double mz)
    {
        for (var i = 0; i < Rows.Count; i++)
            if (Rows[i].Contains(mz))
                return i;
        return -1;
    }

    /// <summary>
    /// The column containing retention time <paramref name="rt"/>, or -1 when it is off the axis. The
    /// sibling of <see cref="RowAt"/> for the hover readouts. Floor rather than a cast: a cast truncates
    /// toward zero, so a cursor just left of the axis landed on column 0 and read out a cell it was not
    /// over.
    /// </summary>
    public int ColumnAt(double rt)
    {
        if (!double.IsFinite(rt) || rt < RtLow)
            return -1;
        var col = (int)Math.Floor((rt - RtLow) / RtBinMin);
        return col < RtBins ? col : -1;
    }
}

/// <summary>
/// The m/z x RT co-elution map behind the tool's Spectrum density tab, computed from the merged
/// transition-level report. What a cell holds is defined once, on <see cref="PrecursorDensityMap"/>.
/// </summary>
/// <remarks>
/// The layout is Cadenza's <c>CoverageCurves.BuildHeatmap</c> - isolation windows across retention time -
/// but not its counting: Cadenza adds one count to every RT bin a peak spans, which pools everything that
/// eluted during the bin, where a cell here is the greatest number co-eluting at any one instant. Sourced
/// from the PRISM report rather than a DIA-NN report: <c>Precursor Mz</c> gives the row,
/// <c>Start Time</c>/<c>End Time</c> give the span, and <c>Detection Q Value</c> decides what counts as
/// detected.
/// </remarks>
public static class PrecursorDensity
{
    /// <summary>Default bin for the APPROXIMATE uniform fallback only (Cadenza's value).</summary>
    public const double DefaultMzBinTh = 2.0;

    /// <summary>
    /// The default RT bin, in minutes.
    /// </summary>
    /// <remarks>
    /// <para><b>About one acquisition cycle, which is what makes a cell one spectrum.</b> A cell holds
    /// the greatest number of precursors co-eluting at any instant inside its column
    /// (<see cref="PrecursorDensityMap"/>), so the busiest cell does not move with the bin - but WHICH
    /// spectrum a cell speaks for does. At about one cycle it is the spectrum acquired then; wider, it is
    /// the worst spectrum inside the column, and the histogram and load curve become distributions of
    /// column maxima rather than of spectra (<see cref="PrecursorDensityMap.RtBinWidened"/>).</para>
    ///
    /// <para>0.01 min is 0.6 s, about one cycle on the cohorts this was built for, and the same bin
    /// the ion accounting views default to. It was 0.1 - ten times a cycle - inherited from Cadenza,
    /// and the tool's RT bin box kept that value after the constant changed, which is why
    /// <c>DensityPaneDefaultsTests</c> pins the two together. Do not widen it for a smoother-looking
    /// plot: the smoothing is what turns "one spectrum" into "the worst of ten".</para>
    /// </remarks>
    public const double DefaultRtBinMin = 0.01;

    /// <summary>Widen the requested m/z bin if needed to keep the grid (and the render) bounded.</summary>
    private const int MaxBinsPerAxis = 4000;

    /// <summary>
    /// The RT axis gets its own, larger bound, because it is the axis a fine bin is actually wanted
    /// on: 0.01 min over a two-hour gradient is 12,000 bins, and the old shared cap of 4,000 would
    /// have widened it back to 0.03 - silently returning a coarser map than the one asked for.
    /// </summary>
    private const int MaxRtBins = 20000;

    /// <summary>
    /// The real bound, which neither axis cap expresses on its own: the grid is
    /// <c>nMz x nRt</c> ints, so a fine bin on both axes at once is what runs the machine out of
    /// memory. At 4 bytes a cell this is about 48 MB, and the RT bin is the one widened to stay
    /// inside it - the m/z rows are the acquisition's own isolation windows and are not PRISM's to
    /// coarsen.
    /// </summary>
    private const long MaxCells = 12_000_000;

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
        MergedDataset dataset, Columns cols, string sample,
        double mzBinTh = DefaultMzBinTh, double rtBinMin = DefaultRtBinMin, double? qValueCutoff = null)
        => Bin(Load(dataset, cols, sample, qValueCutoff), mzBinTh, rtBinMin);

    /// <summary>Load one replicate's precursors, then bin them on a real isolation scheme.</summary>
    public static PrecursorDensityMap Build(
        MergedDataset dataset, Columns cols, string sample, IsolationScheme scheme,
        double rtBinMin = DefaultRtBinMin, double? qValueCutoff = null)
        => Bin(Load(dataset, cols, sample, qValueCutoff), scheme, rtBinMin);

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
        MergedDataset dataset, Columns cols, string sample, double? qValueCutoff = null)
    {
        var qFilter = qValueCutoff is { } q && cols.DetectionQValue is not null
            ? $" AND TRY_CAST(\"{cols.DetectionQValue}\" AS DOUBLE) <= {Num(q)}"
            : "";

        using var conn = new DuckDBConnection("Data Source=:memory:");
        conn.Open();
        DuckDbTuning.Apply(
            conn, DuckDbMerge.AutoMemoryBudgetMb(), DuckDbMerge.ResolveTempDirectory(dataset.Root));
        using var cmd = DuckDbTuning.StreamingCommand(conn, $@"
            SELECT mz, rt0, rt1 FROM (
                SELECT
                    MIN(TRY_CAST(""{cols.PrecursorMz}"" AS DOUBLE)) AS mz,
                    MIN(TRY_CAST(""{cols.StartTime}"" AS DOUBLE)) AS rt0,
                    MAX(TRY_CAST(""{cols.EndTime}"" AS DOUBLE)) AS rt1
                FROM {MergedParquetReader.Scan(dataset.ScanTarget)}
                WHERE ""{cols.Sample}"" = '{Esc(sample)}'{qFilter}
                GROUP BY ""{cols.Peptide}"", ""{cols.PrecursorCharge}"", ""{cols.PrecursorMz}""
            )
            WHERE mz IS NOT NULL AND rt0 IS NOT NULL AND rt1 IS NOT NULL
              AND isfinite(mz) AND isfinite(rt0) AND isfinite(rt1) AND rt1 >= rt0");

        using var reader = cmd.ExecuteReader();
        var result = new List<DetectedPrecursor>();
        while (reader.Read())
            result.Add(new DetectedPrecursor(reader.GetDouble(0), reader.GetDouble(1), reader.GetDouble(2)));
        return result;
    }

    /// <summary>
    /// Bin precursors on the acquisition's REAL isolation windows: each precursor is credited to every
    /// window containing its m/z (more than one only for a staggered/overlapping scheme, where it really
    /// was fragmented twice), and each cell holds the co-elution defined on
    /// <see cref="PrecursorDensityMap"/>. This is the honest version of the map - a cell is a spectrum,
    /// at the m/z boundaries the instrument actually used.
    /// </summary>
    /// <remarks>
    /// <para>Precursors outside every window are counted in
    /// <see cref="PrecursorDensityMap.PrecursorsOutsideRows"/> rather than forced into the nearest row:
    /// a large count there means the scheme does not match the data (usually the wrong scheme picked for
    /// a "Results only" document), and silently clamping would hide exactly that.</para>
    /// <para>A precursor with a non-finite m/z or boundary, or a peak that ends before it starts, is
    /// dropped before anything looks at it - see <see cref="IsValid"/>. It counts as neither placed nor
    /// outside.</para>
    /// </remarks>
    public static PrecursorDensityMap Bin(
        IReadOnlyList<DetectedPrecursor> precursors, IsolationScheme scheme,
        double rtBinMin = DefaultRtBinMin)
    {
        if (!(rtBinMin > 0))
            throw new ArgumentOutOfRangeException(nameof(rtBinMin), rtBinMin, "RT bin must be greater than 0.");
        if (!scheme.HasWindows)
            throw new ArgumentException($"Isolation scheme '{scheme.Name}' defines no windows.", nameof(scheme));
        precursors = Valid(precursors);
        if (precursors.Count == 0)
            return new PrecursorDensityMap(scheme.Windows, 0, rtBinMin, new int[0, 0], scheme.Name);

        var (rtLo, nRt, rtBin) = RtGrid(precursors, rtBinMin, scheme, scheme.Windows.Count);
        var counts = new int[scheme.Windows.Count, nRt];
        var peaks = new List<(double Start, double Stop)>?[scheme.Windows.Count];
        var outside = 0;
        foreach (var p in precursors)
        {
            var matched = false;
            // Covers(), not Contains(): for a scheduled (dynamic DIA) window the peak must also fall
            // inside the interval the window was firing, or a precursor would be credited to a same-m/z
            // window that fired in a different RT segment.
            foreach (var row in scheme.IndicesCovering(p.Mz, p.RtStart, p.RtStop))
            {
                matched = true;
                var window = scheme.Windows[row];

                // Clipped to the stretch this window was actually firing, so an instant when the window
                // was off cannot add to the concurrency of a column whose center it was on for. Covers()
                // has already guaranteed the clip is not empty; an unscheduled window's NaN bounds mean
                // always on.
                var (start, stop) = window.IsScheduled
                    ? (Math.Max(p.RtStart, window.RtStart), Math.Min(p.RtStop, window.RtStop))
                    : (p.RtStart, p.RtStop);
                (peaks[row] ??= new List<(double, double)>()).Add((start, stop));
            }
            if (!matched)
                outside++;
        }
        for (var row = 0; row < scheme.Windows.Count; row++)
            FillRowByConcurrency(counts, row, peaks[row], scheme.Windows[row], rtLo, rtBin);
        return new PrecursorDensityMap(
            scheme.Windows, rtLo, rtBin, counts, scheme.Name, outside, RtBinRequested: rtBinMin);
    }

    /// <summary>
    /// Bin precursors on a uniform m/z grid. This is the FALLBACK for when the acquisition's real windows
    /// are unknown: the cell edges are arbitrary, so a cell only approximates a spectrum. Prefer the
    /// <see cref="IsolationScheme"/> overload, and label the plot honestly when using this one. Invalid
    /// precursors are dropped exactly as there (<see cref="IsValid"/>).
    /// </summary>
    public static PrecursorDensityMap Bin(
        IReadOnlyList<DetectedPrecursor> precursors,
        double mzBinTh = DefaultMzBinTh, double rtBinMin = DefaultRtBinMin)
    {
        if (!(mzBinTh > 0))
            throw new ArgumentOutOfRangeException(nameof(mzBinTh), mzBinTh, "m/z bin must be greater than 0.");
        if (!(rtBinMin > 0))
            throw new ArgumentOutOfRangeException(nameof(rtBinMin), rtBinMin, "RT bin must be greater than 0.");
        precursors = Valid(precursors);
        if (precursors.Count == 0)
            return new PrecursorDensityMap(
                Array.Empty<IsolationWindow>(), 0, rtBinMin, new int[0, 0], UniformSource(mzBinTh),
                RowsAreWindows: false);

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

        var (rtLo, nRt, rtBin) = RtGrid(precursors, rtBinMin, nMz: nMz);
        var counts = new int[nMz, nRt];
        var peaks = new List<(double Start, double Stop)>?[nMz];
        foreach (var p in precursors)
        {
            var row = Math.Clamp((int)((p.Mz - mzLo) / mzBinTh), 0, nMz - 1);
            (peaks[row] ??= new List<(double, double)>()).Add((p.RtStart, p.RtStop));
        }
        for (var row = 0; row < nMz; row++)
            FillRowByConcurrency(counts, row, peaks[row], rows[row], rtLo, rtBin);
        return new PrecursorDensityMap(
            rows, rtLo, rtBin, counts, UniformSource(mzBinTh), RowsAreWindows: false,
            RtBinRequested: rtBinMin);
    }

    /// <summary>
    /// The precursors that can be placed at all: finite m/z and boundaries, and a peak that does not end
    /// before it starts. <see cref="Load"/> filters the same way in SQL; this is for the public Bin, which
    /// takes whatever it is given. Nothing downstream defends against these separately, and each one
    /// fails quietly rather than loudly: a NaN start saturates the column cast to 0 and paints
    /// concurrency from the axis start, an infinite stop makes the whole axis one column, and an inverted
    /// peak closes before it opens and subtracts from its neighbors' concurrency.
    /// </summary>
    private static bool IsValid(DetectedPrecursor p) =>
        double.IsFinite(p.Mz) && double.IsFinite(p.RtStart) && double.IsFinite(p.RtStop)
        && p.RtStop >= p.RtStart;

    /// <summary>The valid precursors (<see cref="IsValid"/>), the input itself when all of them are.</summary>
    private static IReadOnlyList<DetectedPrecursor> Valid(IReadOnlyList<DetectedPrecursor> precursors) =>
        precursors.All(IsValid) ? precursors : precursors.Where(IsValid).ToList();

    /// <summary>
    /// Fill one row with the cell defined on <see cref="PrecursorDensityMap"/>: the greatest number of
    /// precursors co-eluting at any one instant inside each column, found by sweeping the peak
    /// boundaries. Only columns the window was acquiring are written - the same
    /// <see cref="ColumnAcquired"/> rule every view reads with - so a count can never sit in a cell
    /// drawn as a gap.
    /// </summary>
    /// <remarks>
    /// <para>Sweeping the sorted starts and stops gives the exact concurrency between consecutive
    /// boundaries, and each such segment is written to every column it touches as a max. A peak is a
    /// CLOSED interval, and three details follow from that. Opens sort before closes at equal times, so
    /// two peaks meeting at one instant are concurrent there. The column containing a segment's end is
    /// credited (Floor + 1, not Ceiling), because a spectrum acquired exactly at a stop is still inside
    /// the peak. And a zero-length segment - two peaks meeting, or a zero-width peak - still writes the
    /// one column holding its instant, wherever that falls, which is what makes the busiest cell
    /// independent of how the columns are cut: every live instant reaches some column.</para>
    /// <para>The instant at the very top of the axis belongs to the last column, which the clamp on
    /// <c>from</c> provides; without it a zero-width peak at the final boundary had no column at all.</para>
    /// </remarks>
    private static void FillRowByConcurrency(
        int[,] counts, int row, List<(double Start, double Stop)>? peaks, IsolationWindow window,
        double rtLo, double rtBin)
    {
        if (peaks is null || peaks.Count == 0)
            return;

        var nRt = counts.GetLength(1);
        var events = new List<(double Time, int Delta)>(peaks.Count * 2);
        foreach (var (start, stop) in peaks)
        {
            events.Add((start, 1));
            events.Add((stop, -1));
        }
        // Opens before closes at equal times, which is what makes touching peaks concurrent.
        events.Sort((a, b) => a.Time != b.Time ? a.Time.CompareTo(b.Time) : b.Delta.CompareTo(a.Delta));

        var live = 0;
        for (var i = 0; i < events.Count - 1; i++)
        {
            live += events[i].Delta;
            if (live <= 0)
                continue;

            var from = Math.Clamp((int)Math.Floor((events[i].Time - rtLo) / rtBin), 0, nRt - 1);
            var to = Math.Min(nRt, (int)Math.Floor((events[i + 1].Time - rtLo) / rtBin) + 1);
            for (var j = from; j < to; j++)
                if (counts[row, j] < live && ColumnAcquired(window, rtLo, rtBin, j))
                    counts[row, j] = live;
        }
    }

    /// <summary>
    /// Whether <paramref name="window"/> was firing during column <paramref name="j"/> of a grid that
    /// starts at <paramref name="rtLo"/> with columns <paramref name="rtBin"/> minutes wide: on at the
    /// column's center. This is the ONE definition of "this cell is a spectrum". The fill writes only
    /// where it is true, and <see cref="PrecursorDensityMap"/>'s views count and draw only where it is
    /// true, so the two cannot disagree. Always true for an unscheduled window.
    /// </summary>
    internal static bool ColumnAcquired(IsolationWindow window, double rtLo, double rtBin, int j) =>
        window.IsOnAt(rtLo + (j + 0.5) * rtBin);

    /// <summary>Label that marks a map as approximate, so it can never be mistaken for real windows.</summary>
    public static string UniformSource(double mzBinTh) =>
        $"uniform {mzBinTh.ToString("0.###", CultureInfo.InvariantCulture)} Th bins (approximate)";

    private static (double RtLow, int Bins, double BinSize) RtGrid(
        IReadOnlyList<DetectedPrecursor> precursors, double rtBinMin, IsolationScheme? scheme = null,
        int nMz = 1)
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
        // Both bounds, and the cell budget is the one that usually bites: a map with many
        // isolation windows can afford fewer RT bins than one with few, and the axis cap alone cannot
        // know that. Reported back in the map, so the plot never claims a bin it did not use.
        var maxBins = (int)Math.Max(1, Math.Min(MaxRtBins, MaxCells / Math.Max(1, nMz)));
        var bin = Math.Max(rtBinMin, (rtHi - rtLo) / maxBins);
        return (rtLo, Math.Max(1, (int)Math.Ceiling((rtHi - rtLo) / bin)), bin);
    }

    private static string Num(double v) => v.ToString("R", CultureInfo.InvariantCulture);

    private static string Esc(string s) => s.Replace("'", "''");
}
