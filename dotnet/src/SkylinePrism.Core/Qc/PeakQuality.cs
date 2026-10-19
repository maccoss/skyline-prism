using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// Skyline's own peak-quality columns, scored per precursor and run and summarized per run in
/// acquisition order, so that a run which is degraded - or simply different - stands out across a
/// cohort. The QC report's "Peak Quality by Acquisition Order" section.
/// </summary>
/// <remarks>
/// <para><b>The score.</b> For each precursor and each metric, every run's value is turned into a
/// robust z against that precursor's OWN median across runs, scaled by its median absolute
/// deviation (MAD / 0.6745, so a z of 1 is one standard deviation for normal data). Measuring each
/// precursor against itself is what makes the metrics comparable across precursors: a shape
/// correlation of 0.7 is excellent for one peptide and poor for another. The z is one-sided where
/// only one direction is bad - a LOW shape correlation or isotope dot product counts, a high one is
/// not penalized - and two-sided for retention time and FWHM, where a peak eluting early or late, or
/// wider or narrower, differs either way. A precursor's score in a run is the root mean square of
/// those z's, over the metrics it has a value for. Each run is then summarized by the median and
/// 10-90% range of its precursors' scores.</para>
///
/// <para><b>What is left out rather than counted as bad.</b> A metric needs <see cref="MinRuns"/>
/// finite values and a nonzero MAD to define a z for a precursor. A zero MAD is common and is not
/// an error: on a 96-run cohort 40% of precursors had the SAME isotope dot product in most runs -
/// usually exactly 0, no MS1 isotope match - and dividing by that spread would turn any deviation
/// into an infinite score. A missing value in one run is likewise left out of that run's score
/// rather than imputed as a bad one, so the score describes the peaks that were measured.</para>
///
/// <para><b>Why not MSstats+'s anomaly score.</b> MSstats+ (Kohler et al. 2026) scores the same
/// Skyline columns with an isolation forest, which is random; PRISM pins every quantity it reports
/// to a golden, and a random forest cannot be pinned. This is a deterministic score of the same
/// columns, written from the description above rather than translated from MSstats+'s code. On the
/// Verapamil cohort it found the same two runs - the Internal Control injections - that MSstats+
/// found.</para>
///
/// <para><b>Why there is no CUSUM.</b> MSstats+ also feeds the model cumulative sums of each
/// metric in run order. On Verapamil that carried one outlying run's deviation into about ten
/// following runs, which then scored high for a defect they did not have. The plots here are of
/// each run's own metrics only.</para>
///
/// <para><b>Memory.</b> The per-run summaries are quantiles over every precursor, and keeping every
/// precursor x run value would cost memory in proportion to both - gigabytes on a large cohort, on
/// top of the matrices the rest of the report holds. Each run's values are instead counted into
/// fixed-width histograms (<see cref="RunHistograms"/>), about 80 KB per run whatever the number of
/// precursors, and the quantiles read back from them agree with the exact ones to within one bin
/// width - far below what the plots can show.</para>
/// </remarks>
public static class PeakQuality
{
    /// <summary>
    /// Phi^-1(3/4): a normal distribution's MAD is this many standard deviations, so dividing a MAD
    /// by it gives a standard deviation. The value <c>scipy.stats.median_abs_deviation(scale='normal')</c>
    /// divides by. (R's <c>mad()</c> multiplies by 1.4826, its rounded reciprocal, which differs in
    /// the seventh digit.)
    /// </summary>
    public const double NormalQuartile = 0.6744897501960817;

    /// <summary>Fewest runs with a finite value for a precursor's metric to be scored at all.</summary>
    public const int MinRuns = 3;

    /// <summary>A run is labeled when its median score exceeds this multiple of the cohort's.</summary>
    public const double FlagMultiple = 2.0;

    /// <summary>A precursor's center and spread for one metric, over the runs that have a value.</summary>
    /// <param name="Median">Median of the finite values; NaN when there are none.</param>
    /// <param name="Scale">MAD / <see cref="NormalQuartile"/>; NaN when there are no values.</param>
    /// <param name="Count">How many runs had a finite value.</param>
    public readonly record struct Center(double Median, double Scale, int Count)
    {
        /// <summary>Enough runs for the median to be a center worth measuring a deviation from.</summary>
        public bool HasMedian => Count >= MinRuns;

        /// <summary>A center AND a nonzero spread, so a z is defined.</summary>
        public bool HasScale => HasMedian && Scale > 0 && double.IsFinite(Scale);
    }

    /// <summary>
    /// One precursor's z per metric and its score, one entry per run. NaN wherever the run has no
    /// value or the metric has no <see cref="Center.HasScale"/> for this precursor.
    /// </summary>
    public sealed record PrecursorScores(
        Center Shape, Center DotProduct, Center RetentionTime, Center Fwhm,
        double[] ShapeZ, double[] DotProductZ, double[] RetentionTimeZ, double[] FwhmZ,
        double[] Score);

    /// <summary>Median and MAD-based scale of the finite values.</summary>
    public static Center RobustCenter(ReadOnlySpan<double> values)
    {
        // One buffer, sorted twice in place: once for the median, then again as the deviations from it.
        var buffer = new double[values.Length];
        var n = 0;
        foreach (var v in values)
            if (double.IsFinite(v))
                buffer[n++] = v;
        if (n == 0)
            return new Center(double.NaN, double.NaN, 0);

        var finite = buffer.AsSpan(0, n);
        finite.Sort();
        var median = MedianOfSorted(finite);
        for (var i = 0; i < n; i++)
            finite[i] = Math.Abs(finite[i] - median);
        finite.Sort();
        return new Center(median, MedianOfSorted(finite) / NormalQuartile, n);
    }

    // numpy.median's convention: the middle value, or the mean of the two middle values as (a + b) / 2.
    private static double MedianOfSorted(ReadOnlySpan<double> sorted)
    {
        var mid = sorted.Length / 2;
        return (sorted.Length & 1) == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2.0;
    }

    /// <summary>
    /// Score one precursor across runs. The four spans are the precursor's value in each run, all the
    /// same length and indexed the same way; NaN marks a run without one.
    /// </summary>
    public static PrecursorScores ScorePrecursor(
        ReadOnlySpan<double> shapeCorrelation, ReadOnlySpan<double> isotopeDotProduct,
        ReadOnlySpan<double> retentionTime, ReadOnlySpan<double> fwhm)
    {
        var n = shapeCorrelation.Length;
        if (isotopeDotProduct.Length != n || retentionTime.Length != n || fwhm.Length != n)
            throw new ArgumentException("Every metric must have one value per run.");

        var shape = RobustCenter(shapeCorrelation);
        var dotp = RobustCenter(isotopeDotProduct);
        var rt = RobustCenter(retentionTime);
        var width = RobustCenter(fwhm);

        var shapeZ = Low(shapeCorrelation, shape);
        var dotpZ = Low(isotopeDotProduct, dotp);
        var rtZ = TwoSided(retentionTime, rt);
        var fwhmZ = TwoSided(fwhm, width);

        var score = new double[n];
        for (var r = 0; r < n; r++)
        {
            double sum = 0;
            var k = 0;
            // Summed in this order - shape, dot product, RT, FWHM - which the golden repeats.
            AddSquare(shapeZ[r], ref sum, ref k);
            AddSquare(dotpZ[r], ref sum, ref k);
            AddSquare(rtZ[r], ref sum, ref k);
            AddSquare(fwhmZ[r], ref sum, ref k);
            score[r] = k > 0 ? Math.Sqrt(sum / k) : double.NaN;
        }

        return new PrecursorScores(shape, dotp, rt, width, shapeZ, dotpZ, rtZ, fwhmZ, score);
    }

    private static void AddSquare(double z, ref double sum, ref int count)
    {
        if (!double.IsFinite(z))
            return;
        sum += z * z;
        count++;
    }

    // Bad only when LOW: a shape correlation or dot product above the precursor's median scores 0.
    private static double[] Low(ReadOnlySpan<double> values, Center c)
    {
        var z = new double[values.Length];
        for (var i = 0; i < values.Length; i++)
            z[i] = c.HasScale && double.IsFinite(values[i])
                ? Math.Max(0, (c.Median - values[i]) / c.Scale)
                : double.NaN;
        return z;
    }

    // Different either way: eluting early or late, wider or narrower.
    private static double[] TwoSided(ReadOnlySpan<double> values, Center c)
    {
        var z = new double[values.Length];
        for (var i = 0; i < values.Length; i++)
            z[i] = c.HasScale && double.IsFinite(values[i])
                ? Math.Abs(values[i] - c.Median) / c.Scale
                : double.NaN;
        return z;
    }

    /// <summary>
    /// Measure the cohort in an output directory from its <c>merged_data/</c>. Null, with the reason
    /// logged, when there is no merged data or it has none of the four metrics.
    /// </summary>
    /// <param name="overrides">The run's <c>data:</c> column overrides, so a renamed column is found.</param>
    public static PeakQualityResult? Measure(
        string outputDir, ColumnOverrides? overrides = null, Action<string>? log = null,
        int memoryBudgetMb = 0)
    {
        var root = MergedDataset.Locate(outputDir);
        if (root is null)
        {
            log?.Invoke($"  Peak quality skipped: no {MergedDataset.DirectoryName}/ or "
                + $"{MergedDataset.LegacyFileName} in the output directory.");
            return null;
        }

        var timer = Stopwatch.StartNew();
        var dataset = MergedDataset.Open(root);
        var available = ParquetTable.ReadColumnNames(dataset.RepresentativeFile()).ToHashSet();
        var cols = SkylineColumns.Detect(available, overrides);
        var qcols = PrecursorQualityColumns.From(cols, available);
        var metrics = new (string? Column, string Name)[]
        {
            (qcols.ShapeCorrelation, "shape correlation"),
            (qcols.IsotopeDotProduct, "isotope dot product"),
            (qcols.RetentionTime, "retention time"),
            (qcols.Fwhm, "FWHM"),
        }.Where(m => m.Column is not null).Select(m => m.Name).ToList();
        if (metrics.Count == 0)
        {
            log?.Invoke("  Peak quality skipped: the merged data has none of Shape Correlation, "
                + "Isotope Dot Product, Retention Time or Fwhm. Add them to the transition report "
                + "to get this section.");
            return null;
        }

        var samples = MergedParquetReader.GetSortedSamples(dataset, cols.Sample, memoryBudgetMb);
        if (samples.Count == 0)
            return null;
        var times = cols.AcquiredTime is not null
            ? AcquisitionOrder.ReadTimes(dataset, cols.Sample, cols.AcquiredTime)
            : new Dictionary<string, DateTime>(StringComparer.Ordinal);

        var accumulator = new Accumulator(samples.Count);
        foreach (var block in MergedParquetReader.StreamPrecursorQuality(dataset, qcols, samples, memoryBudgetMb))
            accumulator.Add(block);

        var result = accumulator.Summarize(samples, times, metrics);
        log?.Invoke($"  Peak quality: scored {result.Precursors:N0} precursors across "
            + $"{result.Runs.Count:N0} runs on {string.Join(", ", metrics)} "
            + $"({timer.Elapsed.TotalSeconds:0.0} s).");
        return result;
    }

    /// <summary>
    /// Fixed-width histograms, one per run, from which quantiles are read back to within one bin
    /// width. Values outside the range are counted at its nearest edge, and read back as that edge.
    /// </summary>
    internal sealed class RunHistograms
    {
        private readonly int[] _counts;
        private readonly int[] _totals;
        private readonly int _bins;

        public RunHistograms(int runs, double lo, double hi, double width)
        {
            Lo = lo;
            Hi = hi;
            Width = width;
            _bins = (int)Math.Round((hi - lo) / width);
            // Two more per run: one below the range, one above it.
            _counts = new int[runs * (_bins + 2)];
            _totals = new int[runs];
        }

        public double Lo { get; }
        public double Hi { get; }

        /// <summary>The bin width: the bound on how far a quantile read back can be from the exact one.</summary>
        public double Width { get; }

        public void Add(int run, double value)
        {
            if (!double.IsFinite(value))
                return;
            int slot;
            if (value < Lo)
                slot = 0;
            else if (value > Hi)
                slot = _bins + 1;
            else
                // A value exactly at Hi belongs to the last bin, not the overflow slot.
                slot = 1 + Math.Min((int)((value - Lo) / Width), _bins - 1);
            _counts[run * (_bins + 2) + slot]++;
            _totals[run]++;
        }

        public int Count(int run) => _totals[run];

        /// <summary>
        /// The q-quantile of a run's values, interpolated the way <c>numpy.percentile(method='linear')</c>
        /// interpolates between order statistics. Each order statistic is placed inside its own bin,
        /// so it is within one bin width of the true value, and so is any interpolation between two of
        /// them. NaN for a run with no values.
        /// </summary>
        public double Quantile(int run, double q)
        {
            var n = _totals[run];
            if (n == 0)
                return double.NaN;
            var h = q * (n - 1);
            var k = (int)Math.Floor(h);
            var lower = OrderStatistic(run, k);
            if (k + 1 >= n || h == k)
                return lower;
            return lower + (h - k) * (OrderStatistic(run, k + 1) - lower);
        }

        /// <summary>
        /// Estimate of the j-th smallest value (0-based): the bin holding it, with the bin's values
        /// spread evenly across it.
        /// </summary>
        private double OrderStatistic(int run, int j)
        {
            var offset = run * (_bins + 2);
            var below = 0;
            for (var slot = 0; slot < _bins + 2; slot++)
            {
                var c = _counts[offset + slot];
                if (j < below + c)
                {
                    if (slot == 0)
                        return Lo;
                    if (slot == _bins + 1)
                        return Hi;
                    return Lo + Width * (slot - 1 + (j - below + 0.5) / c);
                }
                below += c;
            }
            return Hi;
        }
    }

    /// <summary>
    /// Folds precursors into per-run histograms one at a time, so nothing larger than one precursor's
    /// runs is held beyond them.
    /// </summary>
    internal sealed class Accumulator
    {
        // Ranges and widths per quantity. Wide enough that a real run's median sits well inside them;
        // fine enough that one bin is invisible at the scale each panel is drawn at.
        internal const double ScoreMax = 40, ScoreWidth = 0.01;
        internal const double ShapeWidth = 0.001;
        internal const double DotProductWidth = 0.0005;
        internal const double RtDeviationMax = 4, RtDeviationWidth = 0.001;
        internal const double FwhmRatioMax = 4, FwhmRatioWidth = 0.001;

        private readonly int _runs;
        private readonly int[] _scored;
        private int _precursors;

        public Accumulator(int runs)
        {
            _runs = runs;
            _scored = new int[runs];
            Score = new RunHistograms(runs, 0, ScoreMax, ScoreWidth);
            Shape = new RunHistograms(runs, -1, 1, ShapeWidth);
            DotProduct = new RunHistograms(runs, 0, 1, DotProductWidth);
            RtDeviation = new RunHistograms(runs, -RtDeviationMax, RtDeviationMax, RtDeviationWidth);
            FwhmRatio = new RunHistograms(runs, 0, FwhmRatioMax, FwhmRatioWidth);
        }

        public RunHistograms Score { get; }
        public RunHistograms Shape { get; }
        public RunHistograms DotProduct { get; }
        public RunHistograms RtDeviation { get; }
        public RunHistograms FwhmRatio { get; }

        public void Add(PrecursorQualityBlock block) => Add(
            block.ShapeCorrelation, block.IsotopeDotProduct, block.RetentionTime, block.Fwhm);

        public void Add(
            double[] shapeCorrelation, double[] isotopeDotProduct, double[] retentionTime, double[] fwhm)
        {
            var s = ScorePrecursor(shapeCorrelation, isotopeDotProduct, retentionTime, fwhm);
            var any = false;
            for (var r = 0; r < _runs; r++)
            {
                if (double.IsFinite(s.Score[r]))
                {
                    Score.Add(r, s.Score[r]);
                    _scored[r]++;
                    any = true;
                }
                // The raw values are each run's own, so a precursor seen in too few runs to be scored
                // still has them. The two relative quantities need the precursor's median to mean
                // anything, which takes MinRuns runs.
                Shape.Add(r, shapeCorrelation[r]);
                DotProduct.Add(r, isotopeDotProduct[r]);
                if (s.RetentionTime.HasMedian)
                    RtDeviation.Add(r, retentionTime[r] - s.RetentionTime.Median);
                if (s.Fwhm.HasMedian && s.Fwhm.Median > 0)
                    FwhmRatio.Add(r, fwhm[r] / s.Fwhm.Median);
            }
            if (any)
                _precursors++;
        }

        /// <summary>
        /// One row per run, in acquisition order when every run has an acquired time and in sample
        /// order otherwise - a partial ordering would draw untimed runs at positions that mean nothing.
        /// </summary>
        public PeakQualityResult Summarize(
            IReadOnlyList<string> samples, IReadOnlyDictionary<string, DateTime> times,
            IReadOnlyList<string> metrics)
        {
            var timed = samples.Count(times.ContainsKey);
            var byTime = timed == samples.Count && samples.Count > 0;
            int[] order;
            if (byTime)
            {
                var rank = AcquisitionOrder.Rank(samples.ToDictionary(s => s, s => times[s], StringComparer.Ordinal));
                order = Enumerable.Range(0, samples.Count).OrderBy(i => rank[samples[i]]).ToArray();
            }
            else
            {
                order = Enumerable.Range(0, samples.Count)
                    .OrderBy(i => samples[i], StringComparer.Ordinal).ToArray();
            }

            var runs = new List<PeakQualityRun>(samples.Count);
            for (var pos = 0; pos < order.Length; pos++)
            {
                var r = order[pos];
                runs.Add(new PeakQualityRun(
                    samples[r], pos + 1,
                    times.TryGetValue(samples[r], out var t) ? t : null,
                    _scored[r],
                    Score.Quantile(r, 0.5), Score.Quantile(r, 0.1), Score.Quantile(r, 0.9),
                    Shape.Quantile(r, 0.5), DotProduct.Quantile(r, 0.5),
                    RtDeviation.Quantile(r, 0.5), FwhmRatio.Quantile(r, 0.5)));
            }

            var medians = runs.Select(r => r.ScoreMedian).Where(double.IsFinite).ToArray();
            var cohort = medians.Length > 0 ? Stats.NanMedian(medians) : double.NaN;
            return new PeakQualityResult(runs, _precursors, cohort, byTime, timed, metrics);
        }
    }
}

/// <summary>One run's peak quality: its score distribution and the median of each raw metric.</summary>
/// <param name="SampleId">The merged data's sample id (<c>replicate__@__document</c>).</param>
/// <param name="RunOrder">1-based position on the x axis: acquisition order, or sample order.</param>
/// <param name="ScoredPrecursors">Precursors with a score in this run.</param>
/// <param name="ScoreMedian">Median of the precursors' scores in this run (likewise P10, P90).</param>
/// <param name="ShapeCorrelation">Median over precursors of the precursor's shape correlation.</param>
/// <param name="IsotopeDotProduct">Median over precursors of the isotope dot product.</param>
/// <param name="RtDeviation">Median over precursors of RT minus the precursor's median RT, minutes.</param>
/// <param name="FwhmRatio">Median over precursors of FWHM over the precursor's median FWHM.</param>
public sealed record PeakQualityRun(
    string SampleId, int RunOrder, DateTime? AcquiredTime, int ScoredPrecursors,
    double ScoreMedian, double ScoreP10, double ScoreP90,
    double ShapeCorrelation, double IsotopeDotProduct, double RtDeviation, double FwhmRatio);

/// <summary>A cohort's peak quality, one row per run.</summary>
/// <param name="Runs">In <see cref="PeakQualityRun.RunOrder"/> order.</param>
/// <param name="Precursors">Precursors scored in at least one run.</param>
/// <param name="CohortMedian">Median over runs of each run's median score.</param>
/// <param name="OrderedByAcquiredTime">False when any run lacked an acquired time.</param>
/// <param name="TimedRuns">How many runs had an acquired time.</param>
/// <param name="Metrics">The metrics the merged data had, in the order they enter the score.</param>
public sealed record PeakQualityResult(
    IReadOnlyList<PeakQualityRun> Runs, int Precursors, double CohortMedian,
    bool OrderedByAcquiredTime, int TimedRuns, IReadOnlyList<string> Metrics)
{
    /// <summary>The median score above which a run is labeled.</summary>
    public double FlagThreshold => PeakQuality.FlagMultiple * CohortMedian;

    /// <summary>Runs whose median score exceeds <see cref="FlagThreshold"/>, in run order.</summary>
    public IReadOnlyList<PeakQualityRun> Flagged =>
        Runs.Where(r => r.ScoreMedian > FlagThreshold).ToList();
}
