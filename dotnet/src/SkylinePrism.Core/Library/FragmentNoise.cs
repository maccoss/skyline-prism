using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using SkylinePrism.Core.Numerics;
using SkylinePrism.Core.Rollup;

namespace SkylinePrism.Core.Library;

/// <summary>A set of injections the fragment noise is measured within, as indices into a sample list.</summary>
public sealed record NoiseGroup(string Name, IReadOnlyList<int> SampleIndices);

/// <summary>Binning choices for <see cref="FragmentNoise.Estimate"/>.</summary>
/// <param name="FineWidthLog10">Width of the fine bins cells are first counted into, in log10 units.</param>
/// <param name="TargetBins">How many points each curve aims for.</param>
/// <param name="MinCellsPerBin">Fewest cells a point of the noise curve may rest on.</param>
/// <param name="MinSignalPointsPerBin">Fewest peptides a point of the expected-CV curve may rest on.</param>
public sealed record FragmentNoiseOptions(
    double FineWidthLog10, int TargetBins, int MinCellsPerBin, int MinSignalPointsPerBin)
{
    public static FragmentNoiseOptions Default { get; } = new(0.05, 40, 200, 25);
}

/// <summary>
/// How much a fragment's area varies from injection to injection at a given expected area, measured
/// on the cohort itself - the yardstick library assist's interference removal is gated on - plus, from
/// replicate pools, how precise a sum of fragments is at a given total signal.
/// <para>
/// Two curves, both LINEAR-scale areas:
/// </para>
/// <list type="bullet">
///   <item><b>Noise</b>: expected fragment area -> SD of that fragment's area, after removing the
///   factor every fragment of a precursor shares in an injection (loading, ionization). Interpolated
///   on log-log axes, clamped at both ends (<see cref="SdAt"/>).</item>
///   <item><b>Expected CV</b>, only when measured on replicate pools: total fragment signal -> the
///   median CV (%) of a plain sum of fragments at that total. This is what an estimate built on that
///   much measured signal can be expected to achieve (<see cref="ExpectedCvAt"/>).</item>
/// </list>
/// <para>
/// A cell whose noise model was tried and dropped: an additive-plus-counting noise model with Skyline's
/// clip at zero could not reproduce the zero areas - on FLARE-Extended they rise far more steeply at
/// low intensity than Gaussian noise allows, behaving like a detection threshold. So the curves are
/// used as measured, never as a fitted formula.
/// </para>
/// </summary>
public sealed class FragmentNoiseCurve
{
    /// <summary>Measured within reference/QC pool injections - the cohort's replicate material.</summary>
    public const string PoolReplicates = "pool_replicates";

    /// <summary>
    /// Measured across every sample, because no pool had 3 injections. Biological differences in how
    /// a precursor's fragments vary relative to each other then count as noise, so the curve is wider
    /// and the gate removes less.
    /// </summary>
    public const string AllSamples = "all_samples";

    /// <summary>Not measured: fewer than 3 usable samples. The gate is off.</summary>
    public const string None = "none";

    public string Source { get; init; } = None;
    public IReadOnlyList<GroupSummary> Groups { get; init; } = Array.Empty<GroupSummary>();
    public long Cells { get; init; }
    public double[] NoiseLevel { get; init; } = Array.Empty<double>();
    public double[] NoiseSd { get; init; } = Array.Empty<double>();
    public double[] SignalLevel { get; init; } = Array.Empty<double>();
    public double[] SignalCv { get; init; } = Array.Empty<double>();

    /// <summary>
    /// The <see cref="NoisePlan.Key"/> this curve was measured under - which samples were pools. Kept
    /// in the file so <see cref="Pipeline.ExistingResults"/> can rebuild the transition rollup's stage
    /// fingerprint without the metadata; not recorded in parameters.json.
    /// </summary>
    public string? PlanKey { get; set; }

    public sealed record GroupSummary(string Name, int Injections);

    public bool HasNoise => NoiseLevel.Length > 0;
    public bool HasExpectedCv => SignalLevel.Length > 0;

    /// <summary>A curve from given points (tests, and anything that already has a curve).</summary>
    public static FragmentNoiseCurve FromPoints(
        double[] noiseLevel, double[] noiseSd, double[]? signalLevel = null, double[]? signalCv = null) => new()
    {
        NoiseLevel = noiseLevel,
        NoiseSd = noiseSd,
        SignalLevel = signalLevel ?? Array.Empty<double>(),
        SignalCv = signalCv ?? Array.Empty<double>(),
    };

    /// <summary>
    /// SD of a fragment's area at expected area <paramref name="expectedArea"/>: ln(sd) linear in
    /// ln(area) between the curve's points, clamped at both ends. NaN when there is no curve.
    /// </summary>
    public double SdAt(double expectedArea)
    {
        var lv = NoiseLevel;
        var sd = NoiseSd;
        if (lv.Length == 0)
            return double.NaN;
        if (expectedArea <= lv[0])
            return sd[0];
        if (expectedArea >= lv[^1])
            return sd[^1];
        for (var i = 1; i < lv.Length; i++)
        {
            if (expectedArea <= lv[i])
            {
                var t = (Math.Log(expectedArea) - Math.Log(lv[i - 1])) / (Math.Log(lv[i]) - Math.Log(lv[i - 1]));
                return Math.Exp(Math.Log(sd[i - 1]) + t * (Math.Log(sd[i]) - Math.Log(sd[i - 1])));
            }
        }
        return sd[^1];
    }

    /// <summary>
    /// Expected CV (%) of an estimate resting on <paramref name="signal"/> of measured fragment area:
    /// cv linear in ln(signal) between the curve's points, clamped. NaN when there is no curve (no
    /// replicate pools).
    /// </summary>
    public double ExpectedCvAt(double signal)
    {
        var lv = SignalLevel;
        var cv = SignalCv;
        if (lv.Length == 0)
            return double.NaN;
        if (signal <= lv[0])
            return cv[0];
        if (signal >= lv[^1])
            return cv[^1];
        for (var i = 1; i < lv.Length; i++)
        {
            if (signal <= lv[i])
            {
                var t = (Math.Log(signal) - Math.Log(lv[i - 1])) / (Math.Log(lv[i]) - Math.Log(lv[i - 1]));
                return cv[i - 1] + t * (cv[i] - cv[i - 1]);
            }
        }
        return cv[^1];
    }

    /// <summary>The file this curve is kept in beside the outputs.</summary>
    public const string FileName = "library_assist_noise.json";

    private sealed record Doc(
        [property: JsonPropertyName("source")] string Source,
        [property: JsonPropertyName("groups")] List<GroupDoc> Groups,
        [property: JsonPropertyName("cells")] long Cells,
        [property: JsonPropertyName("noise_curve")] NoiseDoc Noise,
        [property: JsonPropertyName("expected_cv_curve")] SignalDoc Signal,
        [property: JsonPropertyName("plan_key"),
                   JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? PlanKey = null);

    private sealed record GroupDoc(
        [property: JsonPropertyName("name")] string Name,
        [property: JsonPropertyName("injections")] int Injections);

    private sealed record NoiseDoc(
        [property: JsonPropertyName("expected_area")] double[] ExpectedArea,
        [property: JsonPropertyName("sd")] double[] Sd);

    private sealed record SignalDoc(
        [property: JsonPropertyName("signal")] double[] Signal,
        [property: JsonPropertyName("cv_percent")] double[] CvPercent);

    /// <summary>Write the curve as JSON. Every value is finite by construction.</summary>
    public void Save(string path)
    {
        var doc = ToDoc(PlanKey);
        File.WriteAllText(path, JsonSerializer.Serialize(doc, new JsonSerializerOptions { WriteIndented = true }));
    }

    /// <summary>The curve as a JSON-ready object, for parameters.json (without the plan key's sample names).</summary>
    public object ToProvenance() => ToDoc(null);

    private Doc ToDoc(string? planKey) => new(
        Source,
        Groups.Select(g => new GroupDoc(g.Name, g.Injections)).ToList(),
        Cells,
        new NoiseDoc(NoiseLevel, NoiseSd),
        new SignalDoc(SignalLevel, SignalCv),
        planKey);

    /// <summary>Read a saved curve; null when the file is absent or unreadable.</summary>
    public static FragmentNoiseCurve? Load(string path)
    {
        try
        {
            if (!File.Exists(path))
                return null;
            using var stream = IO.ParquetColumnIo.OpenRead(path);
            var doc = JsonSerializer.Deserialize<Doc>(stream);
            if (doc?.Source is null)
                return null;
            var noiseLevel = doc.Noise?.ExpectedArea ?? Array.Empty<double>();
            var noiseSd = doc.Noise?.Sd ?? Array.Empty<double>();
            var signalLevel = doc.Signal?.Signal ?? Array.Empty<double>();
            var signalCv = doc.Signal?.CvPercent ?? Array.Empty<double>();
            // A curve whose two halves disagree in length cannot be interpolated; treat the file as
            // unreadable rather than index past an end later.
            if (noiseLevel.Length != noiseSd.Length || signalLevel.Length != signalCv.Length)
                return null;
            return new FragmentNoiseCurve
            {
                Source = doc.Source,
                Groups = (doc.Groups ?? new List<GroupDoc>())
                    .Select(g => new GroupSummary(g.Name, g.Injections)).ToList(),
                Cells = doc.Cells,
                NoiseLevel = noiseLevel,
                NoiseSd = noiseSd,
                SignalLevel = signalLevel,
                SignalCv = signalCv,
                PlanKey = doc.PlanKey,
            };
        }
        catch (Exception ex) when (ex is IOException or JsonException or UnauthorizedAccessException
                                       or NotSupportedException)
        {
            return null;
        }
    }
}

/// <summary>
/// Which injections the fragment noise is measured within: the reference and QC pools that have at
/// least 3 injections each, or every sample when none does.
/// </summary>
/// <param name="Groups">Each group's sample names, in run order.</param>
public sealed record NoisePlan(string Source, IReadOnlyList<(string Name, IReadOnlyList<string> Samples)> Groups)
{
    /// <summary>
    /// What the curve depends on besides the data: the source and every group's members. Part of the
    /// transition rollup's stage fingerprint, so re-typing a sample re-measures the noise.
    /// </summary>
    public string Key =>
        $"fragment_noise={Source};" + string.Join(";", Groups.Select(g => g.Name + ":" + string.Join("\u0001", g.Samples)));
}

/// <summary>Measures a <see cref="FragmentNoiseCurve"/> from transition rows.</summary>
public static class FragmentNoise
{
    /// <summary>Fewest injections a group needs: a variance needs 2, and the shared factor costs one more.</summary>
    public const int MinInjections = 3;

    /// <summary>Whether <paramref name="config"/> runs the noise-gated library fit, which measures fragment noise.</summary>
    public static bool AppliesTo(Config.PrismConfig config) =>
        config.TransitionRollup.Method?.ToLowerInvariant() is "library_assist" or "library-assisted" or "library_assisted"
        && !string.Equals(config.TransitionRollup.LibraryFittingMethod, "median_polish", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// The groups to measure within. Pool groups are kept apart because the reference and QC materials
    /// differ - pooling them would count the difference between two materials as noise.
    /// </summary>
    public static NoisePlan Plan(IReadOnlyList<string> samples, IReadOnlyDictionary<string, string> sampleTypes)
    {
        var groups = new List<(string, IReadOnlyList<string>)>();
        foreach (var type in new[] { "reference", "qc" })
        {
            var members = samples.Where(s => sampleTypes.TryGetValue(s, out var t) && t == type).ToList();
            if (members.Count >= MinInjections)
                groups.Add((type, members));
        }
        if (groups.Count > 0)
            return new NoisePlan(FragmentNoiseCurve.PoolReplicates, groups);

        var all = samples
            .Where(s => !sampleTypes.TryGetValue(s, out var t) || t is "reference" or "qc" or "experimental")
            .ToList();
        return all.Count >= MinInjections
            ? new NoisePlan(FragmentNoiseCurve.AllSamples, new[] { ("all", (IReadOnlyList<string>)all) })
            : new NoisePlan(FragmentNoiseCurve.None, Array.Empty<(string, IReadOnlyList<string>)>());
    }

    /// <summary>
    /// Measure both curves from <paramref name="blocks"/> (one per peptide; sample indices refer to a
    /// list of <paramref name="nSamples"/> samples). The definitions are those of the golden's
    /// generator, <c>dotnet/tests/fixtures/library_assist/generate.py</c>, which pins this to 1e-12.
    /// </summary>
    /// <remarks>
    /// Per group of n injections and precursor charge with m fragments (areas X, NaN and negatives as
    /// 0): f_j = T_j / mean(T) with T_j the injection's fragment total, e = mu_k * f_j the cell's
    /// expected area, and v = (X - e)^2 * n m / ((n-1)(m-1)) - the squared residual of that rank-1 fit
    /// scaled by its degrees of freedom. Cells are counted into fine log10 bins, which are pooled in
    /// order into points of at least max(MinCellsPerBin, N / TargetBins) cells; a point is the
    /// geometric-mean area and sqrt of the mean v. Memory is the number of fine bins, not the number of
    /// cells, so the all-samples fallback costs no more than the pools.
    /// </remarks>
    public static FragmentNoiseCurve Estimate(
        IEnumerable<PeptideBlock> blocks, int nSamples, IReadOnlyList<NoiseGroup> groups,
        string source, bool excludePrecursor, FragmentNoiseOptions? options = null)
    {
        options ??= FragmentNoiseOptions.Default;
        var fine = new Dictionary<long, FineBin>();
        var points = new List<(double Mean, double Cv)>();
        var pools = source == FragmentNoiseCurve.PoolReplicates;

        foreach (var block in blocks)
            Accumulate(block, nSamples, groups, pools, excludePrecursor, options.FineWidthLog10, fine, points);

        long total = 0;
        foreach (var b in fine.Values)
            total += b.N;
        var target = Math.Max(options.MinCellsPerBin, (long)Math.Ceiling((double)total / options.TargetBins));
        var coarse = new List<FineBin>();
        var acc = new FineBin();
        foreach (var key in fine.Keys.OrderBy(k => k))
        {
            var c = fine[key];
            acc = new FineBin { N = acc.N + c.N, V = acc.V + c.V, L = acc.L + c.L };
            if (acc.N >= target)
            {
                coarse.Add(acc);
                acc = new FineBin();
            }
        }
        if (acc.N > 0)
        {
            if (coarse.Count > 0)
            {
                var last = coarse[^1];
                coarse[^1] = new FineBin { N = last.N + acc.N, V = last.V + acc.V, L = last.L + acc.L };
            }
            else
            {
                coarse.Add(acc);
            }
        }

        var (signalLevel, signalCv) = SignalCurve(points, options);
        return new FragmentNoiseCurve
        {
            Source = source,
            Groups = groups.Select(g => new FragmentNoiseCurve.GroupSummary(g.Name, g.SampleIndices.Count)).ToList(),
            Cells = total,
            NoiseLevel = coarse.Select(c => Math.Exp(c.L / c.N)).ToArray(),
            NoiseSd = coarse.Select(c => Math.Sqrt(c.V / c.N)).ToArray(),
            SignalLevel = signalLevel,
            SignalCv = signalCv,
        };
    }

    private sealed class FineBin
    {
        public long N;
        public double V;
        public double L;
    }

    private static void Accumulate(
        PeptideBlock block, int nSamples, IReadOnlyList<NoiseGroup> groups, bool pools,
        bool excludePrecursor, double width, Dictionary<long, FineBin> fine, List<(double, double)> points)
    {
        // Transition ids in order of first appearance; each cell takes the FIRST non-NaN area - the
        // same matrix the rollup builds.
        var index = new Dictionary<string, int>(StringComparer.Ordinal);
        var charge = new List<int>();
        for (var i = 0; i < block.RowCount; i++)
        {
            if (excludePrecursor && block.IsPrecursor[i])
                continue;
            if (!index.ContainsKey(block.TransitionId[i]))
            {
                index[block.TransitionId[i]] = index.Count;
                charge.Add(block.PrecursorCharge[i]);
            }
        }
        var nt = index.Count;
        if (nt == 0)
            return;
        var raw = new double[nt, nSamples];
        var filled = new bool[nt, nSamples];
        for (var t = 0; t < nt; t++)
            for (var j = 0; j < nSamples; j++)
                raw[t, j] = double.NaN;
        for (var i = 0; i < block.RowCount; i++)
        {
            if (excludePrecursor && block.IsPrecursor[i])
                continue;
            var t = index[block.TransitionId[i]];
            var j = block.SampleIndex[i];
            var area = block.Area[i];
            if (!filled[t, j] && !double.IsNaN(area))
            {
                raw[t, j] = area;
                filled[t, j] = true;
            }
        }
        for (var t = 0; t < nt; t++)
            for (var j = 0; j < nSamples; j++)
                raw[t, j] = Clean(raw[t, j]);

        // Charge 0 is an unparseable charge, which the rollup never fits; its rows describe no fit.
        var charges = charge.Distinct().Where(z => z > 0).OrderBy(z => z).ToList();
        foreach (var group in groups)
        {
            var idx = group.SampleIndices;
            var n = idx.Count;
            if (n < MinInjections)
                continue;

            if (pools)
            {
                var totals = new double[n];
                for (var b = 0; b < n; b++)
                {
                    double s = 0;
                    for (var t = 0; t < nt; t++)
                        s += raw[t, idx[b]];
                    totals[b] = s;
                }
                double sum = 0;
                foreach (var x in totals)
                    sum += x;
                var mean = sum / n;
                if (mean > 0)
                {
                    double ss = 0;
                    foreach (var x in totals)
                        ss += (x - mean) * (x - mean);
                    points.Add((mean, 100.0 * Math.Sqrt(ss / (n - 1)) / mean));
                }
            }

            foreach (var z in charges)
            {
                var ks = new List<int>();
                for (var t = 0; t < nt; t++)
                    if (charge[t] == z)
                        ks.Add(t);
                var m = ks.Count;
                if (m < 2)
                    continue;
                var mu = new double[m];
                for (var a = 0; a < m; a++)
                {
                    double s = 0;
                    foreach (var j in idx)
                        s += raw[ks[a], j];
                    mu[a] = s / n;
                }
                var tj = new double[n];
                for (var b = 0; b < n; b++)
                {
                    double s = 0;
                    foreach (var k in ks)
                        s += raw[k, idx[b]];
                    tj[b] = s;
                }
                double tsum = 0;
                foreach (var x in tj)
                    tsum += x;
                var tbar = tsum / n;
                if (tbar <= 0)
                    continue;
                var factor = (double)(n * m) / ((n - 1) * (m - 1));
                for (var a = 0; a < m; a++)
                {
                    if (mu[a] <= 0)
                        continue;
                    for (var b = 0; b < n; b++)
                    {
                        var e = mu[a] * (tj[b] / tbar);
                        if (e <= 0)
                            continue;
                        var r = raw[ks[a], idx[b]] - e;
                        var key = (long)Math.Floor(Math.Log10(e) / width);
                        if (!fine.TryGetValue(key, out var bin))
                            fine[key] = bin = new FineBin();
                        bin.N += 1;
                        bin.V += r * r * factor;
                        bin.L += Math.Log(e);
                    }
                }
            }
        }
    }

    private static (double[] Level, double[] Cv) SignalCurve(List<(double Mean, double Cv)> points, FragmentNoiseOptions options)
    {
        var sorted = points.OrderBy(p => p.Mean).ThenBy(p => p.Cv).ToList();
        var p = sorted.Count;
        var nbins = Math.Min(options.TargetBins, p / options.MinSignalPointsPerBin);
        if (nbins == 0 && p >= MinInjections)
            nbins = 1;
        var level = new double[nbins];
        var cv = new double[nbins];
        if (nbins == 0)
            return (level, cv);
        var size = p / nbins;
        var extra = p % nbins;
        var start = 0;
        for (var i = 0; i < nbins; i++)
        {
            var len = size + (i < extra ? 1 : 0);
            var chunk = sorted.GetRange(start, len);
            start += len;
            level[i] = Stats.Median(chunk.Select(q => q.Mean).ToArray());
            cv[i] = Stats.Median(chunk.Select(q => q.Cv).ToArray());
        }
        return (level, cv);
    }

    private static double Clean(double v) => double.IsNaN(v) ? 0.0 : Math.Max(v, 0.0);
}
