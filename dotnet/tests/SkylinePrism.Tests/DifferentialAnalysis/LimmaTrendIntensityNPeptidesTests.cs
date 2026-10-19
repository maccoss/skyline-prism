using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.DifferentialAnalysis;
using Xunit;
using Xunit.Abstractions;

namespace SkylinePrism.Tests.DifferentialAnalysis;

/// <summary>
/// <see cref="VariancePrior.LimmaTrendIntensityNPeptides"/>, the default prior chosen by level, and
/// the protein-level check both peptide-count priors make - each as <c>proteomics-toolkit</c> does it,
/// so that one setting runs the same prior, gives the same numbers and refuses the same input in both
/// tools. The numbers themselves are pinned by
/// <see cref="DifferentialGoldenTests.LimmaTrendIntensityNPeptides_MatchesTheLimmaTranscription"/>.
/// </summary>
public class LimmaTrendIntensityNPeptidesTests
{
    private readonly ITestOutputHelper _out;

    public LimmaTrendIntensityNPeptidesTests(ITestOutputHelper output) => _out = output;

    /// <summary>
    /// A protein-level null like the toolkit's <c>_protein_calibration_null</c>: each protein a rollup
    /// of a log-uniform 1-64 peptides, technical variance falling with intensity and divided by the
    /// count (scaled to match at 8 peptides), biology that does not depend on the count, and each
    /// component a scaled inverse chi-square on 10 df. Returns the LOG2 matrix (5 vs 5), the counts,
    /// and each protein's intensity and peptide-count fifth.
    /// </summary>
    private static (double[,] Expr, double[] Counts, int[] IntensityFifth, int[] CountFifth) ProteinNull(
        int seed, int nF = 2000, int perArm = 5)
    {
        var rng = new Random(seed);
        double Normal() => Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
        double InvChi2(int df)
        {
            double s = 0;
            for (var i = 0; i < df; i++)
            {
                var z = Normal();
                s += z * z;
            }

            return df / s;
        }

        var expr = new double[nF, 2 * perArm];
        var level = new double[nF];
        var counts = new double[nF];
        for (var f = 0; f < nF; f++)
        {
            var l = 14 + 14 * rng.NextDouble();
            var n = Math.Max(1, Math.Round(Math.Pow(2, 6 * rng.NextDouble())));
            level[f] = l;
            counts[f] = n;
            var tech = (0.0049 + 0.25 * Math.Exp(-(l - 14) / 2.5)) * InvChi2(10) * 8.0 / n;
            var bio = 0.09 * InvChi2(10);
            for (var j = 0; j < 2 * perArm; j++)
                expr[f, j] = l + Math.Sqrt(bio) * Normal() + Math.Sqrt(tech) * Normal();
        }

        int[] Fifths(double[] by) =>
            Enumerable.Range(0, nF).OrderBy(f => by[f]).ThenBy(f => f)
                .Select((f, rank) => (f, rank)).OrderBy(x => x.f).Select(x => x.rank * 5 / nF).ToArray();
        return (expr, counts, Fifths(level), Fifths(counts));
    }

    private static DifferentialResult Run(double[,] expr, double[]? counts, VariancePrior? prior, string[]? ids = null,
        int perArm = 5)
    {
        ids ??= Enumerable.Range(0, expr.GetLength(0)).Select(f => "P" + f).ToArray();
        return Differential.Run(expr, ids, Enumerable.Range(0, perArm).ToArray(),
            Enumerable.Range(perArm, perArm).ToArray(),
            new DifferentialOptions { Prior = prior, PeptideCounts = counts, Correction = MultipleTesting.None });
    }

    /// <summary>
    /// Calibrated by intensity AND by peptide count, where limma-trend - which cannot see the counts -
    /// tilts by them. Six seeded draws of 2,000 proteins give 2,400 per fifth, so a 5% rate has a
    /// binomial sd of 0.44 points: the 3.5-6.5% bound is more than three sd each side, and a 2-point
    /// tilt is more than three sd of a difference between two fifths.
    /// </summary>
    [Fact]
    public void UnderAProteinLevelNull_ItIsCalibratedByIntensityAndByPeptideCount_AndLimmaTrendIsNotByCount()
    {
        var hits = new Dictionary<string, (double[] ByIntensity, double[] ByCount)>
        {
            ["new"] = (new double[5], new double[5]),
            ["limma-trend"] = (new double[5], new double[5]),
        };
        var total = new double[5];
        for (var draw = 0; draw < 6; draw++)
        {
            var (expr, counts, iq, nq) = ProteinNull(20261019 + draw);
            for (var f = 0; f < iq.Length; f++)
                total[iq[f]]++;
            foreach (var (key, prior) in new[] { ("new", (VariancePrior?)null), ("limma-trend", VariancePrior.LimmaTrend) })
            {
                var res = Run(expr, counts, prior);
                Assert.Equal(key == "new" ? "limma-trend-intensity-n-peptides" : "limma-trend", res.VariancePrior);
                foreach (var row in res.Rows.Where(r => r.PValue < 0.05))
                {
                    var f = int.Parse(row.FeatureId.AsSpan(1));
                    hits[key].ByIntensity[iq[f]]++;
                    hits[key].ByCount[nq[f]]++;
                }
            }
        }

        double[] Rate(double[] h) => h.Select((x, q) => x / total[q]).ToArray();
        foreach (var (key, (byI, byN)) in hits)
            _out.WriteLine($"{key}: by intensity {string.Join(" ", Rate(byI).Select(r => r.ToString("P1")))}; "
                + $"by count {string.Join(" ", Rate(byN).Select(r => r.ToString("P1")))}");

        Assert.All(Rate(hits["new"].ByIntensity), r => Assert.InRange(r, 0.035, 0.065));
        Assert.All(Rate(hits["new"].ByCount), r => Assert.InRange(r, 0.035, 0.065));
        var limmaByCount = Rate(hits["limma-trend"].ByCount);
        Assert.True(limmaByCount[0] - limmaByCount[4] > 0.02,
            $"limma-trend no longer tilts by peptide count ({limmaByCount[0]:P1} fewest, {limmaByCount[4]:P1} most); "
            + "the docs say it does - update them");
    }

    [Fact]
    public void TheDefaultIsChosenByLevel()
    {
        Assert.Null(DifferentialOptions.Default.Prior);
        Assert.Equal(VariancePrior.LimmaTrend, new DifferentialOptions().ResolvedPrior);
        Assert.Equal(VariancePrior.LimmaTrendIntensityNPeptides,
            new DifferentialOptions { PeptideCounts = new double[] { 1, 2 } }.ResolvedPrior);
        Assert.Equal(VariancePrior.Global,
            new DifferentialOptions { Prior = VariancePrior.Global, PeptideCounts = new double[] { 1, 2 } }.ResolvedPrior);

        var (expr, counts, _, _) = ProteinNull(7, nF: 300);
        Assert.Equal("limma-trend-intensity-n-peptides", Run(expr, counts, null).VariancePrior);
        Assert.Equal("limma-trend", Run(expr, null, null).VariancePrior); // peptide level: no counts
    }

    [Fact]
    public void TheDefaultOnTooFewProteinsUsesLimmaTrendAndSaysSo_WhileTheNamedPriorRefuses()
    {
        var (expr, counts, _, _) = ProteinNull(8, nF: 4);
        var res = Run(expr, counts, null);
        Assert.Equal("limma-trend", res.VariancePrior);
        Assert.Contains(res.Messages, m => m.Contains("too few to fit the default limma-trend-intensity-n-peptides prior",
            StringComparison.Ordinal));
        var ex = Assert.Throws<ArgumentException>(() => Run(expr, counts, VariancePrior.LimmaTrendIntensityNPeptides));
        Assert.Contains("too few to fit a prior on intensity and peptide count", ex.Message, StringComparison.Ordinal);
    }

    public static IEnumerable<object[]> CountPriors() => new[]
    {
        new object[] { VariancePrior.LimmaTrendIntensityNPeptides },
        new object[] { VariancePrior.PeptideCount },
    };

    /// <summary>
    /// Both peptide-count priors refuse what is not protein-level, with the toolkit's reasons, in its
    /// order: no counts, a missing / fractional / sub-1 count, counts without spread, repeated ids.
    /// </summary>
    [Theory]
    [MemberData(nameof(CountPriors))]
    public void TheCountPriorsRefuseInputThatIsNotProteinLevel(VariancePrior prior)
    {
        var (expr, counts, _, _) = ProteinNull(9, nF: 80);

        string Refusal(double[]? c, string[]? ids = null) =>
            Assert.Throws<ArgumentException>(() => Run(expr, c, prior, ids)).Message;

        Assert.Contains("only the protein-level matrix carries", Refusal(null), StringComparison.Ordinal);
        Assert.Contains("have no peptide count", Refusal(counts.Select((c, i) => i == 3 ? double.NaN : c).ToArray()),
            StringComparison.Ordinal);
        Assert.Contains("whole-number peptide counts of 1 or more",
            Refusal(counts.Select((c, i) => i == 3 ? 2.5 : c).ToArray()), StringComparison.Ordinal);
        Assert.Contains("whole-number peptide counts of 1 or more",
            Refusal(counts.Select((c, i) => i == 3 ? 0.0 : c).ToArray()), StringComparison.Ordinal);
        Assert.Contains("protein-level analysis only", Refusal(Enumerable.Repeat(1.0, counts.Length).ToArray()),
            StringComparison.Ordinal);
        var peptideRows = Enumerable.Range(0, counts.Length).Select(i => "P" + (i / 4)).ToArray();
        Assert.Contains("repeat a feature id", Refusal(counts, peptideRows), StringComparison.Ordinal);
    }

    [Fact]
    public void TheDefaultWithCountsThatDoNotLookProteinLevelStopsRatherThanFallBack()
    {
        var (expr, counts, _, _) = ProteinNull(10, nF: 80);
        var ex = Assert.Throws<ArgumentException>(() => Run(expr, Enumerable.Repeat(1.0, counts.Length).ToArray(), null));
        Assert.Contains("protein-level analysis only", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// The names are proteomics-toolkit's <c>MODERATIONS</c> (less its deprecated, toolkit-only
    /// intensity-peptide-trend), read back by the parser, with underscores accepted for the hyphens.
    /// </summary>
    [Fact]
    public void TheNamesAreTheToolkits_AndParseBack()
    {
        var names = Enum.GetValues<VariancePrior>().Select(DifferentialTokens.Prior).OrderBy(n => n).ToArray();
        Assert.Equal(new[] { "global", "intensity-trend", "limma-trend", "limma-trend-intensity-n-peptides",
            "peptide-count" }, names);
        foreach (var p in Enum.GetValues<VariancePrior>())
        {
            Assert.Equal(p, DifferentialTokens.ParsePrior(DifferentialTokens.Prior(p)));
            Assert.Equal(p, DifferentialTokens.ParsePrior(DifferentialTokens.Prior(p).Replace('-', '_').ToUpperInvariant()));
        }

        Assert.Null(DifferentialTokens.ParsePrior("deqms"));
        Assert.True(DifferentialTokens.UsesPeptideCounts(VariancePrior.LimmaTrendIntensityNPeptides));
        Assert.True(DifferentialTokens.UsesPeptideCounts(VariancePrior.PeptideCount));
        Assert.False(DifferentialTokens.UsesPeptideCounts(VariancePrior.LimmaTrend));
    }
}
