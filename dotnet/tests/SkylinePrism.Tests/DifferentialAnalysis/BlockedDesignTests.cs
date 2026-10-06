using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.DifferentialAnalysis;
using SkylinePrism.Core.DifferentialAnalysis.Detection;
using Xunit;
using Xunit.Abstractions;

namespace SkylinePrism.Tests.DifferentialAnalysis;

/// <summary>
/// <see cref="DifferentialDesign.BlockedBySubject"/> through <see cref="Differential.Run"/>: that the
/// public entry point reproduces limma (<c>blocked.json</c>), refuses what it cannot honor, and says
/// what it left out. The estimator itself is pinned in <see cref="BlockedGoldenTests"/>.
/// </summary>
public class BlockedDesignTests
{
    private const string File = "blocked.json";
    private readonly ITestOutputHelper _out;

    public BlockedDesignTests(ITestOutputHelper output) => _out = output;

    /// <summary>A golden case as a matrix, arms and per-column subject labels.</summary>
    private static (double[,] Expr, string[] Ids, int[] A, int[] B, string?[] Subjects) Load(string name)
    {
        var c = Golden.Case(File, name);
        var expr = Golden.Mat(c, "expr");
        var design = Golden.Mat(c, "design");
        var subjects = c.GetProperty("block").EnumerateArray().Select(e => (string?)e.GetString()).ToArray();
        var ids = Enumerable.Range(0, expr.GetLength(0)).Select(i => $"F{i:000}").ToArray();
        var a = Enumerable.Range(0, subjects.Length).Where(s => design[s, 1] == 0).ToArray();
        var b = Enumerable.Range(0, subjects.Length).Where(s => design[s, 1] == 1).ToArray();
        return (expr, ids, a, b, subjects);
    }

    private static DifferentialOptions Blocked(string?[] subjects, VariancePrior prior) => new()
    {
        Design = DifferentialDesign.BlockedBySubject,
        SubjectLabels = subjects,
        Prior = prior,
        Correction = MultipleTesting.None,
    };

    /// <summary>
    /// The whole path - design, correlation, GLS, moderation, p-values - against limma, for both
    /// priors limma has. The golden's design is exactly [1, groupB], which is what Run builds.
    /// </summary>
    [Theory]
    [InlineData(VariancePrior.Global, "global")]
    [InlineData(VariancePrior.LimmaTrend, "trend")]
    public void Run_ReproducesLimma(VariancePrior prior, string goldenPrior)
    {
        var c = Golden.Case(File, "nested_unequal_visits");
        var (expr, ids, a, b, subjects) = Load("nested_unequal_visits");

        var result = Differential.Run(expr, ids, a, b, Blocked(subjects, prior));

        Assert.True(result.IsBlocked);
        Assert.Equal(Golden.Num(c, "consensus_correlation"), result.BlockCorrelation, 12);
        Assert.Equal(15, result.NSubjects);
        Assert.Equal((7, 8), (result.SubjectsA, result.SubjectsB));
        Assert.Equal("A n = 28 (7 subjects), B n = 35 (8 subjects)", result.DescribeArms());
        Assert.Equal("intra-subject correlation 0.495 across 15 subjects", result.DescribeBlocking());

        var t = Golden.Mat(c.GetProperty(goldenPrior), "t");
        var p = Golden.Mat(c.GetProperty(goldenPrior), "p");
        var coef = Golden.Mat(c, "coefficients");
        foreach (var row in result.Rows)
        {
            var i = Array.IndexOf(ids, row.FeatureId);
            Assert.True(Math.Abs(row.LogFc - coef[i, 1]) <= 1e-10 * Math.Max(Math.Abs(coef[i, 1]), 1), row.FeatureId);
            Assert.True(Math.Abs(row.T - t[i, 1]) <= 1e-10 * Math.Max(Math.Abs(t[i, 1]), 1), row.FeatureId);
            Golden.Close(p[i, 1], row.PValue, 1e-9, $"p[{row.FeatureId}]");
        }
    }

    /// <summary>
    /// The lab's default prior also runs on the blocked fit: its shape from the design groups, its
    /// level calibrated to the GLS residual variances.
    /// </summary>
    /// <remarks>
    /// There is no external golden for this composition - limma has no intensity trend and the
    /// toolkit has no blocked fit. Each half is pinned on its own (the GLS by <c>blocked.json</c>,
    /// the calibration by <c>toolkit_end_to_end.json</c>), so this asserts only that the two are
    /// joined: the trend was fitted, not silently replaced by the global prior.
    /// </remarks>
    [Fact]
    public void IntensityTrendPrior_IsFittedOnTheBlockedResiduals()
    {
        var (expr, ids, a, b, subjects) = Load("nested_unequal_visits");

        var result = Differential.Run(expr, ids, a, b, Blocked(subjects, VariancePrior.IntensityTrend));

        Assert.Equal("intensity-trend from design groups", result.VariancePrior);
        Assert.True(double.IsFinite(result.PriorLevel));
        Assert.NotNull(result.DescribePriorFit());
        Assert.True(result.IsBlocked);
    }

    [Fact]
    public void WithoutASubjectColumn_IsRefused()
    {
        var (expr, ids, a, b, _) = Load("nested_unequal_visits");
        var options = new DifferentialOptions { Design = DifferentialDesign.BlockedBySubject };

        var ex = Assert.Throws<ArgumentException>(() => Differential.Run(expr, ids, a, b, options));
        Assert.Contains("subject column", ex.Message);
    }

    [Theory]
    [InlineData(DifferentialTest.WelchT)]
    [InlineData(DifferentialTest.MannWhitney)]
    public void ATestThatCannotUseTheCorrelation_IsRefused(DifferentialTest test)
    {
        var (expr, ids, a, b, subjects) = Load("nested_unequal_visits");
        var options = Blocked(subjects, VariancePrior.Global) with { Test = test };

        var ex = Assert.Throws<ArgumentException>(() => Differential.Run(expr, ids, a, b, options));
        Assert.Contains("moderated t", ex.Message);
    }

    [Fact]
    public void SamplesWithNoSubject_AreLeftOut_AndCounted()
    {
        var (expr, ids, a, b, subjects) = Load("nested_unequal_visits");
        subjects[a[0]] = null;
        subjects[b[0]] = " ";

        var result = Differential.Run(expr, ids, a, b, Blocked(subjects, VariancePrior.Global));

        Assert.Equal((a.Length - 1, b.Length - 1), (result.NA, result.NB));
        Assert.Contains(result.Messages, m => m.Contains("2 sample(s) have no subject"));
        var (usedA, usedB) = PairedSamples.ColumnsUsed(DifferentialDesign.BlockedBySubject, subjects, a, b);
        Assert.Equal((a.Length - 1, b.Length - 1), (usedA.Count, usedB.Count));
    }

    /// <summary>
    /// limma answers a correlation of 0 when nothing is repeated, and the fit is then exactly the
    /// unpaired one - said, not silent.
    /// </summary>
    [Fact]
    public void NoRepeatedSubject_IsTheUnpairedFit_AndSaysSo()
    {
        var (expr, ids, a, b, subjects) = Load("nested_unequal_visits");
        var own = subjects.Select((_, i) => (string?)$"S{i}").ToArray();

        var blocked = Differential.Run(expr, ids, a, b, Blocked(own, VariancePrior.Global));
        var unpaired = Differential.Run(expr, ids, a, b,
            new DifferentialOptions { Prior = VariancePrior.Global, Correction = MultipleTesting.None });

        Assert.Equal(0.0, blocked.BlockCorrelation);
        Assert.Contains(blocked.Messages, m => m.Contains("set to 0"));
        foreach (var row in blocked.Rows)
        {
            var other = unpaired.Rows.Single(r => r.FeatureId == row.FeatureId);
            Assert.Equal(other.T, row.T, 10);
        }
    }

    /// <summary>
    /// What the design is for, checked by simulation: under a null group effect with a real
    /// intra-subject correlation, the blocked test holds its nominal false-positive rate while the
    /// unpaired test - counting each subject's samples as independent - does not.
    /// </summary>
    /// <remarks>
    /// 15 subjects with 3-6 samples each, correlation 0.5, every subject in one group only (the shape
    /// of a sex contrast). The design effect <c>1 + (m - 1) rho</c> is about 2.2 here, so the unpaired
    /// test's standard error is roughly 1.5x too small. 2,000 null features put the binomial sd of a
    /// 5% rate at 0.5%, so the bounds below are several sd wide on the blocked side and the unpaired
    /// rate clears them by far more. A fixed seed keeps the outcome the same on every run.
    /// </remarks>
    [Fact]
    public void UnderTheNull_BlockedHoldsItsFalsePositiveRate_AndUnpairedDoesNot()
    {
        var visits = new[] { 3, 4, 4, 5, 4, 3, 5, 5, 4, 6, 3, 4, 5, 4, 4 };
        var subjects = visits.SelectMany((n, s) => Enumerable.Repeat((string?)$"P{s:00}", n)).ToArray();
        var inB = visits.SelectMany((n, s) => Enumerable.Repeat(s >= 7, n)).ToArray();
        var nSamples = subjects.Length;
        const int nFeatures = 2000;

        var rng = new Random(20261006);
        double Normal() => Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
        var expr = new double[nFeatures, nSamples];
        for (var f = 0; f < nFeatures; f++)
        {
            var level = 14 + 2 * rng.NextDouble();
            var subjectEffect = visits.Select(_ => Normal()).ToArray(); // var 1 between, 1 within: rho 0.5
            var s = 0;
            for (var subj = 0; subj < visits.Length; subj++)
                for (var k = 0; k < visits[subj]; k++, s++)
                    expr[f, s] = level + subjectEffect[subj] + Normal();
        }

        var ids = Enumerable.Range(0, nFeatures).Select(i => $"F{i}").ToArray();
        var a = Enumerable.Range(0, nSamples).Where(i => !inB[i]).ToArray();
        var b = Enumerable.Range(0, nSamples).Where(i => inB[i]).ToArray();
        double Rate(DifferentialResult r) => r.Rows.Count(x => x.PValue < 0.05) / (double)r.Rows.Count;

        var blocked = Differential.Run(expr, ids, a, b, Blocked(subjects, VariancePrior.Global));
        var unpaired = Differential.Run(expr, ids, a, b,
            new DifferentialOptions { Prior = VariancePrior.Global, Correction = MultipleTesting.None });

        _out.WriteLine($"correlation {blocked.BlockCorrelation:0.000}; false-positive rate at 0.05: "
            + $"blocked {Rate(blocked):0.0000}, unpaired {Rate(unpaired):0.0000}");
        Assert.InRange(blocked.BlockCorrelation, 0.4, 0.6);
        Assert.InRange(Rate(blocked), 0.03, 0.075);
        Assert.True(Rate(unpaired) > 0.15, $"unpaired false-positive rate {Rate(unpaired):0.000}");
    }

    /// <summary>
    /// The CSV's <c># n:</c> line counts subjects only for a blocked result; every other design's
    /// header is exactly what it was before the blocked design existed.
    /// </summary>
    [Fact]
    public void CsvHeader_CountsSubjectsOnlyWhenBlocked()
    {
        var (ds, _, a, b) = DetectionAnalysisTests.Setup();
        var subjects = DetectionAnalysisTests.Subjects(ds, a, b);
        var unpairedOptions = new DifferentialOptions { Prior = VariancePrior.Global };
        var blockedOptions = Blocked(subjects, VariancePrior.Global);
        var unpaired = Differential.Run(ds.ExprLog2, ds.FeatureIds, a, b, unpairedOptions);
        var blocked = Differential.Run(ds.ExprLog2, ds.FeatureIds, a, b, blockedOptions);

        string Header(DifferentialResult r, DifferentialOptions o)
        {
            var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"prism-blocked-{Guid.NewGuid():N}.csv");
            try
            {
                DifferentialCsv.Write(path, r, ds, o, SignificanceRule.Default, "g", "A", "B");
                return string.Join("\n", System.IO.File.ReadLines(path).TakeWhile(l => l.StartsWith('#')));
            }
            finally
            {
                System.IO.File.Delete(path);
            }
        }

        Assert.Contains($"# n: {a.Count} vs {b.Count}; tested", Header(unpaired, unpairedOptions));
        var blockedHeader = Header(blocked, blockedOptions);
        Assert.Contains($"# n: {blocked.NA} ({blocked.SubjectsA} subjects) vs {blocked.NB} ({blocked.SubjectsB} subjects)",
            blockedHeader);
        Assert.Contains("# blocked by subject: intra-subject correlation", blockedHeader);
    }

    [Fact]
    public void Detection_UnderTheBlockedDesign_SaysItDoesNotAccountForRepeats()
    {
        Assert.Contains("NOT account for repeated samples",
            DetectionAnalysis.UnpairedNote(UnpairedReason.RepeatedSubjects));
    }

    [Fact]
    public void Describe_NamesTheDesign()
    {
        var options = new DifferentialOptions { Design = DifferentialDesign.BlockedBySubject, Prior = VariancePrior.Global };
        Assert.Equal("moderated t (global prior), unpaired, blocked by subject", options.Describe());
        Assert.True(DifferentialDesign.BlockedBySubject.UsesSubjects());
        Assert.False(DifferentialDesign.BlockedBySubject.IsTrend());
    }
}
