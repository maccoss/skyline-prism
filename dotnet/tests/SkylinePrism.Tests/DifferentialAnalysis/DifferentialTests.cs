using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.DifferentialAnalysis;
using Xunit;

namespace SkylinePrism.Tests.DifferentialAnalysis;

/// <summary>
/// End-to-end parity tests for <see cref="Differential"/> against the Python PRISM Differential
/// Explorer's <c>differential()</c> (which drives inmoose 0.9.1 lmFit + squeezeVar + scipy t.cdf +
/// statsmodels BH). The input matrix is a deterministic rational grid reproduced bit-for-bit here,
/// and reference values come from running that same grid through the explorer.
/// </summary>
public class DifferentialTests
{
    private const int F = 8;
    private const int S = 6;

    // expr[f, s] = 10 + noise (+3/-3 planted in group B for f0/f2); noise scaled per feature when
    // hetero, which spreads the residual variances enough to give a finite df_prior.
    private static double[,] BuildMatrix(bool hetero)
    {
        var m = new double[F, S];
        for (var f = 0; f < F; f++)
        for (var s = 0; s < S; s++)
        {
            var noise = ((f * 3 + s * 2) % 7) / 3.0;
            if (hetero)
                noise *= 1.0 + f * 0.5;
            var v = 10.0 + noise;
            if (s >= 3)
            {
                if (f == 0)
                    v += 3.0;
                if (f == 2)
                    v -= 3.0;
            }

            m[f, s] = v;
        }

        return m;
    }

    private static readonly string[] Ids = Enumerable.Range(0, F).Select(f => $"f{f}").ToArray();
    private static readonly int[] GroupA = { 0, 1, 2 };
    private static readonly int[] GroupB = { 3, 4, 5 };

    private static void AssertRel(double expected, double actual, double rtol)
    {
        var denom = Math.Abs(expected);
        var err = denom > 0 ? Math.Abs(actual - expected) / denom : Math.Abs(actual - expected);
        Assert.True(err <= rtol, $"expected {expected:R}, actual {actual:R}, rel err {err:R} > {rtol:R}");
    }

    [Fact]
    public void Run_UniformVariance_InfinitePriorMatchesExplorer()
    {
        var res = Differential.Run(BuildMatrix(hetero: false), Ids, GroupA, GroupB);

        Assert.Equal(8, res.NFeaturesTested);
        Assert.Equal(0, res.NFeaturesDropped);
        Assert.Equal(4.0, res.DfResidual, 12);
        Assert.True(double.IsPositiveInfinity(res.DfPrior));
        Assert.Equal("global", res.VariancePrior);

        // Top two hits, in order.
        Assert.Equal("f0", res.Rows[0].FeatureId);
        Assert.Equal("f2", res.Rows[1].FeatureId);

        var byId = res.Rows.ToDictionary(r => r.FeatureId);
        var f0 = byId["f0"];
        Assert.Equal(3.44444444444444, f0.LogFc, 12);
        Assert.Equal(5.490812290348585, f0.T, 11);
        AssertRel(4.0008939083426606e-08, f0.PValue, 1e-9);
        AssertRel(3.2007151266741285e-07, f0.AdjPValue, 1e-9);
        Assert.Equal(10.666666666666666, f0.MeanA, 12);
        Assert.Equal(14.111111111111112, f0.MeanB, 12);

        var f2 = byId["f2"];
        Assert.Equal(-3.3333333333333344, f2.LogFc, 12);
        Assert.Equal(-5.313689313240575, f2.T, 11);
        AssertRel(1.0742771784671925e-07, f2.PValue, 1e-9);

        // A null feature.
        AssertRel(0.4786398352857747, byId["f4"].PValue, 1e-9);
    }

    [Fact]
    public void Run_HeterogeneousVariance_FinitePriorMatchesExplorer()
    {
        var res = Differential.Run(BuildMatrix(hetero: true), Ids, GroupA, GroupB);

        Assert.Equal(4.0, res.DfResidual, 12);
        Assert.Equal(5.30370628191611, res.DfPrior, 9);

        var byId = res.Rows.ToDictionary(r => r.FeatureId);
        var f0 = byId["f0"];
        Assert.Equal(3.44444444444444, f0.LogFc, 12);
        Assert.Equal(2.7075922559178185, f0.T, 11);
        AssertRel(0.023410739862572128, f0.PValue, 1e-9);
        AssertRel(0.14646999341291478, f0.AdjPValue, 1e-9);

        var f2 = byId["f2"];
        Assert.Equal(-3.6666666666666665, f2.LogFc, 12);
        Assert.Equal(-2.438707720888919, f2.T, 11);
        AssertRel(0.036617498353228695, f2.PValue, 1e-9);
        Assert.Equal(12.222222222222221, f2.MeanA, 12);
        Assert.Equal(8.555555555555557, f2.MeanB, 12);
    }

    [Fact]
    public void Run_TrendPrior_MatchesExplorer()
    {
        var res = Differential.Run(BuildMatrix(hetero: true), Ids, GroupA, GroupB, trend: true);

        Assert.Equal("intensity-trend", res.VariancePrior);
        Assert.True(double.IsPositiveInfinity(res.DfPrior));

        var byId = res.Rows.ToDictionary(r => r.FeatureId);
        Assert.Equal(3.44444444444444, byId["f0"].LogFc, 12);
        Assert.Equal(2.6829479370580227, byId["f0"].T, 9);
        AssertRel(0.007297635004183641, byId["f0"].PValue, 1e-9);
        AssertRel(0.034911717404574286, byId["f0"].AdjPValue, 1e-9);
        Assert.Equal(-2.622531131612897, byId["f2"].T, 9);
        AssertRel(0.008727929351143571, byId["f2"].PValue, 1e-9);
    }

    [Fact]
    public void Run_WithCovariates_MatchesExplorer()
    {
        // age (numeric, mean-centered) + sex (categorical -> sex_M dummy after dropping F).
        var covariates = new Covariate[]
        {
            new NumericCovariate("age", new[] { 50.0, 60.0, 70.0, 55.0, 65.0, 75.0 }),
            new CategoricalCovariate("sex", new string?[] { "M", "F", "M", "F", "M", "F" }),
        };
        var res = Differential.Run(BuildMatrix(hetero: true), Ids, GroupA, GroupB, covariates: covariates);

        Assert.Equal(new[] { "age", "sex_M" }, res.CovariatesUsed);
        Assert.Empty(res.Messages);
        Assert.Equal(2.0, res.DfResidual, 12);
        Assert.Equal(0.31371980324769216, res.DfPrior, 9);

        var byId = res.Rows.ToDictionary(r => r.FeatureId);
        var f4 = byId["f4"];
        Assert.Equal(3.2499999999999987, f4.LogFc, 12);
        AssertRel(421.7642356973605, f4.T, 1e-9);
        AssertRel(1.05164524369471e-06, f4.PValue, 1e-9);
        AssertRel(8.41316194955768e-06, f4.AdjPValue, 1e-9);

        var f0 = byId["f0"];
        Assert.Equal(3.2083333333333304, f0.LogFc, 12);
        AssertRel(4.0187111479273625, f0.T, 1e-9);
        AssertRel(0.044252193135609935, f0.PValue, 1e-9);
    }

    [Fact]
    public void Run_ConfoundedCategorical_DroppedWithMessage_RevertsToNoCovariate()
    {
        // cond exactly matches the group split, so its dummy is dropped as confounded and the design
        // reverts to [intercept, groupB] - i.e. the no-covariate hetero result (df_prior 5.3037).
        var covariates = new Covariate[]
        {
            new CategoricalCovariate("cond", new string?[] { "c1", "c1", "c1", "c2", "c2", "c2" }),
        };
        var res = Differential.Run(BuildMatrix(hetero: true), Ids, GroupA, GroupB, covariates: covariates);

        Assert.Empty(res.CovariatesUsed);
        Assert.Contains(res.Messages, m => m.Contains("confounded with group"));
        Assert.Equal(5.30370628191611, res.DfPrior, 9);
        Assert.Equal(2.7075922559178185, res.Rows.Single(r => r.FeatureId == "f0").T, 11);
    }

    [Fact]
    public void Run_ConstantCovariate_SkippedWithMessage()
    {
        var covariates = new Covariate[]
        {
            new NumericCovariate("batch", new[] { 5.0, 5.0, 5.0, 5.0, 5.0, 5.0 }),
        };
        var res = Differential.Run(BuildMatrix(hetero: true), Ids, GroupA, GroupB, covariates: covariates);

        Assert.Empty(res.CovariatesUsed);
        Assert.Contains(res.Messages, m => m.Contains("constant"));
        Assert.Equal(5.30370628191611, res.DfPrior, 9);
    }

    [Fact]
    public void Run_CovariateMissingInSelectedSample_Skipped()
    {
        var covariates = new Covariate[]
        {
            new NumericCovariate("pmi", new[] { 5.0, double.NaN, 7.0, 6.0, 8.0, 9.0 }),
        };
        var res = Differential.Run(BuildMatrix(hetero: true), Ids, GroupA, GroupB, covariates: covariates);

        Assert.Empty(res.CovariatesUsed);
        Assert.Contains(res.Messages, m => m.Contains("missing values"));
    }

    [Fact]
    public void Run_CovariateCollinearWithGroup_RejectedAsRankDeficient()
    {
        // A numeric covariate equal to the group indicator is not caught as a categorical confound;
        // it makes the design rank-deficient, which must throw (matching the Python reference).
        var covariates = new Covariate[]
        {
            new NumericCovariate("grpdup", new[] { 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 }),
        };
        Assert.Throws<ArgumentException>(() =>
            Differential.Run(BuildMatrix(hetero: true), Ids, GroupA, GroupB, covariates: covariates));
    }

    [Fact]
    public void FromMetadata_InfersNumericVsCategorical()
    {
        // All non-null values parse as numbers -> numeric (integer-coded batch is centered, not dummied).
        var numeric = Assert.IsType<NumericCovariate>(
            Covariate.FromMetadata("batch", new string?[] { "1", "2", "3", null }));
        Assert.Equal(1.0, numeric.Values[0], 12);
        Assert.True(double.IsNaN(numeric.Values[3])); // null -> NaN (missing)

        // Any non-numeric value -> categorical.
        Assert.IsType<CategoricalCovariate>(
            Covariate.FromMetadata("sex", new string?[] { "M", "F", "M" }));

        // All-null -> categorical (will be skipped as missing downstream).
        Assert.IsType<CategoricalCovariate>(
            Covariate.FromMetadata("empty", new string?[] { null, null }));
    }

    [Fact]
    public void Run_RejectsOverlappingGroups()
    {
        Assert.Throws<ArgumentException>(() =>
            Differential.Run(BuildMatrix(false), Ids, new[] { 0, 1, 2 }, new[] { 2, 3, 4 }));
    }

    [Fact]
    public void Run_RejectsTooSmallGroup()
    {
        Assert.Throws<ArgumentException>(() =>
            Differential.Run(BuildMatrix(false), Ids, new[] { 0 }, new[] { 3, 4, 5 }));
    }

    [Fact]
    public void Run_DropsFeaturesWithMissingSelectedValues()
    {
        var m = BuildMatrix(false);
        m[5, 4] = double.NaN; // f5 has a missing value in a selected column
        var res = Differential.Run(m, Ids, GroupA, GroupB);
        Assert.Equal(7, res.NFeaturesTested);
        Assert.Equal(1, res.NFeaturesDropped);
        Assert.DoesNotContain(res.Rows, r => r.FeatureId == "f5");
    }
}
