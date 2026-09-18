using System;
using SkylinePrism.Core.DifferentialAnalysis;
using Xunit;

namespace SkylinePrism.Tests.DifferentialAnalysis;

/// <summary>
/// Parity tests for <see cref="EmpiricalBayes"/> against inmoose 0.9.1 (the limma port the Python
/// PRISM Differential Explorer uses) and scipy.special.polygamma. Reference values were generated
/// with <c>inmoose.limma.squeezeVar.squeezeVar</c>, <c>fitFDist.trigammaInverse</c> and
/// <c>scipy.special.polygamma(1|2, .)</c> on the exact inputs below.
/// </summary>
public class EmpiricalBayesTests
{
    private static void AssertRel(double expected, double actual, double rtol)
    {
        var denom = Math.Abs(expected);
        var err = denom > 0 ? Math.Abs(actual - expected) / denom : Math.Abs(actual - expected);
        Assert.True(err <= rtol, $"expected {expected:R}, actual {actual:R}, rel err {err:R} > {rtol:R}");
    }

    [Fact]
    public void SqueezeVarGlobal_FiniteDfPrior_MatchesInmoose()
    {
        // squeezeVar([0.05,0.5,0.1,1.2,0.3,0.8,0.02,0.15,2.0,0.25,0.6,0.09,0.35,1.5,0.04,0.7,0.12,
        //             0.9,0.2,0.45], df=7)
        var var = new[]
        {
            0.05, 0.5, 0.1, 1.2, 0.3, 0.8, 0.02, 0.15, 2.0, 0.25,
            0.6, 0.09, 0.35, 1.5, 0.04, 0.7, 0.12, 0.9, 0.2, 0.45,
        };
        var res = EmpiricalBayes.SqueezeVarGlobal(var, 7.0);

        Assert.Equal(2.3826351810043684, res.DfPrior, 9);
        Assert.Equal(0.20167589574870898, res.VarPrior[0], 9);
        Assert.Equal(0.08851671927443948, res.VarPost[0], 9);
        Assert.Equal(0.42424329706756725, res.VarPost[1], 9);
        Assert.Equal(0.1258196723625648, res.VarPost[2], 9);
        Assert.Equal(0.9464846403013215, res.VarPost[3], 9);
        Assert.Equal(0.38694034397944194, res.VarPost[19], 9);
        Assert.Empty(res.Warnings);
    }

    [Fact]
    public void SqueezeVarGlobal_NegativeVariance_ExcludedFromFitButSqueezed()
    {
        // squeezeVar([0.1,0.2,-0.3,0.4,0.5], df=5): -0.3 is excluded from the prior fit but still
        // gets a posterior. Large df_prior (51.19) is exactly the regime the polygamma threshold
        // must hold to 1e-9 on.
        var res = EmpiricalBayes.SqueezeVarGlobal(new[] { 0.1, 0.2, -0.3, 0.4, 0.5 }, 5.0);
        Assert.Equal(51.19236370256727, res.DfPrior, 8);
        Assert.Equal(0.3051681091118398, res.VarPrior[0], 9);
        Assert.Equal(0.2869122380652134, res.VarPost[0], 9);
        Assert.Equal(0.2958102442542137, res.VarPost[1], 9);
        Assert.Equal(0.2513202133092121, res.VarPost[2], 9);
        Assert.Equal(0.3136062566322143, res.VarPost[3], 9);
        Assert.Equal(0.32250426282121464, res.VarPost[4], 9);
    }

    [Fact]
    public void SqueezeVarGlobal_InfiniteDfPrior_ShrinksToPooledMean()
    {
        // squeezeVar([0.1,0.2,0.15,0.3,0.25,0.05,0.4,0.12,0.22,0.18], df=6) -> evar<=0:
        //   df_prior = inf, var_prior = pooled mean 0.197, every var_post = 0.197.
        var var = new[] { 0.1, 0.2, 0.15, 0.3, 0.25, 0.05, 0.4, 0.12, 0.22, 0.18 };
        var res = EmpiricalBayes.SqueezeVarGlobal(var, 6.0);

        Assert.True(double.IsPositiveInfinity(res.DfPrior));
        Assert.Equal(0.197, res.VarPrior[0], 12);
        foreach (var v in res.VarPost)
            Assert.Equal(0.197, v, 12);
    }

    [Fact]
    public void SqueezeVarGlobal_ZeroVariances_FlooredAndWarned()
    {
        // squeezeVar([0,0,0,1,2], df=5): median is 0 (3 of 5 zero), so the "more than half zero"
        // branch fires (m set to 1) and the floor becomes 1e-5.
        var res = EmpiricalBayes.SqueezeVarGlobal(new[] { 0.0, 0.0, 0.0, 1.0, 2.0 }, 5.0);
        Assert.Equal(0.3145401482748184, res.DfPrior, 9);
        Assert.Equal(1.1092097785147672e-05, res.VarPrior[0], 15);
        Assert.Equal(6.564839072956646e-07, res.VarPost[0], 15);
        Assert.Contains(res.Warnings, w => w.Contains("residual variances are exactly zero"));
    }

    [Fact]
    public void SqueezeVarGlobal_SomeZeroVariances_WarnsOffsetAwayFromZero()
    {
        // Median is non-zero (1.5) but a zero is present -> the "offset away from zero" warning.
        var res = EmpiricalBayes.SqueezeVarGlobal(new[] { 0.0, 1.0, 2.0, 3.0 }, 5.0);
        Assert.Contains(res.Warnings, w => w.Contains("offset away from zero"));
    }

    [Fact]
    public void SqueezeVarGlobal_MixedNaNInf_MatchesInmoose()
    {
        // squeezeVar([0.3, nan, inf], df=5): only 0.3 is usable (nok==1) -> df_prior 0; the posterior
        // still passes NaN and Inf through.
        var res = EmpiricalBayes.SqueezeVarGlobal(new[] { 0.3, double.NaN, double.PositiveInfinity }, 5.0);
        Assert.Equal(0.0, res.DfPrior, 12);
        Assert.Equal(0.3, res.VarPost[0], 12);
        Assert.True(double.IsNaN(res.VarPost[1]));
        Assert.True(double.IsPositiveInfinity(res.VarPost[2]));
    }

    [Fact]
    public void SqueezeVarTrend_FiniteDfPrior_MatchesInmoose()
    {
        // squeezeVar(var, df=8, covariate=cov) with a covariate-driven trend plus residual scatter
        // (splineDf=4). Validates the natural-spline trend fit + the finite-prior path.
        var mult = new[] { 0.3, 0.6, 1.0, 1.8, 3.0 };
        var var = new double[40];
        var cov = new double[40];
        for (var i = 0; i < 40; i++)
        {
            var[i] = 0.2 * (1.0 + 0.1 * (i % 11)) * mult[i % 5];
            cov[i] = 5.0 + (i % 11) * 0.3;
        }

        var res = EmpiricalBayes.SqueezeVarTrend(var, 8.0, cov);

        Assert.Equal(5.78755728336074, res.DfPrior, 8);
        Assert.Equal(0.19404267835533553, res.VarPrior[0], 9);
        Assert.Equal(0.2333032011592363, res.VarPrior[3], 9);
        Assert.Equal(0.24843734065948617, res.VarPrior[10], 9);
        Assert.Equal(0.1162666514055277, res.VarPost[0], 9);
        Assert.Equal(0.3694821016082684, res.VarPost[3], 9);
        Assert.Equal(0.17391371735487748, res.VarPost[10], 9);
    }

    [Fact]
    public void TrigammaInverse_MatchesInmoose()
    {
        // inmoose.limma.fitFDist.trigammaInverse (Newton path)
        Assert.Equal(2.459952948352307, EmpiricalBayes.TrigammaInverse(0.5), 9);
        Assert.Equal(10.49168182107842, EmpiricalBayes.TrigammaInverse(0.1), 9);
        Assert.Equal(100000.49999916666, EmpiricalBayes.TrigammaInverse(1e-5), 6);
    }

    [Fact]
    public void TrigammaInverse_SpecialBranches()
    {
        // x > 1e7 -> 1/sqrt(x); x < 1e-6 -> 1/x; x < 0 or NaN -> NaN.
        Assert.Equal(3.162277502054508e-4, EmpiricalBayes.TrigammaInverse(1e7 + 1), 15);
        Assert.Equal(1e8, EmpiricalBayes.TrigammaInverse(1e-8), 6);
        Assert.True(double.IsNaN(EmpiricalBayes.TrigammaInverse(-1.0)));
        Assert.True(double.IsNaN(EmpiricalBayes.TrigammaInverse(double.NaN)));
    }

    [Fact]
    public void Trigamma_MatchesScipyAcrossRange()
    {
        // scipy.special.polygamma(1, x) - including x in (0,1) and either side of the recurrence
        // threshold, verified to 1e-12 relative (the fix that keeps a large df_prior accurate).
        (double X, double Y)[] cases =
        {
            (0.05, 401.5323573421151), (0.3, 12.245364546107734), (0.7, 2.8340491566946113),
            (1.0, 1.6449340668482266), (2.5, 0.4903577561002349), (3.5, 0.3303577561002349),
            (11.9999999, 0.08690187362648895), (12.0, 0.08690187287176838),
            (12.0000001, 0.08690187211704786), (40.0, 0.025315103841291025),
        };
        foreach (var (x, y) in cases)
            AssertRel(y, EmpiricalBayes.Trigamma(x), 1e-12);
    }

    [Fact]
    public void Tetragamma_MatchesScipyAcrossRange()
    {
        // scipy.special.polygamma(2, x), verified to 1e-12 relative.
        (double X, double Y)[] cases =
        {
            (0.05, -16002.108158021945), (0.3, -75.27253658872603), (0.7, -6.434992874190922),
            (1.0, -2.404113806319188), (2.5, -0.2362040516417274), (3.5, -0.1082040516417274),
            (11.9999999, -0.0075472055000082375), (12.0, -0.007547205368998913),
            (12.0000001, -0.007547205237989593), (40.0, -0.0006408202718352984),
        };
        foreach (var (x, y) in cases)
            AssertRel(y, EmpiricalBayes.Tetragamma(x), 1e-12);
    }

    [Fact]
    public void Trigamma_MatchesClosedForms()
    {
        // trigamma(1) = pi^2/6, trigamma(1/2) = pi^2/2, trigamma(2) = pi^2/6 - 1.
        Assert.Equal(Math.PI * Math.PI / 6.0, EmpiricalBayes.Trigamma(1.0), 12);
        Assert.Equal(Math.PI * Math.PI / 2.0, EmpiricalBayes.Trigamma(0.5), 12);
        Assert.Equal(Math.PI * Math.PI / 6.0 - 1.0, EmpiricalBayes.Trigamma(2.0), 12);
    }
}
