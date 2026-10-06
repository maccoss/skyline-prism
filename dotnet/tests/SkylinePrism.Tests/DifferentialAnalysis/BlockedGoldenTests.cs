using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using SkylinePrism.Core.DifferentialAnalysis;
using Xunit;
using Xunit.Abstractions;

namespace SkylinePrism.Tests.DifferentialAnalysis;

/// <summary>
/// The subject-blocked design against <c>blocked.json</c>, which R's limma wrote:
/// <c>duplicateCorrelation</c>, then <c>lmFit(block=, correlation=)</c>, then <c>eBayes</c>.
/// See <c>generate_blocked.R</c> and the fixture README for what each case covers.
/// </summary>
public class BlockedGoldenTests
{
    private const string File = "blocked.json";
    private readonly ITestOutputHelper _out;

    public BlockedGoldenTests(ITestOutputHelper output) => _out = output;

    public static IEnumerable<object[]> Cases() => Golden.CaseNames(File);

    /// <summary>
    /// Every case but <c>missing_values</c>, whose features have different residual df - which only
    /// limma's per-feature path produces, and PRISM never fits.
    /// </summary>
    public static IEnumerable<object[]> CompleteCases() =>
        Cases().Where(c => (string)c[0] != "missing_values");

    private static string[] Block(JsonElement c) =>
        c.GetProperty("block").EnumerateArray().Select(e => e.GetString()!).ToArray();

    /// <summary>
    /// Each feature's correlation, and the consensus.
    /// </summary>
    /// <remarks>
    /// PRISM replays statmod's iterations, including the 20-iteration cap and the 1e-6 stopping
    /// rule that leave limma short of the REML optimum, so it is held to limma's values rather than
    /// to <c>atanh_correlations_converged</c>. What it cannot replay is the floating-point path:
    /// the residual space is reached through a different basis (an SVD of the design rather than
    /// lm.fit's Householder Q), so every quantity entering the scoring differs in the last few bits.
    /// The tolerance is set from what that leaves, measured on these cases.
    /// </remarks>
    [Theory]
    [MemberData(nameof(Cases))]
    public void Correlation_MatchesLimma(string name)
    {
        var c = Golden.Case(File, name);
        var result = DuplicateCorrelation.Estimate(Golden.Mat(c, "expr"), Golden.Mat(c, "design"), Block(c));

        var expected = Golden.Vec(c, "atanh_correlations");
        var worst = 0.0;
        for (var i = 0; i < expected.Length; i++)
            if (double.IsFinite(expected[i]))
                worst = Math.Max(worst, Math.Abs(expected[i] - result.AtanhCorrelations[i]));
        _out.WriteLine($"{name}: worst per-feature |atanh diff| = {worst:R}; consensus {result.Consensus:R}");

        for (var i = 0; i < expected.Length; i++)
            NearOrNaN(expected[i], result.AtanhCorrelations[i], AtanhTolerance, $"{name} atanh rho[{i}]");
        Near(Golden.Num(c, "consensus_correlation"), result.Consensus, ConsensusTolerance,
            $"{name} consensus");
        Assert.Null(result.Degenerate);
    }

    // Measured worst cases on the committed fixture, each with about 100x of margin for another
    // platform's floating point: per-feature atanh 3.9e-14 absolute; the GLS and eBayes scale
    // quantities (sigma, unscaled SE, Amean, prior df and variance) 4.8e-14 relative; coefficients
    // and t 9e-13 against max(|x|, 1) (see Near); p 7.4e-11 relative, where a t near zero puts p
    // near 1 and carries t's absolute error.
    private const double AtanhTolerance = 1e-12;
    private const double ConsensusTolerance = 1e-12;
    private const double ScaleTolerance = 1e-12;
    private const double CoefTolerance = 1e-10;
    private const double PTolerance = 1e-9;

    /// <summary>
    /// The GLS fit at limma's consensus correlation. Closed-form arithmetic, so it is held tightly.
    /// </summary>
    /// <remarks>
    /// <c>missing_values</c> is compared on its complete features only: PRISM fits complete features,
    /// and for those limma's per-feature path uses the same correlation matrix as its shared one.
    /// </remarks>
    [Theory]
    [MemberData(nameof(Cases))]
    public void GlsFit_MatchesLimma(string name)
    {
        var c = Golden.Case(File, name);
        var expr = Golden.Mat(c, "expr");
        var complete = CompleteRows(expr);
        var fit = LinearModel.FitBlocked(Rows(expr, complete), Golden.Mat(c, "design"), Block(c),
            Golden.Num(c, "consensus_correlation"));

        var coef = Golden.Mat(c, "coefficients");
        var stdev = Golden.Mat(c, "stdev_unscaled");
        var sigma = Golden.Vec(c, "sigma");
        var amean = Golden.Vec(c, "amean");
        var dfRes = Golden.Vec(c, "df_residual");
        for (var k = 0; k < complete.Length; k++)
        {
            var i = complete[k];
            for (var j = 0; j < coef.GetLength(1); j++)
            {
                Near(coef[i, j], fit.Coefficients[k, j], CoefTolerance, $"{name} coef[{i},{j}]");
                Golden.Close(stdev[i, j], fit.StdevUnscaled[j], ScaleTolerance, $"{name} stdev_unscaled[{i},{j}]");
            }
            Golden.Close(sigma[i], fit.Sigma[k], ScaleTolerance, $"{name} sigma[{i}]");
            Golden.Close(amean[i], fit.Amean[k], ScaleTolerance, $"{name} amean[{i}]");
            Assert.Equal(dfRes[i], fit.DfResidual);
        }
    }

    /// <summary>
    /// eBayes on the GLS fit, both priors, composed from PRISM's own squeezeVar exactly as eBayes
    /// composes it.
    /// </summary>
    [Theory]
    [MemberData(nameof(CompleteCases))]
    public void ModeratedT_MatchesEBayes(string name)
    {
        var c = Golden.Case(File, name);
        var fit = LinearModel.FitBlocked(Golden.Mat(c, "expr"), Golden.Mat(c, "design"), Block(c),
            Golden.Num(c, "consensus_correlation"));
        var variances = fit.Sigma.Select(s => s * s).ToArray();

        foreach (var (prior, squeezed) in new[]
                 {
                     ("global", EmpiricalBayes.SqueezeVarGlobal(variances, fit.DfResidual)),
                     ("trend", EmpiricalBayes.SqueezeVarTrend(variances, fit.DfResidual, fit.Amean)),
                 })
        {
            var e = c.GetProperty(prior);
            Golden.Close(Golden.Num(e, "df_prior"), squeezed.DfPrior, ScaleTolerance, $"{name} {prior} df_prior");
            var s2Prior = Golden.Vec(e, "s2_prior");
            var t = Golden.Mat(e, "t");
            var p = Golden.Mat(e, "p");
            for (var i = 0; i < variances.Length; i++)
            {
                Golden.Close(s2Prior.Length == 1 ? s2Prior[0] : s2Prior[i],
                    squeezed.VarPrior.Length == 1 ? squeezed.VarPrior[0] : squeezed.VarPrior[i],
                    ScaleTolerance, $"{name} {prior} s2_prior[{i}]");
                for (var j = 0; j < t.GetLength(1); j++)
                {
                    var tj = fit.Coefficients[i, j] / (fit.StdevUnscaled[j] * Math.Sqrt(squeezed.VarPost[i]));
                    Near(t[i, j], tj, CoefTolerance, $"{name} {prior} t[{i},{j}]");
                    Golden.Close(p[i, j], Distributions.TwoSidedT(tj, fit.DfResidual + squeezed.DfPrior),
                        PTolerance, $"{name} {prior} p[{i},{j}]");
                }
            }
        }
    }

    /// <summary>
    /// Relative error against <c>max(|expected|, 1)</c>, for quantities that can sit at zero.
    /// </summary>
    /// <remarks>
    /// A coefficient that is truly null comes out of the GLS as a difference of values near 14
    /// (log2) and lands at about 1e-6, so its last bits are noise of absolute size ~1e-15 and a
    /// relative comparison against 1e-6 reports 1e-9 for agreement to the last bit of the data.
    /// Measured worst case on these cases: 1.0e-9 relative on such a coefficient, 9e-13 everywhere
    /// else.
    /// </remarks>
    private static void Near(double expected, double actual, double tol, string what) =>
        Assert.True(Math.Abs(actual - expected) <= tol * Math.Max(Math.Abs(expected), 1.0),
            $"{what}: expected {expected:R}, actual {actual:R}");

    private static void NearOrNaN(double expected, double actual, double tol, string what)
    {
        if (double.IsNaN(expected) || double.IsNaN(actual))
        {
            Assert.True(double.IsNaN(expected) && double.IsNaN(actual), $"{what}: expected {expected:R}, actual {actual:R}");
            return;
        }

        Near(expected, actual, tol, what);
    }

    [Fact]
    public void DegenerateBlocks_AreZero_AndSaySo()
    {
        foreach (var d in Golden.Load(File).GetProperty("degenerate").EnumerateArray())
        {
            var result = DuplicateCorrelation.Estimate(Golden.Mat(d, "expr"), Golden.Mat(d, "design"), Block(d));
            Assert.Equal(Golden.Num(d, "consensus_correlation"), result.Consensus);
            Assert.NotNull(result.Degenerate);
        }
    }

    private static int[] CompleteRows(double[,] m) =>
        Enumerable.Range(0, m.GetLength(0))
            .Where(i => Enumerable.Range(0, m.GetLength(1)).All(j => double.IsFinite(m[i, j])))
            .ToArray();

    private static double[,] Rows(double[,] m, int[] rows)
    {
        var r = new double[rows.Length, m.GetLength(1)];
        for (var i = 0; i < rows.Length; i++)
            for (var j = 0; j < m.GetLength(1); j++)
                r[i, j] = m[rows[i], j];
        return r;
    }
}
