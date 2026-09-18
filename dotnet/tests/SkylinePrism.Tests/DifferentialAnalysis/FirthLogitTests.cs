using System;
using SkylinePrism.Core.DifferentialAnalysis.Detection;
using Xunit;

namespace SkylinePrism.Tests.DifferentialAnalysis;

/// <summary>
/// Parity tests for <see cref="FirthLogit"/> against the Python explorer's <c>_firth_logit</c>.
/// </summary>
public class FirthLogitTests
{
    [Fact]
    public void Fit_SeparableData_GivesFiniteEstimates()
    {
        // Perfectly separable: the MLE diverges, Firth stays finite.
        var x = new[,]
        {
            { 1.0, -2.0 }, { 1.0, -1.0 }, { 1.0, -0.5 }, { 1.0, 0.5 }, { 1.0, 1.0 }, { 1.0, 2.0 },
        };
        var y = new[] { 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 };

        var res = FirthLogit.Fit(x, y);

        Assert.True(res.Converged);
        Assert.True(Math.Abs(res.Beta[0]) < 1e-9);
        Assert.Equal(1.5829695629713765, res.Beta[1], 9);
        Assert.Equal(-1.4997056968696356, res.PenalizedLogLik, 9);
    }

    [Fact]
    public void Fit_MixedData_MatchesExplorer()
    {
        var x = new[,]
        {
            { 1.0, -2.0 }, { 1.0, -1.0 }, { 1.0, 0.0 }, { 1.0, 1.0 },
            { 1.0, 2.0 }, { 1.0, 3.0 }, { 1.0, 4.0 }, { 1.0, 5.0 },
        };
        var y = new[] { 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0 };

        var res = FirthLogit.Fit(x, y);

        Assert.True(res.Converged);
        Assert.Equal(-0.3332674042118441, res.Beta[0], 9);
        Assert.Equal(0.2221782694745628, res.Beta[1], 9);
        Assert.Equal(-3.697167700849171, res.PenalizedLogLik, 9);
    }
}
