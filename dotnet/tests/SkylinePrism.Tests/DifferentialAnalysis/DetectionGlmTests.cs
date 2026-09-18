using System;
using System.Linq;
using SkylinePrism.Core.DifferentialAnalysis;
using SkylinePrism.Core.DifferentialAnalysis.Detection;
using Xunit;

namespace SkylinePrism.Tests.DifferentialAnalysis;

/// <summary>
/// Parity tests for <see cref="DetectionGlm"/> against the explorer's <c>detection_test_glm</c>.
/// </summary>
public class DetectionGlmTests
{
    private static readonly double[,] Det =
    {
        { 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1 }, // pep0: 2/6 A, 5/6 B
        { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 }, // pep1: tracks batch b1
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, // pep2: all detected
    };

    private static readonly string[] Peptides = { "pep0", "pep1", "pep2" };
    private static readonly int[] GroupA = { 0, 1, 2, 3, 4, 5 };
    private static readonly int[] GroupB = { 6, 7, 8, 9, 10, 11 };

    [Fact]
    public void Run_Identifiable_MatchesExplorer()
    {
        var batch = new Covariate[]
        {
            new CategoricalCovariate("batch",
                new string?[] { "b1", "b2", "b1", "b2", "b1", "b2", "b1", "b2", "b1", "b2", "b1", "b2" }),
        };

        var res = DetectionGlm.Run(Det, Peptides, GroupA, GroupB, batch);

        Assert.True(res.Identifiable);
        Assert.Equal(0.0, res.GroupCollinearityR2, 12);
        Assert.Equal(3, res.NParams);
        Assert.Equal(new[] { "batch_b2" }, res.CovariatesUsed);

        var byId = res.Rows.ToDictionary(r => r.PeptideId);
        var pep0 = byId["pep0"];
        Assert.Equal("pep0", res.Rows[0].PeptideId); // smallest p sorts first
        Assert.Equal(2, pep0.DetA);
        Assert.Equal(5, pep0.DetB);
        Assert.Equal(1.7987657834327673, pep0.LogOr, 9);
        Assert.Equal(0.13209640900416442, pep0.P, 9);
        Assert.Equal(0.39628922701249325, pep0.Q, 9);

        Assert.True(Math.Abs(byId["pep1"].LogOr) < 1e-9);
        Assert.Equal(1.0, byId["pep1"].P, 12);
        Assert.Equal(0.0, byId["pep2"].LogOr, 12); // all-detected -> (0, 1)
        Assert.Equal(1.0, byId["pep2"].P, 12);
    }

    [Fact]
    public void Run_ConfoundedCovariate_NotIdentifiable()
    {
        // cohort nested in group (grp = c3 + c4), so the group term adds no rank over the covariates.
        var cohort = new Covariate[]
        {
            new CategoricalCovariate("cohort",
                new string?[] { "c1", "c1", "c1", "c2", "c2", "c2", "c3", "c3", "c3", "c4", "c4", "c4" }),
        };

        var res = DetectionGlm.Run(Det, Peptides, GroupA, GroupB, cohort);

        Assert.False(res.Identifiable);
        Assert.Equal(1.0, res.GroupCollinearityR2, 9);
        Assert.Empty(res.Rows);
        Assert.NotNull(res.Warning);
    }

    [Fact]
    public void Run_NoCovariates_IsIdentifiable()
    {
        var res = DetectionGlm.Run(Det, Peptides, GroupA, GroupB);
        Assert.True(res.Identifiable);
        Assert.Equal(2, res.NParams); // intercept + group
        Assert.Empty(res.CovariatesUsed);
    }
}
