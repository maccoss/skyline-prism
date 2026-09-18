using System.Linq;
using SkylinePrism.Core.DifferentialAnalysis;
using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.DifferentialAnalysis;

/// <summary>
/// Tests for <see cref="MarkerPanel"/>: gene-symbol matching, row z-scoring, group-mean vs per-sample
/// heatmap columns, the per-group panel-score for the boxplot, and the found/not-detected report.
/// </summary>
public class MarkerPanelTests
{
    // Two matched markers (ALB, CD9) and one unmatched feature (XXX); the panel also names ZZZ, absent.
    private static readonly string[] Ids = { "PG1", "PG2", "PG3" };
    private static readonly string[] Labels = { "ALB", "CD9", "XXX" };
    private static readonly string?[] Groups = { "A", "A", "B", "B" };
    private static readonly string[] Samples = { "s1", "s2", "s3", "s4" };

    private static double[,] Expr() => new double[,]
    {
        { 1, 1, 3, 3 }, // ALB: mean 2, sd 1 -> z -1,-1,1,1
        { 1, 1, 3, 3 }, // CD9: same
        { 5, 6, 7, 8 }, // XXX: not in panel
    };

    private static ProteinList Panel() =>
        new() { Name = "Test panel", Members = { "ALB", "CD9", "ZZZ" } };

    [Fact]
    public void Evaluate_GroupMeans_MatchesExpected()
    {
        var r = MarkerPanel.Evaluate(Expr(), Ids, Labels, Groups, Samples, Panel(), perSample: false);

        Assert.Equal(new[] { "ALB", "CD9" }, r.MarkerLabels);
        Assert.Equal(new[] { "A", "B" }, r.ColumnLabels);
        Assert.Equal(new[] { "A", "B" }, r.GroupNames);

        // Row z-scored group means: A = -1, B = +1 for both markers.
        Assert.Equal(-1.0, r.Heatmap[0, 0], 9);
        Assert.Equal(1.0, r.Heatmap[0, 1], 9);
        Assert.Equal(-1.0, r.Heatmap[1, 0], 9);
        Assert.Equal(1.0, r.Heatmap[1, 1], 9);

        Assert.Equal(2, r.Found);
        Assert.Equal(3, r.Total);
        Assert.Equal(new[] { "ZZZ" }, r.NotDetected.ToArray());
        Assert.Equal(1.0, r.SymmetricMax, 9);

        // Panel score per group: each sample's mean marker z-score.
        Assert.Equal(new[] { -1.0, -1.0 }, r.PanelScoreByGroup[0]);
        Assert.Equal(new[] { 1.0, 1.0 }, r.PanelScoreByGroup[1]);
    }

    [Fact]
    public void Evaluate_PerSample_ColumnsAreSamplesOrderedByGroup()
    {
        var r = MarkerPanel.Evaluate(Expr(), Ids, Labels, Groups, Samples, Panel(), perSample: true);

        Assert.Equal(new[] { "s1", "s2", "s3", "s4" }, r.ColumnLabels);
        Assert.Equal(-1.0, r.Heatmap[0, 0], 9); // ALB, s1
        Assert.Equal(1.0, r.Heatmap[0, 3], 9);  // ALB, s4
    }

    [Fact]
    public void Evaluate_ZeroVarianceRow_IsNaN()
    {
        var expr = new double[,]
        {
            { 2, 2, 2, 2 }, // ALB: no variance -> NaN row
            { 1, 1, 3, 3 }, // CD9
            { 5, 6, 7, 8 },
        };

        var r = MarkerPanel.Evaluate(expr, Ids, Labels, Groups, Samples, Panel(), perSample: false);
        Assert.True(double.IsNaN(r.Heatmap[0, 0]));
        Assert.True(double.IsNaN(r.Heatmap[0, 1]));
        Assert.Equal(-1.0, r.Heatmap[1, 0], 9);
    }
}
