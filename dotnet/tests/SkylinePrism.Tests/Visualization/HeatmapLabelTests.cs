using System;
using System.Linq;
using ScottPlot;
using SkylinePrism.Core.Visualization;
using Xunit;

namespace SkylinePrism.Tests.Visualization;

/// <summary>
/// The marker heatmap labels only what can be read. A per-sample heatmap on a 96-sample cohort drew 96
/// sample ids rotated ACROSS the grid - over the data, the title and the row labels - and a large panel
/// drew every row label at 6pt, overlapping into a smear. These pin the replacement rules: name the
/// GROUP under each run of sample columns, name at most a fixed number of rows (every k-th), and
/// stagger group names that are wider than the run they label.
/// </summary>
public class HeatmapLabelTests
{
    private static Plot Draw(int rows, int cols, int[]? groups = null, string[]? groupNames = null)
    {
        var values = new double[rows, cols];
        var rnd = new Random(1);
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                values[i, j] = rnd.NextDouble() * 2 - 1;

        var plt = new Plot();
        PlotRenderer.DrawValueHeatmap(plt, values,
            Enumerable.Range(0, cols).Select(j => $"sample-{j:000}").ToArray(),
            Enumerable.Range(0, rows).Select(i => $"MARKER{i}").ToArray(),
            1.0, "row z-score", annotate: false, columnGroups: groups, groupNames: groupNames);
        plt.GetImageBytes(1400, 900, ImageFormat.Png); // ticks are generated at render time
        return plt;
    }

    private static string[] Labels(IAxis axis) =>
        axis.TickGenerator.Ticks.Where(t => t.IsMajor).Select(t => t.Label).ToArray();

    // 6 + 6 + 84: the shape of the cohort that exposed it - two small control groups and the study.
    private static (int[] Groups, string[] Names) Cohort()
    {
        var g = Enumerable.Repeat(0, 6).Concat(Enumerable.Repeat(1, 6)).Concat(Enumerable.Repeat(2, 84)).ToArray();
        return (g, new[] { "Quality Control", "Standard", "Unknown" });
    }

    [Fact]
    public void ManySampleColumns_AreNamedByGroup_NotBySample()
    {
        var (groups, names) = Cohort();
        var labels = Labels(Draw(9, groups.Length, groups, names).Axes.Bottom);

        Assert.Equal(3, labels.Length);
        Assert.Equal(names, labels.Select(l => l.Trim('\n')).ToArray());
        Assert.DoesNotContain(labels, l => l.StartsWith("sample-", StringComparison.Ordinal));
    }

    [Fact]
    public void EachGroupName_SitsUnderTheMiddleOfItsRun()
    {
        var (groups, names) = Cohort();
        var positions = Draw(9, groups.Length, groups, names).Axes.Bottom.TickGenerator.Ticks
            .Where(t => t.IsMajor).Select(t => t.Position).ToArray();

        // Runs are [0,6), [6,12), [12,96): centres 3, 9, 54.
        Assert.Equal(new[] { 3.0, 9.0, 54.0 }, positions);
    }

    [Fact]
    public void GroupNamesWiderThanTheirRun_AreStaggered()
    {
        // "Quality Control" is 15 characters over 6 columns, so it and "Standard" collide on one line:
        // every other name drops a line instead.
        var (groups, names) = Cohort();
        var labels = Labels(Draw(9, groups.Length, groups, names).Axes.Bottom);

        Assert.False(labels[0].StartsWith('\n'));
        Assert.True(labels[1].StartsWith('\n'));
        Assert.False(labels[2].StartsWith('\n'));
    }

    [Fact]
    public void WideRuns_AreNotStaggered()
    {
        // Two groups of 40 columns each have room for their names, so nothing moves.
        var groups = Enumerable.Repeat(0, 40).Concat(Enumerable.Repeat(1, 40)).ToArray();
        var labels = Labels(Draw(9, groups.Length, groups, new[] { "Control", "Disease" }).Axes.Bottom);

        Assert.All(labels, l => Assert.False(l.StartsWith('\n')));
    }

    [Fact]
    public void FewColumns_AreStillNamedOneByOne()
    {
        // At group-means size there is room for every name, and the groups ARE the columns.
        var labels = Labels(Draw(9, 3, new[] { 0, 1, 2 }, new[] { "A", "B", "C" }).Axes.Bottom);
        Assert.Equal(new[] { "sample-000", "sample-001", "sample-002" }, labels);
    }

    [Fact]
    public void ManyRows_NameEveryKthRow_AndNoMoreThanThirty()
    {
        // The largest shipped panel: 158 members. Every row labelled meant 6pt text in a smear.
        var labels = Labels(Draw(158, 3).Axes.Left);

        Assert.InRange(labels.Length, 20, 30);
        Assert.Contains("MARKER0", labels);
    }

    [Fact]
    public void FewRows_AreAllNamed()
    {
        Assert.Equal(9, Labels(Draw(9, 3).Axes.Left).Length);
    }

    [Fact]
    public void WithoutGroups_ManyColumnsStillGetSampleNames()
    {
        // A heatmap with no grouping to name falls back to sample ids, rotated to hang below the grid.
        var plt = Draw(9, 40);
        Assert.Equal(40, Labels(plt.Axes.Bottom).Length);
        Assert.Equal(-45, plt.Axes.Bottom.TickLabelStyle.Rotation);
    }
}
