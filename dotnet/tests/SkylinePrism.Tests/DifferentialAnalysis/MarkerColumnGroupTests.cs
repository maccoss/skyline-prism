using System.Linq;
using SkylinePrism.Core.DifferentialAnalysis;
using SkylinePrism.Core.Qc;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.DifferentialAnalysis;

/// <summary>
/// <see cref="MarkerPanelResult.ColumnGroups"/> is what lets a per-sample heatmap name its groups rather
/// than every sample. It has to agree with the columns exactly: one entry per column, in column order,
/// and - per sample - non-decreasing, because the columns are ordered by group.
/// </summary>
public class MarkerColumnGroupTests
{
    private static MarkerPanelResult Evaluate(bool perSample)
    {
        var ds = DifferentialDataset.Load(Fixtures.Path2("mini", "e2e-sum", "output"), FeatureLevel.Protein);
        var gene = ds.FeatureGenes.First(g => !string.IsNullOrEmpty(g));
        var panel = new ProteinList { Name = "t" };
        panel.Members.Add(gene);
        var identities = Enumerable.Range(0, ds.FeatureIds.Length).Select(ds.IdentityOf).ToArray();
        return MarkerPanel.Evaluate(ds.ExprLog2, identities, ds.MetadataValues("sample_type"),
            ds.SampleIds, panel, perSample);
    }

    [Fact]
    public void GroupMeans_EachColumnIsItsOwnGroup()
    {
        var r = Evaluate(perSample: false);
        Assert.Equal(Enumerable.Range(0, r.GroupNames.Length), r.ColumnGroups);
    }

    [Fact]
    public void PerSample_OneGroupPerColumn_InColumnOrder()
    {
        var r = Evaluate(perSample: true);

        Assert.Equal(r.ColumnLabels.Length, r.ColumnGroups.Length);
        Assert.All(r.ColumnGroups, g => Assert.InRange(g, 0, r.GroupNames.Length - 1));
        // Columns are ordered by group, so the runs are contiguous.
        Assert.Equal(r.ColumnGroups.OrderBy(g => g), r.ColumnGroups);
        Assert.Equal(r.GroupNames.Length, r.ColumnGroups.Distinct().Count());
    }
}
